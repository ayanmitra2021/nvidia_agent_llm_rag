import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services/model"))
from llm_store import Store
from llm_cache_bundle import export_bundle, import_bundle


class StoreTests(unittest.TestCase):
    def test_completion_tracking_does_not_allocate_requested_choice_count(self):
        from llm_ttft import CompletionState
        state = CompletionState(10**12)
        self.assertEqual(state.expected, 10**12)
        self.assertEqual(state.seen, set())
        state.observe({"choices": [{"index": 0, "finish_reason": "stop"}]})
        with self.assertRaises(ValueError):
            state.complete()

    def test_eviction_expiry_and_canonical_input(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory + "/cache.sqlite", lambda value: value, max_entries=2, max_bytes=512)
            self.assertEqual(store.key("url", {"x": 1, "y": 2}), store.key("url", {"y": 2, "x": 1}))
            self.assertNotEqual(store.key("url", {"temperature": 0}), store.key("url", {"temperature": 1}))
            for key in ["a", "b", "c"]:
                store.put(key, {"value": key})
            self.assertIsNone(store.get("a"))
            self.assertIsNotNone(store.get("c"))
            store.db.execute("UPDATE cache SET expires = 0")
            store.db.commit()
            self.assertIsNone(store.get("c"))
            store.db.close()

    def test_namespace_persistence_and_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/private/store.sqlite"
            first = Store(path, lambda value: value, namespace="revision-a")
            key = first.key("url", {"temperature": 0})
            first.put(key, {"result": "cached"})
            first.db.close()
            again = Store(path, lambda value: value, namespace="revision-a")
            self.assertEqual(again.get(again.key("url", {"temperature": 0})), {"result": "cached"})
            again.db.close()
            revised = Store(path, lambda value: value, namespace="revision-b")
            self.assertIsNone(revised.get(revised.key("url", {"temperature": 0})))
            self.assertEqual(Path(path).parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(Path(path).stat().st_mode & 0o777, 0o600)
            revised.db.close()
            with self.assertRaises(ValueError):
                Store(path, lambda value: value, max_entries=0)

    def test_bundle_excludes_journal_and_requires_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            source = directory + "/source.sqlite"
            store = Store(source, lambda value: value)
            key = store.key("url", {"temperature": 0})
            store.put(key, {"choices": [{"message": {"content": "example"}}]})
            store.record("POST", "url", {"private_prompt": "journal-only-marker"})
            store.db.close()
            bundle = directory + "/cache.json.gz"
            manifest = export_bundle(source, bundle)
            self.assertEqual(manifest["entries"], 1)
            self.assertEqual(Path(bundle).stat().st_mode & 0o777, 0o600)
            self.assertFalse(manifest["request_journal_included"])
            import gzip
            self.assertNotIn(b"journal-only-marker", gzip.decompress(Path(bundle).read_bytes()))
            with self.assertRaises(ValueError):
                import_bundle(directory + "/target.sqlite", bundle, "wrong")
            self.assertEqual(import_bundle(directory + "/target.sqlite", bundle, manifest["sha256"])["imported"], 1)
            target = Store(directory + "/target.sqlite", lambda value: value)
            self.assertEqual(target.get(key)["choices"][0]["message"]["content"], "example")
            target.db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
