import json
import gzip
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services/model"))
from llm_course_cache import CourseCache, MODEL, NVIDIA_URL, package_records, record_response, signature, validate_response


def fixture_bundle(directory):
    capture = str(Path(directory) / "capture.jsonl")
    base = {"model": MODEL, "messages": [{"role": "user", "content": "bundle-case"}], "chat_template_kwargs": {"enable_thinking": False}}
    for text in ["recorded-a", "recorded-b"]:
        response = {"id": text, "object": "chat.completion", "created": 1, "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}
        record_response(capture, NVIDIA_URL, base, response, lambda value: value)
        stream = [{"id": text, "object": "chat.completion.chunk", "created": 1, "model": MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
            {"id": text, "object": "chat.completion.chunk", "created": 1, "model": MODEL,
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
        record_response(capture, NVIDIA_URL, {**base, "stream": True}, stream, lambda value: value)
    path = Path(directory) / "responses.json.gz"
    package_records([json.loads(line) for line in Path(capture).read_text().splitlines()], path, {"fixture": True})
    return path, base


class CourseCacheTests(unittest.TestCase):
    def test_real_sdk_modern_limit_matches_legacy_record_without_rekeying(self):
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            records = json.loads(gzip.decompress(path.read_bytes()))["entries"]
            records = [{**entry, "request": {**entry["request"], "max_tokens": 16}} for entry in records]
            package_records(records, path, {"fixture": True})
            original_bytes = path.read_bytes()
            original_entries = json.loads(gzip.decompress(original_bytes))["entries"]
            client = ChatNVIDIA(model=MODEL, api_key="synthetic", base_url="http://127.0.0.1:1/v1")
            actual = client._get_payload(body["messages"], stop=None, max_completion_tokens=16,
                chat_template_kwargs=body["chat_template_kwargs"])
            self.assertEqual(actual["max_tokens"], 1024)
            cache = CourseCache(path)
            response, metadata = cache.get(actual)
            self.assertEqual(response["id"], "recorded-a")
            compatibility = metadata["request_compatibility"]
            self.assertEqual(compatibility["rule"], "max_completion_tokens_to_max_tokens")
            self.assertEqual(compatibility["requested_key"], signature(actual))
            self.assertEqual(compatibility["recorded_key"], signature({**body, "max_tokens": 16}))
            self.assertEqual(cache.get({**actual, "max_tokens": 1})[1]["request_compatibility"]["effective_key"], compatibility["effective_key"])
            self.assertIsNone(cache.get({**actual, "max_completion_tokens": 17}))
            for field, value in {"messages": [{"role": "user", "content": "Changed"}],
                    "tools": [], "temperature": 0, "seed": 1,
                    "chat_template_kwargs": {"enable_thinking": True}}.items():
                self.assertIsNone(cache.get({**actual, field: value}), field)
            self.assertEqual(path.read_bytes(), original_bytes)
            loaded = sorted((entry for pool in cache.entries.values() for entry in pool), key=lambda entry: entry["id"])
            self.assertEqual(loaded, sorted(original_entries, key=lambda entry: entry["id"]))
            for entry in original_entries:
                self.assertEqual(signature(entry["request"]), entry["key"])

    def test_exact_modern_record_precedes_explicit_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            records = json.loads(gzip.decompress(path.read_bytes()))["entries"]
            legacy = [{**entry, "request": {**entry["request"], "max_tokens": 16}} for entry in records]
            modern = {**legacy[0], "request": {**body, "max_completion_tokens": 16},
                "response": {"id": "modern", "choices": [{"index": 0, "message": {"role": "assistant", "content": "modern"}, "finish_reason": "stop"}]}}
            package_records(legacy + [modern], path, {"fixture": True})
            cache = CourseCache(path)
            response, metadata = cache.get(modern["request"])
            self.assertEqual(response["id"], "modern")
            self.assertNotIn("request_compatibility", metadata)
            response, metadata = cache.get({**modern["request"], "max_tokens": 1024})
            self.assertEqual(response["id"], "modern")
            self.assertEqual(metadata["request_compatibility"]["rule"], "max_completion_tokens_precedence")

    def test_token_limit_alias_preserves_stream_sample_cursor_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            records = json.loads(gzip.decompress(path.read_bytes()))["entries"]
            legacy = [{**entry, "request": {**entry["request"], "max_tokens": 16}} for entry in records]
            seeded = [{**entry, "request": {**entry["request"], "seed": 42}} for entry in legacy]
            package_records(legacy + seeded, path, {"fixture": True})
            cache = CourseCache(path)
            old = {**body, "stream": True, "max_tokens": 16}
            current = {**body, "stream": True, "max_completion_tokens": 16}
            self.assertEqual([cache.get(request, sample=True)[0][0]["id"]
                for request in (current, old, {**current, "max_tokens": 1024})],
                ["recorded-a", "recorded-b", "recorded-a"])
            self.assertEqual(cache.get(current, sample=True, sample_index=1)[0][0]["id"], "recorded-b")
            self.assertEqual([cache.get({**current, "seed": 42}, sample=True)[0][0]["id"] for _ in range(3)],
                ["recorded-a"] * 3)
            self.assertIsNone(cache.get({**current, "max_completion_tokens": True}))

    def test_incomplete_or_error_only_recordings_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            record = next(e for e in json.loads(gzip.decompress(path.read_bytes()))['entries'] if e['request'].get('stream'))
            broken = [[], [{'error': {'message': 'interrupted'}}],
                      [{'choices': [{'delta': {}, 'finish_reason': 'stop'}]}],
                      [{'choices': [{'index': 0, 'delta': {'content': 'unfinished'}, 'finish_reason': None}]}],
                      [{'choices': [{'index': '0', 'delta': {}, 'finish_reason': 'stop'}]}]]
            for response in broken:
                with self.subTest(response=response), self.assertRaises(ValueError):
                    package_records([{**record, 'response': response}], path, {'fixture': True})
            # Token-limit experiments remain valid completed protocol exchanges.
            truncated = [{'choices': [{'index': 0, 'delta': {'content': 'short'}, 'finish_reason': 'length'}]}]
            validate_response(truncated, True)

    def test_seeded_samples_stay_fixed_across_interleaving_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            records = json.loads(gzip.decompress(path.read_bytes()))["entries"]
            seeded = [{**entry, "request": {**entry["request"], "seed": seed}}
                for seed in (0, 12, 42) for entry in records]
            package_records(records + seeded, path, {"fixture": True})
            cache = CourseCache(path)
            for seed in (0, 12, 42):
                request = {**body, "seed": seed}
                first, metadata = cache.get(request, sample=True)
                self.assertEqual(metadata["sample_selection"], "seeded_exact")
                self.assertEqual(cache.get(request, sample=True, sample_index=1)[0]["id"], "recorded-b")
                self.assertEqual(cache.get(body, sample=True, sample_index=1)[1]["sample_selection"], "explicit_index")
                self.assertEqual(cache.get(request, sample=True)[0], first)
                self.assertEqual(CourseCache(path).get(request, sample=True)[0], first)
                self.assertNotIn(signature(request), cache.cursors)
            self.assertEqual([cache.get(body, sample=True)[0]["id"] for _ in range(3)],
                ["recorded-a", "recorded-b", "recorded-a"])

    def test_reviewed_default_preserves_every_recorded_sampling_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            records = json.loads(gzip.decompress(path.read_bytes()))["entries"]
            chosen = next(e for e in records if e["request"] == body and e["response"]["id"] == "recorded-b")
            provenance = {"preferred_entries": {signature(body): chosen["id"]}}
            package_records(records, path, provenance)
            cache = CourseCache(path)
            self.assertEqual(cache.get(body)[0]["id"], "recorded-b")
            self.assertEqual(cache.get(body, sample=True, sample_index=0)[0]["id"], "recorded-a")
            self.assertEqual(cache.get(body, sample=True, sample_index=1)[0]["id"], "recorded-b")
            package_records(records, path, {"preferred_entries": {signature(body): "invented-response"}})
            invalid = CourseCache(path)
            self.assertIsNone(invalid.get(body))
            self.assertTrue(invalid.unavailable)

    def test_recapturing_a_cache_hit_preserves_original_sample_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            entries = json.loads(gzip.decompress(path.read_bytes()))["entries"]
            original = next(e for e in entries if e["request"] == body and e["response"]["id"] == "recorded-a")
            duplicate = {**original, "recorded_at": original["recorded_at"] + 1000}
            package_records(entries + [duplicate], path, {"fixture": True})
            cache = CourseCache(path)
            self.assertEqual(cache.get(body)[0]["id"], "recorded-a")
            self.assertEqual(cache.get(body, sample=True, sample_index=1)[0]["id"], "recorded-b")

    def test_recorded_sampling_preserves_source_and_original_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            cache = CourseCache(path)
            results = [cache.get(body, sample=True) for _ in range(3)]
            self.assertEqual([r[0]["id"] for r in results], ["recorded-a", "recorded-b", "recorded-a"])
            self.assertEqual(cache.get(body, sample=True, sample_index=1)[0]["id"], "recorded-b")
            self.assertEqual(cache.get(body)[0]["id"], "recorded-a")
            self.assertEqual(results[0][0]["usage"]["total_tokens"], 6)
            self.assertFalse(results[0][1]["upstream_request"])
            self.assertEqual(results[0][1]["origin_provider"], NVIDIA_URL)
            self.assertFalse(results[0][1]["provider_usage_reused"])
            self.assertTrue(results[0][1]["recorded_provider_usage"])
            results[0][0]["choices"].clear()
            self.assertTrue(cache.get(body)[0]["choices"])

    def test_complete_semantic_request_and_transport_are_keyed(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            cache = CourseCache(path)
            for field, value in {"temperature": .7, "seed": 42, "stop": "x", "n": 2, "max_tokens": 12,
                    "tools": [{"type": "function", "function": {"name": "different"}}],
                    "response_format": {"type": "json_object"}, "stream_options": {"include_usage": True},
                    "model": "unrelated/model", "messages": [{"role": "user", "content": "novel"}]}.items():
                self.assertIsNone(cache.get({**body, field: value}), field)
            self.assertIsInstance(cache.get({**body, "stream": True})[0], list)
            alias = {**body, "model": "nvidia/Nemotron-3_5-Lightning"}
            self.assertEqual(signature(alias), signature(body))
            self.assertEqual(signature({**body, "use_course_cache": True, "cache_sample": True}), signature(body))
            self.assertEqual(signature({**body, "cache_pacing": "reading"}), signature(body))

    def test_bundle_corruption_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            path.write_bytes(path.read_bytes() + b"changed")
            cache = CourseCache(path)
            self.assertIsNone(cache.get(body))
            self.assertTrue(cache.unavailable)
            self.assertIsNone(CourseCache(Path(directory) / "absent").get(body))

    def test_capture_is_nvidia_only_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captured.jsonl"
            body = {"model": MODEL, "messages": [{"role": "user", "content": "synthetic-secret"}]}
            record_response(path, "https://api.tokenfactory.nebius.com/v1", body, {}, lambda value: value)
            self.assertFalse(path.exists())
            clean = lambda value: json.loads(json.dumps(value).replace("synthetic-secret", "[REDACTED]"))
            record_response(path, NVIDIA_URL, body, {"value": "synthetic-secret"}, clean)
            self.assertNotIn("synthetic-secret", path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.chmod(0o644)
            record_response(path, NVIDIA_URL, body, {"value": "ok"}, clean)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            link = Path(directory) / "link.jsonl"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                record_response(link, NVIDIA_URL, body, {}, clean)
            oversized = Path(directory) / "oversized.jsonl"
            record_response(oversized, NVIDIA_URL, body, {"value": "x" * 4194304}, clean)
            self.assertFalse(oversized.exists())

    def test_namespace_provenance_identity_and_transport_validation(self):
        for mutation in ["namespace", "identity", "model", "stream"]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, body = fixture_bundle(directory)
                data = json.loads(gzip.decompress(path.read_bytes()))
                if mutation == "namespace": data["namespace"] = "other-course"
                if mutation == "identity": data["entries"][0]["id"] = "invalid"
                if mutation == "model": data["entries"][0]["origin_model"] = "other-model"
                if mutation == "stream": data["entries"][0]["response"] = []
                artifact = gzip.compress(json.dumps(data).encode())
                path.write_bytes(artifact)
                manifest_path = Path(str(path) + ".manifest.json")
                manifest = json.loads(manifest_path.read_text())
                manifest["sha256"] = hashlib.sha256(artifact).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                cache = CourseCache(path)
                self.assertIsNone(cache.get(body))
                self.assertTrue(cache.unavailable)


if __name__ == "__main__":
    unittest.main()
