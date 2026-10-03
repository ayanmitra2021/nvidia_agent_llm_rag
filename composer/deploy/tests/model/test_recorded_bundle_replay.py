"""Every shipped recording must be reachable with its unchanged response identity."""
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import recorded_bundle_replay
from recorded_bundle_replay import sample_index_for_record
from test_llm_course_cache import fixture_bundle
from llm_course_cache import CourseCache, package_records


class RecordedBundleReplay(unittest.TestCase):
    def test_every_shipped_record_replays_through_real_asgi_without_inference(self):
        runtime = Path(__file__).resolve().parents[2] / 'services/model'
        result = subprocess.run([sys.executable, recorded_bundle_replay.__file__,
            str(runtime / 'course_cache/responses.json.gz'), '--runtime', str(runtime)],
            env={'PATH': os.environ.get('PATH', ''), 'HOME': tempfile.gettempdir(), 'PYTHONDONTWRITEBYTECODE': '1'},
            text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertGreater(report['entries'], 0)
        self.assertEqual(report['covered_upstream_calls'], 0)
        self.assertEqual(report['failures'], [])

    def test_modern_record_does_not_shadow_legacy_samples_at_the_http_boundary(self):
        from llm_request import completion_limit
        with tempfile.TemporaryDirectory() as directory:
            path, _ = fixture_bundle(directory)
            records = json.loads(gzip.decompress(path.read_bytes()))['entries']
            legacy = [{**entry, 'request': {**entry['request'], 'max_tokens': 16}} for entry in records]
            modern = [{**entry, 'request': completion_limit(entry['request'])} for entry in legacy]
            package_records(legacy + modern, path, {'fixture': True})
            original = path.read_bytes()
            cache = CourseCache(path)
            for entry in json.loads(gzip.decompress(original))['entries']:
                index = sample_index_for_record(cache, entry)
                request = completion_limit(entry['request'])
                response, metadata = cache.get(request, sample=True, sample_index=index)
                self.assertEqual(response, entry['response'])
                self.assertEqual(metadata['entry_id'], entry['id'])
            self.assertEqual(path.read_bytes(), original)

    def test_numeric_aliases_preserve_every_distinct_record_in_the_shared_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = fixture_bundle(directory)
            original = json.loads(gzip.decompress(path.read_bytes()))['entries']
            records = [{**entry, 'request': {**entry['request'], 'temperature': value}}
                       for entry in original for value in (0, 0.0)]
            package_records(records, path, {'fixture': True})
            recorded = json.loads(gzip.decompress(path.read_bytes()))['entries']
            cache = CourseCache(path)
            for record in recorded:
                index = sample_index_for_record(cache, record)
                response, metadata = cache.get(record['request'], sample=True, sample_index=index)
                self.assertEqual(metadata['entry_id'], record['id'])
                self.assertEqual(response, record['response'])
            for field, value in (('id', 'missing-entry'), ('response', {})):
                broken = copy.deepcopy(recorded[0])
                broken[field] = value
                with self.assertRaises(ValueError):
                    sample_index_for_record(cache, broken)


if __name__ == '__main__':
    unittest.main()
