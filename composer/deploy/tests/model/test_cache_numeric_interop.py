"""Numeric JSON compatibility must never rewrite or weaken bundle evidence."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from test_llm_course_cache import fixture_bundle
import llm_course_cache as cache_module
from llm_course_cache import CourseCache, numeric_signature, package_records, record_identity, signature


class NumericInterop(unittest.TestCase):
    def bundle(self, directory, both_forms=False):
        path, body = fixture_bundle(directory)
        rows = json.loads(gzip.decompress(path.read_bytes()))['entries']
        for row in rows:
            row['request']['temperature'] = 0.0
        if both_forms:
            for row in copy.deepcopy(rows):
                row['request']['temperature'] = 0
                rows.append(row)
        package_records(rows, path, {'fixture': True})
        return path, {**body, 'temperature': 0.0}

    def test_json_sse_numeric_variants_preserve_original_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = self.bundle(directory)
            original = path.read_bytes()
            manifest = Path(str(path) + '.manifest.json').read_bytes()
            rows = json.loads(gzip.decompress(original))['entries']
            cache = CourseCache(path)
            for streaming in (False, True):
                request = {**body, **({'stream': True} if streaming else {})}
                js = {**request, 'temperature': 0}
                self.assertNotEqual(signature(js), signature(request))
                value, metadata = cache.get(js)
                self.assertEqual(isinstance(value, list), streaming)
                self.assertEqual(metadata['request_compatibility']['rule'], 'numeric_json_equivalence')
                recorded = next(row for row in rows if row['id'] == metadata['entry_id'])
                self.assertEqual(metadata['request_compatibility']['recorded_key'], recorded['key'])
                self.assertEqual(signature(recorded['request']), recorded['key'])
                self.assertEqual(record_identity(recorded), recorded['id'])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(Path(str(path) + '.manifest.json').read_bytes(), manifest)
            loaded = [entry for pool in cache.entries.values() for entry in pool]
            self.assertEqual(sorted(loaded, key=lambda entry: entry['id']), sorted(rows, key=lambda entry: entry['id']))

    def test_same_pool_cursor_seed_and_explicit_index_for_both_number_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = self.bundle(directory, both_forms=True)
            js = {**body, 'temperature': 0}
            cache = CourseCache(path)
            expected = [cache.get(body, sample=True, sample_index=index)[1]['entry_id'] for index in range(4)]
            self.assertEqual(len(set(expected)), 4)
            actual = [cache.get(request, sample=True)[1]['entry_id'] for request in (body, js, body, js)]
            self.assertEqual(actual, expected)
            self.assertEqual(cache.get(js, sample=True, sample_index=3)[1]['entry_id'], expected[3])
            rows = json.loads(gzip.decompress(path.read_bytes()))['entries']
            for row in rows:
                row['request']['seed'] = 42
            package_records(rows, path, {'fixture': True})
            cache = CourseCache(path)
            ids = [cache.get({**request, 'seed': 42}, sample=True)[1]['entry_id'] for request in (body, js, body, js)]
            self.assertEqual(len(set(ids)), 1)
            self.assertEqual(cache.cursors, {})

    def test_preferences_remain_recorded_and_conflicts_are_resolved_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = self.bundle(directory, both_forms=True)
            rows = json.loads(gzip.decompress(path.read_bytes()))['entries']
            floating = next(row for row in rows if not row['request'].get('stream') and type(row['request']['temperature']) is float)
            integer = next(row for row in rows if not row['request'].get('stream') and type(row['request']['temperature']) is int)
            preference = {floating['key']: floating['id']}
            package_records(rows, path, {'preferred_entries': preference})
            cache = CourseCache(path)
            for temperature in (0, 0.0):
                self.assertEqual(cache.get({**body, 'temperature': temperature})[1]['entry_id'], floating['id'])
            preference[integer['key']] = integer['id']
            package_records(rows, path, {'preferred_entries': preference})
            cache = CourseCache(path)
            for temperature in (0, 0.0):
                _, metadata = cache.get({**body, 'temperature': temperature})
                self.assertEqual(metadata['entry_id'], integer['id'])
                resolution = metadata['numeric_preference_resolution']
                self.assertEqual(resolution['rule'], 'canonical_numeric_key_then_recorded_key')
                self.assertEqual(resolution['selected_entry_id'], integer['id'])
                self.assertEqual(resolution['recorded_preferences'], preference)
            self.assertEqual(cache.preferred, preference)
            self.assertFalse(cache.unavailable)
            # A different reviewed answer must never be chosen by key ordering.
            other = next(row for row in rows if row['key'] == integer['key']
                         and row['response'] != floating['response'])
            preference[integer['key']] = other['id']
            package_records(rows, path, {'preferred_entries': preference})
            invalid = CourseCache(path)
            with self.assertRaisesRegex(ValueError, 'require review'):
                invalid._load()
            self.assertIsNone(invalid.get(body))
            self.assertTrue(invalid.unavailable)

    def test_recursive_numbers_are_lossless_and_do_not_coerce_types_or_strings(self):
        self.assertEqual(numeric_signature({'nested': [{'number': 0.0}, -0.0, 1.0]}),
                         numeric_signature({'nested': [{'number': 0}, 0, 1]}))
        for left, right in [(0.1, 0.2), (True, 1), (False, 0), ('0.0', '0'), (2**53, 2**53 + 1)]:
            self.assertNotEqual(numeric_signature({'nested': [left]}), numeric_signature({'nested': [right]}))
        self.assertNotEqual(numeric_signature({'nested': [float(2**53)]}), numeric_signature({'nested': [2**53 + 1]}))
        for number in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                numeric_signature({'nested': [number]})

    def test_numeric_lookup_composes_with_legacy_limit_without_relaxing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = self.bundle(directory)
            rows = json.loads(gzip.decompress(path.read_bytes()))['entries']
            for row in rows:
                row['request']['max_tokens'] = 16
            package_records(rows, path, {'fixture': True})
            cache = CourseCache(path)
            request = {**body, 'temperature': 0, 'max_completion_tokens': 16}
            _, metadata = cache.get(request)
            compatibility = metadata['request_compatibility']
            self.assertEqual(compatibility['rule'], 'max_completion_tokens_to_max_tokens')
            self.assertTrue(compatibility['numeric_json_equivalence'])
            for field, value in [('temperature', 0.1), ('temperature', True), ('temperature', '0.0'),
                                 ('max_completion_tokens', 17)]:
                self.assertIsNone(cache.get({**request, field: value}))

    def test_numeric_compatibility_does_not_accept_stale_record_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = self.bundle(directory)
            document = json.loads(gzip.decompress(path.read_bytes()))
            document['entries'][0]['request']['temperature'] = 0  # Equal number, changed v1 identity.
            path.write_bytes(gzip.compress(json.dumps(document).encode()))
            manifest_path = Path(str(path) + '.manifest.json')
            manifest = json.loads(manifest_path.read_text())
            manifest['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            invalid = CourseCache(path)
            with self.assertRaisesRegex(ValueError, 'course request digest mismatch'):
                invalid._load()
            self.assertIsNone(invalid.get(body))
            self.assertTrue(invalid.unavailable)

    def test_actual_http_json_and_sse_match_float_recordings_without_provider_calls(self):
        class Provider(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.server.attempts += 1
                self.send_response(500)
                self.end_headers()

        server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        server.attempts = 0
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        provider = f'http://127.0.0.1:{server.server_port}/v1'
        with tempfile.TemporaryDirectory() as directory:
            path, body = self.bundle(directory)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            environment = {'PATH': os.environ['PATH'], 'HOME': directory, 'PYTHONDONTWRITEBYTECODE': '1',
                'LLM_STORE_PATH': str(Path(directory) / 'runtime.sqlite'),
                'LLM_COURSE_CACHE_PATH': str(path), 'LLM_CACHE_PACING': 'off',
                'LLM_RETRY_ATTEMPTS': '1', 'TEST_NUMERIC_KEY': 'synthetic-numeric-key',
                'DLI_LLM_PROVIDERS': json.dumps([{'base_url': provider, 'api_key_env': 'TEST_NUMERIC_KEY'}])}
            source = Path(cache_module.__file__).resolve().parent
            program = f"import uvicorn; uvicorn.run('litellm_service:app', host='127.0.0.1', port={port}, log_config=None, access_log=False)"
            with tempfile.TemporaryFile() as diagnostics:
                process = subprocess.Popen([sys.executable, '-c', program], cwd=source, env=environment,
                                           stdout=diagnostics, stderr=diagnostics)
                try:
                    with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=10) as client:
                        for _ in range(150):
                            self.assertIsNone(process.poll(), 'Numeric fixture wrapper exited')
                            try:
                                if client.get('/health').status_code == 200:
                                    break
                            except httpx.HTTPError:
                                pass
                            time.sleep(.05)
                        else:
                            self.fail('Numeric fixture wrapper did not start')
                        for streaming in (False, True):
                            request = {**body, 'temperature': 0, **({'stream': True} if streaming else {})}
                            wire = json.dumps(request)
                            self.assertIn('"temperature": 0', wire)
                            self.assertNotIn('"temperature": 0.0', wire)
                            response = client.post('/v1/chat/completions', content=wire,
                                headers={'Content-Type': 'application/json', 'x-dli-base-url': provider})
                            self.assertEqual(response.status_code, 200)
                            self.assertEqual(response.headers['x-dli-cache'], 'BUNDLE_HIT')
                            if streaming:
                                self.assertIn('data: [DONE]', response.text)
                                self.assertNotIn('"error"', response.text)
                            else:
                                self.assertTrue(response.json()['choices'][0]['message']['content'])
                        self.assertEqual(server.attempts, 0)
                finally:
                    process.terminate()
                    process.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
