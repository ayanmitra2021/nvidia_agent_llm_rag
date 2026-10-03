"""Mixed provider capture retains canonical inputs and actual served provenance."""
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import httpx

from test_llm_course_cache import fixture_bundle
from llm_course_cache import (CourseCache, MODEL, NVIDIA_URL, ORIGIN_SCHEMA,
    capture_provider_label, package_records, record_response, signature)
import test_provider_circuit as circuit

LOCAL_PROVIDER = 'http://127.0.0.1:9912/v1'


def _wait_for_wrapper(process, client, log, startup_timeout=30):
    """Bound cold imports separately from the provider TTFT exercised below."""
    started = time.monotonic()
    deadline = started + startup_timeout
    last_health = 'not-probed'
    while True:
        returncode = process.poll()
        remaining = deadline - time.monotonic()
        if returncode is not None or remaining <= 0:
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - 8192))
            tail = log.read()
            text = tail.decode('utf-8', errors='replace')
            phases = re.findall(r'^fixture startup: (uvicorn|wrapper|http)$', text, re.MULTILINE)
            errors = re.findall(r'^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))(?::|$)', text, re.MULTILINE)
            # Raw child logs can contain environment values; report only structural diagnostics.
            raise AssertionError(
                f'wrapper startup failed: elapsed={time.monotonic() - started:.2f}s '
                f'exit={returncode} health={last_health} phase={phases[-1:] or ["unknown"]} '
                f'errors={errors[-3:]} log_bytes={size} '
                f'log_tail_sha256={hashlib.sha256(tail).hexdigest()}')
        try:
            response = client.get('/health', timeout=min(1, remaining))
            last_health = response.status_code
            if response.status_code == 200:
                return
        except httpx.HTTPError as error:
            last_health = type(error).__name__
        time.sleep(min(.05, max(0, deadline - time.monotonic())))


class WrapperReadinessTests(unittest.TestCase):
    def test_cold_start_can_exceed_three_seconds_without_changing_request_budget(self):
        clock = [0.0]
        process = Mock()
        process.poll.return_value = None
        client = Mock()
        client.get.side_effect = lambda *a, **kw: httpx.Response(200 if clock[0] >= 4.2 else 503)
        with patch.object(time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(time, 'sleep', side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay)):
            _wait_for_wrapper(process, client, io.BytesIO(), startup_timeout=6)
        self.assertGreaterEqual(clock[0], 4.2)
        self.assertLess(clock[0], 6)
        self.assertTrue(all(call.args == ('/health',) and call.kwargs['timeout'] <= 1
                            for call in client.get.call_args_list))

    def test_exited_child_fails_immediately_with_safe_diagnostics(self):
        process = Mock()
        process.poll.return_value = 7
        client = Mock()
        log = io.BytesIO(b'fixture startup: wrapper\nImportError: secret-value-must-not-appear\n')
        with self.assertRaises(AssertionError) as caught:
            _wait_for_wrapper(process, client, log)
        message = str(caught.exception)
        self.assertIn('exit=7', message)
        self.assertIn("phase=['wrapper']", message)
        self.assertIn("errors=['ImportError']", message)
        self.assertIn('log_tail_sha256=', message)
        self.assertNotIn('secret-value-must-not-appear', message)
        client.get.assert_not_called()

    def test_never_healthy_child_fails_at_monotonic_deadline(self):
        clock = [0.0]
        process = Mock()
        process.poll.return_value = None
        client = Mock()
        client.get.return_value = httpx.Response(503)
        with patch.object(time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(time, 'sleep', side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay)):
            with self.assertRaisesRegex(AssertionError, 'health=503'):
                _wait_for_wrapper(process, client, io.BytesIO(), startup_timeout=.2)
        self.assertAlmostEqual(clock[0], .2)


def completion(model='served/model'):
    return {'id': 'new-recording', 'model': model, 'choices': [{'index': 0,
        'message': {'role': 'assistant', 'content': 'real fixture output'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7}}


def capture(directory, body, response=None, url=LOCAL_PROVIDER):
    path = Path(directory) / 'new.jsonl'
    routed = {**body, 'model': 'routed/model', 'stream': True, 'stream_options': {'include_usage': True}}
    with patch.dict(os.environ, {'NEBIUS_BASE_URL': LOCAL_PROVIDER}):
        record_response(path, url, routed, response or completion(), lambda value: value, canonical_body=body)
    return json.loads(path.read_text().splitlines()[-1])


class OriginTests(unittest.TestCase):
    def test_mixed_bundle_preserves_legacy_records_and_real_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            old = json.loads(gzip.decompress(path.read_bytes()))['entries']
            response = {**completion(), 'dli_routing': {'fallback_used': True}}
            row = capture(directory, body, response)
            self.assertIn('dli_routing', response)
            self.assertNotIn('dli_routing', row['response'])
            self.assertEqual(row['request'], body)
            self.assertEqual(row['origin_schema'], ORIGIN_SCHEMA)
            self.assertEqual(row['origin_provider'], 'nebius')
            self.assertEqual(row['origin_model'], 'served/model')
            self.assertEqual(row['provider_request']['model'], 'routed/model')
            manifest = package_records(old + [row], path, {'fixture': True})
            rows = json.loads(gzip.decompress(path.read_bytes()))['entries']
            self.assertEqual([entry for entry in rows if entry['id'] in {e['id'] for e in old}], old)
            self.assertEqual(manifest['origin_provider'], 'mixed')
            self.assertEqual(manifest['origin_providers'], {'nvidia': 4, 'nebius': 1})
            value, metadata = CourseCache(path).get(body, sample=True, sample_index=2)
            self.assertEqual(value, row['response'])
            self.assertEqual(metadata['usage_origin_provider'], 'nebius')
            self.assertEqual(metadata['origin_model'], 'served/model')
            self.assertEqual(metadata['provider_requested_model'], 'routed/model')
            self.assertNotIn(LOCAL_PROVIDER, json.dumps(row))
            self.assertNotIn(LOCAL_PROVIDER, json.dumps(metadata))

    def test_identical_output_from_two_origins_has_distinct_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            _, body = fixture_bundle(directory)
            nebius = capture(directory, body)
            nvidia = capture(directory, body, url=NVIDIA_URL)
            self.assertEqual(nebius['response'], nvidia['response'])
            self.assertEqual(nebius['key'], nvidia['key'])
            self.assertNotEqual(nebius['id'], nvidia['id'])

    def test_unknown_fallback_needs_explicit_private_mapping(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / 'unknown.jsonl'
            body = {'model': MODEL, 'messages': []}
            record_response(path, LOCAL_PROVIDER, body, completion(), lambda value: value, canonical_body=body)
            self.assertFalse(path.exists())
            self.assertIsNone(capture_provider_label(LOCAL_PROVIDER))
            with patch.dict(os.environ, {'LLM_CAPTURE_PROVIDER_LABELS': json.dumps({LOCAL_PROVIDER: 'nebius'})}):
                record_response(path, LOCAL_PROVIDER, body, completion(), lambda value: value, canonical_body=body)
            self.assertEqual(json.loads(path.read_text())['origin_provider'], 'nebius')
            for mapping in [{LOCAL_PROVIDER: 'unknown'}, {LOCAL_PROVIDER: []},
                    {LOCAL_PROVIDER: 'nebius', LOCAL_PROVIDER + '/': 'nvidia'}]:
                with patch.dict(os.environ, {'LLM_CAPTURE_PROVIDER_LABELS': json.dumps(mapping)}):
                    with self.assertRaises(ValueError):
                        capture_provider_label(LOCAL_PROVIDER)

    def test_stream_round_trip_and_incomplete_capture_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            body = {'model': MODEL, 'messages': [], 'stream': True}
            chunks = [{'model': 'served/model', 'choices': [{'index': 0,
                'delta': {'content': 'part'}, 'finish_reason': None}]},
                {'model': 'served/model', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                 'usage': {'completion_tokens': 1}}]
            row = capture(directory, body, chunks)
            path = Path(directory) / 'bundle.gz'
            package_records([row], path, {})
            self.assertEqual(CourseCache(path).get(body)[0], chunks)
            for bad in [chunks[:1], [{**chunks[0], 'model': 'other/model'}, chunks[1]],
                    [{'choices': chunks[1]['choices']}]]:
                with self.assertRaises(ValueError):
                    capture(directory, body, bad)

    def test_loader_rejects_tampered_provenance_and_manifest_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            body = {'model': MODEL, 'messages': []}
            row = capture(directory, body)
            path = Path(directory) / 'bundle.gz'
            mutations = [{'origin_provider': 'unknown'}, {'origin_schema': 'unknown'},
                {'origin_model': 'false-model'}, {'provider_request': {'model': 'routed/model', 'messages': [], 'api_key': 'synthetic'}},
                {'origin_provider': 'nvidia'}]
            for changed in mutations:
                package_records([row], path, {})
                data = json.loads(gzip.decompress(path.read_bytes()))
                data['entries'][0].update(changed)
                artifact = gzip.compress(json.dumps(data).encode())
                path.write_bytes(artifact)
                manifest_path = Path(str(path) + '.manifest.json')
                manifest = json.loads(manifest_path.read_text())
                manifest['sha256'] = hashlib.sha256(artifact).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                cache = CourseCache(path)
                self.assertIsNone(cache.get(body), changed)
                self.assertTrue(cache.unavailable)
            for changed in [{'origin_providers': {'nvidia': 1}}, {'origin_provider': NVIDIA_URL}]:
                manifest = package_records([row], path, {})
                manifest.update(changed)
                Path(str(path) + '.manifest.json').write_text(json.dumps(manifest))
                self.assertIsNone(CourseCache(path).get(body))

    def test_legacy_token_alias_never_selects_new_provider_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            old = json.loads(gzip.decompress(path.read_bytes()))['entries']
            legacy = [{**entry, 'request': {**entry['request'], 'max_tokens': 16}} for entry in old]
            canonical = {**body, 'max_tokens': 16}
            new = capture(directory, canonical)
            package_records(legacy + [new], path, {'preferred_entries': {new['key']: new['id']}})
            request = {**body, 'max_completion_tokens': 16}
            cache = CourseCache(path)
            value, metadata = cache.get(request)
            self.assertEqual(value['id'], 'recorded-a')
            self.assertEqual(metadata['origin_provider'], NVIDIA_URL)
            self.assertEqual(metadata['sample_count'], 2)
            package_records([new], path, {})
            self.assertIsNone(CourseCache(path).get(request))
            exact = capture(directory, request)
            package_records(legacy + [new, exact], path, {})
            self.assertEqual(CourseCache(path).get(request)[1]['origin_provider'], 'nebius')


class CaptureHTTPTests(unittest.TestCase):
    setUpClass = classmethod(circuit.ProviderCircuitTests.setUpClass.__func__)
    tearDownClass = classmethod(circuit.ProviderCircuitTests.tearDownClass.__func__)
    setUp = circuit.ProviderCircuitTests.setUp

    def test_actual_fallback_json_and_sse_capture_canonical_and_wire_requests(self):
        self.servers[0].state['mode'] = 503
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            capture_path = Path(directory) / 'capture.jsonl'
            env = {key: value for key, value in os.environ.items()
                if not key.startswith(('LLM_', 'DLI_LLM_', 'NEBIUS_', 'NVIDIA_', 'OPENAI_'))}
            env.update(LLM_STORE_PATH=str(Path(directory) / 'requests.sqlite'),
                LLM_COURSE_CACHE_ENABLED='false', LLM_CACHE_ENABLED='false', LLM_CACHE_PACING='off',
                LLM_CAPTURE_COURSE_CACHE=str(capture_path), LLM_MAX_TTFT_SECONDS='1',
                LLM_RETRY_ATTEMPTS='1', NEBIUS_BASE_URL=self.provider_urls[1],
                NEBIUS_TOKEN_FACTORY_KEY='synthetic', TEST_PRIMARY_KEY='synthetic',
                NEBIUS_MODEL_ALIASES=json.dumps({MODEL: 'nvidia/Nemotron-3_5-Lightning'}),
                DLI_LLM_PROVIDERS=json.dumps([{'base_url': self.provider_urls[0], 'api_key_env': 'TEST_PRIMARY_KEY'}]))
            program = ("print('fixture startup: uvicorn', flush=True); import uvicorn; "
                "print('fixture startup: wrapper', flush=True); from litellm_service import app; "
                "print('fixture startup: http', flush=True); "
                f"uvicorn.run(app, host='127.0.0.1', port={port}, log_config=None, access_log=False)")
            with tempfile.TemporaryFile() as log:
                process = subprocess.Popen([sys.executable, '-c', program], cwd=circuit.SOURCE, env=env, stdout=log, stderr=log)
                try:
                    with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5, trust_env=False) as client:
                        _wait_for_wrapper(process, client, log)
                        for streaming in (False, True):
                            body = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'capture fixture'}],
                                'stream': streaming, 'max_completion_tokens': 19}
                            response = client.post('/v1/chat/completions', json=body,
                                headers={'x-dli-base-url': self.provider_urls[0]})
                            self.assertEqual(response.status_code, 200)
                            self.assertEqual(response.headers['x-dli-fallback'], 'true')
                            row = json.loads(capture_path.read_text().splitlines()[-1])
                            self.assertEqual(row['request'], body)
                            self.assertEqual(row['provider_request'], self.servers[1].state['requests'][-1])
                            self.assertEqual(row['origin_provider'], 'nebius')
                            self.assertEqual(row['origin_model'], 'nvidia/Nemotron-3_5-Lightning')
                            self.assertEqual(isinstance(row['response'], list), streaming)
                            self.assertNotIn('max_tokens', row['provider_request'])
                            self.assertNotIn('synthetic', json.dumps(row['provider_request']))
                        self.assertEqual(len(capture_path.read_text().splitlines()), 2)
                finally:
                    process.terminate()
                    process.wait(timeout=10)
