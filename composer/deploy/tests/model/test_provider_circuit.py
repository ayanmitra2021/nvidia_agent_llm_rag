"""Real HTTP circuit regressions; synthetic providers, isolated wrapper processes."""
import concurrent.futures
import contextlib
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

SOURCE = Path(__file__).resolve().parents[2] / 'services/model'
MODEL = 'course/circuit-test'
OTHER_MODEL = 'course/circuit-other'


class CircuitProvider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        state = self.server.state
        with state['lock']:
            mode = state['mode']
            state['requests'].append(body)
        state['started'].set()
        time.sleep(state.get('response_delay', 0))
        if isinstance(mode, int):
            payload = json.dumps({'error': {'message': 'synthetic failure'}}).encode()
            self.send_response(mode)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if mode == 'transport':
            self.close_connection = True
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()

        def send(delta, finish=None):
            value = {'id': 'synthetic-circuit', 'object': 'chat.completion.chunk',
                'created': 1, 'model': body['model'], 'choices': [
                    {'index': 0, 'delta': delta, 'finish_reason': finish}]}
            self.wfile.write(('data: ' + json.dumps(value) + '\n\n').encode())
            self.wfile.flush()

        if mode == 'sse401':
            self.wfile.write(b'data: {"error":{"code":401,"message":"synthetic-circuit-primary"}}\n\n')
            self.wfile.flush()
            return
        try:
            send({'role': 'assistant', 'content': 'visible-' + state['name']})
            if mode in {'hold', 'stall-after'}:
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    if select.select([self.connection], [], [], 0.01)[0]:
                        if not self.connection.recv(1, socket.MSG_PEEK):
                            state['closed'].set()
                            return
                    if mode == 'hold' and state['release'].is_set():
                        break
                else:
                    return
            if mode != 'truncated':
                send({}, 'stop')
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            state['closed'].set()


class ProviderCircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servers = []
        for name in ('primary', 'fallback'):
            server = ThreadingHTTPServer(('127.0.0.1', 0), CircuitProvider)
            server.daemon_threads = True
            server.state = {'name': name, 'mode': 'success', 'requests': [],
                'lock': threading.Lock(), 'started': threading.Event(),
                'closed': threading.Event(), 'release': threading.Event()}
            threading.Thread(target=server.serve_forever, daemon=True).start()
            cls.servers.append(server)
        cls.provider_urls = [f'http://127.0.0.1:{s.server_port}/v1' for s in cls.servers]

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers:
            server.state['release'].set()
            server.shutdown()
            server.server_close()

    def setUp(self):
        for server in self.servers:
            with server.state['lock']:
                server.state['mode'] = 'success'
                server.state['response_delay'] = 0
                server.state['requests'].clear()
                for key in ('started', 'closed', 'release'):
                    server.state[key].clear()

    def mode(self, provider, value):
        state = self.servers[provider].state
        with state['lock']:
            state['mode'] = value
            for key in ('started', 'closed', 'release'):
                state[key].clear()

    def count(self, provider):
        return len(self.servers[provider].state['requests'])

    @contextlib.contextmanager
    def wrapper(self, threshold=3, cooldown=30, cap=300, retries=1, ttft=5, request_timeout=5, fallback=True, fallback_allowed=True, primary_ttft=None):
        from test_llm_course_cache import fixture_bundle
        with tempfile.TemporaryDirectory(prefix='provider-circuit-') as directory:
            bundle, _ = fixture_bundle(directory)
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            env = {key: value for key, value in os.environ.items()
                if not key.startswith(('LLM_', 'DLI_LLM_', 'NEBIUS_', 'NVIDIA_', 'OPENAI_'))}
            env.update(LLM_STORE_PATH=str(Path(directory) / 'requests.sqlite'),
                LLM_COURSE_CACHE_PATH=str(bundle), LLM_COURSE_CACHE_ENABLED='false',
                LLM_CACHE_ENABLED='false', LLM_CACHE_PACING='off',
                LLM_MAX_TTFT_SECONDS=str(ttft), LLM_REQUEST_TIMEOUT_SECONDS=str(request_timeout),
                LLM_RETRY_ATTEMPTS=str(retries),
                LLM_CIRCUIT_FAILURE_THRESHOLD=str(threshold),
                LLM_CIRCUIT_BASE_COOLDOWN_SECONDS=str(cooldown),
                LLM_CIRCUIT_MAX_COOLDOWN_SECONDS=str(cap),
                CIRCUIT_PRIMARY_KEY='synthetic-circuit-primary',
                CIRCUIT_FALLBACK_KEY='synthetic-circuit-fallback',
                CIRCUIT_FALLBACK_URL=self.provider_urls[1],
                DLI_LLM_PROVIDERS=json.dumps([{'base_url': self.provider_urls[0],
                    'api_key_env': 'CIRCUIT_PRIMARY_KEY',
                    'model_aliases': {MODEL: 'primary/model', OTHER_MODEL: 'primary/model'}}]),
                DLI_LLM_FALLBACK_PROVIDER=json.dumps({'base_url_env': 'CIRCUIT_FALLBACK_URL',
                    'api_key_env': 'CIRCUIT_FALLBACK_KEY',
                    'model_aliases': {MODEL: 'fallback/model', OTHER_MODEL: 'fallback/model'}}))
            if not fallback:
                env.pop('DLI_LLM_FALLBACK_PROVIDER')
            elif not fallback_allowed:
                entry = json.loads(env['DLI_LLM_FALLBACK_PROVIDER'])
                entry['api_key_env'] = 'NEBIUS_TOKEN_FACTORY_KEY'
                env['NEBIUS_TOKEN_FACTORY_KEY'] = 'synthetic-circuit-fallback'
                env['DLI_LLM_FALLBACK_PROVIDER'] = json.dumps(entry)
            if ttft is None:
                env.pop('LLM_MAX_TTFT_SECONDS')
            if primary_ttft is not None:
                env['LLM_PRIMARY_ONLY_TTFT_SECONDS'] = str(primary_ttft)
            program = ('from llm_output import protect_output; protect_output(); import uvicorn; '
                f"uvicorn.run('litellm_service:app', host='127.0.0.1', port={port}, log_config=None, access_log=False)")
            with tempfile.TemporaryFile() as log:
                process = subprocess.Popen([sys.executable, '-c', program], cwd=SOURCE,
                    env=env, stdout=log, stderr=log)
                url = f'http://127.0.0.1:{port}'
                try:
                    with httpx.Client(timeout=1, trust_env=False) as client:
                        for _ in range(180):
                            if process.poll() is not None:
                                self.fail('wrapper exited during circuit startup')
                            try:
                                if client.get(url + '/health').status_code == 200:
                                    break
                            except httpx.HTTPError:
                                pass
                            time.sleep(0.05)
                        else:
                            self.fail('wrapper did not become ready')
                    yield url
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                    log.seek(0)
                    diagnostics = log.read()
                    self.assertNotIn(b'synthetic-circuit-primary', diagnostics)
                    self.assertNotIn(b'synthetic-circuit-fallback', diagnostics)

    def body(self, model=MODEL, stream=False):
        return {'model': model, 'messages': [{'role': 'user', 'content': 'circuit regression'}],
            'stream': stream, 'use_cache': False, 'use_course_cache': False}

    def test_client_initialization_failure_releases_probe_and_transport(self):
        program = '''
import asyncio, json
from unittest.mock import patch, AsyncMock
import openai
import litellm_service as service
from llm_circuit import Circuits
class BrokenSDK:
    @property
    def chat(self):
        raise TypeError('synthetic resource initialization failure')
async def check():
    url = next(iter(service.PROVIDERS))
    model = 'course/constructor-probe'
    body = {'model': model, 'messages': [{'role': 'user', 'content': 'probe'}]}
    results = []
    for fault in ('sdk', 'resource', 'transport', 'cancelled'):
        clock = [0.0]
        service.CIRCUITS = Circuits(1, .01, .02, clock=lambda: clock[0])
        service.runtime().CIRCUITS = service.CIRCUITS
        ticket = service.CIRCUITS.acquire(url, model)
        service.CIRCUITS.finish(ticket, 'failure', 'transport_error')
        clock[0] = 1.0
        transport = AsyncMock()
        sdk_error = asyncio.CancelledError() if fault == 'cancelled' else TypeError('synthetic initialization failure')
        with patch.object(openai, 'AsyncOpenAI', side_effect=None if fault == 'resource' else sdk_error,
                return_value=BrokenSDK()), patch.object(service, 'http_client',
                side_effect=TypeError('synthetic transport failure') if fault == 'transport' else None,
                return_value=transport):
            try:
                await service.perform_completion(url, 'unused', body, None, body)
            except (TypeError, asyncio.CancelledError):
                pass
            else:
                raise AssertionError('initialization failure was suppressed')
        state = service.CIRCUITS.snapshot(url, model)
        assert not state['probe_in_flight'], fault
        assert state['consecutive_failures'] == 1 and state['tier'] == 1, fault
        assert service.CIRCUITS.states[(url, model)].active == 0, fault
        assert transport.aclose.await_count == (0 if fault == 'transport' else 1), fault
        assert service.CIRCUITS.acquire(url, model) is not None, fault
        results.append(fault)
    print(json.dumps(results))
asyncio.run(check())
'''
        with tempfile.TemporaryDirectory(prefix='circuit-constructor-') as directory:
            env = {key: value for key, value in os.environ.items()
                if not key.startswith(('LLM_', 'DLI_LLM_', 'NEBIUS_', 'NVIDIA_', 'OPENAI_'))}
            env.update(LLM_STORE_PATH=str(Path(directory) / 'store.sqlite'),
                CONSTRUCTOR_TEST_KEY='synthetic-constructor-key',
                DLI_LLM_PROVIDERS=json.dumps([{'base_url': self.provider_urls[0],
                    'api_key_env': 'CONSTRUCTOR_TEST_KEY'}]))
            result = subprocess.run([sys.executable, '-c', program], cwd=SOURCE,
                env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), ['sdk', 'resource', 'transport', 'cancelled'])
            self.assertNotIn('synthetic-constructor-key', result.stderr)

    def post(self, url, **kwargs):
        return httpx.post(url + '/v1/chat/completions',
            headers={'x-dli-base-url': self.provider_urls[0]}, json=self.body(**kwargs),
            timeout=15, trust_env=False)

    def test_primary_only_default_and_explicit_deadlines(self):
        with self.wrapper(ttft=None, fallback=False) as url:
            health = httpx.get(url + '/health', trust_env=False).json()
            self.assertEqual(health['max_ttft_seconds'], 10)
            self.assertEqual(health['primary_only_ttft_seconds'], 45)
        self.servers[0].state['response_delay'] = .6
        with self.wrapper(ttft=.3, fallback=False) as url:
            response = self.post(url)
            self.assertEqual(response.status_code, 504, response.text)

    def test_eligible_fallback_selects_deadline_without_changing_primary_only_models(self):
        self.servers[0].state['response_delay'] = .6
        for configured, allowed in ((False, True), (True, False), (True, True)):
            with self.subTest(configured=configured, allowed=allowed):
                with self.wrapper(ttft=.3, primary_ttft=1.5, fallback=configured, fallback_allowed=allowed) as url:
                    response = self.post(url)
                    self.assertEqual(response.status_code, 200, response.text)
                    routing = response.json()['dli_routing']
                    self.assertEqual(routing['fallback_used'], configured and allowed)
                    self.assertEqual(routing['max_ttft_seconds'], .3 if configured and allowed else 1.5)

    def routing(self, url, model=MODEL, provider=0):
        response = httpx.get(url + '/v1/routing', params={'model': model},
            headers={'x-dli-base-url': self.provider_urls[provider]}, timeout=2, trust_env=False)
        self.assertEqual(response.status_code, 200, response.text)
        for secret in (*self.provider_urls, 'synthetic-circuit-primary', 'synthetic-circuit-fallback'):
            self.assertNotIn(secret, response.text)
        value = response.json()
        self.assertEqual(value['model'], model)
        self.assertIn(value['effective_route'], {'primary', 'fallback', 'unavailable'})
        self.assertIn('policy', value)
        return value

    def wait_eligible(self, url):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = self.routing(url)['primary']
            if state['state'] == 'half_open' and state['retry_after_seconds'] <= 0:
                return state
            time.sleep(min(state['retry_after_seconds'] + 0.02, 0.1))
        self.fail('cooldown did not expire within configured bound')

    def test_three_logical_failures_open_and_subsequent_request_skips_primary(self):
        self.mode(0, 503)
        with self.wrapper() as url:
            policy = self.routing(url)['policy']
            self.assertEqual(policy['failure_threshold'], 3)
            self.assertEqual(policy['base_cooldown_seconds'], 30)
            self.assertEqual(policy['max_cooldown_seconds'], 300)
            for expected in range(1, 4):
                self.post(url)
                state = self.routing(url)['primary']
                self.assertEqual(state['consecutive_failures'], expected)
                self.assertEqual(state['state'], 'open' if expected == 3 else 'closed')
            before = self.count(0)
            response = self.post(url)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()['dli_routing']['fallback_used'])
            self.assertEqual(self.count(0), before)

    def test_transient_http_classes_all_count(self):
        with self.wrapper() as url:
            for status in (408, 429, 500, 501, 502, 503, 504, 505, 507, 599):
                with self.subTest(status=status):
                    self.mode(0, status)
                    # HTTP classification must survive a response slower than the old 0.4s fixture.
                    self.servers[0].state['response_delay'] = 0.55 if status == 408 else 0
                    model = f'course/http-{status}'
                    self.post(url, model=model)
                    state = self.routing(url, model)['primary']
                    self.assertEqual(state['consecutive_failures'], 1)
                    self.assertEqual(state['last_failure'], 'transient_http')

    def test_transport_failures_count(self):
        self.mode(0, 'transport')
        with self.wrapper() as url:
            self.post(url)
            state = self.routing(url)['primary']
            self.assertEqual(state['consecutive_failures'], 1)
            self.assertEqual(state['last_failure'], 'transport_error')

    def test_failed_recovery_probes_escalate_to_cooldown_cap(self):
        self.mode(0, 503)
        with self.wrapper(threshold=1, cooldown=0.2, cap=0.8) as url:
            for tier in range(1, 5):
                if tier > 1:
                    self.wait_eligible(url)
                started = time.monotonic()
                self.post(url)
                state = self.routing(url)['primary']
                elapsed = time.monotonic() - started
                self.assertEqual(state['tier'], tier)
                expected = min(0.8, 0.2 * 2 ** (tier - 1))
                # HTTP fallback and status polling consume part of the cooldown.
                # Bound the remaining time by measured elapsed time, not a CPU-speed assumption.
                remaining = max(0, expected - elapsed)
                self.assertGreaterEqual(state['retry_after_seconds'], max(0, remaining - .001))
                self.assertLessEqual(state['retry_after_seconds'], expected)
                self.assertIn(state['state'], {'open'} if remaining > .001 else {'open', 'half_open'})
                self.assertFalse(state['probe_in_flight'])

    def test_permanent_http_failures_do_not_retry_or_count(self):
        self.servers[0].state['response_delay'] = 0.55
        with self.wrapper(retries=3) as url:
            for status in (400, 403):
                with self.subTest(status=status):
                    self.mode(0, status)
                    before = self.count(0)
                    self.assertEqual(self.post(url).status_code, status)
                    self.assertEqual(self.count(0) - before, 1)
                    state = self.routing(url)['primary']
                    self.assertEqual(state['state'], 'closed')
                    self.assertEqual(state['consecutive_failures'], 0)
            self.assertEqual(self.count(1), 0)

    def test_authentication_failure_falls_back_without_same_provider_retry(self):
        for mode in (401, 'sse401'):
            for streaming in (False, True):
                with self.subTest(mode=mode, streaming=streaming):
                    self.setUp()
                    self.mode(0, mode)
                    with self.wrapper(retries=3) as url:
                        response = self.post(url, stream=streaming)
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(response.headers['x-dli-route'], 'fallback')
                        self.assertIn('primary_authentication_failed', response.text)
                        self.assertIn('visible-fallback', response.text)
                        self.assertNotIn('synthetic-circuit-primary', response.text)
                        self.assertEqual([self.count(0), self.count(1)], [1, 1])
                        for provider, served in ((0, 'primary/model'), (1, 'fallback/model')):
                            self.assertEqual(self.servers[provider].state['requests'][0]['model'], served)
                        state = self.routing(url)
                        self.assertEqual(state['primary']['consecutive_failures'], 1)
                        self.assertEqual(state['primary']['last_failure'], 'provider_authentication_failed')
                        self.assertEqual(state['fallback']['consecutive_failures'], 0)
                        self.assertEqual(self.routing(url, OTHER_MODEL)['primary']['consecutive_failures'], 0)
                        if streaming:
                            self.assertIn('data: [DONE]', response.text)
                        else:
                            self.assertEqual(response.json()['dli_routing']['requested_model'], MODEL)
                            self.assertEqual(response.json()['dli_routing']['served_model'], 'fallback/model')

    def test_authentication_without_working_fallback_returns_provider_503(self):
        for fallback in (False, True):
            for streaming in (False, True):
                with self.subTest(fallback=fallback, streaming=streaming):
                    self.setUp()
                    self.mode(0, 401)
                    self.mode(1, 401)
                    with self.wrapper(retries=3, fallback=fallback) as url:
                        response = self.post(url, stream=streaming)
                        self.assertEqual(response.status_code, 503)
                        self.assertEqual(response.json()['dli_routing']['reason'], 'provider_authentication_failed')
                        self.assertNotIn('synthetic-circuit', response.text)
                        self.assertEqual([self.count(0), self.count(1)], [1, int(fallback)])
                        self.assertEqual(self.routing(url)['primary']['consecutive_failures'], 1)
                        if fallback:
                            self.assertEqual(self.routing(url)['fallback']['consecutive_failures'], 1)

    def test_authentication_failure_cannot_bypass_fallback_model_policy(self):
        self.mode(0, 401)
        with self.wrapper(retries=3, fallback_allowed=False) as url:
            response = self.post(url)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()['dli_routing']['reason'], 'fallback_model_not_allowed')
            self.assertEqual([self.count(0), self.count(1)], [1, 0])

    def test_retry_attempts_are_one_logical_failure(self):
        self.mode(0, 503)
        # Include the real 1s/2s backoff plus three scheduled HTTP attempts.
        with self.wrapper(retries=3, ttft=10) as url:
            self.post(url)
            self.assertEqual(self.count(0), 3)
            state = self.routing(url)['primary']
            self.assertEqual(state['consecutive_failures'], 1)
            self.assertEqual(state['state'], 'closed')

    def test_primary_and_fallback_circuits_and_canonical_models_are_independent(self):
        self.mode(0, 503)
        self.mode(1, 503)
        with self.wrapper(threshold=1) as url:
            self.post(url)
            # Directly target fallback too: do not assume primary errors select it.
            httpx.post(url + '/v1/chat/completions',
                headers={'x-dli-base-url': self.provider_urls[1]},
                json=self.body(), timeout=3, trust_env=False)
            status = self.routing(url)
            self.assertEqual(status['primary']['state'], 'open')
            self.assertEqual(status['fallback']['state'], 'open')
            self.assertEqual(status['effective_route'], 'unavailable')
            before = [self.count(0), self.count(1)]
            self.assertGreaterEqual(self.post(url).status_code, 400)
            self.assertEqual([self.count(0), self.count(1)], before)
            # Both canonical names map to the same served-model alias above.
            alternate = self.routing(url, OTHER_MODEL)
            self.assertEqual(alternate['primary']['state'], 'closed')
            self.assertEqual(alternate['fallback']['state'], 'closed')
            self.assertEqual(alternate['effective_route'], 'primary')

    def test_one_half_open_probe_remains_pending_until_complete_output(self):
        self.mode(0, 503)
        with self.wrapper(threshold=1, cooldown=0.2) as url:
            self.post(url)
            self.wait_eligible(url)
            self.mode(0, 'hold')
            before = self.count(0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                probe = pool.submit(self.post, url)
                self.assertTrue(self.servers[0].state['started'].wait(1))
                status = self.routing(url)['primary']
                self.assertEqual(status['state'], 'half_open')
                self.assertTrue(status['probe_in_flight'])
                self.assertEqual(status['consecutive_failures'], 1)
                competing = self.post(url)
                self.assertEqual(competing.status_code, 200, competing.text)
                self.assertTrue(competing.json()['dli_routing']['fallback_used'])
                self.assertEqual(self.count(0) - before, 1)
                self.servers[0].state['release'].set()
                self.assertEqual(probe.result(timeout=2).status_code, 200)
            status = self.routing(url)['primary']
            self.assertEqual(status['state'], 'closed')
            self.assertEqual(status['consecutive_failures'], 0)
            self.assertEqual(status['tier'], 0)
            self.assertFalse(status['probe_in_flight'])

    def test_cancelled_half_open_probe_releases_without_counting_failure(self):
        self.mode(0, 503)
        with self.wrapper(threshold=1, cooldown=0.2) as url:
            self.post(url)
            self.wait_eligible(url)
            self.mode(0, 'hold')
            with httpx.Client(timeout=3, trust_env=False) as client:
                with client.stream('POST', url + '/v1/chat/completions',
                        headers={'x-dli-base-url': self.provider_urls[0]},
                        json=self.body(stream=True)) as response:
                    self.assertEqual(response.status_code, 200)
                    for line in response.iter_lines():
                        if 'visible-primary' in line:
                            break
                    else:
                        self.fail('half-open probe did not expose primary content')
            self.assertTrue(self.servers[0].state['closed'].wait(1))
            for _ in range(30):
                status = self.routing(url)['primary']
                if not status['probe_in_flight']:
                    break
                time.sleep(0.02)
            self.assertFalse(status['probe_in_flight'])
            self.assertEqual(status['consecutive_failures'], 1)
            self.assertEqual(status['tier'], 1)
            self.mode(0, 'success')
            recovered = self.post(url)
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertFalse(recovered.json()['dli_routing']['fallback_used'])
            self.assertEqual(self.routing(url)['primary']['state'], 'closed')

    def test_optional_null_and_malformed_options_cannot_leak_half_open_probe(self):
        self.mode(0, 503)
        with self.wrapper(threshold=1, cooldown=0.2) as url:
            self.post(url)
            self.wait_eligible(url)
            self.mode(0, 'success')
            rejected = httpx.post(url + '/v1/chat/completions',
                headers={'x-dli-base-url': self.provider_urls[0]},
                json={**self.body(), 'stream_options': 42}, timeout=3, trust_env=False)
            self.assertEqual(rejected.status_code, 400)
            before = self.routing(url)['primary']
            self.assertFalse(before['probe_in_flight'])
            self.assertEqual(before['consecutive_failures'], 1)
            accepted = httpx.post(url + '/v1/chat/completions',
                headers={'x-dli-base-url': self.provider_urls[0]},
                json={**self.body(), 'stream_options': None}, timeout=3, trust_env=False)
            self.assertEqual(accepted.status_code, 200, accepted.text)
            after = self.routing(url)['primary']
            self.assertEqual(after['state'], 'closed')
            self.assertFalse(after['probe_in_flight'])

    def test_incomplete_output_counts_and_never_splices_fallback(self):
        # Deliberately short post-token stall deadline; ordinary HTTP setup uses the default TTFT.
        with self.wrapper(request_timeout=1) as url:
            self.mode(0, 'truncated')
            self.assertGreaterEqual(self.post(url).status_code, 400)
            self.assertEqual(self.routing(url)['primary']['consecutive_failures'], 1)
            self.mode(0, 'stall-after')
            response = self.post(url, stream=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn('visible-primary', response.text)
            self.assertIn('upstream_error', response.text)
            self.assertNotIn('visible-fallback', response.text)
            self.assertNotIn('[DONE]', response.text)
            self.assertEqual(self.count(1), 0)
            state = self.routing(url)['primary']
            self.assertEqual(state['consecutive_failures'], 2)
            self.assertEqual(state['last_failure'], 'response_timeout')
            self.mode(0, 'success')
            self.assertEqual(self.post(url).status_code, 200)
            self.assertEqual(self.routing(url)['primary']['consecutive_failures'], 0)


class CircuitStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SOURCE))
        try:
            from llm_circuit import Circuits
            cls.Circuits = Circuits
        finally:
            sys.path.pop(0)

    def test_concurrent_old_outcomes_cannot_close_new_open_generation(self):
        circuit = self.Circuits()
        tickets = [circuit.acquire('provider', MODEL) for _ in range(6)]
        for ticket in tickets[:3]:
            circuit.finish(ticket, 'failure', 'transient_http')
        opened = circuit.snapshot('provider', MODEL)
        self.assertEqual(opened['state'], 'open')
        self.assertEqual(opened['consecutive_failures'], 3)
        circuit.finish(tickets[3], 'success')
        circuit.finish(tickets[4], 'failure', 'transport_error')
        circuit.finish(tickets[5], 'cancel')
        # Repeated finish must neither decrement active twice nor reset state.
        circuit.finish(tickets[0], 'success')
        after = circuit.snapshot('provider', MODEL)
        self.assertEqual(after['state'], 'open')
        self.assertEqual(after['consecutive_failures'], 3)
        self.assertEqual(after['tier'], 1)
        self.assertIsNone(circuit.acquire('provider', MODEL))

    def test_default_backoff_doubles_to_cap_and_complete_probe_resets(self):
        now = [0.0]
        circuit = self.Circuits(clock=lambda: now[0])
        for _ in range(3):
            circuit.finish(circuit.acquire('provider', MODEL), 'failure', 'transient_http')
        for delay in (30, 60, 120, 240, 300, 300):
            self.assertEqual(circuit.snapshot('provider', MODEL)['retry_after_seconds'], delay)
            self.assertIsNone(circuit.acquire('provider', MODEL))
            now[0] += delay
            ticket = circuit.acquire('provider', MODEL)
            self.assertIsNotNone(ticket)
            self.assertTrue(circuit.snapshot('provider', MODEL)['probe_in_flight'])
            circuit.finish(ticket, 'failure', 'transient_http')
        now[0] += 300
        circuit.finish(circuit.acquire('provider', MODEL), 'success')
        status = circuit.snapshot('provider', MODEL)
        self.assertEqual(status['state'], 'closed')
        self.assertEqual(status['consecutive_failures'], 0)
        self.assertEqual(status['tier'], 0)

    def test_simultaneous_recovery_admits_one_probe_and_cancel_releases(self):
        now = [0.0]
        circuit = self.Circuits(threshold=1, clock=lambda: now[0])
        circuit.finish(circuit.acquire('provider', MODEL), 'failure', 'transport_error')
        now[0] = 30
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            tickets = list(pool.map(lambda _: circuit.acquire('provider', MODEL), range(32)))
        admitted = [ticket for ticket in tickets if ticket is not None]
        self.assertEqual(len(admitted), 1)
        circuit.finish(admitted[0], 'cancel')
        status = circuit.snapshot('provider', MODEL)
        self.assertEqual(status['consecutive_failures'], 1)
        self.assertEqual(status['tier'], 1)
        self.assertFalse(status['probe_in_flight'])
        self.assertIsNotNone(circuit.acquire('provider', MODEL))

    def test_capacity_does_not_evict_active_or_cooling_provider_models(self):
        circuit = self.Circuits(threshold=1)
        tickets = [circuit.acquire('provider', f'model-{index}') for index in range(512)]
        circuit.finish(tickets[0], 'failure', 'transient_http')
        with self.assertRaises(RuntimeError):
            circuit.acquire('provider', 'new-model')
        circuit.finish(tickets[1], 'cancel')
        replacement = circuit.acquire('provider', 'new-model')
        self.assertIsNotNone(replacement)
        self.assertEqual(circuit.snapshot('provider', 'model-0')['state'], 'open')
        for index in range(2, 512):
            self.assertEqual(circuit.snapshot('provider', f'model-{index}')['attempts'], 1)
        self.assertLessEqual(len(circuit.states), 512)


if __name__ == '__main__':
    unittest.main()
