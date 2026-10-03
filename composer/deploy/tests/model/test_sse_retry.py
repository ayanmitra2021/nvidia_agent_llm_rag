"""Synthetic HTTP proves retries stop at the first meaningful provider output."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import httpx
from test_llm_course_cache import MODEL
import llm_ttft


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        scenario = body['messages'][-1]['content']
        self.server.requests.append((scenario, time.monotonic()))
        count = sum(name == scenario for name, _ in self.server.requests)
        headers_error = scenario.startswith('http429') and count == 1
        self.send_response(429 if headers_error else 200)
        self.send_header('Content-Type', 'application/json' if headers_error else 'text/event-stream')
        self.send_header('Retry-After', '3' if scenario == 'deadline' else '0.05')
        self.end_headers()
        if headers_error:
            self.wfile.write(b'{"error":{"message":"synthetic retry"}}')
            return

        def chunk(content, finish=None):
            value = {'id': 'retry-fixture', 'model': MODEL, 'created': 1, 'object': 'chat.completion.chunk',
                'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': content}, 'finish_reason': finish}]}
            self.wfile.write(('data: ' + json.dumps(value) + '\n\n').encode())
            self.wfile.flush()

        if scenario.startswith('after'):
            chunk('visible-primary')
        failed = (scenario.startswith(('always', 'after', 'permanent')) or scenario == 'deadline'
                  or scenario.startswith('once') and count == 1)
        if failed:
            code = 400 if scenario.startswith('permanent') else 503
            self.wfile.write(('data: ' + json.dumps({'error': {'code': code,
                'type': 'service_unavailable' if code == 503 else 'invalid_request_error',
                'message': 'synthetic failure'}}) + '\n\n').encode())
            self.wfile.flush()
            return
        chunk('completed-' + self.server.label, 'stop')
        self.wfile.write(b'data: [DONE]\n\n')


class SSERetryTests(unittest.TestCase):
    @contextmanager
    def wrapper(self, ttft=4):
        servers = []
        for label in ('primary', 'fallback'):
            server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
            server.requests, server.label = [], label
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        urls = [f'http://127.0.0.1:{server.server_port}/v1' for server in servers]
        try:
            with tempfile.TemporaryDirectory() as directory:
                database = str(Path(directory) / 'requests.sqlite')
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                env = {'PATH': os.environ['PATH'], 'HOME': directory, 'PYTHONDONTWRITEBYTECODE': '1',
                    'LLM_STORE_PATH': database, 'LLM_COURSE_CACHE_ENABLED': 'false', 'LLM_CACHE_PACING': 'off',
                    'LLM_MAX_TTFT_SECONDS': str(ttft), 'LLM_RETRY_ATTEMPTS': '3',
                    'TEST_RETRY_KEY': 'synthetic-retry-key', 'TEST_RETRY_FALLBACK': urls[1],
                    'DLI_LLM_PROVIDERS': json.dumps([{'base_url': urls[0], 'api_key_env': 'TEST_RETRY_KEY'}]),
                    'DLI_LLM_FALLBACK_PROVIDER': json.dumps({'base_url_env': 'TEST_RETRY_FALLBACK', 'api_key_env': 'TEST_RETRY_KEY'})}
                program = f"import uvicorn; uvicorn.run('litellm_service:app', host='127.0.0.1', port={port}, log_config=None, access_log=False)"
                with tempfile.TemporaryFile() as log:
                    process = subprocess.Popen([sys.executable, '-c', program], cwd=Path(llm_ttft.__file__).parent,
                        env=env, stdout=log, stderr=log)
                    try:
                        with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=10,
                                          headers={'x-dli-base-url': urls[0]}) as client:
                            for _ in range(150):
                                self.assertIsNone(process.poll(), 'Retry fixture wrapper exited')
                                try:
                                    if client.get('/health').status_code == 200:
                                        break
                                except httpx.HTTPError:
                                    pass
                                time.sleep(.05)
                            else:
                                self.fail('Retry fixture wrapper did not start')
                            yield client, servers, database
                    finally:
                        process.terminate()
                        process.wait(timeout=10)
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()

    def post(self, client, scenario, streaming=False):
        return client.post('/v1/chat/completions', json={'model': MODEL,
            'messages': [{'role': 'user', 'content': scenario}], 'stream': streaming})

    def test_sse_503_retries_primary_then_caches_complete_json_and_stream(self):
        with self.wrapper() as (client, servers, _):
            for streaming in (False, True):
                scenario = 'once-' + str(streaming)
                response = self.post(client, scenario, streaming)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['x-dli-fallback'], 'false')
                self.assertIn('completed-primary', response.text)
                if streaming:
                    self.assertIn('data: [DONE]', response.text)
                attempts = [when for name, when in servers[0].requests if name == scenario]
                self.assertEqual(len(attempts), 2)
                self.assertGreaterEqual(attempts[1] - attempts[0], .045)
                before = len(servers[0].requests)
                again = self.post(client, scenario, streaming)
                self.assertEqual(again.headers['x-dli-cache'], 'HIT')
                self.assertEqual(len(servers[0].requests), before)
            self.assertEqual(servers[1].requests, [])

    def test_http_429_keeps_bounded_primary_retry(self):
        with self.wrapper() as (client, servers, _):
            self.assertEqual(self.post(client, 'http429').status_code, 200)
            self.assertEqual(len(servers[0].requests), 2)
            self.assertGreaterEqual(servers[0].requests[1][1] - servers[0].requests[0][1], .045)
            self.assertEqual(servers[1].requests, [])

    def test_persistent_sse_errors_exhaust_attempts_without_cache(self):
        with self.wrapper() as (client, servers, database):
            self.assertEqual(self.post(client, 'always').status_code, 503)
            self.assertEqual([len(server.requests) for server in servers], [3, 3])
            with sqlite3.connect(database) as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM cache').fetchone()[0], 0)

    def test_content_then_error_never_retries_or_splices_fallback(self):
        with self.wrapper() as (client, servers, database):
            for streaming in (False, True):
                scenario = 'after-' + str(streaming)
                response = self.post(client, scenario, streaming)
                self.assertEqual(response.status_code, 200 if streaming else 502)
                if streaming:
                    self.assertIn('visible-primary', response.text)
                    self.assertIn('upstream_error', response.text)
                    self.assertNotIn('[DONE]', response.text)
                self.assertEqual(sum(name == scenario for name, _ in servers[0].requests), 1)
            self.assertEqual(servers[1].requests, [])
            with sqlite3.connect(database) as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM cache').fetchone()[0], 0)

    def test_retry_after_cannot_extend_each_provider_ttft_deadline(self):
        with self.wrapper(ttft=.4) as (client, servers, _):
            started = time.monotonic()
            self.assertEqual(self.post(client, 'deadline').status_code, 504)
            self.assertLess(time.monotonic() - started, 1.8)
            self.assertEqual([len(server.requests) for server in servers], [1, 1])

    def test_permanent_sse_request_error_does_not_retry_or_fallback(self):
        with self.wrapper() as (client, servers, _):
            self.assertEqual(self.post(client, 'permanent').status_code, 400)
            self.assertEqual([len(server.requests) for server in servers], [1, 0])


if __name__ == '__main__':
    unittest.main()
