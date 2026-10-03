"""Real HTTP live-cache numeric interoperability and exact legacy migration."""
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
from test_llm_course_cache import MODEL  # Installs this repository's runtime path.
import llm_store
from llm_store import Store


class LiveNumericInterop(unittest.TestCase):
    def test_live_fill_then_numeric_variant_hits_chat_and_embeddings(self):
        class Provider(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.server.requests.append((self.path, body))
                self.send_response(200)
                if self.path.endswith('/embeddings'):
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"object":"list","data":[{"index":0,"embedding":[0.1,0.2]}]}')
                else:
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    chunk = {'id': 'numeric-fixture', 'model': MODEL, 'created': 1,
                        'object': 'chat.completion.chunk', 'choices': [{'index': 0,
                            'delta': {'role': 'assistant', 'content': 'Live fixture answer'}, 'finish_reason': 'stop'}]}
                    self.wfile.write(('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        server.requests = []
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        provider = f'http://127.0.0.1:{server.server_port}/v1'
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'runtime.sqlite'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            environment = {'PATH': os.environ['PATH'], 'HOME': directory, 'PYTHONDONTWRITEBYTECODE': '1',
                'LLM_STORE_PATH': str(database), 'LLM_COURSE_CACHE_ENABLED': 'false',
                'LLM_CACHE_PACING': 'off', 'LLM_RETRY_ATTEMPTS': '1', 'TEST_NUMERIC_KEY': 'synthetic-live-key',
                'DLI_LLM_PROVIDERS': json.dumps([{'base_url': provider, 'api_key_env': 'TEST_NUMERIC_KEY'}])}
            program = f"import uvicorn; uvicorn.run('litellm_service:app', host='127.0.0.1', port={port}, log_config=None, access_log=False)"
            with tempfile.TemporaryFile() as diagnostics:
                process = subprocess.Popen([sys.executable, '-c', program], cwd=Path(llm_store.__file__).parent,
                    env=environment, stdout=diagnostics, stderr=diagnostics)
                try:
                    with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=10,
                                      headers={'Content-Type': 'application/json', 'x-dli-base-url': provider}) as client:
                        for _ in range(150):
                            self.assertIsNone(process.poll(), 'Live numeric wrapper exited')
                            try:
                                if client.get('/health').status_code == 200:
                                    break
                            except httpx.HTTPError:
                                pass
                            time.sleep(.05)
                        else:
                            self.fail('Live numeric wrapper did not start')
                        chat = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'live numeric'}], 'temperature': 0.0}
                        for streaming in (False, True):
                            body = {**chat, 'stream': streaming}
                            before = len(server.requests)
                            for temperature, status in [(0.0, 'MISS'), (0, 'HIT')]:
                                response = client.post('/v1/chat/completions', content=json.dumps({**body, 'temperature': temperature}))
                                self.assertEqual(response.status_code, 200)
                                self.assertEqual(response.headers['x-dli-cache'], status)
                                if streaming:
                                    self.assertIn('data: [DONE]', response.text)
                                    self.assertNotIn('"error"', response.text)
                            self.assertEqual(len(server.requests), before + 1)
                            self.assertIs(type(server.requests[-1][1]['temperature']), float)
                        embedding = {'model': MODEL, 'input': 'embedding fixture', 'dimensions': 2.0}
                        before = len(server.requests)
                        for dimensions, status in [(2.0, 'MISS'), (2, 'HIT')]:
                            response = client.post('/v1/embeddings', content=json.dumps({**embedding, 'dimensions': dimensions}))
                            self.assertEqual(response.status_code, 200)
                            self.assertEqual(response.headers['x-dli-cache'], status)
                        self.assertEqual(len(server.requests), before + 1)
                        # A genuinely changed number is still a live miss.
                        for temperature in (0.1, 0.2):
                            response = client.post('/v1/chat/completions', content=json.dumps({**chat, 'temperature': temperature}))
                            self.assertEqual(response.headers['x-dli-cache'], 'MISS')
                        # Old exact lexical keys migrate without a provider request.
                        legacy = {**chat, 'messages': [{'role': 'user', 'content': 'legacy numeric'}]}
                        stored = {'id': 'legacy-fixture', 'model': MODEL, 'object': 'chat.completion', 'created': 1,
                            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Legacy answer'}, 'finish_reason': 'stop'}]}
                        store = Store(database, lambda value: value)
                        old_key = store.legacy_key(provider, legacy)
                        store.put(old_key, stored)
                        before = len(server.requests)
                        for temperature in (0.0, 0):
                            response = client.post('/v1/chat/completions', content=json.dumps({**legacy, 'temperature': temperature}))
                            self.assertEqual(response.headers['x-dli-cache'], 'HIT')
                            self.assertEqual(response.json()['id'], 'legacy-fixture')
                        self.assertEqual(len(server.requests), before)
                        self.assertIsNone(store.get(old_key))
                        self.assertEqual(store.get(store.key(provider, legacy)), stored)
                        store.db.close()
                finally:
                    process.terminate()
                    process.wait(timeout=10)

    def test_legacy_promotion_preserves_expiry_payload_and_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'requests.sqlite', lambda value: value)
            try:
                body = {'nested': [0.0, True, '0.0']}
                old = store.legacy_key('url', body)
                key = store.key('url', body)
                self.assertNotEqual(old, key)
                self.assertEqual(key, store.key('url', {'nested': [0, True, '0.0']}))
                for variant in ({'nested': [0.1, True, '0.0']}, {'nested': [0, 1, '0.0']}, {'nested': [0, True, '0']}):
                    self.assertNotEqual(key, store.key('url', variant))
                store.put(old, {'number': 0.0})
                before = store.db.execute('SELECT value,expires,size FROM cache WHERE key=?', (old,)).fetchone()
                store.record('POST', 'url', body)
                self.assertEqual(store.get(key, old), {'number': 0.0})
                after = store.db.execute('SELECT value,expires,size FROM cache WHERE key=?', (key,)).fetchone()
                self.assertEqual(after, before)
                journal = store.db.execute('SELECT body FROM requests').fetchone()[0]
                self.assertIn('0.0', journal)
                self.assertIs(type(json.loads(journal)['nested'][0]), float)
                self.assertIsNone(store.get(old))
            finally:
                store.db.close()


if __name__ == '__main__':
    unittest.main()
