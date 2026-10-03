"""Run with Python 3.11 and the wrapper dependencies; no real credentials required."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
import sqlite3
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import httpx

SOURCE = Path(__file__).resolve().parents[2] / "services/model"
KEYS = ["synthetic-provider-alpha-secret", "synthetic-provider-beta-secret"]
MODELS = ["nvidia/nemotron-3.5-lightning-30b-a3b", "test/second"]
REQUESTS = []
FAILURES = {}
DISCONNECTED = threading.Event()


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        index = self.server.server_port - 9101
        if self.headers.get("Authorization") != "Bearer " + KEYS[index]:
            return self.reply(401, {"error": "credential mismatch"})
        models = [model.replace("/", "-") if index == 1 and model == MODELS[0] else model for model in MODELS]
        self.reply(200, {"data": [{"id": model, "object": "model"} for model in models]})

    def do_POST(self):
        index = self.server.server_port - 9101
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append((index, self.headers.get("Authorization"), body))
        if self.headers.get("Authorization") != "Bearer " + KEYS[index]:
            return self.reply(401, {"error": "credential mismatch"})
        message = body.get("messages", [{}])[-1].get("content", "")
        response_model = KEYS[index] if message == "echo-secret-model" else body.get("model")
        choice_count = 1 if message == "truncated-missing" else body.get("n", 1)
        if message == "retry":
            FAILURES[index] = FAILURES.get(index, 0) + 1
            if FAILURES[index] < 3:
                return self.reply(429, {"error": {"message": KEYS[index]}})
        if message == "error":
            return self.reply(401, {"error": {"message": KEYS[index]}})
        if message == "redirect":
            self.send_response(307)
            self.send_header("Location", "http://127.0.0.1:9102/v1/chat/completions")
            self.end_headers()
            return
        if self.path.endswith("embeddings"):
            return self.reply(200, {"object": "list", "data": [{"embedding": [0.2, 0.4], "index": 0}]})
        text = KEYS[index] if message == "echo-secret" else "provider-" + str(index)
        if message == "slow":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for _ in range(1000):
                    self.wfile.write(("data: " + json.dumps({"id": "slow", "object": "chat.completion.chunk", "created": 1, "model": response_model, "choices": [{"index": 0, "delta": {"content": "x" * 1000}, "finish_reason": None}]}) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                DISCONNECTED.set()
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta in [{"role": "assistant", "content": text, **({"reasoning": "provider trace"} if index == 1 else {})}, {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "check", "arguments": "{}"}}]}]:
                chunk = {"id": "test", "object": "chat.completion.chunk", "created": 1, "model": response_model, "choices": [{"index": i, "delta": delta, "finish_reason": None} for i in range(choice_count)]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(0.05)
            if message == "truncated-eof":
                return
            if message == "truncated-done":
                self.wfile.write(b'data: [DONE]\n\n')
                return
            final = {"id": "test", "object": "chat.completion.chunk", "created": 1, "model": response_model, "choices": [{"index": i, "delta": {}, "finish_reason": None if message == "truncated-second" and i == 1 else "stop"} for i in range(choice_count)], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}
            self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())
            self.wfile.write(b'data: [DONE]\n\n')
            return
        self.reply(200, {"id": "test", "object": "chat.completion", "created": 1, "model": response_model,
            "choices": [{"index": i, "message": {"role": "assistant", "content": text, **({"reasoning": "provider trace"} if index == 1 else {})}, "finish_reason": "stop"} for i in range(choice_count)],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}})


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store_dir = tempfile.TemporaryDirectory()
        cls.store_path = str(Path(cls.store_dir.name) / "requests.sqlite")
        from test_llm_course_cache import fixture_bundle
        cls.bundle_path, cls.bundle_body = fixture_bundle(cls.store_dir.name)
        cls.servers = []
        cls.processes = []
        for port in [9101, 9102]:
            server = ThreadingHTTPServer(("127.0.0.1", port), Provider)
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, daemon=True).start()
            cls.servers.append(server)

    @classmethod
    def start_services(cls, **overrides):
        # Keep production circuit defaults; each independent scenario owns its process state.
        base_env = {key: value for key, value in os.environ.items()
            if not key.startswith(("LLM_", "DLI_LLM_", "NEBIUS_", "NVIDIA_", "OPENAI_"))}
        env = dict(base_env, LLM_CACHE_PACING="off", LLM_STORE_PATH=cls.store_path, LLM_COURSE_CACHE_PATH=str(cls.bundle_path), LLM_COURSE_CACHE_ENABLED="false", LLM_CACHE_RESPECT_HEADERS="true", DLI_LLM_PROVIDERS=json.dumps([
            {**({"base_url_env": "TEST_SECOND_BASE", "reasoning_budget_parameter": "thinking_token_budget", "reasoning_response_field": "reasoning"} if i else {"base_url": f"http://127.0.0.1:{9101+i}/v1/"}), "api_key_env": f"TEST_KEY_{i}", "model_aliases": {"course-test-alias": MODELS[1], **({MODELS[0]: MODELS[0].replace("/", "-")} if i else {})}} for i in range(2)
        ]), TEST_SECOND_BASE="http://127.0.0.1:9102/v1", LLM_DEFAULT_BASE_URL_ENV="TEST_SECOND_BASE", TEST_KEY_0=KEYS[0], TEST_KEY_1=KEYS[1], DEBUG="1", LLM_DEBUG="1", LITELLM_LOG="DEBUG", OPENAI_LOG="debug")
        env.update(overrides)
        cls.processes.append(subprocess.Popen([sys.executable, "litellm_wrapper.py"], cwd=SOURCE, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        facade_env = dict(base_env, LLM_BASE_URL="http://127.0.0.1:9101/v1", LLM_WRAPPER_URL="http://127.0.0.1:9001", DEBUG="1", LLM_DEBUG="1")
        cls.processes.append(subprocess.Popen([sys.executable, "llm_client.py"], cwd=SOURCE, env=facade_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        cls.wait("http://127.0.0.1:9000/health")

    @classmethod
    def wait(cls, url, status=200):
        for _ in range(120):
            try:
                if httpx.get(url, timeout=1).status_code == status:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise AssertionError("service startup failed")

    @classmethod
    def stop_services(cls):
        while cls.processes:
            process = cls.processes.pop()
            process.terminate()
            stdout, stderr = process.communicate(timeout=10)
            assert not stdout, "unexpected standard output"
            assert all(secret.encode() not in stderr for secret in KEYS), "credential in diagnostic output"
            assert all(line.startswith(b"llm ") for line in stderr.splitlines()), "unstructured diagnostics"
    @classmethod
    def tearDownClass(cls):
        cls.stop_services()
        for server in cls.servers:
            server.shutdown()
            server.server_close()
        cls.store_dir.cleanup()

    def setUp(self):
        self.addCleanup(self.stop_services)
        self.start_services()
        with sqlite3.connect(self.store_path) as connection:
            connection.execute("DELETE FROM cache")
        self.assertEqual(httpx.put("http://127.0.0.1:9000/v1/base-url", json={"base_url": "http://127.0.0.1:9101/v1"}).status_code, 200)
        httpx.post("http://127.0.0.1:9000/v1/course/agent-models", json={"model": MODELS[0]})

    def chat(self, **kwargs):
        return httpx.post("http://127.0.0.1:9000/v1/chat/completions", json={"model": "course/agent", "messages": [{"role": "user", "content": "hello"}], **kwargs}, timeout=20)

    def test_chat_alias_usage_and_lightning_default(self):
        result = self.chat()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers["x-dli-fallback"], "false")
        self.assertEqual(result.json()["usage"]["total_tokens"], 6)
        self.assertEqual(REQUESTS[-1][2]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(REQUESTS[-1][2]["model"], MODELS[0])

    def test_vendored_bundle_opt_in_and_live_fallback(self):
        arguments = {"messages": self.bundle_body["messages"], "use_course_cache": True}
        before = len(REQUESTS)
        for port in (9101, 9102):
            httpx.put("http://127.0.0.1:9000/v1/base-url", json={"base_url": f"http://127.0.0.1:{port}/v1"})
            response = self.chat(**arguments)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], "recorded-a")
            self.assertEqual(response.json()["dli_cache"]["status"], "BUNDLE_HIT")
            self.assertEqual(response.json()["dli_cache"]["origin_provider"], "https://integrate.api.nvidia.com/v1")
            self.assertFalse(response.json()["dli_cache"]["upstream_request"])
        self.assertEqual(len(REQUESTS), before)
        response = self.chat(**{**arguments, "temperature": .4})
        self.assertEqual(response.json()["dli_cache"]["status"], "MISS")
        self.assertEqual(len(REQUESTS), before + 1)
        self.assertNotIn("use_course_cache", REQUESTS[-1][2])
        response = self.chat(messages=self.bundle_body["messages"])
        self.assertEqual(response.json()["dli_cache"]["status"], "MISS")
        self.assertEqual(len(REQUESTS), before + 2)

    def test_recorded_samples_stream_and_explicit_live_controls(self):
        arguments = {"messages": self.bundle_body["messages"], "use_course_cache": True, "cache_sample": True}
        before = len(REQUESTS)
        for index, expected in [(0, "recorded-a"), (1, "recorded-b"), (2, "recorded-a")]:
            response = self.chat(**arguments, cache_sample_index=index)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], expected)
            self.assertEqual(response.json()["dli_cache"]["sample_count"], 2)
        stream = self.chat(**arguments, stream=True, cache_sample_index=1)
        self.assertIn('"content": "recorded-b"', stream.text)
        self.assertIn('"finish_reason": "stop"', stream.text)
        self.assertTrue(stream.text.rstrip().endswith("data: [DONE]"))
        self.assertEqual(len(REQUESTS), before)
        live = self.chat(**arguments, use_cache=False)
        self.assertEqual(live.json()["dli_cache"]["status"], "BYPASS")
        for control in ["no-store", "no-cache"]:
            response = httpx.post("http://127.0.0.1:9000/v1/chat/completions", headers={"Cache-Control": control},
                json={**self.bundle_body, "use_course_cache": True})
            self.assertEqual(response.json()["dli_cache"]["status"], "BYPASS")
        self.assertEqual(len(REQUESTS), before + 3)
        self.assertFalse({"use_course_cache", "cache_sample", "cache_sample_index"}.intersection(REQUESTS[-1][2]))

    def test_course_cache_controls_require_correct_types(self):
        before = len(REQUESTS)
        for body in [{"use_course_cache": "true"}, {"cache_sample": 1}, {"cache_sample_index": True},
                {"cache_sample": True, "cache_sample_index": -1}, {"cache_sample_index": 0},
                *[{"cache_pacing": value} for value in (None, True, 20, {}, "unknown")]]:
            self.assertEqual(self.chat(**body).status_code, 400)
        self.assertEqual(len(REQUESTS), before)

    def test_legacy_guidance_exact_replay_and_portable_facade_contract(self):
        import copy
        import gzip
        from llm_course_cache import package_records
        from llm_request import structured_output
        manifest = Path(str(self.bundle_path) + '.manifest.json')
        original, original_manifest = self.bundle_path.read_bytes(), manifest.read_bytes()
        self.addCleanup(self.bundle_path.write_bytes, original)
        self.addCleanup(manifest.write_bytes, original_manifest)
        schema = {'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer']}
        legacy = {**self.bundle_body, 'guided_json': schema}
        portable = structured_output(legacy)
        records = json.loads(gzip.decompress(original))['entries']
        exemplar = next(record for record in records if not record['request'].get('stream'))
        for body, text in ((legacy, 'legacy-record'), (portable, 'portable-record')):
            record = copy.deepcopy(exemplar)
            record['request'] = body
            record['response']['choices'][0]['message']['content'] = text
            records.append(record)
        package_records(records, self.bundle_path, {'fixture': True})
        capture = Path(self.store_dir.name) / 'guidance-capture.jsonl'
        self.stop_services()
        self.start_services(LLM_CAPTURE_COURSE_CACHE=str(capture),
            LLM_CAPTURE_PROVIDER_LABELS=json.dumps({'http://127.0.0.1:9101/v1': 'nvidia'}))
        def direct(body, **controls):
            return httpx.post('http://127.0.0.1:9001/v1/chat/completions',
                headers={'x-dli-base-url': 'http://127.0.0.1:9101/v1'},
                json={**body, 'use_course_cache': True, **controls}, timeout=20)
        before = len(REQUESTS)
        for body, expected in ((legacy, 'legacy-record'), (portable, 'portable-record')):
            response = direct(body)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['x-dli-cache'], 'BUNDLE_HIT')
            self.assertEqual(response.json()['choices'][0]['message']['content'], expected)
        for guidance in ({'guided_json': schema}, {'nvext': {'guided_json': schema}},
                         {'response_format': portable['response_format']}):
            response = self.chat(messages=self.bundle_body['messages'], use_course_cache=True, **guidance)
            self.assertEqual(response.headers['x-dli-cache'], 'BUNDLE_HIT')
            self.assertEqual(response.json()['choices'][0]['message']['content'], 'portable-record')
        self.assertEqual(len(REQUESTS), before)
        for extra in ({'guided_json': 'bad'}, {'guided_json': schema, 'response_format': {'type': 'json_object'}},
                      {'guided_json': schema, 'nvext': {'guided_json': schema}},
                      {'guided_json': schema, 'unknown_option': True}):
            self.assertEqual(direct({**self.bundle_body, **extra}).status_code, 400)
            self.assertEqual(self.chat(messages=self.bundle_body['messages'], use_course_cache=True, **extra).status_code, 400)
        self.assertEqual(len(REQUESTS), before)
        for changed in ({'guided_json': {**schema, 'additionalProperties': False}}, {'temperature': 0.25}):
            response = self.chat(messages=self.bundle_body['messages'], use_course_cache=True,
                                 **{**{'guided_json': schema}, **changed})
            self.assertEqual(response.status_code, 200)
            self.assertNotEqual(response.headers['x-dli-cache'], 'BUNDLE_HIT')
        response = direct(legacy, use_cache=False)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('guided_json', REQUESTS[-1][2])
        self.assertEqual(REQUESTS[-1][2]['response_format'], portable['response_format'])
        recorded = json.loads(capture.read_text().splitlines()[-1])
        self.assertEqual(recorded['request'], legacy)
        self.assertEqual(recorded['provider_request']['response_format'], portable['response_format'])
        self.assertNotIn('guided_json', recorded['provider_request'])

    def test_pacing_applies_to_hits_without_changing_identity_or_live_inputs(self):
        arguments = {"messages": self.bundle_body["messages"], "use_course_cache": True}
        before = len(REQUESTS)
        for stream in (False, True):
            start = time.monotonic()
            response = self.chat(**arguments, **({"stream": True} if stream else {}), cache_pacing="reading")
            elapsed = time.monotonic() - start
            self.assertEqual(response.headers["x-dli-cache"], "BUNDLE_HIT")
            if stream:
                chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                    if line.startswith("data: {")]
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
                self.assertTrue(response.text.rstrip().endswith("data: [DONE]"))
                metadata = chunks[0]["dli_cache"]
            else:
                metadata = response.json()["dli_cache"]
            self.assertEqual(metadata["pacing"]["profile"], "reading")
            self.assertEqual(metadata["pacing"]["initial_delay_seconds"], 1)
            delay = metadata["pacing"]["total_delay_seconds"]
            self.assertGreaterEqual(delay, 1)
            self.assertGreaterEqual(elapsed, delay * .9)
            self.assertFalse(metadata["upstream_request"])
        self.assertEqual(len(REQUESTS), before)
        novel = {"messages": [{"role": "user", "content": "pacing-control-boundary"}]}
        self.assertEqual(self.chat(**novel, cache_pacing="reading").json()["dli_cache"]["status"], "MISS")
        hit = self.chat(**novel, cache_pacing="off").json()["dli_cache"]
        self.assertEqual(hit["status"], "HIT")
        self.assertEqual(hit["pacing"]["delay_seconds"], 0)
        self.assertEqual(hit["pacing"]["initial_delay_seconds"], 0)
        self.assertEqual(hit["pacing"]["total_delay_seconds"], 0)
        self.assertEqual(self.chat(**novel, cache_pacing="reading", use_cache=False).json()["dli_cache"]["status"], "BYPASS")
        self.assertNotIn("cache_pacing", REQUESTS[-1][2])
        with sqlite3.connect(self.store_path) as connection:
            rows = connection.execute("SELECT body FROM requests WHERE method='POST'").fetchall()
        self.assertTrue(rows)
        self.assertFalse(any("cache_pacing" in json.loads(row[0]) for row in rows))

    def test_unavailable_bundle_keeps_live_fallback(self):
        import gzip
        import hashlib
        from test_llm_course_cache import fixture_bundle
        for mutation in ["missing-manifest", "corrupt-gzip", "bad-entry", "bad-manifest"]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, body = fixture_bundle(directory)
                manifest_path = Path(str(path) + ".manifest.json")
                if mutation == "missing-manifest": manifest_path.unlink()
                elif mutation == "bad-manifest": manifest_path.write_text("not json")
                else:
                    manifest = json.loads(manifest_path.read_text())
                    if mutation == "bad-entry":
                        data = json.loads(gzip.decompress(path.read_bytes()))
                        data["entries"][0]["id"] = "malformed"
                        artifact = gzip.compress(json.dumps(data).encode())
                    else: artifact = b"invalid gzip data"
                    path.write_bytes(artifact)
                    manifest["sha256"] = hashlib.sha256(artifact).hexdigest()
                    manifest_path.write_text(json.dumps(manifest))
                env = dict(os.environ, LLM_STORE_PATH=directory + "/store.sqlite", LLM_COURSE_CACHE_PATH=str(path),
                    DLI_LLM_PROVIDERS=json.dumps([{"base_url": "http://127.0.0.1:9101/v1", "api_key_env": "TEST_KEY_0"}]), TEST_KEY_0=KEYS[0])
                process = subprocess.Popen([sys.executable, "-c", "from llm_output import protect_output; protect_output(); import uvicorn; uvicorn.run('litellm_service:app', host='127.0.0.1', port=9011, log_config=None, access_log=False)"],
                    cwd=SOURCE, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    self.wait("http://127.0.0.1:9011/health", status=503)
                    health = httpx.get("http://127.0.0.1:9011/health").json()
                    self.assertFalse(health["course_cache"]["available"])
                    before = len(REQUESTS)
                    response = httpx.post("http://127.0.0.1:9011/v1/chat/completions", headers={"x-dli-base-url": "http://127.0.0.1:9101/v1"},
                        json={**body, "use_course_cache": True}, timeout=20)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["dli_cache"]["reason"], "course_bundle_unavailable")
                    self.assertEqual(len(REQUESTS), before + 1)
                finally:
                    process.terminate()
                    output, error = process.communicate(timeout=10)
                    self.assertFalse(output)
                    self.assertNotIn(KEYS[0].encode(), error)

    def test_switch_selects_matching_credential(self):
        result = httpx.put("http://127.0.0.1:9000/v1/base-url", json={"base_url": "http://127.0.0.1:9102/v1/"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.chat().json()["choices"][0]["message"]["content"], "provider-1")
        self.assertEqual(REQUESTS[-1][1], "Bearer " + KEYS[1])

    def test_capture_failure_preserves_success_and_stream_completion(self):
        for error_type in ["PermissionError", "ValueError"]:
            with self.subTest(error_type=error_type), tempfile.TemporaryDirectory() as directory:
                env = dict(os.environ, LLM_STORE_PATH=directory + "/store.sqlite",
                    DLI_LLM_PROVIDERS=json.dumps([{"base_url": "http://127.0.0.1:9101/v1", "api_key_env": "TEST_KEY_0"}]), TEST_KEY_0=KEYS[0])
                program = ("from llm_output import protect_output; protect_output(); import litellm_service as s; import uvicorn\n"
                    "def rejected_capture(*args): raise " + error_type + "('private capture path must not escape')\n"
                    "s.record_response = rejected_capture\n"
                    "uvicorn.run(s.app, host='127.0.0.1', port=9011, log_config=None, access_log=False)")
                process = subprocess.Popen([sys.executable, "-c", program], cwd=SOURCE, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    self.wait("http://127.0.0.1:9011/health")
                    for streaming in [False, True]:
                        response = httpx.post("http://127.0.0.1:9011/v1/chat/completions",
                            headers={"x-dli-base-url": "http://127.0.0.1:9101/v1"},
                            json={"model": MODELS[0], "messages": [{"role": "user", "content": "capture-failure"}], "stream": streaming}, timeout=20)
                        self.assertEqual(response.status_code, 200)
                        self.assertIn("provider-0", response.text)
                        self.assertNotIn("stream interrupted", response.text)
                        self.assertNotIn("private capture path", response.text)
                        if streaming: self.assertTrue(response.text.rstrip().endswith("data: [DONE]"))
                finally:
                    process.terminate()
                    output, error = process.communicate(timeout=10)
                    self.assertFalse(output)
                    self.assertNotIn(b"private capture path", error)

    def test_failed_switch_retains_selection(self):
        for url in ["https://unconfigured.invalid/v1", "https://user:password@example.com/v1", "https://example.com/v1?key=bad"]:
            self.assertGreaterEqual(httpx.put("http://127.0.0.1:9000/v1/base-url", json={"base_url": url}).status_code, 400)
        self.assertEqual(self.chat().json()["choices"][0]["message"]["content"], "provider-0")

    def test_no_provider_list(self):
        spec = httpx.get("http://127.0.0.1:9000/openapi.json").json()
        self.assertFalse(any("provider" in path or "backends" in path for path in spec["paths"]))
        self.assertEqual(set(httpx.get("http://127.0.0.1:9000/v1/base-url").json()), {"base_url"})
        source = (SOURCE / "llm_client.py").read_text()
        self.assertNotIn("API_KEY", source)
        self.assertNotIn("PROVIDERS", source)

    def test_override_controls_rejected(self):
        for name in ["api_key", "api_base", "base_url", "extra_headers", "extra_body", "callbacks", "client", "custom_llm_provider"]:
            self.assertEqual(self.chat(**{name: "malicious"}).status_code, 400, name)

    def test_errors_and_success_do_not_echo_key(self):
        for text, status in [("error", 503), ("echo-secret", 200)]:
            result = self.chat(messages=[{"role": "user", "content": text}])
            self.assertEqual(result.status_code, status)
            if text == "error":
                self.assertEqual(result.json()["dli_routing"]["reason"], "provider_authentication_failed")
            for key in KEYS:
                self.assertNotIn(key, result.text)

    def test_provider_and_request_model_metadata_are_redacted(self):
        for streaming in (False, True):
            response = self.chat(messages=[{"role": "user", "content": "echo-secret-model"}], stream=streaming, use_cache=False)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("[REDACTED]", response.text)
            for key in KEYS:
                self.assertNotIn(key, response.text)
            response = httpx.post("http://127.0.0.1:9001/v1/chat/completions",
                headers={"x-dli-base-url": "http://127.0.0.1:9101/v1"},
                json={"model": KEYS[0], "messages": [{"role": "user", "content": "hello"}], "stream": streaming, "use_cache": False}, timeout=20)
            self.assertEqual(response.status_code, 200, response.text)
            for key in KEYS:
                self.assertNotIn(key, response.text)

    def test_truncated_streams_never_succeed_or_enter_cache(self):
        for scenario, n in (("truncated-eof", 1), ("truncated-done", 1), ("truncated-second", 2), ("truncated-missing", 2)):
            for streaming in (False, True):
                with self.subTest(scenario=scenario, stream=streaming):
                    self.stop_services()
                    self.start_services()
                    before = len(REQUESTS)
                    for _ in range(2):
                        response = self.chat(messages=[{"role": "user", "content": scenario}], n=n, stream=streaming, temperature=0, use_cache=True)
                        if streaming:
                            self.assertEqual(response.status_code, 200)
                            self.assertIn('"error"', response.text)
                            self.assertNotIn("data: [DONE]", response.text)
                        else:
                            self.assertEqual(response.status_code, 502, response.text)
                    self.assertEqual(len(REQUESTS), before + 2)
                    with sqlite3.connect(self.store_path) as connection:
                        self.assertEqual(connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0], 0)

    def test_retry(self):
        FAILURES.clear()
        result = self.chat(messages=[{"role": "user", "content": "retry"}])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(FAILURES[0], 3)

    def test_no_redirect_credential_forwarding(self):
        start = len(REQUESTS)
        self.assertGreaterEqual(self.chat(messages=[{"role": "user", "content": "redirect"}]).status_code, 400)
        self.assertFalse(any(index == 1 for index, _, _ in REQUESTS[start:]))

    def test_stream_preserves_tools_and_done(self):
        result = self.chat(stream=True, tools=[{"type": "function", "function": {"name": "check", "parameters": {"type": "object"}}}])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers["x-dli-fallback"], "false")
        self.assertIn('"tool_calls"', result.text)
        self.assertIn("data: [DONE]", result.text)

    def test_completions_both_modes(self):
        for streaming in [False, True]:
            result = httpx.post("http://127.0.0.1:9000/v1/completions", json={"model": "course/agent", "prompt": ["hello", "world"], "stream": streaming}, timeout=20)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.headers["x-dli-fallback"], "false")
            self.assertIn("text_completion", result.text)
            self.assertIn('"text"', result.text)

    def test_embeddings(self):
        result = httpx.post("http://127.0.0.1:9000/v1/embeddings", json={"model": MODELS[1], "input": "hello"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["data"][0]["embedding"], [0.2, 0.4])

    def test_model_catalog_and_selector(self):
        self.assertEqual(len(httpx.get("http://127.0.0.1:9000/v1/models?scope=all").json()["data"]), 3)
        self.assertEqual(httpx.post("http://127.0.0.1:9000/v1/course/agent-models", json={"model": MODELS[1]}).status_code, 200)
        self.assertEqual(self.chat().status_code, 200)
        self.assertEqual(REQUESTS[-1][2]["model"], MODELS[1])

    def test_concurrent_switches_cannot_mix_keys(self):
        def run(index):
            httpx.put("http://127.0.0.1:9000/v1/base-url", json={"base_url": f"http://127.0.0.1:{9101 + index % 2}/v1"}, timeout=20)
            return self.chat().status_code
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(run, range(8))), [200] * 8)
        self.assertTrue(all(auth == "Bearer " + KEYS[index] for index, auth, _ in REQUESTS))

    def test_reasoning_budget_preserved(self):
        result = self.chat(reasoning_budget=1024, chat_template_kwargs={"enable_thinking": True})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(REQUESTS[-1][2]["reasoning_budget"], 1024)
        self.assertEqual(REQUESTS[-1][2]["chat_template_kwargs"], {"enable_thinking": True})

    def test_env_provider_alias_and_reasoning_json_stream_and_replay(self):
        selected = httpx.get("http://127.0.0.1:9001/v1/default-base-url")
        self.assertEqual(selected.json(), {"base_url": "http://127.0.0.1:9102/v1"})
        httpx.put("http://127.0.0.1:9000/v1/base-url", json=selected.json())
        for streaming in (False, True):
            response = self.chat(reasoning_budget=321, stream=streaming, use_cache=False, chat_template_kwargs={"enable_thinking": True})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(REQUESTS[-1][2]["model"], MODELS[0].replace("/", "-"))
            self.assertEqual(REQUESTS[-1][2]["thinking_token_budget"], 321)
            self.assertNotIn("reasoning_budget", REQUESTS[-1][2])
            if streaming:
                values=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
                messages=[choice.get("delta", {}) for value in values for choice in value.get("choices", [])]
                self.assertTrue(any(message.get("reasoning_content")=="provider trace" for message in messages))
            else:
                self.assertEqual(response.json()["choices"][0]["message"]["reasoning_content"], "provider trace")
        first=self.chat(reasoning_budget=322)
        second=self.chat(reasoning_budget=322)
        self.assertEqual(second.headers["x-dli-cache"], "HIT")
        self.assertEqual(second.json()["choices"][0]["message"]["reasoning_content"], "provider trace")

    def test_explicit_alias(self):
        result = self.chat(model="course-test-alias")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(REQUESTS[-1][2]["model"], MODELS[1])

    def test_exact_cache_and_perturbations(self):
        body = {"temperature": 0, "seed": 1111}
        start = len(REQUESTS)
        first, second = self.chat(**body), self.chat(**body)
        self.assertEqual(first.headers["x-dli-cache"], "MISS")
        self.assertEqual(second.headers["x-dli-cache"], "HIT")
        self.assertEqual(len(REQUESTS) - start, 1)
        self.assertTrue(first.json()["dli_cache"]["upstream_request"])
        self.assertFalse(second.json()["dli_cache"]["upstream_request"])
        self.assertTrue(second.json()["dli_cache"]["provider_usage_reused"])
        self.assertIn("dli_routing", first.json())
        self.assertNotIn("dli_routing", second.json())
        self.assertEqual(
            {k:v for k,v in first.json().items() if k not in ("dli_cache", "dli_routing")},
            {k:v for k,v in second.json().items() if k not in ("dli_cache", "dli_routing")},
        )
        self.assertEqual(self.chat(**{**body, "seed": 1112}).headers["x-dli-cache"], "MISS")
        self.assertEqual(self.chat(**body, messages=[{"role": "user", "content": "hello!"}]).headers["x-dli-cache"], "MISS")
        self.assertEqual(self.chat(**body, max_completion_tokens=9).headers["x-dli-cache"], "MISS")
        self.assertEqual(REQUESTS[-1][2]["max_completion_tokens"], 9)
        self.assertNotIn("max_tokens", REQUESTS[-1][2])

    def test_completion_limit_is_modern_on_actual_upstream_requests(self):
        for provider in (9101, 9102):
            self.assertEqual(httpx.put("http://127.0.0.1:9000/v1/base-url",
                json={"base_url": f"http://127.0.0.1:{provider}/v1"}).status_code, 200)
            for limits, expected in (({"max_tokens": 7}, 7), ({"max_completion_tokens": 9}, 9),
                    ({"max_tokens": 7, "max_completion_tokens": 9}, 9)):
                for streaming in (False, True):
                    with self.subTest(provider=provider, limits=limits, stream=streaming):
                        response = self.chat(**limits, stream=streaming, use_cache=False)
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(REQUESTS[-1][2]["max_completion_tokens"], expected)
                        self.assertNotIn("max_tokens", REQUESTS[-1][2])

    def test_cache_defaults_cover_sampling_streams_and_multiple_choices(self):
        for stream in [False, True]:
            for n in [1, 2]:
                for temperature in [None, 0, 0.1, 0.7, 1, 2]:
                    body = {"stream": stream, "n": n}
                    if temperature is not None:
                        body["temperature"] = temperature
                    start = len(REQUESTS)
                    first, second = self.chat(**body), self.chat(**body)
                    self.assertEqual(first.headers["x-dli-cache"], "MISS")
                    self.assertEqual(second.headers["x-dli-cache"], "HIT")
                    self.assertEqual(len(REQUESTS) - start, 1)
                    if not stream:
                        self.assertEqual(len(second.json()["choices"]), n)

    def test_embedding_cache_and_explicit_opt_out(self):
        body = {"model": MODELS[0], "input": "embedding sample"}
        endpoint = "http://127.0.0.1:9000/v1/embeddings"
        start = len(REQUESTS)
        for expected in ["MISS", "HIT"]:
            result = httpx.post(endpoint, json=body)
            self.assertEqual(result.headers["x-dli-cache"], expected)
            self.assertEqual(result.json()["data"][0]["embedding"], [0.2, 0.4])
        self.assertEqual(len(REQUESTS) - start, 1)
        for _ in range(2):
            result = httpx.post(endpoint, json={**body, "use_cache": False})
            self.assertEqual(result.json()["dli_cache"]["reason"], "use_cache_false")
            self.assertNotIn("use_cache", REQUESTS[-1][2])
        self.assertEqual(len(REQUESTS) - start, 3)
        self.assertEqual(httpx.post(endpoint, json={**body, "input": "changed"}).headers["x-dli-cache"], "MISS")

    def test_explicit_cache_parameter_bypasses_reads_and_writes(self):
        for stream in [False, True]:
            body = {"stream": stream, "temperature": 0.7, "seed": 5222}
            self.assertEqual(self.chat(**body).headers["x-dli-cache"], "MISS")
            start = len(REQUESTS)
            for _ in range(2):
                result = self.chat(**body, use_cache=False)
                self.assertEqual(result.headers["x-dli-cache"], "BYPASS")
                self.assertIn("use_cache_false", result.text)
                self.assertNotIn("use_cache", REQUESTS[-1][2])
            self.assertEqual(len(REQUESTS) - start, 2)
            self.assertEqual(self.chat(**body, use_cache=True).headers["x-dli-cache"], "HIT")
            uncached = {**body, "seed": 5333}
            self.chat(**uncached, use_cache=False)
            self.assertEqual(self.chat(**uncached).headers["x-dli-cache"], "MISS")
        for invalid in [None, 0, 1, "false", {}, []]:
            start = len(REQUESTS)
            self.assertEqual(self.chat(use_cache=invalid).status_code, 400)
            self.assertEqual(len(REQUESTS), start)

    def test_cache_bypass_header(self):
        for control in ["no-store", "max-age=0, no-cache"]:
            start = len(REQUESTS)
            body = {"model": "course/agent", "messages": [{"role": "user", "content": "bypass"}], "temperature": 0}
            for _ in range(2):
                result = httpx.post("http://127.0.0.1:9000/v1/chat/completions", json=body, headers={"Cache-Control": control})
                self.assertEqual(result.headers["x-dli-cache"], "BYPASS")
            self.assertEqual(len(REQUESTS) - start, 2)

    def test_stream_cache(self):
        start = len(REQUESTS)
        first = self.chat(stream=True, temperature=0, seed=2111)
        second = self.chat(stream=True, temperature=0, seed=2111)
        self.assertEqual(first.headers["x-dli-cache"], "MISS")
        self.assertEqual(second.headers["x-dli-cache"], "HIT")
        def payloads(text):
            return [{k:v for k,v in json.loads(line[6:]).items() if k not in ("dli_cache", "dli_routing")} for line in text.splitlines() if line.startswith("data: {")]
        self.assertIn('"dli_routing"', first.text)
        self.assertNotIn('"dli_routing"', second.text)
        self.assertEqual(payloads(first.text), payloads(second.text))
        self.assertIn('"status": "HIT"', second.text)
        self.assertEqual(len(REQUESTS) - start, 1)

    def test_journal_records_actual_input_without_headers(self):
        self.chat(seed=777)
        with sqlite3.connect(self.store_path) as connection:
            rows = connection.execute("SELECT body,status FROM requests WHERE source='upstream' AND method='POST' ORDER BY id DESC LIMIT 1").fetchall()
            columns = [row[1] for row in connection.execute("PRAGMA table_info(requests)")]
        self.assertEqual(json.loads(rows[0][0])["seed"], 777)
        self.assertEqual(rows[0][1], 200)
        self.assertNotIn("headers", columns)
        content = Path(self.store_path).read_bytes()
        for key in KEYS:
            self.assertNotIn(key.encode(), content)
        self.assertEqual(Path(self.store_path).stat().st_mode & 0o777, 0o600)

    def test_cache_single_flight(self):
        start = len(REQUESTS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.chat(temperature=0, seed=3111), range(4)))
        self.assertTrue(all(result.status_code == 200 for result in results))
        self.assertEqual(len(REQUESTS) - start, 1)

    def test_disconnect_closes_provider_stream(self):
        DISCONNECTED.clear()
        body = {"model": "course/agent", "stream": True, "messages": [{"role": "user", "content": "slow"}]}
        with httpx.stream("POST", "http://127.0.0.1:9000/v1/chat/completions", json=body, timeout=10) as response:
            self.assertEqual(response.status_code, 200)
            next(response.iter_lines())
        self.assertTrue(DISCONNECTED.wait(5), "provider stream remained open after disconnect")

    def test_malformed_requests(self):
        for body in [[], {"model": []}, {"model": None}]:
            self.assertEqual(httpx.post("http://127.0.0.1:9000/v1/chat/completions", json=body).status_code, 400)

    def test_debug_cannot_enable_dependency_output(self):
        script = '''from llm_output import protect_output, diagnostic
protect_output()
import logging, os
print(os.environ["TEST_SECRET"], flush=True)
logging.error(os.environ["TEST_SECRET"])
os.write(2, os.environ["TEST_SECRET"].encode())
diagnostic("request", 200)
diagnostic(os.environ["TEST_SECRET"], 200)
raise RuntimeError(os.environ["TEST_SECRET"])
'''
        for debug in ["0", "1"]:
            result = subprocess.run([sys.executable, "-c", script], cwd=SOURCE,
                env=dict(os.environ, TEST_SECRET=KEYS[0], DEBUG="1", LLM_DEBUG=debug), capture_output=True)
            self.assertEqual(result.stdout, b"")
            self.assertEqual(result.stderr, b"llm request status=200\n" if debug == "1" else b"")


if __name__ == "__main__":
    unittest.main(verbosity=2)
