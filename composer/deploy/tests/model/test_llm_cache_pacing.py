import asyncio
import gzip
import json
import os
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services/model"))
from llm_cache_pacing import Playback, PROFILES


class PlaybackTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_delay_precedes_metadata_and_off_removes_all_waits(self):
        chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
                  {"choices": [{"delta": {"content": "hello"}}]}]
        for profile in ("chat", "off"):
            now, delays = [0.0], []
            async def sleep(delay):
                delays.append(delay)
                now[0] += delay
            playback = Playback(chunks, profile, clock=lambda: now[0], sleep=sleep)
            await playback.wait(chunks[0])
            self.assertEqual(now[0], 1 if profile == "chat" else 0)
            await playback.wait(chunks[1])
            self.assertEqual(delays.count(1), 1 if profile == "chat" else 0)
            self.assertAlmostEqual(now[0], playback.metadata["total_delay_seconds"])

    async def test_service_defaults_both_layers_and_sse_fidelity(self):
        import httpx
        from test_llm_course_cache import fixture_bundle
        from llm_course_cache import CourseCache, package_records
        from llm_store import Store
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            records = json.loads(gzip.decompress(path.read_bytes()))["entries"]
            stream_record = next(record for record in records if record["request"].get("stream"))
            stream_record["response"] = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}]},
                {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call-test",
                    "function": {"name": "add", "arguments": "{}"}}]}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                 "usage": {"completion_tokens": 4, "total_tokens": 8}},
            ]
            package_records([record for record in records if not record["request"].get("stream")] + [stream_record], path, {"fixture": True})
            environment = {"DLI_LLM_PROVIDERS": json.dumps([{"base_url": "http://127.0.0.1:19101/v1", "api_key_env": "PACING_TEST_KEY"}]),
                "PACING_TEST_KEY": "synthetic-test-only", "LLM_STORE_PATH": directory + "/init.sqlite"}
            with patch.dict(os.environ, environment):
                import litellm_service as service
            class ImmediatePlayback(Playback):
                def __init__(self, value, profile, **kwargs):
                    now = [0.0]
                    async def sleep(delay):
                        now[0] += delay
                    super().__init__(value, profile, clock=lambda: now[0], sleep=sleep, **kwargs)
            with patch.multiple(service, DEFAULT_PACING=None, COURSE_CACHE_ENABLED=True,
                    COURSE_CACHE=CourseCache(path), STORE=Store(directory + "/requests.sqlite", lambda x: x), Playback=ImmediatePlayback), patch.multiple(
                    service.runtime(), DEFAULT_PACING=None, COURSE_CACHE_ENABLED=True, STORE=service.STORE):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url="http://test",
                        headers={"x-dli-base-url": "http://127.0.0.1:19101/v1"}) as client:
                    for streaming in (False, True):
                        request = {**body, **({"stream": True} if streaming else {})}
                        expected = stream_record["response"] if streaming else service.COURSE_CACHE.get(body)[0]
                        for bundle in (True, False):
                            service.STORE.put(service.STORE.key("http://127.0.0.1:19101/v1", request), expected)
                            response = await client.post("/v1/chat/completions", json={**request, "use_course_cache": bundle})
                            self.assertEqual(response.status_code, 200)
                            if streaming:
                                actual = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
                                metadata = [chunk.pop("dli_cache") for chunk in actual]
                                self.assertTrue(response.text.rstrip().endswith("data: [DONE]"))
                            else:
                                actual = response.json()
                                metadata = [actual.pop("dli_cache")]
                            self.assertEqual(actual, expected)
                            self.assertTrue(all(m["pacing"]["profile"] == ("chat" if streaming else "automation") for m in metadata))
                            self.assertTrue(all(m["status"] == ("BUNDLE_HIT" if bundle else "HIT") for m in metadata))

    async def test_profiles_preserve_payload_and_use_cumulative_timing(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "abcd"}}]},
            {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "1234"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
             "usage": {"completion_tokens": 4}},
        ]
        original = deepcopy(chunks)
        for profile, rate in PROFILES.items():
            now, delays = [0.0], []
            async def sleep(delay):
                delays.append(delay)
                now[0] += delay
            playback = Playback(chunks, profile, clock=lambda: now[0], sleep=sleep)
            for chunk in chunks:
                await playback.wait(chunk)
                # Consumer work counts toward the budget; no cumulative sleep drift.
                now[0] += .001
            self.assertEqual(len(delays), 2 if rate else 0)
            self.assertAlmostEqual(sum(delays), 1 + 2 / rate - .001 if rate else 0)
            self.assertEqual(playback.metadata["initial_delay_seconds"], 1 if rate else 0)
            self.assertEqual(playback.metadata["token_basis"], "recorded_visible_usage")
        self.assertEqual(chunks, original)

    async def test_json_fallback_reasoning_and_cap(self):
        payload = {"choices": [{"message": {"content": "x" * 4000}}]}
        sleeps = []
        async def sleep(delay):
            sleeps.append(delay)
        playback = Playback(payload, "reading", clock=lambda: 0, sleep=sleep)
        await playback.wait(payload)
        self.assertEqual(sleeps, [1])
        self.assertFalse(playback.metadata["capped"])
        self.assertEqual(playback.metadata["token_basis"], "estimated_characters_divided_by_four")
        payload["usage"] = {"completion_tokens": 100, "completion_tokens_details": {"reasoning_tokens": 20}}
        self.assertEqual(Playback(payload, "automation").duration, 0)
        self.assertEqual(Playback({"choices": []}, "reading").duration, 0)

    async def test_long_multiturn_streams_finish_without_truncation(self):
        chunks = [{"choices": [{"delta": {"content": "x" * 400}}]} for _ in range(10)]
        original = deepcopy(chunks)
        now = [0.0]
        async def sleep(delay):
            now[0] += delay
        for _ in range(7):
            playback = Playback(chunks, "chat", clock=lambda: now[0], sleep=sleep)
            for chunk in chunks:
                await playback.wait(chunk)
            self.assertTrue(playback.metadata["capped"])
            self.assertEqual(playback.duration, 1)
        self.assertAlmostEqual(now[0], 14)
        self.assertEqual(chunks, original)

    async def test_temperature_samples_work_without_student_cache_controls(self):
        import httpx
        from test_llm_course_cache import fixture_bundle
        from llm_course_cache import CourseCache, package_records
        from llm_store import Store
        with tempfile.TemporaryDirectory() as directory:
            path, body = fixture_bundle(directory)
            entries = json.loads(gzip.decompress(path.read_bytes()))["entries"]
            requests = [{**body, "temperature": t, **seed}
                        for t in (0, .7) for seed in ({}, {"seed": 0}, {"seed": 12})]
            package_records([{**entry, "request": request} for entry in entries
                if not entry["request"].get("stream") for request in requests], path, {"fixture": True})
            with patch.dict(os.environ, {"DLI_LLM_PROVIDERS": json.dumps([
                    {"base_url": "http://127.0.0.1:19101/v1", "api_key_env": "PACING_TEST_KEY"}]),
                    "PACING_TEST_KEY": "synthetic-test-only", "LLM_STORE_PATH": directory + "/init.sqlite"}):
                import litellm_service as service
            with patch.multiple(service, DEFAULT_PACING="off", COURSE_CACHE_ENABLED=True,
                    COURSE_CACHE=CourseCache(path), STORE=Store(directory + "/test.sqlite", lambda x: x)), patch.multiple(
                    service.runtime(), DEFAULT_PACING="off", COURSE_CACHE_ENABLED=True, STORE=service.STORE):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url="http://test",
                        headers={"x-dli-base-url": "http://127.0.0.1:19101/v1"}) as client:
                    for request in requests:
                        replies = [(await client.post("/v1/chat/completions", json=request)).json() for _ in range(3)]
                        self.assertTrue(all(r["dli_cache"]["status"] == "BUNDLE_HIT" for r in replies))
                        expected = ["recorded-a", "recorded-b", "recorded-a"] if request["temperature"] and "seed" not in request else ["recorded-a"] * 3
                        self.assertEqual([r["id"] for r in replies], expected)

    async def test_wait_is_nonblocking_and_cancellation_propagates(self):
        entered = asyncio.Event()
        async def sleep(delay):
            entered.set()
            await asyncio.Event().wait()
        payload = {"choices": [{"message": {"content": "waiting"}}]}
        task = asyncio.create_task(Playback(payload, "reading", sleep=sleep).wait(payload))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    unittest.main()
