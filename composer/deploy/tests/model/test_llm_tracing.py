import asyncio
import json
from pathlib import Path
import sys
import unittest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services/model'))
from llm_tracing import RequestTracing, cache_attributes, trace_headers, provider_attempt


class DeliveryTraceTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, outcome, *, stream=False, cancelled=False, failed=False):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        observations = []
        async def application(scope, receive, send):
            if outcome in {'MISS', 'BYPASS'}:
                provider_attempt('primary')
            cache_attributes(outcome, {"pacing": {"initial_delay_seconds": .02}})
            observations.append(trace_headers())
            await asyncio.sleep(.02)
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"content-type", b"text/event-stream" if stream else b"application/json"),
                (b"x-dli-cache", outcome.encode())]})
            if cancelled:
                raise asyncio.CancelledError()
            content = (b'data: {"error":{"message":"PRIVATE ERROR"}}\n\n' if failed else
                b'data: {"choices":[{"delta":{"content":"PRIVATE RESPONSE"}}]}\n\ndata: [DONE]\n\n') if stream else b'{"secret":"PRIVATE RESPONSE"}'
            await send({"type": "http.response.body", "body": content, "more_body": False})
        async def receive():
            return {"type": "http.request", "body": b'PRIVATE PROMPT'}
        async def send(message):
            pass
        app = RequestTracing(application, provider)
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": [
            (b"authorization", b"PRIVATE KEY"),
            (b"traceparent", b"00-11111111111111111111111111111111-2222222222222222-01")]}
        try:
            await app(scope, receive, send)
        except asyncio.CancelledError:
            if not cancelled:
                raise
        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.context.trace_id, int('1' * 32, 16))
        self.assertEqual(span.parent.span_id, int('2' * 16, 16))
        self.assertIn('traceparent', observations[0])
        self.assertGreaterEqual((span.end_time - span.start_time) / 1e9, .02)
        self.assertEqual(span.attributes['dli.cache.status'], outcome)
        self.assertEqual(span.attributes['dli.inference.executed'], outcome in {'MISS', 'BYPASS'})
        self.assertEqual(span.attributes['dli.response.complete'], not (cancelled or failed))
        self.assertNotIn('PRIVATE', json.dumps(dict(span.attributes)) + str(span.events))
        provider.shutdown()

    async def test_cached_and_live_delivery_keep_truthful_lifetime_and_parent(self):
        for outcome in ('BUNDLE_HIT', 'HIT', 'SHARED', 'MISS', 'BYPASS'):
            for stream in (False, True):
                with self.subTest(outcome=outcome, stream=stream):
                    await self.exercise(outcome, stream=stream)

    async def test_cancelled_and_failed_streams_are_not_reported_complete(self):
        await self.exercise('BUNDLE_HIT', stream=True, cancelled=True)
        await self.exercise('MISS', stream=True, failed=True)
