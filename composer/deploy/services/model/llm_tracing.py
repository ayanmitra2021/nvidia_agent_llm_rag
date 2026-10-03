"""Trace actual proxy delivery, including cached playback, without request content."""
import asyncio
import contextvars
import json
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

PROPAGATOR = TraceContextTextMapPropagator()
ROUTES = {"/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/ranking"}
EXECUTION = contextvars.ContextVar("course_inference_execution", default=None)


def setup_tracing(service):
    provider = TracerProvider(resource=Resource({"service.name": service}))
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=endpoint, insecure=endpoint.startswith("http://"), timeout=2), schedule_delay_millis=500))
    return provider


def trace_headers():
    headers = {}
    PROPAGATOR.inject(headers)
    return headers


def profile_attribute(revision):
    trace.get_current_span().set_attribute("dli.profile.revision", revision)


def provider_attempt(route):
    execution = EXECUTION.get()
    if execution is not None:
        execution["executed"] = True
    span = trace.get_current_span()
    span.set_attribute("dli.inference.executed", True)
    span.add_event("provider.http_attempt", {"dli.route": route})


def cache_attributes(status, details=None):
    span = trace.get_current_span()
    span.set_attribute("dli.cache.status", status)
    span.set_attribute("dli.cache.hit", status in {"HIT", "BUNDLE_HIT", "SHARED"})
    if details and isinstance(details.get("pacing"), dict):
        delay = details["pacing"].get("initial_delay_seconds")
        if type(delay) in (float, int):
            span.set_attribute("dli.cache.playback_initial_delay_seconds", delay)


class RequestTracing:
    """Keep the span open through the last byte or cancellation of an SSE stream."""
    def __init__(self, app, provider):
        self.app = app
        self.tracer = provider.get_tracer("course.proxy")

    async def __call__(self, scope, receive, send):
        token = EXECUTION.set({"executed": False})
        try:
            return await self.deliver(scope, receive, send)
        finally:
            EXECUTION.reset(token)

    async def deliver(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in ROUTES:
            return await self.app(scope, receive, send)
        headers = {k.decode("latin1"): v.decode("latin1") for k, v in scope.get("headers", [])
                   if k in {b"traceparent", b"tracestate"}}
        context = PROPAGATOR.extract(headers)
        with self.tracer.start_as_current_span("POST " + scope["path"], context=context,
                kind=SpanKind.SERVER, record_exception=False, set_status_on_exception=False) as span:
            span.set_attribute("http.request.method", scope.get("method", "POST"))
            span.set_attribute("http.route", scope["path"])
            span.set_attribute("dli.inference.executed", False)
            streaming = done = ended = interrupted = False
            status = 500
            pending = b""

            async def observed(message):
                nonlocal streaming, done, ended, interrupted, status, pending
                if message["type"] == "http.response.start":
                    status = message["status"]
                    span.set_attribute("http.response.status_code", status)
                    response_headers = dict(message.get("headers", []))
                    if response_headers.get(b"x-dli-inference-executed") == b"true":
                        EXECUTION.get()["executed"] = True
                        span.set_attribute("dli.inference.executed", True)
                    response_headers[b"x-dli-inference-executed"] = (
                        b"true" if EXECUTION.get()["executed"] else b"false")
                    message = {**message, "headers": list(response_headers.items())}
                    streaming = b"text/event-stream" in response_headers.get(b"content-type", b"")
                    cache = response_headers.get(b"x-dli-cache", b"").decode("ascii", errors="ignore")
                    if cache in {"HIT", "BUNDLE_HIT", "SHARED", "MISS", "BYPASS"}:
                        cache_attributes(cache)
                    route = response_headers.get(b"x-dli-route", b"").decode("ascii", errors="ignore")
                    if route in {"primary", "fallback", "unavailable", "local"}:
                        span.set_attribute("dli.route", route)
                elif message["type"] == "http.response.body":
                    if streaming:
                        pending += message.get("body", b"")
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if line.strip() == b"data: [DONE]":
                                done = True
                            elif line.startswith(b"data:") and len(line) <= 1048576:
                                try:
                                    value = json.loads(line[5:])
                                    if isinstance(value, dict) and "error" in value:
                                        interrupted = True
                                except (ValueError, UnicodeError):
                                    interrupted = True
                        if len(pending) > 1048576:
                            pending = b""
                    await send(message)
                    if not message.get("more_body", False):
                        ended = True
                    return
                await send(message)

            try:
                await self.app(scope, receive, observed)
            except BaseException as error:
                interrupted = True
                span.set_attribute("dli.response.cancelled", isinstance(error, (asyncio.CancelledError, OSError)))
                raise
            finally:
                complete = ended and status < 400 and not interrupted and (done or not streaming)
                span.set_attribute("dli.response.complete", complete)
                if not complete:
                    span.set_status(Status(StatusCode.ERROR, "Request delivery did not complete"))
