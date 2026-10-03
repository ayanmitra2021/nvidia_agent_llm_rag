# observability_utils.py

import os
import logging
import asyncio
from functools import wraps

def get_observability(service_name: str = "default-service", DEBUG_MODE=None):
    if DEBUG_MODE is None:
        DEBUG_MODE = os.getenv("DEBUG", "0") == "1"

    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger(service_name)

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.trace import Status, StatusCode, SpanKind
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        # Avoid overwriting globally if already set
        if not isinstance(trace.get_tracer_provider(), TracerProvider):
            provider = TracerProvider(resource=Resource.create({
                "service.name": service_name
            }))
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if endpoint:
                processor = BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=endpoint, insecure=True)
                )
                provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

        tracer = trace.get_tracer(service_name)
        propagator = TraceContextTextMapPropagator()

        def traced(name: str):
            """
            Decorator that creates an OTEL span around a function, while
            preserving the original function's signature for FastAPI.
            """
            def decorator(func):
                # Async endpoint / function
                if asyncio.iscoroutinefunction(func):
                    @wraps(func)
                    async def async_wrapper(*args, **kwargs):
                        with tracer.start_as_current_span(
                            name, kind=SpanKind.INTERNAL
                        ) as span:
                            span.set_attribute("function", func.__name__)
                            try:
                                return await func(*args, **kwargs)
                            except Exception as e:
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                                raise
                    return async_wrapper

                # Sync function (in case you decorate non-async code)
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with tracer.start_as_current_span(
                        name, kind=SpanKind.INTERNAL
                    ) as span:
                        span.set_attribute("function", func.__name__)
                        try:
                            return func(*args, **kwargs)
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise
                return sync_wrapper

            return decorator
        
        return logger, tracer, propagator, traced

    except Exception as e:
        logger.warning(f"OpenTelemetry setup failed: {e}")

        # No-op traced decorator for fallback
        def traced(name: str):
            def decorator(f):
                return f
            return decorator

        class StreamingWrapper:
            def __init__(self, inner_gen, span=None):
                self.inner = inner_gen
            async def __aiter__(self):
                async for chunk in self.inner:
                    yield chunk

        return logger, None, None, traced
