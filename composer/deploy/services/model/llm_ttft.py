"""Bound first-token waiting before exposing any provider output."""
import asyncio
import copy
import inspect
import math
import time
from email.utils import parsedate_to_datetime


class CompletionState:
    def __init__(self, expected):
        if type(expected) is not int or expected < 1:
            raise ValueError("invalid completion choice count")
        self.expected = expected
        self.seen = set()
        self.finished = set()

    def observe(self, value):
        for choice in value.get("choices", []):
            index = choice.get("index")
            if type(index) is not int or not 0 <= index < self.expected:
                raise ValueError("unexpected completion choice")
            self.seen.add(index)
            if choice.get("finish_reason") is not None:
                self.finished.add(index)

    def complete(self):
        if len(self.seen) != self.expected or len(self.finished) != self.expected:
            raise ValueError("provider stream ended before all choices finished")


async def close_stream(stream):
    close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if close:
        result = close()
        if inspect.isawaitable(result):
            await result


def assemble(chunks, messages):
    """Preserve all choices and provider usage while assembling with LiteLLM."""
    import litellm
    from litellm.types.utils import ModelResponseStream
    values = [chunk.model_dump(exclude_none=True) for chunk in chunks]
    indices = sorted({choice["index"] for value in values for choice in value.get("choices", [])})
    response = None
    choices = []
    for index in indices:
        selected = []
        for value in values:
            item = copy.deepcopy(value)
            item["choices"] = [c for c in item.get("choices", []) if c["index"] == index]
            for choice in item["choices"]:
                choice["index"] = 0
            selected.append(ModelResponseStream(**item))
        merged = litellm.stream_chunk_builder(selected, messages=messages)
        if merged is None:
            raise ValueError("provider returned no completion")
        value = merged.model_dump(exclude_none=True)
        response = response or value
        for choice in value["choices"]:
            choice["index"] = index
            choices.append(choice)
    if response is None:
        raise ValueError("provider returned no choices")
    response["choices"] = choices
    response.pop("usage", None)
    for value in values:
        if value.get("usage"):
            response["usage"] = value["usage"]
    return response


def meaningful(chunk):
    for choice in chunk.get("choices", []):
        delta = choice.get("delta", {})
        extra = delta.get("provider_specific_fields") or {}
        if any(delta.get(field) for field in ("content", "reasoning_content", "reasoning", "tool_calls", "function_call")) or extra.get("reasoning"):
            return True
        if choice.get("finish_reason") is not None:
            return True
    return False


def upstream_status(error):
    status = getattr(error, "status_code", None)
    if type(status) is int and 400 <= status <= 599:
        return status
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
        code = body.get("code") if isinstance(body, dict) else None
        if type(code) is int and 400 <= code <= 599:
            return code
        if isinstance(code, str) and len(code) == 3 and code.isdigit() and 400 <= int(code) <= 599:
            return int(code)
    return 502


def retry_delay(error, stream, attempt):
    response = getattr(error, "response", None)
    if response is None:
        response = getattr(stream, "response", None)
    value = getattr(response, "headers", {}).get("retry-after")
    if value is not None:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            try:
                delay = parsedate_to_datetime(value).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                delay = None
        if delay is not None and math.isfinite(delay):
            return max(0, delay)
    return 2 ** attempt


async def first_output(open_stream, timeout, attempts=1):
    """One deadline includes headers, retries, heartbeat and role-only chunks."""
    stream = None
    try:
        async with asyncio.timeout(timeout):
            for attempt in range(attempts):
                try:
                    stream = await open_stream()
                    iterator = stream.__aiter__()
                    buffered = []
                    size = 0
                    async for chunk in iterator:
                        value = chunk.model_dump(exclude_none=True)
                        size += len(str(value))
                        if size > 1048576:
                            raise ValueError("provider preamble exceeds limit")
                        buffered.append(chunk)
                        if meaningful(value):
                            return stream, iterator, buffered
                    raise ValueError("provider ended without a completion")
                except Exception as error:
                    delay = retry_delay(error, stream, attempt)
                    if stream is not None:
                        await close_stream(stream)
                        stream = None
                    if upstream_status(error) not in {408, 429, *range(500, 600)} or attempt + 1 == attempts:
                        raise
                    # Header and SSE-error retries share the original deadline.
                    # Returning meaningful output above ends the retry boundary.
                    await asyncio.sleep(delay)
    except BaseException:
        if stream is not None:
            await close_stream(stream)
        raise
