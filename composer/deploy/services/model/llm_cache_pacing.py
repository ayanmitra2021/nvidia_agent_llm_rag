"""Cancellable playback timing; never changes a recorded response or provider input."""
import asyncio
import json
import os
import time

PROFILES = {"reading": 12, "chat": 24, "coding": 50, "automation": 120, "off": 0}
MAX_STREAM_DELAY_SECONDS = 1
MAX_RESPONSE_DELAY_SECONDS = 0.1
INITIAL_DELAY_SECONDS = 1
if os.getenv("LLM_CACHE_SPEEDS"):
    rates = json.loads(os.environ["LLM_CACHE_SPEEDS"])
    if not isinstance(rates, dict) or set(rates) - (set(PROFILES) - {"off"}):
        raise ValueError("invalid cache speed profiles")
    if any(type(rate) not in (int, float) or not 1 <= rate <= 10000 for rate in rates.values()):
        raise ValueError("invalid cache speed")
    PROFILES.update(rates)
MAX_STREAM_DELAY_SECONDS = float(os.getenv("LLM_CACHE_MAX_STREAM_DELAY", str(MAX_STREAM_DELAY_SECONDS)))
MAX_RESPONSE_DELAY_SECONDS = float(os.getenv("LLM_CACHE_MAX_RESPONSE_DELAY", str(MAX_RESPONSE_DELAY_SECONDS)))
INITIAL_DELAY_SECONDS = float(os.getenv("LLM_CACHE_INITIAL_DELAY_SECONDS", str(INITIAL_DELAY_SECONDS)))
if not 0 <= MAX_STREAM_DELAY_SECONDS <= 30 or not 0 <= MAX_RESPONSE_DELAY_SECONDS <= 5 or not 0 <= INITIAL_DELAY_SECONDS <= 5:
    raise ValueError("invalid cache delay limits")


def payload_size(chunk):
    """Visible text and tool output only; role, usage and finish events cost no time."""
    size = 0
    for choice in chunk.get("choices", []):
        message = choice.get("delta", choice.get("message", {}))
        content = message.get("content")
        if isinstance(content, str):
            size += len(content)
        calls = message.get("tool_calls") or []
        functions = [call.get("function", {}) for call in calls]
        if message.get("function_call"):
            functions.append(message["function_call"])
        for function in functions:
            size += sum(len(function.get(key) or "") for key in ("name", "arguments"))
    return size


class Playback:
    def __init__(self, value, profile, *, clock=time.monotonic, sleep=asyncio.sleep, settings=None):
        self.clock, self.sleep = clock, sleep
        chunks = value if isinstance(value, list) else [value]
        self.size = sum(payload_size(chunk) for chunk in chunks)
        first_size = next((payload_size(chunk) for chunk in chunks if payload_size(chunk)), 0)
        self.remaining_size = self.size - first_size
        counts = []
        for chunk in chunks:
            usage = chunk.get("usage") or {}
            count = usage.get("completion_tokens")
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            if type(count) is int and count >= 0 and type(reasoning) is int and 0 <= reasoning <= count:
                counts.append(count - reasoning)
        # Stored streams may omit usage. This fallback is an estimate, not token accounting.
        tokens = max(counts) if counts else self.size / 4
        rates = {**settings["speeds"], "off": 0} if settings else PROFILES
        rate = rates[profile]
        requested = tokens * self.remaining_size / self.size / rate if rate and self.size else 0
        # Complete replies have nothing to read while waiting. Long streams must
        # also finish promptly when a notebook chains several model turns.
        cap = ((settings["max_stream_delay"] if isinstance(value, list) else settings["max_response_delay"])
               if settings else MAX_STREAM_DELAY_SECONDS if isinstance(value, list) else MAX_RESPONSE_DELAY_SECONDS)
        self.duration = min(requested, cap)
        self.initial_delay = (settings["initial_delay_seconds"] if settings else INITIAL_DELAY_SECONDS) if rate else 0
        self.started = None
        self.emitted = 0
        self.visible_started = False
        self.metadata = {"profile": profile, "tokens_per_second": rate,
            "token_basis": "recorded_visible_usage" if counts else "estimated_characters_divided_by_four",
            "delay_seconds": self.duration, "initial_delay_seconds": self.initial_delay,
            "total_delay_seconds": self.initial_delay + self.duration,
            "capped": requested > self.duration}

    async def wait(self, chunk):
        if self.started is None:
            if self.initial_delay:
                await self.sleep(self.initial_delay)
            self.started = self.clock()
        size = payload_size(chunk)
        if size and not self.visible_started:
            self.visible_started = True
            self.started = self.clock()
            return
        self.emitted += size
        if not size or not self.duration:
            return
        delay = self.started + self.duration * self.emitted / self.remaining_size - self.clock()
        if delay > 0:
            await self.sleep(delay)
