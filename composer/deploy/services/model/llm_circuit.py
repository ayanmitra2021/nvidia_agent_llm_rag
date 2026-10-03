"""Process-local provider/model cooldowns; no credentials or request content in state."""
from dataclasses import dataclass
import math
import threading
import time


@dataclass
class State:
    failures: int = 0
    tier: int = 0
    until: float = 0
    probe: bool = False
    generation: int = 0
    active: int = 0
    attempts: int = 0
    successes: int = 0
    failed_requests: int = 0
    last_failure: str | None = None


@dataclass
class Ticket:
    state: State
    generation: int
    probe: bool
    done: bool = False


class Circuits:
    def __init__(self, threshold=3, base=30, maximum=300, clock=time.monotonic):
        if type(threshold) is not int or not 1 <= threshold <= 100:
            raise ValueError("invalid circuit failure threshold")
        if not all(math.isfinite(x) for x in (base, maximum)) or not 0.01 <= base <= maximum <= 3600:
            raise ValueError("invalid circuit cooldown range")
        self.threshold, self.base, self.maximum = threshold, base, maximum
        self.clock = clock
        self.states = {}
        self.lock = threading.Lock()

    def acquire(self, provider, model):
        with self.lock:
            key = (provider, model)
            if key not in self.states:
                if len(self.states) >= 512:
                    idle = next((k for k, s in self.states.items() if not s.active and not s.tier), None)
                    if idle is None:
                        raise RuntimeError("provider health capacity reached")
                    del self.states[idle]
                self.states[key] = State()
            state = self.states[key]
            if state.probe or state.until > self.clock():
                return None
            probe = state.tier > 0
            state.probe = probe
            state.active += 1
            state.attempts += 1
            return Ticket(state, state.generation, probe)

    def finish(self, ticket, outcome, reason=None):
        """Count a logical request once; old concurrent outcomes cannot undo a cooldown."""
        if outcome not in {"success", "failure", "cancel"}:
            raise ValueError("invalid circuit outcome")
        with self.lock:
            if ticket.done:
                return
            ticket.done = True
            state = ticket.state
            state.active -= 1
            if outcome == "success":
                state.successes += 1
            elif outcome == "failure":
                state.failed_requests += 1
            if ticket.generation != state.generation:
                return
            state.probe = False
            if outcome == "cancel":
                return
            if outcome == "success":
                state.failures = state.tier = 0
                state.until = 0
                return
            state.failures += 1
            state.last_failure = reason
            if ticket.probe or state.failures >= self.threshold:
                # Bound the exponent even during a prolonged outage.
                state.tier = min(state.tier + 1, 32)
                state.until = self.clock() + min(self.maximum, self.base * 2 ** (state.tier - 1))
                state.generation += 1

    def snapshot(self, provider, model):
        with self.lock:
            state = self.states.get((provider, model), State())
            remaining = max(0, state.until - self.clock())
            mode = "open" if remaining > 0 else "half_open" if state.tier else "closed"
            return {"state": mode, "consecutive_failures": state.failures, "tier": state.tier,
                "retry_after_seconds": round(remaining, 3), "probe_in_flight": state.probe,
                "last_failure": state.last_failure, "attempts": state.attempts,
                "successes": state.successes, "failed_requests": state.failed_requests,
                "available": not state.probe and remaining == 0}

    def policy(self):
        return {"failure_threshold": self.threshold, "base_cooldown_seconds": self.base,
            "max_cooldown_seconds": self.maximum, "scope": "provider_and_model",
            "persistence": "proxy_process", "recovery_probes": 1}
