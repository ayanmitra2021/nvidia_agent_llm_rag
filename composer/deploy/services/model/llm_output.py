"""A closed diagnostic channel for credential-bearing services."""
import os

_EVENTS = frozenset({"ready", "request", "failure", "configuration_error"})
_fd = None


def protect_output():
    global _fd
    if _fd is not None:
        return
    _fd = os.dup(2)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)


def diagnostic(event, status=0):
    if os.getenv("LLM_DEBUG") == "1" and event in _EVENTS and type(status) is int:
        if _fd is not None and 0 <= status <= 599:
            os.write(_fd, f"llm {event} status={status}\n".encode())
