"""Shared schema for settings that the running LLM proxy owns."""
import copy

DEFAULT_PROFILE = {
    "cache": {"enabled": True, "vendored": True, "respect_http_headers": False,
              "ttl_seconds": 86400, "max_entries": 2000, "max_bytes": 67108864},
    "playback": {"profile": "chat", "speeds": {"reading": 12, "chat": 24, "coding": 50, "automation": 120},
                 "initial_delay_seconds": 1, "max_stream_delay": 1, "max_response_delay": 0.1},
    "requests": {"timeout_seconds": 300, "max_ttft_seconds": 10, "retry_attempts": 3,
                 "max_concurrency": 20, "failure_threshold": 3, "base_cooldown_seconds": 30,
                 "max_cooldown_seconds": 300},
}


def default_profile():
    return copy.deepcopy(DEFAULT_PROFILE)


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_PROFILE):
        raise ValueError("Supply cache, playback, and request settings")
    for section, defaults in DEFAULT_PROFILE.items():
        if not isinstance(settings[section], dict) or set(settings[section]) != set(defaults):
            raise ValueError("Unknown or missing profile setting")
    cache, playback, requests = (settings[k] for k in DEFAULT_PROFILE)
    for name in ("enabled", "vendored", "respect_http_headers"):
        if type(cache[name]) is not bool:
            raise ValueError("Cache switches must be booleans")
    def number(value, low, high, integer=False):
        if type(value) not in ((int,) if integer else (int, float)) or not low <= value <= high:
            raise ValueError("Profile number is outside its supported range")
    number(cache["ttl_seconds"], 0, 31536000, True)
    number(cache["max_entries"], 1, 1000000, True)
    number(cache["max_bytes"], 1, 1073741824, True)
    if not isinstance(playback["speeds"], dict) or set(playback["speeds"]) != set(DEFAULT_PROFILE["playback"]["speeds"]):
        raise ValueError("Unknown playback speed")
    if playback["profile"] not in {*playback["speeds"], "off"}:
        raise ValueError("Unknown playback profile")
    for speed in playback["speeds"].values():
        number(speed, 1, 10000)
    number(playback["initial_delay_seconds"], 0, 5)
    number(playback["max_stream_delay"], 0, 30)
    number(playback["max_response_delay"], 0, 5)
    number(requests["timeout_seconds"], 1, 1800)
    number(requests["max_ttft_seconds"], 0.01, 1800)
    number(requests["retry_attempts"], 1, 5, True)
    number(requests["max_concurrency"], 1, 64, True)
    number(requests["failure_threshold"], 1, 100, True)
    number(requests["base_cooldown_seconds"], 0.01, 3600)
    number(requests["max_cooldown_seconds"], requests["base_cooldown_seconds"], 3600)
    return copy.deepcopy(settings)
