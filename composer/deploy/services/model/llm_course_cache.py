"""Immutable course recordings with exact requests and explicit provider provenance."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time
from llm_request import completion_limit, numeric_json

FORMAT = "dli-course-responses-v1"
NAMESPACE = "c-fx-25-v1:course-responses-v1"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
MODEL_ALIASES = {MODEL: MODEL, "nvidia/Nemotron-3_5-Lightning": MODEL}
CONTROLS = {"use_cache", "use_course_cache", "cache_sample", "cache_sample_index", "cache_pacing"}
ORIGIN_SCHEMA = "dli-course-origin-v2"
ORIGIN_LABELS = {"nvidia", "nebius"}


def capture_provider_label(url):
    """Resolve recording identity only from explicit server-owned configuration."""
    normalized = url.rstrip("/")
    if normalized == NVIDIA_URL:
        return "nvidia"
    if normalized and os.getenv("NEBIUS_BASE_URL", "").rstrip("/") == normalized:
        return "nebius"
    mapping = json.loads(os.getenv("LLM_CAPTURE_PROVIDER_LABELS", "{}"))
    if (not isinstance(mapping, dict) or any(not isinstance(key, str) or not key
            or not isinstance(label, str) or label not in ORIGIN_LABELS for key, label in mapping.items())):
        raise ValueError("Invalid private capture provider labels")
    if len({key.rstrip("/") for key in mapping}) != len(mapping):
        raise ValueError("Ambiguous private capture provider labels")
    return next((label for key, label in mapping.items() if key.rstrip("/") == normalized), None)


def response_model(response):
    chunks = response if isinstance(response, list) else [response]
    models = {chunk.get("model") for chunk in chunks if isinstance(chunk, dict)
              and isinstance(chunk.get("model"), str) and chunk["model"]}
    if len(models) != 1:
        raise ValueError("Recording needs one provider-reported served model")
    return next(iter(models))


def validate_origin(entry):
    if "origin_schema" not in entry:
        if entry.get("origin_provider") != NVIDIA_URL or entry.get("origin_model") not in MODEL_ALIASES:
            raise ValueError("Invalid legacy NVIDIA recording provenance")
        return
    if entry.get("origin_schema") != ORIGIN_SCHEMA or entry.get("origin_provider") not in ORIGIN_LABELS:
        raise ValueError("Unrecognized recording origin schema or provider")
    model = entry.get("origin_model")
    if not isinstance(model, str) or not model or len(model) > 512 or "://" in model:
        raise ValueError("Invalid provider-reported recording model")
    routed = entry.get("provider_request")
    if (not isinstance(routed, dict) or not isinstance(routed.get("model"), str)
            or not routed["model"] or not isinstance(routed.get("messages"), list)
            or set(routed) & {"api_key", "api_base", "base_url", "headers", "extra_headers", "authorization"}):
        raise ValueError("Invalid credential-free provider request evidence")
    if response_model(entry.get("response")) != model:
        raise ValueError("Recording model differs from actual response")


def record_identity(entry):
    values = [entry["key"], entry["response"]]
    if "origin_schema" in entry:
        values.append({field: entry[field] for field in
                       ("origin_schema", "origin_provider", "origin_model", "provider_request")})
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def validate_response(response, streaming):
    """Check completion structure; content quality requires scenario review."""
    if streaming != isinstance(response, list):
        raise ValueError("course response transport mismatch")
    chunks = response if streaming else [response]
    seen, finished = set(), set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or "error" in chunk or not isinstance(chunk.get("choices"), list):
            raise ValueError("invalid course completion envelope")
        for choice in chunk["choices"]:
            index = choice.get("index") if isinstance(choice, dict) else None
            if type(index) is not int or index < 0:
                raise ValueError("invalid course completion choice")
            if not isinstance(choice.get("delta" if streaming else "message"), dict):
                raise ValueError("missing course completion payload")
            seen.add(index)
            if choice.get("finish_reason") is not None:
                finished.add(index)
    if not seen or seen != finished:
        raise ValueError("incomplete course completion")


def signature(body):
    request = {k: v for k, v in body.items() if k not in CONTROLS}
    request["model"] = MODEL_ALIASES.get(request.get("model"), request.get("model"))
    encoded = json.dumps([FORMAT, NAMESPACE, request], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def numeric_signature(body):
    return signature(numeric_json(body))


class CourseCache:
    def __init__(self, bundle=None):
        self.path = Path(bundle or os.getenv("LLM_COURSE_CACHE_PATH", str(Path(__file__).parent / "course_cache/responses.json.gz")))
        self.entries = None
        self.digest = None
        self.cursors = {}
        self.lock = threading.Lock()
        self.unavailable = False
        self.preferred = {}
        self.numeric_entries = {}
        self.numeric_preferred = {}
        self.numeric_preference_resolutions = {}

    def _load(self):
        if self.entries is not None:
            return
        if not self.path.exists():
            self.entries = {}
            self.unavailable = True
            return
        manifest = json.loads(Path(str(self.path) + ".manifest.json").read_text())
        if self.path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("course bundle exceeds compressed size limit")
        artifact = self.path.read_bytes()
        self.digest = hashlib.sha256(artifact).hexdigest()
        if self.digest != manifest.get("sha256"):
            raise ValueError("course bundle checksum mismatch")
        if manifest.get("namespace") != NAMESPACE or not isinstance(manifest.get("provenance"), dict):
            raise ValueError("invalid course bundle namespace or provenance")
        with gzip.open(self.path, "rb") as stream:
            raw = stream.read(128 * 1024 * 1024 + 1)
        if len(raw) > 128 * 1024 * 1024:
            raise ValueError("course bundle exceeds expanded size limit")
        data = json.loads(raw)
        if data.get("format") != FORMAT or data.get("namespace") != NAMESPACE or not isinstance(data.get("entries"), list):
            raise ValueError("invalid course bundle format")
        if data.get("provenance") != manifest["provenance"]:
            raise ValueError("course provenance mismatch")
        entries = {}
        for entry in data["entries"]:
            if not isinstance(entry.get("request"), dict):
                raise ValueError("invalid course response provenance")
            if not isinstance(entry.get("response"), (dict, list)):
                raise ValueError("invalid course response")
            validate_origin(entry)
            key = signature(entry["request"])
            if key != entry.get("key"):
                raise ValueError("course request digest mismatch")
            identity = record_identity(entry)
            if entry.get("id") != identity:
                raise ValueError("course response digest mismatch")
            streaming = bool(entry["request"].get("stream"))
            if streaming != isinstance(entry["response"], list):
                raise ValueError("course response transport mismatch")
            validate_response(entry["response"], streaming)
            entries.setdefault(key, []).append(entry)
        if sum(map(len, entries.values())) != manifest.get("entries"):
            raise ValueError("course bundle entry count mismatch")
        origins = {}
        for entry in data["entries"]:
            label = "nvidia" if entry["origin_provider"] == NVIDIA_URL else entry["origin_provider"]
            origins[label] = origins.get(label, 0) + 1
        if "origin_providers" in manifest or any("origin_schema" in entry for entry in data["entries"]):
            if manifest.get("origin_providers") != origins:
                raise ValueError("course bundle provider counts mismatch")
            summary = NVIDIA_URL if set(origins) == {"nvidia"} else "mixed"
            if manifest.get("origin_provider") != summary:
                raise ValueError("course bundle provider summary mismatch")
        preferences = data["provenance"].get("preferred_entries", {})
        if not isinstance(preferences, dict):
            raise ValueError("invalid course response preferences")
        for key, identity in preferences.items():
            if not any(entry["id"] == identity for entry in entries.get(key, [])):
                raise ValueError("preferred course response is not recorded")
        numeric_entries, numeric_preferences = {}, {}
        for entry in data["entries"]:
            numeric_key = numeric_signature(entry["request"])
            numeric_entries.setdefault(numeric_key, []).append(entry)
            preferred = preferences.get(entry["key"])
            if preferred is not None:
                numeric_preferences.setdefault(numeric_key, {})[entry["key"]] = preferred
        numeric_preferred, resolutions = {}, {}
        for numeric_key, choices in numeric_preferences.items():
            # A legacy bundle may prefer the identical response twice under 0
            # and 0.0. Only identical response and provider evidence can resolve
            # automatically; different reviewed answers require explicit review.
            if len(set(choices.values())) > 1:
                selected_entries = [entry for entry in numeric_entries[numeric_key]
                                    if entry["id"] in choices.values()]
                evidence = {json.dumps({
                    "response": entry["response"],
                    "provider_request": entry.get("provider_request"),
                    **{name: value for name, value in entry.items() if name.startswith("origin_")},
                }, sort_keys=True, separators=(",", ":")) for entry in selected_entries}
                if len(evidence) != 1:
                    raise ValueError("Conflicting preferred responses for numeric-equivalent course requests require review")
            selected_key = numeric_key if numeric_key in choices else min(choices)
            numeric_preferred[numeric_key] = choices[selected_key]
            if len(set(choices.values())) > 1:
                resolutions[numeric_key] = {
                    "rule": "canonical_numeric_key_then_recorded_key",
                    "scope": "default_selection",
                    "selected_recorded_key": selected_key,
                    "selected_entry_id": choices[selected_key],
                    "recorded_preferences": dict(sorted(choices.items())),
                }
        self.preferred = preferences
        self.numeric_entries = numeric_entries
        self.numeric_preferred = numeric_preferred
        self.numeric_preference_resolutions = resolutions
        self.entries = entries

    def get(self, body, sample=False, sample_index=None):
        with self.lock:
            try:
                self._load()
            except (OSError, ValueError, TypeError, KeyError, EOFError):
                self.entries = {}
                self.unavailable = True
                return None
            # NVIDIA gives the modern limit precedence over an SDK-injected legacy
            # default. Keep stored requests and identities immutable; only lookup
            # candidates use this documented, provider-tested parameter alias.
            requested_key = signature(body)
            current = body
            modern = body.get("max_completion_tokens")
            if modern is not None:
                try:
                    current = completion_limit(body)
                except ValueError:
                    return None
            effective_key = signature(current)
            key = effective_key
            try:
                pool_key = numeric_signature(current)
            except ValueError:
                return None
            pool = self.numeric_entries.get(pool_key, [])
            legacy_alias = False
            if not pool and modern is not None and body.get("model") in MODEL_ALIASES:
                legacy = {k: v for k, v in current.items() if k != "max_completion_tokens"}
                legacy["max_tokens"] = modern
                key = signature(legacy)
                pool_key = numeric_signature(legacy)
                pool = [entry for entry in self.numeric_entries.get(pool_key, [])
                        if "origin_schema" not in entry and entry["origin_provider"] == NVIDIA_URL]
                legacy_alias = bool(pool)
            mixed_limit_pool = False
            if sample and body.get("model") in MODEL_ALIASES:
                try:
                    normalized = completion_limit(body)
                    limit = normalized.get("max_completion_tokens")
                    if limit is not None:
                        canonical_key = numeric_signature(normalized)
                        older = {k: v for k, v in normalized.items() if k != "max_completion_tokens"}
                        older["max_tokens"] = limit
                        exact = self.numeric_entries.get(canonical_key, [])
                        compatible = [entry for entry in self.numeric_entries.get(numeric_signature(older), [])
                                      if "origin_schema" not in entry and entry["origin_provider"] == NVIDIA_URL]
                        if exact and compatible:
                            # New recordings must not shadow earlier reviewed samples.
                            # Both wire forms use the same pool without rekeying evidence.
                            pool = exact + compatible
                            pool_key = canonical_key
                            mixed_limit_pool = True
                except ValueError:
                    return None
            if not pool:
                return None
            index = 0
            if not sample and pool_key in self.numeric_preferred:
                index = next((i for i, entry in enumerate(pool) if entry["id"] == self.numeric_preferred[pool_key]), 0)
            selection = "default"
            if sample:
                if sample_index is not None:
                    index = sample_index % len(pool)
                    selection = "explicit_index"
                elif body.get("seed") is not None:
                    selection = "seeded_exact"
                else:
                    index = self.cursors.get(pool_key, 0) % len(pool)
                    self.cursors[pool_key] = index + 1
                    selection = "cycle"
            entry = pool[index]
            metadata = {"status": "BUNDLE_HIT", "layer": "course_bundle", "reason": "recorded_sample" if sample else "vendored_exact_match",
                "upstream_request": False, "provider_usage_reused": False, "recorded_provider_usage": True,
                "usage_origin_provider": entry["origin_provider"], "namespace": NAMESPACE,
                "origin_provider": entry["origin_provider"], "origin_model": entry.get("origin_model"),
                "bundle_sha256": self.digest, "entry_id": entry["id"], "sample_index": index, "sample_count": len(pool),
                "sample_selection": selection}
            if "origin_schema" in entry:
                metadata["origin_schema"] = entry["origin_schema"]
                metadata["provider_requested_model"] = entry["provider_request"]["model"]
            if pool_key in self.numeric_preference_resolutions:
                metadata["numeric_preference_resolution"] = copy.deepcopy(self.numeric_preference_resolutions[pool_key])
            numeric_alias = entry["key"] != key
            if legacy_alias or requested_key != effective_key or numeric_alias:
                metadata["request_compatibility"] = {
                    "rule": ("max_completion_tokens_to_max_tokens" if legacy_alias else
                             "max_completion_tokens_precedence" if requested_key != effective_key else
                             "numeric_json_equivalence"),
                    "requested_key": requested_key, "effective_key": effective_key,
                    "recorded_key": entry["key"], "max_completion_tokens": modern,
                }
                if legacy_alias:
                    metadata["reason"] = "vendored_completion_limit_alias"
                if numeric_alias:
                    metadata["request_compatibility"]["numeric_json_equivalence"] = True
                    metadata["request_compatibility"]["lookup_key"] = pool_key
                    if not legacy_alias:
                        metadata["reason"] = "vendored_numeric_json_equivalence"
            if mixed_limit_pool:
                metadata["request_compatibility"] = {
                    "rule": "nvidia_completion_limit_sampling", "requested_key": requested_key,
                    "recorded_key": entry["key"], "lookup_key": pool_key,
                }
            return copy.deepcopy(entry["response"]), metadata


def record_response(path, url, body, response, clean, *, canonical_body=None):
    if not path or canonical_body is None and url != NVIDIA_URL:
        return
    label = capture_provider_label(url) if canonical_body is not None else None
    if canonical_body is not None and label is None:
        return
    request = clean({k: v for k, v in (canonical_body if canonical_body is not None else body).items() if k not in CONTROLS})
    response = clean(copy.deepcopy(response))
    if isinstance(response, dict):
        response.pop("dli_cache", None)
        response.pop("dli_routing", None)
    else:
        for chunk in response:
            chunk.pop("dli_cache", None)
            chunk.pop("dli_routing", None)
    key = signature(request)
    payload = {"key": key, "request": request, "response": response, "origin_provider": url,
        "origin_model": body.get("model"), "recorded_at": time.time()}
    if canonical_body is not None:
        payload.update(origin_schema=ORIGIN_SCHEMA, origin_provider=label,
                       origin_model=response_model(response),
                       provider_request=clean({k: v for k, v in body.items() if k not in CONTROLS}))
        validate_origin(payload)
        validate_response(response, bool(request.get("stream")))
    payload["id"] = record_identity(payload)
    encoded = json.dumps(payload, sort_keys=True) + "\n"
    if len(encoded.encode()) > 4194304:
        return
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise ValueError("capture directory must not be a symlink")
    os.chmod(target.parent, 0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("capture target must be a regular file")
        os.fchmod(stream.fileno(), 0o600)
        stream.write(encoded)


def package_records(records, output, provenance):
    entries = {}
    for entry in records:
        validate_origin(entry)
        if not isinstance(entry.get("response"), (dict, list)):
            raise ValueError("invalid recorded response")
        validate_response(entry['response'], bool(entry['request'].get('stream')))
        entry = copy.deepcopy(entry)
        entry["key"] = signature(entry["request"])
        entry["id"] = record_identity(entry)
        previous = entries.get(entry["id"])
        if previous is None or entry["recorded_at"] < previous["recorded_at"]:
            entries[entry["id"]] = entry
    ordered = sorted(entries.values(), key=lambda item: (item["key"], item["recorded_at"], item["id"]))
    data = {"format": FORMAT, "namespace": NAMESPACE, "provenance": provenance, "entries": ordered}
    artifact = gzip.compress(json.dumps(data, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(artifact)
    origins = {}
    for entry in ordered:
        label = "nvidia" if entry["origin_provider"] == NVIDIA_URL else entry["origin_provider"]
        origins[label] = origins.get(label, 0) + 1
    manifest = {"format": FORMAT, "namespace": NAMESPACE, "sha256": hashlib.sha256(artifact).hexdigest(), "entries": len(ordered),
        "distinct_requests": len({entry["key"] for entry in ordered}), "compressed_bytes": len(artifact),
        "origin_provider": NVIDIA_URL if set(origins) == {"nvidia"} else "mixed",
        "origin_providers": origins, "provenance": provenance}
    Path(str(target) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
