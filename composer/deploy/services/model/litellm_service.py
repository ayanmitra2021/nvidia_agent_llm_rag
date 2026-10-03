"""OpenAI-compatible provider routing; credentials never cross the private hop."""
import asyncio
import json
import uuid
import os
import re
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["OPENAI_LOG"] = "error"

import httpx
import anyio
import litellm
from llm_store import Store
from llm_course_cache import CourseCache, record_response
from llm_cache_pacing import Playback, PROFILES
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from llm_output import diagnostic
from llm_request import structured_output
from llm_ttft import first_output, close_stream, assemble, CompletionState, upstream_status
from llm_circuit import Circuits
from provider_compat import provider_url, compatibility, request_body, response_body, env_name
from provider_policy import allowed_models, model_allowed
from pathlib import Path
from llm_profile import default_profile, validate_settings
from llm_runtime import Runtime, RuntimeManager, RuntimeRequests
from llm_tracing import RequestTracing, setup_tracing, cache_attributes, provider_attempt

litellm.set_verbose = False
litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.callbacks = []
litellm.success_callback = []
litellm.failure_callback = []


def canonical_url(value):
    if not isinstance(value, str):
        raise ValueError("invalid provider URL")
    parsed = urlsplit(value)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (parsed.scheme != "https" and not (parsed.scheme == "http" and local)) or not parsed.hostname:
        raise ValueError("provider URL must use HTTPS")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or any(c.isspace() for c in value):
        raise ValueError("invalid provider URL")
    return value.rstrip("/")


def fallback_entry():
    configured = os.getenv("DLI_LLM_FALLBACK_PROVIDER")
    if configured:
        return json.loads(configured)
    if os.getenv("NEBIUS_BASE_URL"):
        return {"base_url_env": "NEBIUS_BASE_URL", "api_key_env": "NEBIUS_TOKEN_FACTORY_KEY",
            "model_aliases": json.loads(os.getenv("NEBIUS_MODEL_ALIASES",
                '{"nvidia/nemotron-3.5-lightning-30b-a3b":"nvidia/Nemotron-3_5-Lightning",'
                '"nvidia/nemotron-3-super-120b-a12b":"nvidia/nemotron-3-super-120b-a12b"}'))}
    return None


def load_providers():
    configured = os.getenv("DLI_LLM_PROVIDERS")
    entries = json.loads(configured) if configured else [
        {"base_url": "https://integrate.api.nvidia.com/v1", "api_key_env": "NVIDIA_API_KEY"},
        {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
        {"base_url": "https://inference-api.nvidia.com/v1", "api_key_env": "NVINFR_API_KEY"},
    ]
    if not isinstance(entries, list):
        raise ValueError("provider configuration must be an array")
    fallback = fallback_entry()
    if fallback is not None:
        if not isinstance(fallback, dict) or "base_url_env" not in fallback or "base_url" in fallback:
            raise ValueError("fallback requires environment references")
        fallback_url = canonical_url(provider_url(fallback))
        if not any(canonical_url(provider_url(entry)) == fallback_url for entry in entries):
            entries = entries + [fallback]
    providers = {}
    for entry in entries:
        if not isinstance(entry, dict) or "api_key_env" not in entry or set(entry) - {"base_url", "base_url_env", "api_key_env", "model_aliases", "reasoning_budget_parameter", "reasoning_response_field"}:
            raise ValueError("invalid provider configuration")
        name = entry["api_key_env"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ValueError("invalid credential reference")
        url = canonical_url(provider_url(entry))
        if url in providers:
            raise ValueError("duplicate provider URL")
        key = os.getenv(name)
        if not key:
            if configured or entry is fallback:
                raise ValueError("missing provider credential")
            continue
        aliases = entry.get("model_aliases", {})
        if not isinstance(aliases, dict) or any(not isinstance(a, str) or not isinstance(b, str) or not a or not b for a, b in aliases.items()):
            raise ValueError("invalid model aliases")
        providers[url] = (key, aliases, compatibility(entry), allowed_models(entry))
    if not providers:
        raise ValueError("no configured providers")
    return providers


PROVIDERS = load_providers()
FALLBACK_URL = canonical_url(provider_url(fallback_entry())) if fallback_entry() is not None else None
if FALLBACK_URL is not None and FALLBACK_URL not in PROVIDERS:
    raise ValueError("fallback provider is not configured")
SECRETS = tuple(value[0] for value in PROVIDERS.values())


def clean(value):
    return runtime().clean(value)


STORE_PATH = os.getenv("LLM_STORE_PATH", "/var/lib/llm/requests.sqlite")
STORE = Store(STORE_PATH, clean,
    ttl=int(os.getenv("LLM_CACHE_TTL_SECONDS", "86400")),
    max_bytes=int(os.getenv("LLM_CACHE_MAX_BYTES", "67108864")),
    max_entries=int(os.getenv("LLM_CACHE_MAX_ENTRIES", "2000")),
    namespace=os.getenv("LLM_CACHE_NAMESPACE", "c-fx-25-v1:litellm-1.100.1"))
ACTIVITY_MODELS = set()
PENDING = {}
PENDING_USERS = {}
COURSE_CACHE = CourseCache()
COURSE_CACHE_ENABLED = os.getenv("LLM_COURSE_CACHE_ENABLED", "true").lower() in {"true", "1"}
COURSE_CAPTURE = os.getenv("LLM_CAPTURE_COURSE_CACHE", "")
CAPTURE_FAILURES = 0
DEFAULT_PACING = os.getenv("LLM_CACHE_PACING")
if DEFAULT_PACING is not None and DEFAULT_PACING not in PROFILES:
    raise ValueError("Invalid LLM_CACHE_PACING profile")
DEFAULT_USE_CACHE = os.getenv("LLM_CACHE_ENABLED", "true").lower() in {"true", "1"}
RESPECT_CACHE_HEADERS = os.getenv("LLM_CACHE_RESPECT_HEADERS", "false").lower() in {"true", "1"}
REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "300"))
TTFT_TIMEOUT = float(os.getenv("LLM_MAX_TTFT_SECONDS", "10"))
PRIMARY_ONLY_TTFT_TIMEOUT = float(os.getenv("LLM_PRIMARY_ONLY_TTFT_SECONDS",
    os.getenv("LLM_MAX_TTFT_SECONDS", "45")))
if not all(0 < limit <= 1800 for limit in (TTFT_TIMEOUT, PRIMARY_ONLY_TTFT_TIMEOUT)):
    raise ValueError("invalid first-token limit")
RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
if not 1 <= REQUEST_TIMEOUT <= 1800 or not 1 <= RETRY_ATTEMPTS <= 5:
    raise ValueError("invalid request limits")
CIRCUITS = Circuits(int(os.getenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "3")),
    float(os.getenv("LLM_CIRCUIT_BASE_COOLDOWN_SECONDS", "30")),
    float(os.getenv("LLM_CIRCUIT_MAX_COOLDOWN_SECONDS", "300")))
INSTANCE_ID = uuid.uuid4().hex

settings = default_profile()
settings["cache"].update(enabled=DEFAULT_USE_CACHE, vendored=COURSE_CACHE_ENABLED,
    respect_http_headers=RESPECT_CACHE_HEADERS, ttl_seconds=STORE.ttl,
    max_entries=STORE.max_entries, max_bytes=STORE.max_bytes)
from llm_cache_pacing import INITIAL_DELAY_SECONDS, MAX_STREAM_DELAY_SECONDS, MAX_RESPONSE_DELAY_SECONDS
settings["playback"].update(profile=DEFAULT_PACING or "chat", speeds={k: v for k, v in PROFILES.items() if k != "off"},
    initial_delay_seconds=INITIAL_DELAY_SECONDS, max_stream_delay=MAX_STREAM_DELAY_SECONDS,
    max_response_delay=MAX_RESPONSE_DELAY_SECONDS)
settings["requests"].update(timeout_seconds=REQUEST_TIMEOUT, max_ttft_seconds=TTFT_TIMEOUT,
    retry_attempts=RETRY_ATTEMPTS, failure_threshold=CIRCUITS.threshold,
    base_cooldown_seconds=CIRCUITS.base, max_cooldown_seconds=CIRCUITS.maximum)
INITIAL = Runtime(id="deployment", mode="course", settings=settings,
    base_url=os.getenv(os.getenv("LLM_DEFAULT_BASE_URL_ENV", "")) or
        ("https://integrate.api.nvidia.com/v1" if "https://integrate.api.nvidia.com/v1" in PROVIDERS else next(iter(PROVIDERS))),
    PROVIDERS=PROVIDERS, FALLBACK_URL=FALLBACK_URL, STORE=STORE, CIRCUITS=CIRCUITS,
    COURSE_CACHE_ENABLED=COURSE_CACHE_ENABLED, DEFAULT_USE_CACHE=DEFAULT_USE_CACHE,
    DEFAULT_PACING=DEFAULT_PACING, RESPECT_CACHE_HEADERS=RESPECT_CACHE_HEADERS,
    REQUEST_TIMEOUT=REQUEST_TIMEOUT, TTFT_TIMEOUT=TTFT_TIMEOUT,
    PRIMARY_ONLY_TTFT_TIMEOUT=PRIMARY_ONLY_TTFT_TIMEOUT, RETRY_ATTEMPTS=RETRY_ATTEMPTS)
STORE.clean = INITIAL.clean
MANAGER = RuntimeManager(INITIAL, Path(STORE_PATH).with_name("runtime-profile.json"))
runtime = MANAGER.get
TRACING = setup_tracing("course-llm-wrapper")


def validated_catalog(state, url, catalog):
    if not isinstance(catalog, dict) or any(not isinstance(name, str) or not isinstance(value, dict)
            or value.get("id") != name for name, value in catalog.items()):
        raise ValueError("Provider returned an invalid model catalog")
    original = dict(catalog)
    catalog = dict(original)
    for alias, target in state.PROVIDERS[url][1].items():
        if target not in original:
            raise ValueError("A model mapping target is absent from the provider catalog")
        catalog[alias] = {**original[target], "id": alias}
    catalog = state.clean({name: value for name, value in catalog.items() if model_allowed(state.PROVIDERS[url], name)})
    if not catalog:
        raise ValueError("Provider has no available models")
    return catalog


def recorded_catalog(state, url):
    """Discovery failure must not make verified recordings unreachable.

    This catalog promises a recorded interface, not live provider availability.
    Custom credentials and disabled course caches never inherit these entries.
    """
    if state.mode != "course" or not state.COURSE_CACHE_ENABLED:
        return {}
    COURSE_CACHE.get({})
    if COURSE_CACHE.unavailable:
        return {}
    return {model: {"id": model, "object": "model", "owned_by": "course-recordings",
                    "dli_catalog_source": "recorded"}
            for model in {entry["request"]["model"] for pool in COURSE_CACHE.entries.values() for entry in pool}
            if model_allowed(state.PROVIDERS[url], model)}


async def fetch_catalog(state, url):
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False, trust_env=False) as connection:
            response = await connection.get(url + "/models", headers={"Authorization": "Bearer " + state.PROVIDERS[url][0]})
        response.raise_for_status()
        body = response.json()
        if not isinstance(body.get("data"), list):
            raise ValueError("Provider returned an invalid model catalog")
        catalog = {m["id"]: m for m in body["data"] if isinstance(m, dict) and isinstance(m.get("id"), str)}
        catalog = validated_catalog(state, url, catalog)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        catalog = recorded_catalog(state, url)
        if not catalog:
            raise
    state.catalog[url] = catalog
    ACTIVITY_MODELS.update(catalog)
    return catalog


async def make_runtime(profile, saved_catalog=None, revision=None):
    if not isinstance(profile, dict) or set(profile) != {"mode", "settings", "provider"}:
        raise ValueError("Supply mode, settings, and provider")
    configured = validate_settings(profile["settings"])
    mode, custom = profile["mode"], profile["provider"]
    if mode == "course" and custom is None:
        providers, fallback, base_url = INITIAL.PROVIDERS, INITIAL.FALLBACK_URL, INITIAL.base_url
    elif mode == "custom" and isinstance(custom, dict) and set(custom) == {
            "base_url", "api_key", "model_aliases", "reasoning_budget_parameter", "reasoning_response_field"}:
        base_url = canonical_url(custom["base_url"])
        key = custom["api_key"]
        if (not isinstance(key, str) or not key.strip() or key != key.strip() or len(key) > 16384
                or any(ord(c) < 32 or ord(c) == 127 for c in key)):
            raise ValueError("Enter a nonempty API key without surrounding whitespace")
        aliases = custom["model_aliases"]
        if not isinstance(aliases, dict) or len(aliases) > 200 or any(
                not isinstance(v, str) or not v.strip() or len(v) > 1024 or any(ord(c) < 32 for c in v)
                for pair in aliases.items() for v in pair):
            raise ValueError("Invalid model mappings")
        dialect = compatibility(custom)
        providers, fallback = {base_url: (key, dict(aliases), dialect, None)}, None
        configured["cache"]["vendored"] = False
    else:
        raise ValueError("Choose the course profile or supply your own provider credential")
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{32}", revision)):
        raise ValueError("Invalid stored profile revision")
    revision = revision or uuid.uuid4().hex
    cache, limits = configured["cache"], configured["requests"]
    state = Runtime(id=revision, mode=mode, settings=configured, base_url=base_url,
        PROVIDERS=providers, FALLBACK_URL=fallback, COURSE_CACHE_ENABLED=cache["vendored"],
        DEFAULT_USE_CACHE=cache["enabled"], RESPECT_CACHE_HEADERS=cache["respect_http_headers"],
        DEFAULT_PACING=configured["playback"]["profile"], REQUEST_TIMEOUT=limits["timeout_seconds"],
        TTFT_TIMEOUT=limits["max_ttft_seconds"], RETRY_ATTEMPTS=limits["retry_attempts"],
        PRIMARY_ONLY_TTFT_TIMEOUT=limits["max_ttft_seconds"],
        CIRCUITS=Circuits(limits["failure_threshold"], limits["base_cooldown_seconds"], limits["max_cooldown_seconds"]))
    state.STORE = Store(STORE_PATH, state.clean,
        ttl=cache["ttl_seconds"], max_bytes=cache["max_bytes"], max_entries=cache["max_entries"],
        namespace="course-profile:" + revision)
    try:
        if saved_catalog is not None:
            if not isinstance(saved_catalog, dict) or base_url not in saved_catalog or not isinstance(saved_catalog[base_url], dict):
                raise ValueError("Invalid stored profile catalog")
            state.catalog = {base_url: validated_catalog(state, base_url, saved_catalog[base_url])}
        else:
            for previous in (MANAGER.current, INITIAL):
                if previous.PROVIDERS.get(base_url) == providers[base_url] and previous.catalog.get(base_url):
                    state.catalog[base_url] = validated_catalog(state, base_url, previous.catalog[base_url])
                    break
            if base_url not in state.catalog:
                await fetch_catalog(state, base_url)
    except BaseException:
        state.STORE.db.close()
        raise
    return state


def profile_view(state, *, internal=False):
    result = {"revision": state.id, "mode": state.mode, "settings": state.settings, "active": True}
    if internal:
        result.update(base_url=state.base_url, models=state.catalog.get(state.base_url, {}))
    return result


def capture_response(url, body, response, *, canonical_body):
    """Optional private recording must never change a successful completion."""
    global CAPTURE_FAILURES
    try:
        record_response(COURSE_CAPTURE, url, body, response, clean, canonical_body=canonical_body)
    except (OSError, ValueError, TypeError):
        CAPTURE_FAILURES += 1


def http_client(timeout):
    state = runtime()
    async def on_request(request):
        if request.method == "POST":
            fallback = state.FALLBACK_URL and str(request.url).startswith(state.FALLBACK_URL + "/")
            provider_attempt("fallback" if fallback else "primary")
        await state.STORE.on_request(request)
    return httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=timeout,
        event_hooks={"request": [on_request], "response": [state.STORE.on_response]})


def cache_controls(request):
    return set(request.headers.get("cache-control", "").lower().replace(" ", "").split(",")) if runtime().RESPECT_CACHE_HEADERS else set()


def cacheable(request, use_cache):
    return (use_cache and runtime().STORE.ttl > 0
        and not {"no-cache", "no-store"}.intersection(cache_controls(request)))


def cache_metadata(response, status, reason, details=None):
    if response.status_code != 200:
        return response
    cache_attributes(status, details)
    metadata = {"status": status, "layer": "local_response", "reason": reason,
        "upstream_request": status in {"MISS", "BYPASS"},
        "provider_usage_reused": status in {"HIT", "SHARED"}}
    if details:
        metadata.update(details)
    if isinstance(response, JSONResponse):
        payload = json.loads(response.body)
        payload["dli_cache"] = metadata
        headers = {"X-DLI-Cache": status}
        for name in ("X-DLI-Fallback", "X-DLI-Route", "X-DLI-Circuit"):
            if name in response.headers:
                headers[name] = response.headers[name]
        return JSONResponse(payload, headers=headers)
    original = response.body_iterator
    async def annotated():
        try:
            async for event in original:
                if isinstance(event, str) and event.startswith("data: {"):
                    payload = json.loads(event[6:])
                    payload["dli_cache"] = metadata
                    yield "data: " + json.dumps(payload, sort_keys=True) + "\n\n"
                else:
                    yield event
        finally:
            with anyio.CancelScope(shield=True):
                await original.aclose()
    response.body_iterator = annotated()
    response.headers["X-DLI-Cache"] = status
    return response


def bypass_reason(request, use_cache):
    if not use_cache:
        return "use_cache_false"
    controls = cache_controls(request)
    if "no-store" in controls:
        return "cache_control_no_store"
    if "no-cache" in controls:
        return "cache_control_no_cache"
    if runtime().STORE.ttl <= 0:
        return "cache_disabled"
    return "cache_disabled"


async def cached_response(value, streaming, playback):
    if streaming:
        async def events():
            for chunk in value:
                chunk = {key: item for key, item in chunk.items() if key not in ("dli_cache", "dli_routing")}
                await playback.wait(chunk)
                yield "data: " + json.dumps(chunk, sort_keys=True) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream", headers={"X-DLI-Cache": "HIT"})
    value = {key: item for key, item in value.items() if key not in ("dli_cache", "dli_routing")}
    await playback.wait(value)
    return JSONResponse(value, headers={"X-DLI-Cache": "HIT"})


def failure(status=502):
    diagnostic("failure", status)
    return JSONResponse({"error": {"message": "Provider request failed", "type": "upstream_error"}}, status_code=status)


@asynccontextmanager
async def lifespan(app):
    # OpenAI lazily imports its resource/schema tree on the first chat call.
    # Finish local initialization before readiness, not inside a provider's TTFT.
    import openai.resources
    await MANAGER.restore(make_runtime)
    diagnostic("ready", 200)
    try:
        yield
    finally:
        await asyncio.to_thread(TRACING.shutdown)


app = FastAPI(title="Private LiteLLM wrapper", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(RuntimeRequests, manager=MANAGER)
app.add_middleware(RequestTracing, provider=TRACING)


@app.get("/v1/profile")
async def active_profile(known: str = ""):
    if known == MANAGER.current.id:
        return {"revision": known}
    return profile_view(MANAGER.current, internal=True)


@app.post("/v1/profile")
async def apply_profile(request: Request):
    # This route is private to the container network and is never forwarded by the facade.
    try:
        if len(await request.body()) > 131072:
            return failure(413)
        state = await MANAGER.activate(await request.json(), make_runtime)
        return profile_view(state)
    except (ValueError, TypeError, KeyError):
        return JSONResponse({"error": {"message": "Invalid profile settings or credential"}}, status_code=400)
    except Exception:
        return JSONResponse({"error": {"message": "Profile could not be activated; the previous profile remains active"}}, status_code=502)


@app.get("/v1/default-base-url")
async def default_base_url():
    name = os.getenv("LLM_DEFAULT_BASE_URL_ENV", "")
    url = canonical_url(os.getenv(env_name(name), "")) if name else runtime().base_url
    if url not in runtime().PROVIDERS:
        raise HTTPException(503, "Deployment default is not configured")
    return {"base_url": url}


@app.exception_handler(Exception)
async def unexpected(request, exc):
    return failure()


@app.exception_handler(RequestValidationError)
async def validation(request, exc):
    return failure(400)


def model_denied():
    return JSONResponse({"error": {
        "message": "Model is not available with the managed course credential. Select an available model or configure your own provider credential.",
        "type": "permission_error", "code": "model_not_allowed",
    }}, status_code=403)


def provider(request):
    try:
        url = canonical_url(request.headers.get("x-dli-base-url"))
    except ValueError:
        raise HTTPException(400, "Invalid selected base URL") from None
    if url not in runtime().PROVIDERS:
        raise HTTPException(400, "Selected base URL is not configured")
    return url, runtime().PROVIDERS[url][0]


@app.get("/health")
async def health():
    COURSE_CACHE.get({})
    ready = not runtime().COURSE_CACHE_ENABLED or not COURSE_CACHE.unavailable
    return JSONResponse({"status": "ready" if ready else "cache_unavailable", "instance_id": INSTANCE_ID, "course_cache": {"enabled": runtime().COURSE_CACHE_ENABLED,
        "available": not COURSE_CACHE.unavailable, "entries": sum(map(len, (COURSE_CACHE.entries or {}).values())),
        "sha256": COURSE_CACHE.digest}, "max_ttft_seconds": runtime().TTFT_TIMEOUT,
        "primary_only_ttft_seconds": runtime().PRIMARY_ONLY_TTFT_TIMEOUT,
        "fallback_configured": runtime().FALLBACK_URL is not None}, status_code=200 if ready else 503)


def routing_state(url, model):
    primary = runtime().CIRCUITS.snapshot(url, model)
    fallback = runtime().CIRCUITS.snapshot(runtime().FALLBACK_URL, model) if runtime().FALLBACK_URL and runtime().FALLBACK_URL != url else None
    primary["model_allowed"] = model_allowed(runtime().PROVIDERS[url], model)
    if fallback:
        fallback["model_allowed"] = model_allowed(runtime().PROVIDERS[runtime().FALLBACK_URL], model)
    route = "primary" if primary["available"] and primary["model_allowed"] else "fallback" if fallback and fallback["available"] and fallback["model_allowed"] and primary["model_allowed"] else "unavailable"
    return clean({"model": model, "effective_route": route, "primary": primary,
        "proxy_instance_id": INSTANCE_ID, "course_cache_sha256": COURSE_CACHE.digest,
        "fallback": fallback, "policy": runtime().CIRCUITS.policy()})


@app.get("/v1/activity")
async def activity_status(request: Request, window: int = Query(3600, ge=60, le=86400),
                          limit: int = Query(100, ge=1, le=200), model: str = Query("", max_length=256)):
    provider(request)
    labels = {name: name for name in ACTIVITY_MODELS}
    public_catalog = os.getenv("LLM_ACTIVITY_PUBLIC_CATALOG")
    if public_catalog:
        try:
            names = json.loads(Path(public_catalog).read_text())["model_registry"]
            if not isinstance(names, dict) or not all(isinstance(name, str) and len(name) <= 256 for name in names):
                raise ValueError("invalid public model catalog")
            labels.update({name: name for name in names})
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(503, "Public activity catalog is unavailable") from None
    for entry in runtime().PROVIDERS.values():
        for alias, target in entry[1].items():
            labels[alias] = alias
            labels[target] = alias
    return await MANAGER.run_in_thread(runtime().STORE.activity, window, limit, model, runtime().FALLBACK_URL, labels)


@app.get("/v1/routing")
async def routing_status(request: Request, model: str):
    url, _ = provider(request)
    return routing_state(url, model)


def routing_failure(status, url, model, reason):
    response = failure(status)
    value = json.loads(response.body)
    value["dli_routing"] = {"reason": reason, **routing_state(url, model)}
    headers = {"X-DLI-Route": "unavailable"}
    if status == 503:
        waits = [s["retry_after_seconds"] for s in (value["dli_routing"]["primary"], value["dli_routing"]["fallback"]) if s]
        headers["Retry-After"] = str(max(1, int(min(waits, default=1) + 0.999)))
    return JSONResponse(value, status_code=status, headers=headers)


@app.get("/v1/models")
async def models(request: Request):
    url, _ = provider(request)
    try:
        catalog = await fetch_catalog(runtime(), url)
        return clean({"object": "list", "data": list(catalog.values())})
    except Exception:
        return failure()


_CHAT_FIELDS = frozenset({
    "messages", "temperature", "top_p", "n", "stream", "stop", "max_tokens", "max_completion_tokens",
    "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format", "seed", "tools",
    "tool_choice", "parallel_tool_calls", "stream_options", "logprobs", "top_logprobs", "reasoning_effort",
})
_EXTRA_FIELDS = frozenset({"chat_template_kwargs", "reasoning_budget", "thinking_token_budget", "top_k", "min_p", "repetition_penalty"})


@app.post("/v1/chat/completions")
async def completion(request: Request):
    url, key = provider(request)
    try:
        body = await request.json()
    except Exception:
        return failure(400)
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        return failure(400)
    if not model_allowed(runtime().PROVIDERS[url], body["model"]):
        return model_denied()
    try:
        from llm_request import completion_limit
        body = completion_limit(body)
    except ValueError:
        return failure(400)
    if type(body.get("n", 1)) is not int or body.get("n", 1) < 1:
        return failure(400)
    if body.get("stream_options") is not None and not isinstance(body["stream_options"], dict):
        return failure(400)
    use_cache = body.pop("use_cache", runtime().DEFAULT_USE_CACHE)
    use_course_cache = body.pop("use_course_cache", runtime().COURSE_CACHE_ENABLED)
    sample = body.pop("cache_sample", False)
    sample_index = body.pop("cache_sample_index", None)
    pacing = body.pop("cache_pacing", runtime().DEFAULT_PACING or ("chat" if body.get("stream") else "automation"))
    if not isinstance(pacing, str) or pacing not in PROFILES:
        return failure(400)
    if any(type(value) is not bool for value in (use_cache, use_course_cache, sample)):
        return failure(400)
    if sample_index is not None and (type(sample_index) is not int or sample_index < 0 or not sample):
        return failure(400)
    try:
        provider_body = structured_output(body)
    except ValueError:
        return failure(400)
    if set(provider_body) - _CHAT_FIELDS - _EXTRA_FIELDS - {"model"}:
        return failure(400)
    controls = cache_controls(request)
    bundle_unavailable = False
    explicit_bypass = not use_cache or bool({"no-cache", "no-store"}.intersection(controls))
    if use_course_cache and use_cache and not {"no-cache", "no-store"}.intersection(controls):
        temperature = body.get("temperature")
        varied = type(temperature) in (int, float) and temperature > 0
        hit = COURSE_CACHE.get(body, sample or varied, sample_index)
        if hit:
            value, metadata = hit
            runtime().STORE.record("POST", url + "/chat/completions", body, "course_bundle", 200)
            playback = Playback(value, pacing, settings=runtime().settings["playback"])
            metadata["pacing"] = playback.metadata
            return cache_metadata(await cached_response(value, bool(body.get("stream")), playback), "BUNDLE_HIT", metadata["reason"], metadata)
        bundle_unavailable = COURSE_CACHE.unavailable
    if sample:
        use_cache = False
    canonical_body = dict(body)
    body = provider_body
    body["model"] = runtime().PROVIDERS[url][1].get(body["model"], body["model"])
    dialect = runtime().PROVIDERS[url][2]
    try:
        body = request_body(body, dialect)
    except ValueError:
        return failure(400)
    cache_body = body if dialect == compatibility({}) else {"request": body, "compatibility": dialect}
    cache_key = runtime().STORE.key(url, cache_body) if cacheable(request, use_cache) else None
    if cache_key:
        cached = runtime().STORE.get(cache_key, runtime().STORE.legacy_key(url, cache_body))
        if cached is not None:
            runtime().STORE.record("POST", url + "/chat/completions", body, "cache", 200)
            playback = Playback(cached, pacing, settings=runtime().settings["playback"])
            return cache_metadata(await cached_response(cached, bool(body.get("stream")), playback), "HIT", "exact_match", {"pacing": playback.metadata})
        if not body.get("stream"):
            shared = cache_key in PENDING
            if not shared:
                async def fill():
                    result = await perform_completion(url, key, body, cache_key, canonical_body)
                    if result.status_code == 200 and result.headers.get("x-dli-fallback") != "true":
                        payload = json.loads(result.body)
                        runtime().STORE.put(cache_key, {key: item for key, item in payload.items() if key not in ("dli_cache", "dli_routing")})
                    return result
                task = asyncio.create_task(fill())
                PENDING[cache_key] = task
                task.add_done_callback(lambda done: PENDING.pop(cache_key, None))
            task = PENDING[cache_key]
            PENDING_USERS[task] = PENDING_USERS.get(task, 0) + 1
            try:
                result = await connected_result(request, task)
            finally:
                PENDING_USERS[task] -= 1
                if not PENDING_USERS[task]:
                    del PENDING_USERS[task]
                    if not task.done():
                        task.cancel()
            if shared:
                runtime().STORE.record("POST", url + "/chat/completions", body, "shared", result.status_code)
            reason = "course_bundle_unavailable" if bundle_unavailable else ("in_flight_match" if shared else "exact_match_absent")
            return cache_metadata(result, "SHARED" if shared else "MISS", reason)
    task = asyncio.create_task(perform_completion(url, key, body, cache_key, canonical_body))
    try:
        result = await connected_result(request, task)
    finally:
        if not task.done():
            task.cancel()
    reason = "exact_match_absent" if cache_key else bypass_reason(request, use_cache)
    if sample and not explicit_bypass:
        reason = "course_sample_absent" if use_course_cache else "course_samples_disabled"
    if bundle_unavailable:
        reason = "course_bundle_unavailable"
    return cache_metadata(result, "MISS" if cache_key else "BYPASS", reason)


async def connected_result(request, task):
    while not task.done():
        if await request.is_disconnected():
            raise HTTPException(499, "Request cancelled")
        await asyncio.wait({task}, timeout=0.05)
    response = task.result()
    if isinstance(response, StreamingResponse):
        response.body_iterator = connected_stream(request, response.body_iterator)
    return response


async def connected_stream(request, original):
    # ASGI 2.4 servers can report disconnect only on send; a stalled upstream
    # needs a receive-side monitor to release its connection immediately.
    with anyio.CancelScope() as scope:
        async def watch():
            while True:
                if await request.is_disconnected():
                    scope.cancel()
                    return
                await asyncio.sleep(0.05)
        watcher = asyncio.create_task(watch())
        try:
            async for event in original:
                yield event
        finally:
            watcher.cancel()
            with anyio.CancelScope(shield=True):
                await original.aclose()


async def perform_completion(url, key, body, cache_key, canonical_body):
    from openai import AsyncOpenAI
    selected = url
    fallback = False
    sdk_client = None
    result = None
    ticket = None
    model = canonical_body["model"]
    eligible_fallback = (runtime().FALLBACK_URL and runtime().FALLBACK_URL != url
        and model_allowed(runtime().PROVIDERS[runtime().FALLBACK_URL], model))
    ttft_timeout = runtime().TTFT_TIMEOUT if eligible_fallback else runtime().PRIMARY_ONLY_TTFT_TIMEOUT
    reason = "primary"
    transient = {408, 429, *range(500, 600)}
    for candidate in [url] + ([runtime().FALLBACK_URL] if runtime().FALLBACK_URL and runtime().FALLBACK_URL != url else []):
        selected = candidate
        if not model_allowed(runtime().PROVIDERS[selected], model):
            return routing_failure(503, url, model, "fallback_model_not_allowed")
        try:
            ticket = runtime().CIRCUITS.acquire(selected, model)
        except RuntimeError:
            return routing_failure(503, url, model, "health_capacity_reached")
        if ticket is None:
            if selected == url:
                reason = "primary_cooldown"
            continue
        fallback = selected != url
        if ticket.probe and not fallback:
            reason = "primary_recovery_probe"
        routed = structured_output(canonical_body)
        routed["model"] = runtime().PROVIDERS[selected][1].get(routed["model"], routed["model"])
        try:
            routed = request_body(routed, runtime().PROVIDERS[selected][2])
        except ValueError:
            runtime().CIRCUITS.finish(ticket, "cancel")
            return routing_failure(400, url, model, "invalid_request")
        kwargs = {k: v for k, v in routed.items() if k in _CHAT_FIELDS}
        kwargs["stream"] = True
        kwargs["stream_options"] = {**(kwargs.get("stream_options") or {}), "include_usage": True}
        extras = {k: v for k, v in routed.items() if k in _EXTRA_FIELDS}
        provider_request = {"model": routed["model"], **kwargs, **extras}
        transport = None
        try:
            transport = http_client(runtime().REQUEST_TIMEOUT)
            sdk_client = AsyncOpenAI(api_key=runtime().PROVIDERS[selected][0], base_url=selected,
                http_client=transport, max_retries=0)
            # Resolve lazy SDK resources before timing provider output.
            create_completion = sdk_client.chat.completions.create
        except BaseException:
            # Local initialization owns the acquired probe until an SDK client exists.
            runtime().CIRCUITS.finish(ticket, "cancel")
            if transport is not None:
                with anyio.CancelScope(shield=True):
                    await transport.aclose()
            raise

        async def open_stream():
            return await create_completion(model=routed["model"],
                timeout=runtime().REQUEST_TIMEOUT, extra_body=extras, **kwargs)
        try:
            result, iterator, buffered = await first_output(open_stream, ttft_timeout, runtime().RETRY_ATTEMPTS)
            break
        except asyncio.CancelledError:
            runtime().CIRCUITS.finish(ticket, "cancel")
            await sdk_client.close()
            raise
        except Exception as exc:
            timeout = isinstance(exc, TimeoutError)
            status = 504 if timeout else upstream_status(exc)
            status = status if type(status) is int and 400 <= status <= 599 else 502
            retryable = status in transient
            authentication_failed = status == 401
            failure_reason = ("provider_authentication_failed" if authentication_failed else
                              "ttft_timeout" if timeout else "transient_http" if getattr(exc, "status_code", None)
                              else "provider_stream_error" if isinstance(getattr(exc, "body", None), dict)
                              else "transport_error")
            runtime().CIRCUITS.finish(ticket, "failure" if retryable or authentication_failed else "cancel", failure_reason)
            await sdk_client.close()
            if selected == url and runtime().FALLBACK_URL and runtime().FALLBACK_URL != url and (retryable or authentication_failed):
                reason = ("primary_authentication_failed" if authentication_failed else
                          "primary_ttft_exceeded" if timeout else "primary_transient_failure")
                continue
            if authentication_failed:
                return routing_failure(503, url, model, failure_reason)
            return routing_failure(status, url, model, failure_reason if retryable else "request_rejected")
    else:
        return routing_failure(503, url, model, "providers_cooling_down")

    routing = clean({"fallback_used": fallback, "reason": reason,
        "max_ttft_seconds": ttft_timeout, "requested_model": model,
        "served_model": buffered[-1].model or routed["model"],
        "health_at_dispatch": routing_state(url, model)})
    headers = {"X-DLI-Cache": "MISS" if cache_key else "BYPASS", "X-DLI-Fallback": str(fallback).lower(),
        "X-DLI-Route": "fallback" if fallback else "primary",
        "X-DLI-Circuit": runtime().CIRCUITS.snapshot(url, model)["state"]}

    async def chunks():
        for chunk in buffered:
            yield chunk
        async for chunk in iterator:
            yield chunk

    async def close():
        with anyio.CancelScope(shield=True):
            try:
                await close_stream(result)
            finally:
                await sdk_client.close()

    if not canonical_body.get("stream"):
        try:
            async with asyncio.timeout(runtime().REQUEST_TIMEOUT):
                collected = []
                state = CompletionState(canonical_body.get("n", 1))
                size = 0
                async for chunk in chunks():
                    state.observe(chunk.model_dump(exclude_none=True))
                    size += len(str(chunk))
                    if size > 16777216:
                        raise ValueError("provider response exceeds limit")
                    collected.append(chunk)
                state.complete()
                value = clean(response_body(assemble(collected, routed.get("messages")), runtime().PROVIDERS[selected][2]))
                capture_response(selected, provider_request, value, canonical_body=canonical_body)
                runtime().CIRCUITS.finish(ticket, "success")
                value["dli_routing"] = routing
                diagnostic("request", 200)
                return JSONResponse(value, headers=headers)
        except TimeoutError:
            runtime().CIRCUITS.finish(ticket, "failure", "response_timeout")
            return routing_failure(504, url, model, "response_timeout")
        except Exception:
            runtime().CIRCUITS.finish(ticket, "failure", "stream_incomplete")
            return routing_failure(502, url, model, "stream_incomplete")
        finally:
            runtime().CIRCUITS.finish(ticket, "cancel")
            await close()

    async def events():
        collected, size = [], 0
        state = CompletionState(canonical_body.get("n", 1))
        try:
            async with asyncio.timeout(runtime().REQUEST_TIMEOUT):
                async for chunk in chunks():
                    value = clean(response_body(chunk.model_dump(exclude_none=True), runtime().PROVIDERS[selected][2]))
                    state.observe(value)
                    value.pop("dli_cache", None)
                    value.pop("dli_routing", None)
                    size += len(json.dumps(value, sort_keys=True).encode())
                    if (cache_key or COURSE_CAPTURE) and size <= 4194304:
                        collected.append(value)
                    else:
                        collected.clear()
                    encoded = json.dumps({**value, "dli_routing": routing}, sort_keys=True)
                    yield "data: " + encoded + "\n\n"
                state.complete()
                runtime().CIRCUITS.finish(ticket, "success")
                if cache_key and not fallback and size <= 4194304:
                    runtime().STORE.put(cache_key, collected)
                if size <= 4194304:
                    capture_response(selected, provider_request, collected, canonical_body=canonical_body)
                yield "data: [DONE]\n\n"
        except TimeoutError:
            runtime().CIRCUITS.finish(ticket, "failure", "response_timeout")
            yield 'data: {"error":{"message":"Provider stream timed out","type":"upstream_error"}}\n\n'
        except Exception:
            runtime().CIRCUITS.finish(ticket, "failure", "stream_incomplete")
            yield 'data: {"error":{"message":"Provider stream interrupted","type":"upstream_error"}}\n\n'
        finally:
            runtime().CIRCUITS.finish(ticket, "cancel")
            await close()
    return StreamingResponse(events(), media_type="text/event-stream", headers=headers)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    url, key = provider(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            return failure(400)
        use_cache = body.pop("use_cache", runtime().DEFAULT_USE_CACHE)
        if type(use_cache) is not bool or set(body) - {"model", "input", "encoding_format", "dimensions", "user"}:
            return failure(400)
        if not isinstance(body.get("model"), str):
            return failure(400)
        if not model_allowed(runtime().PROVIDERS[url], body["model"]):
            return model_denied()
        body["model"] = runtime().PROVIDERS[url][1].get(body["model"], body["model"])
        cache_key = runtime().STORE.key(url + "/embeddings", body) if cacheable(request, use_cache) else None
        if cache_key:
            cached = runtime().STORE.get(cache_key, runtime().STORE.legacy_key(url + "/embeddings", body))
            if cached is not None:
                runtime().STORE.record("POST", url + "/embeddings", body, "cache", 200)
                return cache_metadata(JSONResponse(cached), "HIT", "exact_match")
        async with http_client(runtime().REQUEST_TIMEOUT) as client:
            response = await client.post(url + "/embeddings", json=body, headers={"Authorization": "Bearer " + key})
        if response.status_code != 200:
            return failure(response.status_code if 400 <= response.status_code <= 599 else 502)
        result = clean(response.json())
        if cache_key:
            runtime().STORE.put(cache_key, result)
        return cache_metadata(JSONResponse(result), "MISS" if cache_key else "BYPASS",
            "exact_match_absent" if cache_key else bypass_reason(request, use_cache))
    except Exception:
        return failure()
