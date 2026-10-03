"""Course OpenAI facade with one selected upstream URL and no provider registry."""
from llm_output import protect_output, diagnostic
from llm_request import facade_chat_request
from llm_tracing import RequestTracing, setup_tracing, trace_headers, profile_attribute

if __name__ == "__main__":
    protect_output()

import asyncio
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
import anyio
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

LIGHTNING_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
SUPER_MODEL = "nvidia/nemotron-3-super-120b-a12b"
COURSE_AGENT_MODEL = "course/agent"
COURSE_AGENT_MODELS = tuple(json.loads(os.getenv("LLM_COURSE_MODELS", json.dumps([LIGHTNING_MODEL, SUPER_MODEL]))))
LOCAL_RETRIEVAL = os.getenv("LLM_LOCAL_RETRIEVAL", "false").lower() in {"true", "1"}
if LOCAL_RETRIEVAL:
    import local_retrieval
WRAPPER_URL = os.getenv("LLM_WRAPPER_URL", "http://litellm_wrapper:9001").rstrip("/")
REQUIRED_MODELS = json.loads(os.getenv("LLM_REQUIRED_MODELS", "[]"))
if not isinstance(REQUIRED_MODELS, list) or any(not isinstance(m, str) or not m.strip() for m in REQUIRED_MODELS):
    raise ValueError("LLM_REQUIRED_MODELS must be a JSON array of model IDs")


@dataclass(frozen=True)
class Selection:
    base_url: str
    models: dict
    agent_model: str
    profile_id: str = "deployment"
    timeout_seconds: float = 300


selection = Selection(os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"), {}, LIGHTNING_MODEL)
selection_lock = asyncio.Lock()
profile_lock = asyncio.Lock()
TRACING = setup_tracing("course-llm-client")


def client():
    return httpx.AsyncClient(timeout=httpx.Timeout(max(selection.timeout_seconds,
        float(os.getenv("LLM_CLIENT_TIMEOUT_SECONDS", "300"))), connect=15), follow_redirects=False, trust_env=False)


def selected_headers(current):
    profile_attribute(current.profile_id)
    return {"x-dli-base-url": current.base_url, "x-dli-profile": current.profile_id, **trace_headers()}


async def refresh_profile():
    global selection
    async with profile_lock:
        try:
            async with client() as connection:
                response = await connection.get(WRAPPER_URL + "/v1/profile",
                    params={"known": selection.profile_id}, timeout=5)
            if response.status_code == 404:
                return  # Compatibility with an existing wrapper during a rolling update.
            response.raise_for_status()
            profile = response.json()
            if profile["revision"] == selection.profile_id:
                return
            catalog = profile["models"]
            if not isinstance(catalog, dict) or not catalog:
                raise ValueError("Empty profile catalog")
            async with selection_lock:
                agent = selection.agent_model if selection.agent_model in catalog else (
                    LIGHTNING_MODEL if LIGHTNING_MODEL in catalog else next(iter(catalog)))
                limits = profile["settings"]["requests"]
                selection = Selection(profile["base_url"], catalog, agent, profile["revision"],
                    limits["timeout_seconds"] + 2 * limits["max_ttft_seconds"] + 30)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise HTTPException(503, "Active profile is unavailable") from None


async def select_url(url):
    global selection
    if not isinstance(url, str):
        raise HTTPException(400, "base_url must be a URL")
    parsed = urlsplit(url)
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise HTTPException(400, "Invalid base_url")
    url = url.rstrip("/")
    async with selection_lock:
        try:
            async with client() as connection:
                response = await connection.get(WRAPPER_URL + "/v1/models",
                    headers={**selected_headers(selection), "x-dli-base-url": url})
            if response.status_code != 200:
                raise HTTPException(503, "Selected provider is unavailable or unconfigured")
            catalog = {item["id"]: item for item in response.json()["data"] if isinstance(item.get("id"), str)}
            if not catalog:
                raise HTTPException(503, "Selected provider has no available models")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, "Selected provider discovery failed") from None
        selection = Selection(url, catalog, selection.agent_model, selection.profile_id, selection.timeout_seconds)
    return {"base_url": url, "models": len(catalog)}


@asynccontextmanager
async def lifespan(app):
    async def initialize():
        while not selection.models:
            try:
                await refresh_profile()
                if selection.models:
                    return
                if os.getenv("LLM_USE_WRAPPER_DEFAULT", "false").lower() in {"true", "1"}:
                    async with client() as connection:
                        response = await connection.get(WRAPPER_URL + "/v1/default-base-url")
                    if response.status_code != 200:
                        raise HTTPException(503, "Deployment default is unavailable")
                    await select_url(response.json()["base_url"])
                    continue
                await select_url(selection.base_url)
            except (HTTPException, httpx.HTTPError, ValueError, KeyError, TypeError):
                diagnostic("failure", 503)
                await asyncio.sleep(2)
    async def initialize_retrieval():
        try:
            await asyncio.to_thread(local_retrieval.load_local_models)
        except Exception:
            diagnostic("failure", 503)
    retrieval = asyncio.create_task(initialize_retrieval()) if LOCAL_RETRIEVAL else None
    initialization = asyncio.create_task(initialize())
    try:
        yield
    finally:
        if retrieval is not None:
            retrieval.cancel()
            with suppress(asyncio.CancelledError):
                await retrieval
        initialization.cancel()
        with suppress(asyncio.CancelledError):
            await initialization
        await asyncio.to_thread(TRACING.shutdown)


app = FastAPI(title="Course LLM client", lifespan=lifespan, docs_url=None)
app.add_middleware(RequestTracing, provider=TRACING)


@app.get("/docs", include_in_schema=False)
async def api_reference():
    return get_swagger_ui_html(openapi_url="openapi.json", title="Course LLM client - API reference")


@app.exception_handler(Exception)
async def unexpected(request, exc):
    return JSONResponse({"error": {"message": "LLM request failed"}}, status_code=502)


@app.get("/health")
async def health():
    await refresh_profile()
    current = selection
    missing = sorted(set(REQUIRED_MODELS) - current.models.keys())
    local_ready = not LOCAL_RETRIEVAL or local_retrieval.ready()
    return JSONResponse({"backends": int(bool(current.models)), "models": len(current.models),
        "missing_required_models": missing, "local_retrieval_ready": local_ready,
        "course_agent_model": current.agent_model,
        "course_agent_ready": current.agent_model in current.models,
        "course_agent_models": [m for m in COURSE_AGENT_MODELS if m in current.models]},
        status_code=200 if current.models and not missing and local_ready else 503)


@app.get("/v1/base-url")
async def get_base_url():
    await refresh_profile()
    return {"base_url": selection.base_url}


@app.get("/status", response_class=HTMLResponse)
async def status_page():
    return HTMLResponse(Path(__file__).with_name("proxy_status.html").read_text())


@app.get("/v1/activity")
async def activity_status(window: int = Query(3600, ge=60, le=86400),
                          limit: int = Query(100, ge=1, le=200), model: str = Query("", max_length=256)):
    await refresh_profile()
    current = selection
    try:
        async with client() as connection:
            response = await connection.get(WRAPPER_URL + "/v1/activity",
                params={"window": window, "limit": limit, "model": model},
                headers=selected_headers(current))
        if response.status_code != 200:
            raise HTTPException(502, "Activity history is unavailable")
        return response.json()
    except httpx.HTTPError:
        raise HTTPException(502, "Activity history is unavailable") from None


@app.get("/v1/routing")
async def routing_status(model: str = COURSE_AGENT_MODEL):
    await refresh_profile()
    current = selection
    model = current.agent_model if model == COURSE_AGENT_MODEL else model
    if model not in current.models:
        raise HTTPException(400, "Model is not available at the selected provider")
    async with client() as connection:
        response = await connection.get(WRAPPER_URL + "/v1/routing", params={"model": model},
            headers=selected_headers(current))
        if response.status_code != 200:
            raise HTTPException(502, "Routing status is unavailable")
        return response.json()


async def json_object(request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be a JSON object") from None
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")
    return body


@app.put("/v1/base-url")
async def set_base_url(request: Request):
    await refresh_profile()
    body = await json_object(request)
    if set(body) != {"base_url"}:
        raise HTTPException(400, "Supply only base_url")
    return await select_url(body["base_url"])


@app.get("/v1/models")
async def list_models(scope: str = "course"):
    await refresh_profile()
    current = selection
    if scope == "all":
        models = list(current.models.values())
        if LOCAL_RETRIEVAL:
            models.extend(local_retrieval.LOCAL_MODELS.values())
    elif scope == "course":
        models = []
        if current.agent_model in current.models and os.getenv("LLM_AGENT_ALIAS_ENABLED", "true").lower() in {"true", "1"}:
            models.append({"id": COURSE_AGENT_MODEL, "object": "model", "owned_by": "course-runtime", "selected_model": current.agent_model})
        models.extend(current.models[m] for m in dict.fromkeys((*COURSE_AGENT_MODELS, current.agent_model)) if m in current.models)
    else:
        raise HTTPException(400, "scope must be 'course' or 'all'")
    return {"object": "list", "data": models}


@app.get("/v1/course/agent-models")
async def course_models():
    await refresh_profile()
    current = selection
    return {"active": current.agent_model, "default": LIGHTNING_MODEL,
        "data": [{"id": m, "available": m in current.models} for m in dict.fromkeys((*COURSE_AGENT_MODELS, current.agent_model))]}


@app.post("/v1/course/agent-models")
async def set_agent_model(request: Request):
    global selection
    await refresh_profile()
    body = await json_object(request)
    model = body.get("model")
    async with selection_lock:
        if not isinstance(model, str) or model not in selection.models:
            raise HTTPException(400, "Model is not available at the selected provider")
        selection = Selection(selection.base_url, selection.models, model, selection.profile_id, selection.timeout_seconds)
    return {"active": model}


@app.get("/v1/models/{model:path}")
async def get_model(model: str):
    if LOCAL_RETRIEVAL and model in local_retrieval.LOCAL_MODELS:
        return local_retrieval.LOCAL_MODELS[model]
    await refresh_profile()
    current = selection
    resolved = current.agent_model if model == COURSE_AGENT_MODEL else model
    if resolved not in current.models:
        raise HTTPException(404, "Model is not available at the selected provider")
    result = dict(current.models[resolved])
    if model == COURSE_AGENT_MODEL:
        result.update(id=model, selected_model=resolved)
    return result


def text_completion(data, streaming=False):
    if "error" in data:
        return data
    data["object"] = "text_completion"
    for choice in data.get("choices", []):
        choice["text"] = (choice.pop("delta" if streaming else "message", None) or {}).get("content") or ""
    return data


@app.post("/v1/{path:path}")
async def proxy(request: Request, path: str):
    if LOCAL_RETRIEVAL and path in {"embeddings", "ranking"}:
        body = await json_object(request)
        expected = local_retrieval.LOCAL_EMBED_MODEL if path == "embeddings" else local_retrieval.LOCAL_RERANK_MODEL
        if body.get("model") == expected:
            result = await getattr(local_retrieval, path)(request)
            return JSONResponse(result, headers={"X-DLI-Route": "local", "X-DLI-Cache": "BYPASS", "X-DLI-Inference-Executed": "true"})
    if path not in {"chat/completions", "completions", "embeddings"}:
        raise HTTPException(404, "Unsupported endpoint")
    await refresh_profile()
    body = await json_object(request)
    current = selection
    model = body.get("model")
    model = current.agent_model if model == COURSE_AGENT_MODEL else model
    if not isinstance(model, str) or model not in current.models:
        raise HTTPException(400, "Model is not available at the selected provider")
    body["model"] = model
    shim = path == "completions"
    if shim:
        prompt = body.pop("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(str(p) for p in prompt)
        body["messages"] = [{"role": "user", "content": prompt}]
        path = "chat/completions"
    if body.get("stop") == []:
        body.pop("stop")
    if path == "chat/completions":
        try:
            body = facade_chat_request(body, LIGHTNING_MODEL)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
    connection = client()
    try:
        response = await connection.send(connection.build_request("POST", WRAPPER_URL + "/v1/" + path,
            json=body, headers={**selected_headers(current), "cache-control": request.headers.get("cache-control", "")}), stream=True)
        if response.status_code != 200:
            data = json.loads(await response.aread())
            await response.aclose()
            await connection.aclose()
            return JSONResponse(data, status_code=response.status_code, headers=proxy_headers(response))
        if not body.get("stream"):
            data = json.loads(await response.aread())
            await response.aclose()
            await connection.aclose()
            return JSONResponse(text_completion(data) if shim else data, headers=proxy_headers(response))
    except BaseException:
        await connection.aclose()
        raise

    async def events():
        try:
            if not shim:
                async for chunk in response.aiter_bytes():
                    yield chunk
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    value = line[5:].strip()
                    if value == "[DONE]":
                        yield "data: [DONE]\n\n"
                    else:
                        yield "data: " + json.dumps(text_completion(json.loads(value), True)) + "\n\n"
        except Exception:
            yield 'data: {"error":{"message":"Provider stream interrupted"}}\n\n'
        finally:
            with anyio.CancelScope(shield=True):
                await response.aclose()
                await connection.aclose()
    return StreamingResponse(events(), media_type="text/event-stream", headers=proxy_headers(response))


def proxy_headers(response):
    headers = {}
    for name in ("X-DLI-Cache", "X-DLI-Inference-Executed", "X-DLI-Fallback", "X-DLI-Route", "X-DLI-Circuit", "Retry-After"):
        if name in response.headers:
            headers[name] = response.headers[name]
    return headers


@app.post("/rediscover")
async def rediscover():
    await refresh_profile()
    result = await select_url(selection.base_url)
    return {"backends": 1, "models": result["models"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000, access_log=False, log_config=None)
