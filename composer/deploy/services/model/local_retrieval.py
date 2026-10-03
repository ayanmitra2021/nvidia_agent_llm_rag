"""Pinned, offline CPU retrieval models behind the existing NVIDIA API."""
import asyncio
import os
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder

READY = False
LOCAL_EMBEDDER = None
LOCAL_RERANKER = None
LOCAL_EMBED_MODEL = "course/embedding"
LOCAL_EMBED_IMPLEMENTATION = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_RERANK_MODEL = "course/reranker"
LOCAL_RERANK_IMPLEMENTATION = "cross-encoder/ms-marco-MiniLM-L6-v2"
LOCAL_RERANK_PATH = "/opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2"
LOCAL_MODELS = {
    LOCAL_EMBED_MODEL: {
        "id": LOCAL_EMBED_MODEL,
        "object": "model",
        "owned_by": "course-runtime",
        "model_type": "embedding",
        "implementation": LOCAL_EMBED_IMPLEMENTATION,
    },
    LOCAL_RERANK_MODEL: {
        "id": LOCAL_RERANK_MODEL,
        "object": "model",
        "owned_by": "course-runtime",
        "model_type": "ranking",
        "implementation": LOCAL_RERANK_IMPLEMENTATION,
    },
}

def load_local_models():
    """Load the pinned retrieval models once for the life of the service."""
    global LOCAL_EMBEDDER, LOCAL_RERANKER, READY

    LOCAL_EMBEDDER = TextEmbedding(
        model_name=LOCAL_EMBED_IMPLEMENTATION,
        cache_dir="/opt/models/fastembed",
        local_files_only=True,
        threads=2,
    )
    if not any(
        model["model"] == LOCAL_RERANK_IMPLEMENTATION
        for model in TextCrossEncoder.list_supported_models()
    ):
        TextCrossEncoder.add_custom_model(
            model=LOCAL_RERANK_IMPLEMENTATION,
            sources=ModelSource(hf=LOCAL_RERANK_IMPLEMENTATION),
            model_file="onnx/model.onnx",
            description="MS MARCO MiniLM cross-encoder reranker",
            license="apache-2.0",
            size_in_gb=0.091,
        )
    LOCAL_RERANKER = TextCrossEncoder(
        model_name=LOCAL_RERANK_IMPLEMENTATION,
        specific_model_path=LOCAL_RERANK_PATH,
        local_files_only=True,
        threads=2,
    )

    # Force both ONNX sessions to execute before the service reports healthy.
    list(LOCAL_EMBEDDER.embed(["model service ready"]))
    list(LOCAL_RERANKER.rerank("model service ready", ["model service ready"]))
    READY = True


def ready():
    return READY

async def embeddings(request: Request):
    """Serve the pinned CPU embedder or forward another discovered model."""
    if not ready():
        raise HTTPException(503, "Local retrieval models are still loading")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")

    inputs = body.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or not inputs or not all(
        isinstance(value, str) and value for value in inputs
    ):
        raise HTTPException(400, "input must be a non-empty string or list of strings")
    if body.get("encoding_format", "float") != "float":
        raise HTTPException(400, "only float embeddings are supported")
    if body.get("dimensions") not in (None, 384):
        raise HTTPException(400, "this embedding model returns 384 dimensions")

    input_type = body.get("input_type")
    if input_type == "query":
        embed = LOCAL_EMBEDDER.query_embed
    elif input_type == "passage":
        embed = LOCAL_EMBEDDER.passage_embed
    else:
        embed = LOCAL_EMBEDDER.embed

    vectors = await asyncio.to_thread(lambda: list(embed(inputs)))
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector.tolist()}
            for index, vector in enumerate(vectors)
        ],
        "model": LOCAL_EMBED_MODEL,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

async def ranking(request: Request):
    """Serve the pinned CPU cross-encoder or forward another discovered model."""
    if not ready():
        raise HTTPException(503, "Local retrieval models are still loading")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")

    query_body = body.get("query")
    query = query_body.get("text") if isinstance(query_body, dict) else None
    passages = body.get("passages")
    if not isinstance(query, str) or not query:
        raise HTTPException(400, "query.text must be a non-empty string")
    if not isinstance(passages, list) or not passages or len(passages) > 64:
        raise HTTPException(400, "passages must contain between 1 and 64 entries")

    texts = [passage.get("text") for passage in passages if isinstance(passage, dict)]
    if len(texts) != len(passages) or not all(
        isinstance(text, str) and text for text in texts
    ):
        raise HTTPException(400, "each passage must contain non-empty text")

    scores = await asyncio.to_thread(
        lambda: list(LOCAL_RERANKER.rerank(query, texts))
    )
    rankings = sorted(
        ({"index": index, "logit": float(score)} for index, score in enumerate(scores)),
        key=lambda result: result["logit"],
        reverse=True,
    )
    return {"rankings": rankings}
