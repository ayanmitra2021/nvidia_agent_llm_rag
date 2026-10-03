#!/usr/bin/env bash
set -Eeuo pipefail

phase="startup"
trap 'status=$?; echo "Canary failed: phase=${phase} line=${LINENO} exit_code=${status}" >&2' ERR

CANARY_NOTEBOOK_ROOT="${CANARY_NOTEBOOK_ROOT:-/dli/task}"
CANARY_LLM_URL="${CANARY_LLM_URL:-http://llm_client:9000}"
CANARY_DOCKER_ROUTER_URL="${CANARY_DOCKER_ROUTER_URL:-http://docker_router:8070}"
CANARY_FRONTEND_URL="${CANARY_FRONTEND_URL:-http://frontend:8090}"
CANARY_CHATBOT_URL="${CANARY_CHATBOT_URL:-http://chatbot:8990}"

wait_for_http_200() {
    local name="$1"
    local url="$2"
    local response="/tmp/cfx15-canary-${name}.json"
    local code
    local attempt

    for attempt in $(seq 1 30); do
        if code="$(curl -sS --max-time 10 -o "$response" -w '%{http_code}' "$url")" && [ "$code" = "200" ]; then
            echo "Canary ready: service=${name} attempt=${attempt}"
            return 0
        fi
        if [ "$attempt" -eq 1 ] || [ $((attempt % 10)) -eq 0 ]; then
            echo "Canary waiting: service=${name} attempt=${attempt} http=${code:-unavailable}"
        fi
        sleep 2
    done

    echo "Canary timeout: service=${name} last_response=$(head -c 500 "$response" 2>/dev/null || true)" >&2
    return 1
}

phase="localized notebook projection"
workdir="$(mktemp -d /tmp/cfx15-canary-notebooks.XXXXXX)"
cp -a "$CANARY_NOTEBOOK_ROOT"/. "$workdir/"

WORKDIR="$workdir" python3 - <<'PY'
import json
import os
from pathlib import Path

notebook_path = Path(os.environ["WORKDIR"]) / "00_jupyterlab.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
switch_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "course-language-switch")
(Path(os.environ["WORKDIR"]) / ".preserved-source").write_text(
    "".join(switch_cell["source"]), encoding="utf-8"
)
for cell in notebook["cells"]:
    cell.pop("id", None)
notebook["nbformat_minor"] = 4
notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
PY

for language in en tw zh ja en; do
    (cd "$workdir" && python3 composer/switch_language.py "$language")
done

WORKDIR="$workdir" python3 - <<'PY'
import json
import os
import re
from pathlib import Path

root = Path(os.environ["WORKDIR"])
graph = json.loads((root / "composer/translation_web.json").read_text(encoding="utf-8"))
assert graph["languages"] == ["en", "tw", "zh", "ja"]
assert set(graph["notebooks"]) == {path.name for path in root.glob("*.ipynb")}
for relative, entry in graph["notebooks"].items():
    notebook = json.loads((root / relative).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5, relative
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(ids), f"{relative}: missing cell ID"
    assert len(ids) == len(set(ids)), f"{relative}: duplicate cell ID"
    assert all(re.fullmatch(r"[A-Za-z0-9-_]+", value) for value in ids), f"{relative}: invalid cell ID"
    expected = [cell["id"] for cell in entry["cells"] if "en" not in cell.get("exclude_languages", [])]
    assert ids == expected, f"{relative}: graph projection differs"

first = json.loads((root / "00_jupyterlab.ipynb").read_text(encoding="utf-8"))
switch_cell = next(cell for cell in first["cells"] if cell.get("id") == "course-language-switch")
assert "".join(switch_cell["source"]) == (root / ".preserved-source").read_text(encoding="utf-8")
print("Canary passed: locale graph and preserved cell")
PY

phase="course model catalog"
wait_for_http_200 llm-client "$CANARY_LLM_URL/health"
curl -fsS --max-time 10 "$CANARY_LLM_URL/v1/models?scope=course" -o /tmp/cfx15-canary-models.json
python3 - <<'PY'
import json

models = json.load(open("/tmp/cfx15-canary-models.json", encoding="utf-8"))["data"]
actual = [model["id"] for model in models]
expected = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "meta/llama-3.2-11b-vision-instruct",
]
assert actual == expected, (actual, expected)
print("Canary passed: course model catalog")
PY

phase="local embedding"
curl -fsS --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"model":"course/embedding","input":["retrieval augmented generation","unrelated weather report"]}' \
    "$CANARY_LLM_URL/v1/embeddings" \
    -o /tmp/cfx15-canary-embedding.json
python3 - <<'PY'
import json

result = json.load(open("/tmp/cfx15-canary-embedding.json", encoding="utf-8"))
assert len(result["data"]) == 2
assert all(len(item["embedding"]) == 384 for item in result["data"])
print("Canary passed: local embedding")
PY

phase="local reranking"
curl -fsS --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"model":"course/reranker","query":{"text":"retrieval augmented generation"},"passages":[{"text":"RAG retrieves context before generation."},{"text":"A recipe for apple pie."}]}' \
    "$CANARY_LLM_URL/v1/ranking" \
    -o /tmp/cfx15-canary-ranking.json
python3 - <<'PY'
import json

result = json.load(open("/tmp/cfx15-canary-ranking.json", encoding="utf-8"))
assert [item["index"] for item in result["rankings"]] == [0, 1], result
print("Canary passed: local reranking")
PY

phase="non-reasoning text model"
curl -fsS --max-time 180 \
    -H 'Content-Type: application/json' \
    -d '{"model":"nvidia/nemotron-3.5-lightning-30b-a3b","messages":[{"role":"user","content":"Reply with CANARY_OK."}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}' \
    "$CANARY_LLM_URL/v1/chat/completions" \
    -o /tmp/cfx15-canary-text-chat.json
python3 - <<'PY'
import json

result = json.load(open("/tmp/cfx15-canary-text-chat.json", encoding="utf-8"))
content = result["choices"][0]["message"]["content"]
assert isinstance(content, str) and content.strip(), result
assert "<think>" not in content and "</think>" not in content, content
print("Canary passed: non-reasoning text model")
PY

phase="reasoning text model"
curl -fsS --max-time 180 \
    -H 'Content-Type: application/json' \
    -d '{"model":"nvidia/nemotron-3.5-lightning-30b-a3b","messages":[{"role":"user","content":"Tell me about yourself in exactly two sentences."}],"temperature":0,"max_tokens":2048,"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget":512}' \
    "$CANARY_LLM_URL/v1/chat/completions" \
    -o /tmp/cfx15-canary-reasoning-chat.json
python3 - <<'PY'
import json

result = json.load(open("/tmp/cfx15-canary-reasoning-chat.json", encoding="utf-8"))
message = result["choices"][0]["message"]
assert isinstance(message.get("reasoning_content"), str) and message["reasoning_content"].strip(), result
assert isinstance(message.get("content"), str) and message["content"].strip(), result
assert result["choices"][0].get("finish_reason") == "stop", result
print("Canary passed: reasoning text model")
PY

phase="vision-capable chat model"
curl -fsS --max-time 180 \
    -H 'Content-Type: application/json' \
    -d '{"model":"meta/llama-3.2-11b-vision-instruct","messages":[{"role":"user","content":"Reply with CANARY_OK."}],"temperature":0,"max_tokens":128}' \
    "$CANARY_LLM_URL/v1/chat/completions" \
    -o /tmp/cfx15-canary-vision-chat.json
python3 - <<'PY'
import json

result = json.load(open("/tmp/cfx15-canary-vision-chat.json", encoding="utf-8"))
content = result["choices"][0]["message"]["content"]
assert isinstance(content, str) and content.strip(), result
print("Canary passed: vision-capable chat model")
PY

phase="course interfaces"
wait_for_http_200 docker-router "$CANARY_DOCKER_ROUTER_URL/healthy"
wait_for_http_200 assessment-frontend "$CANARY_FRONTEND_URL/config"
wait_for_http_200 content-chatbot "$CANARY_CHATBOT_URL/config"
python3 - <<'PY'
import json

frontend = json.load(open("/tmp/cfx15-canary-assessment-frontend.json", encoding="utf-8"))
api_names = {item.get("api_name") for item in frontend.get("dependencies", [])}
assert {"bot", "rag_eval"}.issubset(api_names), api_names
chatbot = json.load(open("/tmp/cfx15-canary-content-chatbot.json", encoding="utf-8"))
assert chatbot.get("components") and chatbot.get("dependencies")
print("Canary passed: Docker Router, assessment frontend, and content chatbot")
PY

echo "Canary smoke test passed: c-fx-15-v1"
