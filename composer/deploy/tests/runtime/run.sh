#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
name="cfx15-runtime-${CI_JOB_ID:-local-$$}"
cleanup() {
    docker rm -f "$name-compose" "$name-check" "$name-facade" "$name-wrapper" "$name-provider" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM
# Parse both launch configurations with the platform's actual legacy Compose.
# Copy only public YAML; no repository or host credential file enters the fixture.
docker run -d --name "$name-compose" -e DEV_NGINX_PORT=80 -e DEV_ASSESSMENT_PORT=8080 \
    --entrypoint /bin/sh docker/compose:1.23.2 -c 'mkdir -p /task /dli/secrets; touch /task/.env-dev /dli/secrets/.env; sleep 3600' >/dev/null
for file in docker-compose.yml docker-compose.override.yml docker-compose.deployment.yml; do
    docker cp "$file" "$name-compose:/task/$file"
done
# Real build-context files are checked by test_rag_runtime; this fixture parses YAML only.
find . -type d -print | docker exec -i -w /task "$name-compose" sh -c 'while IFS= read -r directory; do mkdir -p "$directory"; done'
docker exec -w /task "$name-compose" docker-compose -f docker-compose.yml -f docker-compose.override.yml config --services
docker exec -w /task "$name-compose" docker-compose -f docker-compose.yml -f docker-compose.deployment.yml config --services
# Build the exact production recipes; credentials never enter either build.
docker build -t "$name-facade" -f services/model/llm_client.Dockerfile services/model
docker build -t "$name-wrapper" -f services/model/litellm_wrapper.Dockerfile services/model
docker run -d --name "$name-provider" --entrypoint python "$name-facade" -c 'import time; time.sleep(3600)' >/dev/null
docker cp tests/runtime/provider_fixture.py "$name-provider:/app/provider_fixture.py"
docker exec -d "$name-provider" python /app/provider_fixture.py
docker run -d --name "$name-wrapper" --network "container:$name-provider" \
    -e TEST_PRIMARY=synthetic-primary -e TEST_FALLBACK=synthetic-fallback \
    -e TEST_FALLBACK_BASE=http://127.0.0.1:9102/v1 \
    -e 'DLI_LLM_PROVIDERS=[{"base_url":"http://127.0.0.1:9101/v1","api_key_env":"TEST_PRIMARY"}]' \
    -e 'DLI_LLM_FALLBACK_PROVIDER={"base_url_env":"TEST_FALLBACK_BASE","api_key_env":"TEST_FALLBACK"}' \
    -e LLM_RETRY_ATTEMPTS=2 -e LLM_CACHE_PACING=off "$name-wrapper" >/dev/null
docker run -d --name "$name-facade" --network "container:$name-provider" \
    -e LLM_WRAPPER_URL=http://127.0.0.1:9001 -e LLM_USE_WRAPPER_DEFAULT=true \
    -e LLM_LOCAL_RETRIEVAL=true \
    -e 'LLM_REQUIRED_MODELS=["nvidia/nemotron-3.5-lightning-30b-a3b"]' "$name-facade" >/dev/null
docker build -t "$name-check" --build-arg FACADE_IMAGE="$name-facade" --build-arg WRAPPER_IMAGE="$name-wrapper" tests/runtime
docker run --name "$name-check" --network "container:$name-provider" "$name-check"
