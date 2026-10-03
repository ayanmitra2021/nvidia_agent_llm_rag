#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
name="course-entry-${CI_JOB_ID:-local-$$}"
course=notebooks
workspace=/dli/task
context=notebooks
recipe=notebooks/composer/Dockerfile
if [ -d dli ]; then
    course=dli
    workspace=/workspace/dli
    context=.
    recipe=Dockerfile
fi
mkdir -p tests/entry/evidence
cleanup() {
    docker cp "$name-browser:/evidence/." tests/entry/evidence/ >/dev/null 2>&1 || true
    docker rm -f "$name-browser" "$name-lab" >/dev/null 2>&1 || true
    docker network rm "$name" >/dev/null 2>&1 || true
    docker image rm "$name-lab" "$name-browser" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM
docker info >/dev/null || { echo 'Course entry test requires a reachable Docker daemon' >&2; exit 1; }
# Build the shipped recipe and use its entrypoint, with no preview-only config.
docker build -t "$name-lab" -f "$recipe" "$context"
docker build -t "$name-browser" - <<'DOCKERFILE'
FROM mcr.microsoft.com/playwright:v1.62.0-noble
WORKDIR /test
RUN npm install --no-audit --no-fund playwright@1.62.0 && mkdir /evidence
CMD ["node", "browser.cjs"]
DOCKERFILE
# Neither the notebook nor the browser may contact inference providers.
docker network create --internal "$name" >/dev/null
docker create --name "$name-lab" --network "$name" --network-alias course \
    -e JUPYTER_TOKEN=foyer-browser-fixture "$name-lab" >/dev/null
docker cp "$course/." "$name-lab:$workspace/"
docker start "$name-lab" >/dev/null
docker create --name "$name-browser" --network "$name" "$name-browser" >/dev/null
docker cp tests/entry/browser.cjs "$name-browser:/test/browser.cjs"
ready=false
for attempt in $(seq 1 90); do
    if docker exec "$name-lab" python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8888/lab/login", timeout=2)' >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done
[ "$ready" = true ] || { echo 'Course entry did not become ready'; exit 1; }
docker start -a "$name-browser"
code=$(docker inspect --format '{{.State.ExitCode}}' "$name-browser")
exit "$code"
