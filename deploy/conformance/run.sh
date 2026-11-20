#!/usr/bin/env bash
# Build arc-serve natively, start it, run conformance.py (routes + realtime + parity with arc_audio), stop it.
# Then the same against the Docker image when a Docker daemon is already running (it never starts Docker Desktop).
#
#   deploy/conformance/run.sh MINI_DIR [--python PY] [--skip-reference] [--combined ARCTAG_REPO TAGGER_MODEL_DIR] [--no-docker]
#
# MINI_DIR: the Arc Mini folder (WhissleAI/arc-v1.1 mini/). PY: an interpreter with arc_audio's dependencies (torch,
# transformers, ...) for the reference run. Builds run under nice -n 19 with make -j2.
# Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MINI="${1:?usage: run.sh MINI_DIR [--python PY] [--skip-reference] [--combined ARCTAG_REPO TAGGER_MODEL_DIR] [--no-docker]}"
MINI="$(cd "$MINI" && pwd)"
shift
PY=python3
REF=()
ARCTAG=""
TAGMODEL=""
DOCKER=1
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2;;
    --skip-reference) REF=(--skip-reference); shift;;
    --combined) ARCTAG="$(cd "$2" && pwd)"; TAGMODEL="$(cd "$3" && pwd)"; shift 3;;
    --no-docker) DOCKER=0; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
OUT="${CONFORMANCE_OUT:-$ROOT/deploy/build/conformance}"
mkdir -p "$OUT"
PORT="${PORT:-18$(( RANDOM % 900 + 100 ))}"
URL="http://127.0.0.1:$PORT"
PID=""
cleanup() { [ -n "$PID" ] && kill "$PID" 2>/dev/null || true; [ -n "${CID:-}" ] && docker rm -f "$CID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

wait_ready() {
  for _ in $(seq 1 120); do
    if curl -fs "$1/v1/health/ready" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "server at $1 did not become ready" >&2
  return 1
}

if [ -n "$ARCTAG" ]; then
  BIN="$ROOT/deploy/build/arc-serve-combined"; MK=(ARCTAG="$ARCTAG"); FLAGS=(--combined); NAME=combined
else
  BIN="$ROOT/deploy/build/arc-serve"; MK=(); FLAGS=(); NAME=transcriber
fi
echo "== native build ($NAME)"
nice -n 19 make -C "$ROOT/deploy" -j2 ${MK[@]+"${MK[@]}"} >"$OUT/build-native.log" 2>&1 || { tail -30 "$OUT/build-native.log"; exit 1; }
echo "== native server on $URL"
ARC_MODEL_DIR="$MINI" ARCTAG_MODEL_DIR="$TAGMODEL" ARC_FILE_ROOT="$ROOT/examples" PORT="$PORT" HOST=127.0.0.1 ARC_WORKERS=2 ARC_THREADS=2 ARC_ACCESS_LOG=0 \
  nice -n 19 "$BIN" >"$OUT/server-native.log" 2>&1 &
PID=$!
wait_ready "$URL"
rc=0
nice -n 19 "$PY" "$HERE/conformance.py" --url "$URL" --model-dir "$MINI" --python "$PY" ${REF[@]+"${REF[@]}"} ${FLAGS[@]+"${FLAGS[@]}"} \
  --stage-dir "$ROOT/examples" --report "$OUT/report-native-$NAME.json" || rc=$?
kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
PID=""

if [ "$DOCKER" = 1 ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "== docker build ($NAME)"
  if [ -n "$ARCTAG" ]; then
    CTX="$(cd "$ROOT/.." && pwd)"
    TAG=arc-serve-combined:conformance
    nice -n 19 docker build -q -f "$ROOT/deploy/Dockerfile.combined" --build-arg ARCTAG_DIR="$(basename "$ARCTAG")" \
      --build-arg ARC_DIR="$(basename "$ROOT")" -t "$TAG" "$CTX" >"$OUT/build-docker.log" 2>&1 || { tail -30 "$OUT/build-docker.log"; exit 1; }
    MOUNTS=(-v "$MINI:/opt/ml/model/mini:ro" -v "$TAGMODEL:/opt/ml/model/arc-tagger:ro")
  else
    TAG=arc-serve:conformance
    nice -n 19 docker build -q -f "$ROOT/deploy/Dockerfile" --target runtime -t "$TAG" "$ROOT" >"$OUT/build-docker.log" 2>&1 || { tail -30 "$OUT/build-docker.log"; exit 1; }
    MOUNTS=(-v "$MINI:/opt/ml/model:ro")
  fi
  # SageMaker's contract: `docker run IMAGE serve`, port 8080, model at /opt/ml/model; network isolation: it must start
  # and answer its health check with no network at all
  CID=$(docker run -d --network none ${MOUNTS[@]+"${MOUNTS[@]}"} "$TAG" serve)
  ok=0
  for _ in $(seq 1 60); do
    if docker exec "$CID" /opt/arc/bin/serve --healthcheck >/dev/null 2>&1; then ok=1; break; fi
    sleep 0.5
  done
  docker rm -f "$CID" >/dev/null; CID=""
  [ "$ok" = 1 ] && echo "PASS  docker: starts and is ready with --network none" || { echo "FAIL  docker: not ready with --network none"; rc=1; }
  CID=$(docker run -d -p "127.0.0.1:$PORT:8080" ${MOUNTS[@]+"${MOUNTS[@]}"} -v "$ROOT/examples:/stage:ro" -e ARC_FILE_ROOT=/stage \
    -e ARC_WORKERS=2 -e ARC_THREADS=2 "$TAG" serve)
  wait_ready "$URL"
  nice -n 19 "$PY" "$HERE/conformance.py" --url "$URL" --model-dir "$MINI" --python "$PY" ${REF[@]+"${REF[@]}"} ${FLAGS[@]+"${FLAGS[@]}"} \
    --stage-dir /stage --report "$OUT/report-docker-$NAME.json" || rc=$?
  docker rm -f "$CID" >/dev/null; CID=""
else
  echo "== docker: no running Docker daemon (or --no-docker): the image is UNTESTED in this run"
fi
exit $rc
