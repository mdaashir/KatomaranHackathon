#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p reports/e2e

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  PYTHON_CMD=(uv run --no-sync python)
fi

COMPOSE=(docker compose -f docker-compose.yml)
RTSP_COMPOSE=(
  docker compose
  -f docker-compose.yml
  -f tests/e2e/docker-compose.rtsp.yml
  --profile rtsp-test
)

cleanup() {
  "${RTSP_COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p data logs
# The app runs as UID 10001; use a root helper to remove its bind-mount files
# safely before a fresh run.
docker run --rm \
  -v "$ROOT/data:/data" \
  -v "$ROOT/logs:/logs" \
  busybox:1.36 \
  sh -c "rm -rf /data/e2e_file.db* /data/e2e_rtsp.db* /logs/e2e_file /logs/e2e_rtsp"

# File path: run the same fixture twice so the second visit must re-identify
# the Face ID without incrementing the unique count.
for _ in 1 2; do
  "${COMPOSE[@]}" run --rm \
    -v "$ROOT/tests:/app/tests:ro" \
    app python main.py \
    --config tests/e2e/config.file.json \
    --start-time 2025-06-20T18:39:03Z
done

"${PYTHON_CMD[@]}" tests/e2e/verify_outputs.py \
  --db data/e2e_file.db \
  --log-dir logs/e2e_file \
  --expected tests/e2e/expected_file.json \
  | tee reports/e2e/file.json

# RTSP path: MediaMTX + an ffmpeg publisher looping the fixture. The pipeline
# stops after 150 frames, which must flush active tracks as exits.
"${RTSP_COMPOSE[@]}" up -d mediamtx rtsp-publisher
sleep 8

"${COMPOSE[@]}" run --rm \
  -v "$ROOT/tests:/app/tests:ro" \
  app python main.py \
  --config tests/e2e/config.rtsp.json \
  --max-frames 300

"${PYTHON_CMD[@]}" tests/e2e/verify_outputs.py \
  --db data/e2e_rtsp.db \
  --log-dir logs/e2e_rtsp \
  --expected tests/e2e/expected_rtsp.json \
  | tee reports/e2e/rtsp.json
