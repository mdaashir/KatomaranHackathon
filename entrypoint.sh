#!/usr/bin/env sh
set -eu

# Model weights are never baked into the image. The named volume is checked
# first; the downloader is idempotent and verifies pinned SHA256 values.
if [ ! -f models/yolov8n-face.pt ] || [ ! -f models/insightface/models/buffalo_l/w600k_r50.onnx ]; then
    python scripts/download_models.py
fi

exec "$@"
