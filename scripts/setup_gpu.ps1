$ErrorActionPreference = "Stop"

# GPU setup for Windows: uv owns the project environment; the CUDA wheels and
# ONNX Runtime GPU are layered on top of the resolved project groups.
uv sync --extra gpu --no-dev

# Replace the default Torch wheel with the CUDA 12.8 build.
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128

# The gpu extra provides onnxruntime-gpu; force the pinned CUDA 12.x build.
uv pip install --reinstall "onnxruntime-gpu==1.22.0"

# Fetch and verify the InsightFace models.
uv run --no-sync python scripts\download_models.py

Write-Host "GPU setup complete. Verify with:"
Write-Host "  uv run --no-sync python -c `"import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())`""
