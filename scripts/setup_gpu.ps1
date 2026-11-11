$ErrorActionPreference = "Stop"

# Phase 4 GPU setup for Windows.
# 1. Replace the CPU Torch wheel with the CUDA 12.8 build.
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install the project dependencies (may pull the CPU onnxruntime wheel).
uv pip install -r requirements.txt

# 3. Remove the CPU ONNX Runtime so the GPU package provides the module.
uv pip uninstall onnxruntime

# 4. Pin ONNX Runtime GPU 1.22, which targets CUDA 12.x.
uv pip install --reinstall "onnxruntime-gpu==1.22.0"

# 5. Fetch and verify the InsightFace models.
.venv\Scripts\python.exe scripts\download_models.py

Write-Host "GPU setup complete. Verify with:"
Write-Host "  .venv\Scripts\python.exe -c `"import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())`""
