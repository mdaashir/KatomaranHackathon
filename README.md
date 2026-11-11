# Intelligent Face Tracker

Real-time face detection, recognition, tracking, auto-registration and unique
visitor counting from a video file or RTSP stream.

> Status: **Phase 4 — recognition foundations.** Configuration, storage,
> logging, video inputs, YOLO face detection, and InsightFace/ArcFace
> embedding are implemented. Tracking, event logging, and the pipeline are
> next. See the build phases in [`docs/PLAN.md`](docs/PLAN.md).

## Documentation

- [Plan](docs/PLAN.md)
- [Features](docs/FEATURES.md)
- [Compute budget](docs/COMPUTE.md)
- [Architecture and diagrams](docs/ARCHITECTURE.md)
- [Prompts used](docs/PROMPTS.md)

## GPU setup (Windows, recommended)

Requires an NVIDIA GPU with a recent driver. This installs the CUDA 12.8
PyTorch build, ONNX Runtime GPU 1.22, and the InsightFace models:

```powershell
uv python install 3.11
uv venv --python 3.11
.\scripts\setup_gpu.ps1
```

Verify:

```powershell
.venv\Scripts\python.exe -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())"
```

## CPU setup

```powershell
uv python install 3.11
uv venv --python 3.11
uv pip install -r requirements.txt
.venv\Scripts\python.exe scripts\download_models.py
```

Set `detector.device` and `recognition.device` to `cpu` in `config.json` to
force CPU execution; `auto` prefers CUDA when it is available.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

## Models

- **No model weights are committed.** All `*.pt`, `*.onnx` and `*.zip` files
  under `models/` are ignored by git.
- `scripts/download_models.py` fetches and SHA256-verifies the 6.2 MB YOLO
  checkpoint plus the 174 MB InsightFace recognizer and 5 MB landmark model.
- See [`models/README.md`](models/README.md) and
  [`models/insightface/README.md`](models/insightface/README.md).

## Configuration

Copy the shape of [`config.json`](config.json). Every tunable is loaded once
into a typed config object and injected into the pipeline.

## Assumptions and known limitations

- The local dataset in `src/video/` is 22 one-minute clips of a mall front
  entry, plus one 8-second 4K crowded clip used as a stress test.
- A face-only detector cannot count visitors who never face the camera. Such
  visitors are not registered and do not affect the unique count. An optional
  person-detector fallback is described in `docs/PLAN.md`.
- Face crops are biometric data and stay local. Nothing is uploaded.

This project is a part of a hackathon run by https://katomaran.com
