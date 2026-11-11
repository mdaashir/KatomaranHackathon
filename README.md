# Intelligent Face Tracker

Real-time face detection, recognition, tracking, auto-registration and unique
visitor counting from a video file or RTSP stream.

> Status: **Phase 1 — foundations (configuration, storage, logging).** The
> detection/recognition pipeline is not wired yet. See the build phases in
> [`docs/PLAN.md`](docs/PLAN.md).

## Documentation

- [Plan](docs/PLAN.md)
- [Features](docs/FEATURES.md)
- [Compute budget](docs/COMPUTE.md)
- [Architecture and diagrams](docs/ARCHITECTURE.md)
- [Prompts used](docs/PROMPTS.md)

## Setup

Python 3.10+ is required; 3.11 is recommended.

```powershell
uv python install 3.11
uv venv --python 3.11
uv pip install -r requirements.txt
```

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

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
