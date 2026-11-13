# Intelligent Face Tracker

[![CI](https://github.com/mdaashir/KatomaranHackathon/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mdaashir/KatomaranHackathon/actions/workflows/ci.yml)
[![Demo](https://img.shields.io/badge/Watch%20the%20demo-YouTube-red?logo=youtube&logoColor=white)](https://youtu.be/bekQvFoOWTY)

Real-time face detection, recognition, tracking, auto-registration and unique
visitor counting for a mall front-entry camera. It accepts a video file, a
folder of clips, an RTSP stream or a local webcam, and logs exactly one entry
and one exit per visit.

**Quick links:** [Problem statement](<docs/Katomarans Problem Statement.pdf>) ·
[Architecture](#architecture) · [Compliance audit](docs/COMPLIANCE.md) ·
[Submission checklist](docs/SUBMISSION.md)

## Demo video

[![Watch the Intelligent Face Tracker walkthrough](https://img.youtube.com/vi/bekQvFoOWTY/hqdefault.jpg)](https://youtu.be/bekQvFoOWTY)

**Watch:** [YouTube](https://youtu.be/bekQvFoOWTY) · **Local copy:** [`demo.mkv`](demo.mkv)

## What it does

- **Detect** faces with a YOLOv8 face model on every `frame_skip` frames.
- **Track** them with ByteTrack; skipped frames reuse the last boxes.
- **Recognize** with InsightFace `buffalo_l` (ArcFace embeddings) against a
  SQLite registry; a new face is auto-registered, a returning face reuses its
  Face ID.
- **Log exactly one entry and one exit** per visit through a per-track state
  machine (`NEW → ACTIVE → LOST → EXITED`), flushing open tracks at end of
  stream or Ctrl+C.
- **Count unique visitors from the database** (`COUNT` of registered faces),
  never from an in-memory counter.
- **Write the audit trail**: `events.log`, dated entry/exit JPEG crops, and
  SQLite rows for every event.

## Architecture

Video/folder/RTSP/webcam input flows through the detector, tracker and ArcFace
recognition into SQLite; entry/exit events also write crops and `events.log`.
The unique count is always a SQL query.

`Video / RTSP / webcam → YOLOv8 → ByteTrack → lifecycle → ArcFace → SQLite → unique count`

```mermaid
flowchart LR
    A[Video / RTSP / webcam] --> B[VideoSource]
    B --> C[YOLOv8 face detector]
    C --> D[ByteTrack]
    D --> E[Track lifecycle]
    E --> F[Quality gate + ArcFace]
    F --> G[(SQLite WAL)]
    E --> H[Crops + events.log]
    H --> G
    G --> I[Unique count]
```

Full component, sequence, identity, state-machine and ER diagrams:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start

### Prerequisites

- Python 3.11 (managed with `uv`).
- An NVIDIA GPU is recommended (validated on an RTX 3050 6 GB); CPU works with
  a lower frame rate.
- ~2 GB free for PyTorch, ONNX Runtime and the InsightFace models.

### Fresh clone

```powershell
git clone https://github.com/mdaashir/KatomaranHackathon.git
cd KatomaranHackathon
```

The input videos are intentionally not committed. Place the provided clips in
`src/video/`, or pass any video with `--source file --path <file>`. A fresh
clone can still inspect [`samples/`](samples/README.md) without videos.

### GPU setup (Windows, recommended)

```powershell
uv python install 3.11
uv venv --python 3.11
.\scripts\setup_gpu.ps1
```

This installs the CUDA 12.8 PyTorch build, ONNX Runtime GPU 1.22, the remaining
dependencies, and the SHA256-verified models. Verify:

```powershell
uv run --no-sync python -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())"
```

### CPU setup

```powershell
uv python install 3.11
uv venv --python 3.11
uv sync --extra cpu
uv run --no-sync python scripts\download_models.py
```

`pyproject.toml` + `uv.lock` are the authoritative dependency definition.
`requirements.txt` is kept as a compact pip fallback for environments without
`uv`.

Set `detector.device` and `recognition.device` to `cpu` in `config.json` to
force CPU execution; `auto` prefers CUDA when available.

## Run

```powershell
# One clip (explicit start time avoids the file mtime)
uv run --no-sync python main.py --source file `
  --path src/video/record_20250620_183903.mp4 `
  --start-time 2025-06-20T18:39:03Z --max-frames 300

# All front-entry clips (filename timestamps + per-clip flush)
uv run --no-sync python main.py --source folder --path src/video

# Interview RTSP stream
uv run --no-sync python main.py --source rtsp --rtsp-url rtsp://camera/stream

# Local host webcam (device index 0)
uv run --no-sync python main.py --source webcam --device-index 0

# Read the DB summary without processing
uv run --no-sync python main.py --report
```

Ctrl+C (or SIGTERM) stops the stream, flushes every open track as an exit, and
writes a `SUMMARY` line. Use `--verify-db` to log `PRAGMA integrity_check` at
startup. `--start-time`, `--source`, `--path`, `--rtsp-url` and
`--device-index` override the config at runtime.

## Docker

The CPU image is the default; the GPU target is built from the same multi-stage
Dockerfile. Models are not baked into the image: they download into the
`models` named volume on first start. The measured CPU image is **2.23 GB**;
the model volume is 181 MB and the first download takes about 100 s here.

```powershell
docker compose up --build
```

The container reads the single `config.json` and processes
`src/video/record_*.mp4`; compose mounts `src/video` read-only,
`data/` and `logs/` are writable for inspection. The API/Streamlit console runs
separately with `uvicorn`/`streamlit`.

```powershell
# GPU (NVIDIA Container Toolkit required)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# One short headless run
docker compose run --rm app python main.py `
  --config config.json --source file `
  --path /app/src/video/record_20250620_183903.mp4 `
  --start-time 2025-06-20T18:39:03Z --max-frames 300

# Linux webcam inside Docker (pass the device explicitly)
docker compose run --rm --device=/dev/video0 app python main.py `
  --source webcam --device-index 0

# Stop
docker compose down
```

The image runs as a non-root user with `init: true` and a 30-second stop grace
period so active tracks flush as exits. For a host webcam, run the API directly
on the machine that owns the camera (`uv run uvicorn ...`); browser camera
permissions do not apply to this server-side source.

## Output layout

```text
data/faces.db                  SQLite (WAL) registry, visits and events
logs/events.log                mandatory audit log (rotating)
logs/app.log                   application log
logs/entries/YYYY-MM-DD/*.jpg  cropped face image for every entry
logs/exits/YYYY-MM-DD/*.jpg    cropped face image for every exit
samples/                       committed real run: CSVs, crops, log, summary
```

## Sample configuration

```json
{
  "source": {"type": "folder", "path": "src/video", "glob": "record_*.mp4", "device_index": 0},
  "frame_skip": 3,
  "detector": {
    "model_path": "models/yolov8n-face.pt",
    "conf_threshold": 0.5,
    "imgsz": 1600,
    "min_face_size": 40,
    "device": "auto"
  },
  "recognition": {
    "model_name": "buffalo_l",
    "model_dir": "models/insightface/models/buffalo_l",
    "device": "auto",
    "similarity_threshold": 0.45,
    "min_face_size": 60,
    "blur_threshold": 60.0
  },
  "tracking": {
    "tracker": "bytetrack",
    "confirm_hits": 3,
    "exit_timeout_frames": 75
  },
  "storage": {"db_path": "data/faces.db", "log_dir": "logs"}
}
```

Every tunable is loaded once into a typed config object and injected into the
pipeline. The complete shape, including RTSP and logging settings, is in
[`config.json`](config.json).

## Measured results

Validated on an RTX 3050 6 GB with `imgsz=1600`, `frame_skip=3`:

| Metric | Result |
|---|---:|
| Clips / frames / detection cycles | 22 / 33,044 / 11,022 |
| Unique faces | 26 |
| Visits (anonymous) | 59 (33) |
| Entries / exits / open | 59 / 59 / 0 |
| Events | 118 |
| One clip incl. model load | 83.5 s |
| Full 22-clip useful time | ~27 min |
| Effective throughput | ~21 fps against a 25 fps source |
| 4K crowd stress | 240 frames, 32 visits, 0 registered (motion blur) |

Face 13 reappeared in a second clip (visits 29 and 51) and was counted once.
The same person's event crops are in
[`samples/images/reentry`](samples/images/reentry).

## Samples

A complete run of the 22 front-entry clips is in
[`samples/`](samples/README.md): 24 cropped images, the Face 13 cross-clip
re-identification, `faces.csv`, `visits.csv`, `events.csv`, an `events.log`
excerpt and `summary.json`.

```powershell
uv run --no-sync python scripts\export_samples.py --config config.json
```

## Operations console (FastAPI + Streamlit)

The API exposes the dashboard/event/log reads plus upload, folder/file/RTSP/
webcam job control, live MJPEG, an SSE activity feed, validated runtime
overrides, a visitor gallery and host/model system info. The Streamlit UI is a
pure HTTP client; the pipeline runs independently.

```powershell
uv sync --extra cpu --extra ui

# Terminal 1
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# Terminal 2
uv run streamlit run ui/app.py
```

API docs: `http://127.0.0.1:8000/docs` · UI: `http://localhost:8501`.
Uploads land in `src/video/`; start a file, folder, RTSP or webcam job from
**Sources & Jobs**, tune the five hot fields in **Config**, browse faces in
**Visitors**, and inspect CPU/memory/ONNX providers in **System**. `frame_skip`
stays owned by `config.json`; **Save as default** is explicit. See [`ui/README.md`](ui/README.md).

## Tests and verification

```powershell
uv run --no-sync python -m pytest
uv run --no-sync python scripts\verify_outputs.py --config config.json
```

`pytest` covers the registry/re-ID matching, the entry/exit state machine, the
unique counter, all video sources, detection, recognition, tracking, events,
storage, the CLI and the API/Streamlit console. `verify_outputs.py` checks SQLite
integrity, foreign keys, exactly one entry and one exit per visit, no open
visits, and that every referenced image exists.

## Continuous integration

GitHub Actions runs `.github/workflows/ci.yml`:

- Ruff check and format check.
- Mypy (non-blocking while the type baseline is cleaned up).
- Pytest with an 85% coverage gate and uploaded JUnit/coverage artifacts.
- CPU and GPU Docker image builds with GitHub Actions layer cache.
- E2E: the trimmed fixture is processed twice (re-ID check), then an RTSP smoke
  runs through MediaMTX with a 300-frame flush check.
- Security: CodeQL, dependency review, gitleaks, pip-audit and Trivy.
- Tagged releases: versioned GHCR images, generated notes and a samples zip.
- Dependabot: grouped weekly updates for uv, Docker and GitHub Actions.

The API/UI layer is covered by contract tests and a Streamlit boot smoke; the
container e2e covers the pipeline.

## Assumptions and known limitations

- The provided dataset is 22 one-minute 2K clips of a mall front entry, plus
  one 8-second 4K crowded stress clip. The 4K clip produced only anonymous
  visits because every face failed the motion-blur gate; the 2K clips are the
  graded dataset.
- A face-only detector cannot count visitors who never face the camera. Such
  visitors are not registered and do not affect the unique count. A person
  detector fallback remains future work.
- Entry/exit is lifecycle-based (first/last sight), not direction-based.
- The local gates are deliberately conservative (`min_face_size=60`,
  `blur_threshold=60`); 33 of 59 real visits were anonymous for this reason.
  Lowering them raises registrations but risks duplicate identities.
- Long GPU sessions can stall after ~20 minutes on this laptop; restart the
  process or run in segments, then re-run `verify_outputs.py`. See PLAN R13.
- The long 22-clip pass averaged ~21 fps against a 25 fps source; the RTSP
  queue drops stale frames. Tune `imgsz` / `frame_skip` on the interview GPU if
  the source cadence is 25 fps.
- Face crops are biometric data and stay local. Nothing is uploaded.

This project is a part of a hackathon run by https://katomaran.com
