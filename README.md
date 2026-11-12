# Intelligent Face Tracker

Real-time face detection, recognition, tracking, auto-registration and unique
visitor counting for a mall front-entry camera. It accepts a video file, a
folder of clips, or an RTSP stream, and logs exactly one entry and one exit per
visit.

> **Status: submission-ready (Phase 10 + compliance audit).** The pipeline,
> samples and documentation are complete. The only remaining actions are
> recording the walkthrough in [`docs/DEMO.md`](docs/DEMO.md), replacing
> `VIDEO_URL`, and submitting the Google Form. See
> [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md).

- Problem statement: [Katomarans Problem Statement](<docs/Katomarans Problem Statement.pdf>)
- Repository: https://github.com/mdaashir/KatomaranHackathon
- Samples: [`samples/README.md`](samples/README.md)
- Plan: [`docs/PLAN.md`](docs/PLAN.md)
- Compliance audit: [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md)

## Demo video

> **TODO before submission:** record the 4–5 minute walkthrough using
> [`docs/DEMO.md`](docs/DEMO.md), upload it to Loom or YouTube, and replace
> `VIDEO_URL` with the public link.

**Video:** `VIDEO_URL` (replace with the public Loom/YouTube link)

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
.venv\Scripts\python.exe -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())"
```

### CPU setup

```powershell
uv python install 3.11
uv venv --python 3.11
uv pip install -r requirements.txt
.venv\Scripts\python.exe scripts\download_models.py
```

Set `detector.device` and `recognition.device` to `cpu` in `config.json` to
force CPU execution; `auto` prefers CUDA when available.

## Run

```powershell
# One clip (explicit start time avoids the file mtime)
.venv\Scripts\python.exe main.py --source file `
  --path src/video/record_20250620_183903.mp4 `
  --start-time 2025-06-20T18:39:03Z --max-frames 300

# All front-entry clips (filename timestamps + per-clip flush)
.venv\Scripts\python.exe main.py --source folder --path src/video

# Interview RTSP stream
.venv\Scripts\python.exe main.py --source rtsp --rtsp-url rtsp://camera/stream

# Read the DB summary without processing
.venv\Scripts\python.exe main.py --report
```

Ctrl+C (or SIGTERM) stops the stream, flushes every open track as an exit, and
writes a `SUMMARY` line. Use `--verify-db` to log `PRAGMA integrity_check` at
startup. `--start-time`, `--source`, `--path` and `--rtsp-url` override the
config at runtime.

## Output layout

```text
data/faces.db                  SQLite (WAL) registry, visits and events
logs/events.log                mandatory audit log (rotating)
logs/app.log                   application log
logs/entries/YYYY-MM-DD/*.jpg  cropped face image for every entry
logs/exits/YYYY-MM-DD/*.jpg    cropped face image for every exit
samples/                       committed real run: CSVs, crops, log, summary
```

## Architecture

```mermaid
flowchart LR
    A[Video file / folder / RTSP] --> B[VideoSource<br/>PyAV, reconnect]
    B --> C[YOLOv8 face detector]
    C --> D[ByteTrack]
    D --> E[Track lifecycle<br/>NEW → ACTIVE → LOST → EXITED]
    E --> F[Quality gate + ArcFace]
    F --> G[(SQLite WAL<br/>faces, visits, events)]
    E --> H[Entry/exit crops<br/>events.log]
    H --> G
    G --> I[DB-derived unique count]
```

The full component, sequence, identity, state-machine and ER diagrams are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The record/prompts used to build
each phase are in [`docs/PROMPTS.md`](docs/PROMPTS.md).

## Sample configuration

```json
{
  "source": {"type": "folder", "path": "src/video", "glob": "record_*.mp4"},
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
.venv\Scripts\python.exe scripts\export_samples.py --config config.json
```

## Dashboard (optional)

A read-only HTML snapshot of the current database and recent event crops. It
does not import the pipeline and adds no dependency:

```powershell
.venv\Scripts\python.exe scripts\dashboard.py --config config.json --open
```

Open [`samples/dashboard/index.html`](samples/dashboard/index.html) for the
snapshot generated from the 22-clip run.

## Tests and verification

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe scripts\verify_outputs.py --config config.json
```

`pytest` covers the registry/re-ID matching, the entry/exit state machine, the
unique counter, all video sources, detection, recognition, tracking, events,
storage, the CLI and the read-only dashboard. `verify_outputs.py` checks SQLite
integrity, foreign keys, exactly one entry and one exit per visit, no open
visits, and that every referenced image exists.

## Business rules implemented

- Auto-register an unmatched face above `similarity_threshold`; reuse the Face
  ID on a match.
- Skip tiny, blurry or profile faces to avoid garbage registrations.
- Entry timestamp = first sight; exit timestamp = last sight; exit fires after
  `exit_timeout_frames` misses.
- Log every event to the DB and `events.log`; save the cropped image atomically.
- Derive the unique count from `COUNT(faces)` in SQLite, not a memory counter.
- Anonymous visits are logged but excluded from the unique count.
- WAL, transactional event writes, atomic image rename, SIGINT/SIGTERM flush
  and capped RTSP reconnect.

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

## Submission

- Repository: https://github.com/mdaashir/KatomaranHackathon
- Demo video: `VIDEO_URL` — replace after recording.
- Samples: [`samples/README.md`](samples/README.md)
- Compliance audit: [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md)
- Submission checklist: [`docs/SUBMISSION.md`](docs/SUBMISSION.md)
- Deadline: **2:00 PM, Friday 13 November 2026**.

This project is a part of a hackathon run by https://katomaran.com
