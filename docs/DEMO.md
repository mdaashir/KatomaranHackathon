# Demo video script

> **Published:** [YouTube walkthrough](https://youtu.be/bekQvFoOWTY) ·
> local [`demo.mkv`](../demo.mkv). This recording and the Google Form
> submission are complete.
>
> Compliance audit: [`COMPLIANCE.md`](COMPLIANCE.md).

Target length: **4–5 minutes**. Record the terminal at 1080p with a readable
font, and keep `samples/` open in Explorer or the editor.

## Before recording

1. Activate the environment and confirm the GPU:
   ```powershell
   uv run --no-sync python -c "import torch; print('cuda', torch.cuda.is_available())"
   ```
2. Confirm the models exist:
   ```powershell
   Test-Path models\yolov8n-face.pt
   Test-Path models\insightface\models\buffalo_l\w600k_r50.onnx
   ```
3. Open these tabs/windows:
   - `README.md`
   - `samples/README.md`
   - `samples/events.log`
   - `samples/images/reentry/`
4. For a clean live count, remove `data/demo.db` and `logs/demo` if they exist
   from an earlier rehearsal. `docs/demo_config.json` points there, so the live
   run does not touch the main database.

## 0:00–0:30 — Problem and result

Show the repository README.

> “This is a face detection, recognition and tracking pipeline for a mall front
> entry. It logs every entry and exit, counts unique visitors from SQLite, and
> produces the required audit log and cropped images. On the 22 provided clips
> it registered 26 unique visitors across 59 visits, with 59 entries and 59
> exits and zero open visits.”

## 0:30–1:00 — Architecture

Scroll to the Mermaid diagram, then open `docs/ARCHITECTURE.md`.

> “Video or RTSP goes through PyAV, YOLO detects faces every third frame,
> ByteTrack keeps IDs alive on skipped frames, a quality gate filters tiny,
> blurry and profile faces, and ArcFace matches or auto-registers the identity.
> The lifecycle emits exactly one entry and one exit, and the unique count is a
> SQL query, not a memory counter.”

## 1:00–1:40 — Setup

Show the README quick start and run the GPU check.

```powershell
uv run --no-sync python -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())"
```

> “Models are downloaded and SHA256-verified by `scripts/download_models.py`;
> no weights are committed. The pinned stack is Torch CUDA 12.8 and ONNX
> Runtime GPU 1.22.”

## 1:40–3:00 — Live run and artifacts

Run one clip into the demo database:

```powershell
uv run --no-sync python main.py --config docs/demo_config.json `
  --start-time 2025-06-20T18:39:03Z --max-frames 300
```

Then show:

```powershell
Get-Content logs\demo\events.log -Tail 12
uv run --no-sync python main.py --config docs/demo_config.json --report
```

Open one entry image and one exit image from `logs\demo\entries\` and
`logs\demo\exits\`.

> “The log shows the embedding, the auto-registration, the entry, the exit and
> the summary. The report is read back from SQLite. The images are the crops
> attached to the events.”

## 3:00–3:45 — Full 22-clip sample

Open `samples/README.md`, `summary.json`, `events.log` and
`samples/images/reentry/`.

> “This is the committed full run: 26 unique, 59 visits, 33 anonymous, 118
> events and zero open. Face 13 appears in two different clips and is counted
> once — that is the re-identification path. The 33 anonymous visits are
> far, blurred or back-facing people who never produced a usable embedding.”

## 3:45–4:30 — Verification and resilience

```powershell
uv run --no-sync python scripts\verify_outputs.py --config docs/demo_config.json
uv run --no-sync python -m pytest -q
```

> “The verifier checks SQLite integrity, foreign keys, exactly one entry and one
> exit per visit, no open visits, and that every image exists. The suite has
> 210 tests, including the registry, state machine and unique counter.”

Show the RTSP command and explain Ctrl+C:

```powershell
uv run --no-sync python main.py --source rtsp --rtsp-url rtsp://camera/stream
```

> “At the interview I switch the source to RTSP. The source reconnects with
> capped backoff, and Ctrl+C flushes every active track as an exit before
> writing the summary.”

## 4:30–5:00 — Close

> “Known limitation: a face-only detector cannot count someone who never faces
> the camera; those visits are logged as anonymous. The unique count is always
> derived from SQLite. The repo and samples are linked in the README.”

## Recording status

**Completed 2026-11-13.** The walkthrough is published at
[https://youtu.be/bekQvFoOWTY](https://youtu.be/bekQvFoOWTY) and a local copy is
in [`demo.mkv`](../demo.mkv). The link was verified in a private browser window
and submitted with the repository through the Katomaran Google Form.
