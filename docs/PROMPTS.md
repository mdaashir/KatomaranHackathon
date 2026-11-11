# PROMPTS — key AI prompts used

> Required by `AGENTS.md`: the prompts are evaluated in the interview. This is
> an honest log of the prompts that materially shaped the project, not a
> transcript of every message.

---

## 2026-11-11 — Phase 0: architecture and planning

### Prompt 1 (planning session)

> read @AGENTS.md @"docs\Katomarans Problem Statement.pdf" before start.
> @src\video has video datasets mentioned. Read
> docs/Katomarans_Problem_Statement.pdf. Produce docs/PLAN.md, FEATURES.md,
> COMPUTE.md and ARCHITECTURE.md (Mermaid).

**Purpose:** establish the architecture, feature list, compute budget and
diagrams before writing any code.

**What the model did:**

1. Extracted the PDF with `pdftotext` and read all 143 lines.
2. Probed all 23 videos with `ffprobe`.
3. Extracted representative frames with `ffmpeg` and inspected them visually.
4. Discovered that the local dataset is two different things:
   - 22 × 60 s HEVC clips (2688×1520, 25 fps) of a mall front entry, sparse
     traffic, many back-facing people;
   - one 8 s 4K H.264 clip of a crowded bazaar, suitable as a stress test.
5. Used that evidence to choose native-resolution detection, separate
   track/register size gates, per-track embedding caching, and per-clip
   track flushing.
6. Wrote the four documents and cross-checked terminology, state machine,
   schema and compute profiles for consistency.

**Key follow-up prompts used during the session:**

- “Confirm the entry/exit definition: lifecycle-based or direction-based?”
  → resolved to lifecycle-based (frame appearance → exit timeout), direction
  stored as metadata only.
- “How should back-facing visitors be handled?”
  → documented limitation with an optional person-detector fallback.
- “What happens when the same face reappears after an exit?”
  → new visit, same Face ID, unique count unchanged (no visit reopening).
- “Is `imgsz=1920` real-time on CPU?”
  → no; split into 1600 CPU / 1920 GPU profiles.

**Outputs:** `docs/PLAN.md`, `docs/FEATURES.md`, `docs/COMPUTE.md`,
`docs/ARCHITECTURE.md`.

**Decisions still awaiting author sign-off:** see `PLAN.md §10`.

---

## 2026-11-11 — Phase 1: foundations

### Prompt

> commit the current staged and start proceeding make sure you always
> following the agents.md

**Purpose:** commit the approved planning baseline and start Phase 1 of the
build without crossing the architecture boundaries.

**What the model did:**

1. Restored project context via `/remember restore` (no `memory.md` yet;
   `AGENTS.md` was the only context file).
2. Committed the baseline on `main`, including the docs and project files.
3. Discovered that committing the 232 MB of input `.mp4` files made the push
   fail; removed them from git history, kept them on disk, added
   `src/video/*.mp4` to `.gitignore`, and documented `src/video/README.md`.
4. Built the Phase 1 foundations:
   - `src/config/` — frozen dataclasses, strict loader, `${ENV_VAR}`
     expansion, key-level validation.
   - `src/storage/` — SQLite WAL connection, schema, transaction context,
     repositories for faces, embeddings, visits and events.
   - `src/logging_setup.py` — rotating console/app/debug logs and the
     pipe-delimited `events.log` formatter.
5. Wrote 56 pytest tests and ran the `/review` skill. Fixed the review findings:
   quality range validation, anonymous/linked event consistency, negative
   track IDs, and log field escaping.
6. Pushed `main` and `feature/config-storage-logging`; opened PR #1.

**Key follow-up prompts used during the session:**

- “Why did the push time out?” → auth was fine; the 232 MB input dataset was
  the problem. Removed it from git history.
- “Does the DB enforce one entry and one exit per visit?” → added
  `UNIQUE(visit_id, event_type)` and a duplicate-event test.
- “Are anonymous and linked events allowed to mix?” → no; added validation and
  tests.

**Tests:** `56 passed`.

**Outputs:** `src/config/`, `src/storage/`, `src/logging_setup.py`,
`tests/`, PR #1.

**Open questions:** the five sign-off items in `PLAN.md §10` are still
unresolved; implementation proceeded with the documented defaults.

---

## 2026-11-11 — Phase 2: video sources

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding
> next make sure you always following the agents.md

**Purpose:** implement Phase 2 (`PLAN.md §7`) on a fresh branch after the
Phase 1 merge.

**What the model did:**

1. Restored context from `memory.md`, confirmed PR #1 was merged, pulled
   `main`, deleted the merged local branch, and created
   `feature/video-source`.
2. Added the `Frame` contract with `global_index`, `window_id`, and
   `is_window_end`, so the pipeline can keep a stable cadence and flush tracks
   at every clip boundary.
3. Added `FileSource` (PyAV decode, PTS timestamps, one-frame lookahead for the
   final-frame marker), `FileSequenceSource` (ordered clips, filename timestamp
   parsing, corrupt-clip skip), and `RTSPSource` (bounded drop-oldest queue,
   exponential reconnect, safe shutdown).
4. Added 19 tests (75 total) and a real-data smoke test on
   `record_20250620_183903.mp4`: 1,502 frames, 59.96 s, 2688×1520, 74.8–76.7
   decode fps.
5. Ran `/review`. Fixed a Windows access violation caused by closing a PyAV
   container while its generator was suspended; decoder lifetime now belongs to
   the generator. Removed the equivalent cross-thread close from `RTSPSource`.

**Problems solved:**

- The rebase onto the rewritten baseline deleted the ignored `.mp4` files
  because the old baseline tracked them. Restored all 23 from the dangling
  baseline commit without re-adding them to git.
- PyAV is not thread-safe for container close. `FileSource.close()` now only
  requests a stop; `RTSPSource.close()` stops and joins, and never closes the
  container from the consumer thread.

**Tests:** `75 passed`.

**Outputs:** `src/video/`, `tests/test_video.py`, updated `requirements.txt`.

**Remaining manual check:** killing and restoring a real RTSP loop. The
automated tests cover reconnect/backoff and a fake-container decode path; no
local RTSP server is available in this environment.

---

## 2026-11-11 — Phase 3: YOLO face detector

### Prompt

> successfully merged the pr, deleted the branch in remote, always append
> memory dont overwrite. start proceeding next make sure you always following
> the agents.md

**Purpose:** implement Phase 3 (`PLAN.md §7`): the YOLO face detector wrapper,
native-coordinate filtering, and measured latency.

**What the model did:**

1. Restored `memory.md`, confirmed PR #2 was merged, pulled `main`, deleted the
   merged local branch, and created `feature/yolo-face-detector`.
2. Downloaded and vendored `models/yolov8n-face.pt` (6,247,374 bytes, SHA256
   recorded in `models/README.md`).
3. Added `src/detection/`: `FaceDetection`, `DetectorError`, and
   `YoloFaceDetector` with a lazy Ultralytics import, an injectable model
   factory, native-coordinate clamping, the 40 px size gate, and device
   mapping.
4. Added 14 fake-model tests and 2 real-image tests (91 total).
5. Benchmarked the real model at `imgsz` 1280/1600/1920 on a 2K front-entry
   frame and the 4K crowd frame; recorded counts and latency in `COMPUTE.md`.

**Problems solved:**

- The checkpoint needs `omegaconf`, which Ultralytics does not install; added
  it to `requirements.txt`.
- The checkpoint is **detect-only** (`task=detect`, no keypoint head). The
  interface returns `keypoints=None`; Phase 4 will use InsightFace landmarks
  for ArcFace alignment. `PLAN.md` and `ARCHITECTURE.md` were updated.
- The first CPU inference includes ~9 s of PyTorch warmup; documented and
  excluded from steady-state measurements.
- Measured CPU latency is 196/297/417 ms at 1280/1600/1920, materially higher
  than the estimate. The compute budget and CPU profiles were corrected.

**Tests:** `91 passed`.

**Outputs:** `src/detection/`, `models/yolov8n-face.pt`,
`models/README.md`, `tests/test_detector*.py`, updated `requirements.txt`,
`PLAN.md`, `ARCHITECTURE.md`, `COMPUTE.md`.

**Remaining manual check:** far-face recall is still partial (1–2 faces at
40 s depending on `imgsz`); optional 2×2 tiling remains a Phase 8 decision.

---

## 2026-11-11 — Phase 4: InsightFace embeddings and registry

### Prompt

> successfully merged the pr, deleted the branch in remote, commit memory.md
> too, use gpu. start proceeding next make sure you always following the
> agents.md

**Purpose:** implement Phase 4 with CUDA execution: quality gate, InsightFace
alignment, ArcFace embeddings, and the cosine registry.

**What the model did:**

1. Synced `main`, deleted the merged local branch, and created
   `feature/insightface-embedder-registry`.
2. Confirmed an RTX 3050 6 GB and installed `torch 2.11.0+cu128`,
   `insightface 2.1`, and `onnxruntime-gpu==1.22.0`.
3. Downloaded the official `buffalo_l` pack and loaded only
   `w600k_r50.onnx` (ArcFace) and `2d106det.onnx` (landmarks) through
   `ArcFaceONNX` / `Landmark`; InsightFace's detector is never run.
4. Added `src/recognition/`: quality gate + crop, 106→5 alignment, CUDA/CPU
   provider selection, lazy runtime, embedder, and SQLite-backed cosine
   registry. Added `recognition.model_dir` / `recognition.device` to config
   and `count_embeddings()` to storage.
5. Added 36 tests (127 total), including a real GPU validation on the provided
   clips.
6. Added `scripts/download_models.py`, `scripts/setup_gpu.ps1`,
   `models/insightface/README.md`, and GPU setup instructions in the README.

**Problems solved:**

- ORT GPU 1.31 requires CUDA 13 DLLs (`cublasLt64_13.dll`), but Torch's cu128
  wheel ships CUDA 12.8. Pinned `onnxruntime-gpu==1.22.0`, which uses CUDA 12.
- `insightface` pulls the CPU `onnxruntime`; the setup script removes it so the
  GPU module wins. PyTorch's CUDA/cuDNN DLL directory is added before ONNX
  Runtime import.
- `FaceAnalysis` requires a detection module even when only recognition and
  landmarks are wanted. Loading `ArcFaceONNX` and `Landmark` directly avoids
  that and skips the unused 17 MB detector.
- The initial 106→5 index guess landed on eyebrow points. Labeled the 106-point
  crop and confirmed: eye clusters `33:43` / `87:97`, nose `86`, mouth corners
  `52` / `61`.
- The 174 MB recognizer exceeds GitHub's per-file limit, so both models are
  fetched by a verified download script instead of committed.
- Most 4K crowd faces fail `blur_threshold=60` due to motion blur; the real
  test selects a quality-passing face. Phase 8 can tune the threshold.

**Measured:** GPU providers `['CUDAExecutionProvider', 'CPUExecutionProvider']`;
landmarks **4.1 ms**, ArcFace **12.7 ms**; CPU 10.1 ms / 96.6 ms; same-person
cosine **0.733 / 0.701**, different person **0.118**.

**Tests:** `127 passed`.

**Outputs:** `src/recognition/`, config/storage updates, 36 tests, GPU setup +
model download scripts, model documentation, updated plan/architecture/compute.

**Remaining:** tracking and visit lifecycle (Phase 5); tune blur/registration
thresholds in Phase 8; real RTSP kill/restore remains a manual check.

---

## 2026-11-11 — Phase 5: tracking and visit lifecycle

### Prompt

> successfully merged the pr, deleted the branch in remote, exclude all models
> from git. start proceeding next make sure you always following the agents.md

**Purpose:** implement Phase 5 (ByteTrack + exactly one entry/exit per visit)
and remove every model weight from git tracking.

**What the model did:**

1. Synced `main`, deleted the merged local branch, and created
   `feature/tracking-lifecycle`.
2. Excluded all `*.pt`, `*.onnx` and `*.zip` files under `models/` from git,
   removed the tracked YOLO checkpoint from the index, and extended
   `scripts/download_models.py` to fetch and verify the YOLO checkpoint too.
3. Pinned `supervision==0.25.1`; 0.31 removed the ByteTrack module.
4. Added `src/tracking/`:
   - `ByteTrackAdapter` configured with `frame_rate = fps / frame_skip`.
   - `TrackLifecycle` implementing `NEW → ACTIVE → LOST → EXITED` plus
     `DROPPED`, confirmation, reacquisition, window/stream flush, and
     identity-anchored open visits.
   - `TrackedBox`, `Track`, `Visit`, `VisitEvent`, `LifecycleSummary` types.
5. Added tracking thresholds (`track_activation_threshold`,
   `minimum_matching_threshold`) to config and validation.
6. Added 23 tests (154 total), including a ByteTrack→lifecycle seam test.

**Problems solved:**

- `supervision` 0.31 has no `tracker` package at all; pin 0.25.1.
- Removing the YOLO checkpoint from git would break model loading, so the
  downloader now fetches and SHA256-verifies it alongside buffalo_l.
- Identity anchoring can only be reliable if recognition runs before
  confirmation. `observe(..., face_ids=...)` supports that; `assign_face()`
  fills an anonymous open visit later, and Phase 6 updates the stored entry.

**Measured:** ByteTrack update with 10 boxes is **1.77 ms median** / 4.22 ms
p95. Lifecycle cost is negligible.

**Tests:** `154 passed`.

**Outputs:** `src/tracking/`, config thresholds, 23 tests, model-exclusion
updates, updated plan/architecture/compute/README/memory.

**Remaining:** persist events/images/DB and derive the unique count (Phase 6);
wire the full pipeline (Phase 7).

---

## 2026-11-11 — Phase 6: event logging and counting

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding
> next make sure you always following the agents.md

**Purpose:** persist lifecycle events as images + DB rows + `events.log`, and
expose the DB-derived unique count and run summary.

**What the model did:**

1. Synced `main`, deleted the merged branch, and created
   `feature/event-logging-and-counter`.
2. Added per-window `video_time_s` and optional `similarity` to lifecycle
   events, so the DB carries `events.video_time_s` correctly.
3. Added cross-table repository operations:
   - `create_visit_with_entry()` — visit + entry in one transaction.
   - `close_visit_with_exit()` — exit + close in one transaction.
   - `link_face()` / `visit_face()` — retroactive anonymous-to-identity link.
4. Added `src/events/`:
   - `ImageStore` — atomic JPEG writes with fsync + rename, dated folders,
     safe filenames, black placeholder for a missing crop.
   - `EventLogger` — image → DB transaction → `events.log`, lifecycle→DB visit
     mapping, rollback of the image on DB failure.
5. Added `src/counting/`:
   - `UniqueCounter.get_unique_count()` from `faces WHERE is_active=1`.
   - `get_summary()` with visits, open/anonymous visits, entries/exits and the
     event time range.
6. Added 29 tests (185 total) for image storage, event persistence, counters,
   and the new repository methods.

**Problems solved:**

- FK violations in tests showed that seeded Face IDs are required; production
  IDs always come from the registry. Fixtures now seed faces.
- The merged PR deleted the previously tracked YOLO checkpoint from disk during
  checkout. `scripts/download_models.py` restored and verified it — exactly the
  workflow the model-exclusion change was meant to support.
- Exits were re-logging `VISIT_LINKED` for already-linked faces. Added
  `visit_face()` and a conflicting-face guard.

**Measured:** Entry persistence **4.78 ms** median / 6.36 ms p95; exit
persistence **4.76 ms** / 5.54 ms p95 (256 px JPEG + SQLite transaction).

**Tests:** `185 passed`.

**Outputs:** `src/events/`, `src/counting/`, lifecycle `video_time_s`, storage
cross-table methods, 29 tests, updated plan/architecture/compute/README/memory.

**Remaining:** wire the pipeline + CLI and graceful shutdown (Phase 7).

---

## 2026-11-11 — Phase 7: pipeline and CLI

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding
> next make sure you always following the agents.md

**Purpose:** wire every module into a runnable pipeline and CLI, with graceful
shutdown and an end-to-end smoke run.

**What the model did:**

1. Synced `main`, deleted the merged branch, and created
   `feature/pipeline-cli`.
2. Added `src/pipeline/`:
   - `Pipeline` frame loop with `frame_skip`, recognition-before-confirmation,
     per-window flush/reset, best-crop cache, and per-track recognition retry.
   - `CropCache` for exit images when the face is no longer in frame.
   - `PipelineResult` summary type.
3. Added `Frame.fps` to all sources so ByteTrack can be created with the real
   detection cadence.
4. Added `main.py`:
   - Composition root, dependency wiring, `--config`, `--source`, `--path`,
     `--rtsp-url`, `--start-time`, `--max-frames`, `--log-level`, `--report`,
     `--verify-db`.
   - SIGINT/SIGTERM set a stop event and close the source so an RTSP stall
     cannot block shutdown.
   - `--report` prints the DB summary as JSON.
5. Added 21 tests (208 total) for the pipeline, crop cache, CLI overrides,
   report, and error paths.

**Problems solved:**

- Ctrl+C could not wake a blocked RTSP queue. `main` now owns the source and
  closes it from the signal handler; `run_pipeline` also closes it if pipeline
  construction fails.
- A Windows-authored config with a UTF-8 BOM failed JSON parsing. The loader now
  reads `utf-8-sig`.
- A failed embedding-cache write discarded a valid recognition; it is now
  non-fatal and logged.
- Added `--start-time` so a single file does not inherit its mtime.

**Measured:** 30 frames / 10 detection cycles, 1 unique, 1 entry, 1 exit,
0 open; 9.3 KB entry crop and 8.8 KB exit crop; first event
`2025-06-20T18:39:03.000Z`.

**Tests:** `208 passed`.

**Outputs:** `src/pipeline/`, `main.py`, `Frame.fps`, BOM-safe config loading,
21 tests, updated plan/architecture/compute/README/memory.

**Remaining:** run all 22 clips, calibrate thresholds, produce `samples/`
(Phase 8), then final README/demo (Phase 9).

---

## Template for future phases

```markdown
## YYYY-MM-DD — Phase N: <name>

### Prompt
> <the request, verbatim>

**Purpose:** <why this prompt was used>

**What the model did:**
1. <step>
2. <step>

**Key follow-up prompts:**
- <question> → <resolution>

**Outputs:** <files, branches, PR>

**Corrections made by the author:** <what the author changed and why>
```
