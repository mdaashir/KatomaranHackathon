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

## 2026-11-11 — Phase 8: end-to-end validation and samples

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding next make sure you always following the agents.md

**Purpose:** run the whole 22-clip dataset, verify the event/visit invariants,
calibrate the conservative gates, and produce the required `samples/` folder.

**What the model did:**

1. Synced `main`, removed `feature/pipeline-cli`, and created
   `test/e2e-and-calibration`.
2. Timed one 60 s clip (1,502 frames / 501 cycles / 83.5 s including model
   load), then ran all 22 `record_*.mp4` clips on a fresh SQLite DB at
   `imgsz=1600`, `frame_skip=3`.
3. Added `scripts/verify_outputs.py` (DB integrity, FK check, one entry + one
   exit per visit, no open visits, every referenced image exists).
4. Added `scripts/export_samples.py` (CSV exports, log excerpt, 20 crop images,
   re-entry images for faces seen in more than one visit).
5. Ran `video_sample1.mp4` as a 4K crowded stress test.
6. Found one cross-clip re-identification: Face 13 in visits 29 and 51; the
   unique count stayed 26.
7. Fixed `CropCache` so a quality-passing face crop wins over a merely large
   fallback crop (the old cache could pick a back-of-head image).
8. Updated PLAN/COMPUTE/README/memory and added two crop-cache tests.

**Problems solved:**

- A transient GPU stall after 21 clips left the process with no CPU progress
  and the GPU spinning. A fresh process finished the last clips into the same
  DB; `verify_outputs.py` then passed all checks. `nvidia-smi` later showed
  510 MHz / 13.5 W, so this is recorded as a driver/power-state risk (PLAN R13).
- The sample export initially took the largest crop, which produced
  back-of-head images for faces recognized later. `CropCache` now prefers the
  quality crop; the existing sample notes the earlier run's back-of-head
  re-entry views.
- Scripts launched from `scripts/` could not import `src`; both add the project
  root to `sys.path`.

**Measured:** 22 clips / 33,044 frames, 26 unique, 59 visits, 33 anonymous,
118 events, 0 open; 24 sample images; 4K stress 240 frames / 32 visits /
0 registered (motion blur).

**Tests:** `210 passed`.

**Outputs:** `samples/`, `scripts/verify_outputs.py`,
`scripts/export_samples.py`, quality-first `CropCache`, updated docs.

**Remaining:** Phase 9 README, demo video, submission.

---

## 2026-11-11 — Phase 9: README, demo and submission

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding next make sure you always following the agents.md

**Purpose:** make the repository self-contained for review: final README,
recording script, submission checklist and a reproducible demo config.

**What the model did:**

1. Synced `main`, removed `test/e2e-and-calibration`, and created
   `docs/readme-and-demo`.
2. Rewrote `README.md` with setup for GPU and CPU, a fresh-clone path, run
   commands, output layout, a Mermaid architecture diagram, sample config,
   measured results, samples, tests, limitations, submission links and the
   mandatory Katomaran line.
3. Added `docs/DEMO.md`: a 4–5 minute, section-by-section walkthrough with the
   exact commands and the artifacts to show.
4. Added `docs/demo_config.json` (one clip, isolated `data/demo.db` and
   `logs/demo`) so the live demo does not touch the main database.
5. Added `docs/SUBMISSION.md` with the repository link, video placeholder and
   pre-submission checks.
6. Re-ran the demo command: 300 frames / 100 cycles / 50.2 s, 1 unique, 1
   visit, 1 entry, 1 exit, 0 open; `verify_outputs.py` clean.

**Problems solved:**

- The demo needs a clean count without deleting the real DB; the dedicated
  `docs/demo_config.json` isolates `data/demo.db` and `logs/demo`.
- A fake/relative video link would 404; the README uses a clearly marked
  `VIDEO_URL` placeholder until the author records the walkthrough.

**Tests:** `210 passed` (unchanged); demo verifier clean.

**Outputs:** final README, `docs/DEMO.md`, `docs/demo_config.json`,
`docs/SUBMISSION.md`, updated memory/plan.

**Remaining:** record the Loom/YouTube video, replace `VIDEO_URL` in README and
SUBMISSION, and submit through the Katomaran form. Do not mark the PR as the
final submission until that is done.

---

## 2026-11-11 — Phase 10: optional read-only dashboard

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding next make sure you always following the agents.md

**Purpose:** add the optional read-only dashboard without touching the core
pipeline, and ship a committed snapshot generated from the real 22-clip run.

**What the model did:**

1. Synced `main`, removed `docs/readme-and-demo`, and created
   `feature/dashboard`.
2. Added `src/dashboard/`: `DashboardBuilder` reads SQLite, copies recent event
   thumbnails, and writes a standalone `index.html`; `render_dashboard` is a
   pure HTML renderer with escaped DB text.
3. Added `scripts/dashboard.py` with `--config`, `--db`, `--output`,
   `--max-events`, `--max-visits` and `--open`.
4. Added six tests: summary/rows, image copying, missing-image placeholder,
   truncation/validation, empty states and safe image cleanup.
5. Generated `samples/dashboard/index.html` from `data/e2e.db`: 26 unique,
   59 visits, 24 recent events and 24 thumbnails.
6. Updated README, PLAN, FEATURES, ARCHITECTURE, sample README and `.gitignore`
   (`/dashboard/`).

**Design:** static generation rather than a server. It adds no dependency,
cannot block the pipeline, and can be refreshed on demand. Live serving can be
added later without changing the renderer.

**Tests:** `216 passed`.

**Outputs:** `src/dashboard/`, `scripts/dashboard.py`,
`tests/test_dashboard.py`, `samples/dashboard/`, updated docs.

**Remaining:** the public demo video is still the only blocker before the
submission form; the dashboard is an optional extra.

---

## 2026-11-11 — Compliance audit against the problem statement

### Prompt

> successfully merged the pr, deleted the branch in remote. cross verify with @docs\"Katomarans Problem Statement.pdf" if all done as per what asked and stated if in case of missing do it next iteration. first create a checklist and validate. make sure you always following the agents.md. update all docs mds to current

**Purpose:** verify the submission against the actual problem statement rather
than against the project's own plan, and refresh every document to the final
state.

**What the model did:**

1. Extracted the four-page PDF with `pypdf` in the scratchpad (not added to the
   project dependencies) and read all 43 stated requirements.
2. Created `docs/COMPLIANCE.md` mapping every requirement to its implementation
   and evidence, with `DONE`, `PARTIAL`, `MISSING` and `AUTHOR` statuses.
3. Re-ran `scripts/verify_outputs.py` on `data/e2e.db` and the full suite.
4. Updated README, PLAN, FEATURES, COMPUTE, ARCHITECTURE, DEMO, SUBMISSION,
   sample README, PROMPTS and memory to the final audit state.
5. Recorded the remaining actions: public video, Google Form, real-time tuning
   on the interview GPU, and an optional real-camera RTSP confirmation.

**Result:** 216 tests pass; verifier is clean (26 unique, 59 visits, 59/59
entries/exits, 0 open, 0 missing images). All functional requirements are done;
real-time throughput is a documented performance caveat (~21 fps against 25
fps in the long run). The public video is the only missing deliverable; the
form is the only other external action.

**Outputs:** `docs/COMPLIANCE.md` plus updated documentation.

---

## 2026-11-12 — P1 Docker and uv packaging

### Prompt

> successfully merged the pr, deleted the branch in remote, I had updated @AGENTS.md included new 2 md in docs. start proceeding make sure you always following the agents.md

### Correction prompts

> wait i have serveral corrections update accordingly and work on
> 1. move entrypoint.sh to root
> 2. no more inputs folder use src/video itself
> 3. remove requirements folder use only single requirements.txt in root with all properly commented and seperated.
> 4. run uv init that might add some files modify it for easy use also use uv everywhere.
> 5. no sperate config for docker.
> 6. why makefile remove it if not needed.

> why .env? remove if not needed

> why to still use requirement.txt use uv's toml which provides all these packages segreateions like cpu, gpu etc. have requirements.txt as compactibility or fallback

**Purpose:** ship the P1 headless container layer while keeping a single source
of truth for configuration and dependencies.

**What the model did:**

1. Added a multi-stage CPU/GPU Dockerfile with `uv` in the builder, root
   `entrypoint.sh`, compose + GPU override, and a single `config.json`.
2. Moved dependencies into `pyproject.toml`: core runtime, `cpu`/`gpu`/`ui`
   extras, `dev` dependency group; generated `uv.lock`; kept a compact
   `requirements.txt` fallback.
3. Removed the separate `inputs/`, `config.docker.json`, `.env.example`,
   `requirements/` and `Makefile`; compose mounts `src/video` and uses
   `config.json`.
4. Built the CPU image with `uv sync`, loaded models into a named volume, and
   ran the pipeline in the container.

**Problems solved:**

- A `.dockerignore` entry excluded all of `src/video` instead of only its
  videos; changed to `src/video/*.mp4`.
- `uv run` would re-sync the GPU environment back to the CPU lock; all user
  commands now use `uv run --no-sync` after setup.
- The GPU image installs the CUDA Torch wheel after `uv sync`; the GPU build is
  still pending a local run.

**Validated:** CPU image **2.23 GB**; clean 30-frame container run **1 unique /
1 entry / 1 exit / 0 open** in 29.5 s; first-start model download 101.8 s
(181 MB); `docker compose config` OK; entrypoint OK; `pytest` **216 passed**;
e2e verifier clean.

**Remaining:** `ci/core-workflow`, e2e fixture/RTSP smoke, GPU build, then
P2/P3. Do not claim P1 CI complete until those branches land.

---

## 2026-11-12 — P1 core CI

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding next make sure you always following the agents.md

**Purpose:** add the core CI workflow after Docker: lint, typecheck, tests with
coverage, and CPU/GPU image builds. The trimmed e2e fixture and RTSP smoke stay
in the next P1 branch per `docs/DEVOPS.md`.

**What the model did:**

1. Synced `main`, created `ci/core-workflow`.
2. Installed ruff/pytest-cov, added the ruff/coverage config, applied safe
   fixes and formatted 26 Python files.
3. Added `.github/workflows/ci.yml`: ruff, non-blocking mypy, pytest at an 85%
   coverage gate with JUnit/coverage artifacts, and CPU/GPU Buildx builds with
   GitHub Actions cache.
4. Added the CI badge and sections to README, and updated PLAN/DEVOPS/FEATURES.

**Problems solved:**

- Ruff initially found 58 issues and 27 unformatted files; after config, safe
  fixes and formatting it is clean. One unused `setup` local was removed by
  hand.
- Coverage is 92.72%, so the gate is set to 85% with headroom.

**Validated:** `ruff check` clean; `ruff format --check` clean; **216 passed**;
coverage **92.72%**; workflow YAML parses.

**Remaining:** e2e fixture + MediaMTX/RTSP smoke, PR-hygiene, security/release.
GPU image build still needs a green CI run.

---

## 2026-11-12 — P1 e2e fixture and RTSP smoke

### Prompt

> successfully merged the pr, deleted the branch in remote, along with a pr raised by dependencebot. start proceeding next make sure you always following the agents.md

**Purpose:** complete the P1 e2e branch: a committed trimmed fixture, DB/image/
log verification, and a real MediaMTX RTSP smoke test in CI.

**What the model did:**

1. Trimmed the first front-entry clip to a 3.04 s / 76-frame, 603 KB fixture.
2. Added isolated file and RTSP configs, expected-count files and a
   `tests/e2e/verify_outputs.py` that runs inside the app container.
3. Added `tests/e2e/run_e2e.sh`: file run twice (re-ID), then MediaMTX plus an
   ffmpeg publisher looping the fixture, with a 300-frame RTSP run and exits
   flushed.
4. Added `init-dirs` to compose so the non-root app can write bind-mounted
   data/logs, plus a root cleanup helper in the e2e script.
5. Added the `e2e` CI job, wired after `build`, with CPU image cache and
   artifact upload.
6. Updated README/PLAN/DEVOPS/FEATURES and the e2e README.

**Problems solved:**

- The first RTSP window of 150 frames ended before the track confirmed; 300
  frames gives two complete visits.
- `jrottenberg/ffmpeg` uses ffmpeg as its entrypoint, so `sh -c` was invalid;
  the override now passes ffmpeg arguments directly.
- Bind-mounted data/logs are owned by UID 10001; `init-dirs` chowns them and a
  root helper cleans previous runs. The verifier runs inside the app container
  so SQLite integrity checks have the same permissions as the writer.
- `docker compose run` in Compose 5.7 has no `--no-build`; removed it.

**Validated:** file e2e 1 unique / 2 visits / 2 entry+exit / 0 open; RTSP e2e
1 unique / 2 visits / 2 entry+exit / 0 open; ruff/format clean; 216 tests; compose
and YAML valid.

**Remaining:** PR-hygiene/security workflows and P2 API/UI assertions.

---

## 2026-11-12 — P1 PR hygiene and Dependabot

### Prompt

> successfully merged the pr, deleted the branch in remote, i have an doubt if direct commit on main runs 5 workflows is okay, but pr merge also does it making it run twice. start proceeding next make sure you always following the agents.md

**Purpose:** add the P1 PR-hygiene workflow, PR template and grouped
Dependabot updates, and document the intentional duplicate CI run.

**What the model did:**

1. Added `.github/workflows/pr-checks.yml`: conventional PR title check and a
   full-history commit-trailer check.
2. Added `scripts/check_pr_trailers.py` (fails on `Co-authored-by`,
   `Claude-Session` or `Generated with Claude Code`).
3. Added `.github/pull_request_template.md` and grouped weekly Dependabot
   updates for uv, Docker and GitHub Actions.
4. Documented in `ci.yml` that a PR merge runs both the PR head and main merge
   commit; kept both for the review gate and the direct-commit safety net.
5. Updated README/PLAN/DEVOPS/FEATURES.

**Duplicate-run decision:** PR merge → `pull_request` run on the head commit +
`push` run on `main`; direct commit to main → only the push run. Keeping both is
intentional; dropping push would remove the safety net. Caching makes the second
pass cheaper.

**Validated:** trailer script runs clean on the branch range; workflows and
Dependabot YAML parse; ruff clean.

**Remaining:** `ci/security-and-release` and P2 API/UI assertions.

---

## 2026-11-12 — P3 security and release

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding next make sure you always following the agents.md

### Correction prompt

> dont need pull_request_template.md, pr-checks.yml, docker-publish.ynl, nightly.yml so modify accordingly

**Purpose:** add the P3 security and tagged-release automation while keeping
the workflow surface small.

**What the model did:**

1. Added `security.yml`: CodeQL, dependency review, gitleaks, non-blocking
   pip-audit, Trivy filesystem scan on PRs and built-image scan on pushes/
   schedule.
2. Added `release.yml`: tag-triggered versioned GHCR images, generated release
   notes and a `samples.zip` attachment.
3. Kept Dependabot grouped updates.
4. Removed `pr-checks.yml`, `pull_request_template.md`, `docker-publish.yml`,
   `nightly.yml` and the now-unused trailer script per the author.
5. Updated README/PLAN/DEVOPS/FEATURES/AGENTS.

**Validated:** YAML parses for `security.yml` and `release.yml`; no Python/code
changes; e2e/tests remain green from main.

**Remaining:** P2 API/UI (bonus). The public video and form remain the core
submission actions.

---

## 2026-11-12 — P2 F1 API and Streamlit views

### Prompt

> successfully merged the 2 prs, deleted the branch in remote. start proceeding next make sure you always following the agents.md

**Purpose:** start P2 with the read-only API and the first three Streamlit
views, keeping the pipeline completely independent.

**What the model did:**

1. Added `src/api/`: a read-only `ReadModel` over SQLite/`events.log` and a
   FastAPI app with `/health`, `/stats`, `/events`, `/logs/tail`, `/images` and
   redacted `/config`.
2. Added `ui/app.py`: Dashboard, Event Log and System Logs, using `requests`
   only; no DB/pipeline imports.
3. Added API contract tests and a Streamlit `AppTest` boot smoke.
4. Added UI dependencies to the `ui` extra/requirements, `httpx` to dev, and
   switched CI typecheck/test to `--extra ui`.
5. Updated README/PLAN/FRONTEND/FEATURES and added `ui/README.md`.

**Problems solved:**

- `load_config` requires an explicit path; the API defaults to `config.json`.
- API health/event reads use normal SQLite connections; the earlier read-only
  URI caused Linux WAL/integrity issues in CI.
- The UI handles API-down states with `st.error` instead of raising.

**Validated:** 223 tests, 92.21% coverage, ruff/format clean, Streamlit boot
smoke passes with no API running.

**Remaining:** F2 job control/live view/uploads, F3 folder/config, F4 visitors/
system pages; P0 submission video/form still the priority.

---

## 2026-11-12 — P2 F2 jobs, uploads, live view

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding next make sure you always following the agents.md

Previous PR follow-up (PR #22): **“see skipped workflow and comments fix it”** —
corrected the skipped `trivy-image` PR job and confirmed the CodeQL review
threads resolved.

**Purpose:** add the F2 control surface without putting the graded pipeline at
risk.

**What the model did:**

1. Added an optional pipeline `frame_sink`; it copies a frame only when a live
   viewer is attached and never raises into the frame loop.
2. Added `JobManager` (one daemon worker) over `run_pipeline`, with queued/
   running/stopped/failed states, filename-only input paths, file/folder/RTSP
   sources and a stop event.
3. Added upload (2 GiB cap, atomic `.part` rename), `/videos`, `/jobs`,
   `/stream/live.mjpg` and `/events/stream` SSE.
4. Added Streamlit Live Monitor and Sources & Jobs pages; pure `requests`.
5. Added `python-multipart`, API/job/live tests and a frame-sink test.

**Problems solved:**

- UI AppTest timed out because auto-refresh fragments kept scheduling with the
  API down; `FACETRACKER_REFRESH=0` disables refresh for the smoke test.
- Job inputs are filenames inside `src/video`, not arbitrary paths; this is
  sufficient for uploads and keeps path traversal out.
- Live frames are copied only when viewers exist, so the headless pipeline
  pays nothing.

**Validated:** 230 tests, 90.58% coverage, ruff/format clean, Streamlit smoke
passes with the API down.

**Remaining:** F3 folder/config, F4 visitors/system; P0 video/form priority.

---

## 2026-11-12 — P2 F3 folder jobs and runtime config

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding next make sure you always following the agents.md

**Purpose:** add folder processing and the validated config pane without
changing the pipeline's file-owned requirements.

**What the model did:**

1. Added `GET /folders` and recursive (`**/*.mp4`) folder jobs through the
   existing `FileSequenceSource`.
2. Added `ConfigStore`: base `config.json` + persisted
   `data/runtime_overrides.json`, five hot fields, loader-equivalent
   validation, atomic write, reset and explicit save-default.
3. Added `PUT /config`, reset/save-default routes, and `JobManager.update_config`
   so overrides reach the next job only.
4. Added Streamlit folder mode and the Config page (diff, disabled
   `frame_skip`, explicit save confirmation).
5. Added config/folder contract tests.

**Problems solved:**

- Persisted overrides can be stale or cross-field-invalid; startup logs and
  drops them instead of failing the API.
- `frame_skip` is rejected by the override model (extra key), preserving the
  PDF requirement that it originates in `config.json`.
- Save-default is atomic and clears overrides only after the write succeeds.

**Validated:** 235 tests, 90.65% coverage, ruff/format clean, Streamlit smoke
passes with the API down.

**Remaining:** F4 visitors/system/polish; P0 video/form priority.

---

## 2026-11-12 — P2 F4 visitors, system and polish

### Prompt

> successfully merged the pr, deleted the branch in remote. start proceeding next make sure you always following the agents.md

**Purpose:** finish the bonus UI without touching the graded pipeline.

**What the model did:**

1. Added `/visitors` and `/visitors/{face_id}` from faces/visits/events,
   including thumbnails and entry/exit image URLs.
2. Added `/system` with CPU, memory, disk, process RSS, DB size, uptime, ONNX
   providers, model settings and GPU availability (`psutil`).
3. Added the Streamlit Visitors gallery/timeline and System page; Event Log
   CSV export and System Logs download.
4. Added visitor/system contract tests and updated all status docs.

**Problems solved:**

- Visitor queries use event IDs rather than image paths, preserving the
  traversal-safe image endpoint.
- System diagnostics are read-only and never instantiate a model or touch the
  pipeline.
- CSV/download actions stay in the UI; no extra API writes.

**Validated:** 237 tests, 90.81% coverage, ruff/format clean, Streamlit smoke
passes with the API down.

**Remaining:** P0 public video and Google Form; no more P2 build slices.

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
