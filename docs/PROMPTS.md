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
