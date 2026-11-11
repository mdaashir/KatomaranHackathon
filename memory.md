# Memory — Intelligent Face Tracker

Last updated: 2026-11-11

## What was built

- Planning baseline: `docs/PLAN.md`, `FEATURES.md`, `COMPUTE.md`,
  `ARCHITECTURE.md`, `PROMPTS.md`.
- Baseline commit `e309029` on `main`; `origin` is
  `https://github.com/mdaashir/KatomaranHackathon.git`; `main` pushed.
- Phase 1 branch `feature/config-storage-logging` with commits for config,
  storage, logging, and the prompt log.
- `src/config/`: frozen dataclasses, strict JSON loader, `${ENV_VAR}`
  expansion, key-level validation.
- `src/storage/`: SQLite WAL connection, schema, transaction context,
  repositories for faces, embeddings, visits, events.
- `src/logging_setup.py`: rotating console/app/debug logs and the
  pipe-delimited `events.log` formatter.
- `tests/`: 56 passing tests.
- PR #1: https://github.com/mdaashir/KatomaranHackathon/pull/1

## Decisions made

- Use frozen dataclasses rather than Pydantic; reject unknown config keys.
- SQLite WAL with explicit `BEGIN IMMEDIATE`; one writer connection.
- `UNIQUE(visit_id, event_type)` enforces one entry and one exit per visit.
- `VisitRepository` and `EventRepository` enforce anonymous/linked consistency.
- `events.log` is the INFO audit log; per-frame detail goes to `debug.log`
  only at DEBUG.
- Raw `.mp4` datasets are not committed (232 MB of input data); they stay in
  `src/video/` locally and are ignored via `.gitignore`.
- Detection defaults: `imgsz=1600` CPU / 1920 GPU; track at 40 px native,
  register at 60 px; `exit_timeout_frames=75` (~3 s at 25 fps).
- Recognition is per-track (cached), clips are flushed independently, and
  closed visits are never reopened.

## Problems solved

- `git push` timed out on the 232 MB video pack. Rewrote the local baseline
  without the videos, rebased the Phase 1 commits, and pushed a 3.97 MB `main`.
  The 23 video files remain on disk.
- `gh` authentication and the Git credential manager both work; the failure was
  pack size, not credentials.
- `/review` findings fixed: quality range validation, anonymous/linked event
  consistency, negative `track_id` rejection, and quote escaping in log fields.

## Current state

- PR #1 is **OPEN** and awaiting author review/merge.
- Working tree is clean on `feature/config-storage-logging`.
- `pytest`: 56 passed.
- Do **not** start Phase 2 until PR #1 is merged, per `AGENTS.md`.

## Next session starts with

- Ask the author to review/merge PR #1. After merge: `git checkout main`,
  `git pull`, create `feature/video-source`, and implement Phase 2
  (`FileSource`, `FileSequenceSource`, `RTSPSource` with reconnect) per
  `PLAN.md §7`.

## Open questions

- `PLAN.md §10` sign-offs are still unanswered:
  1. Lifecycle-based vs direction-based entry/exit.
  2. Face-only limitation vs optional person-detector fallback.
  3. `record_*` clips as the primary dataset.
  4. 40 px tracking / 60 px registration gates.
  5. Frontend deferred until Phases 0–9.

---

# Memory — Phase 2 append

Last updated: 2026-11-11

## What was built

- Phase 2 branch `feature/video-source`, PR #2 merged.
- `src/video/`: `Frame` contract (`global_index`, `window_id`, `is_window_end`),
  `FileSource` (PyAV, PTS timestamps, final-frame lookahead),
  `FileSequenceSource` (22 clips in order, filename timestamps, corrupt-clip
  skip), `RTSPSource` (bounded drop-oldest queue, capped reconnect, safe stop),
  `BoundedFrameQueue`.
- `tests/test_video.py` plus the ffmpeg `make_video` fixture.
- `requirements.txt` now includes `numpy` and `av`.

## Decisions

- Decoder lifetime belongs to the generator; `FileSource.close()` only sets a
  stop flag because closing PyAV from a suspended generator caused a Windows
  access violation.
- `RTSPSource.close()` stops and joins but never closes the container from the
  consumer thread, for the same thread-safety reason.
- Tracks must flush on `is_window_end` **and** on a `window_id` change, because
  a corrupt clip cannot mark its final frame.
- Filename timestamps are treated as UTC.

## State at end of Phase 2

- 75 tests passed.
- Real clip: 1,502 frames, 59.96 s, 2688×1520, 74.8–76.7 decode fps.
- PR #2 merged at 2026-11-11T17:10:47Z.
- No local RTSP server: kill/restore remains a manual check.

---

# Memory — Phase 3 append

Last updated: 2026-11-11

## What was built

- Phase 3 branch `feature/yolo-face-detector`, PR #3 open.
- `src/detection/`: `FaceDetection`, `DetectorError`, `YoloFaceDetector`
  (lazy Ultralytics import, injectable model factory, native-coordinate
  clamping, 40 px size gate, device mapping).
- Vendored `models/yolov8n-face.pt` (6,247,374 bytes, SHA256
  `6217b2550b792b789b3e42a55453c6c4326d6293d27f1a6ae56da715c9fbc668`).
- `models/README.md` documents source, size, checksum, AGPL-3.0 license,
  pickle trust warning, and the `omegaconf` dependency.
- `tests/test_detector.py` (14 fake-model tests) and
  `tests/test_detector_real.py` (2 real-image tests).
- `requirements.txt` adds `ultralytics>=8.4` and `omegaconf>=2.3`.

## Decisions

- The checkpoint is **detect-only** (`task=detect`, one class `face`) and has no
  keypoint head. `FaceDetection.keypoints` is `None`; Phase 4 must derive
  ArcFace alignment points with InsightFace landmarks.
- Model is vendored for offline interview use. Ultralytics is AGPL-3.0.
- Lazy model loading keeps import/tests fast; first CPU inference includes ~9 s
  of PyTorch warmup.
- CPU latency is materially higher than estimated: 196 ms @1280, 297 ms @1600,
  417 ms @1920 on a 2K frame. `COMPUTE.md` now recommends 1280 + skip 4–6 on
  4-core machines and estimates 47–49 min for the 22-minute dataset at
  1280/skip 3.

## State at end of Phase 3

- 91 tests passed.
- Real 4K crowd: 11–13 faces at 1280/1600/1920.
- Real front entry: 3 faces at 0 s, 4 at 20 s, 1–2 at 40 s.
- PR #3 is open and waiting for review/merge.
- Do not start Phase 4 until PR #3 is merged.

## Next session starts with

- After PR #3 merges: `git checkout main`, pull, delete the merged local branch,
  create `feature/insightface-embedder-registry`, and implement Phase 4:
  quality gate, InsightFace landmark alignment (YOLO has no keypoints),
  ArcFace embeddings, cosine registry, and registration/re-ID tests.

## Open questions still standing

1. Lifecycle-based vs direction-based entry/exit.
2. Face-only limitation vs optional person-detector fallback.
3. `record_*` clips as the primary dataset.
4. 40 px tracking / 60 px registration gates.
5. Frontend deferred until Phases 0–9.
6. Whether Phase 8 needs 2×2 tiling to improve far-face recall.

---

# Memory — Phase 4 append

Last updated: 2026-11-11

## What was built

- Phase 4 branch `feature/insightface-embedder-registry`.
- `src/recognition/`:
  - `quality.py` — size/confidence/blur gate and face crop.
  - `alignment.py` — 106→5 landmark reduction, ArcFace 112×112 warp, pose gate.
  - `providers.py` — CUDA/CPU provider selection and Torch CUDA DLL setup.
  - `runtime.py` — lazy `ArcFaceONNX` + `Landmark` loading.
  - `embedder.py` — landmarks/pose/align/embed chain.
  - `registry.py` — SQLite-backed cosine matrix, register, match, add embedding.
- Config adds `recognition.model_dir` and `recognition.device`.
- Storage adds `FaceRepository.count_embeddings()`.
- `scripts/download_models.py` downloads/verifies the two InsightFace models.
- `scripts/setup_gpu.ps1` installs CUDA Torch, ORT GPU 1.22, and models.
- `models/insightface/README.md` documents source, sizes, hashes, license.
- 36 new tests (127 total), including real GPU validation.

## Decisions made

- Use CUDA for both YOLO and InsightFace. `device="auto"` selects
  `CUDAExecutionProvider`; CPU is the fallback.
- Pin `onnxruntime-gpu==1.22.0` because 1.31 requests CUDA 13 DLLs while Torch
  cu128 ships CUDA 12.8.
- Load InsightFace `ArcFaceONNX` and `Landmark` directly; do not run
  InsightFace's detector on full frames.
- 106→5 mapping: eye centroids `33:43` and `87:97`, nose `86`, mouth corners
  `52` and `61` (validated visually on a real face).
- The 174 MB recognizer is not committed; `scripts/download_models.py` fetches
  and SHA256-verifies it.
- `memory.md` is now tracked in git (removed from `.gitignore`).

## Problems solved

- ORT GPU 1.31 / CUDA 13 vs Torch cu128 / CUDA 12.8 mismatch.
- CPU `onnxruntime` pulled by insightface; setup script removes it.
- `FaceAnalysis` requires a detector; direct model loading avoids it.
- Initial 106→5 indices were wrong; corrected after labeled-crop inspection.
- 4K crowd motion blur fails the 60 blur threshold for most faces; the real
  test picks a quality-passing face. Phase 8 may tune the threshold.

## State at end of Phase 4

- `pytest`: 127 passed.
- GPU providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`.
- GPU latency: landmarks 4.1 ms, ArcFace 12.7 ms; CPU 10.1 ms / 96.6 ms.
- Same-person cosine 0.733/0.701; different person 0.118.
- PR pending creation; waiting for review/merge.
- Do not start Phase 5 until the PR is merged.

## Next session starts with

- After merge: `git checkout main`, pull, delete the merged branch, create
  `feature/tracking-lifecycle`, and implement Phase 5: ByteTrack wrapper,
  `NEW → ACTIVE → LOST → EXITED`, identity-anchored open visits, reacquisition,
  end-of-stream flush, and lifecycle tests.

## Open questions still standing

1. Lifecycle-based vs direction-based entry/exit.
2. Face-only limitation vs optional person-detector fallback.
3. `record_*` clips as the primary dataset.
4. 40 px tracking / 60 px registration gates.
5. Frontend deferred until Phases 0–9.
6. Whether Phase 8 needs 2×2 tiling to improve far-face recall.
7. Whether `blur_threshold=60` is too strict for the 4K crowd clip.

## Phase 4 PR

- PR #4: https://github.com/mdaashir/KatomaranHackathon/pull/4
- State: open, waiting for review/merge. Do not start Phase 5 until merged.

---

# Memory — Phase 5 append

Last updated: 2026-11-11

## What was built

- Phase 5 branch `feature/tracking-lifecycle`.
- `src/tracking/`:
  - `types.py` — `TrackedBox`, `Track`, `Visit`, `VisitEvent`, `TrackState`.
  - `tracker.py` — `ByteTrackAdapter` with `frame_rate = fps / frame_skip`.
  - `lifecycle.py` — `TrackLifecycle` state machine and visit manager.
- Config adds `tracking.track_activation_threshold` and
  `tracking.minimum_matching_threshold`.
- 23 new tests (154 total), including a ByteTrack→lifecycle seam test.
- All model weights are now excluded from git (`models/**/*.pt`, `*.onnx`,
  `*.zip`); `models/yolov8n-face.pt` was removed from the index.
- `scripts/download_models.py` now fetches and verifies both the YOLO
  checkpoint and the two InsightFace models.

## Decisions made

- Pin `supervision==0.25.1`: 0.31 removed the tracker module.
- ByteTrack is updated once per detection cycle and receives
  `frame_rate = fps / frame_skip`.
- Lifecycle is storage-agnostic: returns `VisitEvent` objects with an optional
  sink; Phase 6 owns images/DB/events.log.
- Entry is emitted after `confirm_hits` but timestamped at first sight; exit is
  emitted at timeout but timestamped at last sight.
- `LOST → ACTIVE` absorbs short gaps; a closed visit is never reopened.
- Identity anchoring attaches a new track to an existing open visit when it
  carries the same Face ID. Recognition must run before confirmation;
  `assign_face()` can fill an anonymous open visit later.
- Window changes flush and clear tracks; end-of-stream calls `flush()`.

## Problems solved

- Supervision 0.31 had no `ByteTrack`; downgraded to 0.25.1.
- Model exclusion would break loading; the downloader now fetches YOLO too.
- Remote history still contains the previously committed YOLO checkpoint;
  future commits do not. Purging history was not done because it would rewrite
  the merged remote.

## State at end of Phase 5

- `pytest`: 154 passed.
- ByteTrack: 1.77 ms median / 4.22 ms p95 for 10 boxes.
- PR pending creation; waiting for review/merge.
- Do not start Phase 6 until the PR is merged.

## Next session starts with

- After merge: sync `main`, delete the merged branch, create
  `feature/event-logging-and-counter`, and implement Phase 6: atomic image
  store, `EventLogger` (image → DB transaction → events.log), unique counter,
  and summary/report. Wire `TrackLifecycle` events into the DB.

## Open questions still standing

1. Lifecycle-based vs direction-based entry/exit.
2. Face-only limitation vs optional person-detector fallback.
3. `record_*` clips as the primary dataset.
4. 40 px tracking / 60 px registration gates.
5. Frontend deferred until Phases 0–9.
6. Whether Phase 8 needs 2×2 tiling to improve far-face recall.
7. Whether `blur_threshold=60` is too strict for the 4K crowd clip.

## Phase 5 PR

- PR #5: https://github.com/mdaashir/KatomaranHackathon/pull/5
- State: open, waiting for review/merge. Do not start Phase 6 until merged.
