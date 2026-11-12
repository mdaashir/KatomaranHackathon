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

---

# Memory — Phase 6 append

Last updated: 2026-11-11

## What was built

- Phase 6 branch `feature/event-logging-and-counter`.
- `src/events/`:
  - `image_store.py` — atomic 256 px JPEG writes to dated entry/exit folders;
    black placeholder for a missing crop.
  - `event_logger.py` — image → DB transaction → `events.log`, lifecycle→DB
    visit mapping, image rollback on DB failure, `link_face()`.
- `src/counting/counter.py` — DB-derived `get_unique_count()` and
  `CounterSummary`.
- `src/tracking/types.py` / `lifecycle.py` now carry `video_time_s` and optional
  `similarity` per event.
- Storage adds `create_visit_with_entry`, `close_visit_with_exit`,
  `link_face`, `visit_face`, `count_anonymous`, `time_range`.
- 29 new tests (185 total).

## Decisions made

- `EventLogger` is synchronous in Phase 6; Phase 7 can offload it.
- A lifecycle visit ID maps to a DB visit ID in memory. Image filenames use the
  lifecycle ID; logs carry both `visit` and `lifecycle_visit`.
- Image writes happen before the DB transaction; on failure the image is
  removed. A missing crop writes a black placeholder so no event is lost.
- `link_face()` updates the visit and its entry event. The exit path rejects a
  conflicting Face ID.
- Unique count is always `COUNT(*) FROM faces WHERE is_active=1`; anonymous
  visits are excluded.
- Cross-table persistence lives in `EventRepository`, not in the event logger.

## Problems solved

- Test FK violations required seeding Face IDs; fixtures now seed two faces.
- The merge deleting the previously tracked YOLO model removed it from disk.
  The new downloader restored and verified it, validating the model-exclusion
  workflow.
- Exits were re-logging `VISIT_LINKED`; added `visit_face()` and a conflict
  guard.

## State at end of Phase 6

- `pytest`: 185 passed.
- Entry persistence 4.78 ms median / 6.36 p95.
- Exit persistence 4.76 ms / 5.54 p95.
- PR pending creation; waiting for review/merge.
- Do not start Phase 7 until the PR is merged.

## Next session starts with

- After merge: sync `main`, delete the merged branch, create
  `feature/pipeline-cli`, and wire the composition root: config → video source
  → detector → tracker → lifecycle → recognizer/registry → event logger →
  counter, with frame_skip, window resets, signal handling, flush, and
  `--report`.

## Open questions still standing

1. Lifecycle-based vs direction-based entry/exit.
2. Face-only limitation vs optional person-detector fallback.
3. `record_*` clips as the primary dataset.
4. 40 px tracking / 60 px registration gates.
5. Frontend deferred until Phases 0–9.
6. Whether Phase 8 needs 2×2 tiling to improve far-face recall.
7. Whether `blur_threshold=60` is too strict for the 4K crowd clip.

## Phase 6 PR

- PR #6: https://github.com/mdaashir/KatomaranHackathon/pull/6
- State: open, waiting for review/merge. Do not start Phase 7 until merged.

---

# Memory — Phase 7 append

Last updated: 2026-11-11

## What was built

- Phase 7 branch `feature/pipeline-cli`.
- `src/pipeline/`:
  - `Pipeline` frame loop, `frame_skip`, recognition-before-confirmation,
    per-window flush/reset, crop cache, per-track recognition retry.
  - `CropCache` for exit images after the face leaves the frame.
  - `PipelineResult` summary type.
- `main.py` composition root and CLI: `--config`, `--source`, `--path`,
  `--rtsp-url`, `--start-time`, `--max-frames`, `--log-level`, `--report`,
  `--verify-db`.
- `Frame.fps` added to file/folder/RTSP sources so ByteTrack gets the real
  detection cadence.
- Config loader accepts UTF-8 with or without BOM.
- 22 new tests (208 total).

## Decisions made

- `main.py` builds the source before installing signals; the signal handler
  sets the stop event and closes the source, waking a blocked RTSP queue.
- `run_pipeline` closes the source if pipeline construction fails.
- Recognition runs before `lifecycle.observe`; late recognition uses
  `assign_face()` + `EventLogger.link_face()`.
- Best crop per track is cached for exits; window end/change flushes and
  resets tracker, lifecycle tracks, crop cache, and retry timers.
- `--start-time` overrides single-file mtime; folder mode uses filename
  timestamps; RTSP uses wall clock.
- A failed embedding-cache write is non-fatal.

## Problems solved

- Ctrl+C during a stalled RTSP stream could hang; closing the source wakes the
  queue.
- PowerShell-written configs have a UTF-8 BOM; the loader now reads
  `utf-8-sig`.
- Source cleanup is guaranteed even if pipeline construction fails.

## State at end of Phase 7

- `pytest`: 208 passed.
- End-to-end smoke: 30 frames / 10 cycles, 1 unique, 1 entry, 1 exit, 0 open;
  crops 9.3 KB / 8.8 KB; first event `2025-06-20T18:39:03Z`.
- `--report` matches the DB summary.
- PR pending creation; waiting for review/merge.
- Do not start Phase 8 until the PR is merged.

## Next session starts with

- After merge: sync `main`, delete the merged branch, create
  `test/e2e-and-calibration`, run all 22 clips and `video_sample1.mp4` on a
  fresh DB, inspect duplicates/far-face recall, tune thresholds, and build
  `samples/` (events.log excerpt, images, DB export, summary).

## Open questions still standing

1. Lifecycle-based vs direction-based entry/exit.
2. Face-only limitation vs optional person-detector fallback.
3. `record_*` clips as the primary dataset.
4. 40 px tracking / 60 px registration gates.
5. Frontend deferred until Phases 0–9.
6. Whether Phase 8 needs 2×2 tiling to improve far-face recall.
7. Whether `blur_threshold=60` is too strict for the 4K crowd clip.

## Phase 7 PR

- PR #7: https://github.com/mdaashir/KatomaranHackathon/pull/7
- State: open, waiting for review/merge. Do not start Phase 8 until merged.

---

## Phase 8 — end-to-end validation and samples (2026-11-11)

- Branch: `test/e2e-and-calibration`.
- Ran all 22 `record_*.mp4` clips on a fresh DB (`data/e2e.db`,
  `logs/e2e`, `imgsz=1600`, `frame_skip=3`): 33,044 frames, 11,022 detection
  cycles, 26 unique faces, 59 visits, 33 anonymous, 59 entries, 59 exits,
  0 open.
- One cross-clip re-identification: Face 13 in visits 29 and 51; unique count
  counts it once.
- Added `scripts/verify_outputs.py`: integrity OK, 0 FK violations, 0 missing
  event/visit images, exactly two events per visit, 0 open.
- Added `scripts/export_samples.py`: `samples/` has 24 images (entries, exits,
  re-entry), `faces.csv`, `visits.csv`, `events.csv`, `events.log` excerpt,
  `summary.json`, `README.md`.
- 4K `video_sample1.mp4` stress: 240 frames / 80 cycles, 32 visits, 0
  registrations (motion blur); 105.3 s wall.
- One transient GPU stall after 21 clips (CPU idle, GPU spinning); a fresh
  process finished the last clips in 105.8 s + 194.3 s. `nvidia-smi` later
  showed 510 MHz / 13.5 W. Recorded as PLAN R13; mitigation is segment restarts.
- Fixed `CropCache` to prefer quality-passing crops over larger fallback crops;
  added two tests. The existing sample's re-entry images predate the fix and are
  back-of-head views; the README explains the DB is authoritative.
- `pytest`: 210 passed.
- Updated PLAN, COMPUTE, PROMPTS, README, memory.
- PR #8: https://github.com/mdaashir/KatomaranHackathon/pull/8
- State: open, waiting for review/merge. Do not start Phase 9 until merged.

## Next session starts with

- After merge: sync `main`, delete `test/e2e-and-calibration`, start Phase 9
  (`docs/readme-and-demo`): final README, Loom/YouTube walkthrough, repository
  link and submission form. Keep the mandatory Katomaran line last.

---

## Phase 9 — README, demo and submission (2026-11-11)

- Branch: `docs/readme-and-demo`.
- Rewrote `README.md`: final status, GPU/CPU setup, fresh-clone path, run
  commands, output layout, Mermaid architecture diagram, sample config,
  measured results, samples, tests, business rules, limitations, submission
  links and the mandatory Katomaran line.
- Added `docs/DEMO.md` (4–5 minute recording script with exact commands),
  `docs/demo_config.json` (one clip, isolated `data/demo.db`/`logs/demo`) and
  `docs/SUBMISSION.md` (checklist and video placeholder).
- Validated the demo command: 300 frames / 100 cycles / 50.2 s, 1 unique,
  1 visit, 1 entry, 1 exit, 0 open; `verify_outputs.py` clean. `pytest`: 210.
- The only remaining manual work is recording the public Loom/YouTube video
  and replacing `VIDEO_URL` in `README.md` and `docs/SUBMISSION.md`, then
  submitting through the Katomaran form.
- PR #9: https://github.com/mdaashir/KatomaranHackathon/pull/9
- State: open, waiting for review/merge. The public video link is still the only
  blocker before submission.

## Next session starts with

- After merge: sync `main`, delete `docs/readme-and-demo`, and either record the
  demo or start the optional Phase 10 dashboard only after the submission is
  complete. Do not add a dashboard before the core submission is final.

---

## Phase 10 — optional read-only dashboard (2026-11-11)

- Branch: `feature/dashboard`.
- Added `src/dashboard/` (`DashboardBuilder`, pure `render_dashboard`) and
  `scripts/dashboard.py` (`--config`, `--db`, `--output`, `--max-events`,
  `--max-visits`, `--open`). It only reads SQLite, copies event thumbnails and
  writes a standalone HTML snapshot; it never imports the pipeline.
- Added six tests (`tests/test_dashboard.py`); total `216 passed`.
- Generated `samples/dashboard/index.html` from `data/e2e.db`: 26 unique,
  59 visits, 24 recent events, 24 thumbnails. `samples/README.md` links it.
- Updated README, PLAN, FEATURES, ARCHITECTURE, PROMPTS and `.gitignore`
  (`/dashboard/`).
- PR #10: https://github.com/mdaashir/KatomaranHackathon/pull/10
- State: open, waiting for review/merge. The public demo video link remains the
  only blocker before the actual submission.

## Next session starts with

- After merge: sync `main`, delete `feature/dashboard`, record the public demo
  (still required), replace `VIDEO_URL`, and submit. Do not start another
  feature before the submission video is live.

---

## Compliance audit against the problem statement (2026-11-11)

- Branch: `docs/compliance-audit`.
- Extracted the four-page PDF with a scratchpad-only `pypdf` install; created
  `docs/COMPLIANCE.md` with all 43 requirements mapped to evidence.
- Re-ran `pytest`: 216 passed. `verify_outputs.py --db data/e2e.db`: clean
  (26 unique, 59 visits, 59/59 entries/exits, 0 open, 0 missing images).
- All functional requirements are done. Real-time throughput is a documented
  performance caveat (~21 fps against 25 fps in the long run; tune imgsz/frame
  skip on interview hardware). The only missing deliverable is the public demo
  video; the Google Form submission is the only other author action. Real RTSP
  kill/restore remains a hardware confirmation, not a code gap.
- Updated README, PLAN (resolved sign-off defaults), FEATURES, COMPUTE,
  ARCHITECTURE, DEMO, SUBMISSION, sample README, PROMPTS and this memory.
- PR #11: https://github.com/mdaashir/KatomaranHackathon/pull/11
- State: open, waiting for review/merge. The public video and Google Form remain
  the only submission blockers; interview-hardware fps tuning is the only
  performance follow-up.

## Next session starts with

- After merge: record the public video, replace `VIDEO_URL`, verify the link in
  a private browser, submit the repository through the Katomaran form, and keep
  the exact mandatory footer line last. Do not start another feature before the
  submission is submitted.

---

## P1 Docker and uv packaging (2026-11-12)

- Branch: `chore/dockerfile-and-compose`.
- Added the multi-stage CPU/GPU Dockerfile, root `entrypoint.sh`,
  `docker-compose.yml`, `docker-compose.gpu.yml`; compose mounts `src/video`
  and uses the single `config.json`.
- Dependencies moved to `pyproject.toml` (`cpu`/`gpu`/`ui` extras, `dev` group)
  with `uv.lock`; `requirements.txt` is the compact pip fallback. `uv init`
  added `.python-version`; `uv run --no-sync` is used after setup so the CUDA
  Torch wheel is not replaced by a sync.
- Removed `inputs/`, `config.docker.json`, `.env.example`, `requirements/` and
  `Makefile` per author corrections.
- CPU image **2.23 GB**; clean 30-frame container run 1 unique / 1 entry /
  1 exit / 0 open in 29.5 s; first-start model download 101.8 s (181 MB);
  compose config and entrypoint verified.
- `pytest`: 216 passed; e2e verifier clean.
- GPU image build is still pending; `ci/core-workflow`, e2e fixture/RTSP smoke,
  PR checks and release/security follow in separate branches.
- PR #12: https://github.com/mdaashir/KatomaranHackathon/pull/12
- State: open, waiting for review/merge. The CPU image is validated; GPU build
  and core CI/e2e are the next P1 branches.

## Next session starts with

- After merge: sync `main`, delete `chore/dockerfile-and-compose`, then start
  `ci/core-workflow` (lint, typecheck, tests, CPU/GPU builds), followed by the
  trimmed e2e fixture and RTSP smoke. Do not start P2/UI before P1 CI is green.

---

## P1 core CI (2026-11-12)

- Branch: `ci/core-workflow`.
- Added `.github/workflows/ci.yml`: ruff check/format, non-blocking mypy,
  pytest with an 85% coverage gate and JUnit/coverage artifacts, and CPU/GPU
  Buildx image builds with GHA cache. Triggers PRs, `main`, manual dispatch and
  `workflow_call`; least-privilege permissions; PR runs cancel superseded work.
- Added ruff/coverage config to `pyproject.toml`; applied safe ruff fixes and
  formatted 26 Python files; removed one unused test local.
- Validation: ruff clean, format clean, **216 passed**, coverage **92.72%**,
  workflow YAML parses.
- README/PLAN/DEVOPS/FEATURES updated with the CI badge/status.
- e2e fixture/RTSP smoke, PR checks, security/release remain.
- PR #13: https://github.com/mdaashir/KatomaranHackathon/pull/13
- State: open, waiting for review/merge. Core CI is in place; e2e/RTSP,
  PR-hygiene and security/release are the next branches.

## Next session starts with

- After merge: sync `main`, delete `ci/core-workflow`, then create
  `test/e2e-fixture-and-verify`: trim a small clip, add
  `tests/e2e/verify_outputs.py`, MediaMTX/RTSP smoke and the e2e job. Do not
  start P2/UI before P1 e2e is green.

---

## P1 e2e fixture and RTSP smoke (2026-11-12)

- Branch: `test/e2e-fixture-and-verify`.
- Fixture: `tests/fixtures/entry_exit_3s.mp4` — 3.04 s, 76 frames, 603 KB.
- Added isolated `tests/e2e/config.file.json` and `config.rtsp.json`, expected
  counts, and a read-only `verify_outputs.py` (counts, integrity, FK, two events
  per visit, images, required log events).
- Added `tests/e2e/run_e2e.sh`: file fixture twice (re-ID) then MediaMTX + ffmpeg
  publisher, 300-frame RTSP run, flush check, artifact JSON.
- Added compose `init-dirs` (chown bind mounts for UID 10001) and a root cleanup
  helper; verifier opens SQLite read-only so CI does not need write access.
- Added CI `e2e` job after `build`, with CPU image cache and artifacts.
- Validated: file 1 unique / 2 visits / 2 entry+exit / 0 open; RTSP 1 unique /
  2 visits / 2 entry+exit / 0 open; ruff/format clean; 216 tests; compose/YAML
  valid.
- PR pending creation in this phase; stop and wait for review/merge.

## Next session starts with

- After merge: sync `main`, delete `test/e2e-fixture-and-verify`, then create
  `ci/pr-checks` (PR title/trailer checks, template, Dependabot) followed by
  `ci/security-and-release`. Do not start P2/UI before P1 checks are green.
