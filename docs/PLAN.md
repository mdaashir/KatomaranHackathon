# PLAN — Intelligent Face Tracker with Auto-Registration and Visitor Counting

> Status: **final implementation; compliance audit complete** (2026-11-11)
> Deadline: **2:00 PM, Friday 13 November 2026**
> Remaining author actions: public demo video and Google Form submission.
> Interview-hardware fps tuning is the remaining validation step.
> Sources of truth: `docs/Katomarans Problem Statement.pdf`, `AGENTS.md`
> Local data: `src/video/`

---

## 1. Objective

Build a real-time Python pipeline that consumes a video file (development) or an
RTSP stream (interview), detects faces with YOLO, embeds them with InsightFace
(ArcFace), tracks them with ByteTrack, automatically registers every new face
with a stable ID, logs exactly one entry and one exit per visit (image +
timestamp + face ID), and exposes an accurate unique-visitor count derived from
SQLite.

Grading is driven by four artefacts: `events.log`, the cropped entry/exit
images, the database rows, and the final unique count. Every design decision
below serves those four artefacts.

---

## 2. What the local datasets actually are

I probed every file in `src/video` before planning.

| Dataset | Files | Resolution | Codec / FPS | Duration | Content |
|---|---|---|---|---|---|
| `record_20250620_*.mp4` | 22 | 2688 × 1520 | HEVC / 25 | 60 s each (≈22 min total) | Fixed oblique camera labelled **“BBB MALL FRONT ENTRY”**. Sparse traffic, 0–6 people per clip, people walking toward and away from camera, two seated people on the right edge. |
| `video_sample1.mp4` | 1 | 3840 × 2160 | H.264 / 29.97 | 8.0 s | Crowded top-down bazaar floor, 25–40 visible faces, constant motion. |

Consequences that shape the design:

1. **The 22 `record_*` clips are the primary dataset.** They are the only
   footage with genuine entry/exit semantics. `video_sample1.mp4` is the
   stress test for crowded scenes and throughput.
2. **The clips are separate observation windows, not a continuous video.**
   Filenames encode start time (`record_YYYYMMDD_HHMMSS`). Most clips start
   ~61 s apart, but there are real gaps (e.g. 18:52:10 → 19:03:24). Tracks are
   therefore flushed at each file boundary; a person reappearing in a later
   clip reuses the registered Face ID (unique count does **not** increment) but
   starts a new visit and logs a new entry.
3. **Face size varies enormously.** Faces near the camera are ~70–130 px; faces
   at the far end of the corridor are ~25–45 px. A single downscale to 1280
   wide would erase the far faces. Detection therefore runs at
   `imgsz=1600` on CPU / `1920` on GPU for ≥2K sources, and there are two
   separate size gates:
   `detector.min_face_size = 40` (track it) and
   `recognition.min_face_size = 60` (register/embed it).
4. **Many visitors are back-facing.** The camera is behind people entering the
   mall, so a large fraction of “visitors” never show a usable face. A
   face-only detector cannot count them. This is a documented limitation, with
   an optional person-detector fallback listed as a later phase.

---

## 3. Scope

### In scope (must ship)

- File source (single file **and** ordered folder of clips) + RTSP source with
  reconnect.
- YOLO face detection on a configurable frame cadence (`frame_skip`).
- InsightFace/ArcFace embeddings and cosine re-identification.
- Automatic registration of unmatched, sufficiently good faces.
- ByteTrack tracking + per-track state machine (`NEW → ACTIVE → LOST →
  EXITED`).
- Exactly one entry and one exit per visit, with cropped image, timestamp,
  event type and Face ID.
- `logs/events.log`, structured image folders, SQLite WAL persistence.
- `get_unique_count()` derived from the DB.
- Graceful SIGINT/SIGTERM shutdown with active-track flush.
- Tests for registry, lifecycle and counter.
- Sample outputs in `samples/`, README and architecture docs. The public demo
  video is the remaining author action.

### Out of scope (unless time remains)

- Frontend dashboard: optional; completed as a read-only snapshot in Phase 10.
- Multi-camera identity fusion.
- Cloud sync, auth, multi-tenant storage.
- Person/body re-identification (optional enhancement, see R4).

---

## 4. Shared vocabulary

These terms are used consistently in code, schema, logs and docs. **Confirm or
correct these before implementation starts.**

| Term | Meaning |
|---|---|
| **Face ID** | Persistent identity of a person in the registry (`faces.face_id`). Assigned once on first successful registration; reused by recognition forever. This is what the unique counter counts. |
| **Track ID** | Temporary ID from ByteTrack for one continuous detection run. Can change after occlusion; never used for counting. |
| **Visit** | One continuous observed presence of a person in the stream, from confirmed appearance to exit (`visits.visit_id`). A visit has exactly one entry event and one exit event. A reappearance after the exit timeout starts a new visit that reuses the same Face ID, so the unique count is unaffected. |
| **Entry** | The moment a track is confirmed (`confirm_hits` detections) after appearing. Logged once per visit, timestamped at first sight. |
| **Exit** | The moment a track has been missing for `exit_timeout_frames` source frames, or the stream/source ends. Logged once per visit, timestamped at last sight. |
| **Detection cycle** | A source frame on which detection runs (`frame_index % frame_skip == 0`). At 25 fps and `frame_skip = 3` this is 8.33 cycles/s. |
| **Anonymous visit** | A confirmed visit whose face never passed the registration quality gate. It gets entry/exit events and images, but **no Face ID** and is excluded from the unique-visitor count. |
| **Registered (unique) visitor** | A row in `faces`. Unique count = `COUNT(*) FROM faces WHERE is_active = 1`. Re-identification never creates a new row. |
| **Reacquisition** | A `LOST` track reappearing before `exit_timeout_frames` returns to `ACTIVE`; it is the same visit, with no second entry. |

---

## 5. Decisions and rationale

| # | Decision | Rationale |
|---|---|---|
| D1 | **Primary input = ordered `record_*.mp4` sequence; `video_sample1.mp4` = stress test.** | Only the record clips contain real front-entry traffic. The 4K clip validates crowded-scene throughput. |
| D2 | **Each clip is an independent observation window; flush tracks at file end.** | Filename timestamps prove real gaps between recordings. Flushing prevents cross-gap “ghost” tracks; the unique count is unaffected because a reappearance reuses the Face ID. |
| D3 | **Detect at `imgsz=1600` on CPU and `1920` on GPU for ≥2K sources; never below 1280.** | Far faces are 25–45 px; downscaling to 640 destroys them. 1920 keeps far faces detectable but is not real-time on CPU, so CPU uses 1600 and GPU uses 1920. |
| D4 | **Two size gates:** `detector.min_face_size = 40` (track) and `recognition.min_face_size = 60` (embed/register). | You want to track a far face, but not pollute the registry with a 30 px embedding. Registration happens when the person walks closer. |
| D5 | **Recognition is per-track, not per-face-per-frame.** Embed once on confirmation, retry only while unregistered (throttled), then stop. | A crowded 4K frame with 30 faces would otherwise spend ~300 ms/cycle on ArcFace. Caching makes throughput detection-bound. |
| D6 | **Identity-anchored visits.** Visits are keyed by Face ID when known; a new track matching a Face ID with an **open** visit attaches to it instead of creating a second visit. | ByteTrack has no appearance model and can split one person into two tracks. Identity anchoring prevents duplicate entry/exit pairs for the same continuous presence. Closed visits are never reopened. |
| D7 | **Entry is emitted only after `confirm_hits` (default 3) detections.** | Prevents one-frame false positives from creating registry rows and entry events. |
| D8 | **`exit_timeout_frames` is measured in source frames; recommended 75 (3.0 s @ 25 fps), not the 30 in the sample config.** | 30 frames = 1.2 s; the face detector drops frames far more often than that, which would make tracks flap and produce duplicate visits. A gap shorter than the timeout is absorbed by `LOST → ACTIVE`; a longer gap is a real exit followed by a new visit with the same Face ID. The key name is kept for compatibility; the value and semantics are documented. |
| D9 | **SQLite WAL + one transaction per event.** Entry/exit image is written atomically (temp → `os.replace`) before the DB row. | Meets “resilient to unexpected interruptions”; no half-written JPEGs, no orphan DB rows. |
| D10 | **`events.log` is an INFO-level audit log; per-frame detection noise goes to a separate DEBUG log.** | Logging every face on every cycle would generate ~400 MB/day on a live stream. `FACE_DETECTED` is logged on track start/reacquire, which satisfies the mandatory events without log explosion. Rotating handler caps size. |
| D11 | **Anonymous visits are logged and counted separately, never in `unique_visitors`.** | Honest counting. A low-quality face is still a visit with images and events, but it cannot be claimed as a unique identity. |
| D12 | **ByteTrack is updated at detection cadence, not every frame.** Set its `frame_rate = fps / frame_skip` and give it `lost_track_buffer` in cycles. | Keeps its Kalman/`lost` logic consistent with the actual update rate. |
| D13 | **PyAV for decoding (OpenCV as fallback).** | Reliable HEVC and RTSP option handling (`rtsp_transport=tcp`, timeouts, low-delay), plus real timestamps. The clips are HEVC, which not every OpenCV wheel decodes. |
| D14 | **Offline files process as fast as possible; RTSP runs real-time with a bounded frame queue that drops stale frames.** | Files should finish quickly; a live camera must never build latency. |

---

## 6. System at a glance

```
VideoSource (file sequence | RTSP with reconnect)
   -> Pipeline (frame loop, frame_skip)
      -> YoloFaceDetector -> ByteTrack -> Lifecycle/VisitManager
            -> QualityGate -> Aligner -> ArcFaceEmbedder -> FaceRegistry
      -> EventLogger (atomic image + SQLite transaction + events.log)
   -> UniqueCounter (reads SQLite)
```

Full component, sequence, state and ER diagrams are in
[`ARCHITECTURE.md`](./ARCHITECTURE.md). Feature-by-feature acceptance criteria
are in [`FEATURES.md`](./FEATURES.md). Compute estimates are in
[`COMPUTE.md`](./COMPUTE.md).

---

## 7. Build phases (one branch + one PR per phase)

Per `AGENTS.md`: branch from fresh `main`, commit in logical steps, run tests
and a real-video check, open a PR, then **stop and wait for the author to
merge** before starting the next phase.

### Phase 0 — Docs and decisions (this branch: `docs/plan-features-compute-architecture`)

Deliverables: `docs/PLAN.md`, `docs/FEATURES.md`, `docs/COMPUTE.md`,
`docs/ARCHITECTURE.md`, updated `docs/PROMPTS.md`.

Acceptance: documents are internally consistent, diagrams render, author signs
off on the vocabulary and the five open questions in §10.

### Phase 1 — Foundations (`feature/config-storage-logging`)

- `src/config/` typed dataclasses + JSON loader + validation with actionable
  errors.
- `src/storage/` SQLite connection (WAL, foreign keys, busy timeout), schema,
  repositories (`FaceRepository`, `EmbeddingRepository`, `VisitRepository`,
  `EventRepository`).
- `src/logging_setup.py` rotating console/file logging.
- `tests/test_config.py`, `tests/test_storage.py`.

Acceptance: DB is created from schema, CRUD round-trips, invalid config fails
with a clear message, WAL mode confirmed.

### Phase 2 — Video sources (`feature/video-source`)

- `VideoSource` protocol returning `Frame(index, timestamp_s, image, source_id)`.
- `FileSource` (one file) and `FileSequenceSource` (ordered folder, filename
  timestamp parsing, per-file flush signal).
- `RTSPSource` with TCP transport, low-delay, bounded queue, reconnect with
  exponential backoff, `SOURCE_LOST` / `SOURCE_RECONNECTED` events.
- Graceful `close()`.

Acceptance: reads all 22 clips in order with correct timestamps; survives a
killed/restored local RTSP loop; never blocks the pipeline indefinitely.

### Phase 3 — YOLO face detector (`feature/yolo-face-detector`)

- `ultralytics` wrapper, model path from config, `imgsz` and confidence from
  config.
- Returns `FaceDetection(bbox_native, confidence, keypoints)`, boxes mapped
  back to native coordinates. `keypoints` is optional: the bundled
  `yolov8n-face` checkpoint is detect-only and returns `None`.
- Size filtering and NMS already handled by Ultralytics; our gate uses native
  pixels.
- `tests/test_detector.py` with a stubbed model + real-image smoke tests
  (skipped if weights are absent).

Acceptance: detects near and mid-range faces on record frames and at least 5
faces on the 4K crowd frame; far-face recall is measured and reported;
per-frame latency recorded in `COMPUTE.md`.

Measured (Phase 3, CPU): 3 faces at front-entry 0 s, 4 at 20 s, 1–2 at 40 s
depending on `imgsz`, and 11–13 on the 4K crowd. The checkpoint has no keypoint
head, so Phase 4 derives alignment points with InsightFace landmarks.

### Phase 4 — Embeddings and registry (`feature/insightface-embedder-registry`)

- 5-point similarity-transform alignment to the ArcFace 112×112 template,
  using YOLO keypoints when present and InsightFace landmarks otherwise.
- InsightFace `buffalo_l` recognition model loaded directly
  (`model_zoo.ArcFaceONNX`), so detection is **not** run twice. The 106-point
  landmark model supplies the five points when YOLO has no keypoint head.
- CUDA first: `recognition.device="auto"` selects `CUDAExecutionProvider` when
  available and falls back to CPU.
- L2-normalised 512-d embeddings; cosine match against an in-memory matrix
  loaded from SQLite.
- Registration with quality gate; up to `max_embeddings_per_face` embeddings
  per identity.
- `tests/test_registry.py`: match above threshold, reject below, register once,
  re-identification does not create a second face.

Acceptance: same face across different frames matches; a different person does
not; registry reload from DB reproduces matches.

Measured (Phase 4, RTX 3050 6 GB, CUDA 12.8): 106 landmarks **4.1 ms** and
ArcFace **12.7 ms** per face on GPU; CPU fallback is **10.1 ms** and
**96.6 ms**. Same-person cosine is **0.733 / 0.701** at 0.4 s / 0.8 s; a
different person scores **0.118**. The 106→5 mapping uses eye centroids
33:43 and 87:97, nose 86, and mouth corners 52 and 61.

### Phase 5 — Tracking and visit lifecycle (`feature/tracking-lifecycle`)

- ByteTrack wrapper (`supervision` or `boxmot`) configured for detection
  cadence.
- `TrackState` machine + `VisitManager` implementing D6–D8.
- Temporary-loss handling (`LOST → ACTIVE`) so a short detection gap does not
  create a second visit.
- End-of-stream and shutdown flush every active/lost track as an exit.
- `tests/test_lifecycle.py`: exactly one entry/exit per visit, timeout,
  reacquisition, flush on shutdown, anonymous visit handling.

Acceptance: state-machine tests pass; a scripted synthetic track produces one
entry and one exit; a 2 s detector gap does not create a new visit; a 4 s gap
creates one exit and a later reappearance creates a new visit with the same
Face ID.

Implemented with `supervision==0.25.1` ByteTrack (0.31 removed the tracker
module). `TrackLifecycle` is storage-agnostic: it returns `VisitEvent` objects
and supports an optional sink, so Phase 6 can map events to SQLite/images/logs.
Identity anchoring works at confirmation: if a new track already has the same
Face ID as an open visit, it attaches to that visit instead of emitting a
second entry. Recognition must run before confirmation to get the right ID;
`assign_face()` can later fill an anonymous open visit, and Phase 6 updates the
stored entry row. Measured: **1.8 ms median** for a 10-box ByteTrack update;
**154 tests pass**.

### Phase 6 — Event logging and counting (`feature/event-logging-and-counter`)

- `ImageStore` atomic writes to `logs/entries|exits/YYYY-MM-DD/`.
- `EventLogger` writes image → DB transaction → `events.log` line.
- Filenames carry timestamp, face ID, visit ID and track ID.
- `UniqueCounter.get_unique_count()` + `get_summary()` (unique, visits, events,
  anonymous).
- `tests/test_event_logger.py`, `tests/test_counter.py`.

Acceptance: DB `UNIQUE(visit_id, event_type)` rejects duplicates; images are
never truncated; counter returns DB-derived values after restart; re-ID does not
increment.

Implemented with a synchronous `EventLogger`: lifecycle visit IDs map to DB
visit IDs in memory. The entry image filename uses the lifecycle visit ID; the
audit line carries both `visit` and `lifecycle_visit` for traceability. Images
are written atomically first, then the visit+event transaction; a DB failure
removes the image. A missing crop produces a black placeholder so an event is
never lost. `link_face()` updates both the visit and its entry event when
recognition happens after entry. Measured: **4.8 ms median per entry/exit**
(256 px JPEG + SQLite transaction); **185 tests pass**.

### Phase 7 — Pipeline and CLI (`feature/pipeline-cli`)

- `main.py` composition root: load config, wire dependencies, install signal
  handlers, run, flush, report.
- CLI overrides: `--config`, `--source`, `--path`, `--rtsp-url`,
  `--max-frames`, `--log-level`, `--report`.
- Graceful SIGINT/SIGTERM; summary written to log and stdout.

Acceptance: `python main.py --config config.json` runs a clip end-to-end with
no unhandled exception; Ctrl+C produces exit events for active tracks and a
summary.

Implemented: `main.py` is the composition root; the pipeline creates ByteTrack
lazily from `frame.fps`, recognizes before confirmation, caches the best crop
per track for exits, and flushes/resets on `is_window_end` or a window change.
SIGINT/SIGTERM set the stop event and close the source so an RTSP stall cannot
block shutdown. `--start-time` overrides a single-file start, and the config
loader accepts UTF-8 with or without BOM. Measured: the 30-frame smoke run
processed **30 frames / 10 detection cycles**, produced **1 unique, 1 entry,
1 exit, 0 open**, and wrote 9.3 KB / 8.8 KB crops. **208 tests pass.**

### Phase 8 — End-to-end validation and tuning (`test/e2e-and-calibration`)

- Run all 22 record clips in sequence; run `video_sample1.mp4`.
- Inspect entry/exit balance, duplicate registrations, missed far faces.
- Calibrate `similarity_threshold`, `min_face_size`, `exit_timeout_frames`.
- Benchmark on the target CPU (and GPU if available); replace estimates in
  `COMPUTE.md` with measured numbers.
- Produce `samples/`: `events.log` excerpt, 10–20 images, DB CSV/JSON export,
  `summary.json`.

Acceptance: every visit has exactly one entry and one exit; unique count is
stable across a re-run from an empty DB; sampled identities are correct by
visual inspection; measured throughput close to real-time on GPU. On this
laptop the long run averaged ~21 fps against a 25 fps source; one clip is near
1.0× excluding model load. Tuning `imgsz`/`frame_skip` remains an
interview-hardware step.

**Measured (2026-11-11, RTX 3050 6 GB, `imgsz=1600`, `frame_skip=3`):**
22 clips / 33,044 frames produced 26 unique faces, 59 visits (33 anonymous),
118 events and 0 open visits. `scripts/verify_outputs.py` reported integrity
OK, 0 foreign-key violations, 0 missing images and 59/59 entries/exits. Face 13
reappeared in a second clip (visits 29 and 51) and was counted once. One clip
took 83.5 s including model load; the useful full-sequence time was ~27 min,
with one transient GPU stall resumed in a fresh process. The 4K stress clip
produced 32 visits and 0 registrations (motion blur). Calibration kept
`similarity_threshold=0.45`, `min_face_size=60` and `exit_timeout_frames=75`.
`samples/` contains 24 crops, CSV exports, `events.log`, `summary.json` and a
README. The 2×2 tiling option was not needed for the primary 2K clips and was
left out to keep the frame budget.

### Phase 9 — README, demo video, submission (`docs/readme-and-demo`)

- README: setup (`uv` recommended), assumptions, sample `config.json`,
  architecture link, limitations, demo video link, mandatory Katomaran line.
- Record the Loom/YouTube walkthrough.
- Verify the GitHub repo link and submission form.

Acceptance: a fresh clone can run the sample in ≤10 minutes following the
README; video link is public and plays.

**Status (2026-11-11):** README, `docs/DEMO.md`, `docs/SUBMISSION.md` and
`docs/demo_config.json` are complete. The 300-frame demo command was re-run:
1 unique / 1 visit / 1 entry / 1 exit / 0 open, and `verify_outputs.py` is
clean. Public video, `VIDEO_URL` replacement and the Google Form are author
actions; fps tuning on the interview hardware remains the only performance
follow-up.

### Phase 10 — Optional dashboard (`feature/dashboard`) — only after Phases 0–9

Small read-only page or CLI report showing current unique count, recent
entries/exits, and thumbnails. Must not touch the core pipeline.

**Status (2026-11-13):** superseded by the full P2 FastAPI/Streamlit console;
the standalone `src/dashboard`, `scripts/dashboard.py`, static sample and test
artifacts were removed. The core pipeline is untouched.

### Phase 11 — P1 Docker and core CI

- Multi-stage CPU/GPU Dockerfile, compose stack, root `entrypoint.sh`,
  `pyproject.toml` + `uv.lock` groups, and a `requirements.txt` fallback.
- `ci/core-workflow` (next branch): lint, typecheck, tests, image builds.
- `test/e2e-fixture-and-verify` (next branch): trimmed clip, compose e2e,
  RTSP smoke test.
- `ci/security-and-release` (this branch): `security.yml`, `release.yml`,
  Dependabot.

**Status (2026-11-12):** Docker/compose, core `ci.yml`, file+RTSP e2e and P3
security (`security.yml`) + tagged release (`release.yml`) are implemented.
File e2e: 1 unique / 2 visits / 2 entries / 2 exits / 0 open; RTSP e2e: 1
unique, 2 visits, balanced, 0 open through MediaMTX. PR-hygiene, edge publish
and nightly were intentionally dropped; API/UI assertions wait for P2.

---

## 8. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Far faces (25–45 px) are missed or produce garbage embeddings. | Detect at `imgsz=1600` CPU / `1920` GPU, track at 40 px, register only at ≥60 px. Optional 2×2 tiling if recall is still low. |
| R2 | Back-facing visitors cannot be counted by a face-only detector. | Documented limitation; anonymous visits are still logged. Optional YOLO person detector fallback in a later phase. |
| R3 | Same person registered twice because the first embedding was poor. | Higher `register_quality_threshold` than the embedding gate; up to 3 embeddings per face; calibration pass in Phase 8; manual merge script if needed. |
| R4 | ByteTrack ID switches when faces cross. | Identity-anchored open visits; recognition re-anchors the visit to the Face ID. |
| R5 | Detector drops frames → false exits. | 3 s exit timeout; `LOST → ACTIVE` reacquisition; a reappearance after exit reuses the Face ID, so the unique count stays correct even if an extra visit is logged. |
| R6 | Clip gaps confuse tracking. | One source window per file; flush at file end; global clock from filename + frame index. |
| R7 | CPU-only interview machine cannot keep up. | ONNX Runtime CPU, `yolov8n-face`, `imgsz`/`frame_skip` configurable, per-track embedding cache, producer/consumer decode thread. |
| R8 | Model weights unavailable offline at interview. | Fetch both checkpoints with `scripts/download_models.py`; no model weights are committed. Run it before the interview and verify the pinned SHA256 values. |
| R9 | Log explosion on long streams. | Rotating handlers; INFO audit events only; DEBUG per-frame log disabled by default. |
| R10 | Windows path/encoding issues. | `pathlib` everywhere, UTF-8, ISO timestamps with `-` instead of `:` in filenames. |
| R11 | HEVC decode differences. | PyAV primary, OpenCV fallback; decode test in Phase 2 on all 22 clips. |
| R12 | Face images are biometric data. | Local-only storage, no network upload, document retention/deletion in README. |
| R13 | A long GPU session can stall after ~20 min (observed once: CPU idle, GPU spinning, later 510 MHz). | Run in shorter segments or restart the process; verify the DB after resume; use `imgsz=1280`/`frame_skip=3` on thermally limited hosts. The validated run resumed cleanly. |

---

## 9. Definition of done (grading map)

| Requirement (problem statement / AGENTS) | Where satisfied | Proof artefact |
|---|---|---|
| YOLO face detection | Phase 3 | `events.log` detection lines, annotated debug frames |
| InsightFace/ArcFace embeddings | Phase 4 | `EMBEDDING_GENERATED` lines, `embeddings` rows |
| Auto-registration with unique ID | Phase 4, 6 | `FACE_REGISTERED` lines, `faces` rows |
| Re-identification without count increment | Phase 4, 6 | `FACE_RECOGNIZED` lines, counter test |
| Tracking with ByteTrack | Phase 5 | `TRACK_STARTED/LOST` lines |
| `frame_skip` from `config.json` | Phase 1, 7 | config + measured cycle rate |
| Exactly one entry + one exit per visit | Phase 5, 6 | `UNIQUE(visit_id, event_type)` + tests |
| Cropped images in date folders | Phase 6 | `logs/entries|exits/YYYY-MM-DD/` |
| Timestamp, type, Face ID in DB and log | Phase 6 | `events` table, `events.log` |
| Mandatory `events.log` coverage | Phase 6 | log lines for detect/embed/register/recognize/track/entry/exit |
| Unique count from DB | Phase 6 | `get_unique_count()` + `--report` |
| Atomic images, transactions, WAL | Phase 1, 6 | code + tests |
| SIGINT/SIGTERM flush | Phase 5, 7 | shutdown test |
| RTSP + reconnect | Phase 2 | `RTSPSource`, capped backoff/retry tests; real camera is a manual interview check |
| Sample output | Phase 8 | `samples/` |
| README + video + Katomaran line | Phase 9 | README + footer done; public video is an author action |

---

## 10. Resolved defaults (accepted for submission)

These defaults were used for the final implementation; no changes were requested.

1. **Entry/exit meaning.** An entry is “track confirmed after appearing in
   frame”; an exit is “track missing for 3 s or stream ended”. Event type is
   lifecycle-based, **not** direction-based. **Accepted.**
2. **Back-facing visitors.** The face-only limitation is accepted; they are
   logged as anonymous visits and excluded from the unique count. An optional
   person-detector fallback remains future work. **Accepted.**
3. **Dataset priority.** The 22 `record_*` clips are the primary demo;
   `video_sample1.mp4` is the crowded stress test. **Accepted.**
4. **Size gates.** Track at 40 px native, register at 60 px native. This
   deliberately tracks more faces than it registers. **Accepted.**
5. **Frontend.** Deferred until the core was complete; the optional read-only
   dashboard was added in Phase 10. **Accepted.**

---

## 11. Environment note

`python` / `py` are not currently on PATH in this shell. `uv` **is** installed
(`C:\Users\mdaashir\AppData\Local\mise\installs\uv\latest\uv.exe`), so the
project will be set up with:

```powershell
uv python install 3.11
uv venv
uv pip install -r requirements.txt
```

`ffmpeg` / `ffprobe` are available and decode all 22 HEVC clips successfully
(verified). InsightFace will download `buffalo_l` on first run unless the model
cache is vendored; plan for an offline check.

---

### Phase 12 — P2 F1 read-only API and Streamlit views

- FastAPI read layer over SQLite/`events.log`: health, stats, events with filters,
  log tail, images and redacted config.
- Streamlit Dashboard, Event Log and System Logs; pure HTTP client.
- Tests: API contract plus Streamlit `AppTest` boot smoke.

**Status (2026-11-12):** F1 implemented. 223 tests, 92.21% coverage. F2 (job
control/live view/uploads), F3 (folder/config) and F4 (visitors/system) remain.

---

### Phase 13 — P2 F2 jobs, uploads and live view

- One-worker `JobManager` over the existing `run_pipeline` composition; file,
  folder and RTSP requests, queued/running/stopped/failed states, stop.
- Uploads into `src/video`, filename-only job paths, and a 2 GiB cap.
- Live MJPEG through a latest-frame buffer that copies only when a viewer is
  attached; SSE activity feed from new DB events.
- Streamlit Live Monitor and Sources & Jobs pages.

**Status (2026-11-12):** F2 implemented. 230 tests, 90.58% coverage. F3 (folder
polish/config) and F4 (visitors/system) remain.

---

### Phase 14 — P2 F3 folder jobs and runtime config

- `GET /folders`; folder jobs use `**/*.mp4` recursive, sorted, sequential
  processing through the existing `FileSequenceSource`.
- `ConfigStore`: base `config.json` + `data/runtime_overrides.json`, validated
  via the same loader rules; five hot fields only, `frame_skip` stays
  file-owned; atomic save-default.
- `GET/PUT /config`, `POST /config/reset`, `POST /config/save-default`;
  applied config reaches the next job.
- Streamlit folder mode and Config page.

**Status (2026-11-12):** F3 implemented. 235 tests, 90.65% coverage. F4
(visitors/system polish) remains.

---

### Phase 15 — P2 F4 visitors, system and polish

- `GET /visitors` and `/visitors/{id}` from faces/visits/events, with event
  image URLs for thumbnails and visit timelines.
- `GET /system`: CPU/memory/disk, process RSS, DB size, uptime, ONNX providers,
  model settings and GPU availability.
- Streamlit Visitors gallery/timeline and System page; Event Log CSV export;
  System Logs download.

**Status (2026-11-12):** F4 implemented. 237 tests, 90.81% coverage. P2 is
feature-complete; P0 video/form submission actions remain.

---

### Phase 16 — Final submission audit

- Re-audited the README, feature count, UI evidence and test numbers after F4.
- Updated `COMPLIANCE.md` and `SUBMISSION.md` to 35 features, 231 tests,
  90.47% coverage and the full optional P2 UI.
- No repository code changes: the only remaining deliverables are the public
  demo video and Google Form submission.

**Status (2026-11-12):** repository-side submission is complete; external
author actions remain.

---

### Phase 17 — UI modernization and cleanup

- Replaced deprecated `use_container_width` usage with Streamlit's `width`
  API (`"stretch"`) and removed the startup warning.
- Rebuilt `ui/app.py` as a themed operations console: hero headers, KPI cards,
  sidebar health, bordered work areas and responsive gallery/table layouts.
- Kept the UI a pure HTTP client; no pipeline/storage imports.
- Emptied the old static HTML dashboard artifacts and removed their test and
  documentation references; the Streamlit console supersedes it.

**Status (2026-11-13):** 231 tests, 90.47% coverage; ruff clean; Streamlit
smoke passes without an API.

---

## 12. Compliance audit (2026-11-11)

Every requirement in the problem statement is mapped to implementation evidence
in [`COMPLIANCE.md`](./COMPLIANCE.md). Repository-side requirements are complete;
the public demo video and Google Form submission are the only author actions.
