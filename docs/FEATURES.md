# FEATURES — Intelligent Face Tracker

> Companion to [`PLAN.md`](./PLAN.md). Every feature has an ID, priority
> (**M**ust / **S**hould / **C**ould), owning phase and objective acceptance
> criteria. “Done” means the acceptance criteria are demonstrable from a log,
> a DB row, an image file, or a test.
>
> **Final audit (2026-11-13):** F-01–F-36 are implemented and tested. The
> public walkthrough is published at
> [https://youtu.be/bekQvFoOWTY](https://youtu.be/bekQvFoOWTY); the Google Form
> was submitted. See [`COMPLIANCE.md`](./COMPLIANCE.md).

---

## 1. Feature map

| Area | Features |
|---|---|
| Input | F-01 single file, F-02 clip sequence, F-03 RTSP + reconnect, F-04 graceful shutdown |
| Detection | F-05 YOLO face detection, F-06 frame cadence, F-07 native-size gates |
| Recognition | F-08 alignment + ArcFace embedding, F-09 quality gate, F-10 registry match, F-11 auto-registration, F-12 embedding cache/retry, F-13 re-identification |
| Tracking | F-14 ByteTrack wrapper, F-15 track lifecycle, F-16 identity-anchored visits, F-17 temporary-loss recovery |
| Events | F-18 entry event, F-19 exit event, F-20 atomic image store, F-21 events.log, F-22 DB consistency |
| Counting | F-23 unique count, F-24 summary/report |
| Config | F-25 typed config, F-26 validation |
| Ops | F-27 structured logging, F-28 tests, F-29 sample outputs, F-30 docs + demo, F-31 optional dashboard |
| Delivery | F-32 containers, F-33 core CI, F-34 API/UI, F-35 security/release |

---

## 2. Input

### F-01 — Single file source (M, Phase 2)

Read a local video file frame-by-frame; expose `index`, `timestamp_s`, BGR
`image`, `source_id`. Supports H.264 and HEVC.

**Acceptance**
- `video_sample1.mp4` (3840×2160, 240 frames) reads exactly 240 frames.
- First frame timestamp is 0.0; last is ≈8.0 s.
- `close()` releases the decoder; a second run works.

### F-02 — Ordered clip sequence source (M, Phase 2)

Given a folder (or sorted list) of `record_*.mp4`, iterate files in filename
order and treat each as an independent observation window. Parse the start time
from `record_YYYYMMDD_HHMMSS.mp4`; absolute event time =
clip start + `frame_index / fps`.

**Acceptance**
- All 22 clips are read in chronological order.
- A track alive at the end of a clip is flushed (exit) before the next clip
  starts; no track ID leaks across files.
- Events from clip 7 carry clip 7’s absolute timestamp, not wall clock.
- Known filename gaps (e.g. 18:52:10 → 19:03:24) are preserved.

### F-03 — RTSP source with reconnect (M, Phase 2)

Open an RTSP URL with TCP transport and low-delay options. On decode/connection
failure, emit `SOURCE_LOST`, back off 1 s → 2 s → 4 s … capped at 30 s, and
reconnect until shutdown. Bounded queue: when the consumer is slower than the
camera, drop the oldest frame, never grow latency.

**Acceptance**
- Killing and restarting a local RTSP loop reconnects without restarting the
  process; `SOURCE_LOST` and `SOURCE_RECONNECTED` appear in `events.log`.
- After a 30 s outage, the pipeline continues and tracks are flushed only if
  the gap exceeds the exit timeout.
- Memory stays flat for a 10-minute run (queue bounded).

### F-04 — Graceful shutdown (M, Phase 2/5/7)

Handle SIGINT and SIGTERM: stop reading, flush every `ACTIVE`/`LOST` track as a
single exit event, commit open visits, close DB and log, write a summary.

**Acceptance**
- Ctrl+C mid-stream produces an exit event for every open visit.
- `visits.status` is `closed` for every flushed visit.
- No `database is locked` or truncated image after shutdown.

---

## 3. Detection

### F-05 — YOLO face detector (M, Phase 3)

Wrap an Ultralytics YOLO face model (`models/yolov8n-face.pt`) behind
`FaceDetector.detect(image) -> list[FaceDetection]`. Each detection carries
native-coordinate `bbox (x1,y1,x2,y2)`, `confidence`, and 5 facial keypoints
when the model provides them.

**Acceptance**
- On a near-range record frame, detects the foreground face with confidence
  ≥ `conf_threshold`.
- On the 4K crowded frame, detects ≥ 20 faces at `imgsz=1280`.
- Boxes are mapped back to original pixel coordinates (verified by drawing and
  saving one debug frame).
- If weights are missing, startup fails with a message naming the expected
  path and download instructions.

### F-06 — Configurable detection cadence (M, Phase 1/7)

Run detection only when `frame_index % frame_skip == 0`. Every decoded frame is
still advanced through the source clock; tracker state persists on skipped
frames using the last known boxes.

**Acceptance**
- With `frame_skip = 3` at 25 fps, `events.log` shows ≈8.33 detection cycles/s.
- Changing `frame_skip` to 5 changes the cycle count proportionally without a
  code change.
- A face is not lost merely because 1–2 frames were skipped.

### F-07 — Native-size gates (M, Phase 3/4)

Two independent thresholds measured in **native source pixels**:

- `detector.min_face_size = 40` — below this, ignore the detection.
- `recognition.min_face_size = 60` — below this, track but never embed or
  register.

**Acceptance**
- A 35 px face produces no track.
- A 50 px face produces a track and entry/exit events but no embedding and no
  `faces` row.
- A 70 px face produces an embedding and (if unmatched) a registration.
- Gate values are configurable and documented.

---

## 4. Recognition

### F-08 — Alignment + ArcFace embedding (M, Phase 4)

Align the face crop with a 5-point similarity transform to the ArcFace
112×112 template, then run the InsightFace `buffalo_l` **recognition** model
directly (no second face detection). Return an L2-normalised 512-d float32
vector.

**Acceptance**
- Embedding dimension is 512 and `||v|| ≈ 1.0`.
- Two frames of the same face yield cosine similarity ≥ 0.5.
- Two different people yield cosine similarity < 0.35 in the calibration set.
- Per-face embedding latency is measured and recorded in `COMPUTE.md`.
- If the model cache is missing and the machine is offline, startup fails with
  an explicit message (no silent download hang).

### F-09 — Face quality gate (M, Phase 4)

Before embedding/registration, require:

- native face size ≥ `recognition.min_face_size`;
- detection confidence ≥ `recognition.min_confidence`;
- Laplacian variance of the aligned grey crop ≥ `blur_threshold`;
- plausible frontal geometry from keypoints (inter-eye distance and nose
  position within tolerance).

**Acceptance**
- A deliberately blurred crop is rejected with `QUALITY_REJECTED` in DEBUG.
- A profile/back-of-head crop is rejected.
- Rejections never create `faces` rows.
- Thresholds live in `config.json`.

### F-10 — Registry matching (M, Phase 4)

Load all `(face_id, embedding)` rows into a normalised matrix. For a query
embedding, return `(best_face_id, best_similarity)` by dot product. Match if
`best_similarity >= similarity_threshold` (default 0.45).

**Acceptance**
- Matching 1,000 identities takes < 1 ms per query on CPU.
- Reloading the registry from SQLite reproduces the same matches.
- Empty registry returns `(None, -1.0)` and registers cleanly.
- Threshold is configurable and logged with each decision.

### F-11 — Auto-registration (M, Phase 4/6)

On a quality-passing embedding with `best_similarity < similarity_threshold`,
insert one `faces` row and one `embeddings` row in a single transaction, assign
the next `face_id`, update the in-memory registry, and log
`EMBEDDING_GENERATED` + `FACE_REGISTERED`. The registering visit is linked to
the new Face ID.

**Acceptance**
- Exactly one `faces` row per new identity.
- `FACE_REGISTERED` includes `face_id`, `track_id`, `similarity` and
  `threshold`.
- A crash between image write and DB commit does not leave a face without an
  embedding (transaction).
- Registering the same embedding twice does not create a second row
  (re-ID path taken).

### F-12 — Embedding cache and retry policy (M, Phase 5/7)

Once a track has a Face ID, do not embed it again on every cycle. If a track
has no Face ID, retry embedding at most once per
`recognition_retry_interval_sec` while the quality gate passes. Optionally add
an embedding to an existing face when quality improves
(`max_embeddings_per_face = 3`).

**Acceptance**
- A recognized track produces exactly one `EMBEDDING_GENERATED` per attempt
  window, not one per cycle.
- A 30-face crowded frame does not exceed the latency budget in `COMPUTE.md`.
- The number of embeddings per face never exceeds the configured maximum.

### F-13 — Re-identification (M, Phase 4/6)

A returning face with similarity ≥ threshold reuses its existing Face ID,
links to the visit, and does **not** increment the unique count.

**Acceptance**
- Same person in two different clips: one `faces` row, two visits, two entry
  events, unique count unchanged after the second.
- `FACE_RECOGNIZED` logs the matched `face_id` and similarity.
- Counter test asserts `before == after` for a re-ID.

---

## 5. Tracking and visits

### F-14 — ByteTrack wrapper (M, Phase 5)

Wrap ByteTrack (`supervision` or `boxmot`) behind `Tracker.update(detections,
frame_index) -> list[TrackedBox]`. Configure `frame_rate = fps / frame_skip`
and `lost_track_buffer` in detection cycles.

**Acceptance**
- A synthetic face moving 10 px/cycle keeps one track ID.
- A 2-cycle occlusion keeps the track recoverable.
- Two faces crossing do not permanently merge into one track.
- Tracker configuration is documented and configurable.

### F-15 — Track lifecycle state machine (M, Phase 5)

States: `NEW → ACTIVE → LOST → EXITED`, plus `DROPPED` for unconfirmed tracks.

- `NEW`: seen once, no event.
- `ACTIVE`: `confirm_hits` (3) detections reached; emit exactly one entry.
- `LOST`: temporarily missing; no event yet.
- `EXITED`: missing for `exit_timeout_frames` (75 source frames ≈ 3 s); emit
  exactly one exit.

**Acceptance**
- A track that appears, is confirmed and disappears produces exactly one entry
  and one exit.
- A one-frame false detection never reaches `ACTIVE` and produces no events.
- A 2 s gap does not produce an exit.
- A 4 s gap produces exactly one exit.
- Unit tests cover every transition, the reacquisition path and the flush path.

### F-16 — Identity-anchored visits (M, Phase 5)

When a newly confirmed track has a Face ID that already has an **open** visit,
attach the track to that open visit instead of creating a second one. A closed
visit is never reopened; a later appearance creates a new visit with the same
Face ID.

**Acceptance**
- A tracker split that produces two simultaneous tracks for one person creates
  only one visit and one entry/exit pair.
- A person who exits and later reappears creates a new visit, reuses the Face
  ID, and does not change the unique count.
- The DB still enforces at most one entry and one exit per visit.

### F-17 — Temporary-loss recovery (M, Phase 5)

If a `LOST` track is reacquired before `exit_timeout_frames`, it returns to
`ACTIVE` on the same visit: no second entry, no exit. This absorbs detector
dropouts and brief occlusions.

**Acceptance**
- A 2 s detection gap at 25 fps does not produce an exit.
- A 4 s gap produces exactly one exit.
- Reacquisition emits `TRACK_REACQUIRED` and no `ENTRY`.
- The visit’s entry timestamp remains the original first sight.

---

## 6. Events and storage

### F-18 — Entry event (M, Phase 6)

On `NEW → ACTIVE`, write one entry event containing: cropped face image,
absolute timestamp (first sight), event type `entry`, Face ID (or
`is_anonymous = 1`), visit ID, track ID, bbox, confidence, similarity.

**Acceptance**
- Exactly one `events` row with `event_type='entry'` per visit.
- Image path exists and opens; crop is centred on the face.
- DB, filename and `events.log` all carry the same visit/face/track IDs.
- Entry timestamp equals the track’s first sight, not the confirmation time.

### F-19 — Exit event (M, Phase 6)

On `LOST → EXITED` or shutdown flush, write one exit event with the best
quality crop from the visit (falling back to the largest crop if the quality
gate never passed), timestamp of last sight, type `exit`, and the same identity
fields.

**Acceptance**
- Exactly one `events` row with `event_type='exit'` per visit.
- Exit timestamp ≥ entry timestamp.
- End-of-stream flush produces an exit for every open visit.
- Image is saved under `logs/exits/YYYY-MM-DD/`.

### F-20 — Atomic image store (M, Phase 6)

Save 256×256 JPEG crops (quality 90) to
`logs/entries|exits/YYYY-MM-DD/<ISO>_face<id>_visit<id>_track<id>.jpg`.
Write to `.tmp`, flush + `fsync`, `os.replace` to final path, then commit the DB
row. On DB failure, remove the just-written image and log the failure.

**Acceptance**
- No zero-byte or partial JPEG after a forced kill during a write.
- Date folder follows the event’s local date, not the run date.
- Filenames contain only filesystem-safe characters (no `:`).
- A DB failure leaves no orphan image.

### F-21 — Mandatory `events.log` (M, Phase 6)

Human-readable, machine-parseable pipe-delimited lines with UTC timestamp,
event code, and `key=value` fields. Required codes:

`FACE_DETECTED`, `EMBEDDING_GENERATED`, `FACE_REGISTERED`, `FACE_RECOGNIZED`,
`TRACK_STARTED`, `TRACK_LOST`, `TRACK_REACQUIRED`, `ENTRY`, `EXIT`,
`QUALITY_REJECTED` (DEBUG), `SOURCE_LOST`, `SOURCE_RECONNECTED`, `SHUTDOWN`,
`SUMMARY`.

**Acceptance**
- One line per required system event, present in `logs/events.log`.
- Rotating handler caps each file at 10 MB with 5 backups.
- Per-frame detection noise is NOT in `events.log` (goes to `logs/debug.log`).
- Every line parses with a documented regex; fields never contain raw newlines.

### F-22 — Database consistency and resilience (M, Phase 1/6)

SQLite in WAL mode, `foreign_keys=ON`, `busy_timeout=5000`,
`synchronous=NORMAL`. Each event + visit update is one transaction. Unique
index `(visit_id, event_type)` prevents duplicate entry/exit rows.

**Acceptance**
- `PRAGMA journal_mode` returns `wal`.
- Inserting a second entry for the same visit fails at the DB level.
- A hard kill (SIGKILL / Task Manager End Task) during a run leaves a
  recoverable DB (`PRAGMA integrity_check` = ok).
- Restart continues with existing faces and correct unique count.

---

## 7. Counting and reporting

### F-23 — Unique visitor count (M, Phase 6)

`UniqueCounter.get_unique_count()` executes
`SELECT COUNT(*) FROM faces WHERE is_active = 1` inside a read transaction.
Never uses an in-memory counter as the source of truth.

**Acceptance**
- Count matches `SELECT COUNT(*) FROM faces`.
- Re-identification does not change the count.
- Anonymous visits do not change the count.
- Count survives process restart.
- Count can be queried while the pipeline is writing (WAL read).

### F-24 — Report / summary (S, Phase 6/7)

`get_summary()` returns unique faces, total visits, open visits, entries,
exits, anonymous visits, first/last event time. `main.py --report` prints it
and `SUMMARY` is written on shutdown.

**Acceptance**
- `entries == exits` for a clean completed run.
- Open visits are zero after graceful shutdown.
- Summary is emitted to stdout and `events.log`.

---

## 8. Configuration and operations

### F-25 — Typed configuration (M, Phase 1)

Load `config.json` once into frozen dataclasses (or Pydantic models) and inject
them into constructors. No module reads the file directly. No magic numbers in
business code.

**Acceptance**
- All tunables in `ARCHITECTURE.md §7` are present in the typed config.
- Changing a value in JSON changes behaviour without code edits.
- Config is logged once at startup with secrets redacted.

### F-26 — Validation (M, Phase 1)

Validate required fields, types, ranges, enum values, and paths. Fail fast with
a message naming the exact key and expected value. `source.type` decides whether
`path` or `rtsp_url` is required.

**Acceptance**
- Missing `frame_skip` → clear error.
- `frame_skip = 0` → clear error.
- `source.type = "rtsp"` with empty URL → clear error.
- `similarity_threshold` outside `[0,1]` → clear error.
- Valid config loads without warnings.

### F-27 — Structured logging (M, Phase 1)

Python `logging` only, never `print`, for runtime output. Console INFO + rotating
file. `events.log` uses its own formatter. Source, visit, face and track IDs are
included wherever relevant.

**Acceptance**
- No `print()` in `src/` or `main.py` (grep test).
- Log level configurable.
- Exceptions include stack traces in the file log.

### F-28 — Tests (M, every phase)

Pytest suite covering registry, lifecycle, counter, config, image store,
storage, and a synthetic pipeline with fake detector/embedder/source.

**Acceptance**
- `pytest` passes on a clean checkout (model-dependent tests skip cleanly).
- Lifecycle tests cover all transitions, reacquisition and flush.
- Counter tests cover re-ID and anonymous exclusion.
- No test writes outside `tmp_path`.

### F-29 — Sample outputs (M, Phase 8)

`samples/` contains real output from the provided video:

- `events.log` excerpt,
- 10–20 entry/exit crops,
- DB export (CSV or JSON),
- `summary.json` with the final unique count,
- a short `README.md` describing how the sample was produced.

**Acceptance**
- Every image referenced in the sample log/DB exists.
- Summary count matches the DB export.
- Sample includes at least one recognized reappearance (same Face ID, two visits).

### F-30 — Documentation and demo (M, Phase 9)

README with setup, assumptions, sample config, planning/architecture links,
limitations, demo video link, and the exact Katomaran footer line.
`docs/PROMPTS.md` updated with the key prompts used.

**Acceptance**
- Fresh clone → sample run following README only.
- Demo video shows: live run, entry/exit logs, image folders, DB count.
- Footer line exactly: `This project is a part of a hackathon run by https://katomaran.com`

**Status (2026-11-11):** README setup, assumptions, sample config, planning and
architecture links, limitations, footer and prompts are done. The public video
link is still an author action.

### F-31 — Optional dashboard (C, Phase 10)

Read-only view of unique count, recent entries/exits and thumbnails. Separate
process; reads SQLite/logs; never blocks the pipeline.

**Acceptance**
- Dashboard shows a count consistent with the DB.
- Pipeline runs normally with the dashboard stopped.
- No new dependency in the core path.

**Status (2026-11-13):** superseded by the full P2 FastAPI/Streamlit console;
the standalone `src/dashboard`/`scripts/dashboard.py` artifacts were removed.

### F-32 — Containerized runtime (M, P1)

Multi-stage CPU/GPU Dockerfile, compose stack, root `entrypoint.sh`,
`pyproject.toml`/`uv.lock` dependency groups and a `requirements.txt` fallback.
Models stay out of the image and download into a named volume; the pipeline
remains headless.

**Acceptance**
- `docker compose config` passes; CPU image builds and runs a short clip.
- First start downloads and verifies models into the models volume.
- `docker compose up` is the documented one-command start; GPU override exists.
- Image size and build time are recorded in COMPUTE/README.

**Status (2026-11-12):** CPU image built and smoke-tested; model download
verified. GPU build and CI/e2e are next.

### F-33 — Core CI (M, P1)

`ci.yml` with lint, typecheck, tests, CPU/GPU image builds and e2e; later
PR-hygiene, security and release workflows per `docs/DEVOPS.md`.

**Acceptance**
- PRs and `main` run the same checks; e2e uses a trimmed fixture and verifies
  DB/log/image invariants.

**Status:** core CI, e2e, security and tagged release implemented: ruff,
non-blocking mypy, pytest with an 85% coverage gate, CPU/GPU builds, file+RTSP
e2e, CodeQL/dependency-review/gitleaks/pip-audit/Trivy, versioned GHCR release
images and grouped Dependabot. PR-hygiene/edge-publish/nightly were dropped.
API/UI assertions wait for P2.

### F-34 — API and Streamlit UI (C, P2)

Thin FastAPI service and Streamlit client implementing the screens and runtime
overrides in `docs/FRONTEND.md`; the pipeline runs without either.

**Status:** F1–F3 implemented as recorded above. F4 implemented: `/visitors`,
`/visitors/{id}`, `/system`, Visitors gallery/timeline, System metrics/model
info, Event Log CSV export and log download. P2 is feature-complete.

### F-35 — Security and release automation (C, P3)

CodeQL, dependency/secret scanning, Trivy, GHCR tagged releases, release notes
and a samples zip. Edge publishing and nightly were dropped by the author.

**Status:** implemented in `security.yml` + `release.yml`; Dependabot remains.

### F-36 — Host webcam source (C, P2)

Local camera input through OpenCV for a real-time demo without a video file or
RTSP camera. The API process must run on the machine that owns the camera;
Linux Docker needs an explicit `--device`.

**Status (2026-11-13):** implemented in `src/video/webcam.py`, `/jobs`
(`source_type=webcam`, `device_index`), CLI `--source webcam --device-index`,
Sources & Jobs UI and `/stream/status` live health. Fake-capture tests cover
open failure, frame iteration and single-use behavior.

---

## 9. Explicit non-features

- No use of the `face_recognition` library.
- No cloud calls at runtime (after model download).
- No video recording of the source stream.
- No identity merge UI in the core scope.
- No claim that back-facing visitors are counted as unique.

---

## 10. Feature-to-phase matrix

| Phase | Features |
|---|---|
| 1 | F-22, F-25, F-26, F-27 |
| 2 | F-01, F-02, F-03, F-04 |
| 3 | F-05, F-06, F-07 |
| 4 | F-08, F-09, F-10, F-11, F-13 |
| 5 | F-12, F-14, F-15, F-16, F-17 |
| 6 | F-18, F-19, F-20, F-21, F-22, F-23, F-24 |
| 7 | F-04, F-06, F-24, F-25 |
| 8 | F-29, calibration and measured compute |
| 9 | F-30 |
| 10 | F-31 |
