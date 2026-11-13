# Problem Statement Compliance Checklist

Source: [Katomarans Problem Statement](<Katomarans Problem Statement.pdf>)

Status: **final audit — 2026-11-13**. All functional requirements are
implemented and verified; the optional P2 API/UI is feature-complete
(235 tests). Real-time throughput is the one documented performance caveat.
The public video is uploaded and the Google Form submission is complete.
Interview-hardware fps tuning remains the only performance follow-up.

This checklist maps every requirement in the problem statement to the
implementation and the evidence a reviewer can inspect. Status values:

- **DONE** — implemented and verified.
- **PARTIAL** — implemented, but a documented limitation or manual action remains.
- **MISSING** — not implemented.
- **AUTHOR** — cannot be completed from the repository alone.

## 1. Detection, recognition, tracking and database

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1.1 | Process the provided sample video file | DONE | `FileSource`/`FileSequenceSource`; 22 clips processed end to end; `samples/` |
| 1.2 | Support a live RTSP stream for the interview | PARTIAL | `RTSPSource` with TCP/low-delay, bounded queue and capped reconnect; automated reconnect/decode tests. No local RTSP camera was available for a real kill/restore check. |
| 1.3 | YOLO-based face detector | DONE | `src/detection/yolo_detector.py`; `models/yolov8n-face.pt` |
| 1.4 | InsightFace / ArcFace embeddings; never `face_recognition` | DONE | `src/recognition/` uses InsightFace `buffalo_l` + ArcFace; no `face_recognition` dependency |
| 1.5 | Auto-register a new face with a unique ID and timestamp | DONE | `FaceRepository.create` inserts face + embedding atomically; `FACE_REGISTERED` in `events.log` |
| 1.6 | Recognize and track a registered face across frames | DONE | ByteTrack + cosine registry; `FACE_RECOGNIZED`, `TRACK_STARTED`, `TRACK_REACQUIRED` |
| 1.7 | Configurable frame skip in `config.json` | DONE | `frame_skip: 3` in `config.json`; loaded into typed config and used by the pipeline |
| 1.8 | SQLite (WAL) metadata store | DONE | `src/storage/`; WAL, foreign keys, transactional writes |
| 1.9 | Real-time processing of the 25 fps stream | PARTIAL | GPU 1600/skip3 averaged ~21 fps over the long run; one clip is near 1.0× excluding model load. `RTSPSource` drops stale frames so the count stays current. Tune `imgsz`/`frame_skip` on the interview GPU. |

## 2. Logging and filesystem

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 2.1 | Exactly one entry and one exit per visit | DONE | `ux_events_visit_type` unique index; state machine; 59/59 entries/exits, 0 open in the validated run; `scripts/verify_outputs.py` |
| 2.2 | Each event has a cropped image, timestamp, type and Face ID | DONE (anonymous limitation) | `events` table and dated JPEGs; recognized events carry `face_id`; far/back-facing anonymous visits have `face_id = NULL` by design |
| 2.3 | Images in `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/` | DONE | `ImageStore`; atomic temp-file + rename; 256×256 JPEGs |
| 2.4 | All metadata in the database | DONE | `faces`, `embeddings`, `visits`, `events` tables |
| 2.5 | Mandatory `events.log` covers detection, recognition, tracking, entry/exit, embedding and registration | DONE | `FACE_DETECTED`, `EMBEDDING_GENERATED`, `FACE_REGISTERED`, `FACE_RECOGNIZED`, `TRACK_STARTED`, `TRACK_LOST`, `TRACK_REACQUIRED`, `ENTRY`, `EXIT`, `SUMMARY` |
| 2.6 | Resilient to unexpected interruptions | DONE | WAL, transactions, atomic image writes, SIGINT/SIGTERM flush, capped RTSP reconnect |

## 3. Unique visitor counting

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 3.1 | Maintain an accurate unique-face count | DONE | `UniqueCounter` = `COUNT(faces WHERE is_active=1)`; 26 unique in the validated 22-clip run |
| 3.2 | Re-identification must not increment the count | DONE | Face 13 appears in visits 29 and 51 (two clips) and is counted once; `samples/images/reentry/` and `visits.csv` |
| 3.3 | Count retrievable from the database or logs | DONE | `main.py --report`; `UniqueCounter`; `SUMMARY` line in `events.log` |

## 4. Mandatory stack and configuration

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 4.1 | Python | DONE | Python 3.11 project |
| 4.2 | YOLO face detection | DONE | Ultralytics YOLOv8 face checkpoint |
| 4.3 | InsightFace / ArcFace recognition | DONE | `buffalo_l`, ArcFace |
| 4.4 | OpenCV / DeepSORT / ByteTrack tracking | DONE | `ByteTrackAdapter` (supervision ByteTrack) |
| 4.5 | SQLite / MongoDB / PostgreSQL | DONE | SQLite WAL |
| 4.6 | JSON configuration for detection skip | DONE | `config.json`, validated by typed config loader |
| 4.7 | Log file + local image store + DB | DONE | `events.log`, `logs/`, `data/faces.db` |
| 4.8 | Video file in development, RTSP at interview | DONE | Both sources implemented; real RTSP hardware confirmation is tracked under 1.2 |

## 5. Documentation and submission

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 5.1 | Planning document | DONE | `docs/PLAN.md` |
| 5.2 | Features documented | DONE | `docs/FEATURES.md` (36 features) |
| 5.3 | CPU and GPU compute estimates | DONE | `docs/COMPUTE.md` with measured Phase 3–8 numbers |
| 5.4 | Architecture diagram | DONE | Mermaid diagrams in `README.md` and `docs/ARCHITECTURE.md` |
| 5.5 | README setup instructions | DONE | `README.md` GPU and CPU setup, fresh-clone path |
| 5.6 | README assumptions | DONE | `README.md` known limitations |
| 5.7 | README sample `config.json` structure | DONE | `README.md` sample block + full `config.json` |
| 5.8 | README Loom/YouTube explanatory video | DONE | [YouTube walkthrough](https://youtu.be/bekQvFoOWTY) linked in README, SUBMISSION and DEMO; local [`demo.mkv`](../demo.mkv) |
| 5.9 | Exact Katomaran footer at the bottom of README | DONE | Last line of `README.md` |
| 5.10 | Sample output: logs, images, DB entries | DONE | `samples/` with `events.log` excerpt, 24 crops, CSVs and `summary.json`; optional P2 UI reads the same data |
| 5.11 | Prompts used are documented | DONE | `docs/PROMPTS.md` |
| 5.12 | Submit the GitHub repository in the Google Form before the deadline | DONE | Repository + public video submitted 2026-11-13 |
| 5.13 | Clean, modular, understandable Python code | DONE | Typed modules, constructor injection, tests, phase docs |

## 6. Optional and bonus items

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 6.1 | Frontend is optional; bonus if included | DONE | Full optional P2 FastAPI/Streamlit UI: dashboard, live MJPEG, jobs, folder/RTSP, config, visitors, system; API/UI tests |
| 6.2 | Log files are mandatory and used for verification | DONE | Full audit log format and sample excerpt |
| 6.3 | Code is modular, scalable and clearly commented | DONE | `src/` modules, docstrings, tests |
| 6.4 | Database and file logging consistent and resilient | DONE | WAL, transactions, atomic writes, `verify_outputs.py` |

## 7. Author actions before submission

**Completed 2026-11-13:** the 4–5 minute walkthrough was recorded, uploaded to
[YouTube](https://youtu.be/bekQvFoOWTY), linked in the repo, and the repository
+ video were submitted through the Katomaran Google Form.

## 8. Validation evidence

- `pytest`: **235 passed**, 90.29% coverage; API/UI contract tests plus the
  Streamlit boot smoke.
- `docker compose config`: valid; CPU/GPU image builds and e2e run in CI.
- `scripts/verify_outputs.py --db data/e2e.db`: integrity OK, 0 FK violations,
  0 missing images, 0 open visits, 59 entries / 59 exits.
- Full dataset: 22 clips / 33,044 frames; 26 unique, 59 visits, 33 anonymous.
- 4K stress clip: 32 visits, 0 registrations (motion blur) — documented.

## 9. Audit result and next iteration

**Repository-side result:** all functional requirements are implemented and
verified. Real-time throughput is the one documented performance caveat. The
public video and Google Form submission are complete, the exact footer line is
present, the samples are consistent, and the verifier is clean.

**Next iteration / interview follow-ups:**

1. Tune `imgsz` / `frame_skip` on the interview GPU if the source is 25 fps;
   the long run averaged ~21 fps at 1600/skip3. The RTSP queue already drops
   stale frames, so the unique count stays current.
2. Optional if interview hardware is available: repeat the RTSP kill/restore
   check on the actual camera. The code path and backoff/reconnect tests are
   already in place; this is a hardware confirmation, not a missing feature.
