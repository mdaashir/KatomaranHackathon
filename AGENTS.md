# Intelligent Face Tracker — Hackathon Project

Python real-time pipeline: detect, recognize, track faces in a video/RTSP
stream, log every entry/exit, and count unique visitors.
Problem statement: `docs/Katomarans_Problem_Statement.pdf` (source of truth).
Submission deadline: **2:00 PM, Friday 13 Nov 2026**.

## Role

You are a senior Python engineer specialising in real-time computer vision
systems. Prefer clean, modular, production-style architecture over quick
scripts. Explain non-obvious decisions in comments — the author will be
interviewed on every line and on the prompts used.

## Mandatory tech stack (do not deviate)

| Concern          | Choice                                                         |
|------------------|----------------------------------------------------------------|
| Language         | Python 3.10+                                                   |
| Face detection   | YOLO face model (YOLOv8 via `ultralytics`)                     |
| Embeddings       | InsightFace (ArcFace, e.g. `buffalo_l`)                        |
| Tracking         | ByteTrack or DeepSORT (OpenCV tracker only as fallback)        |
| Database         | SQLite (WAL mode)                                              |
| Config           | `config.json`                                                  |
| Input            | Video file in dev, **RTSP stream at interview** — support both |
| API (bonus)      | FastAPI — thin control/streaming layer over the pipeline       |
| UI (bonus)       | Streamlit (client of the API only)                             |
| Containers       | Docker + Docker Compose (CPU default, GPU optional)            |
| CI/CD            | GitHub Actions                                                 |

- NEVER use the `face_recognition` library.

## Scope and priority

The PDF requires no frontend and no CLI: input comes from `config.json`
(video file or RTSP). A frontend is only "an added advantage". So the graded
core comes first and everything else must never put it at risk.

- **P0 — graded core:** pipeline, logging, DB, unique count, `config.json`,
  README, sample outputs, demo video. Must run headless with
  `python main.py --config config.json` (a thin entrypoint, not a CLI
  feature set) — this is what runs against the RTSP stream at the interview.
- **P1 — Docker + core CI:** Dockerfile, compose, `ci.yml` (lint, test,
  build, e2e on a trimmed clip).
- **P2 — API + Streamlit UI:** live monitor, dashboard, upload/folder/RTSP
  input, log view, runtime config pane.
- **P3 — extras:** release/security workflows, visitor gallery
  polish, folder-batch polish.

If time runs short, cut from the bottom (P3, then P2, then P1). Never cut P0,
and never skip the README or the demo video.

Specs live in separate files so this one stays short. Read them on demand,
before starting the matching work:

- UI / API work: `docs/FRONTEND.md`
- Docker / CI / release work: `docs/DEVOPS.md`

UI/API rules that always apply:

- The pipeline must run fully without the API or UI. They are clients, never
  dependencies; the UI contains no business logic and never touches the DB
  or pipeline code directly (it goes through the API).
- The live view must never slow the pipeline (drop frames for the viewer).
- Runtime config overrides use the same validated config model as
  `config.json`; `frame_skip` still originates from `config.json`.

## Code standards

- Type hints and docstrings on every public function/class; comment the why.
- Dependency injection by constructor: classes receive their collaborators
  (db, config, detector, logger). Never instantiate infrastructure inside
  business classes, and no module-level singletons or mutable globals.
- Every integration lives in its own module with one clear interface:
  detector, embedder, tracker, database, event logger, video source.
- No magic numbers. Every tunable comes from `config.json` and is loaded
  once into a typed config object (dataclass/pydantic) and passed in.
- Use the `logging` module, never `print`, for runtime output.
- Keep functions small and unit-testable; add pytest tests for the registry
  (re-ID matching), the entry/exit state machine, and the unique counter.

## Project structure

```
config.json
main.py                     # composition root: wires dependencies, runs pipeline
pyproject.toml  uv.lock  .python-version
Dockerfile  docker-compose.yml  docker-compose.gpu.yml  .dockerignore  entrypoint.sh
.github/                    # workflows/, dependabot.yml
ui/                         # Streamlit app (P2) — talks to src/api only
src/
  config/                   # config loading + validation
  video/                    # VideoSource (file + RTSP, with reconnect)
  detection/                # YOLO face detector
  recognition/              # InsightFace embedder + face registry (cosine match)
  tracking/                 # tracker wrapper + track lifecycle (entry/exit)
  events/                   # EventLogger: image store + DB row + events.log line
  storage/                  # SQLite connection, schema, repositories
  counting/                 # unique visitor counter (derived from DB)
  pipeline/                 # orchestrates one frame -> events
  api/                      # FastAPI: jobs, stream, stats, logs, config (P2)
logs/
  events.log
  entries/YYYY-MM-DD/       # cropped face images on entry
  exits/YYYY-MM-DD/         # cropped face images on exit
data/                       # sqlite db
docs/                       # PLAN.md FEATURES.md COMPUTE.md ARCHITECTURE.md PROMPTS.md
samples/                    # sample logs, images, DB export from the provided video
scripts/                    # download_models.py, e2e helpers
tests/                      # unit/, e2e/, fixtures/ (trimmed sample clip)
```

## Functional rules (these are graded)

1. **Detection cadence:** run YOLO every `frame_skip` frames (from
   `config.json`); the tracker keeps IDs alive on skipped frames.
2. **Auto-registration:** a face with no match above `similarity_threshold`
   gets a new unique face ID, its embedding and timestamp stored in the DB,
   and an "embedding generated" + "registered" line in `events.log`.
3. **Re-identification:** match by cosine similarity against stored
   embeddings. A returning face reuses its ID and must NOT increment the
   unique count. Skip embedding on tiny/blurry/profile faces
   (`min_face_size`, quality checks) to avoid garbage registrations.
4. **Exactly one entry and one exit per visit.** Use a per-track state
   machine (NEW -> ACTIVE -> LOST -> EXITED). Exit fires only after the
   track is missing for `exit_timeout_frames`. On end-of-stream or shutdown,
   flush every active track as an exit event.
5. **Each event records:** cropped face image, timestamp, event type
   (`entry`/`exit`), face ID. Image saved under
   `logs/entries|exits/YYYY-MM-DD/`, metadata saved in the DB.
6. **`events.log` is mandatory** and must record: face detected, embedding
   generated, registered, recognized, tracking started/lost, entry, exit.
7. **Unique count** is derived from the DB (`COUNT` of registered faces),
   never from an in-memory counter alone. Expose a `get_unique_count()`.
8. **Resilience:** write images atomically (temp file then rename), wrap
   DB writes in transactions, use SQLite WAL, handle SIGINT/SIGTERM
   gracefully, and auto-reconnect on RTSP drops.

## Config (`config.json` shape)

```json
{
  "source": { "type": "file", "path": "data/sample.mp4", "rtsp_url": "" },
  "frame_skip": 3,
  "detector": { "model_path": "models/yolov8n-face.pt", "conf_threshold": 0.5 },
  "recognition": { "model_name": "buffalo_l", "similarity_threshold": 0.45, "min_face_size": 60 },
  "tracking": { "tracker": "bytetrack", "exit_timeout_frames": 30 },
  "storage": { "db_path": "data/faces.db", "log_dir": "logs" }
}
```

Runtime overrides (UI config pane) layer on top of this file; see
`docs/FRONTEND.md`. Never edit `config.json` at runtime except
through an explicit "Save as default" action.

## Skills

Do not load any skill by default. Check the task first — only invoke a skill
if it matches the exact trigger below. Never invoke a skill just because it
exists.

- `/architect` — before building something non-trivial with no plan yet
  (use it first for PLAN / FEATURES / COMPUTE / ARCHITECTURE docs)
- `/review` — when a feature is done and needs a production check
- `/recover` — when something is broken and the fix isn't obvious
- `/remember` — at the start of a new session to restore context,
  and at the end to save progress

## Deliverables checklist

Keep these current as the work progresses; do not leave them for the end.

- `docs/PLAN.md` — app plan; `docs/FEATURES.md` — every feature listed;
  `docs/COMPUTE.md` — estimated CPU and GPU load; `docs/ARCHITECTURE.md` —
  diagram (Mermaid).
- `docs/PROMPTS.md` — log the key prompts used (they are evaluated in the
  interview). Append after each major step.
- `README.md` — setup instructions, assumptions, sample `config.json`,
  planning doc + architecture diagram, Loom/YouTube video link (**no video =
  submission not reviewed**), and this exact final line at the bottom:
  `This project is a part of a hackathon run by https://katomaran.com`
- `samples/` — real output from the provided video: `events.log`, cropped
  images, DB entries/export, final unique count.
- `docs/FRONTEND.md` and `docs/DEVOPS.md` — keep
  them in sync with what is actually built; they double as feature docs.
- README also covers: `docker compose up`, UI URL, how to run headless, CI
  badge, and the final image size / compute numbers.
- **Git:** every change is committed and goes through a PR. Follow the
  Git workflow below on every task, without exception.

### Git workflow (every change, no exceptions)

`main` holds only the base folder structure and default files. Never commit
or push feature work directly to `main`, and never merge a PR yourself —
the author reviews and merges every PR. (CI also runs on every push to
`main` as a safety net for the author's own merges and direct commits; it is
not permission for Claude to push there.)

1. **Start from fresh main:** `git checkout main && git pull`, then create
   one branch for one feature / function / requirement, named by what it
   does: `feature/yolo-face-detector`, `feature/insightface-embedder`,
   `feature/entry-exit-state-machine`, `fix/...`, `docs/...`, `test/...`,
   `ci/...`, `chore/...`.
   Keep each branch small and focused on a single requirement.
2. **Build it, committing as you go.** Every change is committed — never
   leave work uncommitted. Small, logical commits with clear messages in
   imperative mood (`feat: add YOLO face detector wrapper`, `fix: ...`,
   `docs: ...`, `test: ...`).
3. **No co-author or tool attribution in commits or PRs.** Never add
   `Co-Authored-By`, `Generated with Claude Code`, `Claude-Session`, or any
   other trailer/footer. Commit messages contain only the message itself.
   If a hook rejects a commit for this reason, amend the message — do not
   bypass the hook with `--no-verify`.
4. **Complete, verify, test before the PR:** run the tests (`pytest`), run
   the affected part of the pipeline on the sample video, and confirm the
   behaviour against the Functional rules above. Run `/review` for any
   non-trivial feature. Fix everything found before moving on. Once CI
   exists, the PR's checks (lint, tests, build, e2e) must be green —
   check with `gh pr checks` and fix failures before telling the author the
   PR is ready.
5. **Raise the PR to `main`:** push the branch and open it with
   `gh pr create --base main` (or give the author the compare link). The PR
   description covers what changed, why, how it was tested, and anything the
   reviewer should check. No tool-attribution footer.
6. **Stop and wait.** After raising the PR, tell the author it is ready and
   do not start the next feature until they confirm it is merged. Then
   return to step 1 from an updated `main`.

## Session continuity

REQUIRED — do not skip, do not wait to be asked:

- **First action of every session:** run `/remember restore` before doing anything else.
- **Last action of every session:** run `/remember save` before closing.
