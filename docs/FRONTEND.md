# Frontend & API Requirements (P2 — bonus layer)

> **Status (2026-11-12):** F1–F3 complete. F4 implemented: `/visitors`,
> `/visitors/{id}`, `/system`, Visitors gallery/timeline, System metrics and
> model info, Event Log CSV export, log download. All P2 screens are
> implemented; API/UI remain clients and never block the pipeline.

**Status of the requirement:** the problem statement says a frontend "is not
required, but including one will be considered an added advantage". It does
not ask for a CLI either; the system is driven by `config.json` and a video
file / RTSP stream. So this layer is a bonus: build it only after P0 (and
ideally P1) are done and merged.

## 1. Decision: Streamlit UI + thin FastAPI service

| Option | For | Against |
|---|---|---|
| **Streamlit (chosen)** | Pure Python, one language and one toolchain, native drag-and-drop uploader, fast to build, easy to containerise | Re-run model needs care for live updates; cannot browse server folders or upload directories natively |
| Separate React/Vite | Folder drop via `webkitdirectory`, full layout control | Second toolchain, more code, more CI and Docker surface, slower to ship |

Streamlit wins on time and risk. Its two weak spots are handled like this:

- **Live video:** the API serves an MJPEG stream (`/stream/live.mjpg`); the UI
  embeds it with an `<img src=...>` inside `st.components.v1.html`, so the
  browser pulls frames directly and Streamlit never re-runs per frame. Metrics
  and activity feed refresh with `st.fragment(run_every=...)`.
- **"Process all files in a folder":** a server-side folder picker over the
  mounted `src/video/` directory (dropdown/tree populated by `GET /videos`)
  plus a "Process all in folder" button. Multi-file drag-and-drop covers
  browser-side selection.

### Architecture

```
Browser ── Streamlit (ui/) ──HTTP──▶ FastAPI (src/api) ──▶ pipeline worker
                 │                         │                    │
                 └──── <img> MJPEG ────────┘                    ├─ SQLite (WAL)
                                                                └─ logs/ (events.log, crops)
```

- `python main.py --config config.json` still runs the pipeline headless with
  no API/UI (interview/RTSP mode).
- The API process owns the pipeline worker (a background thread or process)
  and exposes control + data. The UI is a pure client.

## 2. API (keep it minimal)

| Method & path | Purpose |
|---|---|
| `GET /health` | liveness + pipeline state |
| `POST /upload` | multipart, one or many video files → saved to `src/video/` |
| `GET /videos` | list files/folders under `src/video/` (traversal-safe) |
| `POST /jobs` | start a job: `{type: file\|folder\|rtsp\|webcam, path\|url\|device_index}` |
| `GET /jobs`, `GET /jobs/{id}` | queue, status, per-file progress |
| `POST /jobs/{id}/stop` | graceful stop (flushes active tracks as exits) |
| `GET /stream/live.mjpg` | annotated frames for the live view |
| `GET /stream/status` | viewers, frame age, resolution and frames sent |
| `GET /events/stream` | SSE: live pipeline events (detect, embed, register, recognise, entry, exit) |
| `GET /stats` | unique count, entries, exits, currently in frame, FPS |
| `GET /visitors`, `GET /visitors/{id}` | registered faces + visit history |
| `GET /events` | filter by date, type, face ID; paged |
| `GET /logs/tail` | tail of `events.log`, filter by level/text |
| `GET /images/{event_id}` | serve cropped face images by event ID (DB-resolved) |
| `GET /config`, `PUT /config` | effective config; apply runtime overrides |
| `POST /config/reset` | drop overrides, back to `config.json` |
| `POST /config/save-default` | explicitly write overrides into `config.json` |
| `GET /system` | CPU, memory, GPU (if any), model info |
| `POST /session/new` | start a new counting session (confirmation required) |

## 3. UI screens and acceptance criteria

1. **Live Monitor**
   - Annotated live video: bounding box, face ID label, track state colour
     (new / active / lost), detection confidence.
   - Live pipeline activity feed showing each stage as it happens: detected →
     cropped → embedding generated → matched or registered → tracking →
     entry / exit, with the latest face crops as thumbnails.
   - Counters: unique visitors, currently in frame, entries, exits, FPS,
     frame number, processing latency.
   - Start / Stop controls and a source indicator (file name, masked RTSP or
     webcam index). Live-stream health shows viewers, frame age and resolution.
2. **Sources & Jobs**
   - Drag-and-drop area for one or many video files (validate extension and
     size limit; show errors inline).
   - **Webcam:** choose a host camera index and start it as a real-time source;
     the API must run on the machine that owns the camera.
   - **Process folder:** choose a folder under `src/video/` and process every
     video in it (recursive, sorted, sequential). Per-file status and progress
     bar, continue on failure, summary at the end (entries/exits/new IDs per
     file).
   - **RTSP:** URL input with placeholder
     `rtsp://user:pass@host:554/stream`, prefilled from `config.json` if set,
     "Test connection" button, credentials masked everywhere in UI and logs.
   - Job queue view with stop/cancel.
3. **Dashboard**
   - Big unique-visitor count plus entries/exits today and currently in frame.
   - Charts: visitors over time, entries vs exits per interval, visit
     duration distribution, re-identification matches vs new registrations.
   - Date range filter.
4. **Visitors** — gallery of registered faces (thumbnail, ID, first seen,
   last seen, visit count); click for the full entry/exit timeline and images.
5. **Event Log** — table of entry/exit events with crop thumbnail, timestamp,
   type, face ID; filters (date, type, face ID); CSV export.
6. **System Logs** — live tail of `events.log` with level filter, text
   search, pause/resume, and download.
7. **Config pane** — see section 4.
8. **System** — CPU/GPU/memory, model names and versions, DB size, uptime.

## 4. Runtime config overrides

Three layers: `config.json` (base) → runtime overrides (persisted to
`data/runtime_overrides.json`) → effective config used by the pipeline.

- Form is generated from the same typed config model; invalid values are
  rejected with field-level errors (the same validation as file loading).
- **Hot-reloadable:** `frame_skip`, `conf_threshold`, `similarity_threshold`,
  `min_face_size`, `exit_timeout_frames`. **Restart required:** model
  names/paths, tracker type, DB path, source type. The UI labels which is
  which and offers "Apply & restart pipeline" for the latter.
- Show a diff between base and effective values. "Reset to config.json"
  clears overrides. "Save as default" is an explicit action that rewrites
  `config.json` atomically; nothing else ever writes that file.
- Every applied override is logged to `events.log` (key, old, new, time).
- `frame_skip` must still originate from `config.json` (PDF requirement);
  overrides only change it for the running session.

## 5. Sessions and counting

The unique count must not double count a returning face. Decision to record
in the README assumptions: the face registry persists across files within a
**session**; the unique count is reported per session and all-time. "New
session" (with confirmation) starts a fresh count without deleting history.

## 6. Non-functional requirements

- UI down or closed ⇒ pipeline keeps running and logging.
- The MJPEG stream is downscaled and capped (about 10 fps); if the viewer is
  slow, frames are dropped for the viewer only.
- RTSP credentials are masked in every log line, API response and UI view.
- Upload hardening: extension allow-list, size limit, safe filenames,
  path-traversal protection on `/videos`, `/images`, and job paths.
- API CORS limited to the UI origin; no auth required for the hackathon, but
  document that ports should not be exposed publicly.
- Basic tests: API contract tests (pytest + `TestClient`), and a Streamlit
  smoke test (`streamlit.testing.v1.AppTest`).

## 7. Build order inside P2

1. **F1:** API skeleton, `/stats`, `/events`, `/logs/tail`, Streamlit Dashboard
   + Event Log + System Logs.
2. **F2:** job control, live MJPEG + activity feed, RTSP input, drag-and-drop
   upload.
3. **F3:** folder processing, config pane with runtime overrides.
4. **F4:** visitors gallery, system page, polish.

One feature per branch and PR, as in the Git workflow.
