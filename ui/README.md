# Streamlit client

The UI is a pure HTTP client of `src/api`; it never imports the pipeline or
SQLite.

```powershell
# Terminal 1: control/read API over the current DB/logs
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# Terminal 2: UI
uv run streamlit run ui/app.py
```

Open `http://localhost:8501`. The API defaults to `http://127.0.0.1:8000`; set
`API_URL` to point at another host. API docs are at `/docs`.

F1 provides the Dashboard, Event Log and System Logs views. F2 adds Live
Monitor (MJPEG + activity feed) and Sources & Jobs (upload, file/RTSP start,
queue/stop). F3 adds recursive folder jobs and the Config page for the five
hot-reloadable tunables (`frame_skip` stays file-owned). F4 adds the Visitors
gallery/timeline, the System page, Event Log CSV export and log download.
Uploads go to `src/video/`; job paths are filenames/folder names in that folder.
