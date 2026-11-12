# Streamlit client

The UI is a pure HTTP client of `src/api`; it never imports the pipeline or
SQLite.

```powershell
# Terminal 1: read-only API over the current DB/logs
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# Terminal 2: UI
uv run streamlit run ui/app.py
```

Open `http://localhost:8501`. The API defaults to `http://127.0.0.1:8000`; set
`API_URL` to point at another host. API docs are at `/docs`.

F1 provides the Dashboard, Event Log and System Logs views. Job control, live
MJPEG, uploads, folder processing, the config pane and the visitor gallery are
the later P2 slices in [`../docs/FRONTEND.md`](../docs/FRONTEND.md).
