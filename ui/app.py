"""Streamlit control console for the face tracker.

The UI is a pure HTTP client: it never imports the pipeline, SQLite or storage
modules, and the tracker keeps running when the console is closed.
"""

from __future__ import annotations

import csv
import html
import io
import os

import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
REFRESH_SECONDS = None if os.getenv("FACETRACKER_REFRESH") == "0" else 5
LIVE_REFRESH_SECONDS = None if REFRESH_SECONDS is None else 3

st.set_page_config(
    page_title="Face Tracker · Operations",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
:root {
  --ink: #0f172a;
  --muted: #64748b;
  --line: #e2e8f0;
  --brand: #0e7490;
  --brand-2: #0f766e;
  --surface: #ffffff;
}
.stApp {
  background:
    radial-gradient(circle at 92% 0%, rgba(14, 116, 144, 0.12), transparent 34%),
    radial-gradient(circle at 0% 40%, rgba(15, 118, 110, 0.07), transparent 28%),
    #f4f7fb;
}
.block-container { max-width: 1500px; padding: 1.3rem 2.2rem 3rem; }
[data-testid="stSidebar"] {
  background: #0b1220;
  border-right: 1px solid rgba(148, 163, 184, 0.18);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
[data-testid="stSidebar"] * { color: #cbd5e1; }
[data-testid="stSidebar"] .stRadio > div { gap: 0.15rem; }
[data-testid="stSidebar"] .stRadio label {
  padding: 0.55rem 0.65rem;
  border-radius: 9px;
  font-weight: 500;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(255, 255, 255, 0.06);
}
.brand { display: flex; gap: 0.75rem; align-items: center; padding: 0.2rem 0.35rem 1.15rem; }
.brand-mark {
  width: 40px; height: 40px; border-radius: 13px;
  background: linear-gradient(135deg, #0e7490, #14b8a6);
  display: flex; align-items: center; justify-content: center;
  color: white; font-weight: 800; font-size: 0.8rem; letter-spacing: 0.04em;
  box-shadow: 0 8px 20px rgba(14, 116, 144, 0.35);
}
.brand-title { color: #f8fafc; font-weight: 700; font-size: 1.02rem; }
.brand-sub {
  color: #94a3b8; font-size: 0.68rem; text-transform: uppercase;
  letter-spacing: 0.14em; margin-top: 0.12rem;
}
.sidebar-rule { height: 1px; background: rgba(148, 163, 184, 0.18); margin: 0.7rem 0 1rem; }
.sidebar-meta { color: #94a3b8; font-size: 0.74rem; line-height: 1.65; padding: 0.25rem 0.35rem; }
.sidebar-meta strong { color: #e2e8f0; font-weight: 600; }
.hero {
  background: linear-gradient(120deg, #0b1220 0%, #134e4a 58%, #0e7490 100%);
  border-radius: 22px; padding: 1.65rem 1.9rem;
  color: #f8fafc; box-shadow: 0 20px 45px rgba(15, 23, 42, 0.16);
  margin-bottom: 1.35rem;
}
.hero .eyebrow {
  color: #99f6e4; font-size: 0.69rem; font-weight: 700;
  letter-spacing: 0.18em; text-transform: uppercase;
}
.hero h1 { color: #ffffff; margin: 0.35rem 0 0.4rem; font-size: 2rem; line-height: 1.12; }
.hero p { color: #cbd5e1; margin: 0; max-width: 820px; font-size: 0.95rem; }
.hero-row { display: flex; gap: 0.45rem; flex-wrap: wrap; margin-top: 1rem; }
.pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.28rem 0.65rem; border-radius: 999px;
  background: rgba(255, 255, 255, 0.1); color: #e2e8f0;
  border: 1px solid rgba(255, 255, 255, 0.14);
  font-size: 0.72rem; font-weight: 600;
}
.pill .dot { width: 6px; height: 6px; border-radius: 50%; background: #5eead4; }
.section-head { display: flex; align-items: end; justify-content: space-between; margin: 0.9rem 0 0.45rem; }
.section-head h2 { color: var(--ink); font-size: 1.1rem; margin: 0; }
.section-head span { color: var(--muted); font-size: 0.78rem; }
.muted { color: var(--muted); font-size: 0.82rem; }
[data-testid="stMetric"] {
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
}
[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.76rem; }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 700; }
[data-testid="stDataFrame"] {
  border: 1px solid var(--line); border-radius: 14px;
  overflow: hidden; background: #ffffff;
}
.stButton > button { border-radius: 10px; font-weight: 600; border-color: #cbd5e1; }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0e7490, #0f766e);
  border: 0; color: #ffffff;
}
.stDownloadButton > button { border-radius: 10px; }
[data-testid="stFileUploader"] section {
  border: 1.5px dashed #94a3b8; border-radius: 14px; background: #ffffff;
}
.live-shell {
  background: #0b1220; border-radius: 18px; padding: 0.6rem;
  box-shadow: 0 18px 35px rgba(15, 23, 42, 0.18);
}
.live-shell img { width: 100%; border-radius: 12px; display: block; }
.card {
  background: #ffffff; border: 1px solid var(--line);
  border-radius: 16px; padding: 1rem 1.1rem;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}
.visitor-name { color: var(--ink); font-weight: 700; margin: 0.45rem 0 0.1rem; }
.visitor-meta { color: var(--muted); font-size: 0.76rem; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def _try_get(path: str, params: dict[str, object] | None = None):
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=2)
        if response.ok:
            return response.json()
    except (requests.RequestException, ValueError):
        return None
    return None


def api_get(path: str, params: dict[str, object] | None = None):
    data = _try_get(path, params)
    if data is None:
        st.error(f"API unavailable at {API_URL}. Start `uvicorn src.api.app:app`.")
    return data


def api_post(
    path: str,
    *,
    json_data: dict[str, object] | None = None,
    files=None,
    timeout: int = 15,
    method: str = "POST",
):
    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            json=json_data,
            files=files,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Request failed at {API_URL}{path}: {exc}")
        return None


def _image_url(path: str | None) -> str | None:
    if not path:
        return None
    return f"{API_URL}{path}" if path.startswith("/") else f"{API_URL}/{path}"


def page_header(title: str, subtitle: str, meta: tuple[str, ...] = ()) -> None:
    pills = "".join(
        f'<span class="pill"><span class="dot"></span>{html.escape(item)}</span>'
        for item in meta
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">Intelligent Face Tracker</div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(subtitle)}</p>
          <div class="hero-row">{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="section-head"><h2>{html.escape(title)}</h2>'
        f"<span>{html.escape(hint)}</span></div>",
        unsafe_allow_html=True,
    )


def _event_table(events: list[dict[str, object]]) -> None:
    if not events:
        st.info("No events match the current filters.")
        return
    st.dataframe(
        [
            {
                "Time": event["occurred_at"],
                "Type": event["event_type"],
                "Face": str(event["face_id"]) if event["face_id"] else "anonymous",
                "Visit": event["visit_id"],
                "Track": event["track_id"],
                "Source": event["source_id"],
            }
            for event in events
        ],
        width="stretch",
        hide_index=True,
    )
    columns = st.columns(6)
    for index, event in enumerate(events[:6]):
        with columns[index % 6]:
            url = _image_url(event.get("image_url"))
            if url:
                st.image(url, caption=f"#{event['event_id']}", width="stretch")


@st.fragment(run_every=REFRESH_SECONDS)
def dashboard_body() -> None:
    stats = api_get("/stats")
    if stats:
        columns = st.columns(6)
        values = (
            ("Unique faces", stats["unique_faces"]),
            ("Visits", stats["visits"]),
            ("In frame", stats["currently_in_frame"]),
            ("Entries", stats["entries"]),
            ("Exits", stats["exits"]),
            ("Anonymous", stats["anonymous_visits"]),
        )
        for column, (label, value) in zip(columns, values):
            column.metric(label, value)
        st.caption(
            f"First event: {stats['first_event_at'] or '—'} · "
            f"Last event: {stats['last_event_at'] or '—'}"
        )
    events = api_get("/events", {"limit": 10})
    if events:
        section("Recent activity", "Latest entry and exit events")
        _event_table(events["items"])


def dashboard() -> None:
    page_header(
        "Operations overview",
        "A database-derived view of the front-entry pipeline, unique visitors "
        "and the latest activity.",
        ("Read-only", "DB-derived counts", "5s refresh"),
    )
    dashboard_body()


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def live_body() -> None:
    status = api_get("/stream/status")
    if status:
        columns = st.columns(4)
        columns[0].metric("Viewers", status["viewers"])
        columns[1].metric("Frames sent", status["frames_published"])
        age = status["frame_age_seconds"]
        columns[2].metric("Frame age", f"{age:.1f}s" if age is not None else "—")
        columns[3].metric(
            "Resolution",
            f"{status['width']}×{status['height']}"
            if status["width"] and status["height"]
            else "—",
        )
    jobs = api_get("/jobs")
    if jobs:
        running = [job for job in jobs["jobs"] if job["status"] == "running"]
        if running:
            st.caption(f"Running: {running[0]['label']} · {running[0]['job_id']}")
        else:
            st.caption("No pipeline job is running.")
    stats = api_get("/stats")
    if stats:
        columns = st.columns(4)
        columns[0].metric("Unique faces", stats["unique_faces"])
        columns[1].metric("In frame", stats["currently_in_frame"])
        columns[2].metric("Entries", stats["entries"])
        columns[3].metric("Exits", stats["exits"])
    events = api_get("/events", {"limit": 8})
    if events:
        section("Activity", "Live entry and exit feed")
        _event_table(events["items"])


def live_monitor() -> None:
    page_header(
        "Live monitor",
        "Annotated MJPEG from the pipeline worker, with the latest queue status "
        "and activity. Frames are dropped for the viewer before the pipeline.",
        ("MJPEG", "Viewer-only frame drops", "Activity feed"),
    )
    components.html(
        f'<div class="live-shell"><img src="{API_URL}/stream/live.mjpg" '
        'alt="Live annotated face tracker stream" /></div>',
        height=545,
        scrolling=False,
    )
    live_body()


def sources_jobs() -> None:
    page_header(
        "Sources and jobs",
        "Ingest a video, process a folder, or attach an RTSP camera. One worker "
        "runs the pipeline; jobs are queued and can be stopped.",
        ("Upload", "Folder / RTSP", "One worker"),
    )
    with st.container(border=True):
        section("Ingest a video", "MP4, AVI, MOV, MKV or M4V")
        uploads = st.file_uploader(
            "Upload videos",
            type=["mp4", "avi", "mov", "mkv", "m4v"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads and st.button("Upload selected", type="primary"):
            files = [
                (
                    "files",
                    (upload.name, upload.getvalue(), upload.type or "video/mp4"),
                )
                for upload in uploads
            ]
            result = api_post("/upload", files=files, timeout=300)
            if result:
                st.success(f"Uploaded {len(result['uploaded'])} file(s).")

    videos = api_get("/videos")
    names = videos["videos"] if videos else []
    folders = api_get("/folders")
    folder_names = folders["folders"] if folders else []
    if not names:
        st.info("No videos in the source folder yet.")

    section("Start a job", "The next job uses the current effective config")
    left, right = st.columns([1, 1])
    with left:
        with st.container(border=True):
            source_mode = st.radio(
                "Start from",
                ["file", "folder", "rtsp", "webcam"],
                horizontal=True,
            )
            if source_mode == "file":
                selected = st.selectbox(
                    "Video file",
                    names or ["(upload one first)"],
                )
                payload = {"source_type": "file", "path": selected}
            elif source_mode == "folder":
                selected_folder = st.selectbox(
                    "Folder (all videos, recursive)",
                    folder_names or ["(no folders)"],
                )
                payload = {"source_type": "folder", "path": selected_folder}
            elif source_mode == "webcam":
                device_index = st.number_input(
                    "Camera index",
                    min_value=0,
                    max_value=16,
                    value=0,
                    step=1,
                )
                payload = {
                    "source_type": "webcam",
                    "device_index": int(device_index),
                }
            else:
                rtsp_url = st.text_input(
                    "RTSP URL",
                    placeholder="rtsp://user:pass@camera/stream",
                    type="password",
                )
                payload = {"source_type": "rtsp", "url": rtsp_url}
    with right:
        with st.container(border=True):
            st.markdown("**Execution model**")
            st.caption(
                "The API owns one background worker. Uploads and folders are "
                "confined to the video directory; RTSP credentials are masked."
            )
            if st.button("Start job", type="primary"):
                result = api_post("/jobs", json_data=payload)
                if result:
                    st.success(f"Queued job {result['job_id']}.")

    jobs = api_get("/jobs")
    if jobs and jobs["jobs"]:
        section("Job queue", "Newest first")
        st.dataframe(
            [
                {
                    "Job": job["job_id"],
                    "Source": job["label"],
                    "Status": job["status"],
                    "Started": job["started_at"],
                    "Finished": job["finished_at"],
                    "Error": job["error"],
                }
                for job in jobs["jobs"]
            ],
            width="stretch",
            hide_index=True,
        )
        active = [
            job["job_id"]
            for job in jobs["jobs"]
            if job["status"] in {"queued", "running"}
        ]
        if active:
            stop_id = st.selectbox("Stop job", active)
            if st.button("Stop selected job"):
                result = api_post(f"/jobs/{stop_id}/stop")
                if result:
                    st.success(f"Stop requested for {stop_id}.")


def config_page() -> None:
    page_header(
        "Runtime configuration",
        "Tune the safe hot fields for the next job. The base file remains the "
        "single source of truth; frame skip is intentionally file-owned.",
        ("Validated overrides", "Next job only", "Atomic save"),
    )
    data = api_get("/config")
    if not data:
        return
    base = data["base"]
    effective = data["effective"]
    st.metric("frame_skip (config.json)", data["effective"]["frame_skip"])

    with st.container(border=True):
        with st.form("config_form"):
            left, right = st.columns(2)
            conf = left.number_input(
                "Detector confidence",
                min_value=0.0,
                max_value=1.0,
                value=float(effective["detector"]["conf_threshold"]),
                step=0.01,
            )
            detector_min = left.number_input(
                "Detector min face size",
                min_value=1,
                value=int(effective["detector"]["min_face_size"]),
                step=1,
            )
            similarity = right.number_input(
                "Recognition similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(effective["recognition"]["similarity_threshold"]),
                step=0.01,
            )
            recognition_min = right.number_input(
                "Recognition min face size",
                min_value=1,
                value=int(effective["recognition"]["min_face_size"]),
                step=1,
            )
            exit_timeout = right.number_input(
                "Exit timeout frames",
                min_value=1,
                value=int(effective["tracking"]["exit_timeout_frames"]),
                step=1,
            )
            applied = st.form_submit_button("Apply overrides", type="primary")

    if applied:
        result = api_post(
            "/config",
            method="PUT",
            json_data={
                "detector": {
                    "conf_threshold": conf,
                    "min_face_size": detector_min,
                },
                "recognition": {
                    "similarity_threshold": similarity,
                    "min_face_size": recognition_min,
                },
                "tracking": {"exit_timeout_frames": exit_timeout},
            },
        )
        if result:
            st.success("Overrides applied to the next job.")
            data = result
            base = data["base"]
            effective = data["effective"]

    section("Base vs effective", "Five hot-reloadable fields")
    keys = (
        ("detector.conf_threshold", ("detector", "conf_threshold")),
        ("detector.min_face_size", ("detector", "min_face_size")),
        ("recognition.similarity_threshold", ("recognition", "similarity_threshold")),
        ("recognition.min_face_size", ("recognition", "min_face_size")),
        ("tracking.exit_timeout_frames", ("tracking", "exit_timeout_frames")),
    )
    st.dataframe(
        [
            {
                "Key": label,
                "Base": base[section][key],
                "Effective": effective[section][key],
            }
            for label, (section, key) in keys
        ],
        width="stretch",
        hide_index=True,
    )
    st.json(data["overrides"])

    reset_col, save_col = st.columns(2)
    with reset_col:
        if st.button("Reset to config.json"):
            result = api_post("/config/reset")
            if result:
                st.success("Overrides cleared.")
    with save_col:
        confirmed = st.checkbox("I understand this rewrites config.json")
        if st.button("Save as default", disabled=not confirmed):
            result = api_post("/config/save-default")
            if result:
                st.success("Effective config saved as config.json.")


def visitors_page() -> None:
    page_header(
        "Visitor directory",
        "Registered identities, visit counts and the images that anchor each "
        "re-identification.",
        ("Face registry", "Cross-clip re-ID", "Visit timeline"),
    )
    data = api_get("/visitors")
    if not data:
        return
    visitors = data["items"]
    if not visitors:
        st.info("No registered visitors yet.")
        return
    section("Gallery", f"{data['total']} registered faces")
    columns = st.columns(4)
    for index, visitor in enumerate(visitors[:12]):
        with columns[index % 4]:
            with st.container(border=True):
                url = _image_url(visitor.get("thumbnail_url"))
                if url:
                    st.image(url, width="stretch")
                st.markdown(
                    f'<div class="visitor-name">Face {visitor["face_id"]}</div>'
                    f'<div class="visitor-meta">{visitor["visit_count"]} visits · '
                    f"quality {visitor['best_quality']:.2f}</div>",
                    unsafe_allow_html=True,
                )

    selected = st.selectbox(
        "Open visitor",
        [visitor["face_id"] for visitor in visitors],
        format_func=lambda face_id: f"Face {face_id}",
    )
    detail = api_get(f"/visitors/{selected}")
    if not detail:
        return
    metrics = st.columns(4)
    metrics[0].metric("Face", detail["face_id"])
    metrics[1].metric("Visits", detail["visit_count"])
    metrics[2].metric("Best quality", round(detail["best_quality"], 3))
    metrics[3].metric("Active", "yes" if detail["is_active"] else "no")
    st.caption(
        f"First seen: {detail['first_seen_at']} · Last seen: {detail['last_seen_at']}"
    )
    if detail["visits"]:
        section("Visit timeline", "Entry and exit history")
        st.dataframe(
            [
                {
                    "Visit": visit["visit_id"],
                    "Entry": visit["entry_at"],
                    "Exit": visit["exit_at"] or "open",
                    "Status": visit["status"],
                    "Source": visit["source_id"],
                }
                for visit in detail["visits"]
            ],
            width="stretch",
            hide_index=True,
        )


def _bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def system_page() -> None:
    page_header(
        "System",
        "Host, storage and inference-runtime diagnostics for the API process.",
        ("CPU / memory", "ONNX providers", "DB size"),
    )
    data = api_get("/system")
    if not data:
        return
    memory = data["memory"]
    metrics = st.columns(4)
    metrics[0].metric("CPU", f"{data['cpu']['percent']:.0f}%")
    metrics[1].metric("Memory", f"{memory['percent']:.0f}%")
    metrics[2].metric("DB size", _bytes(data["database"]["size_bytes"]))
    metrics[3].metric("Uptime", f"{data['uptime_seconds'] / 60:.1f} min")
    st.progress(
        min(max(memory["percent"] / 100, 0.0), 1.0),
        text=f"Memory used: {_bytes(memory['used_bytes'])} / {_bytes(memory['total_bytes'])}",
    )
    st.caption(
        f"{data['platform']} · Python {data['python']} · "
        f"GPU: {'available' if data['gpu']['available'] else 'CPU only'}"
    )
    section("Inference runtime", "ONNX execution providers")
    st.write(", ".join(data["gpu"]["providers"]))
    st.json(data["models"])


def event_log() -> None:
    page_header(
        "Event log",
        "Every entry and exit, with the crop that anchors the audit trail.",
        ("Filterable", "CSV export", "DB-backed"),
    )
    with st.container(border=True):
        left, middle, right = st.columns([1, 1, 1])
        event_type = left.selectbox("Type", ["all", "entry", "exit"])
        date = middle.text_input("Date (YYYY-MM-DD)", placeholder="2025-06-20")
        face_id = right.text_input("Face ID", placeholder="any")
    params: dict[str, object] = {"limit": 50}
    if event_type != "all":
        params["event_type"] = event_type
    if date:
        params["date"] = date
    if face_id:
        try:
            params["face_id"] = int(face_id)
        except ValueError:
            st.warning("Face ID must be an integer.")
    data = api_get("/events", params)
    if data:
        st.caption(f"{data['total']} matching events")
        _event_table(data["items"])
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["event_id", "type", "occurred_at", "face_id", "visit_id", "source"]
        )
        for event in data["items"]:
            writer.writerow(
                [
                    event["event_id"],
                    event["event_type"],
                    event["occurred_at"],
                    event["face_id"] or "",
                    event["visit_id"],
                    event["source_id"],
                ]
            )
        st.download_button(
            "Download CSV",
            buffer.getvalue(),
            "events.csv",
            "text/csv",
        )


def system_logs() -> None:
    page_header(
        "System logs",
        "Tail and search the mandatory audit log without touching the pipeline.",
        ("events.log", "Filter", "Download"),
    )
    with st.container(border=True):
        text = st.text_input("Filter text", placeholder="FACE_REGISTERED")
        lines = st.slider("Lines", 50, 1000, 200, step=50)
    data = api_get("/logs/tail", {"lines": lines, "text": text or None})
    if data:
        st.caption(data["path"])
        st.code("\n".join(data["lines"]) or "No matching log lines.", language="text")
        st.download_button(
            "Download tail",
            "\n".join(data["lines"]),
            "events.log",
            "text/plain",
        )


def sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="brand">
          <div class="brand-mark">IF</div>
          <div>
            <div class="brand-title">Face Tracker</div>
            <div class="brand-sub">Operations console</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    view = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Live Monitor",
            "Sources & Jobs",
            "Visitors",
            "Event log",
            "Config",
            "System",
            "System logs",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    health = _try_get("/health")
    if health:
        status = "API online" if health.get("database_ok") else "API degraded"
    else:
        status = "API offline"
    st.sidebar.markdown(
        f'<div class="sidebar-meta"><strong>{status}</strong><br>'
        f"{html.escape(API_URL)}</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Read-only client. The tracker runs independently.")
    return view


view = sidebar()
if view == "Dashboard":
    dashboard()
elif view == "Live Monitor":
    live_monitor()
elif view == "Sources & Jobs":
    sources_jobs()
elif view == "Visitors":
    visitors_page()
elif view == "Event log":
    event_log()
elif view == "Config":
    config_page()
elif view == "System":
    system_page()
else:
    system_logs()
