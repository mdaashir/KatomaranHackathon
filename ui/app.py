"""Streamlit client for the FastAPI control layer.

This app never imports the pipeline, database or storage modules. It is a
plain HTTP client; the API can be down without affecting the tracker.
"""

from __future__ import annotations

import os

import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
REFRESH_SECONDS = None if os.getenv("FACETRACKER_REFRESH") == "0" else 5
LIVE_REFRESH_SECONDS = None if REFRESH_SECONDS is None else 3

st.set_page_config(
    page_title="Intelligent Face Tracker",
    page_icon="👤",
    layout="wide",
)


def api_get(path: str, params: dict[str, object] | None = None):
    try:
        response = requests.get(
            f"{API_URL}{path}",
            params=params,
            timeout=3,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        st.error(f"API unavailable at {API_URL}: {exc}")
        return None


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
        st.error(f"API request failed at {API_URL}{path}: {exc}")
        return None


def _image_url(path: str | None) -> str | None:
    if not path:
        return None
    return f"{API_URL}{path}" if path.startswith("/") else f"{API_URL}/{path}"


def _event_table(events: list[dict[str, object]]) -> None:
    if not events:
        st.info("No events match the current filters.")
        return
    st.dataframe(
        [
            {
                "Time": event["occurred_at"],
                "Type": event["event_type"],
                "Face": event["face_id"] if event["face_id"] else "anonymous",
                "Visit": event["visit_id"],
                "Track": event["track_id"],
                "Source": event["source_id"],
            }
            for event in events
        ],
        use_container_width=True,
        hide_index=True,
    )
    cols = st.columns(6)
    for index, event in enumerate(events[:6]):
        with cols[index % 6]:
            url = _image_url(event.get("image_url"))
            if url:
                st.image(url, caption=f"#{event['event_id']}", use_container_width=True)


@st.fragment(run_every=REFRESH_SECONDS)
def dashboard() -> None:
    stats = api_get("/stats")
    if stats:
        cols = st.columns(6)
        values = (
            ("Unique faces", stats["unique_faces"]),
            ("Visits", stats["visits"]),
            ("In frame", stats["currently_in_frame"]),
            ("Entries", stats["entries"]),
            ("Exits", stats["exits"]),
            ("Anonymous", stats["anonymous_visits"]),
        )
        for column, (label, value) in zip(cols, values):
            column.metric(label, value)
        st.caption(
            f"First event: {stats['first_event_at'] or '—'} · "
            f"Last event: {stats['last_event_at'] or '—'}"
        )
    events = api_get("/events", {"limit": 10})
    if events:
        st.subheader("Recent events")
        _event_table(events["items"])


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def live_refresh() -> None:
    jobs = api_get("/jobs")
    if jobs:
        running = [job for job in jobs["jobs"] if job["status"] == "running"]
        if running:
            st.caption(f"Running: {running[0]['label']} ({running[0]['job_id']})")
        else:
            st.caption("No pipeline job is running.")
    stats = api_get("/stats")
    if stats:
        cols = st.columns(4)
        cols[0].metric("Unique faces", stats["unique_faces"])
        cols[1].metric("In frame", stats["currently_in_frame"])
        cols[2].metric("Entries", stats["entries"])
        cols[3].metric("Exits", stats["exits"])
    events = api_get("/events", {"limit": 8})
    if events:
        st.subheader("Activity")
        _event_table(events["items"])


def live_monitor() -> None:
    st.subheader("Live monitor")
    components.html(
        f'<img src="{API_URL}/stream/live.mjpg" '
        'style="width:100%;max-height:520px;object-fit:contain;background:#111" />',
        height=530,
        scrolling=False,
    )
    live_refresh()


def sources_jobs() -> None:
    st.subheader("Sources & jobs")
    uploads = st.file_uploader(
        "Upload videos",
        type=["mp4", "avi", "mov", "mkv", "m4v"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Upload selected", type="primary"):
        files = [
            (
                "files",
                (
                    upload.name,
                    upload.getvalue(),
                    upload.type or "video/mp4",
                ),
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
    if names:
        st.caption("Available videos: " + ", ".join(names[:12]))
    else:
        st.info("No videos in the source folder yet.")

    left, right = st.columns([1, 1])
    with left:
        source_mode = st.radio(
            "Start from",
            ["file", "folder", "rtsp"],
            horizontal=True,
        )
        if source_mode == "file":
            selected = st.selectbox("Video file", names or ["(upload one first)"])
            payload = {"source_type": "file", "path": selected}
        elif source_mode == "folder":
            selected_folder = st.selectbox(
                "Folder (all videos, recursive)",
                folder_names or ["(no folders)"],
            )
            payload = {"source_type": "folder", "path": selected_folder}
        else:
            rtsp_url = st.text_input(
                "RTSP URL",
                placeholder="rtsp://user:pass@camera/stream",
                type="password",
            )
            payload = {"source_type": "rtsp", "url": rtsp_url}
    with right:
        st.write("The pipeline runs in the API process on one worker thread.")
        if st.button("Start job", type="primary"):
            result = api_post("/jobs", json_data=payload)
            if result:
                st.success(f"Queued job {result['job_id']}.")

    jobs = api_get("/jobs")
    if jobs and jobs["jobs"]:
        st.subheader("Job queue")
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
            use_container_width=True,
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
    st.subheader("Config")
    data = api_get("/config")
    if not data:
        return
    base = data["base"]
    effective = data["effective"]
    st.caption(
        "Runtime overrides apply to the next job. frame_skip, source, storage "
        "and logging stay owned by config.json."
    )
    st.metric("frame_skip (config.json)", data["effective"]["frame_skip"])

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
        use_container_width=True,
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


def event_log() -> None:
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


def system_logs() -> None:
    text = st.text_input("Filter text", placeholder="FACE_REGISTERED")
    lines = st.slider("Lines", 50, 1000, 200, step=50)
    data = api_get("/logs/tail", {"lines": lines, "text": text or None})
    if data:
        st.caption(data["path"])
        st.code("\n".join(data["lines"]) or "No matching log lines.", language="text")


st.sidebar.title("Face Tracker")
view = st.sidebar.radio(
    "View",
    [
        "Dashboard",
        "Live Monitor",
        "Sources & Jobs",
        "Event log",
        "Config",
        "System logs",
    ],
)
st.sidebar.caption(f"API: {API_URL}")
st.sidebar.caption("Read-only client. The tracker runs independently.")

if view == "Dashboard":
    dashboard()
elif view == "Live Monitor":
    live_monitor()
elif view == "Sources & Jobs":
    sources_jobs()
elif view == "Event log":
    event_log()
elif view == "Config":
    config_page()
else:
    system_logs()
