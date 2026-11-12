"""Streamlit client for the read-only FastAPI layer.

This app never imports the pipeline, database or storage modules. It is a
plain HTTP client; the API can be down without affecting the tracker.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

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


@st.fragment(run_every=5)
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
view = st.sidebar.radio("View", ["Dashboard", "Event log", "System logs"])
st.sidebar.caption(f"API: {API_URL}")
st.sidebar.caption("Read-only client. The tracker runs independently.")

if view == "Dashboard":
    dashboard()
elif view == "Event log":
    event_log()
else:
    system_logs()
