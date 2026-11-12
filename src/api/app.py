"""FastAPI control/read layer for the tracker dashboard."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal
import uuid

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from src.api.jobs import (
    JobManager,
    VIDEO_SUFFIXES,
    input_root_for,
    list_videos,
)
from src.api.live import LiveFrameBuffer
from src.api.models import JobRequest
from src.api.reader import ReadModel
from src.config import AppConfig, load_config

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def create_app(
    config: AppConfig | None = None,
    *,
    db_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    input_root: str | Path | None = None,
    job_manager: JobManager | None = None,
    frame_buffer: LiveFrameBuffer | None = None,
) -> FastAPI:
    """Build the API app.

    The UI is a separate HTTP client. The API owns the optional pipeline
    worker, but the headless ``main.py`` path remains fully independent.
    """

    effective = config or load_config("config.json")
    reader = ReadModel(
        db_path or effective.storage.db_path,
        log_dir or effective.storage.log_dir,
        config=effective,
    )
    buffer = frame_buffer or (
        job_manager.frame_buffer if job_manager is not None else LiveFrameBuffer()
    )
    manager = job_manager or JobManager(
        effective,
        input_root=input_root or input_root_for(effective),
        frame_buffer=buffer,
    )
    upload_root = Path(input_root or manager.input_root).resolve()

    app = FastAPI(
        title="Intelligent Face Tracker API",
        version="0.2.0",
        description="Control and read layer over the tracker database.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return reader.health()

    @app.get("/stats")
    def stats() -> dict[str, object]:
        return reader.stats()

    @app.get("/events")
    def events(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
        event_type: Literal["entry", "exit"] | None = Query(None),
        face_id: int | None = Query(None, ge=1),
    ) -> dict[str, object]:
        return reader.events(
            limit=limit,
            offset=offset,
            date=date,
            event_type=event_type,
            face_id=face_id,
        )

    @app.get("/events/stream")
    async def events_stream(request: Request) -> StreamingResponse:
        async def generate():
            last_id = reader.latest_event_id()
            while not await request.is_disconnected():
                for item in reader.events_after(last_id):
                    last_id = int(item["event_id"])
                    yield (f"id: {last_id}\nevent: event\ndata: {json.dumps(item)}\n\n")
                yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/logs/tail")
    def logs_tail(
        lines: int = Query(200, ge=1, le=1000),
        text: str | None = Query(None),
        level: str | None = Query(None),
    ) -> dict[str, object]:
        return reader.log_tail(lines=lines, text=text, level=level)

    @app.get("/images/{event_id}")
    def image(event_id: int) -> FileResponse:
        resolved = reader.image_path(event_id)
        if resolved is None:
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(resolved, media_type="image/jpeg")

    @app.get("/config")
    def get_config() -> dict[str, object]:
        payload = reader.config_payload()
        if payload is None:
            raise HTTPException(status_code=404, detail="config not loaded")
        return {"config": payload}

    @app.get("/videos")
    def videos() -> dict[str, object]:
        return {"videos": list_videos(upload_root)}

    @app.post("/upload")
    async def upload(files: list[UploadFile] = File(...)) -> dict[str, object]:
        upload_root.mkdir(parents=True, exist_ok=True)
        uploaded = []
        for upload_file in files:
            safe_name = Path(upload_file.filename or "").name
            suffix = Path(safe_name).suffix.lower()
            if not safe_name or suffix not in VIDEO_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"unsupported video type: {safe_name or 'unknown'}",
                )
            target = upload_root / f"{uuid.uuid4().hex[:12]}{suffix}"
            temporary = target.with_suffix(target.suffix + ".part")
            size = 0
            try:
                with temporary.open("wb") as output:
                    while chunk := await upload_file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail="upload exceeds 2 GiB",
                            )
                        output.write(chunk)
                temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            uploaded.append(
                {
                    "name": target.name,
                    "size": size,
                    "uploaded_at": (
                        datetime.now(timezone.utc)
                        .isoformat(timespec="seconds")
                        .replace("+00:00", "Z")
                    ),
                }
            )
        return {"uploaded": uploaded, "videos": list_videos(upload_root)}

    @app.get("/jobs")
    def jobs() -> dict[str, object]:
        return {"jobs": [job.to_dict() for job in manager.list()]}

    @app.get("/jobs/{job_id}")
    def job(job_id: str) -> dict[str, object]:
        found = manager.get(job_id)
        if found is None:
            raise HTTPException(status_code=404, detail="job not found")
        return found.to_dict()

    @app.post("/jobs", status_code=201)
    def create_job(request: JobRequest) -> dict[str, object]:
        try:
            created = manager.start(request)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return created.to_dict()

    @app.post("/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, object]:
        stopped = manager.stop(job_id)
        if stopped is None:
            raise HTTPException(status_code=404, detail="job not found")
        return stopped.to_dict()

    @app.get("/stream/live.mjpg")
    def live_stream() -> StreamingResponse:
        return StreamingResponse(
            manager.frame_buffer.mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


app = create_app()
