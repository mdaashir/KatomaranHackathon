"""FastAPI read layer for the tracker dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.reader import ReadModel
from src.config import AppConfig, load_config


def create_app(
    config: AppConfig | None = None,
    *,
    db_path: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> FastAPI:
    """Build the read-only API app.

    The pipeline is never imported or started here. The API only reads the
    SQLite DB and ``events.log``; the UI is a separate HTTP client.
    """

    effective = config or load_config("config.json")
    reader = ReadModel(
        db_path or effective.storage.db_path,
        log_dir or effective.storage.log_dir,
        config=effective,
    )
    app = FastAPI(
        title="Intelligent Face Tracker API",
        version="0.1.0",
        description="Read-only control/read layer over the tracker database.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
        allow_methods=["GET"],
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

    return app


app = create_app()
