"""Read-only data access for the FastAPI control/read layer."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3

from src.config import AppConfig
from src.counting import UniqueCounter
from src.storage import Database


def _empty_summary() -> dict[str, object]:
    return {
        "unique_faces": 0,
        "visits": 0,
        "open_visits": 0,
        "anonymous_visits": 0,
        "entries": 0,
        "exits": 0,
        "first_event_at": None,
        "last_event_at": None,
        "currently_in_frame": 0,
    }


class ReadModel:
    """Read the tracker DB and log folder without owning the pipeline."""

    def __init__(
        self,
        db_path: str | Path,
        log_dir: str | Path,
        *,
        config: AppConfig | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._log_dir = Path(log_dir)
        self._config = config

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def health(self) -> dict[str, object]:
        """Return API/DB/log liveness."""

        database_ok = False
        integrity = None
        if self._db_path.is_file():
            try:
                connection = sqlite3.connect(str(self._db_path))
                try:
                    integrity = connection.execute("PRAGMA integrity_check").fetchone()[
                        0
                    ]
                    database_ok = integrity == "ok"
                finally:
                    connection.close()
            except sqlite3.Error:
                database_ok = False
        return {
            "status": "ok" if database_ok else "degraded",
            "database": str(self._db_path),
            "database_ok": database_ok,
            "integrity": integrity,
            "log_dir": str(self._log_dir),
        }

    def stats(self) -> dict[str, object]:
        """Return DB-derived counters for the dashboard."""

        if not self._db_path.is_file():
            return _empty_summary()
        database = Database(self._db_path)
        try:
            summary = UniqueCounter(database).get_summary().to_dict()
        finally:
            database.close()
        summary["currently_in_frame"] = summary["open_visits"]
        return summary

    def events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        date: str | None = None,
        event_type: str | None = None,
        face_id: int | None = None,
    ) -> dict[str, object]:
        """Return a page of entry/exit events with image URLs."""

        if not self._db_path.is_file():
            return {"items": [], "total": 0, "limit": limit, "offset": offset}

        clauses: list[str] = []
        params: list[object] = []
        if date:
            clauses.append("substr(e.occurred_at, 1, 10) = ?")
            params.append(date)
        if event_type:
            clauses.append("e.event_type = ?")
            params.append(event_type)
        if face_id is not None:
            clauses.append("e.face_id = ?")
            params.append(face_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        try:
            total = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM events e
                JOIN visits v ON v.visit_id = e.visit_id
                {where}
                """,
                params,
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT e.event_id, e.event_type, e.occurred_at, e.face_id,
                       e.visit_id, e.source_id, e.image_path, e.confidence,
                       e.is_anonymous, v.track_id
                FROM events e
                JOIN visits v ON v.visit_id = e.visit_id
                {where}
                ORDER BY e.event_id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        finally:
            connection.close()

        items = [self._event_item(row) for row in rows]
        return {"items": items, "total": int(total), "limit": limit, "offset": offset}

    def events_after(
        self, after_id: int, *, limit: int = 50
    ) -> list[dict[str, object]]:
        """Return new events in ascending ID order for the activity feed."""

        if not self._db_path.is_file():
            return []
        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT e.event_id, e.event_type, e.occurred_at, e.face_id,
                       e.visit_id, e.source_id, e.image_path, e.confidence,
                       e.is_anonymous, v.track_id
                FROM events e
                JOIN visits v ON v.visit_id = e.visit_id
                WHERE e.event_id > ?
                ORDER BY e.event_id ASC
                LIMIT ?
                """,
                (after_id, limit),
            ).fetchall()
        finally:
            connection.close()
        return [self._event_item(row) for row in rows]

    def latest_event_id(self) -> int:
        """Return the newest event ID, or zero for an empty/missing DB."""

        if not self._db_path.is_file():
            return 0
        connection = sqlite3.connect(str(self._db_path))
        try:
            value = connection.execute("SELECT MAX(event_id) FROM events").fetchone()[0]
        finally:
            connection.close()
        return int(value or 0)

    @staticmethod
    def _event_item(row: sqlite3.Row) -> dict[str, object]:
        return {
            "event_id": int(row["event_id"]),
            "event_type": str(row["event_type"]),
            "occurred_at": str(row["occurred_at"]),
            "face_id": (int(row["face_id"]) if row["face_id"] is not None else None),
            "visit_id": int(row["visit_id"]),
            "track_id": int(row["track_id"]),
            "source_id": str(row["source_id"]),
            "confidence": (
                float(row["confidence"]) if row["confidence"] is not None else None
            ),
            "is_anonymous": bool(row["is_anonymous"]),
            "image_url": f"/images/{row['event_id']}",
        }

    def visitors(self, *, limit: int = 100, offset: int = 0) -> dict[str, object]:
        """Return registered faces with visit counts and thumbnails."""

        if not self._db_path.is_file():
            return {"items": [], "total": 0, "limit": limit, "offset": offset}
        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        try:
            total = connection.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            rows = connection.execute(
                """
                SELECT f.face_id, f.created_at, f.first_seen_at, f.model_name,
                       f.best_quality, f.is_active,
                       COUNT(v.visit_id) AS visit_count,
                       COALESCE(MAX(COALESCE(v.exit_at, v.entry_at)),
                                f.first_seen_at) AS last_seen_at,
                       (SELECT e.event_id FROM events e
                        WHERE e.face_id = f.face_id AND e.event_type = 'entry'
                        ORDER BY e.event_id DESC LIMIT 1) AS thumbnail_event_id
                FROM faces f
                LEFT JOIN visits v ON v.face_id = f.face_id
                GROUP BY f.face_id
                ORDER BY last_seen_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        finally:
            connection.close()
        return {
            "items": [self._visitor_item(row) for row in rows],
            "total": int(total),
            "limit": limit,
            "offset": offset,
        }

    def visitor(self, face_id: int) -> dict[str, object] | None:
        """Return one face and its visit timeline, or ``None``."""

        if not self._db_path.is_file():
            return None
        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        try:
            face = connection.execute(
                """
                SELECT f.face_id, f.created_at, f.first_seen_at, f.model_name,
                       f.best_quality, f.is_active,
                       COUNT(v.visit_id) AS visit_count,
                       COALESCE(MAX(COALESCE(v.exit_at, v.entry_at)),
                                f.first_seen_at) AS last_seen_at,
                       (SELECT e.event_id FROM events e
                        WHERE e.face_id = f.face_id AND e.event_type = 'entry'
                        ORDER BY e.event_id DESC LIMIT 1) AS thumbnail_event_id
                FROM faces f
                LEFT JOIN visits v ON v.face_id = f.face_id
                WHERE f.face_id = ?
                GROUP BY f.face_id
                """,
                (face_id,),
            ).fetchone()
            if face is None:
                return None
            visits = connection.execute(
                """
                SELECT v.visit_id, v.face_id, v.track_id, v.source_id,
                       v.entry_at, v.exit_at, v.status, v.is_anonymous,
                       (SELECT e.event_id FROM events e
                        WHERE e.visit_id = v.visit_id AND e.event_type = 'entry'
                        ORDER BY e.event_id DESC LIMIT 1) AS entry_event_id,
                       (SELECT e.event_id FROM events e
                        WHERE e.visit_id = v.visit_id AND e.event_type = 'exit'
                        ORDER BY e.event_id DESC LIMIT 1) AS exit_event_id
                FROM visits v
                WHERE v.face_id = ?
                ORDER BY v.entry_at DESC, v.visit_id DESC
                """,
                (face_id,),
            ).fetchall()
        finally:
            connection.close()
        payload = self._visitor_item(face)
        payload["visits"] = [self._visit_item(row) for row in visits]
        return payload

    @staticmethod
    def _visitor_item(row: sqlite3.Row) -> dict[str, object]:
        thumbnail = row["thumbnail_event_id"]
        return {
            "face_id": int(row["face_id"]),
            "created_at": str(row["created_at"]),
            "first_seen_at": str(row["first_seen_at"]),
            "last_seen_at": str(row["last_seen_at"]),
            "model_name": str(row["model_name"]),
            "best_quality": float(row["best_quality"]),
            "is_active": bool(row["is_active"]),
            "visit_count": int(row["visit_count"]),
            "thumbnail_url": f"/images/{thumbnail}" if thumbnail is not None else None,
        }

    @staticmethod
    def _visit_item(row: sqlite3.Row) -> dict[str, object]:
        entry = row["entry_event_id"]
        exit_event = row["exit_event_id"]
        return {
            "visit_id": int(row["visit_id"]),
            "face_id": int(row["face_id"]),
            "track_id": int(row["track_id"]),
            "source_id": str(row["source_id"]),
            "entry_at": str(row["entry_at"]),
            "exit_at": str(row["exit_at"]) if row["exit_at"] is not None else None,
            "status": str(row["status"]),
            "is_anonymous": bool(row["is_anonymous"]),
            "entry_image_url": f"/images/{entry}" if entry is not None else None,
            "exit_image_url": (
                f"/images/{exit_event}" if exit_event is not None else None
            ),
        }

    def log_tail(
        self,
        *,
        lines: int = 200,
        text: str | None = None,
        level: str | None = None,
    ) -> dict[str, object]:
        """Return a filtered tail of ``events.log``."""

        path = self._log_dir / "events.log"
        if not path.is_file():
            return {"path": str(path), "lines": []}
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = raw[-max(lines * 5, 1000) :]
        if text:
            needle = text.casefold()
            selected = [line for line in selected if needle in line.casefold()]
        if level:
            wanted = level.upper()
            selected = [
                line
                for line in selected
                if len(line.split("|")) > 1 and line.split("|")[1].strip() == wanted
            ]
        return {"path": str(path), "lines": selected[-lines:]}

    def image_path(self, event_id: int) -> Path | None:
        """Resolve the stored image for one event ID."""

        if not self._db_path.is_file():
            return None
        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                "SELECT image_path FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        root = self._log_dir.resolve()
        candidate = Path(str(row["image_path"])).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
        return candidate

    def config_payload(self) -> dict[str, object] | None:
        """Return the effective config with RTSP credentials redacted."""

        if self._config is None:
            return None
        payload = asdict(self._config)
        source = payload.get("source")
        if isinstance(source, dict):
            url = source.get("rtsp_url")
            if isinstance(url, str) and url:
                source["rtsp_url"] = re.sub(r"(rtsp://)[^/@]+@", r"\1***@", url)
        return payload

    def generated_at(self) -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
