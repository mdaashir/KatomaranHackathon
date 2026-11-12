"""Verify one e2e run: DB invariants, images, expected counts and log events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

REQUIRED_LOG_EVENTS = (
    "FACE_DETECTED",
    "EMBEDDING_GENERATED",
    "FACE_REGISTERED",
    "FACE_RECOGNIZED",
    "TRACK_STARTED",
    "TRACK_LOST",
    "ENTRY",
    "EXIT",
    "SUMMARY",
)
EXACT_KEYS = (
    "unique_faces",
    "visits",
    "open_visits",
    "anonymous_visits",
    "entries",
    "exits",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--expected", required=True)
    return parser


def summary(connection: sqlite3.Connection) -> dict[str, object]:
    def scalar(query: str) -> int:
        return int(connection.execute(query).fetchone()[0])

    first, last = connection.execute(
        "SELECT MIN(occurred_at), MAX(occurred_at) FROM events"
    ).fetchone()
    return {
        "unique_faces": scalar("SELECT COUNT(*) FROM faces WHERE is_active = 1"),
        "visits": scalar("SELECT COUNT(*) FROM visits"),
        "open_visits": scalar("SELECT COUNT(*) FROM visits WHERE status = 'open'"),
        "anonymous_visits": scalar(
            "SELECT COUNT(*) FROM visits WHERE is_anonymous = 1"
        ),
        "entries": scalar("SELECT COUNT(*) FROM events WHERE event_type = 'entry'"),
        "exits": scalar("SELECT COUNT(*) FROM events WHERE event_type = 'exit'"),
        "first_event_at": first,
        "last_event_at": last,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db)
    log_path = Path(args.log_dir) / "events.log"
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: dict[str, object] = {}

    if not db_path.is_file():
        errors.append(f"database missing: {db_path}")
    if not log_path.is_file():
        errors.append(f"log missing: {log_path}")
    if errors:
        sys.stdout.write(json.dumps({"ok": False, "errors": errors}, indent=2) + "\n")
        return 1

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        checks["integrity"] = integrity
        if integrity != "ok":
            errors.append(f"integrity_check returned {integrity!r}")

        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        checks["foreign_key_violations"] = len(foreign_keys)
        if foreign_keys:
            errors.append(f"{len(foreign_keys)} foreign-key violations")

        counts = summary(connection)
        checks["summary"] = counts

        for key in EXACT_KEYS:
            if key in expected and expected[key] is not None:
                if counts[key] != expected[key]:
                    errors.append(f"{key}: expected {expected[key]}, got {counts[key]}")

        for key in ("entries", "exits"):
            minimum = expected.get(f"{key}_min")
            if minimum is not None and counts[key] < minimum:
                errors.append(f"{key}: expected at least {minimum}, got {counts[key]}")

        if expected.get("balanced") and counts["entries"] != counts["exits"]:
            errors.append(f"entries ({counts['entries']}) != exits ({counts['exits']})")

        if counts["open_visits"] != 0:
            errors.append(f"{counts['open_visits']} open visit(s)")

        bad_visits = connection.execute(
            """
            SELECT v.visit_id, COUNT(e.event_id) AS event_count
            FROM visits v
            LEFT JOIN events e ON e.visit_id = v.visit_id
            GROUP BY v.visit_id
            HAVING event_count <> 2
            """
        ).fetchall()
        checks["visits_without_two_events"] = len(bad_visits)
        if bad_visits:
            errors.append(
                "visits without exactly two events: "
                + ", ".join(str(row["visit_id"]) for row in bad_visits)
            )

        missing_images = 0
        event_rows = connection.execute(
            "SELECT event_id, image_path FROM events"
        ).fetchall()
        for row in event_rows:
            if not Path(row["image_path"]).is_file():
                missing_images += 1
                errors.append(
                    f"event {row['event_id']} image missing: {row['image_path']}"
                )
        checks["event_images"] = len(event_rows)
        checks["missing_event_images"] = missing_images

        for row in connection.execute(
            "SELECT visit_id, entry_image_path, exit_image_path FROM visits"
        ).fetchall():
            for key in ("entry_image_path", "exit_image_path"):
                value = row[key]
                if value is not None and not Path(value).is_file():
                    errors.append(f"visit {row['visit_id']} image missing: {value}")

        log_text = log_path.read_text(encoding="utf-8")
        missing_events = [
            event for event in REQUIRED_LOG_EVENTS if f"| {event} " not in log_text
        ]
        checks["missing_log_events"] = missing_events
        if missing_events:
            errors.append("missing log event(s): " + ", ".join(missing_events))
    finally:
        connection.close()

    report = {
        "ok": not errors,
        "database": str(db_path),
        "log": str(log_path),
        "expected": expected,
        "checks": checks,
        "errors": errors,
    }
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
