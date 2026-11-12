"""Verify DB invariants and every referenced image for a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.counting import UniqueCounter
from src.storage import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--db",
        default=None,
        help="override the database path from config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = args.db
    if db_path is None:
        db_path = load_config(args.config).storage.db_path

    database = Database(db_path)
    database.initialize()
    connection = database.connect()
    errors: list[str] = []
    checks: dict[str, object] = {}

    integrity = database.integrity_check()
    checks["integrity"] = integrity
    if integrity != "ok":
        errors.append(f"integrity_check returned {integrity!r}")

    foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    checks["foreign_key_violations"] = len(foreign_keys)
    if foreign_keys:
        errors.append(f"{len(foreign_keys)} foreign-key violations")

    unique = UniqueCounter(database).get_unique_count()
    checks["unique_faces"] = unique

    open_visits = connection.execute(
        "SELECT COUNT(*) FROM visits WHERE status = 'open'"
    ).fetchone()[0]
    checks["open_visits"] = int(open_visits)
    if open_visits:
        errors.append(f"{open_visits} open visit(s)")

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

    bad_status = connection.execute(
        """
        SELECT visit_id FROM visits
        WHERE (status = 'open' AND exit_at IS NOT NULL)
           OR (status = 'closed' AND exit_at IS NULL)
        """
    ).fetchall()
    checks["inconsistent_visit_status"] = len(bad_status)
    if bad_status:
        errors.append(f"{len(bad_status)} visits have inconsistent status")

    bad_times = connection.execute(
        """
        SELECT visit_id FROM visits
        WHERE exit_at IS NOT NULL AND exit_at < entry_at
        """
    ).fetchall()
    checks["visits_with_exit_before_entry"] = len(bad_times)
    if bad_times:
        errors.append(f"{len(bad_times)} visits exit before entry")

    missing_images = 0
    event_images = connection.execute(
        "SELECT event_id, image_path FROM events"
    ).fetchall()
    for row in event_images:
        if not Path(row["image_path"]).is_file():
            missing_images += 1
            errors.append(f"event {row['event_id']} image missing: {row['image_path']}")
    checks["event_images"] = len(event_images)
    checks["missing_event_images"] = missing_images

    visit_images = connection.execute(
        """
        SELECT visit_id, entry_image_path, exit_image_path FROM visits
        WHERE entry_image_path IS NOT NULL OR exit_image_path IS NOT NULL
        """
    ).fetchall()
    missing_visit_images = 0
    for row in visit_images:
        for key in ("entry_image_path", "exit_image_path"):
            path = row[key]
            if path is not None and not Path(path).is_file():
                missing_visit_images += 1
                errors.append(f"visit {row['visit_id']} image missing: {path}")
    checks["missing_visit_images"] = missing_visit_images

    entries = connection.execute(
        "SELECT COUNT(*) FROM events WHERE event_type = 'entry'"
    ).fetchone()[0]
    exits = connection.execute(
        "SELECT COUNT(*) FROM events WHERE event_type = 'exit'"
    ).fetchone()[0]
    checks["entries"] = int(entries)
    checks["exits"] = int(exits)
    if entries != exits:
        errors.append(f"entries ({entries}) != exits ({exits})")

    report = {
        "ok": not errors,
        "database": str(db_path),
        "checks": checks,
        "errors": errors,
    }
    database.close()
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
