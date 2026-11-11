"""Generate a read-only HTML dashboard from the tracker database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.dashboard import DashboardBuilder
from src.storage import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--db", default=None, help="override the database path")
    parser.add_argument("--output", default="dashboard")
    parser.add_argument("--max-events", type=int, default=24)
    parser.add_argument("--max-visits", type=int, default=10)
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the generated page in the default browser",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = args.db or load_config(args.config).storage.db_path
    if not Path(db_path).is_file():
        sys.stderr.write(f"database not found: {db_path}\n")
        return 2

    database = Database(db_path)
    try:
        data = DashboardBuilder(
            database,
            args.output,
            max_events=args.max_events,
            max_visits=args.max_visits,
        ).build()
    except (sqlite3.Error, OSError, ValueError) as exc:
        sys.stderr.write(f"dashboard failed: {exc}\n")
        return 1
    finally:
        database.close()

    page = (Path(args.output) / "index.html").resolve()
    if args.open:
        webbrowser.open(page.as_uri())

    sys.stdout.write(
        json.dumps(
            {
                "output": str(page),
                "unique_faces": data.summary.unique_faces,
                "visits": data.summary.visits,
                "events": len(data.events),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
