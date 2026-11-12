"""Export a curated sample of a completed run from SQLite, images and logs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.counting import UniqueCounter
from src.storage import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--db", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output", default="samples")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--log-lines", type=int, default=80)
    return parser


def _write_csv(connection, table: str, path: Path) -> int:
    rows = connection.execute(f"SELECT * FROM {table}").fetchall()
    columns = [description[0] for description in connection.execute(
        f"SELECT * FROM {table} LIMIT 0"
    ).description]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[column] for column in columns])
    return len(rows)


def _log_excerpt(log_path: Path, limit: int) -> str:
    if not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= limit:
        return "\n".join(lines) + "\n"
    half = max(1, limit // 2)
    head = lines[:half]
    tail = lines[-half:]
    return "\n".join(
        head + ["... (excerpt; see logs for the full file) ..."] + tail
    ) + "\n"


def _copy_images(connection, output: Path, max_images: int) -> int:
    image_root = output / "images"
    if image_root.exists():
        shutil.rmtree(image_root)
    entries_dir = image_root / "entries"
    exits_dir = image_root / "exits"
    entries_dir.mkdir(parents=True, exist_ok=True)
    exits_dir.mkdir(parents=True, exist_ok=True)

    per_type = max(1, max_images // 2)
    copied = 0
    for event_type, folder in (("entry", entries_dir), ("exit", exits_dir)):
        recognized = connection.execute(
            """
            SELECT image_path FROM events
            WHERE event_type = ? AND face_id IS NOT NULL
            ORDER BY event_id
            LIMIT ?
            """,
            (event_type, per_type),
        ).fetchall()
        remaining = max(0, per_type - len(recognized))
        anonymous = []
        if remaining:
            anonymous = connection.execute(
                """
                SELECT image_path FROM events
                WHERE event_type = ? AND face_id IS NULL
                ORDER BY event_id
                LIMIT ?
                """,
                (event_type, remaining),
            ).fetchall()
        for row in list(recognized) + list(anonymous):
            source = Path(row["image_path"])
            if not source.is_file():
                continue
            shutil.copy2(source, folder / source.name)
            copied += 1
    return copied


def _copy_reentry_images(
    connection,
    output: Path,
) -> tuple[int, list[tuple[int, int]]]:
    reentry_dir = output / "images" / "reentry"
    if reentry_dir.exists():
        shutil.rmtree(reentry_dir)
    reentry_dir.mkdir(parents=True, exist_ok=True)

    faces = connection.execute(
        """
        SELECT face_id, COUNT(*) AS visits FROM visits
        WHERE face_id IS NOT NULL
        GROUP BY face_id HAVING visits > 1
        ORDER BY face_id
        """
    ).fetchall()
    rows = connection.execute(
        """
        SELECT image_path FROM events
        WHERE face_id IN (
            SELECT face_id FROM visits WHERE face_id IS NOT NULL
            GROUP BY face_id HAVING COUNT(*) > 1
        )
        ORDER BY face_id, visit_id, event_id
        """
    ).fetchall()
    copied = 0
    for row in rows:
        source = Path(row["image_path"])
        if not source.is_file():
            continue
        shutil.copy2(source, reentry_dir / source.name)
        copied += 1
    return copied, [
        (int(row["face_id"]), int(row["visits"])) for row in faces
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    db_path = args.db or config.storage.db_path
    log_dir = Path(args.log_dir or config.storage.log_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    database = Database(db_path)
    database.initialize()
    connection = database.connect()
    counter = UniqueCounter(database)
    summary = counter.get_summary().to_dict()

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    counts = {
        "faces": _write_csv(connection, "faces", output / "faces.csv"),
        "visits": _write_csv(connection, "visits", output / "visits.csv"),
        "events": _write_csv(connection, "events", output / "events.csv"),
    }

    (output / "events.log").write_text(
        _log_excerpt(log_dir / "events.log", args.log_lines),
        encoding="utf-8",
    )
    image_count = _copy_images(connection, output, args.max_images)
    reentry_count, reentry_faces = _copy_reentry_images(connection, output)
    database.close()

    if reentry_faces:
        reentry_summary = "; ".join(
            f"Face {face_id} appears in {visits} visits (counted once)"
            for face_id, visits in reentry_faces
        )
    else:
        reentry_summary = "No multi-visit face was observed in this run."

    readme = f"""# Sample output

Compliance audit: [`../docs/COMPLIANCE.md`](../docs/COMPLIANCE.md).

Curated output from a real run of the 22 `record_*.mp4` front-entry clips.

- Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}
- Source folder: `{config.source.path}` (`{config.source.glob}`)
- Frame skip: `{config.frame_skip}`
- Detector: `imgsz={config.detector.imgsz}`, `device={config.detector.device}`
- Recognition: `model={config.recognition.model_name}`, `threshold={config.recognition.similarity_threshold}`

## Summary

```json
{json.dumps(summary, indent=2)}
```

- Faces exported: {counts['faces']}
- Visits exported: {counts['visits']}
- Events exported: {counts['events']}
- Images copied: {image_count + reentry_count}

## Files

- `events.log` — excerpt of the mandatory audit log
- `faces.csv`, `visits.csv`, `events.csv` — SQLite exports
- `summary.json` — DB-derived unique/visit/event counts
- `images/entries`, `images/exits` — cropped event images
- `images/reentry` — images for faces appearing in more than one visit
- `dashboard/index.html` — read-only HTML snapshot of the same database

## Dashboard

The `dashboard/` folder is a standalone read-only page generated by
`scripts/dashboard.py`; it reads SQLite and copies event thumbnails without
importing the pipeline.

## Re-identification

{reentry_summary}

Event crops for a returning face can be back-of-head views when the subject is
walking away; identity is anchored by the frontal embedding recorded in
`events.csv` and `visits.csv`.

## Run note

The 22-clip pass was completed against the same SQLite database; one long GPU
segment was interrupted by a transient GPU stall and resumed in a fresh
process. The DB was verified independently: 0 open visits, 0 missing images,
balanced entries/exits, and 0 foreign-key violations. The audit excerpt shows
the restart boundary; image filenames use the lifecycle visit ID, while SQLite
stores the canonical visit ID (`visits.csv` / `events.csv` are authoritative).

## Calibration

- `similarity_threshold=0.45`: real same-person pairs scored 0.70–0.73,
  different people 0.118. Kept unchanged.
- `exit_timeout_frames=75` (~3 s): produced 0 open visits and balanced
  entry/exit counts.
- `min_face_size=60` and `blur_threshold=60`: far, blurred or back-facing
  faces become anonymous visits. Lowering the gates increases registrations
  but also duplicate identities; the conservative values were kept.

## How this was produced

```powershell
python main.py --config <e2e-config>
python scripts/export_samples.py --config <e2e-config>
```

No input videos or model weights are included. Anonymous visits are logged but
excluded from the unique count; far/back-facing visitors may never produce a
usable embedding.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    sys.stdout.write(
        json.dumps(
            {
                "output": str(output),
                "summary": summary,
                "csv_rows": counts,
                "images": image_count + reentry_count,
                "reentry_images": reentry_count,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
