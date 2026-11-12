"""Build a self-contained read-only HTML dashboard from SQLite.

The pipeline is never imported here. The builder only reads the database and
copies event images into the output folder, so it cannot affect tracking,
recognition or the unique count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
from pathlib import Path
import shutil

from src.counting import CounterSummary, UniqueCounter
from src.storage import Database


@dataclass(frozen=True)
class DashboardEvent:
    """One recent entry or exit shown on the dashboard."""

    event_id: int
    event_type: str
    occurred_at: str
    face_id: int | None
    visit_id: int
    track_id: int
    source_id: str
    thumbnail: str | None
    is_anonymous: bool
    confidence: float | None


@dataclass(frozen=True)
class DashboardVisit:
    """One recent visit shown on the dashboard."""

    visit_id: int
    face_id: int | None
    track_id: int
    source_id: str
    entry_at: str
    exit_at: str | None
    status: str
    is_anonymous: bool


@dataclass(frozen=True)
class DashboardData:
    """Everything needed to render one dashboard snapshot."""

    summary: CounterSummary
    events: tuple[DashboardEvent, ...]
    visits: tuple[DashboardVisit, ...]
    generated_at: str


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _face_label(face_id: int | None, is_anonymous: bool) -> str:
    if is_anonymous or face_id is None:
        return "Anonymous"
    return f"Face {face_id}"


def _format_confidence(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _badge(event_type: str) -> str:
    css = "entry" if event_type == "entry" else "exit"
    return f'<span class="badge {css}">{html.escape(event_type)}</span>'


def _thumb(event: DashboardEvent) -> str:
    if event.thumbnail is None:
        return '<div class="thumb missing">no image</div>'
    path = html.escape(event.thumbnail, quote=True)
    return (
        f'<a href="{path}" target="_blank" rel="noreferrer">'
        f'<img class="thumb" src="{path}" alt="Event {event.event_id} crop" '
        'loading="lazy"></a>'
    )


def _event_rows(events: tuple[DashboardEvent, ...]) -> str:
    if not events:
        return '<tr><td colspan="8" class="empty">No events recorded yet.</td></tr>'
    rows = []
    for event in events:
        face = _face_label(event.face_id, event.is_anonymous)
        source = html.escape(event.source_id, quote=True)
        rows.append(
            "<tr>"
            f"<td>{_thumb(event)}</td>"
            f"<td>{_badge(event.event_type)}</td>"
            f"<td>{html.escape(event.occurred_at)}</td>"
            f"<td>{html.escape(face)}</td>"
            f"<td>#{event.visit_id}</td>"
            f"<td>{event.track_id}</td>"
            f"<td>{_format_confidence(event.confidence)}</td>"
            f'<td class="source" title="{source}">{source}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def _visit_rows(visits: tuple[DashboardVisit, ...]) -> str:
    if not visits:
        return '<tr><td colspan="7" class="empty">No visits recorded yet.</td></tr>'
    rows = []
    for visit in visits:
        face = _face_label(visit.face_id, visit.is_anonymous)
        exit_at = visit.exit_at if visit.exit_at is not None else "—"
        source = html.escape(visit.source_id, quote=True)
        status_css = "open" if visit.status == "open" else "closed"
        rows.append(
            "<tr>"
            f"<td>#{visit.visit_id}</td>"
            f"<td>{html.escape(face)}</td>"
            f'<td><span class="badge {status_css}">'
            f"{html.escape(visit.status)}</span></td>"
            f"<td>{html.escape(visit.entry_at)}</td>"
            f"<td>{html.escape(exit_at)}</td>"
            f"<td>{visit.track_id}</td>"
            f'<td class="source" title="{source}">{source}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def _stat_cards(summary: CounterSummary) -> str:
    cards = (
        ("Unique faces", summary.unique_faces),
        ("Visits", summary.visits),
        ("Open visits", summary.open_visits),
        ("Anonymous", summary.anonymous_visits),
        ("Entries", summary.entries),
        ("Exits", summary.exits),
    )
    return "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="number">{value}</div></div>'
        for label, value in cards
    )


def render_dashboard(data: DashboardData) -> str:
    """Return a complete standalone HTML document."""

    first = data.summary.first_event_at or "—"
    last = data.summary.last_event_at or "—"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intelligent Face Tracker — Dashboard</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f4f5f7;
  --panel: #ffffff;
  --fg: #172033;
  --muted: #667085;
  --line: #e4e7ec;
  --header: #111827;
  --accent: #2563eb;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}}
header {{
  background: var(--header);
  color: #ffffff;
  padding: 26px 24px;
}}
header .inner {{ max-width: 1240px; margin: 0 auto; }}
header h1 {{ margin: 0 0 6px; font-size: 26px; letter-spacing: -0.02em; }}
header p {{ margin: 0; color: #cbd5e1; font-size: 14px; }}
main {{ max-width: 1240px; margin: 0 auto; padding: 22px 24px 48px; }}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 20px;
}}
.card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
}}
.card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
.card .number {{ font-size: 30px; font-weight: 700; margin-top: 4px; }}
.grid {{
  display: grid;
  grid-template-columns: 1.45fr 1fr;
  gap: 20px;
  align-items: start;
}}
@media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} }}
section {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 20px;
}}
section h2 {{ margin: 0; padding: 14px 16px; font-size: 15px; border-bottom: 1px solid var(--line); }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }}
th {{ color: var(--muted); font-weight: 600; background: #fafbfc; white-space: nowrap; }}
tr:last-child td {{ border-bottom: 0; }}
.thumb {{
  width: 54px; height: 54px; object-fit: cover; border-radius: 8px;
  display: block; background: #eef2f6;
}}
.thumb.missing {{
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 10px; text-align: center;
}}
.badge {{
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.02em;
}}
.badge.entry {{ background: #dcfce7; color: #166534; }}
.badge.exit {{ background: #ffedd5; color: #9a3412; }}
.badge.open {{ background: #fef9c3; color: #854d0e; }}
.badge.closed {{ background: #e0e7ff; color: #3730a3; }}
.source {{ max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }}
.empty {{ color: var(--muted); text-align: center; padding: 24px; }}
.meta {{ color: var(--muted); font-size: 12px; margin: 0 0 18px; }}
footer {{ max-width: 1240px; margin: 0 auto; padding: 0 24px 36px; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<header>
  <div class="inner">
    <h1>Intelligent Face Tracker</h1>
    <p>Read-only snapshot from SQLite · generated {html.escape(data.generated_at)}</p>
  </div>
</header>
<main>
  <div class="cards">
    {_stat_cards(data.summary)}
  </div>
  <p class="meta">First event: {html.escape(first)} · Last event: {html.escape(last)}</p>
  <div class="grid">
    <section>
      <h2>Recent events</h2>
      <table>
        <thead><tr><th>Crop</th><th>Type</th><th>Time</th><th>Identity</th><th>Visit</th><th>Track</th><th>Conf.</th><th>Source</th></tr></thead>
        <tbody>
        {_event_rows(data.events)}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Recent visits</h2>
      <table>
        <thead><tr><th>Visit</th><th>Identity</th><th>Status</th><th>Entry</th><th>Exit</th><th>Track</th><th>Source</th></tr></thead>
        <tbody>
        {_visit_rows(data.visits)}
        </tbody>
      </table>
    </section>
  </div>
</main>
<footer>
  Unique visitors are derived from registered faces in SQLite. Anonymous visits
  are logged but excluded from the unique count.
</footer>
</body>
</html>
"""


class DashboardBuilder:
    """Read the DB and write ``index.html`` plus copied thumbnails."""

    def __init__(
        self,
        database: Database,
        output_dir: str | Path,
        *,
        max_events: int = 24,
        max_visits: int = 10,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events: must be at least 1")
        if max_visits < 1:
            raise ValueError("max_visits: must be at least 1")
        self._database = database
        self._output = Path(output_dir)
        self._max_events = max_events
        self._max_visits = max_visits

    @property
    def output_dir(self) -> Path:
        """The folder where ``index.html`` is written."""

        return self._output

    def build(self) -> DashboardData:
        """Read the DB, copy images and write the dashboard."""

        self._output.mkdir(parents=True, exist_ok=True)
        images_dir = self._output / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for stale in images_dir.glob("event_*"):
            if stale.is_file():
                stale.unlink()

        connection = self._database.connect()
        summary = UniqueCounter(self._database).get_summary()
        events = tuple(self._read_events(connection, images_dir))
        visits = tuple(self._read_visits(connection))
        data = DashboardData(
            summary=summary,
            events=events,
            visits=visits,
            generated_at=_utc_now(),
        )

        html_path = self._output / "index.html"
        temporary = html_path.with_suffix(".html.tmp")
        temporary.write_text(render_dashboard(data), encoding="utf-8")
        temporary.replace(html_path)
        return data

    def _read_events(self, connection, images_dir: Path) -> list[DashboardEvent]:
        rows = connection.execute(
            """
            SELECT e.event_id, e.event_type, e.occurred_at, e.face_id,
                   e.visit_id, e.source_id, e.image_path, e.confidence,
                   e.is_anonymous, v.track_id
            FROM events e
            JOIN visits v ON v.visit_id = e.visit_id
            ORDER BY e.event_id DESC
            LIMIT ?
            """,
            (self._max_events,),
        ).fetchall()

        events: list[DashboardEvent] = []
        for row in rows:
            source = Path(str(row["image_path"]))
            thumbnail = None
            if source.is_file():
                target = images_dir / f"event_{row['event_id']}_{source.name}"
                shutil.copy2(source, target)
                thumbnail = f"images/{target.name}"
            events.append(
                DashboardEvent(
                    event_id=int(row["event_id"]),
                    event_type=str(row["event_type"]),
                    occurred_at=str(row["occurred_at"]),
                    face_id=(
                        int(row["face_id"]) if row["face_id"] is not None else None
                    ),
                    visit_id=int(row["visit_id"]),
                    track_id=int(row["track_id"]),
                    source_id=str(row["source_id"]),
                    thumbnail=thumbnail,
                    is_anonymous=bool(row["is_anonymous"]),
                    confidence=(
                        float(row["confidence"])
                        if row["confidence"] is not None
                        else None
                    ),
                )
            )
        return events

    def _read_visits(self, connection) -> list[DashboardVisit]:
        rows = connection.execute(
            """
            SELECT visit_id, face_id, track_id, source_id, entry_at, exit_at,
                   status, is_anonymous
            FROM visits
            ORDER BY visit_id DESC
            LIMIT ?
            """,
            (self._max_visits,),
        ).fetchall()
        return [
            DashboardVisit(
                visit_id=int(row["visit_id"]),
                face_id=(int(row["face_id"]) if row["face_id"] is not None else None),
                track_id=int(row["track_id"]),
                source_id=str(row["source_id"]),
                entry_at=str(row["entry_at"]),
                exit_at=str(row["exit_at"]) if row["exit_at"] is not None else None,
                status=str(row["status"]),
                is_anonymous=bool(row["is_anonymous"]),
            )
            for row in rows
        ]
