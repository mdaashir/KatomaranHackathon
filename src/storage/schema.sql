PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS faces (
    face_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    first_seen_at TEXT    NOT NULL,
    model_name    TEXT    NOT NULL,
    best_quality  REAL    NOT NULL DEFAULT 0.0,
    is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id         INTEGER NOT NULL REFERENCES faces(face_id) ON DELETE CASCADE,
    vector          BLOB    NOT NULL,
    dim             INTEGER NOT NULL,
    quality         REAL    NOT NULL,
    created_at      TEXT    NOT NULL,
    source_event_id INTEGER
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id          INTEGER REFERENCES faces(face_id),
    track_id         INTEGER NOT NULL,
    source_id        TEXT    NOT NULL,
    entry_at         TEXT    NOT NULL,
    exit_at          TEXT,
    status           TEXT    NOT NULL CHECK (status IN ('open', 'closed')),
    is_anonymous     INTEGER NOT NULL DEFAULT 0 CHECK (is_anonymous IN (0, 1)),
    entry_image_path TEXT,
    exit_image_path  TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id     INTEGER NOT NULL REFERENCES visits(visit_id),
    face_id      INTEGER REFERENCES faces(face_id),
    event_type   TEXT    NOT NULL CHECK (event_type IN ('entry', 'exit')),
    occurred_at  TEXT    NOT NULL,
    video_time_s REAL    NOT NULL,
    source_id    TEXT    NOT NULL,
    image_path   TEXT    NOT NULL,
    bbox_json    TEXT,
    confidence   REAL,
    similarity   REAL,
    is_anonymous INTEGER NOT NULL DEFAULT 0 CHECK (is_anonymous IN (0, 1))
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_events_visit_type
    ON events (visit_id, event_type);
CREATE INDEX IF NOT EXISTS ix_events_time
    ON events (occurred_at);
CREATE INDEX IF NOT EXISTS ix_visits_face
    ON visits (face_id, status);
