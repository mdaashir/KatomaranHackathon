"""SQLite connection management.

The pipeline owns one writer connection. Reports open a separate read-only
connection (SQLite WAL supports concurrent readers while a writer is active).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class Database:
    """Owns a single SQLite connection configured for the tracker."""

    def __init__(
        self,
        db_path: str | Path,
        schema_path: str | Path | None = None,
    ) -> None:
        """
        Args:
            db_path: Database file, or ``":memory:"`` for tests.
            schema_path: Optional override for the schema file. Defaults to
                ``schema.sql`` next to this module.
        """

        self._path: str | Path = (
            db_path if str(db_path) == ":memory:" else Path(db_path)
        )
        self._schema_path = (
            Path(schema_path)
            if schema_path is not None
            else Path(__file__).with_name("schema.sql")
        )
        self._connection: sqlite3.Connection | None = None

    @property
    def path(self) -> str | Path:
        """The configured database path."""

        return self._path

    def connect(self) -> sqlite3.Connection:
        """Open the connection on first use and return it."""

        if self._connection is None:
            if isinstance(self._path, Path):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                target = str(self._path)
            else:
                target = self._path

            connection = sqlite3.connect(
                target,
                timeout=5.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            self._connection = connection

        return self._connection

    def initialize(self) -> None:
        """Create the schema if it does not exist."""

        schema = self._schema_path.read_text(encoding="utf-8")
        self.connect().executescript(schema)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one ``BEGIN IMMEDIATE`` transaction.

        The transaction is rolled back if the block raises. Nested
        transactions are rejected rather than silently flattened.
        """

        connection = self.connect()
        if connection.in_transaction:
            raise RuntimeError("nested transactions are not supported")

        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    def integrity_check(self) -> str:
        """Return SQLite's integrity check result (``"ok"`` when healthy)."""

        row = self.connect().execute("PRAGMA integrity_check").fetchone()
        return str(row[0])

    def close(self) -> None:
        """Close the connection if it is open."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Database:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
