"""Schema migration runner driven by SQLite PRAGMA user_version."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from . import v001_initial

MIGRATIONS: tuple[tuple[int, Callable[[sqlite3.Connection], None]], ...] = (
    (1, v001_initial.upgrade),
)
LATEST_VERSION = MIGRATIONS[-1][0]


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations inside the caller's transaction."""
    row = conn.execute("PRAGMA user_version").fetchone()
    current = int(row[0]) if row else 0
    for version, fn in MIGRATIONS:
        if current < version:
            fn(conn)
            conn.execute(f"PRAGMA user_version = {version}")
            current = version
    return current
