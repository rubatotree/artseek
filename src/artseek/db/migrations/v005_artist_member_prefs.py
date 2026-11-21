"""Schema v5: manual order and active flag for an artist's social accounts."""

from __future__ import annotations

import sqlite3


def upgrade(conn: sqlite3.Connection) -> None:
    existing = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(artist_identity)").fetchall()
    }
    if "sort_order" not in existing:
        conn.execute("ALTER TABLE artist_identity ADD COLUMN sort_order INTEGER")
    if "inactive" not in existing:
        # Inactive accounts stay searchable but are hidden next to artist names.
        conn.execute(
            "ALTER TABLE artist_identity ADD COLUMN inactive INTEGER NOT NULL DEFAULT 0"
        )
