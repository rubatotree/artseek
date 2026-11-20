"""Persist a natural-sort key so name ordering no longer sorts rows in Python."""

from __future__ import annotations

import sqlite3

from ...core.sort_keys import natural_sort_text


def upgrade(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE images ADD COLUMN sort_name TEXT")
    rows = conn.execute("SELECT id, name FROM images").fetchall()
    conn.executemany(
        "UPDATE images SET sort_name = ? WHERE id = ?",
        [(natural_sort_text(str(row[1] or "")), int(row[0])) for row in rows],
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_sort_name ON images(sort_name, point_id)"
    )
