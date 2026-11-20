"""Add covering indexes and query planner statistics for gallery/tag/artist hot paths."""

from __future__ import annotations

import sqlite3

INDEXES_V2_SQL = """
CREATE INDEX IF NOT EXISTS idx_images_folder_cov
    ON images(deleted, folder_id, mtime_ns);
CREATE INDEX IF NOT EXISTS idx_source_author_mtime
    ON image_source(author_key, mtime_ns);
CREATE INDEX IF NOT EXISTS idx_source_tags_tag_point
    ON source_tags(tag, point_id);
CREATE INDEX IF NOT EXISTS idx_tags_name_nocase
    ON tags(name COLLATE NOCASE, id);
"""


def upgrade(conn: sqlite3.Connection) -> None:
    conn.executescript(INDEXES_V2_SQL)
    conn.execute("ANALYZE")
