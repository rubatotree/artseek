"""Schema v4: structured vision attributes, tag doc counts, and per-image task state."""

from __future__ import annotations

import sqlite3
import time

_IMAGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("media_type", "TEXT"),
    ("rating", "TEXT"),
    ("is_monochrome", "INTEGER"),
    ("monochrome_score", "REAL"),
    ("is_ai", "INTEGER"),
    ("ai_source", "TEXT"),
    ("ai_score", "REAL"),
    ("aesthetic_score", "REAL"),
    ("shot_type", "TEXT"),
    ("focal_x", "REAL"),
    ("focal_y", "REAL"),
    ("character_crop_json", "TEXT"),
    ("vision_meta_version", "INTEGER NOT NULL DEFAULT 0"),
    ("vision_meta_at", "INTEGER"),
)


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def upgrade(conn: sqlite3.Connection) -> None:
    existing_img = _existing_columns(conn, "images")
    for name, ddl in _IMAGE_COLUMNS:
        if name not in existing_img:
            conn.execute(f"ALTER TABLE images ADD COLUMN {name} {ddl}")

    existing_tags = _existing_columns(conn, "tags")
    if "doc_count" not in existing_tags:
        conn.execute("ALTER TABLE tags ADD COLUMN doc_count INTEGER NOT NULL DEFAULT 0")

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_images_media_rating
            ON images(media_type, rating) WHERE deleted = 0;
        CREATE INDEX IF NOT EXISTS idx_images_mono_ai
            ON images(is_monochrome, is_ai) WHERE deleted = 0;
        CREATE INDEX IF NOT EXISTS idx_images_aesthetic
            ON images(aesthetic_score DESC, id) WHERE deleted = 0;

        CREATE TABLE IF NOT EXISTS image_task_state (
            point_id TEXT PRIMARY KEY,
            image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
            content_hash TEXT NOT NULL DEFAULT '',
            index_done INTEGER NOT NULL DEFAULT 0,
            url_done INTEGER NOT NULL DEFAULT 0,
            author_done INTEGER NOT NULL DEFAULT 0,
            tag_models_json TEXT NOT NULL DEFAULT '{}',
            descriptor_done INTEGER NOT NULL DEFAULT 0,
            vision_done INTEGER NOT NULL DEFAULT 0,
            ccip_done INTEGER NOT NULL DEFAULT 0,
            caption_done INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_task_state_image
            ON image_task_state(image_id);
        """
    )

    conn.execute(
        """
        UPDATE tags
        SET doc_count = (
            SELECT COUNT(*)
            FROM image_tags AS it
            JOIN images AS i ON i.id = it.image_id AND i.deleted = 0
            WHERE it.tag_id = tags.id
        )
        """
    )

    now = int(time.time())
    conn.execute(
        """
        INSERT OR IGNORE INTO image_task_state(
            point_id, image_id, content_hash, index_done, url_done, author_done,
            tag_models_json, descriptor_done, vision_done, ccip_done, caption_done, updated_at
        )
        SELECT
            i.point_id,
            i.id,
            COALESCE(i.content_hash, ''),
            1,
            CASE
                WHEN s.point_id IS NOT NULL AND (
                    s.status = 'ok'
                    OR s.source_origin = 'manual'
                    OR COALESCE(s.manual_fields, '') LIKE '%url_ignored%'
                ) THEN 1
                ELSE 0
            END,
            CASE
                WHEN s.point_id IS NOT NULL
                     AND s.author_key IS NOT NULL
                     AND s.author_key <> '' THEN 1
                ELSE 0
            END,
            '{}',
            CASE
                WHEN i.descriptor_text IS NOT NULL AND i.descriptor_text <> '' THEN 1
                ELSE 0
            END,
            CASE WHEN COALESCE(i.vision_meta_version, 0) > 0 THEN 1 ELSE 0 END,
            0,
            CASE
                WHEN i.caption_json IS NOT NULL
                     AND i.caption_json <> ''
                     AND i.caption_json <> 'null' THEN 1
                ELSE 0
            END,
            ?
        FROM images AS i
        LEFT JOIN image_source AS s ON s.point_id = i.point_id
        WHERE i.deleted = 0
        """,
        (now,),
    )
