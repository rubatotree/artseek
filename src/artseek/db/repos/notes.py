"""Repository for indexing markdown note metadata in SQLite."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..connection import Database


class NotesIndexRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def sync_hash(
        self, content_hash: str, notes: Sequence[tuple[str, str, int]]
    ) -> None:
        """Replace indexed notes for one canonical content hash."""
        digest = str(content_hash or "").strip().lower()
        if not digest:
            return
        with self.db.write_tx() as conn:
            conn.execute("DELETE FROM notes_index WHERE content_hash = ?", (digest,))
            if notes:
                conn.executemany(
                    """
                    INSERT INTO notes_index(content_hash, name, title, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (digest, str(name), str(title), int(updated_at))
                        for name, title, updated_at in notes
                    ],
                )

    def delete_note(self, content_hash: str, name: str) -> None:
        with self.db.write_tx() as conn:
            conn.execute(
                "DELETE FROM notes_index WHERE content_hash = ? AND name = ?",
                (str(content_hash).lower(), str(name)),
            )

    def delete_hash(self, content_hash: str) -> None:
        with self.db.write_tx() as conn:
            conn.execute(
                "DELETE FROM notes_index WHERE content_hash = ?",
                (str(content_hash).lower(),),
            )

    def list_for_hash(self, content_hash: str) -> list[dict[str, Any]]:
        rows = self.db.read_conn().execute(
            """
            SELECT content_hash, name, title, updated_at
            FROM notes_index
            WHERE content_hash = ?
            ORDER BY name
            """,
            (str(content_hash).lower(),),
        ).fetchall()
        return [
            {
                "content_hash": str(r["content_hash"]),
                "name": str(r["name"]),
                "title": str(r["title"]),
                "updated_at": int(r["updated_at"]),
            }
            for r in rows
        ]

    def counts(self) -> tuple[int, int]:
        row = self.db.read_conn().execute(
            "SELECT COUNT(DISTINCT content_hash) AS folders, COUNT(*) AS notes FROM notes_index"
        ).fetchone()
        return (int(row["folders"]), int(row["notes"])) if row else (0, 0)
