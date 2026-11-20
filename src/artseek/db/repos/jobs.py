"""Repository for unified background job state persistence."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from ..connection import Database

_UNSET = object()


def _decode_json(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "kind": str(row["kind"]),
        "status": str(row["status"]),
        "payload": _decode_json(row["payload_json"], {}),
        "done": int(row["done"]),
        "total": int(row["total"]),
        "message": str(row["message"]),
        "error": str(row["error"]) if row["error"] is not None else None,
        "summary": _decode_json(row["summary_json"], None),
        "created_at": int(row["created_at"]),
        "updated_at": int(row["updated_at"]),
    }


class JobRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        job_id: str,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        status: str = "queued",
        message: str = "",
        total: int = 0,
    ) -> dict[str, Any]:
        now = int(time.time())
        encoded_payload = json.dumps(payload or {}, ensure_ascii=False)
        with self.db.write_tx() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    id, kind, status, payload_json, done, total, message,
                    error, summary_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?, NULL, NULL, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    done = 0,
                    total = excluded.total,
                    message = excluded.message,
                    error = NULL,
                    summary_json = NULL,
                    updated_at = excluded.updated_at
                """,
                (job_id, kind, status, encoded_payload, int(total), message, now, now),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert row is not None
            return _row_to_job(row)

    def update(
        self,
        job_id: str,
        *,
        status: str | Any = _UNSET,
        done: int | Any = _UNSET,
        total: int | Any = _UNSET,
        message: str | Any = _UNSET,
        error: str | None | Any = _UNSET,
        summary: dict[str, Any] | None | Any = _UNSET,
    ) -> dict[str, Any] | None:
        assignments: list[str] = ["updated_at = ?"]
        params: list[Any] = [int(time.time())]
        if status is not _UNSET:
            assignments.append("status = ?")
            params.append(str(status))
        if done is not _UNSET:
            assignments.append("done = ?")
            params.append(int(done))
        if total is not _UNSET:
            assignments.append("total = ?")
            params.append(int(total))
        if message is not _UNSET:
            assignments.append("message = ?")
            params.append(str(message))
        if error is not _UNSET:
            assignments.append("error = ?")
            params.append(str(error) if error is not None else None)
        if summary is not _UNSET:
            assignments.append("summary_json = ?")
            params.append(json.dumps(summary, ensure_ascii=False) if summary is not None else None)
        params.append(job_id)
        with self.db.write_tx() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?",
                tuple(params),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return _row_to_job(row) if row is not None else None

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.db.read_conn().execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return _row_to_job(row) if row is not None else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.read_conn().execute(
            "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def mark_interrupted_running(self) -> int:
        with self.db.write_tx() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'interrupted', updated_at = ?
                WHERE status IN ('queued', 'running', 'paused')
                """,
                (int(time.time()),),
            )
            return int(cur.rowcount or 0)

    def delete(self, job_id: str) -> bool:
        with self.db.write_tx() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return int(cur.rowcount or 0) > 0

    def clear_finished(self) -> int:
        with self.db.write_tx() as conn:
            cur = conn.execute(
                "DELETE FROM jobs WHERE status IN ('completed', 'failed', 'stopped', 'interrupted')"
            )
            return int(cur.rowcount or 0)
