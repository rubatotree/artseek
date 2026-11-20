"""Repository for persistent LLM QueryPlan cache."""

from __future__ import annotations

import json
import time
from typing import Any

from ..connection import Database


class QueryPlanCacheRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, query_norm: str, model: str) -> dict[str, Any] | None:
        row = self.db.read_conn().execute(
            "SELECT plan_json, model FROM query_plan_cache WHERE query_norm = ?",
            (query_norm,),
        ).fetchone()
        if row is None or str(row["model"]) != model:
            return None
        try:
            data = json.loads(str(row["plan_json"]))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def put(self, query_norm: str, model: str, plan: dict[str, Any]) -> None:
        now = int(time.time())
        encoded = json.dumps(plan, ensure_ascii=False)
        with self.db.write_tx() as conn:
            conn.execute(
                """
                INSERT INTO query_plan_cache(query_norm, plan_json, model, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_norm) DO UPDATE SET
                    plan_json = excluded.plan_json,
                    model = excluded.model,
                    created_at = excluded.created_at
                """,
                (query_norm, encoded, model, now),
            )

    def clear(self) -> None:
        with self.db.write_tx() as conn:
            conn.execute("DELETE FROM query_plan_cache")
