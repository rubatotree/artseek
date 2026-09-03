from __future__ import annotations

import json
import re
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_ID_LENGTH = 12
SNAPSHOT_ID_RE = re.compile(rf"^[0-9a-f]{{{SNAPSHOT_ID_LENGTH}}}$")
MAX_SNAPSHOTS = 96
MAX_SNAPSHOT_RESULTS = 400
RESULT_FIELDS = (
    "id",
    "score",
    "path",
    "name",
    "width",
    "height",
    "tags",
    "tag_hits",
    "caption",
    "visual_score",
    "descriptor_score",
)


def new_snapshot_id() -> str:
    return secrets.token_hex(SNAPSHOT_ID_LENGTH // 2)


def is_snapshot_id(value: str | None) -> bool:
    return bool(value) and SNAPSHOT_ID_RE.fullmatch(str(value)) is not None


def sanitize_reference(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    point_id = str(item.get("id") or "").strip()
    if not point_id:
        return None
    try:
        weight = float(item.get("weight", 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    return {"id": point_id, "weight": max(0.0, min(weight, 8.0))}


def sanitize_result(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    point_id = str(item.get("id") or "").strip()
    if not point_id:
        return None
    row = {key: item.get(key) for key in RESULT_FIELDS}
    row["id"] = point_id
    row["tags"] = [str(tag) for tag in (item.get("tags") or []) if str(tag)]
    row["tag_hits"] = [str(tag) for tag in (item.get("tag_hits") or []) if str(tag)]
    return row


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class SearchSnapshotStore:
    """File-backed cache of one search run, keyed by a short hex id."""

    def __init__(self, directory: Path):
        self.root = Path(directory)
        self._lock = threading.RLock()

    def set_root(self, directory: Path) -> None:
        with self._lock:
            self.root = Path(directory)
            self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, snapshot_id: str) -> Path:
        if not is_snapshot_id(snapshot_id):
            raise ValueError("检索快照编号不合法")
        return self.root / f"{snapshot_id}.json"

    def create(
        self,
        *,
        query: str,
        text_weight: float,
        references: list[dict[str, Any]],
        expand: bool,
        strict_tags: bool,
        dedupe: bool,
        limit: int,
        folder_keys: list[str] | None,
        mtime_after: str | None,
        mtime_before: str | None,
        query_plan: dict[str, Any] | None,
        llm_warning: str | None,
        results: list[dict[str, Any]],
        has_more: bool,
        had_file_refs: bool = False,
    ) -> str:
        refs = [item for item in (sanitize_reference(row) for row in references) if item]
        rows = [item for item in (sanitize_result(row) for row in results) if item]
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            snapshot_id = new_snapshot_id()
            while self.path_for(snapshot_id).exists():
                snapshot_id = new_snapshot_id()
            stamp = _now()
            payload = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "id": snapshot_id,
                "created_at": stamp,
                "updated_at": stamp,
                "query": str(query or ""),
                "text_weight": max(0.0, min(float(text_weight), 8.0)),
                "references": refs,
                "expand": bool(expand),
                "strict_tags": bool(strict_tags),
                "dedupe": bool(dedupe),
                "limit": int(limit),
                "folder_keys": list(folder_keys) if folder_keys is not None else None,
                "mtime_after": mtime_after or None,
                "mtime_before": mtime_before or None,
                "query_plan": query_plan if isinstance(query_plan, dict) else {},
                "llm_warning": llm_warning,
                "results": rows[:MAX_SNAPSHOT_RESULTS],
                "has_more": bool(has_more),
                "had_file_refs": bool(had_file_refs),
            }
            self._write(snapshot_id, payload)
            self._evict_locked(keep=snapshot_id)
            return snapshot_id

    def append(
        self,
        snapshot_id: str,
        results: list[dict[str, Any]],
        *,
        has_more: bool,
    ) -> bool:
        rows = [item for item in (sanitize_result(row) for row in results) if item]
        with self._lock:
            payload = self._read_locked(snapshot_id)
            if payload is None:
                return False
            seen = {str(item.get("id") or "") for item in payload.get("results") or []}
            merged = list(payload.get("results") or [])
            for row in rows:
                if row["id"] in seen:
                    continue
                if len(merged) >= MAX_SNAPSHOT_RESULTS:
                    break
                merged.append(row)
                seen.add(row["id"])
            payload["results"] = merged
            payload["has_more"] = bool(has_more) and len(merged) < MAX_SNAPSHOT_RESULTS
            payload["updated_at"] = _now()
            self._write(snapshot_id, payload)
            return True

    def get(self, snapshot_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read_locked(snapshot_id)

    def _read_locked(self, snapshot_id: str) -> dict[str, Any] | None:
        if not is_snapshot_id(snapshot_id):
            return None
        path = self.path_for(snapshot_id)
        if not path.is_file():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(loaded, dict) or loaded.get("id") != snapshot_id:
            return None
        loaded["results"] = [
            item
            for item in (sanitize_result(row) for row in (loaded.get("results") or []))
            if item
        ]
        loaded["references"] = [
            item
            for item in (sanitize_reference(row) for row in (loaded.get("references") or []))
            if item
        ]
        return loaded

    def _write(self, snapshot_id: str, payload: dict[str, Any]) -> None:
        path = self.path_for(snapshot_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _evict_locked(self, *, keep: str) -> None:
        files = [path for path in self.root.glob("*.json") if path.is_file()]
        overflow = len(files) - MAX_SNAPSHOTS
        if overflow <= 0:
            return
        files.sort(key=lambda path: path.stat().st_mtime_ns)
        removed = 0
        for path in files:
            if path.stem == keep:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
            if removed >= overflow:
                return
