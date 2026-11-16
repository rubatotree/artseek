from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .parser import ParsedSource, extract_embedded_refs, parse_filename
from .store import ProvenanceStore

ProgressCallback = Callable[[int, int, str], None]


def parse_payload_path(path: Path, *, deep: bool) -> ParsedSource | None:
    """Embedded URLs win, but avoid opening every already-identified file."""
    filename = parse_filename(path)
    embedded = extract_embedded_refs(path) if deep or filename is None else []
    if embedded:
        primary = embedded[0]
        candidates = tuple(
            ref
            for ref in (*embedded[1:], filename)
            if ref is not None
            and (ref.platform, ref.source_id, ref.page)
            != (primary.platform, primary.source_id, primary.page)
        )
        return ParsedSource(primary=primary, candidates=candidates)
    if filename is not None:
        return ParsedSource(primary=filename)
    return None


def scan_payloads(
    store: ProvenanceStore,
    payloads: Iterable[tuple[str, dict[str, Any]]],
    *,
    total: int | None = None,
    deep: bool = False,
    only_new: bool = False,
    progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Read local filenames/metadata and persist source references.

    No network request is made. `deep=False` opens a file only when its name is
    not already recognizable, which keeps routine scans bounded.
    """
    known = store.known_points() if only_new else {}
    summary = {
        "total": int(total or 0),
        "scanned": 0,
        "identified": 0,
        "unresolved": 0,
        "new": 0,
        "skipped": 0,
        "stale": 0,
    }
    live_hashes: dict[str, str | None] = {}
    for index, (point_id, payload) in enumerate(payloads, start=1):
        if stop_event is not None and stop_event.is_set():
            break
        while pause_event is not None and pause_event.is_set():
            if stop_event is not None and stop_event.is_set():
                break
            time.sleep(0.25)
        if stop_event is not None and stop_event.is_set():
            break
        summary["total"] = max(int(summary["total"]), index)
        digest = str(payload.get("content_hash") or "") or None
        if only_new and point_id in known:
            known_hash, _updated, known_mtime, known_path = known[point_id]
            current_path = _path_key(payload)
            if (
                digest is None
                or (
                    known_hash == digest
                    and known_mtime is not None
                    and (known_path or "") == (current_path or "")
                )
            ):
                summary["skipped"] += 1
                continue
        summary["scanned"] += 1
        if digest:
            live_hashes[point_id] = digest
        path_value = str(payload.get("path") or "")
        path = Path(path_value) if path_value else None
        parsed: ParsedSource | None = None
        if path is not None and path.exists():
            parsed = parse_payload_path(path, deep=deep)
        before = store.get(point_id)
        if (
            before is not None
            and before.get("content_hash")
            and digest
            and before["content_hash"] != digest
        ):
            store.reset_for_replacement(point_id)
            before = store.get(point_id)
        if parsed is None:
            store.upsert_unresolved(point_id, payload, path_key=_path_key(payload))
            summary["unresolved"] += 1
        else:
            store.upsert_scan(
                point_id,
                payload,
                parsed,
                path_key=_path_key(payload),
                origin=parsed.primary.origin,
            )
            if before is None:
                summary["new"] += 1
            summary["identified"] += 1
        if progress is not None and (index == 1 or index % 50 == 0):
            progress(index, int(summary["total"]), path_value)
    if live_hashes:
        summary["stale"] = len(store.stale_points(live_hashes))
    return summary


def _path_key(payload: dict[str, Any]) -> str | None:
    value = payload.get("path_key")
    if value:
        return str(value)
    path = payload.get("path")
    return str(path) if path else None
