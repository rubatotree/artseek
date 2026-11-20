"""Provenance scanning, remote enrichment, manual editing, and visual-neighbor auto-tagging."""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..core.ranking import GENERIC_VISUAL_TAGS, rank_tags_by_idf
from ..core.tag_dictionary import tag_info
from ..db.repos.images import ImageRepo
from ..db.repos.provenance import ProvenanceRepo
from ..provenance.fetch import (
    PIXIV_DOMAIN,
    TWITTER_DOMAIN,
    FxTwitterClient,
    PixivClient,
    ProvenanceService,
)
from ..provenance.parser import parse_source_text
from ..provenance.scan import parse_payload_path, record_unresolved, scan_payloads
from ..provenance.scheduler import BudgetExhausted, FetchError, RateLimiter
from ..provenance.store import (
    normalize_pixiv_user_id,
    normalize_twitter_handle,
)
from ..vectors.index import VectorIndex
from .common import ServiceError, resolve_author_reference


class ProvenanceAppService:
    """Orchestrates v2 provenance records, budgeted remote enrichment, and vector auto-tagging."""

    def __init__(
        self,
        config: AppConfig,
        images: ImageRepo,
        prov: ProvenanceRepo,
        vectors: VectorIndex,
        *,
        on_author_change: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.images = images
        self.prov = prov
        self.vectors = vectors
        self._on_author_change = on_author_change

    def _notify_author_change(self) -> None:
        if callable(self._on_author_change):
            self._on_author_change()

    def _limiter(self, domain: str) -> RateLimiter:
        cfg = self.config.provenance
        if domain == PIXIV_DOMAIN:
            return RateLimiter(
                self.prov,
                PIXIV_DOMAIN,
                min_interval=max(0.0, cfg.pixiv_min_interval),
                daily_limit=cfg.pixiv_daily_limit,
                concurrency=max(1, cfg.pixiv_concurrency),
            )
        return RateLimiter(
            self.prov,
            TWITTER_DOMAIN,
            min_interval=max(0.0, cfg.twitter_min_interval),
            daily_limit=cfg.twitter_daily_limit,
            concurrency=max(1, cfg.twitter_concurrency),
        )

    def budgets(self) -> list[dict[str, Any]]:
        return [
            self._limiter(PIXIV_DOMAIN).snapshot().as_dict(),
            self._limiter(TWITTER_DOMAIN).snapshot().as_dict(),
        ]

    def stats(self) -> dict[str, Any]:
        return self.prov.stats()

    def make_fetch_service(
        self,
        *,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> ProvenanceService:
        return ProvenanceService(
            self.prov,
            pixiv=PixivClient(),
            twitter=FxTwitterClient(),
            pixiv_limiter=self._limiter(PIXIV_DOMAIN),
            twitter_limiter=self._limiter(TWITTER_DOMAIN),
            stop_event=stop_event,
            pause_event=pause_event,
        )

    def _resolve_item(self, point_id: str) -> tuple[dict[str, Any], Path]:
        payload = self.images.get_by_point_id(point_id)
        if not payload:
            raise ServiceError(404, "图片不在索引中")
        path = Path(str(payload["path"]))
        if not path.exists():
            raise ServiceError(404, "原图片已移动或删除，请重新索引")
        return payload, path

    def ensure_record(
        self, point_id: str, payload: dict[str, Any], path: Path
    ) -> dict[str, Any] | None:
        record = self.prov.get(point_id)
        if record is not None and record.get("status") != "unresolved":
            return record
        parsed = parse_payload_path(path, deep=True)
        if parsed is not None:
            self.prov.upsert_scan(
                point_id,
                payload,
                parsed,
                path_key=str(payload.get("path_key") or ""),
                origin=parsed.primary.origin,
            )
        elif record is None:
            record_unresolved(
                self.prov,
                point_id,
                payload,
                path_key=str(payload.get("path_key") or ""),
                path=path,
            )
        return self.prov.get(point_id)

    def _format_provenance_payload(
        self, point_id: str, payload: dict[str, Any], path: Path
    ) -> dict[str, Any]:
        record = self.ensure_record(point_id, payload, path)
        if record is None:
            return {"point_id": point_id, "record": None, "author": None}
        author = None
        if record.get("author_key"):
            author = self.prov.author_summary(str(record["author_key"]))
        page = record.get("page")
        count = record.get("page_count")
        if record.get("platform") == "pixiv" and page is not None:
            page_label = f"p{page}" + (f" / {count}" if count else "")
        elif record.get("platform") == "twitter" and page is not None:
            page_label = f"{page} / {count}" if count else str(page)
        else:
            page_label = ""
        clean = dict(record)
        clean.pop("raw_json", None)
        tag_aliases: dict[str, list[str]] = {}
        for tag in {
            *record.get("tags", []),
            *record.get("manual_tags", []),
            *record.get("auto_tags", []),
        }:
            info = tag_info(str(tag))
            if info is None:
                continue
            values = [info.zh, *info.aliases, info.tag, str(tag)]
            tag_aliases[str(tag)] = list(
                dict.fromkeys(v for v in values if str(v).strip())
            )
        clean["tag_aliases"] = tag_aliases
        return {
            "point_id": point_id,
            "record": clean,
            "author": author,
            "page_label": page_label,
            "path": str(path),
            "reverse_search": {
                "saucenao": "https://saucenao.com/",
                "ascii2d": "https://ascii2d.net/",
            },
        }

    def get_image_provenance(self, point_id: str) -> dict[str, Any]:
        payload, path = self._resolve_item(point_id)
        return self._format_provenance_payload(point_id, payload, path)

    def update_manual(
        self,
        point_id: str,
        *,
        clear_manual: bool = False,
        url: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: Sequence[str] | None = None,
        manual_tags: Sequence[str] | None = None,
        author_key: str | None = None,
        author_query: str | None = None,
        author_name: str | None = None,
        author_handle: str | None = None,
        pixiv_id: str | None = None,
        x_handle: str | None = None,
    ) -> dict[str, Any]:
        payload, path = self._resolve_item(point_id)
        record = self.ensure_record(point_id, payload, path)
        if record is None:
            raise ServiceError(404, "图片不在索引中")

        if clear_manual:
            self.prov.clear_manual(point_id)
            parsed = parse_payload_path(path, deep=True)
            if parsed is not None:
                self.prov.upsert_scan(
                    point_id,
                    payload,
                    parsed,
                    path_key=str(payload.get("path_key") or ""),
                    origin=parsed.primary.origin,
                )
            self._notify_author_change()
            return self._format_provenance_payload(point_id, payload, path)

        fields: dict[str, Any] = {}
        ref = None
        if url:
            ref = parse_source_text(url)
            if ref is None:
                raise ServiceError(400, "请输入 Pixiv/X/小红书作品 URL 或任意 http(s) 来源 URL")
        if title is not None:
            fields["title"] = title.strip() or None
        if description is not None:
            fields["description"] = description

        if tags is not None:
            fields["tags"] = [t.strip() for t in tags if t.strip()]
        if manual_tags is not None:
            fields["manual_tags"] = [t.strip() for t in manual_tags if t.strip()]

        clean_author_key = (author_key or "").strip()
        clean_author_query = (author_query or "").strip()
        if clean_author_key or clean_author_query:
            fields["author_key"] = resolve_author_reference(
                self.prov, clean_author_key or clean_author_query
            )
        elif (
            author_name is not None
            or author_handle is not None
            or pixiv_id is not None
            or x_handle is not None
        ):
            name = (author_name or "").strip()
            raw_pixiv = (pixiv_id or "").strip()
            raw_x = (x_handle or author_handle or "").strip()
            norm_pixiv = normalize_pixiv_user_id(raw_pixiv) if raw_pixiv else ""
            norm_x = normalize_twitter_handle(raw_x) if raw_x else ""
            if raw_pixiv and not norm_pixiv:
                raise ServiceError(400, "Pixiv ID 应为数字 ID 或作者主页 URL")
            if raw_x and not re.fullmatch(r"[A-Za-z0-9_]{1,15}", norm_x):
                raise ServiceError(400, "X handle 格式无效")
            key = ""
            if norm_pixiv or norm_x:
                base_key = str(record.get("author_key") or "")
                single_platform = ""
                single_value = ""
                if bool(norm_pixiv) != bool(norm_x):
                    single_platform = "pixiv" if norm_pixiv else "twitter"
                    single_value = norm_pixiv or norm_x
                owner = (
                    self.prov.identity_owner(single_platform, single_value)
                    if single_platform
                    else None
                )
                if owner is not None:
                    key = owner
                elif base_key.isdigit() and single_platform:
                    key = self.prov.attach_identity(
                        base_key,
                        single_platform,
                        single_value,
                        name=name or None,
                    )
                else:
                    key = self.prov.ensure_linked_author(
                        name=name or None,
                        pixiv_id=norm_pixiv or None,
                        x_handle=norm_x or None,
                    )
            elif name:
                key = self.prov.ensure_manual_author(name)
            if key:
                fields["author_key"] = key

        if ref is not None:
            self.prov.set_manual(point_id, fields, ref=ref)
            latest = self.prov.get(point_id)
            if latest is not None and latest.get("platform") in {"pixiv", "twitter"}:
                domain = PIXIV_DOMAIN if latest["platform"] == "pixiv" else TWITTER_DOMAIN
                state = self.prov.domain_state(domain)
                if int(state.get("blocked_until") or 0) <= int(time.time()):
                    service = self.make_fetch_service()
                    try:
                        service.enrich_record(latest)
                    except (FetchError, BudgetExhausted) as exc:
                        self.prov.set_status(
                            point_id, getattr(exc, "status", "error"), str(exc)
                        )
                    finally:
                        service.close()
        elif fields:
            self.prov.set_manual(point_id, fields)
        self._notify_author_change()
        return self._format_provenance_payload(point_id, payload, path)

    def resolve_url(self, point_id: str, url: str) -> dict[str, Any]:
        payload, path = self._resolve_item(point_id)
        ref = parse_source_text(url)
        if ref is None:
            raise ServiceError(400, "无法识别该 URL")
        self.ensure_record(point_id, payload, path)
        self.prov.set_manual(point_id, {}, ref=ref)
        record = self.prov.get(point_id)
        if record is not None and ref.platform in {"pixiv", "twitter"}:
            domain = PIXIV_DOMAIN if ref.platform == "pixiv" else TWITTER_DOMAIN
            state = self.prov.domain_state(domain)
            if int(state.get("blocked_until") or 0) <= int(time.time()):
                service = self.make_fetch_service()
                try:
                    service.enrich_record(record)
                except (FetchError, BudgetExhausted) as exc:
                    self.prov.set_status(
                        point_id, getattr(exc, "status", "error"), str(exc)
                    )
                finally:
                    service.close()
        self._notify_author_change()
        return self._format_provenance_payload(point_id, payload, path)

    def set_url_ignored(self, point_id: str, *, ignored: bool = True) -> dict[str, Any]:
        payload, path = self._resolve_item(point_id)
        self.ensure_record(point_id, payload, path)
        self.prov.set_url_ignored(point_id, ignored=ignored)
        self._notify_author_change()
        return self._format_provenance_payload(point_id, payload, path)

    # ---------------------------------------------------------------- batch jobs
    def run_scan(
        self,
        *,
        deep: bool = False,
        only_new: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        try:
            total = self.images.count()
            summary = scan_payloads(
                self.prov,
                self.images.iter_scan_payloads(),
                total=total,
                deep=deep,
                only_new=only_new,
                progress=progress,
                stop_event=stop_event,
                pause_event=pause_event,
            )
            return summary
        finally:
            self._notify_author_change()

    def select_enrich_records(
        self,
        *,
        platforms: Sequence[str],
        limit: int | None = None,
        point_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        clean = [p for p in platforms if p in {"pixiv", "twitter"}]
        if not clean:
            return []
        if point_ids:
            records = [
                rec
                for pid in dict.fromkeys(point_ids)
                if (rec := self.prov.get(pid)) is not None
                and rec.get("platform") in clean
            ]
        else:
            records = self.prov.pending_records(platforms=tuple(clean), limit=limit)
        if limit is not None:
            records = records[: max(0, int(limit))]
        return records

    def run_enrich(
        self,
        records: Sequence[dict[str, Any]],
        *,
        progress: Callable[[int, int, str], None] | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if not records:
            return {"total": 0, "ok": 0, "failed": 0, "stopped": False}
        service = self.make_fetch_service(stop_event=stop_event, pause_event=pause_event)
        try:
            return service.enrich_records(list(records), progress=progress)
        finally:
            service.close()
            self._notify_author_change()

    def run_auto_tag(
        self,
        *,
        limit: int | None = 500,
        progress: Callable[[int, int, str], None] | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        batch_size: int = 128,
        neighbor_limit: int = 24,
        top_tags: int = 12,
    ) -> dict[str, Any]:
        point_ids = self.prov.point_ids_without_auto_tags(limit)
        total = len(point_ids)
        summary = {"ok": 0, "skipped": 0, "failed": 0, "total": total}
        if not point_ids:
            return summary

        row_map = self.images.row_ids_for_points(point_ids)
        total_docs, doc_counts = self.images.tag_doc_counts()
        idf_map: dict[str, float] | None = None
        if total_docs > 0 and doc_counts:
            idf_map = {
                tag: (
                    (math.log((total_docs + 1.0) / (1.0 + cnt)) + 0.5)
                    * (0.25 if tag in GENERIC_VISUAL_TAGS else 1.0)
                )
                for tag, cnt in doc_counts.items()
            }
        row_tags_cache: dict[int, list[str]] | None = None
        done = 0

        for start in range(0, len(point_ids), batch_size):
            if stop_event is not None and stop_event.is_set():
                break
            while pause_event is not None and pause_event.is_set():
                if stop_event is not None and stop_event.is_set():
                    break
                time.sleep(0.25)
            if stop_event is not None and stop_event.is_set():
                break

            chunk_pids = point_ids[start : start + batch_size]
            target_pairs = [
                (pid, row_map[pid])
                for pid in chunk_pids
                if pid in row_map
            ]
            target_rows = [rid for _, rid in target_pairs]
            own_tags_map = self.images.row_tag_scores_map(target_rows)
            missing_rows = [rid for rid in target_rows if not own_tags_map.get(rid)]
            neighbor_votes: dict[int, tuple[list[str], dict[str, float]]] = {}
            if missing_rows:
                if row_tags_cache is None:
                    row_tags_cache = self.images.row_tags_map()
                neighbor_votes = self.vectors.neighbor_tag_votes(
                    missing_rows,
                    row_tags_cache,
                    limit=neighbor_limit,
                    top_tags=top_tags,
                    tag_idf=idf_map,
                )

            updates: list[tuple[str, list[str], dict[str, float]]] = []
            for pid, rid in target_pairs:
                own_scores = own_tags_map.get(rid)
                if own_scores:
                    tags, scores = rank_tags_by_idf(
                        own_scores,
                        total_docs=total_docs,
                        doc_counts=doc_counts,
                        limit=top_tags,
                    )
                else:
                    ranked = neighbor_votes.get(rid)
                    if not ranked:
                        summary["skipped"] += 1
                        continue
                    tags, scores = ranked
                updates.append((pid, tags, scores))
                if tags:
                    summary["ok"] += 1
                else:
                    summary["skipped"] += 1
            summary["skipped"] += len(chunk_pids) - len(target_pairs)
            if updates:
                self.prov.set_auto_tags_many(updates)
            done = min(start + len(chunk_pids), total)
            if progress is not None:
                progress(done, total, "")

        return summary
