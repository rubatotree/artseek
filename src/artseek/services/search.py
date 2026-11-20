"""Hybrid vector+SQL search, non-blocking LLM query expansion, and snapshots."""

from __future__ import annotations

import dataclasses
import re
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AppConfig
from ..core.query_plan import QueryPlan, normalize_query, static_expand
from ..core.ranking import compose_visual_vector, fuse_and_rank
from ..core.scope import ImageScope
from ..core.tag_dictionary import expand_tag_aliases
from ..core.tag_query import TagQueryError, parse_tag_expression
from ..db.repos.images import ImageRepo
from ..db.repos.provenance import ProvenanceRepo
from ..db.repos.query_cache import QueryPlanCacheRepo
from ..db.scope_resolver import ScopeResolver
from ..library import day_end_ns, day_start_ns
from ..ml.llm import OllamaUnavailable
from ..search_cache import SearchSnapshotStore, is_snapshot_id
from ..vectors.index import VectorIndex
from .common import (
    ServiceError,
    attach_provenance,
    fast_folder_key,
    gallery_result,
    resolve_author_token,
    resolve_folder_keys_from_db,
)
from .model_lifecycle import ModelLifecycle

MAX_REFERENCE_IMAGES = 8
_FILTER_TOKEN_RE = re.compile(
    r'(?:^|\s)(author|artist|tag):(?:"([^"]+)"|(\S+))',
    re.IGNORECASE,
)


def parse_query_filters(query: str) -> tuple[str, str | None, list[str]]:
    """Strip inline `author:`, `artist:`, `tag:` tokens from a search string."""
    author: str | None = None
    tags: list[str] = []

    def take(match: re.Match[str]) -> str:
        nonlocal author
        key = match.group(1).casefold()
        value = (match.group(2) or match.group(3) or "").strip()
        if key in {"author", "artist"} and value:
            author = value
        elif key == "tag" and value:
            tags.append(value)
        return " "

    cleaned = re.sub(r"\s+", " ", _FILTER_TOKEN_RE.sub(take, query or "")).strip()
    return cleaned, author, tags


def normalize_plan_key(text: str) -> str:
    return normalize_query(text)


class SearchService:
    """Executes non-blocking LLM expansion, hybrid VectorIndex+ScopeResolver search."""

    def __init__(
        self,
        config: AppConfig,
        images: ImageRepo,
        prov: ProvenanceRepo,
        vectors: VectorIndex,
        resolver: ScopeResolver,
        lifecycle: ModelLifecycle,
        query_cache: QueryPlanCacheRepo,
        snapshots: SearchSnapshotStore,
    ) -> None:
        self.config = config
        self.images = images
        self.prov = prov
        self.vectors = vectors
        self.resolver = resolver
        self.lifecycle = lifecycle
        self.query_cache = query_cache
        self.snapshots = snapshots
        self._plan_lock = threading.Lock()
        self._pending_plans: set[tuple[str, str]] = set()
        self._plan_errors: dict[tuple[str, str], tuple[float, str]] = {}

    # ---------------------------------------------------------------- LLM query plan
    def _resolve_plan(
        self,
        search_text: str,
        *,
        expand: bool,
        wait_for_llm: bool,
    ) -> tuple[QueryPlan, str | None, int, bool, str]:
        """Return `(plan, llm_warning, plan_revision, llm_pending, plan_key)`."""
        text = str(search_text or "").strip()
        if not text:
            return QueryPlan("", ""), None, 1, False, ""

        plan_key = normalize_plan_key(text)
        offline_plan = static_expand(text)
        if not expand or not self.config.llm.enabled:
            return offline_plan, None, 1, False, plan_key

        model = str(self.config.llm.query_model or "")
        cached = self.query_cache.get(plan_key, model)
        if cached is not None:
            return QueryPlan.from_dict(cached), None, 2, False, plan_key

        if wait_for_llm:
            try:
                llm_plan = self._invoke_llm(text)
                self.query_cache.put(plan_key, model, llm_plan.to_dict())
                with self._plan_lock:
                    self._plan_errors.pop((plan_key, model), None)
                return llm_plan, None, 2, False, plan_key
            except OllamaUnavailable as exc:
                warning = str(exc)
                with self._plan_lock:
                    self._plan_errors[(plan_key, model)] = (time.monotonic(), warning)
                return offline_plan, warning, 1, False, plan_key

        cache_pair = (plan_key, model)
        with self._plan_lock:
            err_entry = self._plan_errors.get(cache_pair)
            if err_entry is not None and (time.monotonic() - err_entry[0]) < 30.0:
                return offline_plan, err_entry[1], 1, False, plan_key
            already_pending = cache_pair in self._pending_plans
            if not already_pending:
                self._pending_plans.add(cache_pair)
                self._plan_errors.pop(cache_pair, None)

        if not already_pending:
            threading.Thread(
                target=self._background_expand_plan,
                args=(text, plan_key, model),
                daemon=True,
                name="artseek-llm-plan",
            ).start()
        return offline_plan, None, 1, True, plan_key

    def _invoke_llm(self, text: str) -> QueryPlan:
        raw_secs = int(getattr(self.lifecycle, "auto_unload_seconds", 0) or 0)
        keep_alive = -1 if raw_secs <= 0 else raw_secs
        expand_fn = self.lifecycle.llm.expand_query
        try:
            return expand_fn(text, keep_alive=keep_alive)
        except TypeError:
            return expand_fn(text)

    def _background_expand_plan(self, text: str, plan_key: str, model: str) -> None:
        cache_pair = (plan_key, model)
        try:
            llm_plan = self._invoke_llm(text)
            self.query_cache.put(plan_key, model, llm_plan.to_dict())
            with self._plan_lock:
                self._plan_errors.pop(cache_pair, None)
        except OllamaUnavailable as exc:
            with self._plan_lock:
                self._plan_errors[cache_pair] = (time.monotonic(), str(exc))
        except Exception as exc:
            with self._plan_lock:
                self._plan_errors[cache_pair] = (time.monotonic(), str(exc))
        finally:
            with self._plan_lock:
                self._pending_plans.discard(cache_pair)

    def get_query_plan_status(
        self,
        query: str,
        *,
        text_label: str = "",
        expand: bool = True,
    ) -> dict[str, Any]:
        cleaned, _, tag_tokens = parse_query_filters(query)
        parts = [cleaned, *tag_tokens]
        if text_label.strip():
            parts.append(text_label.strip())
        search_text = " ".join(p for p in parts if p).strip()
        if not search_text:
            return {
                "query": query,
                "plan_key": "",
                "ready": True,
                "pending": False,
                "plan_revision": 1,
                "query_plan": dataclasses.asdict(QueryPlan("", "")),
                "llm_warning": None,
            }
        plan_key = normalize_plan_key(search_text)
        model = str(self.config.llm.query_model or "")
        cached = (
            self.query_cache.get(plan_key, model)
            if (expand and self.config.llm.enabled)
            else None
        )
        if cached is not None:
            plan = QueryPlan.from_dict(cached)
            return {
                "query": query,
                "plan_key": plan_key,
                "ready": True,
                "pending": False,
                "plan_revision": 2,
                "query_plan": plan.to_dict(),
                "llm_warning": None,
            }
        cache_pair = (plan_key, model)
        with self._plan_lock:
            pending = cache_pair in self._pending_plans
            err_entry = self._plan_errors.get(cache_pair)
            warning = err_entry[1] if err_entry is not None else None
        offline = static_expand(search_text)
        return {
            "query": query,
            "plan_key": plan_key,
            "ready": not pending,
            "pending": pending,
            "plan_revision": 1,
            "query_plan": offline.to_dict(),
            "llm_warning": warning,
        }

    # ---------------------------------------------------------------- references & snapshots
    def resolve_library_references(
        self, references: Sequence[dict[str, Any]]
    ) -> tuple[list[Path], list[float]]:
        paths: list[Path] = []
        weights: list[float] = []
        seen: set[str] = set()
        for spec in references:
            pid = str(spec.get("id") or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            payload = self.images.get_by_point_id(pid)
            if not payload:
                raise ServiceError(404, "参考图片不在索引中")
            paths.append(Path(str(payload["path"])))
            weights.append(float(spec.get("weight", 1.0)))
        return paths, weights

    def _persist_snapshot(
        self,
        *,
        snapshot_id: str | None,
        query: str,
        text_weight: float,
        references: Sequence[dict[str, Any]],
        expand: bool,
        strict_tags: bool,
        dedupe: bool,
        limit: int,
        folder_keys: list[str] | None,
        mtime_after: str | None,
        mtime_before: str | None,
        offset: int,
        payload: dict[str, Any],
        had_file_refs: bool = False,
    ) -> dict[str, Any]:
        refs = [
            {"id": str(spec["id"]), "weight": float(spec.get("weight", 1.0))}
            for spec in references
            if spec.get("id")
        ]
        results = list(payload.get("results") or [])
        if offset <= 0:
            payload["snapshot_id"] = self.snapshots.create(
                query=query,
                text_weight=text_weight,
                references=refs,
                expand=expand,
                strict_tags=strict_tags,
                dedupe=dedupe,
                limit=limit,
                folder_keys=folder_keys,
                mtime_after=mtime_after,
                mtime_before=mtime_before,
                query_plan=payload.get("query_plan")
                if isinstance(payload.get("query_plan"), dict)
                else {},
                llm_warning=payload.get("llm_warning"),
                results=results,
                has_more=bool(payload.get("has_more")),
                had_file_refs=had_file_refs,
            )
            return payload
        if is_snapshot_id(snapshot_id) and self.snapshots.append(
            snapshot_id or "",
            results,
            has_more=bool(payload.get("has_more")),
        ):
            payload["snapshot_id"] = snapshot_id
        elif snapshot_id:
            payload["snapshot_id"] = snapshot_id
        return payload

    def read_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        if not is_snapshot_id(snapshot_id):
            raise ServiceError(400, "检索快照编号不合法")
        data = self.snapshots.get(snapshot_id)
        if data is None:
            raise ServiceError(404, "检索快照不存在或已过期")
        rows = list(data.get("results") or [])
        point_ids = [str(r.get("id") or "") for r in rows if r.get("id")]
        row_map = self.images.row_ids_for_points(point_ids)
        payloads = self.images.get_payloads_by_ids(list(row_map.values()))
        live: list[dict[str, Any]] = []
        for row in rows:
            pid = str(row.get("id") or "")
            if not pid:
                continue
            rid = row_map.get(pid)
            payload = payloads.get(rid) if rid is not None else None
            if payload is not None:
                item = gallery_result(pid, payload, fallback_name=str(row.get("name") or ""))
                item["score"] = row.get("score")
                item["tag_hits"] = list(row.get("tag_hits") or [])
                item["visual_score"] = row.get("visual_score")
                item["descriptor_score"] = row.get("descriptor_score")
                live.append(item)
            else:
                live.append(row)
        data["results"] = attach_provenance(self.prov, live)
        return data

    # ---------------------------------------------------------------- main search
    def search(
        self,
        *,
        query: str = "",
        references: Sequence[dict[str, Any]] = (),
        reference_id: str | None = None,
        upload_paths: Sequence[Path] = (),
        upload_weights: Sequence[float] | None = None,
        text_weight: float = 1.0,
        limit: int | None = None,
        expand: bool = True,
        strict_tags: bool = False,
        folder_keys: Sequence[str] | None = None,
        use_persisted_scope: bool = True,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
        offset: int = 0,
        exclude_ids: Sequence[str] = (),
        dedupe: bool = False,
        snapshot_id: str | None = None,
        author_key: str | None = None,
        author_keys: Sequence[str] = (),
        title_query: str = "",
        text_label: str = "",
        tag_expression: str = "",
        sort: str = "mtime_desc",
        seed: int | None = None,
        text_tags: Sequence[str] = (),
        tag_text: Sequence[str] = (),
        image_tags: Sequence[str] = (),
        wait_for_llm: bool = False,
        had_file_refs: bool = False,
    ) -> dict[str, Any]:
        specs = [
            {"id": str(s["id"]), "weight": float(s.get("weight", 1.0))}
            for s in references
            if s.get("id")
        ]
        if reference_id:
            specs.append({"id": reference_id, "weight": 1.0})

        lib_paths, lib_weights = self.resolve_library_references(specs)
        up_list = list(upload_paths)
        if upload_weights is not None:
            up_weights = [float(w) for w in upload_weights]
        else:
            up_weights = [1.0] * len(up_list)
        all_paths = [*lib_paths, *up_list]
        all_weights = [*lib_weights, *up_weights]
        if len(all_paths) > MAX_REFERENCE_IMAGES:
            raise ServiceError(400, f"参考图最多 {MAX_REFERENCE_IMAGES} 张")
        if len(all_weights) != len(all_paths):
            raise ServiceError(400, "image_weights 数量必须与参考图一致")

        expr_text = str(tag_expression or "").strip()
        if expr_text:
            try:
                parse_tag_expression(expr_text)
            except TagQueryError as exc:
                raise ServiceError(400, str(exc)) from exc

        cleaned_query, author_token, tag_tokens = parse_query_filters(query)
        requested_authors = [*author_keys]
        if author_key and author_key.strip():
            requested_authors.append(author_key)
        if author_token:
            requested_authors.append(author_token)
        resolved_authors = list(
            dict.fromkeys(
                resolve_author_token(self.prov, token)
                for token in requested_authors
                if str(token or "").strip()
            )
        )
        single_author = resolved_authors[0] if len(resolved_authors) == 1 else None
        merged_text_tags = [
            *[str(t).strip() for t in tag_tokens if str(t).strip()],
            *[str(t).strip() for t in text_tags if str(t).strip()],
        ]
        clean_tag_text = [str(t).strip() for t in tag_text if str(t).strip()]
        clean_image_tags = [str(t).strip() for t in image_tags if str(t).strip()]

        search_text = " ".join(p for p in (cleaned_query, *tag_tokens) if p).strip()
        if text_label.strip():
            search_text = " ".join([search_text, text_label.strip()]).strip()

        resolved_folders: list[str] | None
        if folder_keys is not None or use_persisted_scope:
            resolved_folders = resolve_folder_keys_from_db(self.images, folder_keys)
        else:
            resolved_folders = None

        after_ns = day_start_ns(mtime_after)
        before_ns = day_end_ns(mtime_before)
        time_range = (
            (after_ns, before_ns) if (after_ns is not None or before_ns is not None) else None
        )

        has_metadata_filters = bool(
            resolved_authors
            or merged_text_tags
            or title_query.strip()
            or clean_tag_text
            or clean_image_tags
            or expr_text
        )

        scope = ImageScope(
            folders=tuple(fast_folder_key(k) for k in resolved_folders)
            if resolved_folders is not None
            else None,
            folder_recursive=False,
            time_range=time_range,
            author_keys=tuple(resolved_authors),
            tags_all=tuple(clean_image_tags),
            source_tags_all=tuple(merged_text_tags),
            source_tag_text=tuple(clean_tag_text),
            tag_expression=expr_text,
            title_terms=(title_query.strip(),) if title_query.strip() else (),
        )

        page_limit = max(1, min(int(limit or self.config.search.default_limit), 200))
        page_offset = max(0, int(offset))
        applied_filters = {
            "author_key": single_author,
            "author_keys": resolved_authors,
            "tags": merged_text_tags,
            "title_query": title_query,
            "text_label": text_label,
            "tag_expression": expr_text,
            "tag_text": clean_tag_text,
            "image_tags": clean_image_tags,
        }

        if not search_text and not all_paths and has_metadata_filters:
            # Zero-model-load SQL path for filter-only searches
            row_ids, total, has_more, order, used_seed = self.resolver.page_rows(
                scope,
                sort=sort,
                limit=page_limit,
                offset=page_offset,
                seed=seed,
            )
            payloads_map = self.images.get_payloads_by_ids(row_ids)
            results = attach_provenance(
                self.prov,
                [
                    gallery_result(str(payloads_map[rid]["point_id"]), payloads_map[rid])
                    for rid in row_ids
                    if rid in payloads_map
                ],
            )
            plan = QueryPlan(original=query, semantic_zh="", positive_tags=merged_text_tags)
            out: dict[str, Any] = {
                "query_plan": dataclasses.asdict(plan),
                "plan_revision": 1,
                "llm_pending": False,
                "plan_key": "",
                "results": results,
                "total": total,
                "has_more": has_more,
                "offset": page_offset,
                "sort": order,
                "seed": used_seed if order == "random" else None,
                "applied_filters": applied_filters,
            }
            return self._persist_snapshot(
                snapshot_id=snapshot_id,
                query=query,
                text_weight=text_weight,
                references=specs,
                expand=expand,
                strict_tags=strict_tags,
                dedupe=dedupe,
                limit=page_limit,
                folder_keys=resolved_folders,
                mtime_after=mtime_after,
                mtime_before=mtime_before,
                offset=page_offset,
                payload=out,
                had_file_refs=had_file_refs,
            )

        if not search_text and not all_paths:
            raise ServiceError(400, "请提供文字或参考图片")

        plan, llm_warning, plan_revision, llm_pending, plan_key = self._resolve_plan(
            search_text,
            expand=expand,
            wait_for_llm=wait_for_llm,
        )

        excluded_pids = [pid for pid in dict.fromkeys(exclude_ids) if pid]
        seen = len(excluded_pids) if excluded_pids else page_offset
        multiplier = self.config.search.candidate_multiplier
        if dedupe:
            multiplier = max(multiplier * 2, 16)
        candidate_count = min(max((seen + page_limit) * multiplier, seen + page_limit, 100), 4000)

        scope_rows = self.resolver.rows(scope)
        if scope_rows is not None and scope_rows.size == 0:
            empty_out: dict[str, Any] = {
                "query_plan": dataclasses.asdict(plan),
                "plan_revision": plan_revision,
                "llm_pending": llm_pending,
                "plan_key": plan_key,
                "llm_warning": llm_warning,
                "results": [],
                "has_more": False,
                "offset": page_offset,
                "applied_filters": applied_filters,
            }
            return self._persist_snapshot(
                snapshot_id=snapshot_id,
                query=query,
                text_weight=text_weight,
                references=specs,
                expand=expand,
                strict_tags=strict_tags,
                dedupe=dedupe,
                limit=page_limit,
                folder_keys=resolved_folders,
                mtime_after=mtime_after,
                mtime_before=mtime_before,
                offset=page_offset,
                payload=empty_out,
                had_file_refs=had_file_refs,
            )

        with self.lifecycle.hold_model() as embedder:
            text_vec: np.ndarray | None = None
            if search_text:
                text_vec = embedder.encode_texts([plan.embedding_text])[0]
            active_paths = [
                path for path, w in zip(all_paths, all_weights, strict=True) if w > 0
            ]
            active_weights = [w for w in all_weights if w > 0]
            img_vecs: list[np.ndarray] = []
            if active_paths:
                encoded = embedder.encode_images(active_paths)
                img_vecs = [encoded[i] for i in range(len(active_paths))]

        try:
            visual_vec = compose_visual_vector(
                text_vec,
                img_vecs,
                text_weight=text_weight,
                image_weights=active_weights,
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc

        vis_rows, vis_scores = self.vectors.search(
            visual_vec, "visual", candidate_count, rows=scope_rows
        )
        visual_hits = list(zip(vis_rows.tolist(), vis_scores.tolist(), strict=True))
        descriptor_hits: list[tuple[int, float]] = []
        if text_vec is not None:
            desc_rows, desc_scores = self.vectors.search(
                text_vec, "descriptor", candidate_count, rows=scope_rows
            )
            descriptor_hits = list(zip(desc_rows.tolist(), desc_scores.tolist(), strict=True))

        cand_rows = list(
            dict.fromkeys([r for r, _ in visual_hits] + [r for r, _ in descriptor_hits])
        )
        excluded_row_map = (
            self.images.row_ids_for_points(excluded_pids) if excluded_pids else {}
        )
        excluded_row_ids = set(excluded_row_map.values())
        all_fetch_rows = list(dict.fromkeys([*cand_rows, *excluded_row_ids]))
        cand_payloads = self.images.get_ranking_candidates(
            all_fetch_rows,
            [*plan.positive_tags, *plan.negative_tags],
            include_dedupe_meta=dedupe,
        )
        excluded_payloads = [
            cand_payloads[rid] for rid in excluded_row_ids if rid in cand_payloads
        ]

        desc_weight = self.config.search.descriptor_weight if descriptor_hits else 0.0
        raw_items, has_more = fuse_and_rank(
            visual_hits,
            descriptor_hits,
            payloads=cand_payloads,
            plan=plan,
            descriptor_weight=desc_weight,
            keyword_bonus=self.config.search.keyword_bonus,
            strict_tags=strict_tags,
            dedupe=dedupe,
            excluded_ids=excluded_row_ids or None,
            excluded_payloads=excluded_payloads,
            limit=page_limit,
            offset=page_offset,
        )
        win_rids = [int(item["id"]) for item in raw_items]
        full_payloads = self.images.get_payloads_by_ids(win_rids)
        items: list[dict[str, Any]] = []
        for item in raw_items:
            rid = int(item["id"])
            payload = full_payloads.get(rid)
            if payload is None:
                continue
            item["id"] = str(payload.get("point_id") or payload.get("id") or rid)
            item["path"] = str(payload.get("path", ""))
            item["name"] = str(payload.get("name", ""))
            item["width"] = int(payload.get("width", 0))
            item["height"] = int(payload.get("height", 0))
            item["tags"] = list(payload.get("tags", []))
            item["caption"] = payload.get("caption")
            items.append(item)
        items = attach_provenance(self.prov, items)
        result_out: dict[str, Any] = {
            "query_plan": dataclasses.asdict(plan),
            "plan_revision": plan_revision,
            "llm_pending": llm_pending,
            "plan_key": plan_key,
            "llm_warning": llm_warning,
            "results": items,
            "has_more": has_more,
            "offset": page_offset,
            "applied_filters": applied_filters,
        }
        return self._persist_snapshot(
            snapshot_id=snapshot_id,
            query=query,
            text_weight=text_weight,
            references=specs,
            expand=expand,
            strict_tags=strict_tags,
            dedupe=dedupe,
            limit=page_limit,
            folder_keys=resolved_folders,
            mtime_after=mtime_after,
            mtime_before=mtime_before,
            offset=page_offset,
            payload=result_out,
            had_file_refs=had_file_refs,
        )

    def local_search(
        self,
        *,
        query: str,
        limit: int = 40,
        offset: int = 0,
        folder_keys: Sequence[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
        sort: str = "mtime_desc",
        seed: int | None = None,
    ) -> dict[str, Any]:
        terms = [
            term.lstrip("#")
            for term in re.split(r"\s+", (query or "").strip())
            if term.lstrip("#")
        ]
        if not terms:
            raise ServiceError(400, "请输入 tag、标题或文件名关键词")
        groups: list[tuple[str, list[str]]] = []
        for term in terms:
            aliases = set(expand_tag_aliases(term))
            aliases.update(static_expand(term).positive_tags)
            groups.append((term, sorted(aliases)))
        point_ids = self.prov.point_ids_for_metadata_search(groups)
        resolved_folders = resolve_folder_keys_from_db(self.images, folder_keys)
        after_ns = day_start_ns(mtime_after)
        before_ns = day_end_ns(mtime_before)
        time_range = (
            (after_ns, before_ns) if (after_ns is not None or before_ns is not None) else None
        )
        page_limit = max(1, min(int(limit), 200))
        page_offset = max(0, int(offset))
        scope = ImageScope(
            folders=tuple(resolved_folders),
            folder_recursive=False,
            time_range=time_range,
            restrict_point_ids=tuple(point_ids),
        )
        row_ids, total, has_more, order, used_seed = self.resolver.page_rows(
            scope,
            sort=sort,
            limit=page_limit,
            offset=page_offset,
            seed=seed,
        )
        payloads = self.images.get_payloads_by_ids(row_ids)
        results = attach_provenance(
            self.prov,
            [
                gallery_result(str(payloads[rid]["point_id"]), payloads[rid])
                for rid in row_ids
                if rid in payloads
            ],
        )
        plan = QueryPlan(original=query, semantic_zh="", positive_tags=[])
        return {
            "query_plan": dataclasses.asdict(plan),
            "results": results,
            "total": total,
            "has_more": has_more,
            "offset": page_offset,
            "sort": order,
            "seed": used_seed if order == "random" else None,
            "metadata_search": True,
            "applied_filters": {
                "author_key": None,
                "author_keys": [],
                "tags": [],
                "title_query": "",
                "tag_text": [],
                "image_tags": [],
                "mode": "metadata",
            },
        }
