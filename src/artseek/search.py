from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig
from .library import (
    FolderCatalog,
    build_catalog,
    day_end_ns,
    day_start_ns,
    load_catalog,
    rebuild_catalog,
)
from .llm import OllamaClient, OllamaUnavailable
from .models.embedder import SiglipEmbedder
from .query import QueryPlan, normalize_tag, static_expand
from .store import Neighbor, VectorStore

RankedRow = tuple[str, dict[str, Any], list[str]]
MAX_REFERENCE_IMAGES = 8


def compose_visual_vector(
    text_vector: np.ndarray | None,
    image_vectors: Sequence[np.ndarray],
    *,
    text_weight: float = 1.0,
    image_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted sum of text and reference-image vectors, then L2-normalized."""
    parts: list[np.ndarray] = []
    if text_vector is not None and text_weight > 0:
        parts.append(np.asarray(text_vector, dtype=np.float32) * float(text_weight))
    weights = list(image_weights) if image_weights is not None else [1.0] * len(image_vectors)
    if len(weights) != len(image_vectors):
        raise ValueError("image_weights 数量必须与参考图一致")
    for vector, weight in zip(image_vectors, weights, strict=True):
        if weight > 0:
            parts.append(np.asarray(vector, dtype=np.float32) * float(weight))
    if not parts:
        raise ValueError("文字与参考图的权重不能全为 0")
    combined = np.sum(np.stack(parts, axis=0), axis=0)
    combined /= max(float(np.linalg.norm(combined)), 1e-12)
    return combined.astype(np.float32, copy=False)


def duplicate_key(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    """Identity of a byte-identical copy: content hash, else size plus pixel size."""
    digest = str(payload.get("content_hash") or "").strip()
    if digest:
        return ("h", digest)
    try:
        size = int(payload.get("size_bytes") or 0)
        width = int(payload.get("width") or 0)
        height = int(payload.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if size > 0 and width > 0 and height > 0:
        return ("s", size, width, height)
    return None


def keep_first_unique(
    ranked: Sequence[RankedRow],
    seen_keys: set[tuple[Any, ...]] | None = None,
) -> list[RankedRow]:
    seen: set[tuple[Any, ...]] = set(seen_keys or ())
    unique: list[RankedRow] = []
    for row in ranked:
        key = duplicate_key(row[1]["payload"])
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        unique.append(row)
    return unique


@dataclass(slots=True)
class SearchResult:
    id: str
    score: float
    path: str
    name: str
    width: int
    height: int
    tags: list[str]
    tag_hits: list[str]
    caption: dict[str, Any] | None
    visual_score: float | None = None
    descriptor_score: float | None = None


@dataclass(slots=True)
class SearchResponse:
    query_plan: QueryPlan
    results: list[SearchResult]
    llm_warning: str | None = None
    has_more: bool = False
    offset: int = 0


class SearchEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.embedder = SiglipEmbedder(config.embedding)
        self.store = VectorStore(config)
        self.store.ensure_collection(self.embedder.dimension)
        self.llm = OllamaClient(config.llm)
        self._folder_catalog: FolderCatalog | None = None
        self._folder_catalog_count: int | None = None
        self._plan_cache: dict[str, QueryPlan] = {}

    def close(self) -> None:
        self.store.close()

    def warmup(self, status: Callable[[str], None] | None = None) -> None:
        """Compile inference kernels and preload the query LLM before serving."""
        notify = status or (lambda _message: None)
        notify("正在预热图文向量模型……")
        warmup = getattr(self.embedder, "warmup", None)
        if callable(warmup):
            warmup()
        else:
            self.embedder.encode_texts(["warmup"])
        if not self.config.llm.enabled:
            return
        notify(f"正在载入本地大模型 {self.config.llm.query_model}……")
        try:
            self.llm.preload()
            notify(f"本地大模型 {self.config.llm.query_model} 已载入")
        except OllamaUnavailable as exc:
            notify(f"Ollama 未就绪，查询将使用内置词典：{exc}")

    def invalidate_folder_catalog(self) -> None:
        self._folder_catalog = None
        self._folder_catalog_count = None
        self.store.invalidate_folder_catalog()

    def folder_catalog(self) -> FolderCatalog:
        count = int(self.store.count()) if hasattr(self.store, "count") else None
        cached = getattr(self, "_folder_catalog", None)
        if cached is not None and getattr(self, "_folder_catalog_count", None) == count:
            return cached
        catalog = load_catalog(self.config.folder_catalog_path, count) if count is not None else None
        if catalog is None:
            if count is not None:
                catalog = rebuild_catalog(self.store, self.config)
            else:
                rows = getattr(self.store, "iter_folder_payloads", self.store.iter_payloads)()
                catalog = build_catalog(rows)
        self._folder_catalog = catalog
        self._folder_catalog_count = count
        return catalog

    def _plan(self, text: str, expand: bool) -> tuple[QueryPlan, str | None]:
        if not expand or not self.config.llm.enabled:
            return static_expand(text), None
        cache = getattr(self, "_plan_cache", None)
        if cache is None:
            self._plan_cache = {}
            cache = self._plan_cache
        cached = cache.get(text)
        if cached is not None:
            return cached, None
        try:
            plan = self.llm.expand_query(text)
        except OllamaUnavailable as exc:
            return static_expand(text), str(exc)
        if len(cache) >= 64:
            cache.pop(next(iter(cache)))
        cache[text] = plan
        return plan, None

    @staticmethod
    def _add_ranked(
        combined: dict[str, dict[str, Any]],
        neighbors: list[Neighbor],
        channel: str,
        weight: float,
    ) -> None:
        for rank, neighbor in enumerate(neighbors, start=1):
            item = combined.setdefault(
                neighbor.id,
                {"payload": neighbor.payload, "rrf": 0.0, "raw": {}},
            )
            item["rrf"] += weight / (60.0 + rank)
            item["raw"][channel] = neighbor.score

    def search(
        self,
        text: str = "",
        *,
        image: Path | None = None,
        images: Sequence[Path] | None = None,
        text_weight: float = 1.0,
        image_weights: Sequence[float] | None = None,
        limit: int | None = None,
        expand: bool = True,
        strict_tags: bool = False,
        folder_keys: Sequence[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
        offset: int = 0,
        exclude_ids: Sequence[str] | None = None,
        dedupe: bool = False,
    ) -> SearchResponse:
        refs: list[Path] = []
        if images:
            refs.extend(images)
        if image is not None:
            refs.append(image)
        if len(refs) > MAX_REFERENCE_IMAGES:
            raise ValueError(f"参考图最多 {MAX_REFERENCE_IMAGES} 张")
        if not text.strip() and not refs:
            raise ValueError("请提供文字或参考图片")
        if image_weights is not None and len(image_weights) != len(refs):
            raise ValueError("image_weights 数量必须与参考图一致")
        limit = limit or self.config.search.default_limit
        offset = max(0, offset)
        excluded = {item_id for item_id in (exclude_ids or []) if item_id}
        seen = len(excluded) if excluded else offset
        plan, llm_warning = self._plan(text.strip(), expand) if text.strip() else (
            QueryPlan("", ""),
            None,
        )
        needed = seen + limit
        multiplier = self.config.search.candidate_multiplier
        if dedupe:
            multiplier = max(multiplier * 2, 16)
        candidate_count = min(
            max(needed * multiplier, needed, 100),
            4000,
        )
        after_ns = day_start_ns(mtime_after)
        before_ns = day_end_ns(mtime_before)
        point_ids: list[str] | None = None
        universe: int | None = None
        if folder_keys is not None or after_ns is not None or before_ns is not None:
            catalog = self.folder_catalog()
            selected = folder_keys if folder_keys is not None else list(catalog.all_keys())
            point_ids = catalog.point_ids(selected, after_ns=after_ns, before_ns=before_ns)
            universe = sum(len(folder.ids) for folder in catalog.folders.values())
            if point_ids is not None and not point_ids:
                return SearchResponse(plan, [], llm_warning, has_more=False, offset=offset)

        text_vector = None
        if text.strip():
            text_vector = self.embedder.encode_texts([plan.embedding_text])[0]
        declared_weights = (
            [float(weight) for weight in image_weights]
            if image_weights is not None
            else [1.0] * len(refs)
        )
        active_paths = [path for path, weight in zip(refs, declared_weights) if weight > 0]
        active_weights = [weight for weight in declared_weights if weight > 0]
        image_vectors: list[np.ndarray] = []
        if active_paths:
            encoded = self.embedder.encode_images(active_paths)
            image_vectors = [encoded[index] for index in range(len(active_paths))]
        visual_vector = compose_visual_vector(
            text_vector,
            image_vectors,
            text_weight=text_weight,
            image_weights=active_weights,
        )

        visual = self.store.query(
            visual_vector.tolist(),
            using="visual",
            limit=candidate_count,
            point_ids=point_ids,
            universe=universe,
        )
        descriptor: list[Neighbor] = []
        if text_vector is not None:
            descriptor = self.store.query(
                text_vector.tolist(),
                using="descriptor",
                limit=candidate_count,
                point_ids=point_ids,
                universe=universe,
            )

        combined: dict[str, dict[str, Any]] = {}
        descriptor_weight = self.config.search.descriptor_weight if descriptor else 0.0
        visual_weight = 1.0 - descriptor_weight
        self._add_ranked(combined, visual, "visual", visual_weight)
        self._add_ranked(combined, descriptor, "descriptor", descriptor_weight)

        positive = {normalize_tag(tag) for tag in plan.positive_tags}
        negative = {normalize_tag(tag) for tag in plan.negative_tags}
        ranked: list[tuple[str, dict[str, Any], list[str]]] = []
        for item_id, item in combined.items():
            payload = item["payload"]
            tags = {normalize_tag(tag) for tag in payload.get("tags", [])}
            if tags & negative:
                continue
            hits = sorted(tags & positive)
            if strict_tags and positive and not hits:
                continue
            if positive:
                item["rrf"] += (
                    self.config.search.keyword_bonus * len(hits) / len(positive)
                )
            ranked.append((item_id, item, hits))

        ranked.sort(key=lambda row: row[1]["rrf"], reverse=True)
        max_score = ranked[0][1]["rrf"] if ranked else 1.0
        if dedupe:
            seen_keys: set[tuple[Any, ...]] = set()
            if excluded:
                for payload in self._retrieve_payloads(list(excluded)).values():
                    key = duplicate_key(payload)
                    if key is not None:
                        seen_keys.add(key)
            ranked = keep_first_unique(ranked, seen_keys)
        if excluded:
            ranked = [row for row in ranked if row[0] not in excluded]
        elif offset:
            ranked = ranked[offset:]
        selected = ranked[:limit]
        results: list[SearchResult] = []
        for item_id, item, hits in selected:
            payload = item["payload"]
            raw = item["raw"]
            results.append(
                SearchResult(
                    id=item_id,
                    score=round(100.0 * item["rrf"] / max_score, 2),
                    path=str(payload.get("path", "")),
                    name=str(payload.get("name", "")),
                    width=int(payload.get("width", 0)),
                    height=int(payload.get("height", 0)),
                    tags=list(payload.get("tags", [])),
                    tag_hits=hits,
                    caption=payload.get("caption"),
                    visual_score=raw.get("visual"),
                    descriptor_score=raw.get("descriptor"),
                )
            )
        has_more = len(ranked) > len(results)
        return SearchResponse(
            plan, results, llm_warning, has_more=has_more, offset=offset
        )

    def _retrieve_payloads(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        method = getattr(self.store, "retrieve_payloads", None)
        if callable(method):
            return method(list(ids))
        get_one = getattr(self.store, "get", None)
        if not callable(get_one):
            return {}
        found: dict[str, dict[str, Any]] = {}
        for item_id in ids:
            payload = get_one(item_id)
            if payload:
                found[str(item_id)] = payload
        return found

    def similar(
        self,
        image: Path,
        *,
        limit: int | None = None,
        folder_keys: Sequence[str] | None = None,
        mtime_after: str | None = None,
        mtime_before: str | None = None,
    ) -> SearchResponse:
        return self.search(
            image=image,
            limit=limit,
            expand=False,
            folder_keys=folder_keys,
            mtime_after=mtime_after,
            mtime_before=mtime_before,
        )
