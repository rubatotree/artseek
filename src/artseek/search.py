from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .llm import OllamaClient, OllamaUnavailable
from .models.embedder import SiglipEmbedder
from .query import QueryPlan, normalize_tag, static_expand
from .store import Neighbor, VectorStore


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


class SearchEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.embedder = SiglipEmbedder(config.embedding)
        self.store = VectorStore(config)
        self.store.ensure_collection(self.embedder.dimension)
        self.llm = OllamaClient(config.llm)

    def close(self) -> None:
        self.store.close()

    def _plan(self, text: str, expand: bool) -> tuple[QueryPlan, str | None]:
        if not expand or not self.config.llm.enabled:
            return static_expand(text), None
        try:
            return self.llm.expand_query(text), None
        except OllamaUnavailable as exc:
            return static_expand(text), str(exc)

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
        limit: int | None = None,
        expand: bool = True,
        strict_tags: bool = False,
        image_weight: float = 0.65,
    ) -> SearchResponse:
        if not text.strip() and image is None:
            raise ValueError("请提供文字或参考图片")
        limit = limit or self.config.search.default_limit
        plan, llm_warning = self._plan(text.strip(), expand) if text.strip() else (
            QueryPlan("", ""),
            None,
        )
        candidate_count = max(limit * self.config.search.candidate_multiplier, 100)

        if image is not None and text.strip():
            visual_vector = self.embedder.encode_composed(
                image, plan.embedding_text, image_weight=image_weight
            )
        elif image is not None:
            visual_vector = self.embedder.encode_images([image])[0]
        else:
            visual_vector = self.embedder.encode_texts([plan.embedding_text])[0]

        visual = self.store.query(
            visual_vector.tolist(), using="visual", limit=candidate_count
        )
        descriptor: list[Neighbor] = []
        if text.strip():
            descriptor_vector = self.embedder.encode_texts([plan.embedding_text])[0]
            descriptor = self.store.query(
                descriptor_vector.tolist(), using="descriptor", limit=candidate_count
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
        selected = ranked[:limit]
        max_score = selected[0][1]["rrf"] if selected else 1.0
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
        return SearchResponse(plan, results, llm_warning)

    def similar(self, image: Path, *, limit: int | None = None) -> SearchResponse:
        return self.search(image=image, limit=limit, expand=False)
