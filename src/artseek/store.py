from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from qdrant_client import QdrantClient, models

from .config import AppConfig

COLLECTION = "artseek_images"
FOLDER_PAYLOAD_FIELDS = ("path", "path_key", "root", "root_key", "folder_key", "mtime_ns")
SEARCH_PAYLOAD_FIELDS = (
    "path",
    "name",
    "width",
    "height",
    "size_bytes",
    "content_hash",
    "tags",
    "caption",
)
# Local Qdrant 的 HasIdCondition 对每个点做 list 成员检查，上万 ID 会把查询拖成数十秒。
SUBSET_POSTFILTER_RATIO = 0.35


def query_mode(selected: int, total: int) -> Literal["empty", "all", "postfilter", "subset"]:
    if selected <= 0:
        return "empty"
    if total <= 0 or selected >= total:
        return "all"
    if selected >= max(int(total * SUBSET_POSTFILTER_RATIO), 1):
        return "postfilter"
    return "subset"


@dataclass(slots=True)
class Neighbor:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, config: AppConfig):
        config.ensure_dirs()
        self.config = config
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Local mode is not recommended for collections with more than 20,000 points"
                ),
                category=UserWarning,
            )
            self.client = QdrantClient(path=str(config.qdrant_path))

    def close(self) -> None:
        self.client.close()

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.config.manifest_path.exists():
            return None
        try:
            data = json.loads(self.config.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("索引清单损坏，请运行 artseek rebuild 重建") from exc
        return data if isinstance(data, dict) else None

    def known_dimension(self) -> int | None:
        manifest = self._read_manifest()
        if manifest:
            value = manifest.get("dimension")
            try:
                dimension = int(value)
            except (TypeError, ValueError):
                dimension = 0
            if dimension > 0:
                return dimension
        existing = {item.name for item in self.client.get_collections().collections}
        if COLLECTION not in existing:
            return None
        info = self.client.get_collection(COLLECTION)
        vectors = info.config.params.vectors
        try:
            return int(vectors["visual"].size)  # type: ignore[index]
        except (KeyError, TypeError, AttributeError):
            return None

    def ensure_collection(self, dimension: int) -> None:
        manifest = self._read_manifest()
        if manifest:
            indexed_model = manifest.get("embedding_model")
            if indexed_model and indexed_model != self.config.embedding.model:
                raise RuntimeError(
                    f"索引使用 {indexed_model}，当前配置使用 {self.config.embedding.model}。"
                    "请运行 artseek rebuild <图片目录> 重建索引。"
                )
        existing = {item.name for item in self.client.get_collections().collections}
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config={
                    "visual": models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                    "descriptor": models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    ),
                },
            )
        else:
            info = self.client.get_collection(COLLECTION)
            vectors = info.config.params.vectors
            try:
                current_size = int(vectors["visual"].size)  # type: ignore[index]
            except (KeyError, TypeError, AttributeError) as exc:
                raise RuntimeError("现有向量库不是 ArtSeek 双路索引，请另设 data_dir") from exc
            if current_size != dimension:
                raise RuntimeError(
                    f"向量模型维度发生变化（索引 {current_size}，当前 {dimension}）。"
                    "请执行 artseek rebuild <图片目录> 重建索引。"
                )
        self._write_manifest(dimension)

    def _write_manifest(self, dimension: int) -> None:
        content = {
            "schema_version": 1,
            "collection": COLLECTION,
            "embedding_model": self.config.embedding.model,
            "dimension": dimension,
            "tagger_model": self.config.tagger.model if self.config.tagger.enabled else None,
        }
        self.config.manifest_path.write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def upsert(self, points: Iterable[models.PointStruct]) -> None:
        self.client.upsert(collection_name=COLLECTION, points=list(points), wait=True)
        self.invalidate_folder_catalog()

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=COLLECTION,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )
        self.invalidate_folder_catalog()

    def clear(self) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if COLLECTION in existing:
            self.client.delete_collection(COLLECTION)
        if self.config.manifest_path.exists():
            self.config.manifest_path.unlink()
        self.invalidate_folder_catalog()

    def invalidate_folder_catalog(self) -> None:
        path = self.config.folder_catalog_path
        if path.exists():
            path.unlink()

    def scroll_root(self, root_key: str) -> dict[str, tuple[str, dict[str, Any]]]:
        result: dict[str, tuple[str, dict[str, Any]]] = {}
        offset: Any = None
        root_filter = models.Filter(
            must=[
                models.FieldCondition(key="root_key", match=models.MatchValue(value=root_key))
            ]
        )
        while True:
            records, offset = self.client.scroll(
                collection_name=COLLECTION,
                scroll_filter=root_filter,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = dict(record.payload or {})
                path_key = str(payload.get("path_key", ""))
                if path_key:
                    result[path_key] = (str(record.id), payload)
            if offset is None:
                break
        return result

    def query(
        self,
        vector: list[float],
        using: str,
        limit: int,
        point_ids: Sequence[str] | None = None,
        universe: int | None = None,
    ) -> list[Neighbor]:
        if point_ids is None:
            return self._query_points(vector, using, limit)
        if not point_ids:
            return []
        total = universe if universe is not None else self.count()
        mode = query_mode(len(point_ids), total)
        if mode == "all":
            return self._query_points(vector, using, limit)
        if mode == "postfilter":
            return self._query_then_filter(vector, using, limit, set(point_ids), total)
        return self._score_subset(vector, using, list(point_ids), limit)

    def _query_points(self, vector: list[float], using: str, limit: int) -> list[Neighbor]:
        response = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            using=using,
            limit=max(1, limit),
            with_payload=list(SEARCH_PAYLOAD_FIELDS),
            with_vectors=False,
        )
        return [
            Neighbor(str(point.id), float(point.score), dict(point.payload or {}))
            for point in response.points
        ]

    def _query_then_filter(
        self,
        vector: list[float],
        using: str,
        limit: int,
        allowed: set[str],
        total: int,
    ) -> list[Neighbor]:
        probe = min(max(limit * 10, 400), total)
        hits = [item for item in self._query_points(vector, using, probe) if item.id in allowed]
        if len(hits) >= limit or probe >= total:
            return hits[:limit]
        probe = min(max(limit * 40, 2000), total)
        hits = [item for item in self._query_points(vector, using, probe) if item.id in allowed]
        return hits[:limit]

    def _score_subset(
        self,
        vector: list[float],
        using: str,
        point_ids: list[str],
        limit: int,
    ) -> list[Neighbor]:
        query = np.asarray(vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm > 0:
            query = query / query_norm
        records = self.client.retrieve(
            collection_name=COLLECTION,
            ids=point_ids,
            with_payload=False,
            with_vectors=[using],
        )
        scored: list[tuple[float, str]] = []
        for record in records:
            named = _named_vector(record.vector, using)
            if named is None:
                continue
            denom = float(np.linalg.norm(named))
            if denom == 0:
                continue
            scored.append((float(query @ (named / denom)), str(record.id)))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: max(1, limit)]
        if not top:
            return []
        payloads = {
            str(record.id): dict(record.payload or {})
            for record in self.client.retrieve(
                collection_name=COLLECTION,
                ids=[item_id for _, item_id in top],
                with_payload=list(SEARCH_PAYLOAD_FIELDS),
                with_vectors=False,
            )
        }
        return [
            Neighbor(item_id, score, payloads.get(item_id, {}))
            for score, item_id in top
        ]

    def get(self, point_id: str) -> dict[str, Any] | None:
        records = self.client.retrieve(
            collection_name=COLLECTION,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        return dict(records[0].payload or {}) if records else None

    def retrieve_payloads(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        unique = [item_id for item_id in dict.fromkeys(ids) if item_id]
        if not unique:
            return {}
        fields = list(SEARCH_PAYLOAD_FIELDS)
        result: dict[str, dict[str, Any]] = {}
        for start in range(0, len(unique), 256):
            batch = unique[start : start + 256]
            records = self.client.retrieve(
                collection_name=COLLECTION,
                ids=batch,
                with_payload=fields,
                with_vectors=False,
            )
            for record in records:
                result[str(record.id)] = dict(record.payload or {})
        return result

    def count(self) -> int:
        existing = {item.name for item in self.client.get_collections().collections}
        if COLLECTION not in existing:
            return 0
        return int(self.client.count(COLLECTION, exact=True).count)

    def iter_payloads(self) -> Iterable[tuple[str, dict[str, Any]]]:
        yield from self._scroll_payloads(with_payload=True)

    def iter_folder_payloads(self) -> Iterable[tuple[str, dict[str, Any]]]:
        yield from self._scroll_payloads(with_payload=list(FOLDER_PAYLOAD_FIELDS))

    def _scroll_payloads(
        self,
        *,
        with_payload: bool | Sequence[str],
    ) -> Iterable[tuple[str, dict[str, Any]]]:
        existing = {item.name for item in self.client.get_collections().collections}
        if COLLECTION not in existing:
            return
        offset: Any = None
        while True:
            records, offset = self.client.scroll(
                collection_name=COLLECTION,
                limit=2048,
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )
            for record in records:
                yield str(record.id), dict(record.payload or {})
            if offset is None:
                break


def _named_vector(vector: Any, using: str) -> np.ndarray | None:
    if vector is None:
        return None
    if isinstance(vector, dict):
        data = vector.get(using)
    else:
        data = vector
    if data is None:
        return None
    return np.asarray(data, dtype=np.float32)
