from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from .config import AppConfig

COLLECTION = "artseek_images"


@dataclass(slots=True)
class Neighbor:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, config: AppConfig):
        config.ensure_dirs()
        self.config = config
        self.client = QdrantClient(path=str(config.qdrant_path))

    def close(self) -> None:
        self.client.close()

    def ensure_collection(self, dimension: int) -> None:
        if self.config.manifest_path.exists():
            try:
                manifest = json.loads(
                    self.config.manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("索引清单损坏，请运行 artseek rebuild 重建") from exc
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

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=COLLECTION,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )

    def clear(self) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if COLLECTION in existing:
            self.client.delete_collection(COLLECTION)
        if self.config.manifest_path.exists():
            self.config.manifest_path.unlink()

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

    def query(self, vector: list[float], using: str, limit: int) -> list[Neighbor]:
        response = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            using=using,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            Neighbor(str(point.id), float(point.score), dict(point.payload or {}))
            for point in response.points
        ]

    def get(self, point_id: str) -> dict[str, Any] | None:
        records = self.client.retrieve(
            collection_name=COLLECTION,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        return dict(records[0].payload or {}) if records else None

    def count(self) -> int:
        existing = {item.name for item in self.client.get_collections().collections}
        if COLLECTION not in existing:
            return 0
        return int(self.client.count(COLLECTION, exact=True).count)

    def iter_payloads(self) -> Iterable[tuple[str, dict[str, Any]]]:
        existing = {item.name for item in self.client.get_collections().collections}
        if COLLECTION not in existing:
            return
        offset: Any = None
        while True:
            records, offset = self.client.scroll(
                collection_name=COLLECTION,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                yield str(record.id), dict(record.payload or {})
            if offset is None:
                break
