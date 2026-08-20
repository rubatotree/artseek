from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from qdrant_client import models

from .config import AppConfig
from .image_io import canonical_path, image_metadata, load_rgb, point_id, scan_images
from .llm import OllamaClient, OllamaUnavailable, flatten_caption
from .models.embedder import SiglipEmbedder
from .models.tagger import PredictedTags, WDTagger
from .store import VectorStore

ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class IndexSummary:
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    llm_warning: str | None = None


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class ImageIndexer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.embedder = SiglipEmbedder(config.embedding)
        self.tagger = (
            WDTagger(config.tagger, config.embedding) if config.tagger.enabled else None
        )
        self.llm = OllamaClient(config.llm)
        self.store = VectorStore(config)
        self.store.ensure_collection(self.embedder.dimension)

    def close(self) -> None:
        self.store.close()

    def _describe_batch(
        self,
        batch: Sequence[tuple[Path, Path]],
        images: Sequence[Image.Image],
        use_captions: bool,
        summary: IndexSummary,
    ) -> tuple[list[PredictedTags], list[dict[str, Any] | None]]:
        if self.tagger:
            tags = self.tagger.predict(images)
        else:
            tags = [PredictedTags([], {}) for _ in images]

        captions: list[dict[str, Any] | None] = [None] * len(batch)
        if use_captions:
            for index, (_, path) in enumerate(batch):
                try:
                    captions[index] = self.llm.caption_image(path)
                except OllamaUnavailable as exc:
                    summary.llm_warning = str(exc)
                    break
        return tags, captions

    def index(
        self,
        roots: Sequence[Path],
        *,
        recursive: bool = True,
        force: bool = False,
        captions: bool | None = None,
        remove_missing: bool = True,
        progress: ProgressCallback | None = None,
    ) -> IndexSummary:
        summary = IndexSummary()
        use_captions = self.config.llm.caption_on_index if captions is None else captions
        batch_size = min(
            self.config.embedding.batch_size,
            self.config.tagger.batch_size if self.tagger else self.config.embedding.batch_size,
        )

        for root in roots:
            resolved_root = root.expanduser().resolve()
            root_key = canonical_path(resolved_root)
            existing = self.store.scroll_root(root_key)
            scanned = list(scan_images([resolved_root], recursive=recursive))
            summary.discovered += len(scanned)
            present_keys = {canonical_path(path) for _, path in scanned}

            changed: list[tuple[Path, Path]] = []
            for pair in scanned:
                _, path = pair
                key = canonical_path(path)
                previous = existing.get(key)
                stat = path.stat()
                unchanged = (
                    previous
                    and int(previous[1].get("mtime_ns", -1)) == stat.st_mtime_ns
                    and int(previous[1].get("size_bytes", -1)) == stat.st_size
                )
                if unchanged and not force:
                    summary.unchanged += 1
                else:
                    changed.append(pair)

            completed = 0
            for raw_batch in chunks(changed, batch_size):
                batch = list(raw_batch)
                valid_pairs: list[tuple[Path, Path]] = []
                images: list[Image.Image] = []
                for pair in batch:
                    try:
                        images.append(load_rgb(pair[1]))
                        valid_pairs.append(pair)
                    except (OSError, ValueError) as exc:
                        summary.failed.append((str(pair[1]), str(exc)))
                if not valid_pairs:
                    continue

                visual_vectors = self.embedder.encode_images(images)
                predicted_tags, structured_captions = self._describe_batch(
                    valid_pairs, images, use_captions, summary
                )
                descriptor_texts: list[str] = []
                for tags, caption in zip(
                    predicted_tags, structured_captions, strict=True
                ):
                    tag_text = ", ".join(tag.replace("_", " ") for tag in tags.tags)
                    parts = [flatten_caption(caption), tag_text]
                    descriptor_texts.append("。".join(filter(None, parts)))

                descriptor_vectors = np.array(visual_vectors, copy=True)
                nonempty_indices = [
                    index for index, text in enumerate(descriptor_texts) if text.strip()
                ]
                if nonempty_indices:
                    encoded = self.embedder.encode_texts(
                        [descriptor_texts[index] for index in nonempty_indices]
                    )
                    for target, vector in zip(nonempty_indices, encoded, strict=True):
                        descriptor_vectors[target] = vector

                points: list[models.PointStruct] = []
                for index, ((_, path), image, tags, caption) in enumerate(
                    zip(
                        valid_pairs,
                        images,
                        predicted_tags,
                        structured_captions,
                        strict=True,
                    )
                ):
                    payload = image_metadata(path, image)
                    payload.update(
                        {
                            "root": str(resolved_root),
                            "root_key": root_key,
                            "tags": tags.tags,
                            "tag_scores": tags.scores,
                            "caption": caption,
                            "descriptor_text": descriptor_texts[index],
                        }
                    )
                    points.append(
                        models.PointStruct(
                            id=point_id(path),
                            vector={
                                "visual": visual_vectors[index].tolist(),
                                "descriptor": descriptor_vectors[index].tolist(),
                            },
                            payload=payload,
                        )
                    )
                self.store.upsert(points)
                summary.indexed += len(points)
                completed += len(valid_pairs)
                if progress:
                    progress(completed, len(changed), str(resolved_root))

            if remove_missing:
                stale_ids = [
                    item_id
                    for key, (item_id, _) in existing.items()
                    if key not in present_keys
                ]
                self.store.delete_ids(stale_ids)
                summary.deleted += len(stale_ids)
        return summary
