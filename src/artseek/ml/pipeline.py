"""Producer-consumer GPU-saturated indexing pipeline for ArtSeek v2."""

from __future__ import annotations

import hashlib
import io
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import AppConfig
from ..db.repos.images import ImageRepo, ImageUpsertRecord
from ..image_io import canonical_path, composite_on_white, perceptual_hash, point_id
from ..vectors.index import VectorIndex
from .embedder import SiglipEmbedder
from .preprocess import PRE_SHRINK_SHORT_EDGE, draft_target, shrink_to_short_edge
from .tag_postprocess import (
    CAT_RATING,
    build_high_density_descriptor,
)

ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class PreparedImageItem:
    root: Path
    root_key: str
    root_path: str
    folder_key: str
    folder_path: str
    folder_mtime_ns: int
    path: Path
    path_str: str
    path_key: str
    point_id: str
    name: str
    ext: str
    mtime_ns: int
    size_bytes: int
    orig_width: int
    orig_height: int
    content_hash: str
    phash: str
    rgb_arr: np.ndarray


@dataclass(slots=True)
class FailedImageItem:
    root: Path
    path: Path
    reason: str


@dataclass(slots=True)
class PipelineBatchStats:
    indexed: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    decode_ms: float = 0.0
    gpu_ms: float = 0.0
    persist_ms: float = 0.0
    stopped: bool = False


def decode_and_prepare_image(
    root: Path,
    path: Path,
    *,
    root_key: str | None = None,
    root_path: str | None = None,
    folder_mtime_ns: int = 0,
    short_edge_limit: int = PRE_SHRINK_SHORT_EDGE,
    existing_mtime_ns: int | None = None,
    existing_size_bytes: int | None = None,
) -> PreparedImageItem | FailedImageItem:
    """Read image bytes once, compute SHA-256 + true dimensions + RGB array + dHash."""
    try:
        if existing_mtime_ns is not None and existing_size_bytes is not None:
            mtime_ns = existing_mtime_ns
            size_bytes = existing_size_bytes
        else:
            st = path.stat()
            mtime_ns = st.st_mtime_ns
            size_bytes = st.st_size

        raw = path.read_bytes()
        content_hash = hashlib.sha256(raw).hexdigest()

        with Image.open(io.BytesIO(raw)) as img:
            img.seek(0)
            raw_w, raw_h = img.size
            orientation = int(img.getexif().get(0x0112, 1) or 1)
            orig_w, orig_h = (
                (raw_h, raw_w) if orientation in {5, 6, 7, 8} else (raw_w, raw_h)
            )
            target = (
                draft_target(img, short_edge_limit)
                if (img.format or "").upper() in {"JPEG", "MPO", "WEBP"}
                else None
            )
            if target is not None:
                img.draft("RGB", target)
            transposed = ImageOps.exif_transpose(img)
            rgb = composite_on_white(transposed)

        rgb = shrink_to_short_edge(rgb, short_edge_limit)
        phash = perceptual_hash(rgb)
        rgb_arr = np.array(rgb, dtype=np.uint8, copy=True)
        resolved_root = root_path or str(root.resolve())
        resolved_root_key = root_key or canonical_path(root)
        parent = path.parent
        folder_p = str(parent.resolve())
        folder_k = canonical_path(parent)
        p_str = str(path.resolve())
        p_key = canonical_path(path)
        pid = point_id(path)
        return PreparedImageItem(
            root=root,
            root_key=resolved_root_key,
            root_path=resolved_root,
            folder_key=folder_k,
            folder_path=folder_p,
            folder_mtime_ns=folder_mtime_ns,
            path=path,
            path_str=p_str,
            path_key=p_key,
            point_id=pid,
            name=path.name,
            ext=path.suffix.lower(),
            mtime_ns=mtime_ns,
            size_bytes=size_bytes,
            orig_width=orig_w,
            orig_height=orig_h,
            content_hash=content_hash,
            phash=phash,
            rgb_arr=rgb_arr,
        )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return FailedImageItem(root=root, path=path, reason=str(exc))


class IndexingPipeline:
    """8-thread CPU decode producer + single GPU batch consumer + atomic SQLite/Vector writer."""

    def __init__(
        self,
        config: AppConfig,
        images_repo: ImageRepo,
        vectors: VectorIndex,
        *,
        embedder: SiglipEmbedder,
        tagger: Any = None,
        cpu_workers: int = 4,
        short_edge_limit: int = PRE_SHRINK_SHORT_EDGE,
        flush_every_batches: int = 8,
    ) -> None:
        self.config = config
        self.images_repo = images_repo
        self.vectors = vectors
        self.embedder = embedder
        self.tagger = tagger
        self.cpu_workers = max(1, int(cpu_workers))
        self.short_edge_limit = max(256, int(short_edge_limit))
        self.flush_every_batches = max(1, int(flush_every_batches))

    def process_items(
        self,
        items: Sequence[tuple[Path, Path, int, int, int]],
        *,
        batch_size: int | None = None,
        progress: ProgressCallback | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> PipelineBatchStats:
        """Process `(root, path, mtime_ns, size_bytes, folder_mtime_ns)` tuples."""
        stats = PipelineBatchStats()
        total = len(items)
        if total == 0:
            return stats

        eff_batch = batch_size or min(
            32,
            max(1, int(self.config.embedding.batch_size)),
            max(1, int(self.config.tagger.batch_size))
            if self.tagger is not None
            else max(1, int(self.config.embedding.batch_size)),
        )
        q_max = max(eff_batch * 2, 32)
        work_q: queue.Queue[PreparedImageItem | FailedImageItem | None] = queue.Queue(
            maxsize=q_max
        )
        producer_stop = threading.Event()

        root_cache: dict[Path, tuple[str, str]] = {}
        for root_dir, _, _, _, _ in items:
            if root_dir not in root_cache:
                root_cache[root_dir] = (canonical_path(root_dir), str(root_dir.resolve()))

        def _decode_one(
            entry: tuple[Path, Path, int, int, int],
        ) -> PreparedImageItem | FailedImageItem | None:
            if producer_stop.is_set() or (stop_event is not None and stop_event.is_set()):
                return None
            r_dir, img_path, mtime_ns, size_bytes, f_mtime_ns = entry
            r_key, r_path = root_cache[r_dir]
            return decode_and_prepare_image(
                r_dir,
                img_path,
                root_key=r_key,
                root_path=r_path,
                folder_mtime_ns=f_mtime_ns,
                short_edge_limit=self.short_edge_limit,
                existing_mtime_ns=mtime_ns if mtime_ns > 0 else None,
                existing_size_bytes=size_bytes if size_bytes >= 0 else None,
            )

        def _stopping() -> bool:
            return producer_stop.is_set() or (
                stop_event is not None and stop_event.is_set()
            )

        def _put(value: PreparedImageItem | FailedImageItem | None) -> bool:
            while True:
                try:
                    work_q.put(value, timeout=0.2)
                    return True
                except queue.Full:
                    if _stopping():
                        return False

        def _producer() -> None:
            try:
                with ThreadPoolExecutor(
                    max_workers=self.cpu_workers, thread_name_prefix="artseek-decode"
                ) as pool:
                    for res in pool.map(_decode_one, items):
                        if _stopping():
                            break
                        if res is not None and not _put(res):
                            break
            finally:
                _put(None)

        prod_thread = threading.Thread(target=_producer, daemon=True, name="artseek-producer")
        prod_thread.start()

        batch: list[PreparedImageItem] = []
        completed = 0
        batches_since_flush = 0
        dirty_vectors = False

        try:
            while True:
                if pause_event is not None:
                    while pause_event.is_set():
                        if stop_event is not None and stop_event.is_set():
                            break
                        time.sleep(0.05)
                if stop_event is not None and stop_event.is_set():
                    stats.stopped = True
                    producer_stop.set()
                    break

                t_wait0 = time.perf_counter()
                item = work_q.get()
                stats.decode_ms += (time.perf_counter() - t_wait0) * 1000.0

                if item is None:
                    if batch:
                        self._flush_gpu_batch(batch, stats)
                        dirty_vectors = True
                        completed += len(batch)
                        if progress is not None:
                            progress(completed, total, str(batch[-1].root))
                        batch.clear()
                    break

                if isinstance(item, FailedImageItem):
                    stats.failed.append((str(item.path), item.reason))
                    completed += 1
                    if progress is not None:
                        progress(completed, total, str(item.root))
                    continue

                batch.append(item)
                if len(batch) >= eff_batch:
                    self._flush_gpu_batch(batch, stats)
                    dirty_vectors = True
                    batches_since_flush += 1
                    completed += len(batch)
                    last_root = str(batch[-1].root)
                    batch.clear()
                    if batches_since_flush >= self.flush_every_batches:
                        t_p0 = time.perf_counter()
                        self.vectors.flush()
                        stats.persist_ms += (time.perf_counter() - t_p0) * 1000.0
                        dirty_vectors = False
                        batches_since_flush = 0
                    if progress is not None:
                        progress(completed, total, last_root)
        finally:
            producer_stop.set()
            if dirty_vectors:
                t_p0 = time.perf_counter()
                self.vectors.flush()
                stats.persist_ms += (time.perf_counter() - t_p0) * 1000.0

        return stats

    def _flush_gpu_batch(
        self,
        batch: Sequence[PreparedImageItem],
        stats: PipelineBatchStats,
    ) -> None:
        """Commit visuals first, then tags.

        Rows become readable as soon as the visual vectors are written, so the
        gallery can show new images while the (much slower) tagger is still
        working on them. Tags and the descriptor vector are patched in place by
        the second stage of the same batch.
        """
        t_gpu0 = time.perf_counter()
        rgb_arrays = [item.rgb_arr for item in batch]

        # ---- Stage A: file rows + visual vectors, visible immediately.
        visual_vectors = np.asarray(
            self.embedder.encode_images(rgb_arrays), dtype=np.float32
        )
        time.sleep(0)
        records = [
            ImageUpsertRecord(
                point_id=item.point_id,
                root_key=item.root_key,
                root_path=item.root_path,
                folder_key=item.folder_key,
                folder_path=item.folder_path,
                folder_mtime_ns=item.folder_mtime_ns,
                path=item.path_str,
                path_key=item.path_key,
                name=item.name,
                ext=item.ext,
                mtime_ns=item.mtime_ns,
                size_bytes=item.size_bytes,
                width=item.orig_width,
                height=item.orig_height,
                content_hash=item.content_hash,
                phash=item.phash,
                descriptor_text="",
                caption=None,
                tags=[],
                tag_scores={},
                tag_categories={},
            )
            for item in batch
        ]
        t_db0 = time.perf_counter()
        row_ids = self.images_repo.upsert_images_batch(records, keep_enrichment=True)
        # Placeholder descriptor: identical to the visual vector, so visual search
        # works before any tag-derived text embedding exists.
        self.vectors.set_rows(row_ids, visual_vectors, np.array(visual_vectors, copy=True))
        stats.persist_ms += (time.perf_counter() - t_db0) * 1000.0
        stats.indexed += len(records)

        if self.tagger is None:
            stats.gpu_ms += (time.perf_counter() - t_gpu0) * 1000.0
            return

        # ---- Stage B: tags, descriptor text, descriptor vectors.
        predicted_tags = self.tagger.predict(rgb_arrays)
        time.sleep(0)

        descriptor_texts: list[str] = []
        for pred in predicted_tags:
            tags = list(pred.tags)
            cats = dict(getattr(pred, "categories", {}) or {})
            if getattr(pred, "rating", None):
                r_tag = f"rating:{pred.rating}"
                if r_tag not in cats:
                    cats[r_tag] = CAT_RATING
                    tags.append(r_tag)
            descriptor_texts.append(
                build_high_density_descriptor(
                    tags,
                    cats,
                    drop_overlap=bool(self.config.tagger.drop_overlap),
                )
            )

        tag_updates: list[
            tuple[int, Sequence[str], dict[str, float], dict[str, int], str]
        ] = []
        for idx, pred in enumerate(predicted_tags):
            tags = list(pred.tags)
            scores = dict(pred.scores)
            cats = dict(getattr(pred, "categories", {}) or {})
            if getattr(pred, "rating", None):
                r_tag = f"rating:{pred.rating}"
                if r_tag not in scores:
                    tags.append(r_tag)
                    scores[r_tag] = 1.0
                    cats[r_tag] = CAT_RATING
            tag_updates.append((row_ids[idx], tags, scores, cats, descriptor_texts[idx]))

        nonempty_indices = [
            idx for idx, text in enumerate(descriptor_texts) if text.strip()
        ]
        descriptor_vectors = np.array(visual_vectors, copy=True)
        if nonempty_indices:
            encoded = self.embedder.encode_texts(
                [descriptor_texts[idx] for idx in nonempty_indices]
            )
            time.sleep(0)
            for target_idx, vec in zip(nonempty_indices, encoded, strict=True):
                descriptor_vectors[target_idx] = vec
        stats.gpu_ms += (time.perf_counter() - t_gpu0) * 1000.0

        t_db1 = time.perf_counter()
        self.images_repo.update_tags_and_descriptor_batch(tag_updates)
        if nonempty_indices:
            self.vectors.set_descriptor_rows(
                [row_ids[idx] for idx in nonempty_indices],
                descriptor_vectors[nonempty_indices],
            )
        tag_versions = dict(getattr(self.tagger, "model_versions", {}) or {})
        if tag_versions:
            self.images_repo.mark_tag_models_done(
                [
                    (item.point_id, row_ids[idx], item.content_hash, True)
                    for idx, item in enumerate(batch)
                ],
                tag_versions,
            )
        stats.persist_ms += (time.perf_counter() - t_db1) * 1000.0
