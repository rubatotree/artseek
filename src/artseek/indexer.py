from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from qdrant_client import models

from .config import AppConfig
from .image_io import (
    IMAGE_EXTENSIONS,
    canonical_path,
    file_digest,
    image_metadata,
    load_rgb,
    point_id,
)
from .incremental_state import (
    DirectoryState,
    IncrementalIndexState,
    IndexedImageState,
    RootIndexState,
)
from .library import FolderCatalog, build_catalog, rebuild_catalog, save_catalog
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


def _is_within(path_key: str, ancestor_key: str) -> bool:
    try:
        Path(path_key).relative_to(Path(ancestor_key))
    except ValueError:
        return False
    return True


class ImageIndexer:
    def __init__(
        self,
        config: AppConfig,
        *,
        embedder: SiglipEmbedder | None = None,
        store: VectorStore | None = None,
        tagger: WDTagger | None = None,
        llm: OllamaClient | None = None,
    ):
        self.config = config
        self._owns_embedder = embedder is None
        self.embedder = embedder or SiglipEmbedder(config.embedding)
        if store is None:
            self.store = VectorStore(config)
            self.store.ensure_collection(self.embedder.dimension)
            self._owns_store = True
        else:
            self.store = store
            self._owns_store = False
        if tagger is not None:
            self.tagger = tagger
            self._owns_tagger = False
        elif config.tagger.enabled:
            self.tagger = WDTagger(config.tagger, config.embedding)
            self._owns_tagger = True
        else:
            self.tagger = None
            self._owns_tagger = False
        self.llm = llm or OllamaClient(config.llm)
        self.state = IncrementalIndexState(config.index_state_path)
        # Set only after a vector mutation. Callers can adopt it directly and
        # avoid reopening Qdrant to rebuild the same catalog a second time.
        self.folder_catalog: FolderCatalog | None = None

    def close(self) -> None:
        if self._owns_embedder:
            closer = getattr(self.embedder, "close", None)
            if callable(closer):
                closer()
            self.embedder = None  # type: ignore[assignment]
        if self._owns_tagger:
            closer = getattr(self.tagger, "close", None)
            if callable(closer):
                closer()
            self.tagger = None
        if self._owns_store:
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
        """Update roots using directory snapshots, with periodic full verification.

        A directory mtime does not change for every in-place file rewrite on
        every filesystem. The fast path is therefore an I/O optimization, not
        a correctness proof. Periodic verification enumerates every file and
        catches normal rewrites whose file mtime changes, but a same-size edit
        with a manually restored mtime remains indistinguishable without
        hashing every file on every update.
        """
        summary = IndexSummary()
        self.folder_catalog = None
        resolved_roots = self._resolve_roots(roots)
        self._assert_non_overlapping_roots(resolved_roots)
        use_captions = self.config.llm.caption_on_index if captions is None else captions
        batch_size = min(
            self.config.embedding.batch_size,
            self.config.tagger.batch_size if self.tagger else self.config.embedding.batch_size,
        )
        changed_vectors = False

        for resolved_root in resolved_roots:
            root_key = canonical_path(resolved_root)
            dirty_recovery = self.state.is_dirty(root_key)
            snapshot = self.state.load_root(root_key)
            failed_before = len(summary.failed)
            full_verification = self._requires_full_verification(
                snapshot,
                recursive=recursive,
                force=force or dirty_recovery,
                remove_missing=remove_missing,
            )
            try:
                if full_verification:
                    mutated = self._index_full_root(
                        resolved_root,
                        root_key,
                        recursive=recursive,
                        # A dirty marker means the prior process may have died
                        # after only some force/upsert batches reached Qdrant.
                        # Re-embed every file before clearing it.
                        force=force or dirty_recovery,
                        remove_missing=remove_missing,
                        use_captions=use_captions,
                        batch_size=batch_size,
                        summary=summary,
                        progress=progress,
                        failed_before=failed_before,
                    )
                else:
                    assert snapshot is not None
                    mutated = self._index_delta_root(
                        resolved_root,
                        snapshot,
                        use_captions=use_captions,
                        batch_size=batch_size,
                        summary=summary,
                        progress=progress,
                        failed_before=failed_before,
                    )
            except Exception:
                # Qdrant may already have accepted part of the work. Discarding
                # this snapshot makes the next run reconcile it from Qdrant,
                # rather than making a stale cache authoritative.
                self.state.invalidate_root(root_key)
                raise
            changed_vectors = changed_vectors or mutated

        if changed_vectors or not self.config.folder_catalog_path.exists():
            self.folder_catalog = self._catalog_from_state() or rebuild_catalog(
                self.store, self.config
            )
        return summary

    def _index_full_root(
        self,
        root: Path,
        root_key: str,
        *,
        recursive: bool,
        force: bool,
        remove_missing: bool,
        use_captions: bool,
        batch_size: int,
        summary: IndexSummary,
        progress: ProgressCallback | None,
        failed_before: int,
    ) -> bool:
        """Enumerate a root and scroll Qdrant to initialize/repair its cache."""
        scanned, directories = self._scan_subtree(root, recursive=recursive)
        existing = self._states_from_qdrant(self.store.scroll_root(root_key), root_key)
        changed = self._changed_paths(scanned, existing, force=force)
        summary.discovered += len(scanned)
        summary.unchanged += len(scanned) - len(changed)

        stale_ids: list[str] = []
        if remove_missing:
            stale_ids.extend(item.point_id for key, item in existing.items() if key not in scanned)
            stale_ids.extend(
                item.point_id
                for key, item in existing.items()
                if key in changed and item.point_id != point_id(changed[key])
            )
        stale_ids = list(dict.fromkeys(stale_ids))
        if changed or stale_ids:
            self.state.mark_dirty(root_key)
        updated = self._upsert_changed(
            root,
            root_key,
            changed,
            use_captions=use_captions,
            batch_size=batch_size,
            summary=summary,
            progress=progress,
        )
        if stale_ids:
            self.store.delete_ids(stale_ids)
            summary.deleted += len(stale_ids)

        if len(summary.failed) > failed_before:
            self.state.invalidate_root(root_key)
        elif remove_missing:
            final_images = {
                key: updated[key] if key in updated else existing[key]
                for key in scanned
                if key in updated or key in existing
            }
            self.state.replace_root(
                RootIndexState(
                    root_key=root_key,
                    root_path=str(root),
                    recursive=recursive,
                    last_full_verification_ns=time.time_ns(),
                    images=final_images,
                    directories=directories,
                )
            )
        else:
            # A no-delete update cannot form an exact snapshot: retaining a
            # removed file would make a later delta update silently miss it.
            self.state.invalidate_root(root_key)
        return bool(updated or stale_ids)

    def _index_delta_root(
        self,
        root: Path,
        snapshot: RootIndexState,
        *,
        use_captions: bool,
        batch_size: int,
        summary: IndexSummary,
        progress: ProgressCallback | None,
        failed_before: int,
    ) -> bool:
        if not root.exists() or not root.is_dir():
            # Preserve the historical FileNotFound contract. In particular, do
            # not retain a zero-directory root snapshot that would later block
            # indexing an enclosing root as an apparent overlap.
            self.state.invalidate_root(snapshot.root_key)
            raise FileNotFoundError(f"图片目录不存在：{root}")
        changed_directories = self._changed_directories(snapshot)
        if not changed_directories:
            summary.discovered += len(snapshot.images)
            summary.unchanged += len(snapshot.images)
            return False

        current_images = dict(snapshot.images)
        current_directories = dict(snapshot.directories)
        rescanned: dict[str, Path] = {}
        known_directories = set(snapshot.directories)
        for directory_key in changed_directories:
            directory = Path(directory_key)
            if not directory.exists() or not directory.is_dir():
                self._drop_subtree(current_images, current_directories, directory_key)
                continue

            # A changed directory is enumerated only one level deep. Existing
            # children were already stat'ed above, so recursing into them would
            # turn a root-level file change into a full tree scan. New children
            # do need a complete walk because no snapshot exists for them yet.
            scanned = self._scan_changed_directory(directory, known_directories)
            if scanned is None:
                self._drop_subtree(current_images, current_directories, directory_key)
                continue
            paths, directories = scanned
            self._drop_direct_images(current_images, directory_key)
            rescanned.update(paths)
            current_directories.update(directories)

        changed = self._changed_paths(rescanned, snapshot.images, force=False)
        summary.discovered += len(current_images) + len(rescanned)
        summary.unchanged += len(current_images) + len(rescanned) - len(changed)
        expected_current_keys = set(current_images) | set(rescanned)
        may_delete = any(key not in expected_current_keys for key in snapshot.images)
        may_replace = any(
            item.point_id != point_id(changed[key])
            for key, item in snapshot.images.items()
            if key in changed
        )
        if changed or may_delete or may_replace:
            self.state.mark_dirty(snapshot.root_key)
        updated = self._upsert_changed(
            root,
            snapshot.root_key,
            changed,
            use_captions=use_captions,
            batch_size=batch_size,
            summary=summary,
            progress=progress,
        )

        for key in rescanned:
            previous = snapshot.images.get(key)
            if key in updated:
                current_images[key] = updated[key]
            elif previous is not None:
                current_images[key] = previous
            else:
                # A load/encode failure leaves this new file unindexed. The
                # invalidation below forces a safe full retry next time.
                current_images.pop(key, None)

        stale_ids = [
            item.point_id for key, item in snapshot.images.items() if key not in current_images
        ]
        stale_ids.extend(
            item.point_id
            for key, item in snapshot.images.items()
            if key in updated and item.point_id != updated[key].point_id
        )
        stale_ids = list(dict.fromkeys(stale_ids))
        if stale_ids:
            self.store.delete_ids(stale_ids)
            summary.deleted += len(stale_ids)

        if len(summary.failed) > failed_before:
            self.state.invalidate_root(snapshot.root_key)
        else:
            self.state.replace_root(
                RootIndexState(
                    root_key=snapshot.root_key,
                    root_path=str(root),
                    recursive=True,
                    last_full_verification_ns=snapshot.last_full_verification_ns,
                    images=current_images,
                    directories=current_directories,
                )
            )
        return bool(updated or stale_ids)

    def _catalog_from_state(self) -> FolderCatalog | None:
        """Avoid a post-update Qdrant scroll when committed sidecar data is complete."""
        rows = self.state.catalog_payloads()
        if rows is None or not hasattr(self.store, "count"):
            return None
        if int(self.store.count()) != len(rows):
            return None
        catalog = build_catalog(rows)
        save_catalog(self.config.folder_catalog_path, catalog, count=len(rows))
        return catalog

    @staticmethod
    def _drop_subtree(
        images: dict[str, IndexedImageState],
        directories: dict[str, DirectoryState],
        directory_key: str,
    ) -> None:
        for key in [
            key
            for key, item in images.items()
            if _is_within(key, directory_key) or _is_within(item.folder_key, directory_key)
        ]:
            del images[key]
        for key in [key for key in directories if _is_within(key, directory_key)]:
            del directories[key]

    @staticmethod
    def _drop_direct_images(images: dict[str, IndexedImageState], directory_key: str) -> None:
        for key in [
            key for key, item in images.items() if item.folder_key == directory_key
        ]:
            del images[key]

    def _upsert_changed(
        self,
        root: Path,
        root_key: str,
        changed: dict[str, Path],
        *,
        use_captions: bool,
        batch_size: int,
        summary: IndexSummary,
        progress: ProgressCallback | None,
    ) -> dict[str, IndexedImageState]:
        updated: dict[str, IndexedImageState] = {}
        completed = 0
        pairs = [(root, path) for path in changed.values()]
        for raw_batch in chunks(pairs, batch_size):
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
            for tags, caption in zip(predicted_tags, structured_captions, strict=True):
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
            states: list[IndexedImageState] = []
            for index, ((_, path), image, tags, caption) in enumerate(
                zip(valid_pairs, images, predicted_tags, structured_captions, strict=True)
            ):
                payload = image_metadata(path, image)
                payload.update(
                    {
                        "root": str(root),
                        "root_key": root_key,
                        "folder_key": canonical_path(path.parent),
                        "content_hash": file_digest(path),
                        "tags": tags.tags,
                        "tag_scores": tags.scores,
                        "caption": caption,
                        "descriptor_text": descriptor_texts[index],
                    }
                )
                item_id = point_id(path)
                points.append(
                    models.PointStruct(
                        id=item_id,
                        vector={
                            "visual": visual_vectors[index].tolist(),
                            "descriptor": descriptor_vectors[index].tolist(),
                        },
                        payload=payload,
                    )
                )
                states.append(
                    IndexedImageState(
                        path_key=str(payload["path_key"]),
                        root_key=root_key,
                        point_id=item_id,
                        mtime_ns=int(payload["mtime_ns"]),
                        size_bytes=int(payload["size_bytes"]),
                        folder_key=str(payload["folder_key"]),
                        name=str(payload["name"]),
                    )
                )
            self.store.upsert(points)
            updated.update({item.path_key: item for item in states})
            summary.indexed += len(points)
            completed += len(valid_pairs)
            if progress:
                progress(completed, len(changed), str(root))
        return updated

    def _scan_subtree(
        self, root: Path, *, recursive: bool
    ) -> tuple[dict[str, Path], dict[str, DirectoryState]]:
        """Scan one tree once, retaining directory mtimes for the next update."""
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"图片目录不存在：{root}")
        images: dict[str, Path] = {}
        directories: dict[str, DirectoryState] = {}
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                stat = directory.stat()
            except FileNotFoundError:
                continue
            directory_key = canonical_path(directory)
            directories[directory_key] = DirectoryState(directory_key, stat.st_mtime_ns)
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        path = Path(entry.path)
                        try:
                            if entry.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                                images[canonical_path(path)] = path
                            elif recursive and entry.is_dir(follow_symlinks=False):
                                pending.append(path)
                        except OSError:
                            continue
            except FileNotFoundError:
                directories.pop(directory_key, None)
            except OSError:
                # A partial traversal cannot safely be used to infer deletions.
                raise
        return images, directories

    def _scan_changed_directory(
        self, directory: Path, known_directories: set[str]
    ) -> tuple[dict[str, Path], dict[str, DirectoryState]] | None:
        """Read direct entries, descending only into directories new to the snapshot."""
        try:
            stat = directory.stat()
            directory_key = canonical_path(directory)
            images: dict[str, Path] = {}
            directories = {
                directory_key: DirectoryState(directory_key, stat.st_mtime_ns)
            }
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    try:
                        if entry.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                            images[canonical_path(path)] = path
                            continue
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    child_key = canonical_path(path)
                    if child_key not in known_directories:
                        child_images, child_directories = self._scan_subtree(path, recursive=True)
                        images.update(child_images)
                        directories.update(child_directories)
            return images, directories
        except FileNotFoundError:
            return None

    def _changed_directories(self, snapshot: RootIndexState) -> list[str]:
        changed: list[str] = []
        for path_key, previous in snapshot.directories.items():
            path = Path(path_key)
            try:
                stat = path.stat()
            except FileNotFoundError:
                changed.append(path_key)
                continue
            if not path.is_dir() or stat.st_mtime_ns != previous.mtime_ns:
                changed.append(path_key)
        return changed

    @staticmethod
    def _changed_paths(
        scanned: dict[str, Path],
        existing: dict[str, IndexedImageState],
        *,
        force: bool,
    ) -> dict[str, Path]:
        changed: dict[str, Path] = {}
        for key, path in scanned.items():
            stat = path.stat()
            previous = existing.get(key)
            if (
                force
                or previous is None
                or previous.mtime_ns != stat.st_mtime_ns
                or previous.size_bytes != stat.st_size
                or previous.name != path.name
            ):
                changed[key] = path
        return changed

    @staticmethod
    def _states_from_qdrant(
        records: dict[str, tuple[str, dict[str, Any]]], root_key: str
    ) -> dict[str, IndexedImageState]:
        states: dict[str, IndexedImageState] = {}
        for key, (item_id, payload) in records.items():
            try:
                states[key] = IndexedImageState(
                    path_key=key,
                    root_key=root_key,
                    point_id=item_id,
                    mtime_ns=int(payload.get("mtime_ns", -1)),
                    size_bytes=int(payload.get("size_bytes", -1)),
                    folder_key=str(payload.get("folder_key") or Path(key).parent),
                    name=str(payload.get("name") or Path(key).name),
                )
            except (TypeError, ValueError):
                # Treat a malformed legacy payload as changed, preserving it
                # until the complete scan has successfully replaced it.
                states[key] = IndexedImageState(
                    path_key=key,
                    root_key=root_key,
                    point_id=item_id,
                    mtime_ns=-1,
                    size_bytes=-1,
                    folder_key=canonical_path(Path(key).parent),
                    name=Path(key).name,
                )
        return states

    def _requires_full_verification(
        self,
        snapshot: RootIndexState | None,
        *,
        recursive: bool,
        force: bool,
        remove_missing: bool,
    ) -> bool:
        if force or not remove_missing or snapshot is None or snapshot.recursive != recursive:
            return True
        # Non-recursive scans intentionally retain their prior behavior. They
        # cannot safely share a root snapshot with later recursive updates.
        if not recursive:
            return True
        try:
            interval = max(0, int(self.config.indexing.full_verification_interval_seconds))
        except (TypeError, ValueError):
            interval = 0
        return interval == 0 or (
            time.time_ns() - snapshot.last_full_verification_ns >= interval * 1_000_000_000
        )

    def _resolve_roots(self, roots: Sequence[Path]) -> list[Path]:
        resolved: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            path = root.expanduser().resolve()
            key = canonical_path(path)
            if key not in seen:
                seen.add(key)
                resolved.append(path)
        return resolved

    def _assert_non_overlapping_roots(self, roots: Sequence[Path]) -> None:
        root_keys = [canonical_path(root) for root in roots]
        known_keys = self.state.root_keys()
        candidates = [*root_keys, *(key for key in known_keys if key not in root_keys)]
        for index, key in enumerate(candidates):
            for other in candidates[index + 1 :]:
                if _is_within(key, other) or _is_within(other, key):
                    raise ValueError(
                        "索引根目录不能相互嵌套："
                        f"{key} 与 {other}。请只保留父目录，避免同一图片的状态互相覆盖。"
                    )
