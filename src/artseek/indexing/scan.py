"""SQLite-backed incremental directory scanner and indexer for ArtSeek v2."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..db.connection import Database
from ..db.repos.images import ImageRepo
from ..image_io import IMAGE_EXTENSIONS, canonical_path, point_id
from ..ml.embedder import SiglipEmbedder
from ..ml.pipeline import IndexingPipeline
from ..ml.tagger import WDTagger
from ..vectors.index import VectorIndex

ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class IndexSummary:
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    llm_warning: str | None = None
    stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovered": self.discovered,
            "indexed": self.indexed,
            "unchanged": self.unchanged,
            "deleted": self.deleted,
            "failed": len(self.failed),
            "failed_items": [
                {"path": path, "reason": reason} for path, reason in self.failed[:50]
            ],
            "llm_warning": self.llm_warning,
            "stopped": self.stopped,
        }


@dataclass(slots=True)
class _DbRootState:
    root_id: int
    root_key: str
    root_path: str
    recursive: bool
    last_full_verification_ns: int
    dirty: bool
    folders: dict[str, tuple[int, str, int]]
    images: dict[str, tuple[str, str, int, int, str]]


def _is_within(path_key: str, ancestor_key: str) -> bool:
    norm_p = os.path.normcase(os.path.normpath(path_key)).rstrip("\\/")
    norm_a = os.path.normcase(os.path.normpath(ancestor_key)).rstrip("\\/")
    if norm_p == norm_a:
        return True
    prefix = norm_a + os.sep
    return norm_p.startswith(prefix)


class IncrementalScanner:
    """Directory-mtime delta and periodic full verification scans on SQLite + VectorIndex."""

    def __init__(
        self,
        config: AppConfig,
        db: Database,
        images_repo: ImageRepo,
        vectors: VectorIndex,
        *,
        embedder_factory: Callable[[], SiglipEmbedder] | None = None,
        tagger_factory: Callable[[], WDTagger | None] | None = None,
        cpu_workers: int = 8,
    ) -> None:
        self.config = config
        self.db = db
        self.images_repo = images_repo
        self.vectors = vectors
        self._embedder_factory = embedder_factory or (
            lambda: SiglipEmbedder(self.config.embedding)
        )
        self._tagger_factory = tagger_factory or (
            lambda: WDTagger(self.config.tagger, self.config.embedding)
            if self.config.tagger.enabled
            else None
        )
        self.cpu_workers = max(1, int(cpu_workers))

    def index_roots(
        self,
        roots: Sequence[Path],
        *,
        recursive: bool = True,
        force: bool = False,
        remove_missing: bool = True,
        embedder: SiglipEmbedder | None = None,
        tagger: WDTagger | None = None,
        progress: ProgressCallback | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> IndexSummary:
        summary = IndexSummary()
        resolved_roots = self._resolve_roots(roots)
        self._assert_non_overlapping_roots(resolved_roots)

        # Phase 1: Discover changed files and stale point_ids before loading GPU models.
        root_plans: list[
            tuple[
                Path,
                str,
                bool,
                list[tuple[Path, Path, int, int, int]],
                list[str],
                dict[str, tuple[str, int]],
                set[str],
            ]
        ] = []
        total_todo = 0
        for resolved_root in resolved_roots:
            if stop_event is not None and stop_event.is_set():
                summary.stopped = True
                return summary
            root_key = canonical_path(resolved_root)
            state = self._load_root_state(root_key)
            full_verify = self._requires_full_verification(
                state,
                recursive=recursive,
                force=force or (state.dirty if state is not None else True),
                remove_missing=remove_missing,
            )
            if full_verify:
                todo, stale_pids, dir_mtimes, removed_dirs, disc_cnt, unch_cnt = (
                    self._plan_full_root(
                        resolved_root,
                        root_key,
                        state,
                        recursive=recursive,
                        force=force or (state.dirty if state is not None else False),
                        remove_missing=remove_missing,
                    )
                )
            else:
                assert state is not None
                todo, stale_pids, dir_mtimes, removed_dirs, disc_cnt, unch_cnt = (
                    self._plan_delta_root(resolved_root, state)
                )
            summary.discovered += disc_cnt
            summary.unchanged += unch_cnt
            total_todo += len(todo)
            root_plans.append(
                (
                    resolved_root,
                    root_key,
                    full_verify,
                    todo,
                    stale_pids,
                    dir_mtimes,
                    removed_dirs,
                )
            )

        # Fast exit when nothing changed (0 new/modified images and 0 deletions).
        any_mutations = any(
            todo or stale_pids for _, _, _, todo, stale_pids, _, _ in root_plans
        )
        if not any_mutations:
            for resolved_root, root_key, full_verify, _, _, dir_mtimes, _ in root_plans:
                if full_verify and remove_missing:
                    self._commit_root_directory_state(
                        resolved_root,
                        root_key,
                        recursive=recursive,
                        dir_mtimes=dir_mtimes,
                        removed_dirs=set(),
                        mark_verified=True,
                    )
            return summary

        owns_embedder = embedder is None
        owns_tagger = tagger is None
        active_embedder = embedder
        active_tagger = tagger
        try:
            if total_todo > 0:
                if active_embedder is None:
                    active_embedder = self._embedder_factory()
                if active_tagger is None and self.config.tagger.enabled:
                    active_tagger = self._tagger_factory()
                dim = int(getattr(active_embedder, "dimension", 0) or 0)
                if dim > 0 and self.images_repo.known_dimension() is None:
                    self.images_repo.meta_set("dimension", str(dim))
                self.images_repo.meta_set("embedding_model", self.config.embedding.model)
                if self.config.tagger.enabled:
                    self.images_repo.meta_set("tagger_model", self.config.tagger.model)

            for (
                resolved_root,
                root_key,
                full_verify,
                todo,
                stale_pids,
                dir_mtimes,
                removed_dirs,
            ) in root_plans:
                if stop_event is not None and stop_event.is_set():
                    summary.stopped = True
                    break

                self._set_root_dirty(
                    root_key, str(resolved_root), recursive=recursive, dirty=True
                )
                failed_before = len(summary.failed)
                if todo:
                    assert active_embedder is not None
                    pipeline = IndexingPipeline(
                        self.config,
                        self.images_repo,
                        self.vectors,
                        embedder=active_embedder,
                        tagger=active_tagger,
                        cpu_workers=self.cpu_workers,
                    )
                    b_stats = pipeline.process_items(
                        todo,
                        progress=progress,
                        stop_event=stop_event,
                        pause_event=pause_event,
                    )
                    summary.indexed += b_stats.indexed
                    summary.failed.extend(b_stats.failed)
                    if b_stats.stopped:
                        summary.stopped = True

                if stale_pids and not summary.stopped:
                    del_rows = self.images_repo.soft_delete_by_point_ids(stale_pids)
                    if del_rows:
                        self.vectors.zero_rows(del_rows)
                        self.vectors.flush()
                    summary.deleted += len(stale_pids)

                if summary.stopped or len(summary.failed) > failed_before or not remove_missing:
                    self._set_root_dirty(
                        root_key, str(resolved_root), recursive=recursive, dirty=True
                    )
                else:
                    self._commit_root_directory_state(
                        resolved_root,
                        root_key,
                        recursive=recursive,
                        dir_mtimes=dir_mtimes,
                        removed_dirs=removed_dirs,
                        mark_verified=full_verify,
                    )
        finally:
            if owns_tagger and active_tagger is not None:
                closer = getattr(active_tagger, "close", None)
                if callable(closer):
                    closer()
            if owns_embedder and active_embedder is not None:
                closer = getattr(active_embedder, "close", None)
                if callable(closer):
                    closer()

        return summary

    # ---------------------------------------------------------------- planning
    def _load_root_state(self, root_key: str) -> _DbRootState | None:
        conn = self.db.read_conn()
        r = conn.execute(
            """
            SELECT id, root_key, root_path, recursive, last_full_verification_ns, dirty
            FROM roots WHERE root_key = ?
            """,
            (root_key,),
        ).fetchone()
        if r is None:
            return None
        root_id = int(r["id"])
        folder_rows = conn.execute(
            "SELECT id, folder_key, path, mtime_ns FROM folders WHERE root_id = ?",
            (root_id,),
        ).fetchall()
        folders = {
            str(fr["folder_key"]): (int(fr["id"]), str(fr["path"]), int(fr["mtime_ns"]))
            for fr in folder_rows
        }
        if root_key not in folders:
            return None
        img_rows = conn.execute(
            """
            SELECT i.path_key, i.point_id, f.folder_key, i.mtime_ns, i.size_bytes, i.name
            FROM images AS i
            JOIN folders AS f ON f.id = i.folder_id
            WHERE i.root_id = ? AND i.deleted = 0
            """,
            (root_id,),
        ).fetchall()
        images = {
            str(ir["path_key"]): (
                str(ir["point_id"]),
                str(ir["folder_key"]),
                int(ir["mtime_ns"]),
                int(ir["size_bytes"]),
                str(ir["name"]),
            )
            for ir in img_rows
        }
        return _DbRootState(
            root_id=root_id,
            root_key=str(r["root_key"]),
            root_path=str(r["root_path"]),
            recursive=bool(r["recursive"]),
            last_full_verification_ns=int(r["last_full_verification_ns"]),
            dirty=bool(r["dirty"]),
            folders=folders,
            images=images,
        )

    def _requires_full_verification(
        self,
        state: _DbRootState | None,
        *,
        recursive: bool,
        force: bool,
        remove_missing: bool,
    ) -> bool:
        if (
            force
            or not remove_missing
            or state is None
            or state.dirty
            or state.recursive != recursive
            or not recursive
            or state.last_full_verification_ns <= 0
        ):
            return True
        try:
            interval = max(0, int(self.config.indexing.full_verification_interval_seconds))
        except (TypeError, ValueError):
            interval = 0
        return interval == 0 or (
            time.time_ns() - state.last_full_verification_ns >= interval * 1_000_000_000
        )

    def _plan_full_root(
        self,
        root: Path,
        root_key: str,
        state: _DbRootState | None,
        *,
        recursive: bool,
        force: bool,
        remove_missing: bool,
    ) -> tuple[
        list[tuple[Path, Path, int, int, int]],
        list[str],
        dict[str, tuple[str, int]],
        set[str],
        int,
        int,
    ]:
        scanned_files, dir_mtimes = self._scan_subtree(root, recursive=recursive)
        existing = state.images if state is not None else {}

        todo: list[tuple[Path, Path, int, int, int]] = []
        changed_keys: set[str] = set()
        for pkey, (path, mtime_ns, size_bytes, folder_key) in scanned_files.items():
            prev = existing.get(pkey)
            f_mtime = dir_mtimes.get(folder_key, ("", 0))[1]
            if (
                force
                or prev is None
                or prev[2] != mtime_ns
                or prev[3] != size_bytes
                or prev[4] != path.name
            ):
                todo.append((root, path, mtime_ns, size_bytes, f_mtime))
                changed_keys.add(pkey)

        stale_pids: list[str] = []
        removed_dirs: set[str] = set()
        if remove_missing and existing:
            for pkey, (pid, _, _, _, _) in existing.items():
                if pkey not in scanned_files:
                    stale_pids.append(pid)
                elif pkey in changed_keys and pid != point_id(scanned_files[pkey][0]):
                    stale_pids.append(pid)
            if state is not None:
                removed_dirs = {fk for fk in state.folders if fk not in dir_mtimes}
        stale_pids = list(dict.fromkeys(stale_pids))
        discovered = len(scanned_files)
        unchanged = discovered - len(todo)
        return todo, stale_pids, dir_mtimes, removed_dirs, discovered, unchanged

    def _plan_delta_root(
        self,
        root: Path,
        state: _DbRootState,
    ) -> tuple[
        list[tuple[Path, Path, int, int, int]],
        list[str],
        dict[str, tuple[str, int]],
        set[str],
        int,
        int,
    ]:
        if not root.exists() or not root.is_dir():
            self._set_root_dirty(state.root_key, state.root_path, recursive=True, dirty=True)
            raise FileNotFoundError(f"图片目录不存在：{root}")

        changed_dirs: list[str] = []
        for fkey, (_, fpath, prev_mtime) in state.folders.items():
            p = Path(fpath)
            try:
                st = p.stat()
            except FileNotFoundError:
                changed_dirs.append(fkey)
                continue
            if not p.is_dir() or st.st_mtime_ns != prev_mtime or prev_mtime <= 0:
                changed_dirs.append(fkey)

        if not changed_dirs:
            cnt = len(state.images)
            return [], [], {}, set(), cnt, cnt

        current_images = dict(state.images)
        dir_mtimes: dict[str, tuple[str, int]] = {
            fk: (fp, mt) for fk, (_, fp, mt) in state.folders.items()
        }
        removed_dirs: set[str] = set()
        rescanned: dict[str, tuple[Path, int, int, str]] = {}
        known_dirs = set(state.folders)

        for dkey in changed_dirs:
            dpath = Path(state.folders[dkey][1]) if dkey in state.folders else Path(dkey)
            if not dpath.exists() or not dpath.is_dir():
                self._drop_subtree(current_images, dir_mtimes, removed_dirs, dkey)
                continue
            scanned = self._scan_changed_directory(dpath, known_dirs)
            if scanned is None:
                self._drop_subtree(current_images, dir_mtimes, removed_dirs, dkey)
                continue
            files_found, dirs_found = scanned
            for pkey in [k for k, v in current_images.items() if v[1] == dkey]:
                del current_images[pkey]
            rescanned.update(files_found)
            dir_mtimes.update(dirs_found)

        todo: list[tuple[Path, Path, int, int, int]] = []
        changed_keys: set[str] = set()
        for pkey, (path, mtime_ns, size_bytes, folder_key) in rescanned.items():
            prev = state.images.get(pkey)
            f_mtime = dir_mtimes.get(folder_key, ("", 0))[1]
            if (
                prev is None
                or prev[2] != mtime_ns
                or prev[3] != size_bytes
                or prev[4] != path.name
            ):
                todo.append((root, path, mtime_ns, size_bytes, f_mtime))
                changed_keys.add(pkey)

        expected_keys = set(current_images) | set(rescanned)
        stale_pids = [
            pid for pkey, (pid, _, _, _, _) in state.images.items() if pkey not in expected_keys
        ]
        stale_pids.extend(
            pid
            for pkey, (pid, _, _, _, _) in state.images.items()
            if pkey in changed_keys and pid != point_id(rescanned[pkey][0])
        )
        stale_pids = list(dict.fromkeys(stale_pids))
        discovered = len(current_images) + len(rescanned)
        unchanged = discovered - len(todo)
        return todo, stale_pids, dir_mtimes, removed_dirs, discovered, unchanged

    @staticmethod
    def _drop_subtree(
        images: dict[str, tuple[str, str, int, int, str]],
        dir_mtimes: dict[str, tuple[str, int]],
        removed_dirs: set[str],
        directory_key: str,
    ) -> None:
        for pkey in [
            k
            for k, v in images.items()
            if _is_within(k, directory_key) or _is_within(v[1], directory_key)
        ]:
            del images[pkey]
        for fk in [k for k in dir_mtimes if _is_within(k, directory_key)]:
            del dir_mtimes[fk]
            removed_dirs.add(fk)

    def _scan_subtree(
        self, root: Path, *, recursive: bool
    ) -> tuple[dict[str, tuple[Path, int, int, str]], dict[str, tuple[str, int]]]:
        """Traverse `root` using `os.scandir`, collecting DirEntry file stats."""
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"图片目录不存在：{root}")
        images: dict[str, tuple[Path, int, int, str]] = {}
        directories: dict[str, tuple[str, int]] = {}
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                dir_stat = directory.stat()
            except FileNotFoundError:
                continue
            d_path = str(directory.resolve())
            d_key = os.path.normcase(d_path)
            directories[d_key] = (d_path, dir_stat.st_mtime_ns)
            try:
                with os.scandir(d_path) as entries:
                    for entry in entries:
                        try:
                            if entry.is_file():
                                ext = os.path.splitext(entry.name)[1].lower()
                                if ext in IMAGE_EXTENSIONS:
                                    fst = entry.stat()
                                    f_path = Path(entry.path)
                                    p_key = os.path.normcase(entry.path)
                                    images[p_key] = (
                                        f_path,
                                        fst.st_mtime_ns,
                                        fst.st_size,
                                        d_key,
                                    )
                            elif recursive and entry.is_dir(follow_symlinks=False):
                                pending.append(Path(entry.path))
                        except OSError:
                            continue
            except FileNotFoundError:
                directories.pop(d_key, None)
            except OSError:
                raise
        return images, directories

    def _scan_changed_directory(
        self, directory: Path, known_directories: set[str]
    ) -> tuple[dict[str, tuple[Path, int, int, str]], dict[str, tuple[str, int]]] | None:
        try:
            dir_stat = directory.stat()
            d_path = str(directory.resolve())
            d_key = os.path.normcase(d_path)
            images: dict[str, tuple[Path, int, int, str]] = {}
            directories: dict[str, tuple[str, int]] = {
                d_key: (d_path, dir_stat.st_mtime_ns)
            }
            with os.scandir(d_path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file():
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in IMAGE_EXTENSIONS:
                                fst = entry.stat()
                                images[os.path.normcase(entry.path)] = (
                                    Path(entry.path),
                                    fst.st_mtime_ns,
                                    fst.st_size,
                                    d_key,
                                )
                            continue
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    child_key = os.path.normcase(entry.path)
                    if child_key not in known_directories:
                        c_imgs, c_dirs = self._scan_subtree(
                            Path(entry.path), recursive=True
                        )
                        images.update(c_imgs)
                        directories.update(c_dirs)
            return images, directories
        except FileNotFoundError:
            return None

    # ---------------------------------------------------------------- DB state commits
    def _set_root_dirty(
        self, root_key: str, root_path: str, *, recursive: bool, dirty: bool
    ) -> None:
        with self.db.write_tx() as conn:
            conn.execute(
                """
                INSERT INTO roots(root_key, root_path, recursive, last_full_verification_ns, dirty)
                VALUES (?, ?, ?, 0, ?)
                ON CONFLICT(root_key) DO UPDATE SET
                    root_path = excluded.root_path,
                    recursive = excluded.recursive,
                    dirty = excluded.dirty
                """,
                (root_key, root_path, int(recursive), int(dirty)),
            )

    def _commit_root_directory_state(
        self,
        root: Path,
        root_key: str,
        *,
        recursive: bool,
        dir_mtimes: dict[str, tuple[str, int]],
        removed_dirs: set[str],
        mark_verified: bool,
    ) -> None:
        root_path = str(root.resolve())
        now_ns = time.time_ns()
        with self.db.write_tx() as conn:
            root_id = self.images_repo._ensure_root_locked(
                conn, root_key, root_path, recursive=recursive
            )
            if mark_verified:
                conn.execute(
                    """
                    UPDATE roots
                    SET recursive = ?, last_full_verification_ns = ?, dirty = 0
                    WHERE id = ?
                    """,
                    (int(recursive), now_ns, root_id),
                )
            else:
                conn.execute(
                    "UPDATE roots SET dirty = 0 WHERE id = ?",
                    (root_id,),
                )

            folder_cache: dict[str, int] = {}
            for fkey, (fpath, fmtime) in sorted(
                dir_mtimes.items(), key=lambda kv: len(kv[0])
            ):
                self.images_repo._ensure_folder_chain_locked(
                    conn,
                    root_id,
                    root_key,
                    root_path,
                    fkey,
                    fpath,
                    mtime_ns=fmtime,
                    cache=folder_cache,
                )

            if removed_dirs:
                for fkey in removed_dirs:
                    row = conn.execute(
                        "SELECT id FROM folders WHERE folder_key = ?", (fkey,)
                    ).fetchone()
                    if row is None:
                        continue
                    fid = int(row["id"])
                    has_live = conn.execute(
                        "SELECT 1 FROM images WHERE folder_id = ? AND deleted = 0 LIMIT 1",
                        (fid,),
                    ).fetchone()
                    if has_live is None:
                        conn.execute(
                            "UPDATE folders SET mtime_ns = 0, selected = 0 WHERE id = ?",
                            (fid,),
                        )

    def _resolve_roots(self, roots: Sequence[Path]) -> list[Path]:
        resolved: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            p = Path(root).expanduser().resolve()
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"图片目录不存在：{p}")
            key = canonical_path(p)
            if key not in seen:
                seen.add(key)
                resolved.append(p)
        return resolved

    def _assert_non_overlapping_roots(self, roots: Sequence[Path]) -> None:
        root_keys = [canonical_path(r) for r in roots]
        rows = self.db.read_conn().execute(
            """
            SELECT r.root_key
            FROM roots AS r
            WHERE EXISTS (
                SELECT 1 FROM images AS i WHERE i.root_id = r.id AND i.deleted = 0
            )
            """
        ).fetchall()
        known_keys = [str(r["root_key"]) for r in rows]
        candidates = [*root_keys, *(k for k in known_keys if k not in root_keys)]
        for idx, key in enumerate(candidates):
            for other in candidates[idx + 1 :]:
                if _is_within(key, other) or _is_within(other, key):
                    raise ValueError(
                        "索引根目录不能相互嵌套："
                        f"{key} 与 {other}。请只保留父目录，避免同一图片的状态互相覆盖。"
                    )
