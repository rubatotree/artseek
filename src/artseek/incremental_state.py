from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

STATE_VERSION = "2"


@dataclass(frozen=True, slots=True)
class IndexedImageState:
    path_key: str
    root_key: str
    point_id: str
    mtime_ns: int
    size_bytes: int
    folder_key: str
    name: str


@dataclass(frozen=True, slots=True)
class DirectoryState:
    path_key: str
    mtime_ns: int


@dataclass(slots=True)
class RootIndexState:
    root_key: str
    root_path: str
    recursive: bool
    last_full_verification_ns: int
    images: dict[str, IndexedImageState]
    directories: dict[str, DirectoryState]


class IncrementalIndexState:
    """Transactional snapshot of completed filesystem/Qdrant index updates.

    This is deliberately a cache rather than an independent source of truth.
    A missing, corrupt, or explicitly invalidated root therefore makes callers
    perform a full filesystem/Qdrant reconciliation before using it again.
    """

    def __init__(self, path: Path):
        self.path = path

    def load_root(self, root_key: str) -> RootIndexState | None:
        try:
            connection = self._connect()
            try:
                dirty = connection.execute(
                    "SELECT 1 FROM dirty_roots WHERE root_key = ?", (root_key,)
                ).fetchone()
                if dirty is not None:
                    return None
                row = connection.execute(
                    """
                    SELECT root_path, recursive, last_full_verification_ns
                    FROM roots WHERE root_key = ?
                    """,
                    (root_key,),
                ).fetchone()
                if row is None:
                    return None
                image_rows = connection.execute(
                    """
                    SELECT path_key, root_key, point_id, mtime_ns, size_bytes, folder_key, name
                    FROM images WHERE root_key = ?
                    """,
                    (root_key,),
                ).fetchall()
                directory_rows = connection.execute(
                    "SELECT path_key, mtime_ns FROM directories WHERE root_key = ?",
                    (root_key,),
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            self._discard_corrupt_cache()
            return None

        images = {
            str(item[0]): IndexedImageState(
                path_key=str(item[0]),
                root_key=str(item[1]),
                point_id=str(item[2]),
                mtime_ns=int(item[3]),
                size_bytes=int(item[4]),
                folder_key=str(item[5]),
                name=str(item[6]),
            )
            for item in image_rows
        }
        directories = {
            str(item[0]): DirectoryState(path_key=str(item[0]), mtime_ns=int(item[1]))
            for item in directory_rows
        }
        # A root row without its root directory can only arise from a damaged
        # or hand-edited cache; never use such a partial snapshot for deletion.
        if root_key not in directories:
            return None
        return RootIndexState(
            root_key=root_key,
            root_path=str(row[0]),
            recursive=bool(row[1]),
            last_full_verification_ns=int(row[2]),
            images=images,
            directories=directories,
        )

    def root_keys(self) -> set[str]:
        try:
            connection = self._connect()
            try:
                rows = connection.execute("SELECT root_key FROM roots").fetchall()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            self._discard_corrupt_cache()
            return set()
        return {str(row[0]) for row in rows}

    def mark_dirty(self, root_key: str) -> None:
        """Persist a write-ahead marker before mutating this root in Qdrant."""
        try:
            connection = self._connect()
            try:
                connection.execute(
                    "INSERT OR IGNORE INTO dirty_roots(root_key) VALUES (?)", (root_key,)
                )
                connection.commit()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            self._discard_corrupt_cache()
            raise

    def is_dirty(self, root_key: str) -> bool:
        """Whether this root needs conservative full re-embedding recovery."""
        try:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT 1 FROM dirty_roots WHERE root_key = ?", (root_key,)
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            self._discard_corrupt_cache()
            return True
        return row is not None

    def mark_all_dirty(self) -> None:
        """Persist write-ahead markers for every valid root before a global clear."""
        try:
            connection = self._connect()
            try:
                connection.execute(
                    "INSERT OR IGNORE INTO dirty_roots(root_key) SELECT root_key FROM roots"
                )
                connection.commit()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            self._discard_corrupt_cache()
            raise

    def catalog_payloads(self) -> list[tuple[str, dict[str, int | str]]] | None:
        """Return catalog rows only when every snapshot is committed and valid."""
        try:
            connection = self._connect()
            try:
                if connection.execute("SELECT 1 FROM dirty_roots LIMIT 1").fetchone():
                    return None
                incomplete = connection.execute(
                    """
                    SELECT 1
                    FROM roots AS root
                    LEFT JOIN directories AS directory
                      ON directory.root_key = root.root_key
                     AND directory.path_key = root.root_key
                    WHERE directory.path_key IS NULL
                    LIMIT 1
                    """
                ).fetchone()
                if incomplete is not None:
                    return None
                rows = connection.execute(
                    """
                    SELECT
                        image.point_id,
                        image.path_key,
                        root.root_path,
                        image.root_key,
                        image.folder_key,
                        image.name,
                        image.mtime_ns
                    FROM images AS image
                    JOIN roots AS root ON root.root_key = image.root_key
                    """
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            self._discard_corrupt_cache()
            return None
        return [
            (
                str(point_id),
                {
                    "path": str(path_key),
                    "path_key": str(path_key),
                    "root": str(root_path),
                    "root_key": str(root_key),
                    "folder_key": str(folder_key),
                    "name": str(name),
                    "mtime_ns": int(mtime_ns),
                },
            )
            for point_id, path_key, root_path, root_key, folder_key, name, mtime_ns in rows
        ]

    def replace_root(self, state: RootIndexState) -> None:
        """Atomically replace one root only after its Qdrant update has succeeded."""
        try:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM images WHERE root_key = ?", (state.root_key,))
                connection.execute("DELETE FROM directories WHERE root_key = ?", (state.root_key,))
                connection.execute("DELETE FROM roots WHERE root_key = ?", (state.root_key,))
                connection.execute(
                    """
                    INSERT INTO roots(root_key, root_path, recursive, last_full_verification_ns)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        state.root_key,
                        state.root_path,
                        int(state.recursive),
                        state.last_full_verification_ns,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO images(
                        root_key, path_key, point_id, mtime_ns, size_bytes, folder_key, name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.root_key,
                            item.path_key,
                            item.point_id,
                            item.mtime_ns,
                            item.size_bytes,
                            item.folder_key,
                            item.name,
                        )
                        for item in state.images.values()
                    ],
                )
                connection.executemany(
                    "INSERT INTO directories(root_key, path_key, mtime_ns) VALUES (?, ?, ?)",
                    [
                        (state.root_key, item.path_key, item.mtime_ns)
                        for item in state.directories.values()
                    ],
                )
                connection.execute("DELETE FROM dirty_roots WHERE root_key = ?", (state.root_key,))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            self._discard_corrupt_cache()
            raise

    def invalidate_root(self, root_key: str) -> None:
        """Remove a root snapshot so the next update must reconcile it fully."""
        try:
            connection = self._connect()
            try:
                connection.execute("DELETE FROM roots WHERE root_key = ?", (root_key,))
                connection.commit()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            self._discard_corrupt_cache()

    def invalidate_all(self) -> None:
        try:
            connection = self._connect()
            try:
                connection.execute("DELETE FROM roots")
                connection.commit()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            self._discard_corrupt_cache()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS roots(
                root_key TEXT PRIMARY KEY,
                root_path TEXT NOT NULL,
                recursive INTEGER NOT NULL,
                last_full_verification_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images(
                root_key TEXT NOT NULL REFERENCES roots(root_key) ON DELETE CASCADE,
                path_key TEXT NOT NULL,
                point_id TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                folder_key TEXT NOT NULL,
                name TEXT NOT NULL,
                PRIMARY KEY(root_key, path_key)
            );
            CREATE TABLE IF NOT EXISTS directories(
                root_key TEXT NOT NULL REFERENCES roots(root_key) ON DELETE CASCADE,
                path_key TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                PRIMARY KEY(root_key, path_key)
            );
            CREATE TABLE IF NOT EXISTS dirty_roots(
                root_key TEXT PRIMARY KEY
            );
            CREATE INDEX IF NOT EXISTS images_root_path ON images(root_key, path_key);
            CREATE INDEX IF NOT EXISTS directories_root_path ON directories(root_key, path_key);
            """
        )
        version = connection.execute(
            "SELECT value FROM metadata WHERE key = 'version'"
        ).fetchone()
        if version is None:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('version', ?)", (STATE_VERSION,)
            )
        elif str(version[0]) != STATE_VERSION:
            connection.executescript(
                "DELETE FROM images; DELETE FROM directories; DELETE FROM roots; "
                "DELETE FROM dirty_roots; "
                "DELETE FROM metadata;"
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('version', ?)", (STATE_VERSION,)
            )
        connection.commit()
        return connection

    def _discard_corrupt_cache(self) -> None:
        # The sidecar is regenerable. A corrupt cache must never block a safe
        # full reconciliation or be mistaken for a valid empty root.
        for suffix in ("", "-wal", "-shm"):
            try:
                self.path.with_name(self.path.name + suffix).unlink(missing_ok=True)
            except OSError:
                pass
