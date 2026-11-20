"""Thread-safe WAL SQLite connection manager with serialized write transactions."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .migrations import apply_migrations


class Database:
    """Manages per-thread SQLite connections and single-writer WAL transactions."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._pool_lock = threading.Lock()
        self._connections: set[sqlite3.Connection] = set()
        self._write_lock = threading.RLock()
        self._closed = False
        self.write_generation: int = 0
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.write_tx() as conn:
            apply_migrations(conn)

    def _open_connection(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.path),
            timeout=15.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = -32000")
        return conn

    def read_conn(self) -> sqlite3.Connection:
        """Return the current thread's connection, opening one if needed."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        with self._pool_lock:
            if self._closed:
                self._closed = False
            if conn is not None and conn in self._connections:
                return conn
            conn = self._open_connection()
            self._connections.add(conn)
            self._local.conn = conn
            return conn

    @contextmanager
    def write_tx(self) -> Iterator[sqlite3.Connection]:
        """Execute a serialized BEGIN IMMEDIATE transaction (reentrant on the same thread)."""
        with self._write_lock:
            conn = self.read_conn()
            if conn.in_transaction:
                yield conn
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            else:
                if conn.in_transaction:
                    conn.execute("COMMIT")
                self.write_generation += 1

    def checkpoint(self) -> None:
        """Flush WAL pages back into the main database file."""
        with self._write_lock:
            conn = self.read_conn()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def reopen(self, new_path: Path) -> None:
        """Close all connections, switch to `new_path`, and ensure schema is ready."""
        with self._write_lock:
            self.close()
            self.path = Path(new_path)
            self._closed = False
            self.write_generation += 1
            self._initialize_schema()

    def close(self) -> None:
        """Close all tracked thread-local connections."""
        with self._pool_lock:
            self._closed = True
            conns = list(self._connections)
            self._connections.clear()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        if hasattr(self._local, "conn"):
            self._local.conn = None
