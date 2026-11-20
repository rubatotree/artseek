"""Reference-counted lazy model lifecycle manager with idle auto-unloading."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from ..config import (
    DEFAULT_AUTO_UNLOAD_SECONDS,
    AppConfig,
    clamp_auto_unload_seconds,
    load_auto_unload_seconds,
    save_auto_unload_seconds,
)
from ..ml.llm import OllamaClient, OllamaUnavailable
from ..vectors.index import VectorIndex


def _default_embedder(config: AppConfig) -> Any:
    from ..models.embedder import SiglipEmbedder

    return SiglipEmbedder(config.embedding)


class ModelLifecycle:
    """Lazy-loads SigLIP on demand, tracks active users, and auto-unloads when idle."""

    def __init__(
        self,
        config: AppConfig,
        *,
        embedder_factory: Callable[[], Any] | None = None,
        llm: OllamaClient | None = None,
        vectors: VectorIndex | None = None,
        on_dimension_known: Callable[[int], None] | None = None,
        auto_unload_seconds: int | None = None,
    ) -> None:
        self.config = config
        self._embedder_factory = embedder_factory or (lambda: _default_embedder(config))
        self.llm = llm if llm is not None else OllamaClient(config.llm)
        self.vectors = vectors
        self._on_dimension_known = on_dimension_known
        self.embedder: Any = None
        self._lock = threading.RLock()
        self._busy = 0
        self._loading = False
        self._idle_timer: threading.Timer | None = None
        self._last_used_at: float | None = None
        if auto_unload_seconds is None:
            auto_unload_seconds = load_auto_unload_seconds(config)
        self.auto_unload_seconds = clamp_auto_unload_seconds(
            auto_unload_seconds, DEFAULT_AUTO_UNLOAD_SECONDS
        )

    def close(self) -> None:
        self.unload_model(force=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded = self.embedder is not None
            remaining = None
            if loaded and self.auto_unload_seconds > 0 and self._last_used_at is not None:
                remaining = max(
                    0, int(self.auto_unload_seconds - (time.monotonic() - self._last_used_at))
                )
            minutes, extra = divmod(max(0, self.auto_unload_seconds), 60)
            if extra:
                minutes += 1
            return {
                "loaded": loaded,
                "loading": self._loading,
                "busy": self._busy > 0,
                "model": self.config.embedding.model,
                "auto_unload_seconds": self.auto_unload_seconds,
                "auto_unload_minutes": minutes,
                "idle_remaining_seconds": remaining,
            }

    def set_auto_unload_seconds(self, seconds: int) -> None:
        value = clamp_auto_unload_seconds(seconds, DEFAULT_AUTO_UNLOAD_SECONDS)
        with self._lock:
            self.auto_unload_seconds = value
            self._last_used_at = time.monotonic() if self.embedder is not None else None
            if self._busy == 0:
                self._schedule_unload_locked()
        save_auto_unload_seconds(self.config, value)

    def load_model(self) -> None:
        with self._lock:
            self._load_if_needed_locked()
            self._last_used_at = time.monotonic()
            if self._busy == 0:
                self._schedule_unload_locked()

    def unload_model(self, *, force: bool = False) -> bool:
        with self._lock:
            if self._busy > 0 and not force:
                return False
            self._unload_locked()
            return True

    @contextmanager
    def hold_model(self) -> Iterator[Any]:
        self.ensure_loaded()
        try:
            yield self.embedder
        finally:
            self.release()

    def ensure_loaded(self) -> None:
        with self._lock:
            self._load_if_needed_locked()
            self._busy += 1
            self._last_used_at = time.monotonic()
            self._cancel_timer_locked()

    def release(self) -> None:
        with self._lock:
            self._busy = max(0, self._busy - 1)
            self._last_used_at = time.monotonic()
            if self._busy == 0:
                self._schedule_unload_locked()

    def warmup(self, status: Callable[[str], None] | None = None) -> None:
        notify = status or (lambda _msg: None)
        with self.hold_model() as embedder:
            notify("正在预热图文向量模型……")
            warmup = getattr(embedder, "warmup", None)
            if callable(warmup):
                warmup()
            else:
                embedder.encode_texts(["warmup"])
            if not self.config.llm.enabled:
                return
            notify(f"正在载入本地大模型 {self.config.llm.query_model}……")
            try:
                self.llm.preload(keep_alive=self._ollama_keep_alive())
                notify(f"本地大模型 {self.config.llm.query_model} 已载入")
            except OllamaUnavailable as exc:
                notify(f"Ollama 未就绪，查询将使用内置词典：{exc}")

    def _ollama_keep_alive(self) -> int:
        seconds = int(getattr(self, "auto_unload_seconds", 0) or 0)
        return -1 if seconds <= 0 else seconds

    def _load_if_needed_locked(self) -> None:
        if self.embedder is not None:
            return
        self._loading = True
        try:
            self.embedder = self._embedder_factory()
            dimension = getattr(self.embedder, "dimension", None)
            if dimension is not None and callable(self._on_dimension_known):
                self._on_dimension_known(int(dimension))
            device = getattr(self.embedder, "device", None)
            if (
                self.vectors is not None
                and device is not None
                and getattr(device, "type", None) == "cuda"
            ):
                try:
                    self.vectors.upload_cuda(device)
                except Exception:
                    pass
        finally:
            self._loading = False

    def _unload_locked(self) -> None:
        self._cancel_timer_locked()
        embedder = self.embedder
        self.embedder = None
        self._last_used_at = None
        if embedder is not None:
            closer = getattr(embedder, "close", None)
            if callable(closer):
                closer()
        if self.vectors is not None:
            self.vectors.release_cuda()
        unload_llm = getattr(self.llm, "unload", None)
        if callable(unload_llm):
            try:
                unload_llm()
            except Exception:
                pass

    def _cancel_timer_locked(self) -> None:
        timer = self._idle_timer
        self._idle_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_unload_locked(self) -> None:
        self._cancel_timer_locked()
        if self.auto_unload_seconds <= 0 or self.embedder is None:
            return
        timer = threading.Timer(self.auto_unload_seconds, self._idle_unload)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_unload(self) -> None:
        try:
            with self._lock:
                if self._busy > 0 or self.embedder is None:
                    return
                if self.auto_unload_seconds <= 0 or self._last_used_at is None:
                    return
                elapsed = time.monotonic() - self._last_used_at
                if elapsed < self.auto_unload_seconds:
                    remaining = self.auto_unload_seconds - elapsed
                    timer = threading.Timer(remaining, self._idle_unload)
                    timer.daemon = True
                    self._idle_timer = timer
                    timer.start()
                    return
                self._unload_locked()
        except Exception:
            pass
