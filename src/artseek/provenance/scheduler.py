"""Per-domain provenance fetch scheduler and rate-limit circuit breaker."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


class FetchError(RuntimeError):
    def __init__(self, message: str, *, status: str = "error", retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class NotFoundError(FetchError):
    def __init__(self, message: str = "作品或推文不存在"):
        super().__init__(message, status="not_found")


class ProtectedError(FetchError):
    def __init__(self, message: str = "作品受保护或需要登录"):
        super().__init__(message, status="protected")


class RateLimitError(FetchError):
    def __init__(self, message: str = "请求过于频繁", retry_after: int | None = None):
        super().__init__(message, status="rate_limited", retry_after=retry_after)


class BudgetExhausted(FetchError):
    def __init__(self, message: str = "今日请求预算已用完"):
        super().__init__(message, status="rate_limited")


@dataclass(slots=True)
class BudgetSnapshot:
    domain: str
    min_interval: float
    concurrency: int
    daily_limit: int | None
    used_today: int
    blocked_until: int
    long_remaining: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "min_interval": self.min_interval,
            "concurrency": self.concurrency,
            "daily_limit": self.daily_limit,
            "used_today": self.used_today,
            "blocked_until": self.blocked_until,
            "long_remaining": self.long_remaining,
        }


_DOMAIN_LOCKS: dict[tuple[str, str], threading.RLock] = {}
_DOMAIN_SEMAPHORES: dict[tuple[str, str], threading.BoundedSemaphore] = {}
_DOMAIN_LOCKS_GUARD = threading.Lock()


def _domain_lock(store: Any, domain: str) -> threading.RLock:
    key = (str(store.path.resolve()).casefold(), domain)
    with _DOMAIN_LOCKS_GUARD:
        return _DOMAIN_LOCKS.setdefault(key, threading.RLock())


def _domain_semaphore(store: Any, domain: str, concurrency: int) -> threading.BoundedSemaphore:
    key = (str(store.path.resolve()).casefold(), domain)
    with _DOMAIN_LOCKS_GUARD:
        return _DOMAIN_SEMAPHORES.setdefault(
            key, threading.BoundedSemaphore(max(1, int(concurrency)))
        )


class RateLimiter:
    def __init__(
        self,
        store: Any,
        domain: str,
        *,
        min_interval: float,
        daily_limit: int,
        concurrency: int = 1,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.domain = domain
        self.min_interval = max(0.0, float(min_interval))
        limit = int(daily_limit)
        self.daily_limit = limit if limit > 0 else None
        self.sleep = sleep
        self.clock = clock
        self._lock = _domain_lock(store, domain)
        self.concurrency = max(1, int(concurrency))
        self._semaphore = _domain_semaphore(store, domain, self.concurrency)

    def snapshot(self) -> BudgetSnapshot:
        state = self.store.domain_state(self.domain)
        today = time.strftime("%Y-%m-%d", time.gmtime(int(self.clock())))
        used_today = int(state.get("day_count") or 0) if state.get("day") == today else 0
        return BudgetSnapshot(
            domain=self.domain,
            min_interval=self.min_interval,
            concurrency=self.concurrency,
            daily_limit=self.daily_limit,
            used_today=used_today,
            blocked_until=int(state.get("blocked_until") or 0),
            long_remaining=state.get("long_remaining"),
        )

    def wait(
        self,
        *,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> None:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise BudgetExhausted("任务已停止")
            while pause_event is not None and pause_event.is_set():
                if stop_event is not None and stop_event.is_set():
                    raise BudgetExhausted("任务已停止")
                self.sleep(0.25)
            state = self.store.domain_state(self.domain)
            now = int(self.clock())
            blocked_until = int(state.get("blocked_until") or 0)
            if blocked_until > now:
                self.sleep(min(5.0, max(0.25, blocked_until - now)))
                continue
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            used = int(state.get("day_count") or 0) if state.get("day") == day else 0
            if self.daily_limit is not None and used >= self.daily_limit:
                raise BudgetExhausted(
                    f"{self.domain} 今日预算已用完（{used}/{self.daily_limit}）"
                )
            last = float(state.get("last_request") or 0.0)
            wait_for = last + self.min_interval - self.clock()
            if wait_for > 0:
                self.sleep(wait_for)
                continue
            return

    def record(self) -> None:
        self.store.note_request(self.domain)

    def _reserve(
        self,
        *,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> None:
        """Reserve a request start time while allowing concurrent in-flight calls."""
        while True:
            if stop_event is not None and stop_event.is_set():
                raise BudgetExhausted("任务已停止")
            while pause_event is not None and pause_event.is_set():
                if stop_event is not None and stop_event.is_set():
                    raise BudgetExhausted("任务已停止")
                self.sleep(0.25)
            with self._lock:
                state = self.store.domain_state(self.domain)
                now = int(self.clock())
                blocked_until = int(state.get("blocked_until") or 0)
                if blocked_until > now:
                    wait_for = min(5.0, max(0.25, blocked_until - now))
                else:
                    day = time.strftime("%Y-%m-%d", time.gmtime(now))
                    used = (
                        int(state.get("day_count") or 0)
                        if state.get("day") == day
                        else 0
                    )
                    if self.daily_limit is not None and used >= self.daily_limit:
                        raise BudgetExhausted(
                            f"{self.domain} 今日预算已用完（{used}/{self.daily_limit}）"
                        )
                    start = max(
                        float(self.clock()),
                        float(state.get("last_request") or 0.0) + self.min_interval,
                    )
                    if start <= self.clock():
                        self.store.note_request(self.domain)
                        return
                    wait_for = start - self.clock()
            self.sleep(min(5.0, max(0.05, wait_for)))

    @contextmanager
    def slot(
        self,
        *,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ):
        """Allow up to `concurrency` in-flight requests, spaced by the interval."""
        with self._semaphore:
            self._reserve(stop_event=stop_event, pause_event=pause_event)
            yield

    def block(self, seconds: int, error: str) -> int:
        return self.store.set_blocked(self.domain, seconds, error)
