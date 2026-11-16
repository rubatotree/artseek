from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

import httpx

from .parser import SourceRef, hashtags, pixiv_user_id_from_url, twitter_handle_from_url
from .store import ProvenanceStore

PIXIV_DOMAIN = "pixiv.net"
TWITTER_DOMAIN = "api.fxtwitter.com"
PIXIV_MIN_INTERVAL = 0.75
TWITTER_MIN_INTERVAL = 0.2
# 0 means no daily request cap; the per-domain interval remains the guardrail.
PIXIV_DAILY_LIMIT = 0
TWITTER_DAILY_LIMIT = 0
DEFAULT_TIMEOUT = 20.0
PIXIV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.pixiv.net/",
    "Accept": "application/json",
}

ProgressCallback = Callable[[int, int, str], None]


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
class FetchResult:
    platform: str
    source_id: str
    page: int | None
    url: str
    status: str = "ok"
    title: str | None = None
    description: str | None = None
    author_platform: str | None = None
    author_id: str | None = None
    author_name: str | None = None
    author_handle: str | None = None
    author_profile_url: str | None = None
    author_avatar_url: str | None = None
    author_social: dict[str, Any] = field(default_factory=dict)
    posted_at: str | None = None
    page_count: int | None = None
    width: int | None = None
    height: int | None = None
    tags: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "source_id": self.source_id,
            "page": self.page,
            "url": self.url,
            "status": self.status,
            "title": self.title,
            "description": self.description,
            "posted_at": self.posted_at,
            "page_count": self.page_count,
            "width": self.width,
            "height": self.height,
            "tags": list(self.tags),
            "raw": self.raw,
            "confidence": self.confidence,
        }


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


def _domain_lock(store: ProvenanceStore, domain: str) -> threading.RLock:
    key = (str(store.path.resolve()).casefold(), domain)
    with _DOMAIN_LOCKS_GUARD:
        return _DOMAIN_LOCKS.setdefault(key, threading.RLock())


def _domain_semaphore(
    store: ProvenanceStore, domain: str, concurrency: int
) -> threading.BoundedSemaphore:
    key = (str(store.path.resolve()).casefold(), domain)
    with _DOMAIN_LOCKS_GUARD:
        return _DOMAIN_SEMAPHORES.setdefault(
            key, threading.BoundedSemaphore(max(1, int(concurrency)))
        )


class RateLimiter:
    def __init__(
        self,
        store: ProvenanceStore,
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


class PixivClient:
    def __init__(self, client: httpx.Client | None = None):
        self._client = client
        self._owns_client = client is None

    def _get(self, url: str) -> Any:
        client = self._client
        if client is None:
            client = httpx.Client(
                headers=PIXIV_HEADERS,
                follow_redirects=True,
                timeout=DEFAULT_TIMEOUT,
            )
            self._client = client
        try:
            response = client.get(url, headers=PIXIV_HEADERS)
        except httpx.HTTPError as exc:
            raise FetchError(f"Pixiv 请求失败：{exc}") from exc
        if response.status_code == 404:
            raise NotFoundError("Pixiv 作品不存在或已删除")
        if response.status_code in (401, 403):
            content_type = response.headers.get("Content-Type", "").casefold()
            if "text/html" in content_type:
                raise RateLimitError("Pixiv 返回验证页/风控挑战", retry_after=1800)
            raise ProtectedError("Pixiv 作品受保护或需要登录")
        if response.status_code == 429:
            retry_after = _retry_after(response)
            raise RateLimitError("Pixiv 返回 429", retry_after=retry_after)
        if response.status_code >= 400:
            raise FetchError(f"Pixiv 返回 HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise FetchError("Pixiv 返回了非 JSON 内容") from exc
        if not isinstance(payload, dict):
            raise FetchError("Pixiv 响应格式异常")
        if payload.get("error"):
            message = str(payload.get("message") or "")
            if "not found" in message.casefold() or not message:
                raise NotFoundError("Pixiv 作品不存在或已删除")
            raise FetchError(f"Pixiv 错误：{message}")
        body = payload.get("body")
        if not isinstance(body, (dict, list)):
            raise FetchError("Pixiv 响应缺少 body")
        return body

    def fetch(self, ref: SourceRef, *, local_size: tuple[int, int] | None = None) -> FetchResult:
        body = self._get(f"https://www.pixiv.net/ajax/illust/{ref.source_id}")
        if not isinstance(body, dict):
            raise FetchError("Pixiv 作品响应格式异常")
        page_count = _int_or_none(body.get("pageCount"))
        page = ref.page
        width = _int_or_none(body.get("width"))
        height = _int_or_none(body.get("height"))
        confidence = ref.confidence
        pages_unavailable = False
        if page is None:
            if page_count == 1:
                page = 0
            else:
                confidence = min(confidence, 0.5)
        if page is not None and (page > 0 or (page_count and page_count > 1)):
            try:
                pages = self._get(
                    f"https://www.pixiv.net/ajax/illust/{ref.source_id}/pages"
                )
            except (NotFoundError, ProtectedError):
                # R18 / login-only works often expose the illustration metadata
                # but return 404/403 for /pages without a session. Keep the
                # source and author instead of marking the work deleted.
                pages = None
                pages_unavailable = True
                confidence = min(confidence, 0.6)
            if isinstance(pages, list) and 0 <= page < len(pages):
                info = pages[page]
                if isinstance(info, dict):
                    width = _int_or_none(info.get("width")) or width
                    height = _int_or_none(info.get("height")) or height
                    page_count = len(pages)
            elif pages is not None:
                confidence = min(confidence, 0.5)
        if local_size and width and height and tuple(local_size) != (width, height):
            confidence = min(confidence, 0.5)
        tags_block = body.get("tags")
        tags = []
        if isinstance(tags_block, dict):
            for item in tags_block.get("tags") or []:
                if isinstance(item, dict) and item.get("tag"):
                    tags.append(str(item["tag"]))
        return FetchResult(
            platform="pixiv",
            source_id=ref.source_id,
            page=page,
            url=ref.url,
            title=_str_or_none(body.get("illustTitle") or body.get("title")),
            description=_str_or_none(body.get("description") or body.get("illustComment")),
            author_platform="pixiv",
            author_id=_str_or_none(body.get("userId")),
            author_name=_str_or_none(body.get("userName")),
            author_handle=_str_or_none(body.get("userAccount")),
            author_profile_url=(
                f"https://www.pixiv.net/users/{body.get('userId')}"
                if body.get("userId")
                else None
            ),
            posted_at=_str_or_none(body.get("createDate") or body.get("uploadDate")),
            page_count=page_count,
            width=width,
            height=height,
            tags=tags,
            raw={
                "illustType": body.get("illustType"),
                "xRestrict": body.get("xRestrict"),
                "sl": body.get("sl"),
                "isLoginOnly": body.get("isLoginOnly"),
                "pageCount": page_count,
                "pages_unavailable": pages_unavailable,
            },
            confidence=confidence,
        )

    def fetch_author(self, author_id: str) -> dict[str, Any]:
        body = self._get(f"https://www.pixiv.net/ajax/user/{author_id}?full=1")
        if not isinstance(body, dict):
            raise FetchError("Pixiv 作者响应格式异常")
        social = body.get("social")
        return {
            "name": _str_or_none(body.get("name")),
            "handle": _str_or_none(body.get("pixivId")),
            "profile_url": f"https://www.pixiv.net/users/{author_id}",
            "avatar_url": _str_or_none(body.get("imageBig") or body.get("image")),
            "social": social if isinstance(social, dict) else {},
        }

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
        self._client = None


class FxTwitterClient:
    def __init__(self, client: httpx.Client | None = None):
        self._client = client
        self._owns_client = client is None

    def _get(self, url: str) -> dict[str, Any]:
        client = self._client
        if client is None:
            client = httpx.Client(
                headers={"User-Agent": "artseek/0.1"},
                follow_redirects=True,
                timeout=DEFAULT_TIMEOUT,
            )
            self._client = client
        try:
            response = client.get(url)
        except httpx.HTTPError as exc:
            raise FetchError(f"FxTwitter 请求失败：{exc}") from exc
        if response.status_code == 404:
            raise NotFoundError("推文不存在或已删除")
        if response.status_code in (401, 403):
            raise ProtectedError("推文受保护或需要登录")
        if response.status_code == 429:
            raise RateLimitError("FxTwitter 返回 429", retry_after=_retry_after(response))
        if response.status_code >= 400:
            raise FetchError(f"FxTwitter 返回 HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise FetchError("FxTwitter 返回了非 JSON 内容") from exc
        if not isinstance(payload, dict):
            raise FetchError("FxTwitter 响应格式异常")
        tweet = payload.get("tweet")
        if not isinstance(tweet, dict):
            code = payload.get("code")
            if code == 404:
                raise NotFoundError("推文不存在或已删除")
            if code in (401, 403):
                raise ProtectedError("推文受保护或需要登录")
            raise FetchError(str(payload.get("message") or "FxTwitter 响应缺少 tweet"))
        return tweet

    def fetch(self, ref: SourceRef, *, local_size: tuple[int, int] | None = None) -> FetchResult:
        tweet = self._get(f"https://api.fxtwitter.com/status/{ref.source_id}")
        media_block = tweet.get("media")
        media = media_block.get("all") if isinstance(media_block, dict) else []
        if not isinstance(media, list):
            media = []
        page = ref.page
        confidence = ref.confidence
        chosen: dict[str, Any] | None = None
        if page and 1 <= page <= len(media) and isinstance(media[page - 1], dict):
            chosen = media[page - 1]
        elif page is not None:
            confidence = min(confidence, 0.5)
        elif local_size:
            matches = [
                item
                for item in media
                if isinstance(item, dict)
                and _int_or_none(item.get("width")) == int(local_size[0])
                and _int_or_none(item.get("height")) == int(local_size[1])
            ]
            if len(matches) == 1:
                chosen = matches[0]
                page = media.index(chosen) + 1
            else:
                confidence = min(confidence, 0.6)
        width = _int_or_none(chosen.get("width")) if chosen else None
        height = _int_or_none(chosen.get("height")) if chosen else None
        if chosen and local_size and width and height:
            if (int(local_size[0]), int(local_size[1])) != (width, height):
                confidence = min(confidence, 0.5)
        author = tweet.get("author") if isinstance(tweet.get("author"), dict) else {}
        description = _str_or_none(author.get("description"))
        website = author.get("website")
        website_url = (
            _str_or_none(website.get("url")) if isinstance(website, dict) else None
        )
        text = _str_or_none(tweet.get("text")) or ""
        social: dict[str, Any] = {}
        for candidate in (description, website_url, text):
            if not candidate:
                continue
            pixiv_id = pixiv_user_id_from_url(candidate)
            if pixiv_id:
                social["pixiv_user_id"] = pixiv_id
                break
        return FetchResult(
            platform="twitter",
            source_id=ref.source_id,
            page=page,
            url=ref.url,
            title=None,
            description=text or None,
            author_platform="twitter",
            author_id=_str_or_none(author.get("id")),
            author_name=_str_or_none(author.get("name")),
            author_handle=_str_or_none(author.get("screen_name")),
            author_profile_url=_str_or_none(author.get("url")),
            author_avatar_url=_str_or_none(author.get("avatar_url")),
            author_social=social,
            posted_at=_str_or_none(tweet.get("created_at")),
            page_count=len(media),
            width=width,
            height=height,
            tags=hashtags(text),
            raw={
                "possibly_sensitive": tweet.get("possibly_sensitive"),
                "lang": tweet.get("lang"),
                "media_count": len(media),
                "author_description": description,
            },
            confidence=confidence,
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
        self._client = None


class ProvenanceService:
    """One-record enrichment that never runs two requests concurrently."""

    def __init__(
        self,
        store: ProvenanceStore,
        *,
        pixiv: PixivClient | None = None,
        twitter: FxTwitterClient | None = None,
        pixiv_limiter: RateLimiter | None = None,
        twitter_limiter: RateLimiter | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ):
        self.store = store
        self.pixiv = pixiv or PixivClient()
        self.twitter = twitter or FxTwitterClient()
        self.pixiv_limiter = pixiv_limiter or RateLimiter(
            store, PIXIV_DOMAIN, min_interval=PIXIV_MIN_INTERVAL, daily_limit=PIXIV_DAILY_LIMIT
        )
        self.twitter_limiter = twitter_limiter or RateLimiter(
            store,
            TWITTER_DOMAIN,
            min_interval=TWITTER_MIN_INTERVAL,
            daily_limit=TWITTER_DAILY_LIMIT,
        )
        self.stop_event = stop_event or threading.Event()
        self.pause_event = pause_event or threading.Event()

    def budgets(self) -> list[dict[str, Any]]:
        return [self.pixiv_limiter.snapshot().as_dict(), self.twitter_limiter.snapshot().as_dict()]

    def enrich_record(self, record: dict[str, Any]) -> FetchResult:
        platform = str(record.get("platform") or "")
        source_id = str(record.get("source_id") or "")
        if platform not in {"pixiv", "twitter"} or not source_id:
            raise FetchError("只有 Pixiv/X 来源可以自动补全")
        page = record.get("page")
        ref = SourceRef(
            platform=platform,  # type: ignore[arg-type]
            source_id=source_id,
            page=int(page) if page is not None else None,
            url=str(record.get("url") or ""),
            origin=str(record.get("source_origin") or "filename"),  # type: ignore[arg-type]
            confidence=float(record.get("confidence") or 0.9),
        )
        local_size = None
        if record.get("width") and record.get("height"):
            local_size = (int(record["width"]), int(record["height"]))
        if platform == "pixiv":
            with self.pixiv_limiter.slot(
                stop_event=self.stop_event, pause_event=self.pause_event
            ):
                result = self.pixiv.fetch(ref, local_size=local_size)
            author_key = self._resolve_pixiv_author(result)
            self.store.set_fetch_result(record["point_id"], result.as_dict(), author_key=author_key)
            return result
        with self.twitter_limiter.slot(
            stop_event=self.stop_event, pause_event=self.pause_event
        ):
            result = self.twitter.fetch(ref, local_size=local_size)
        author_key = self._resolve_twitter_author(result)
        self.store.set_fetch_result(record["point_id"], result.as_dict(), author_key=author_key)
        return result

    def _resolve_pixiv_author(self, result: FetchResult) -> str | None:
        if not result.author_id:
            return None
        key = self.store.ensure_author(
            "pixiv",
            result.author_id,
            name=result.author_name,
            handle=result.author_handle,
            profile_url=result.author_profile_url,
            avatar_url=result.author_avatar_url,
        )
        if not self.store.author_profile_fresh(key):
            try:
                with self.pixiv_limiter.slot(
                    stop_event=self.stop_event, pause_event=self.pause_event
                ):
                    profile = self.pixiv.fetch_author(result.author_id)
            except (RateLimitError, BudgetExhausted):
                raise
            except FetchError:
                profile = {}
            if profile:
                key = self.store.ensure_author(
                    "pixiv",
                    result.author_id,
                    name=profile.get("name") or result.author_name,
                    handle=profile.get("handle") or result.author_handle,
                    profile_url=profile.get("profile_url") or result.author_profile_url,
                    avatar_url=profile.get("avatar_url") or result.author_avatar_url,
                    social=profile.get("social") or {},
                    profile_fetched=True,
                )
                twitter_url = _twitter_social_url(profile.get("social"))
                handle = twitter_handle_from_url(twitter_url or "")
                if handle:
                    twitter_key = self.store.ensure_author(
                        "twitter", handle, handle=handle
                    )
                    self.store.link_authors(
                        twitter_key, key, evidence="pixiv_social", confidence=0.95
                    )
        return key

    def _resolve_twitter_author(self, result: FetchResult) -> str | None:
        if not result.author_handle:
            return None
        key = self.store.ensure_author(
            "twitter",
            result.author_handle,
            name=result.author_name,
            handle=result.author_handle,
            profile_url=result.author_profile_url,
            avatar_url=result.author_avatar_url,
        )
        pixiv_id = result.author_social.get("pixiv_user_id")
        if pixiv_id:
            pixiv_key = self.store.ensure_author("pixiv", str(pixiv_id))
            self.store.link_authors(pixiv_key, key, evidence="twitter_bio", confidence=0.7)
        return key

    def enrich_records(
        self,
        records: Iterable[dict[str, Any]],
        *,
        progress: ProgressCallback | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        rows = list(records)
        if limit is not None:
            rows = rows[: max(0, int(limit))]
        summary: dict[str, Any] = {
            "total": len(rows),
            "ok": 0,
            "failed": 0,
            "stopped": False,
        }
        if not rows:
            return summary
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in rows:
            groups.setdefault(str(record.get("platform") or ""), []).append(record)
        lock = threading.Lock()
        done = [0]

        def report(record: dict[str, Any]) -> None:
            with lock:
                done[0] += 1
                current = done[0]
                if progress is not None:
                    label = str(record.get("title") or record.get("source_id") or "")
                    progress(current, len(rows), label)

        def handle(record: dict[str, Any], abort: threading.Event) -> None:
            if self.stop_event.is_set() or abort.is_set():
                return
            while self.pause_event.is_set():
                if self.stop_event.is_set():
                    return
                time.sleep(0.25)
            try:
                self.enrich_record(record)
                with lock:
                    summary["ok"] += 1
            except BudgetExhausted:
                with lock:
                    summary["stopped"] = True
                abort.set()
            except RateLimitError as exc:
                seconds = exc.retry_after or 30 * 60
                self._block_for(record.get("platform"), seconds, str(exc))
                self.store.set_status(record["point_id"], "rate_limited", str(exc))
                with lock:
                    summary["failed"] += 1
                    summary["stopped"] = True
                abort.set()
            except ProtectedError as exc:
                self.store.set_status(record["point_id"], "protected", str(exc))
                with lock:
                    summary["failed"] += 1
            except NotFoundError as exc:
                self.store.set_status(record["point_id"], "not_found", str(exc))
                with lock:
                    summary["failed"] += 1
            except FetchError as exc:
                self.store.set_status(record["point_id"], exc.status, str(exc))
                with lock:
                    summary["failed"] += 1
            except Exception as exc:  # keep one bad record from killing the job
                self.store.set_status(record["point_id"], "error", str(exc))
                with lock:
                    summary["failed"] += 1
            report(record)

        def run_group(platform: str, group: list[dict[str, Any]]) -> None:
            concurrency = 1
            if platform == "pixiv":
                concurrency = self.pixiv_limiter.concurrency
            elif platform == "twitter":
                concurrency = self.twitter_limiter.concurrency
            abort = threading.Event()
            queue: Queue[dict[str, Any]] = Queue()
            for record in group:
                queue.put(record)

            def worker() -> None:
                while not self.stop_event.is_set() and not abort.is_set():
                    try:
                        record = queue.get_nowait()
                    except Empty:
                        return
                    handle(record, abort)

            threads = [
                threading.Thread(target=worker, daemon=True)
                for _ in range(max(1, concurrency))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        threads = [
            threading.Thread(
                target=run_group,
                args=(platform, group),
                daemon=True,
                name=f"artseek-provenance-{platform or 'other'}",
            )
            for platform, group in groups.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if self.stop_event.is_set():
            summary["stopped"] = True
        return summary

    def _block_for(self, platform: Any, seconds: int, error: str) -> None:
        if platform == "pixiv":
            self.pixiv_limiter.block(seconds, error)
        elif platform == "twitter":
            self.twitter_limiter.block(seconds, error)

    def close(self) -> None:
        self.pixiv.close()
        self.twitter.close()


def _retry_after(response: httpx.Response) -> int | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(1, int(float(value)))
    except ValueError:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _twitter_social_url(social: Any) -> str | None:
    if not isinstance(social, dict):
        return None
    twitter = social.get("twitter")
    if isinstance(twitter, dict):
        return _str_or_none(twitter.get("url"))
    if isinstance(twitter, str):
        return twitter
    return None
