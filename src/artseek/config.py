from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_config_dir, user_data_dir

APP_NAME = "ArtSeek"
DEFAULT_AUTO_UNLOAD_SECONDS = 600
DEFAULT_FULL_VERIFICATION_SECONDS = 24 * 60 * 60
DEFAULT_WEB_SEARCH_LIMIT = 40
MIN_WEB_SEARCH_LIMIT = 12
MAX_WEB_SEARCH_LIMIT = 80
WEB_SEARCH_LIMIT_STEP = 4
INDEX_STATE_NAME = "index_state.sqlite3"
PROVENANCE_NAME = "provenance.sqlite3"
DB_NAME = "artseek.sqlite3"
VISUAL_VECTORS_NAME = "visual.f32.npy"
DESCRIPTOR_VECTORS_NAME = "descriptor.f32.npy"
CCIP_VECTORS_NAME = "ccip.f32.npy"
INDEX_SIDECAR_FILES = (
    DB_NAME,
    VISUAL_VECTORS_NAME,
    DESCRIPTOR_VECTORS_NAME,
    CCIP_VECTORS_NAME,
    INDEX_STATE_NAME,
    PROVENANCE_NAME,
    "folder_catalog.json",
    "folder_scope.json",
    "collections.json",
    "manifest.json",
)
INDEX_SQLITE_EXTRAS = (
    f"{DB_NAME}-wal",
    f"{DB_NAME}-shm",
    f"{DB_NAME}-journal",
    f"{INDEX_STATE_NAME}-wal",
    f"{INDEX_STATE_NAME}-shm",
    f"{INDEX_STATE_NAME}-journal",
    f"{PROVENANCE_NAME}-wal",
    f"{PROVENANCE_NAME}-shm",
    f"{PROVENANCE_NAME}-journal",
)
_RUNTIME_SETTINGS_LOCK = threading.RLock()


@dataclass(slots=True)
class EmbeddingConfig:
    model: str = "google/siglip2-base-patch16-256"
    batch_size: int = 16
    device: str = "auto"
    dtype: str = "auto"


@dataclass(slots=True)
class TaggerConfig:
    enabled: bool = True
    backend: str = "camie+pixai0.9"
    model: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    camie_model: str = "Camais03/camie-tagger-v2"
    pixai09_model: str = "deepghs/pixai-tagger-v0.9-onnx"
    pixai10_model: str = "pixai-labs/pixai-tagger-v1.0"
    threshold: float = 0.35
    general_threshold: float = 0.35
    character_threshold: float = 0.65
    copyright_threshold: float = 0.60
    meta_threshold: float = 0.45
    max_tags: int = 80
    batch_size: int = 16
    include_character_tags: bool = True
    mcut_enabled: bool = True
    drop_overlap: bool = True


@dataclass(slots=True)
class LLMConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    query_model: str = "qwen3:4b"
    vision_model: str = "qwen3-vl:4b"
    timeout_seconds: int = 120
    caption_on_index: bool = False


@dataclass(slots=True)
class SearchConfig:
    visual_weight: float = 0.72
    descriptor_weight: float = 0.28
    keyword_bonus: float = 0.08
    candidate_multiplier: int = 8
    default_limit: int = 40


@dataclass(slots=True)
class IndexingConfig:
    """Safety settings for the local incremental-index cache."""

    full_verification_interval_seconds: int = DEFAULT_FULL_VERIFICATION_SECONDS


@dataclass(slots=True)
class ProvenanceConfig:
    """Rate and automation policy for source metadata enrichment."""

    auto_enrich: bool = True
    pixiv_min_interval: float = 0.75
    twitter_min_interval: float = 0.2
    pixiv_concurrency: int = 1
    twitter_concurrency: int = 4
    pixiv_daily_limit: int = 0
    twitter_daily_limit: int = 0
    author_ttl_days: int = 30


@dataclass(slots=True)
class AppConfig:
    data_dir: Path = field(default_factory=lambda: Path(user_data_dir(APP_NAME)))
    cache_dir: Path = field(default_factory=lambda: Path(user_cache_dir(APP_NAME)))
    qdrant_path_override: Path | None = None
    notes_dir_override: Path | None = None
    index_dir_override: Path | None = None
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    tagger: TaggerConfig = field(default_factory=TaggerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)

    @property
    def qdrant_path(self) -> Path:
        return self.qdrant_path_override or (self.data_dir / "qdrant")

    @property
    def notes_dir(self) -> Path:
        return self.notes_dir_override or (self.data_dir / "notes")

    @property
    def index_dir(self) -> Path:
        return self.index_dir_override or self.data_dir

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def index_state_path(self) -> Path:
        return self.index_dir / INDEX_STATE_NAME

    @property
    def thumbnail_dir(self) -> Path:
        return self.cache_dir / "thumbnails"

    @property
    def search_snapshot_dir(self) -> Path:
        return self.cache_dir / "search-snapshots"

    @property
    def folder_scope_path(self) -> Path:
        return self.index_dir / "folder_scope.json"

    @property
    def folder_catalog_path(self) -> Path:
        return self.index_dir / "folder_catalog.json"

    @property
    def runtime_path(self) -> Path:
        return self.data_dir / "runtime.json"

    @property
    def collections_path(self) -> Path:
        return self.index_dir / "collections.json"

    @property
    def provenance_path(self) -> Path:
        return self.index_dir / PROVENANCE_NAME

    @property
    def db_path(self) -> Path:
        return self.index_dir / DB_NAME

    @property
    def visual_vectors_path(self) -> Path:
        return self.index_dir / VISUAL_VECTORS_NAME

    @property
    def descriptor_vectors_path(self) -> Path:
        return self.index_dir / DESCRIPTOR_VECTORS_NAME

    @property
    def ccip_vectors_path(self) -> Path:
        return self.index_dir / CCIP_VECTORS_NAME

    @property
    def artist_assets_dir(self) -> Path:
        return self.index_dir / "artist-assets"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.search_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class WebSearchSettings:
    """User-owned query defaults shared by every local web client."""

    use_llm: bool = True
    strict_tags: bool = False
    dedupe: bool = True
    limit: int = DEFAULT_WEB_SEARCH_LIMIT


def default_config_path() -> Path:
    override = os.environ.get("ARTSEEK_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path(user_config_dir(APP_NAME)) / "config.toml"


def default_config_text() -> str:
    data = Path(user_data_dir(APP_NAME)).as_posix()
    cache = Path(user_cache_dir(APP_NAME)).as_posix()
    return f'''# ArtSeek 本地配置。Windows 路径建议写成 D:/Pictures 这种形式。
data_dir = "{data}"
cache_dir = "{cache}"
# 可选。默认是 data_dir 下的 qdrant 与 notes。网页「高级」改路径写入 runtime.json，并优先于这里。
# qdrant_path = "D:/ArtSeek/qdrant"
# notes_dir = "D:/ArtSeek/notes"
# index_dir = "D:/ArtSeek/index"
# cache_dir 也可在网页里改。索引数据库默认与 data_dir 相同，
# 含增量 SQLite、目录树、勾选范围、收藏夹与 manifest。

[embedding]
# 轻量而且支持中文/日语/韩语的图文向量模型。
model = "google/siglip2-base-patch16-256"
batch_size = 16
device = "auto"       # auto / cuda / cpu
dtype = "auto"        # auto / float16 / bfloat16 / float32

[tagger]
# 默认启用 Camie v2 + PixAI v0.9 融合标签器（支持 IP / 角色 / 分级 / 风格与细节），
# 可选后端：camie+pixai0.9 | camie | pixai0.9 | pixai1.0 | wd
enabled = true
backend = "camie+pixai0.9"
model = "SmilingWolf/wd-eva02-large-tagger-v3"
camie_model = "Camais03/camie-tagger-v2"
pixai09_model = "deepghs/pixai-tagger-v0.9-onnx"
pixai10_model = "pixai-labs/pixai-tagger-v1.0"
threshold = 0.35
general_threshold = 0.35
character_threshold = 0.65
copyright_threshold = 0.60
meta_threshold = 0.45
max_tags = 80
batch_size = 8
include_character_tags = true
mcut_enabled = true
drop_overlap = true

[llm]
# Ollama 不存在或未启动时会自动回退到内置中英日服装/动作词典。
enabled = true
base_url = "http://127.0.0.1:11434"
query_model = "qwen3:4b"
vision_model = "qwen3-vl:4b"
timeout_seconds = 120
# 给每张图生成结构化长描述更准，但速度慢；首次使用建议保持 false。
caption_on_index = false

[search]
visual_weight = 0.72
descriptor_weight = 0.28
keyword_bonus = 0.08
candidate_multiplier = 8
default_limit = 40

[indexing]
# 目录 mtime 无法可靠捕捉文件原地改写，故定期完整扫描；仍只比较文件 mtime/大小。
# 同大小且人为恢复 mtime 的内容改写无法在不逐文件哈希的前提下被检测；0 表示每次校验。
full_verification_interval_seconds = 86400

[provenance]
# 索引完成后自动为能识别的 Pixiv/X 文件名补全来源；不自动调用反向搜图。
auto_enrich = true
# 按域名限速；Pixiv 默认单并发，FxTwitter 默认 4 并发。
pixiv_min_interval = 0.75
twitter_min_interval = 0.2
pixiv_concurrency = 1
twitter_concurrency = 4
# 0 表示不设每日请求上限，仅保留每个域名的请求间隔与并发数。
pixiv_daily_limit = 0
twitter_daily_limit = 0
author_ttl_days = 30
'''


def write_default_config(path: Path | None = None, *, force: bool = False) -> Path:
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    path.write_text(default_config_text(), encoding="utf-8")
    return path


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif isinstance(current, Path):
            setattr(instance, key, Path(value).expanduser())
        else:
            setattr(instance, key, value)


def coerce_directory_path(value: Any, *, create: bool = False) -> Path:
    """Expand, make absolute, and optionally create a user-supplied directory."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("路径不能为空")
    text = str(value).strip()
    if "\x00" in text:
        raise ValueError("路径不合法")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        path = path.resolve()
    except OSError as exc:
        raise ValueError(f"无法解析路径：{exc}") from exc
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"无法使用该目录：{exc}") from exc
    return path


def paths_equal(left: Path, right: Path) -> bool:
    try:
        return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))
    except OSError:
        return os.path.normcase(str(left)) == os.path.normcase(str(right))


_STORAGE_LOCK_NAMES = frozenset({".lock"})


def _resolved_or_absolute(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path if path.is_absolute() else Path.cwd() / path


def paths_overlap(left: Path, right: Path) -> bool:
    """True when either path is the other, or one directory contains the other."""
    left_resolved = _resolved_or_absolute(left)
    right_resolved = _resolved_or_absolute(right)
    if paths_equal(left_resolved, right_resolved):
        return True
    return left_resolved.is_relative_to(right_resolved) or right_resolved.is_relative_to(
        left_resolved
    )


def storage_visible_entries(path: Path) -> list[Path]:
    """Directory children excluding Qdrant lock files."""
    if not path.exists():
        return []
    if not path.is_dir():
        raise ValueError("路径不是目录")
    return [item for item in path.iterdir() if item.name not in _STORAGE_LOCK_NAMES]


def _ignore_storage_locks(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _STORAGE_LOCK_NAMES}


def migrate_storage_tree(source: Path, destination: Path | str) -> tuple[Path, bool]:
    """Copy a vector or notes directory to an empty destination. Source is kept.

    Returns `(destination, copied)`. `copied` is False when the source had no
    visible files, in which case only the destination directory is created.
    """
    dest = coerce_directory_path(destination, create=False)
    if dest.exists() and not dest.is_dir():
        raise ValueError("目标路径不是目录")
    if source.exists() and not source.is_dir():
        raise ValueError("当前路径不是目录")
    if paths_equal(source, dest):
        raise ValueError("目标路径与当前路径相同")
    if paths_overlap(source, dest):
        raise ValueError("不能迁到当前目录内部或其上级目录")
    if storage_visible_entries(dest):
        raise ValueError("目标目录不是空目录，请另选路径以免覆盖已有库")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not storage_visible_entries(source):
        dest.mkdir(parents=True, exist_ok=True)
        return dest, False
    try:
        shutil.copytree(
            source,
            dest,
            dirs_exist_ok=dest.is_dir(),
            ignore=_ignore_storage_locks,
        )
    except OSError as exc:
        raise ValueError(f"复制失败：{exc}") from exc
    return dest, True


def _checkpoint_sqlite(path: Path) -> None:
    if not path.is_file():
        return
    try:
        connection = sqlite3.connect(str(path))
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()
    except sqlite3.Error:
        return


def migrate_index_sidecars(source_dir: Path, destination: Path | str) -> tuple[Path, bool]:
    """Copy index sidecars into an empty-of-index destination. Source is kept.

    Includes the incremental SQLite file (and WAL extras), folder catalog,
    selected-folder scope, collections, and the embedding manifest. Does not
    copy Qdrant, notes, cache, runtime.json, or config.toml.
    """
    dest = coerce_directory_path(destination, create=False)
    if dest.exists() and not dest.is_dir():
        raise ValueError("目标路径不是目录")
    if source_dir.exists() and not source_dir.is_dir():
        raise ValueError("当前路径不是目录")
    if paths_equal(source_dir, dest):
        raise ValueError("目标路径与当前路径相同")
    if paths_overlap(source_dir, dest):
        raise ValueError("不能迁到当前目录内部或其上级目录")
    dest.mkdir(parents=True, exist_ok=True)
    occupied = [name for name in INDEX_SIDECAR_FILES if (dest / name).exists()]
    if occupied:
        raise ValueError("目标目录已有索引文件，请另选路径以免覆盖")
    copied = False
    try:
        for name in (*INDEX_SIDECAR_FILES, *INDEX_SQLITE_EXTRAS):
            src = source_dir / name
            if not src.is_file():
                continue
            shutil.copy2(src, dest / name)
            copied = True
    except OSError as exc:
        raise ValueError(f"复制失败：{exc}") from exc
    _checkpoint_sqlite(dest / DB_NAME)
    _checkpoint_sqlite(dest / INDEX_STATE_NAME)
    _checkpoint_sqlite(dest / PROVENANCE_NAME)
    return dest, copied


def _optional_directory_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return coerce_directory_path(value, create=False)
    except ValueError:
        return None


def apply_runtime_paths(config: AppConfig) -> None:
    """Runtime path overrides win over config.toml so CLI and web share locations."""
    data = _load_runtime_data(config)
    if "qdrant_path" in data:
        override = _optional_directory_path(data.get("qdrant_path"))
        config.qdrant_path_override = override
    if "notes_dir" in data:
        override = _optional_directory_path(data.get("notes_dir"))
        config.notes_dir_override = override
    if "cache_dir" in data:
        override = _optional_directory_path(data.get("cache_dir"))
        if override is not None:
            config.cache_dir = override
    if "index_dir" in data:
        config.index_dir_override = _optional_directory_path(data.get("index_dir"))


def load_config(path: Path | None = None) -> AppConfig:
    path = path or default_config_path()
    cfg = AppConfig()
    if path.exists():
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        qdrant = raw.pop("qdrant_path", None)
        notes = raw.pop("notes_dir", None)
        index_dir = raw.pop("index_dir", None)
        raw.pop("index_state_path", None)
        _merge_dataclass(cfg, raw)
        if isinstance(qdrant, str) and qdrant.strip():
            cfg.qdrant_path_override = coerce_directory_path(qdrant, create=False)
        if isinstance(notes, str) and notes.strip():
            cfg.notes_dir_override = coerce_directory_path(notes, create=False)
        if isinstance(index_dir, str) and index_dir.strip():
            cfg.index_dir_override = coerce_directory_path(index_dir, create=False)
    apply_runtime_paths(cfg)
    cfg.ensure_dirs()
    return cfg


def clamp_auto_unload_seconds(value: Any, default: int = DEFAULT_AUTO_UNLOAD_SECONDS) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        seconds = default
    return max(0, min(seconds, 86400))


def clamp_web_search_limit(value: Any, default: int = DEFAULT_WEB_SEARCH_LIMIT) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = default
    limit = max(MIN_WEB_SEARCH_LIMIT, min(limit, MAX_WEB_SEARCH_LIMIT))
    return (
        MIN_WEB_SEARCH_LIMIT
        + round((limit - MIN_WEB_SEARCH_LIMIT) / WEB_SEARCH_LIMIT_STEP) * WEB_SEARCH_LIMIT_STEP
    )


def _load_runtime_data(config: AppConfig) -> dict[str, Any]:
    path = config.runtime_path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_runtime_data(config: AppConfig, payload: dict[str, Any]) -> None:
    config.ensure_dirs()
    path = config.runtime_path
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_auto_unload_seconds(config: AppConfig) -> int:
    data = _load_runtime_data(config)
    return clamp_auto_unload_seconds(data.get("auto_unload_seconds", DEFAULT_AUTO_UNLOAD_SECONDS))


def save_auto_unload_seconds(config: AppConfig, seconds: int) -> None:
    with _RUNTIME_SETTINGS_LOCK:
        payload = _load_runtime_data(config)
        payload["auto_unload_seconds"] = clamp_auto_unload_seconds(seconds)
        _save_runtime_data(config, payload)


def load_web_search_settings(config: AppConfig) -> WebSearchSettings:
    raw = _load_runtime_data(config).get("web_search_settings")
    if not isinstance(raw, dict):
        return WebSearchSettings()
    return WebSearchSettings(
        use_llm=raw["use_llm"] if isinstance(raw.get("use_llm"), bool) else True,
        strict_tags=raw["strict_tags"] if isinstance(raw.get("strict_tags"), bool) else False,
        dedupe=raw["dedupe"] if isinstance(raw.get("dedupe"), bool) else True,
        limit=clamp_web_search_limit(raw.get("limit")),
    )


def save_web_search_settings(config: AppConfig, settings: WebSearchSettings) -> None:
    with _RUNTIME_SETTINGS_LOCK:
        payload = _load_runtime_data(config)
        payload["web_search_settings"] = {
            "use_llm": bool(settings.use_llm),
            "strict_tags": bool(settings.strict_tags),
            "dedupe": bool(settings.dedupe),
            "limit": clamp_web_search_limit(settings.limit),
        }
        _save_runtime_data(config, payload)


def save_storage_paths(
    config: AppConfig,
    *,
    qdrant_path: Path | str | None = None,
    notes_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
    update_qdrant: bool = False,
    update_notes: bool = False,
    update_cache: bool = False,
    update_index: bool = False,
) -> None:
    """Persist path overrides. Empty/None clears an override back to the default."""
    with _RUNTIME_SETTINGS_LOCK:
        payload = _load_runtime_data(config)
        if update_qdrant:
            if qdrant_path in (None, ""):
                payload.pop("qdrant_path", None)
                config.qdrant_path_override = None
            else:
                path = coerce_directory_path(qdrant_path, create=False)
                payload["qdrant_path"] = str(path)
                config.qdrant_path_override = path
        if update_notes:
            if notes_dir in (None, ""):
                payload.pop("notes_dir", None)
                config.notes_dir_override = None
            else:
                path = coerce_directory_path(notes_dir, create=True)
                payload["notes_dir"] = str(path)
                config.notes_dir_override = path
        if update_cache:
            if cache_dir in (None, ""):
                payload.pop("cache_dir", None)
            else:
                path = coerce_directory_path(cache_dir, create=True)
                payload["cache_dir"] = str(path)
                config.cache_dir = path
        if update_index:
            if index_dir in (None, ""):
                payload.pop("index_dir", None)
                config.index_dir_override = None
            else:
                path = coerce_directory_path(index_dir, create=True)
                payload["index_dir"] = str(path)
                config.index_dir_override = path
        _save_runtime_data(config, payload)
