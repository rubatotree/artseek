from __future__ import annotations

import json
import os
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
    model: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    threshold: float = 0.35
    max_tags: int = 80
    batch_size: int = 8
    include_character_tags: bool = False


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
class AppConfig:
    data_dir: Path = field(default_factory=lambda: Path(user_data_dir(APP_NAME)))
    cache_dir: Path = field(default_factory=lambda: Path(user_cache_dir(APP_NAME)))
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    tagger: TaggerConfig = field(default_factory=TaggerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def index_state_path(self) -> Path:
        return self.data_dir / "index_state.sqlite3"

    @property
    def thumbnail_dir(self) -> Path:
        return self.cache_dir / "thumbnails"

    @property
    def folder_scope_path(self) -> Path:
        return self.data_dir / "folder_scope.json"

    @property
    def folder_catalog_path(self) -> Path:
        return self.data_dir / "folder_catalog.json"

    @property
    def runtime_path(self) -> Path:
        return self.data_dir / "runtime.json"

    @property
    def collections_path(self) -> Path:
        return self.data_dir / "collections.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)


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

[embedding]
# 轻量而且支持中文/日语/韩语的图文向量模型。
model = "google/siglip2-base-patch16-256"
batch_size = 16
device = "auto"       # auto / cuda / cpu
dtype = "auto"        # auto / float16 / bfloat16 / float32

[tagger]
# 面向二次元插画的 WD EVA02 标签器，能补足服装、动作、构图细节。
enabled = true
model = "SmilingWolf/wd-eva02-large-tagger-v3"
threshold = 0.35
max_tags = 80
batch_size = 8
include_character_tags = false

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


def load_config(path: Path | None = None) -> AppConfig:
    path = path or default_config_path()
    cfg = AppConfig()
    if path.exists():
        with path.open("rb") as handle:
            _merge_dataclass(cfg, tomllib.load(handle))
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
