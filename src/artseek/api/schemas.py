"""Pydantic request schemas and form-parsing helpers for ArtSeek HTTP routes."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..collections import COLLECTION_SORT_KEYS
from ..config import (
    DEFAULT_WEB_SEARCH_LIMIT,
    MAX_WEB_SEARCH_LIMIT,
    MIN_WEB_SEARCH_LIMIT,
    WEB_SEARCH_LIMIT_STEP,
)
from ..library import DEFAULT_SORT, SORT_KEYS
from ..services.common import ServiceError


def parse_id_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServiceError(400, "exclude_ids 不是合法 JSON") from exc
    if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
        raise ServiceError(400, "exclude_ids 必须是字符串数组")
    return loaded


def parse_weight_list(raw: str, expected: int) -> list[float]:
    if not raw.strip():
        return [1.0] * expected
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServiceError(400, "file_weights 不是合法 JSON") from exc
    if not isinstance(loaded, list) or not all(isinstance(item, (int, float)) for item in loaded):
        raise ServiceError(400, "file_weights 必须是数字数组")
    weights = [max(0.0, min(float(item), 8.0)) for item in loaded]
    if len(weights) < expected:
        weights.extend([1.0] * (expected - len(weights)))
    return weights[:expected]


def parse_folder_keys(raw: str) -> list[str] | None:
    if not raw.strip():
        return None
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServiceError(400, "folder_keys 不是合法 JSON") from exc
    if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
        raise ServiceError(400, "folder_keys 必须是字符串数组")
    return loaded


def parse_reference_specs(raw: str) -> list[ReferenceSpec]:
    if not raw.strip():
        return []
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServiceError(400, "references 不是合法 JSON") from exc
    if not isinstance(loaded, list):
        raise ServiceError(400, "references 必须是数组")
    try:
        return [ReferenceSpec.model_validate(item) for item in loaded]
    except Exception as exc:
        raise ServiceError(400, "references 格式无效") from exc


class ReferenceSpec(BaseModel):
    id: str
    weight: float = Field(default=1.0, ge=0, le=8)


class TextSearchRequest(BaseModel):
    query: str = ""
    text_weight: float = Field(default=1.0, ge=0, le=8)
    reference_id: str | None = None
    references: list[ReferenceSpec] = Field(default_factory=list)
    limit: int = Field(default=40, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=4000)
    expand: bool = True
    wait_for_llm: bool = False
    strict_tags: bool = False
    folder_keys: list[str] | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    exclude_ids: list[str] = Field(default_factory=list)
    dedupe: bool = False
    snapshot_id: str | None = None
    author_key: str | None = None
    author_keys: list[str] = Field(default_factory=list)
    title_query: str = ""
    text_label: str = ""
    tag_expression: str = ""
    sort: str = "mtime_desc"
    seed: int | None = None
    text_tags: list[str] = Field(default_factory=list)
    tag_text: list[str] = Field(default_factory=list)
    image_tags: list[str] = Field(default_factory=list)
    media_types: list[str] = Field(default_factory=list)
    ratings: list[str] = Field(default_factory=list)
    monochrome: bool | None = None
    exclude_ai: bool | None = None


class RandomBrowseRequest(BaseModel):
    """A stable random ordering for one paginated gallery session."""

    folder_keys: list[str] | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    seed: int | None = None
    limit: int = Field(default=40, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=50_000)
    author_key: str | None = None
    ratings: list[str] = Field(default_factory=list)


class FolderScopeRequest(BaseModel):
    sort: str = DEFAULT_SORT
    selected: list[str] = Field(default_factory=list)
    mtime_after: str | None = None
    mtime_before: str | None = None

    @field_validator("sort", mode="before")
    @classmethod
    def coerce_sort(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text if text in SORT_KEYS else DEFAULT_SORT

    @field_validator("selected", mode="before")
    @classmethod
    def coerce_selected(cls, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if not isinstance(value, list):
            return []
        keys: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                item = item.get("key") or item.get("path") or ""
            text = str(item).strip()
            if text:
                keys.append(text)
        return keys

    @field_validator("mtime_after", "mtime_before", mode="before")
    @classmethod
    def coerce_day(cls, value: Any) -> str | None:
        if value is None or value == "":
            return None
        return str(value)


class ScopeImagesRequest(BaseModel):
    folder_keys: list[str] = Field(default_factory=list)
    sort: str = DEFAULT_SORT
    seed: int | None = None
    limit: int = Field(default=40, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=50_000)
    author_key: str | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    ratings: list[str] = Field(default_factory=list)

    @field_validator("sort", mode="before")
    @classmethod
    def coerce_sort(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text if text in SORT_KEYS or text == "random" else DEFAULT_SORT

    @field_validator("folder_keys", mode="before")
    @classmethod
    def coerce_folder_keys(cls, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if item is not None and str(item).strip()]

    @field_validator("mtime_after", "mtime_before", mode="before")
    @classmethod
    def coerce_day(cls, value: Any) -> str | None:
        if value is None or value == "":
            return None
        return str(value)


class SuggestionRequest(BaseModel):
    q: str = ""
    limit: int = Field(default=200, ge=1, le=500)
    offset: int = Field(default=0, ge=0, le=50_000)
    folder_keys: list[str] | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    kind: str = "source"


class ArtistScopeRequest(BaseModel):
    folder_keys: list[str] | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    author_keys: list[str] = Field(default_factory=list)
    title_query: str = ""
    text_tags: list[str] = Field(default_factory=list)
    tag_expression: str = ""
    collection_id: str = ""
    snapshot_id: str = ""
    ratings: list[str] = Field(default_factory=list)


class ArtistGalleryRequest(ArtistScopeRequest):
    q: str = ""
    sort: str = "latest_desc"
    include_unassigned: bool = False
    offset: int = Field(default=0, ge=0, le=50_000)
    limit: int = Field(default=30, ge=1, le=100)
    image_sort: str = "mtime_desc"
    image_limit: int = Field(default=20, ge=1, le=60)
    image_seed: int | None = None


class ArtistImagesRequest(ArtistScopeRequest):
    sort: str = "mtime_desc"
    seed: int | None = None
    offset: int = Field(default=0, ge=0, le=50_000)
    limit: int = Field(default=20, ge=1, le=60)


class ArtistIdentityRequest(BaseModel):
    kind: str = "link"
    value: str = ""
    label: str = ""


class ArtistIdentityRemoveRequest(BaseModel):
    kind: str = "link"
    value: str = ""


class ArtistNameRequest(BaseModel):
    name: str = ""


class ArtistCoverRequest(BaseModel):
    point_id: str = ""


class ArtistMergeRequest(BaseModel):
    source: str = ""


class IndexAddRequest(BaseModel):
    path: str = ""
    paths: list[str] = Field(default_factory=list)
    recursive: bool = True


class IndexRemoveRequest(BaseModel):
    folder_key: str


class ModelControlRequest(BaseModel):
    loaded: bool | None = None
    auto_unload_minutes: int | None = Field(default=None, ge=0, le=1440)


class WebSearchSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_llm: bool = True
    strict_tags: bool = False
    dedupe: bool = True
    limit: int = Field(
        default=DEFAULT_WEB_SEARCH_LIMIT,
        ge=MIN_WEB_SEARCH_LIMIT,
        le=MAX_WEB_SEARCH_LIMIT,
    )

    @field_validator("limit")
    @classmethod
    def validate_limit_step(cls, value: int) -> int:
        if (value - MIN_WEB_SEARCH_LIMIT) % WEB_SEARCH_LIMIT_STEP:
            raise ValueError(f"每批数量必须以 {WEB_SEARCH_LIMIT_STEP} 为步长")
        return value


class CollectionNameRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = value.strip()
        if len(name) > 80:
            raise ValueError("收藏夹名称不能超过 80 个字符")
        return name


class CollectionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    sort: str | None = None
    seed_sort: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        name = value.strip()
        if len(name) > 80:
            raise ValueError("收藏夹名称不能超过 80 个字符")
        return name

    @field_validator("sort", "seed_sort")
    @classmethod
    def validate_sort(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in COLLECTION_SORT_KEYS:
            raise ValueError("未知的收藏夹排序方式")
        return value

    @model_validator(mode="after")
    def validate_update(self) -> CollectionUpdateRequest:
        if self.name is None and self.sort is None:
            raise ValueError("请提供新的收藏夹名称或排序方式")
        return self


class CollectionItemOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    before_id: str | None = None
    after_id: str | None = None

    @field_validator("image_id")
    @classmethod
    def validate_image_id(cls, value: str) -> str:
        image_id = value.strip()
        if not image_id:
            raise ValueError("图片 ID 不能为空")
        return image_id

    @model_validator(mode="after")
    def validate_anchor(self) -> CollectionItemOrderRequest:
        if self.before_id is not None and self.after_id is not None:
            raise ValueError("before_id 与 after_id 只能提供一个")
        return self


class CollectionOrderRequest(BaseModel):
    collection_ids: list[str]

    @field_validator("collection_ids")
    @classmethod
    def validate_collection_ids(cls, value: list[str]) -> list[str]:
        if any(not collection_id.strip() for collection_id in value):
            raise ValueError("收藏夹 ID 不能为空")
        if len(set(value)) != len(value):
            raise ValueError("收藏夹 ID 不能重复")
        return value


class CollectionItemRequest(BaseModel):
    image_id: str

    @field_validator("image_id")
    @classmethod
    def validate_image_id(cls, value: str) -> str:
        point_id = value.strip()
        if not point_id:
            raise ValueError("请选择要收藏的图片")
        return point_id


class StoragePathsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qdrant_path: str | None = None
    notes_dir: str | None = None
    cache_dir: str | None = None
    index_dir: str | None = None


class StorageMigrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["qdrant", "notes", "cache", "index"]
    destination: str


class NoteCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    content: str = ""


class NoteWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def validate_write(self) -> NoteWriteRequest:
        if self.title is None and self.content is None:
            raise ValueError("请提供笔记标题或内容")
        return self


class NoteOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str]

    @field_validator("names")
    @classmethod
    def validate_names(cls, value: list[str]) -> list[str]:
        if any(not str(name).strip() for name in value):
            raise ValueError("笔记名称不能为空")
        return value


class ProvenanceScanRequest(BaseModel):
    deep: bool = False
    only_new: bool = False
    auto_enrich: bool | None = None


class AutoTagRequest(BaseModel):
    limit: int | None = Field(default=500, ge=1, le=100_000)


class ProvenanceEnrichRequest(BaseModel):
    platforms: list[Literal["pixiv", "twitter"]] = Field(
        default_factory=lambda: ["pixiv", "twitter"]
    )
    limit: int | None = Field(default=None, ge=1, le=100_000)
    point_ids: list[str] = Field(default_factory=list)


class ProvenanceUrlRequest(BaseModel):
    url: str = Field(default="", max_length=2000)


class ProvenanceManualRequest(BaseModel):
    url: str | None = Field(default=None, max_length=2000)
    title: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=20_000)
    author_name: str | None = Field(default=None, max_length=200)
    author_handle: str | None = Field(default=None, max_length=100)
    pixiv_id: str | None = Field(default=None, max_length=200)
    x_handle: str | None = Field(default=None, max_length=200)
    author_key: str | None = Field(default=None, max_length=200)
    author_query: str | None = Field(default=None, max_length=200)
    tags: list[str] | None = None
    manual_tags: list[str] | None = None
    clear_manual: bool = False


class RetagJobRequest(BaseModel):
    backend: str | None = None
    folder_keys: list[str] | None = None
    skip_completed: bool = True
    limit: int | None = Field(default=None, ge=1, le=200_000)


class VisionTaskRequest(BaseModel):
    folder_keys: list[str] | None = None
    skip_completed: bool = True
    limit: int | None = Field(default=None, ge=1, le=200_000)


class CcipTaskRequest(VisionTaskRequest):
    """CCIP needs `character_crop_json`; set this to accept whole-image fallback."""

    allow_incomplete_vision: bool = False


class CcipResetRequest(BaseModel):
    folder_keys: list[str] | None = None


class SingleRetagRequest(BaseModel):
    backend: str | None = None


class AutoFillQueueRequest(BaseModel):
    backend: str | None = None
    folder_keys: list[str] | None = None
    include_provenance: bool = False


class JobMoveRequest(BaseModel):
    direction: Literal["up", "down", "top", "bottom"]


class UrlIgnoredRequest(BaseModel):
    ignored: bool = True


class CharacterSearchRequest(BaseModel):
    reference_id: str | None = None
    bbox: list[float] | None = None
    folder_keys: list[str] | None = None
    mtime_after: str | None = None
    mtime_before: str | None = None
    media_types: list[str] = Field(default_factory=list)
    ratings: list[str] = Field(default_factory=list)
    monochrome: bool | None = None
    exclude_ai: bool | None = None
    dedupe: bool = False
    limit: int = Field(default=40, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=4000)
