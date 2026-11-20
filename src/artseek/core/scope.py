"""Pure immutable image scope descriptor used across gallery, search, and artists."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageScope:
    """Unified scope specification resolved by `ScopeResolver`.

    - `folders is None`: no folder filter; `folders == ()`: explicit empty selection (0 images).
    - `folder_recursive=True`: includes all descendant folders under each folder in `folders`.
    - `time_range`: inclusive `(after_ns, before_ns)` bounds in nanoseconds; either bound
      may be None.
    - `author_keys`: union across listed author keys (supports `"none"` pseudo-author).
    """

    folders: tuple[str, ...] | None = None
    folder_recursive: bool = False
    time_range: tuple[int | None, int | None] | None = None
    collection_id: str | None = None
    author_keys: tuple[str, ...] = ()
    tags_all: tuple[str, ...] = ()
    tags_none: tuple[str, ...] = ()
    source_tags_all: tuple[str, ...] = ()
    source_tags_any: tuple[str, ...] = ()
    source_tag_text: tuple[str, ...] = ()
    tag_expression: str | None = None
    title_terms: tuple[str, ...] = ()
    media_types: tuple[str, ...] = ()
    ratings: tuple[str, ...] = ()
    monochrome: bool | None = None
    exclude_ai: bool | None = None
    shot_types: tuple[str, ...] = ()
    snapshot_ids: tuple[str, ...] | None = None
    restrict_point_ids: tuple[str, ...] | None = None

    @property
    def is_unfiltered(self) -> bool:
        return (
            self.folders is None
            and (
                self.time_range is None
                or (self.time_range[0] is None and self.time_range[1] is None)
            )
            and self.collection_id is None
            and not self.author_keys
            and not self.tags_all
            and not self.tags_none
            and not self.source_tags_all
            and not self.source_tags_any
            and not self.source_tag_text
            and not (self.tag_expression or "").strip()
            and not self.title_terms
            and not self.media_types
            and not self.ratings
            and self.monochrome is None
            and self.exclude_ai is None
            and not self.shot_types
            and self.snapshot_ids is None
            and self.restrict_point_ids is None
        )
