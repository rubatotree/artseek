"""Pure vector composition, weighted RRF fusion, tag filtering, and deduplication."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .query_plan import QueryPlan, normalize_tag

RankedRow = tuple[Any, dict[str, Any], list[str]]

GENERIC_VISUAL_TAGS = frozenset(
    {
        "1girl",
        "solo",
        "1boy",
        "multiple_girls",
        "2girls",
        "3girls",
        "looking_at_viewer",
        "simple_background",
        "white_background",
        "transparent_background",
        "open_mouth",
        "closed_mouth",
    }
)


def order_tags_for_descriptor(tags: Sequence[str]) -> list[str]:
    """Place distinctive tags ahead of ultra-common boilerplate for 64-token text encoders."""
    specific: list[str] = []
    generic: list[str] = []
    for tag in tags:
        clean = str(tag or "").strip()
        if not clean:
            continue
        if normalize_tag(clean) in GENERIC_VISUAL_TAGS:
            generic.append(clean)
        else:
            specific.append(clean)
    return [*specific, *generic]


def rank_tags_by_idf(
    tag_scores: Sequence[tuple[str, float]],
    *,
    total_docs: int,
    doc_counts: Mapping[str, int],
    limit: int = 12,
    demote_generic: bool = True,
) -> tuple[list[str], dict[str, float]]:
    """Select the highest-information tags using confidence * IDF and generic demotion."""
    if not tag_scores or limit <= 0:
        return [], {}
    n_docs = max(1, int(total_docs))
    weighted: list[tuple[float, float, str]] = []
    for raw_tag, raw_score in tag_scores:
        tag = str(raw_tag or "").strip()
        if not tag:
            continue
        score = float(raw_score)
        df = max(0, int(doc_counts.get(tag, 0)))
        idf = math.log((n_docs + 1.0) / (1.0 + df)) + 0.5
        penalty = (
            0.25
            if demote_generic and normalize_tag(tag) in GENERIC_VISUAL_TAGS
            else 1.0
        )
        weighted.append((score * idf * penalty, score, tag))
    weighted.sort(key=lambda item: (-item[0], -item[1], item[2]))
    chosen = weighted[: max(1, int(limit))]
    return (
        [tag for _, _, tag in chosen],
        {tag: round(score, 4) for _, score, tag in chosen},
    )


def compose_visual_vector(
    text_vector: np.ndarray | None,
    image_vectors: Sequence[np.ndarray],
    *,
    text_weight: float = 1.0,
    image_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted sum of text and reference-image vectors, then L2-normalized."""
    parts: list[np.ndarray] = []
    if text_vector is not None and text_weight > 0:
        parts.append(np.asarray(text_vector, dtype=np.float32) * float(text_weight))
    weights = list(image_weights) if image_weights is not None else [1.0] * len(image_vectors)
    if len(weights) != len(image_vectors):
        raise ValueError("image_weights 数量必须与参考图一致")
    for vector, weight in zip(image_vectors, weights, strict=True):
        if weight > 0:
            parts.append(np.asarray(vector, dtype=np.float32) * float(weight))
    if not parts:
        raise ValueError("文字与参考图的权重不能全为 0")
    combined = np.sum(np.stack(parts, axis=0), axis=0)
    combined /= max(float(np.linalg.norm(combined)), 1e-12)
    return combined.astype(np.float32, copy=False)


def duplicate_keys(payload: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    """Identities used by 「去除重复」: content hash, perceptual hash, else size."""
    keys: list[tuple[Any, ...]] = []
    digest = str(payload.get("content_hash") or "").strip().casefold()
    if digest:
        keys.append(("h", digest))
    phash = str(payload.get("phash") or "").strip().casefold()
    if phash:
        keys.append(("p", phash))
    if digest:
        return tuple(keys)
    try:
        size = int(payload.get("size_bytes") or 0)
        width = int(payload.get("width") or 0)
        height = int(payload.get("height") or 0)
    except (TypeError, ValueError):
        return tuple(keys)
    if size > 0 and width > 0 and height > 0:
        keys.append(("s", size, width, height))
    return tuple(keys)


def duplicate_key(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    """Primary duplicate identity: content hash, else perceptual hash, else size."""
    keys = duplicate_keys(payload)
    return keys[0] if keys else None


def keep_first_unique(
    ranked: Sequence[RankedRow],
    seen_keys: set[tuple[Any, ...]] | None = None,
) -> list[RankedRow]:
    seen: set[tuple[Any, ...]] = set(seen_keys or ())
    unique: list[RankedRow] = []
    for row in ranked:
        keys = duplicate_keys(row[1]["payload"])
        if keys and any(key in seen for key in keys):
            continue
        seen.update(keys)
        unique.append(row)
    return unique


def fuse_and_rank(
    visual_hits: Sequence[tuple[Any, float]],
    descriptor_hits: Sequence[tuple[Any, float]],
    *,
    payloads: dict[Any, dict[str, Any]],
    plan: QueryPlan,
    descriptor_weight: float,
    keyword_bonus: float,
    strict_tags: bool = False,
    dedupe: bool = False,
    excluded_ids: set[Any] | None = None,
    excluded_payloads: Sequence[dict[str, Any]] = (),
    limit: int = 40,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], bool]:
    """Fuse visual + descriptor hits with weighted RRF, tag bonus/exclusion, and dedupe."""
    combined: dict[Any, dict[str, Any]] = {}
    desc_w = float(descriptor_weight) if descriptor_hits else 0.0
    vis_w = 1.0 - desc_w

    for channel, hits, weight in (
        ("visual", visual_hits, vis_w),
        ("descriptor", descriptor_hits, desc_w),
    ):
        for rank, (item_key, score) in enumerate(hits, start=1):
            payload = payloads.get(item_key)
            if payload is None:
                continue
            entry = combined.setdefault(
                item_key,
                {"payload": payload, "rrf": 0.0, "raw": {}},
            )
            entry["rrf"] += weight / (60.0 + rank)
            entry["raw"][channel] = float(score)

    positive = {normalize_tag(tag) for tag in plan.positive_tags}
    negative = {normalize_tag(tag) for tag in plan.negative_tags}
    ranked: list[RankedRow] = []
    for item_key, entry in combined.items():
        payload = entry["payload"]
        tags = {normalize_tag(tag) for tag in payload.get("tags", [])}
        if tags & negative:
            continue
        hits = sorted(tags & positive)
        if strict_tags and positive and not hits:
            continue
        if positive:
            entry["rrf"] += float(keyword_bonus) * len(hits) / len(positive)
        ranked.append((item_key, entry, hits))

    ranked.sort(key=lambda row: row[1]["rrf"], reverse=True)
    max_score = ranked[0][1]["rrf"] if ranked else 1.0

    if dedupe:
        seen_keys: set[tuple[Any, ...]] = set()
        for exc_payload in excluded_payloads:
            seen_keys.update(duplicate_keys(exc_payload))
        ranked = keep_first_unique(ranked, seen_keys)

    if excluded_ids:
        ranked = [row for row in ranked if row[0] not in excluded_ids]
    elif offset:
        ranked = ranked[offset:]

    selected = ranked[: max(1, int(limit))]
    items: list[dict[str, Any]] = []
    for item_key, entry, hits in selected:
        payload = entry["payload"]
        raw = entry["raw"]
        items.append(
            {
                "id": str(payload.get("point_id") or payload.get("id") or item_key),
                "score": round(100.0 * entry["rrf"] / max_score, 2),
                "path": str(payload.get("path", "")),
                "name": str(payload.get("name", "")),
                "width": int(payload.get("width", 0)),
                "height": int(payload.get("height", 0)),
                "tags": list(payload.get("tags", [])),
                "tag_hits": hits,
                "caption": payload.get("caption"),
                "visual_score": raw.get("visual"),
                "descriptor_score": raw.get("descriptor"),
            }
        )
    has_more = len(ranked) > len(items)
    return items, has_more
