"""Tag categories, MCut thresholding, semantic overlap pruning, fusion, and descriptor packing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.query_plan import normalize_tag
from ..core.ranking import order_tags_for_descriptor

CAT_GENERAL = 0
CAT_ARTIST_STYLE = 1
CAT_COPYRIGHT = 3
CAT_CHARACTER = 4
CAT_META = 5
CAT_RATING = 6
CAT_MEDIUM = 7
CAT_ATTRIBUTE = 8
CAT_POSE = 9

CAMIE_CATEGORY_MAP: dict[str, int] = {
    "general": CAT_GENERAL,
    "artist": CAT_ARTIST_STYLE,
    "copyright": CAT_COPYRIGHT,
    "character": CAT_CHARACTER,
    "meta": CAT_META,
    "year": CAT_META,
    "rating": CAT_RATING,
}

PIXAI10_CATEGORY_MAP: dict[str, int] = {
    "general": CAT_GENERAL,
    "character": CAT_CHARACTER,
    "copyright": CAT_COPYRIGHT,
    "ip": CAT_COPYRIGHT,
    "style": CAT_ARTIST_STYLE,
    "artist": CAT_ARTIST_STYLE,
    "meta": CAT_META,
    "rating": CAT_RATING,
}

RATING_CANONICAL: dict[str, str] = {
    "rating_general": "safe",
    "general": "safe",
    "safe": "safe",
    "rating_sensitive": "r15",
    "sensitive": "r15",
    "r15": "r15",
    "rating_questionable": "r18",
    "questionable": "r18",
    "rating_explicit": "r18",
    "explicit": "r18",
    "r18": "r18",
}

# Specific child tag -> redundant generic parent tags pruned in compact summaries/descriptors.
OVERLAP_IMPLICATIONS: dict[str, tuple[str, ...]] = {
    "serafuku": ("school_uniform", "sailor_collar", "shirt", "skirt"),
    "school_swimsuit": ("one-piece_swimsuit", "swimsuit"),
    "one-piece_swimsuit": ("swimsuit",),
    "bikini": ("swimsuit",),
    "micro_bikini": ("bikini", "swimsuit"),
    "gothic_lolita": ("lolita_fashion", "dress"),
    "sweet_lolita": ("lolita_fashion", "dress"),
    "sundress": ("dress",),
    "puffy_sleeves": ("sleeves",),
    "wide_sleeves": ("sleeves",),
    "sleeves_past_wrists": ("long_sleeves",),
    "twintails": ("hair",),
    "ponytail": ("hair",),
    "braid": ("hair",),
    "twin_braids": ("braid", "hair"),
    "knees_to_chest": ("sitting",),
    "hugging_own_legs": ("sitting",),
    "seiza": ("kneeling", "sitting"),
    "lying_on_side": ("lying",),
    "on_back": ("lying",),
    "on_stomach": ("lying",),
    "thighhighs": ("legwear",),
    "kneehighs": ("legwear",),
    "pantyhose": ("legwear",),
    "loose_socks": ("socks",),
    "cat_ears": ("animal_ears",),
    "fox_ears": ("animal_ears",),
    "rabbit_ears": ("animal_ears",),
    "dog_ears": ("animal_ears",),
    "wolf_ears": ("animal_ears",),
}


@dataclass(slots=True)
class PredictedTags:
    tags: list[str]
    scores: dict[str, float]
    categories: dict[str, int] = field(default_factory=dict)
    rating: str | None = None


def mcut_threshold(
    probs: Sequence[float],
    *,
    min_threshold: float = 0.25,
    max_threshold: float = 0.65,
) -> float:
    """Compute Multilabel Cut (MCut) threshold from sorted probabilities."""
    vals = sorted((float(p) for p in probs if float(p) > 0.0), reverse=True)
    if len(vals) < 2:
        return min_threshold
    best_gap = -1.0
    cut = min_threshold
    for i in range(len(vals) - 1):
        gap = vals[i] - vals[i + 1]
        if gap > best_gap:
            best_gap = gap
            cut = 0.5 * (vals[i] + vals[i + 1])
    return max(min_threshold, min(max_threshold, cut))


def prune_overlapping_for_summary(tags: Sequence[str]) -> list[str]:
    """Remove redundant generic parent tags when a specific child tag is present."""
    normalized = [normalize_tag(t) for t in tags if normalize_tag(t)]
    present = set(normalized)
    drop: set[str] = set()
    for tag in normalized:
        parents = OVERLAP_IMPLICATIONS.get(tag)
        if parents:
            for parent in parents:
                if parent in present:
                    drop.add(parent)
    return [tag for tag in dict.fromkeys(normalized) if tag not in drop]


def merge_predicted_tags(
    predictions: Sequence[PredictedTags],
    *,
    max_tags: int = 80,
) -> PredictedTags:
    """Fuse predictions from multiple taggers (e.g. Camie v2 + PixAI 0.9)."""
    if not predictions:
        return PredictedTags([], {})
    if len(predictions) == 1:
        single = predictions[0]
        return PredictedTags(
            tags=list(single.tags[:max_tags]),
            scores=dict(single.scores),
            categories=dict(single.categories),
            rating=single.rating,
        )

    all_scores: dict[str, list[float]] = {}
    categories: dict[str, int] = {}
    rating: str | None = None
    for pred in predictions:
        if rating is None and pred.rating:
            rating = pred.rating
        for raw_tag in pred.tags:
            tag = normalize_tag(raw_tag)
            if not tag:
                continue
            score = float(pred.scores.get(raw_tag, pred.scores.get(tag, 0.0)))
            all_scores.setdefault(tag, []).append(score)
            cat = int(pred.categories.get(raw_tag, pred.categories.get(tag, CAT_GENERAL)))
            if tag not in categories or (cat != CAT_GENERAL and categories[tag] == CAT_GENERAL):
                categories[tag] = cat

    fused_scores: dict[str, float] = {}
    for tag, vals in all_scores.items():
        if len(vals) == 1:
            fused = vals[0]
        else:
            vals_sorted = sorted(vals, reverse=True)
            # Cross-model agreement boost capped cleanly below 1.0.
            fused = min(0.9999, vals_sorted[0] + 0.15 * vals_sorted[1])
        fused_scores[tag] = round(fused, 4)

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: (
            -(
                item[1]
                + (
                    0.12
                    if categories.get(item[0], 0) in {CAT_CHARACTER, CAT_COPYRIGHT}
                    else 0.0
                )
            ),
            item[0],
        ),
    )[: max(1, int(max_tags))]

    chosen_tags = [tag for tag, _ in ranked]
    return PredictedTags(
        tags=chosen_tags,
        scores={tag: fused_scores[tag] for tag in chosen_tags},
        categories={tag: categories.get(tag, CAT_GENERAL) for tag in chosen_tags},
        rating=rating,
    )


def build_high_density_descriptor(
    tags: Sequence[str],
    categories: Mapping[str, int] | None = None,
    *,
    caption: dict[str, Any] | None = None,
    title: str | None = None,
    drop_overlap: bool = True,
    max_general: int = 20,
) -> str:
    """Pack IP, character, medium/rating, title, and key tags into SigLIP's 64-token budget."""
    cats = categories or {}
    ips: list[str] = []
    chars: list[str] = []
    attrs: list[str] = []
    general: list[str] = []

    for raw in tags:
        tag = normalize_tag(raw)
        if not tag:
            continue
        cat = int(cats.get(raw, cats.get(tag, CAT_GENERAL)))
        if cat == CAT_COPYRIGHT:
            if tag not in ips:
                ips.append(tag)
        elif cat == CAT_CHARACTER:
            if tag not in chars:
                chars.append(tag)
        elif cat in {CAT_MEDIUM, CAT_RATING, CAT_ATTRIBUTE}:
            if tag not in attrs:
                attrs.append(tag)
        elif cat in {CAT_GENERAL, CAT_POSE, CAT_ARTIST_STYLE}:
            if tag not in general:
                general.append(tag)

    if drop_overlap and general:
        general = prune_overlapping_for_summary(general)
    general = order_tags_for_descriptor(general)[:max_general]

    ordered_tags = [*ips[:4], *chars[:6], *attrs[:4], *general]
    segments: list[str] = []
    if ordered_tags:
        segments.append(", ".join(t.replace("_", " ") for t in ordered_tags))
    if title and str(title).strip():
        segments.append(str(title).strip())
    if caption:
        from .captioner import flatten_caption

        flat = flatten_caption(caption)
        if flat:
            segments.append(flat)
    return "。".join(filter(None, segments))
