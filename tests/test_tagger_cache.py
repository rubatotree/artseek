from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from artseek.models.tagger import _cached_tagger_snapshot


def test_cached_tagger_snapshot_requires_matching_complete_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / "revision-a"
    snapshot.mkdir()
    for name in ("config.json", "selected_tags.csv", "model.safetensors"):
        (snapshot / name).write_text("cached", encoding="utf-8")
    cached = {name: str(snapshot / name) for name in (
        "config.json", "selected_tags.csv", "model.safetensors"
    )}

    with patch(
        "artseek.models.tagger.try_to_load_from_cache",
        side_effect=lambda _id, name: cached.get(name),
    ):
        assert _cached_tagger_snapshot("example/tagger") == snapshot

    other = tmp_path / "revision-b"
    other.mkdir()
    (other / "selected_tags.csv").write_text("cached", encoding="utf-8")
    cached["selected_tags.csv"] = str(other / "selected_tags.csv")
    with patch(
        "artseek.models.tagger.try_to_load_from_cache",
        side_effect=lambda _id, name: cached.get(name),
    ):
        assert _cached_tagger_snapshot("example/tagger") is None


def test_cached_tagger_snapshot_falls_back_if_weights_missing(tmp_path: Path) -> None:
    for name in ("config.json", "selected_tags.csv"):
        (tmp_path / name).write_text("cached", encoding="utf-8")
    cached = {name: str(tmp_path / name) for name in ("config.json", "selected_tags.csv")}
    with patch(
        "artseek.models.tagger.try_to_load_from_cache",
        side_effect=lambda _id, name: cached.get(name),
    ):
        assert _cached_tagger_snapshot("example/tagger") is None
