from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from PIL import Image

from ..compute import configure_inference, release_accelerator, resolve_device, resolve_dtype
from ..config import EmbeddingConfig, TaggerConfig
from ..image_io import composite_on_white


@dataclass(slots=True)
class PredictedTags:
    tags: list[str]
    scores: dict[str, float]


def _cached_tagger_snapshot(model_id: str) -> Path | None:
    """Use a complete cached revision without making Hub metadata requests."""
    cached: dict[str, Path] = {}
    for filename in ("config.json", "selected_tags.csv"):
        location = try_to_load_from_cache(model_id, filename)
        if not isinstance(location, str) or not Path(location).is_file():
            return None
        cached[filename] = Path(location)
    if cached["config.json"].parent != cached["selected_tags.csv"].parent:
        return None
    snapshot = cached["config.json"].parent
    for filename in ("model.safetensors", "pytorch_model.bin"):
        location = try_to_load_from_cache(model_id, filename)
        if (
            isinstance(location, str)
            and Path(location).is_file()
            and Path(location).parent == snapshot
        ):
            return snapshot
    return None


class WDTagger:
    def __init__(self, config: TaggerConfig, runtime: EmbeddingConfig):
        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as exc:
            raise RuntimeError('缺少插画标签器依赖，请执行：pip install -e ".[tagger]"') from exc

        self.config = config
        self.device = resolve_device(runtime.device)
        self.dtype = resolve_dtype(runtime.dtype, self.device)
        configure_inference(self.device)
        snapshot = _cached_tagger_snapshot(config.model)
        source = f"local-dir:{snapshot}" if snapshot is not None else f"hf_hub:{config.model}"
        self.model = timm.create_model(source, pretrained=True).eval()
        self.model.to(device=self.device, dtype=self.dtype)
        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)
        input_size = data_config.get("input_size", (3, 448, 448))
        self._input_h = int(input_size[-2])
        self._input_w = int(input_size[-1])
        raw_mean = data_config.get("mean", (0.5, 0.5, 0.5))
        raw_std = data_config.get("std", (0.5, 0.5, 0.5))
        self._mean = tuple(float(v) for v in raw_mean[:3])
        self._std = tuple(float(v) for v in raw_std[:3])
        labels_path = snapshot / "selected_tags.csv" if snapshot is not None else Path(
            hf_hub_download(config.model, "selected_tags.csv")
        )
        self.names, self.categories = self._load_labels(Path(labels_path))

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            model.to("cpu")
        except Exception:
            pass
        self.model = None
        self.transform = None
        release_accelerator()

    @staticmethod
    def _load_labels(path: Path) -> tuple[list[str], list[int]]:
        names: list[str] = []
        categories: list[int] = []
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                names.append(row["name"])
                categories.append(int(row["category"]))
        return names, categories

    def _preprocess_one(self, image: Image.Image) -> torch.Tensor:
        rgb = composite_on_white(image)
        w, h = rgb.size
        if w != h:
            max_dim = max(w, h)
            padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded.paste(rgb, ((max_dim - w) // 2, (max_dim - h) // 2))
            rgb = padded
        if rgb.size != (self._input_w, self._input_h):
            rgb = rgb.resize((self._input_w, self._input_h), Image.Resampling.BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        arr = arr[..., ::-1]  # RGB -> BGR
        mean = np.asarray(self._mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self._std, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    def predict(self, images: Sequence[Image.Image]) -> list[PredictedTags]:
        if not images:
            return []
        tensors = torch.stack([self._preprocess_one(image) for image in images])
        tensors = tensors.to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            output = self.model(tensors)
            logits = output.logits if hasattr(output, "logits") else output
            probabilities = torch.sigmoid(logits.float()).cpu()

        results: list[PredictedTags] = []
        for row in probabilities:
            candidates: list[tuple[str, float]] = []
            for name, category, score in zip(
                self.names, self.categories, row.tolist(), strict=True
            ):
                allowed = category == 0 or (self.config.include_character_tags and category == 4)
                if allowed and score >= self.config.threshold:
                    candidates.append((name, float(score)))
            candidates.sort(key=lambda item: item[1], reverse=True)
            selected = candidates[: self.config.max_tags]
            results.append(
                PredictedTags(
                    tags=[name for name, _ in selected],
                    scores={name: round(score, 4) for name, score in selected},
                )
            )
        return results
