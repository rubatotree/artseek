from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from ..compute import configure_inference, release_accelerator, resolve_device, resolve_dtype
from ..config import EmbeddingConfig, TaggerConfig


@dataclass(slots=True)
class PredictedTags:
    tags: list[str]
    scores: dict[str, float]


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
        self.model = timm.create_model(f"hf_hub:{config.model}", pretrained=True).eval()
        self.model.to(device=self.device, dtype=self.dtype)
        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)
        labels_path = hf_hub_download(config.model, "selected_tags.csv")
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

    def predict(self, images: Sequence[Image.Image]) -> list[PredictedTags]:
        tensors = torch.stack([self.transform(image.convert("RGB")) for image in images])
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

