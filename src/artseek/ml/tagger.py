"""WD-Tagger v3 with GPU batch preprocessing and vectorized GPU postprocessing."""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from PIL import Image

from ..compute import configure_inference, release_accelerator, resolve_device, resolve_dtype
from ..config import EmbeddingConfig, TaggerConfig


@dataclass(slots=True)
class PredictedTags:
    tags: list[str]
    scores: dict[str, float]


def _cached_tagger_snapshot(model_id: str) -> Path | None:
    """Use a complete cached revision without making Hub metadata requests."""
    if Path(model_id).is_dir():
        return Path(model_id)
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
    """GPU-accelerated WD-Tagger with GPU resize/crop/normalize and vectorized mask+topk."""

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
        self.data_config: dict[str, Any] = dict(data_config)
        self.transform = create_transform(**data_config, is_training=False)
        labels_path = (
            snapshot / "selected_tags.csv"
            if snapshot is not None
            else Path(hf_hub_download(config.model, "selected_tags.csv"))
        )
        self.names, self.categories = self._load_labels(Path(labels_path))
        self._input_h, self._input_w, self._scale_size, self._interp_mode, self._mean, self._std = (
            self._extract_preprocess_params(self.data_config)
        )
        allowed = [
            cat == 0 or (self.config.include_character_tags and cat == 4)
            for cat in self.categories
        ]
        self._allowed_mask = torch.tensor(allowed, dtype=torch.bool, device=self.device)

    @staticmethod
    def _extract_preprocess_params(
        data_config: dict[str, Any],
    ) -> tuple[int, int, int, str, tuple[float, float, float], tuple[float, float, float]]:
        input_size = data_config.get("input_size", (3, 448, 448))
        h = int(input_size[-2])
        w = int(input_size[-1])
        crop_pct = float(data_config.get("crop_pct") or 1.0)
        scale_size = int(math.floor(min(h, w) / crop_pct))
        interp = str(data_config.get("interpolation", "bicubic")).lower()
        mode = interp if interp in {"nearest", "bilinear", "bicubic"} else "bicubic"
        raw_mean = data_config.get("mean", (0.5, 0.5, 0.5))
        raw_std = data_config.get("std", (0.5, 0.5, 0.5))
        mean = (float(raw_mean[0]), float(raw_mean[1]), float(raw_mean[2]))
        std = (float(raw_std[0]), float(raw_std[1]), float(raw_std[2]))
        return h, w, scale_size, mode, mean, std

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
        self._allowed_mask = None  # type: ignore[assignment]
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

    def _preprocess_images_gpu(
        self, images: Sequence[Image.Image | np.ndarray]
    ) -> torch.Tensor:
        """Exact GPU equivalent of timm Resize(short_edge=scale_size) + CenterCrop + Normalize."""
        target_h = self._input_h
        target_w = self._input_w
        scale_size = self._scale_size
        mode = self._interp_mode
        use_aa = mode in {"bilinear", "bicubic"}
        non_blocking = self.device.type == "cuda"

        cropped: list[torch.Tensor] = []
        for item in images:
            if isinstance(item, np.ndarray):
                arr = np.array(item, dtype=np.uint8, copy=not item.flags.writeable)
            else:
                rgb = item if item.mode == "RGB" else item.convert("RGB")
                arr = np.array(rgb, dtype=np.uint8, copy=True)
            tensor = torch.from_numpy(arr).to(
                device=self.device, non_blocking=non_blocking
            )
            chw = tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0
            cur_h, cur_w = int(chw.shape[-2]), int(chw.shape[-1])
            if cur_h <= cur_w:
                new_h = scale_size
                new_w = max(target_w, int(scale_size * cur_w / max(cur_h, 1)))
            else:
                new_w = scale_size
                new_h = max(target_h, int(scale_size * cur_h / max(cur_w, 1)))
            if new_h != cur_h or new_w != cur_w:
                chw = F.interpolate(
                    chw,
                    size=(new_h, new_w),
                    mode=mode,
                    align_corners=False if use_aa else None,
                    antialias=use_aa,
                )
            top = max(0, (new_h - target_h) // 2)
            left = max(0, (new_w - target_w) // 2)
            chw = chw[:, :, top : top + target_h, left : left + target_w]
            cropped.append(chw)
            time.sleep(0)

        batch = torch.cat(cropped, dim=0)
        mean = torch.tensor(self._mean, device=self.device, dtype=torch.float32).view(
            1, 3, 1, 1
        )
        std = torch.tensor(self._std, device=self.device, dtype=torch.float32).view(
            1, 3, 1, 1
        )
        batch = (batch - mean) / std
        return batch.to(dtype=self.dtype)

    def predict(
        self, images: Sequence[Image.Image | np.ndarray]
    ) -> list[PredictedTags]:
        if not images:
            return []
        tensors = self._preprocess_images_gpu(images)
        with torch.inference_mode():
            output = self.model(tensors)
            logits = output.logits if hasattr(output, "logits") else output
            probabilities = torch.sigmoid(logits.float())
            if self._allowed_mask.device != probabilities.device:
                self._allowed_mask = self._allowed_mask.to(probabilities.device)
            masked = probabilities.masked_fill(~self._allowed_mask.unsqueeze(0), -1.0)
            k = min(max(1, int(self.config.max_tags)), masked.shape[-1])
            top_scores, top_indices = torch.topk(masked, k=k, dim=-1, largest=True, sorted=True)
            if top_scores.is_cuda:
                ev = torch.cuda.Event()
                ev.record()
                while not ev.query():
                    time.sleep(0.001)
            scores_cpu = top_scores.cpu().tolist()
            indices_cpu = top_indices.cpu().tolist()

        threshold = float(self.config.threshold)
        names = self.names
        results: list[PredictedTags] = []
        for row_scores, row_indices in zip(scores_cpu, indices_cpu, strict=True):
            selected_tags: list[str] = []
            selected_scores: dict[str, float] = {}
            for score, idx in zip(row_scores, row_indices, strict=True):
                if score < threshold:
                    break
                name = names[idx]
                selected_tags.append(name)
                selected_scores[name] = round(float(score), 4)
            results.append(PredictedTags(tags=selected_tags, scores=selected_scores))
        return results
