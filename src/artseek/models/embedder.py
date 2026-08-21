from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from ..compute import configure_inference, release_accelerator, resolve_device, resolve_dtype
from ..config import EmbeddingConfig
from ..image_io import load_rgb


def pooled_features(output: Any) -> torch.Tensor:
    """transformers v4 returns a tensor; v5 returns BaseModelOutputWithPooling."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(output, (tuple, list)) and output:
        candidate = output[1] if len(output) > 1 else output[0]
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TypeError(f"Expected a feature tensor or pooler_output, got {type(output)!r}")


def _load_auto_model(name: str, dtype: torch.dtype) -> Any:
    load_kwargs: dict[str, Any] = {"attn_implementation": "sdpa"}
    try:
        model = AutoModel.from_pretrained(name, dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(name, torch_dtype=dtype, **load_kwargs)
    return model.eval()


class SiglipEmbedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        self.processor = AutoProcessor.from_pretrained(config.model)
        configure_inference(self.device)
        self.model = _load_auto_model(config.model, self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)
        self._dimension = self._infer_dimension()

    def _infer_dimension(self) -> int:
        config = self.model.config
        text_config = getattr(config, "text_config", None)
        for source in (
            getattr(config, "projection_dim", None),
            getattr(text_config, "projection_size", None),
            getattr(text_config, "hidden_size", None),
            getattr(config, "hidden_size", None),
            getattr(getattr(config, "vision_config", None), "hidden_size", None),
        ):
            if source:
                return int(source)
        return int(self.encode_texts(["test"]).shape[1])

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _normalize(features: Any) -> np.ndarray:
        features = torch.nn.functional.normalize(pooled_features(features).float(), dim=-1)
        return features.cpu().numpy().astype(np.float32, copy=False)

    def _move_inputs(self, inputs: Any) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        non_blocking = self.device.type == "cuda"
        for key, value in dict(inputs).items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
                continue
            if value.is_floating_point():
                moved[key] = value.to(
                    device=self.device, dtype=self.dtype, non_blocking=non_blocking
                )
            else:
                moved[key] = value.to(device=self.device, non_blocking=non_blocking)
        return moved

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            model.to("cpu")
        except Exception:
            pass
        self.model = None
        self.processor = None
        release_accelerator()

    def warmup(self) -> None:
        """Run a dummy forward pass so CUDA kernels compile before the first query."""
        self.encode_texts(["warmup"])
        dummy = Image.new("RGB", (256, 256), (128, 128, 128))
        self.encode_images([dummy])
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def encode_images(self, images: Sequence[Path | Image.Image]) -> np.ndarray:
        opened = [load_rgb(item) if isinstance(item, (str, Path)) else item.convert("RGB") for item in images]
        inputs = self._move_inputs(self.processor(images=opened, return_tensors="pt"))
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return self._normalize(features)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        text_config = getattr(self.model.config, "text_config", None)
        max_length = getattr(text_config, "max_position_embeddings", None) or 64
        inputs = self._move_inputs(
            self.processor(
                text=list(texts),
                padding="max_length",
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
        )
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return self._normalize(features)

    def encode_composed(
        self,
        image: Path | Image.Image,
        text: str,
        *,
        text_weight: float = 1.0,
        image_weight: float = 1.0,
    ) -> np.ndarray:
        image_vector = self.encode_images([image])[0]
        text_vector = self.encode_texts([text])[0]
        vector = float(text_weight) * text_vector + float(image_weight) * image_vector
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        return vector.astype(np.float32, copy=False)

