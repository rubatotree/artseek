"""SigLIP v2 embedder with fast local snapshot loading and GPU batch preprocessing."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache
from PIL import Image

from ..compute import configure_inference, release_accelerator, resolve_device, resolve_dtype
from ..config import EmbeddingConfig
from ..image_io import load_rgb
from .preprocess import shrink_to_short_edge


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


def _cached_siglip_snapshot(model_id: str) -> Path | None:
    """Return local HF cache snapshot directory if complete, avoiding network metadata requests."""
    if Path(model_id).is_dir():
        return Path(model_id)
    cfg_loc = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(cfg_loc, str) or not Path(cfg_loc).is_file():
        return None
    snapshot = Path(cfg_loc).parent
    for weight_name in ("model.safetensors", "pytorch_model.bin"):
        w_loc = try_to_load_from_cache(model_id, weight_name)
        if (
            isinstance(w_loc, str)
            and Path(w_loc).is_file()
            and Path(w_loc).parent == snapshot
        ):
            return snapshot
    return None


def _load_auto_model(source: str, dtype: torch.dtype, *, local_only: bool = False) -> Any:
    from transformers import AutoModel

    load_kwargs: dict[str, Any] = {
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "local_files_only": local_only,
    }
    try:
        model = AutoModel.from_pretrained(source, dtype=dtype, **load_kwargs)
    except TypeError:
        try:
            model = AutoModel.from_pretrained(source, torch_dtype=dtype, **load_kwargs)
        except TypeError:
            load_kwargs.pop("low_cpu_mem_usage", None)
            load_kwargs.pop("local_files_only", None)
            try:
                model = AutoModel.from_pretrained(source, dtype=dtype, **load_kwargs)
            except TypeError:
                model = AutoModel.from_pretrained(source, torch_dtype=dtype, **load_kwargs)
    return model.eval()


_PIL_RESAMPLE_TO_TORCH: dict[int, str] = {
    0: "nearest",
    2: "bilinear",
    3: "bicubic",
}


class SiglipEmbedder:
    """GPU-accelerated SigLIP embedder with local cache fast-path and GPU image preprocessing."""

    def __init__(self, config: EmbeddingConfig):
        from transformers import AutoProcessor

        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        snapshot = _cached_siglip_snapshot(config.model)
        source = str(snapshot) if snapshot is not None else config.model
        local_only = snapshot is not None
        try:
            self.processor = AutoProcessor.from_pretrained(
                source, local_files_only=local_only
            )
        except Exception:
            self.processor = AutoProcessor.from_pretrained(config.model)
            source = config.model
            local_only = False
        configure_inference(self.device)
        self.model = _load_auto_model(source, self.dtype, local_only=local_only)
        self.model.to(device=self.device, dtype=self.dtype)
        self._dimension = self._infer_dimension()
        self._target_h, self._target_w, self._interp_mode, self._mean, self._std = (
            self._extract_image_preprocess_params()
        )

    def _extract_image_preprocess_params(
        self,
    ) -> tuple[int, int, str, tuple[float, float, float], tuple[float, float, float]]:
        ip = getattr(self.processor, "image_processor", None)
        size = getattr(ip, "size", None)
        target_h = 256
        target_w = 256
        if isinstance(size, dict):
            target_h = int(size.get("height") or size.get("shortest_edge") or 256)
            target_w = int(size.get("width") or size.get("shortest_edge") or target_h)
        elif size is not None:
            h = getattr(size, "height", None) or getattr(size, "shortest_edge", None)
            w = getattr(size, "width", None) or getattr(size, "shortest_edge", None)
            if h:
                target_h = int(h)
            if w:
                target_w = int(w)
        resample = int(getattr(ip, "resample", 2) or 2)
        mode = _PIL_RESAMPLE_TO_TORCH.get(resample, "bilinear")
        raw_mean = getattr(ip, "image_mean", (0.5, 0.5, 0.5)) or (0.5, 0.5, 0.5)
        raw_std = getattr(ip, "image_std", (0.5, 0.5, 0.5)) or (0.5, 0.5, 0.5)
        mean = (float(raw_mean[0]), float(raw_mean[1]), float(raw_mean[2]))
        std = (float(raw_std[0]), float(raw_std[1]), float(raw_std[2]))
        return target_h, target_w, mode, mean, std

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

    def _wait_gpu_cooperative(self, tensor: torch.Tensor) -> None:
        if tensor.is_cuda:
            ev = torch.cuda.Event()
            ev.record()
            while not ev.query():
                time.sleep(0.001)

    @staticmethod
    def _normalize(features: Any) -> np.ndarray:
        features = F.normalize(pooled_features(features).float(), dim=-1)
        if features.is_cuda:
            ev = torch.cuda.Event()
            ev.record()
            while not ev.query():
                time.sleep(0.001)
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

    def _preprocess_images_gpu(
        self, images: Sequence[Image.Image | np.ndarray]
    ) -> torch.Tensor:
        """Resize and normalize RGB images directly on `self.device` with antialiasing."""
        target_h = self._target_h
        target_w = self._target_w
        mode = self._interp_mode
        use_aa = mode in {"bilinear", "bicubic"}
        non_blocking = self.device.type == "cuda"

        resized: list[torch.Tensor] = []
        for item in images:
            if isinstance(item, np.ndarray):
                arr = np.array(item, dtype=np.uint8, copy=not item.flags.writeable)
            else:
                rgb = item if item.mode == "RGB" else item.convert("RGB")
                arr = np.array(rgb, dtype=np.uint8, copy=True)
            tensor = torch.from_numpy(arr).to(
                device=self.device, non_blocking=non_blocking
            )
            # (H, W, 3) -> (1, 3, H, W) float32 in [0, 1]
            chw = tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0
            if chw.shape[-2] != target_h or chw.shape[-1] != target_w:
                chw = F.interpolate(
                    chw,
                    size=(target_h, target_w),
                    mode=mode,
                    align_corners=False if use_aa else None,
                    antialias=use_aa,
                )
            resized.append(chw)
            time.sleep(0)

        batch = torch.cat(resized, dim=0)
        mean = torch.tensor(self._mean, device=self.device, dtype=torch.float32).view(
            1, 3, 1, 1
        )
        std = torch.tensor(self._std, device=self.device, dtype=torch.float32).view(
            1, 3, 1, 1
        )
        batch = (batch - mean) / std
        return batch.to(dtype=self.dtype)

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

    def encode_images(
        self, images: Sequence[Path | str | Image.Image | np.ndarray]
    ) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dimension), dtype=np.float32)
        opened: list[Image.Image | np.ndarray] = [
            load_rgb(item)
            if isinstance(item, (str, Path))
            else (item if isinstance(item, np.ndarray) else item.convert("RGB"))
            for item in images
        ]
        opened = [
            item if isinstance(item, np.ndarray) else shrink_to_short_edge(item)
            for item in opened
        ]
        pixel_values = self._preprocess_images_gpu(opened)
        with torch.inference_mode():
            features = self.model.get_image_features(pixel_values=pixel_values)
        return self._normalize(features)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
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
