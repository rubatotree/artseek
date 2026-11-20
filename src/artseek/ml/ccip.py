"""CCIP character identity feature extraction for cross-outfit character retrieval."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
from PIL import Image

from ..config import EmbeddingConfig
from ..image_io import composite_on_white
from .embedder import resolve_device
from .tagger import _cached_or_download, create_ort_session

_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _crop_character_region(
    image: Image.Image,
    *,
    bbox: Sequence[float] | None = None,
    crop_json: str | None = None,
) -> Image.Image:
    """Crop a character region using an explicit bbox or cached `character_crop_json`."""
    rgb = composite_on_white(image)
    w, h = rgb.size
    chosen_box: Sequence[float] | None = bbox
    if chosen_box is None and crop_json:
        try:
            data = json.loads(crop_json)
            if isinstance(data, dict):
                chosen_box = data.get("body") or data.get("head")
        except (json.JSONDecodeError, TypeError):
            chosen_box = None

    if chosen_box is not None and len(chosen_box) == 4:
        x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in chosen_box]
        px1 = max(0, min(w - 1, int(round(x1 * w))))
        py1 = max(0, min(h - 1, int(round(y1 * h))))
        px2 = max(px1 + 1, min(w, int(round(x2 * w))))
        py2 = max(py1 + 1, min(h, int(round(y2 * h))))
        if (px2 - px1) >= 8 and (py2 - py1) >= 8:
            rgb = rgb.crop((px1, py1, px2, py2))

    # Preserve character aspect ratio inside 384x384 canvas.
    cw, ch = rgb.size
    scale = 384.0 / max(1, max(cw, ch))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = rgb.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (384, 384), (128, 128, 128))
    canvas.paste(resized, ((384 - nw) // 2, (384 - nh) // 2))
    return canvas


class CCIPExtractor:
    """Extracts 768-D normalized CCIP embeddings via `deepghs/ccip_onnx`."""

    def __init__(
        self,
        runtime: EmbeddingConfig,
        *,
        repo_id: str = "deepghs/ccip_onnx",
        model_file: str = "ccip-caformer-24-randaug-pruned/model_feat.onnx",
    ) -> None:
        self.device = resolve_device(runtime.device)
        model_path = _cached_or_download(repo_id, model_file)
        self.session = create_ort_session(model_path, self.device)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    def close(self) -> None:
        self.session = None

    def extract_batch(
        self,
        items: Sequence[tuple[Image.Image, str | None]],
    ) -> np.ndarray:
        """Compute L2-normalized `(N, 768)` CCIP embeddings for `(image, crop_json)` items."""
        if not items or self.session is None:
            return np.zeros((0, 768), dtype=np.float32)
        tensors: list[np.ndarray] = []
        for img, crop_json in items:
            square = _crop_character_region(img, crop_json=crop_json)
            arr = np.asarray(square, dtype=np.float32) / 255.0
            arr = (arr - _CLIP_MEAN) / _CLIP_STD
            tensors.append(arr.transpose(2, 0, 1))
        batch = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
        feats = np.asarray(
            self.session.run([self._output_name], {self._input_name: batch})[0],
            dtype=np.float32,
        )
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return (feats / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)

    def extract_one(
        self,
        image: Image.Image,
        *,
        bbox: Sequence[float] | None = None,
        crop_json: str | None = None,
    ) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("CCIPExtractor is closed")
        square = _crop_character_region(image, bbox=bbox, crop_json=crop_json)
        arr = (np.asarray(square, dtype=np.float32) / 255.0 - _CLIP_MEAN) / _CLIP_STD
        batch = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)
        feats = np.asarray(
            self.session.run([self._output_name], {self._input_name: batch})[0],
            dtype=np.float32,
        )
        vec = feats[0]
        norm = float(np.linalg.norm(vec))
        return (vec / max(norm, 1e-12)).astype(np.float32, copy=False)
