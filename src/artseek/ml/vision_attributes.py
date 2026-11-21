"""Vision attribute analyzers, YOLOv8 head focal detection, and embedded AI metadata parsing."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..config import EmbeddingConfig
from ..image_io import composite_on_white
from .embedder import resolve_device
from .tag_postprocess import CAT_ATTRIBUTE, CAT_MEDIUM, CAT_RATING
from .tagger import _cached_or_download, create_ort_session

VISION_META_VERSION = 2

_MEDIUM_LABELS = ("3d", "bangumi", "comic", "illustration", "not_painting")
_RATING_LABELS = ("safe", "r15", "r18")
_MONO_LABELS = ("monochrome", "normal")
_AI_LABELS = ("ai", "human")
_COMPLETENESS_LABELS = ("polished", "rough", "monochrome")
_AESTHETIC_LABELS = ("masterpiece", "best", "great", "good", "normal", "low", "worst")
_AESTHETIC_WEIGHTS: dict[str, float] = {
    "masterpiece": 1.0,
    "best": 0.84,
    "great": 0.68,
    "good": 0.52,
    "normal": 0.34,
    "low": 0.16,
    "worst": 0.02,
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], others[:, 0])
    yy1 = np.maximum(box[1], others[:, 1])
    xx2 = np.minimum(box[2], others[:, 2])
    yy2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    area_b = np.maximum(0.0, (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]))
    union = np.maximum(1e-9, area_a + area_b - inter)
    return inter / union


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _box_iou(boxes[idx], boxes[rest])
        order = rest[ious <= iou_threshold]
    return keep


def extract_embedded_ai_metadata(image: Image.Image) -> tuple[bool, str | None]:
    """Inspect PNG text chunks / WebP / EXIF comments for AI generator signatures."""
    info = getattr(image, "info", None) or {}
    if not isinstance(info, dict):
        return False, None

    software = str(info.get("Software") or info.get("software") or "").casefold()
    source = str(info.get("Source") or info.get("source") or "").casefold()
    comment = str(info.get("Comment") or info.get("comment") or "")
    params = str(info.get("parameters") or "")

    if "novelai" in software or "novelai" in source:
        return True, "novelai"
    if comment:
        lowered = comment.casefold()
        if (
            '"steps"' in lowered
            and '"sampler"' in lowered
            and ('"uc"' in lowered or "novelai" in lowered)
        ):
            return True, "novelai"

    if "prompt" in info and "workflow" in info:
        return True, "comfyui"
    for key in ("prompt", "workflow"):
        raw = info.get(key)
        if isinstance(raw, str) and '"class_type"' in raw:
            return True, "comfyui"

    if params and ("Steps:" in params and ("Sampler:" in params or "CFG scale:" in params)):
        return True, "sd_webui"

    exif_raw = info.get("exif")
    if isinstance(exif_raw, (bytes, bytearray)):
        text = bytes(exif_raw).decode("latin-1", errors="ignore")
        lower_text = text.casefold()
        if "novelai" in lower_text:
            return True, "novelai"
        if "steps:" in lower_text and ("sampler:" in lower_text or "cfg scale:" in lower_text):
            return True, "sd_webui"
        if '"class_type"' in text:
            return True, "comfyui"

    return False, None


@dataclass(slots=True)
class HeadDetectionResult:
    focal_x: float
    focal_y: float
    shot_type: str
    crop_json: str | None
    head_count: int


class HeadFocalDetector:
    """YOLOv8 anime head detector computing gallery focal points and character crop boxes."""

    def __init__(
        self,
        device: torch.device,
        *,
        repo_id: str = "deepghs/anime_head_detection",
        model_file: str = "head_detect_v2.0_s/model.onnx",
        input_size: int = 640,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
    ) -> None:
        self.device = device
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        model_path = _cached_or_download(repo_id, model_file)
        self.session = create_ort_session(model_path, device)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    def close(self) -> None:
        self.session = None

    def _preprocess_one(self, image: Image.Image) -> tuple[np.ndarray, float, int, int, int, int]:
        rgb = composite_on_white(image)
        orig_w, orig_h = rgb.size
        size = self.input_size
        scale = min(size / max(1, orig_w), size / max(1, orig_h))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        arr = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        return arr, scale, pad_x, pad_y, orig_w, orig_h

    def _decode_one(
        self,
        raw_pred: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> HeadDetectionResult:
        # YOLOv8 head outputs [5, anchors] (or [anchors, 5]): cx, cy, w, h, conf.
        pred = raw_pred
        if pred.ndim == 2 and pred.shape[0] in (5, 6) and pred.shape[1] not in (5, 6):
            pred = pred.T
        if pred.ndim != 2 or pred.shape[1] < 5:
            return HeadDetectionResult(0.5, 0.35, "scenery_or_object", None, 0)

        confs = pred[:, 4]
        mask = confs >= self.conf_threshold
        if not np.any(mask):
            return HeadDetectionResult(0.5, 0.35, "scenery_or_object", None, 0)

        sel = pred[mask]
        sel_confs = confs[mask]
        cx = sel[:, 0]
        cy = sel[:, 1]
        bw = sel[:, 2]
        bh = sel[:, 3]
        x1 = (cx - 0.5 * bw - pad_x) / max(scale, 1e-6)
        y1 = (cy - 0.5 * bh - pad_y) / max(scale, 1e-6)
        x2 = (cx + 0.5 * bw - pad_x) / max(scale, 1e-6)
        y2 = (cy + 0.5 * bh - pad_y) / max(scale, 1e-6)

        boxes = np.stack(
            [
                np.clip(x1 / max(1, orig_w), 0.0, 1.0),
                np.clip(y1 / max(1, orig_h), 0.0, 1.0),
                np.clip(x2 / max(1, orig_w), 0.0, 1.0),
                np.clip(y2 / max(1, orig_h), 0.0, 1.0),
            ],
            axis=1,
        )
        keep = _nms_xyxy(boxes, sel_confs, self.iou_threshold)
        if not keep:
            return HeadDetectionResult(0.5, 0.35, "scenery_or_object", None, 0)

        kept_boxes = [boxes[i] for i in keep]
        kept_confs = [float(sel_confs[i]) for i in keep]
        # Primary head = highest confidence * area
        ranked_idx = sorted(
            range(len(kept_boxes)),
            key=lambda i: kept_confs[i]
            * max(
                1e-4,
                float(
                    (kept_boxes[i][2] - kept_boxes[i][0])
                    * (kept_boxes[i][3] - kept_boxes[i][1])
                ),
            ),
            reverse=True,
        )
        primary = kept_boxes[ranked_idx[0]]
        px1, py1, px2, py2 = [float(v) for v in primary]
        pw = max(0.01, px2 - px1)
        ph = max(0.01, py2 - py1)

        # Focal point centers on top heads weighted by confidence*area (up to 3).
        w_sum = 0.0
        fx_sum = 0.0
        fy_sum = 0.0
        for ridx in ranked_idx[:3]:
            b = kept_boxes[ridx]
            area = max(1e-4, float((b[2] - b[0]) * (b[3] - b[1])))
            weight = kept_confs[ridx] * math.sqrt(area)
            w_sum += weight
            fx_sum += weight * float(0.5 * (b[0] + b[2]))
            fy_sum += weight * float(0.5 * (b[1] + b[3]))
        focal_x = _clamp01(fx_sum / max(w_sum, 1e-6))
        focal_y = _clamp01(fy_sum / max(w_sum, 1e-6))

        # Expand head box to cover character upper/mid body for CCIP extraction.
        body_box = [
            round(_clamp01(px1 - 0.9 * pw), 4),
            round(_clamp01(py1 - 0.35 * ph), 4),
            round(_clamp01(px2 + 0.9 * pw), 4),
            round(_clamp01(py2 + 3.0 * ph), 4),
        ]
        head_box = [round(px1, 4), round(py1, 4), round(px2, 4), round(py2, 4)]

        count = len(kept_boxes)
        if count >= 3 and ph < 0.25:
            shot = "group"
        elif ph >= 0.42:
            shot = "close_up"
        elif ph >= 0.20:
            shot = "upper_body"
        else:
            shot = "full_body"

        crop_payload = json.dumps(
            {"head": head_box, "body": body_box, "heads": count},
            separators=(",", ":"),
        )
        return HeadDetectionResult(focal_x, focal_y, shot, crop_payload, count)

    def predict(self, images: Sequence[Image.Image]) -> list[HeadDetectionResult]:
        if not images or self.session is None:
            return []
        tensors: list[np.ndarray] = []
        metas: list[tuple[float, int, int, int, int]] = []
        for img in images:
            arr, scale, pad_x, pad_y, orig_w, orig_h = self._preprocess_one(img)
            tensors.append(arr)
            metas.append((scale, pad_x, pad_y, orig_w, orig_h))
        batch = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
        raw_out = self.session.run([self._output_name], {self._input_name: batch})[0]
        results: list[HeadDetectionResult] = []
        for idx, meta in enumerate(metas):
            results.append(self._decode_one(raw_out[idx], *meta))
        return results


class _OnnxImageClassifier:
    """Shared wrapper for DeepGHS normalized `[-1, 1]` RGB ONNX image classifiers."""

    def __init__(
        self,
        device: torch.device,
        repo_id: str,
        model_file: str,
        meta_file: str,
        default_labels: tuple[str, ...],
        input_size: int = 384,
        *,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.input_size = int(input_size)
        self._mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        model_path = _cached_or_download(repo_id, model_file)
        meta_path = _cached_or_download(repo_id, meta_file)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.labels: list[str] = [str(x) for x in (meta.get("labels") or default_labels)]
        self.session = create_ort_session(model_path, device)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    def close(self) -> None:
        self.session = None

    def _preprocess_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        size = self.input_size
        tensors: list[np.ndarray] = []
        for img in images:
            rgb = composite_on_white(img)
            if rgb.size != (size, size):
                rgb = rgb.resize((size, size), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            arr = (arr - self._mean) / self._std
            tensors.append(arr.transpose(2, 0, 1))
        return np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)

    def predict_probs(self, images: Sequence[Image.Image]) -> list[dict[str, float]]:
        if not images or self.session is None:
            return []
        batch = self._preprocess_batch(images)
        raw = np.asarray(
            self.session.run([self._output_name], {self._input_name: batch})[0],
            dtype=np.float32,
        )
        row_sums = np.sum(raw, axis=-1)
        already_probs = (
            float(np.min(raw)) >= -1e-5
            and float(np.max(raw)) <= 1.0 + 1e-5
            and bool(np.all(np.abs(row_sums - 1.0) <= 1e-2))
        )
        probs = raw if already_probs else _softmax(raw)
        out: list[dict[str, float]] = []
        for row in probs:
            out.append(
                {
                    self.labels[i]: float(row[i])
                    for i in range(min(len(self.labels), len(row)))
                }
            )
        return out


class VisionAttributeAnalyzer:
    """Runs head focal detection + medium, rating, monochrome, AI, completeness & aesthetic."""

    def __init__(self, runtime: EmbeddingConfig) -> None:
        self.device = resolve_device(runtime.device)
        self.head_detector = HeadFocalDetector(self.device)
        self.medium_clf = _OnnxImageClassifier(
            self.device,
            "deepghs/anime_classification",
            "caformer_s36_v1.5_focal/model.onnx",
            "caformer_s36_v1.5_focal/meta.json",
            _MEDIUM_LABELS,
            384,
        )
        self.rating_clf = _OnnxImageClassifier(
            self.device,
            "deepghs/anime_rating",
            "caformer_s36_v1_pruned_ls0.1/model.onnx",
            "caformer_s36_v1_pruned_ls0.1/meta.json",
            _RATING_LABELS,
            384,
        )
        self.mono_clf = _OnnxImageClassifier(
            self.device,
            "deepghs/monochrome_detect",
            "caformer_s36_plus_safe2/model.onnx",
            "caformer_s36_plus_safe2/meta.json",
            _MONO_LABELS,
            384,
        )
        self.ai_clf = _OnnxImageClassifier(
            self.device,
            "deepghs/anime_ai_check",
            "caformer_s36_plus_sce/model.onnx",
            "caformer_s36_plus_sce/meta.json",
            _AI_LABELS,
            384,
        )
        self.completeness_clf = _OnnxImageClassifier(
            self.device,
            "deepghs/anime_completeness",
            "caformer_s36_v2.2/model.onnx",
            "caformer_s36_v2.2/meta.json",
            _COMPLETENESS_LABELS,
            384,
        )
        self.aesthetic_clf = _OnnxImageClassifier(
            self.device,
            "deepghs/anime_aesthetic",
            "swinv2pv3_v0_448_ls0.2_x/model.onnx",
            "swinv2pv3_v0_448_ls0.2_x/meta.json",
            _AESTHETIC_LABELS,
            448,
        )

    def close(self) -> None:
        for component in (
            self.head_detector,
            self.medium_clf,
            self.rating_clf,
            self.mono_clf,
            self.ai_clf,
            self.completeness_clf,
            self.aesthetic_clf,
        ):
            component.close()

    def analyze_batch(
        self,
        items: Sequence[tuple[int, Image.Image]],
    ) -> list[dict[str, Any]]:
        """Analyze `(image_id, PIL.Image)` pairs and emit attribute update payloads."""
        if not items:
            return []
        images = [img for _, img in items]
        heads = self.head_detector.predict(images)
        mediums = self.medium_clf.predict_probs(images)
        ratings = self.rating_clf.predict_probs(images)
        monos = self.mono_clf.predict_probs(images)
        ai_probs = self.ai_clf.predict_probs(images)
        comps = self.completeness_clf.predict_probs(images)
        aesthetics = self.aesthetic_clf.predict_probs(images)

        updates: list[dict[str, Any]] = []
        for idx, (img_id, pil_img) in enumerate(items):
            head_res = heads[idx]
            med_dist = mediums[idx]
            rat_dist = ratings[idx]
            mono_dist = monos[idx]
            ai_dist = ai_probs[idx]
            comp_dist = comps[idx]
            aes_dist = aesthetics[idx]

            media_type = max(med_dist.items(), key=lambda kv: kv[1])[0]
            rating = max(rat_dist.items(), key=lambda kv: kv[1])[0]
            mono_score = float(mono_dist.get("monochrome", 0.0))
            is_mono = 1 if mono_score >= 0.55 else 0

            embedded_ai, embedded_source = extract_embedded_ai_metadata(pil_img)
            if embedded_ai:
                is_ai = 1
                ai_source = embedded_source or "metadata"
                ai_score = 0.99
            elif media_type in ("not_painting", "3d"):
                # `deepghs/anime_ai_check` is trained strictly on 2D anime illustrations
                # and emits false positives on real photos, BJD dolls, and 3D renders.
                is_ai = 0
                ai_source = "human"
                ai_score = 0.0
            else:
                ai_score = float(ai_dist.get("ai", 0.0))
                is_ai = 1 if ai_score >= 0.70 else 0
                ai_source = "model" if is_ai else "human"

            aes_num = sum(
                float(prob) * _AESTHETIC_WEIGHTS[label]
                for label, prob in aes_dist.items()
                if label in _AESTHETIC_WEIGHTS
            )
            aes_den = sum(
                float(prob)
                for label, prob in aes_dist.items()
                if label in _AESTHETIC_WEIGHTS
            )
            if aes_den > 1e-6:
                aesthetic = _clamp01(aes_num / aes_den)
            else:
                polished_p = float(comp_dist.get("polished", 0.0))
                rough_p = float(comp_dist.get("rough", 0.0))
                illus_p = float(med_dist.get("illustration", 0.0))
                aesthetic = _clamp01(0.55 * polished_p + 0.30 * illus_p + 0.15 * (1.0 - rough_p))

            attr_tags: dict[str, int] = {
                f"medium:{media_type}": CAT_MEDIUM,
                f"rating:{rating}": CAT_RATING,
                ("ai_generated" if is_ai else "hand_drawn"): CAT_ATTRIBUTE,
            }
            if is_mono:
                attr_tags["monochrome"] = CAT_ATTRIBUTE
            comp_label = max(comp_dist.items(), key=lambda kv: kv[1])[0]
            if comp_label in {"polished", "rough"}:
                attr_tags[comp_label] = CAT_ATTRIBUTE

            updates.append(
                {
                    "image_id": int(img_id),
                    "media_type": media_type,
                    "rating": rating,
                    "is_monochrome": is_mono,
                    "monochrome_score": round(mono_score, 4),
                    "is_ai": is_ai,
                    "ai_source": ai_source,
                    "ai_score": round(ai_score, 4),
                    "aesthetic_score": round(aesthetic, 4),
                    "shot_type": head_res.shot_type,
                    "focal_x": round(head_res.focal_x, 4),
                    "focal_y": round(head_res.focal_y, 4),
                    "character_crop_json": head_res.crop_json,
                    "vision_meta_version": VISION_META_VERSION,
                    "attr_tags": attr_tags,
                }
            )
        return updates
