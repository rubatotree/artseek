"""Multi-backend GPU/ONNX taggers: Camie v2, PixAI v0.9, PixAI v1.0, WD v3, and fusion."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from PIL import Image

from ..compute import configure_inference, release_accelerator, resolve_device, resolve_dtype
from ..config import EmbeddingConfig, TaggerConfig
from ..core.query_plan import normalize_tag
from ..image_io import composite_on_white
from .tag_postprocess import (
    CAMIE_CATEGORY_MAP,
    CAT_ARTIST_STYLE,
    CAT_CHARACTER,
    CAT_COPYRIGHT,
    CAT_GENERAL,
    CAT_META,
    CAT_RATING,
    PIXAI10_CATEGORY_MAP,
    RATING_CANONICAL,
    PredictedTags,
    mcut_threshold,
    merge_predicted_tags,
)


def _cached_or_download(repo_id: str, filename: str) -> Path:
    """Resolve a Hugging Face file from local cache first, avoiding network when present."""
    local_dir = Path(repo_id)
    if local_dir.is_dir():
        candidate = local_dir / filename
        if candidate.is_file():
            return candidate
    cached = try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str) and Path(cached).is_file():
        return Path(cached)
    return Path(hf_hub_download(repo_id, filename))


def create_ort_session(model_path: Path, device: torch.device) -> Any:
    """Create an ONNX Runtime session sharing PyTorch's CUDA runtime when available."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if device.type == "cuda":
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(directory="")
            except Exception:
                pass
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)


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


def _to_rgb_chw_float(
    item: Image.Image | np.ndarray,
    device: torch.device,
    *,
    non_blocking: bool,
) -> torch.Tensor:
    if isinstance(item, np.ndarray):
        arr = np.array(item, dtype=np.uint8, copy=not item.flags.writeable)
    else:
        rgb = composite_on_white(item)
        arr = np.array(rgb, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(arr).to(device=device, non_blocking=non_blocking)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0


def _pad_square_and_resize(
    chw: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    pad_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mode: str = "bicubic",
) -> torch.Tensor:
    cur_h, cur_w = int(chw.shape[-2]), int(chw.shape[-1])
    if cur_h != cur_w:
        max_dim = max(cur_h, cur_w)
        pad_top = (max_dim - cur_h) // 2
        pad_bottom = max_dim - cur_h - pad_top
        pad_left = (max_dim - cur_w) // 2
        pad_right = max_dim - cur_w - pad_left
        if pad_rgb[0] == pad_rgb[1] == pad_rgb[2]:
            chw = F.pad(
                chw,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=float(pad_rgb[0]),
            )
        else:
            canvas = torch.empty(
                (1, 3, max_dim, max_dim), device=chw.device, dtype=chw.dtype
            )
            for c_idx, val in enumerate(pad_rgb):
                canvas[:, c_idx, :, :] = float(val)
            canvas[:, :, pad_top : pad_top + cur_h, pad_left : pad_left + cur_w] = chw
            chw = canvas
    if int(chw.shape[-2]) != target_h or int(chw.shape[-1]) != target_w:
        use_aa = mode in {"bilinear", "bicubic"}
        chw = F.interpolate(
            chw,
            size=(target_h, target_w),
            mode=mode,
            align_corners=False if use_aa else None,
            antialias=use_aa,
        )
    return chw


class WDTagger:
    """GPU-accelerated WD-Tagger v3 with BGR white-pad preprocessing and vectorized topk."""

    model_versions: dict[str, int] = {"wd_v3": 1}

    def __init__(self, config: TaggerConfig, runtime: EmbeddingConfig):
        try:
            import timm
            from timm.data import resolve_model_data_config
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
        labels_path = (
            snapshot / "selected_tags.csv"
            if snapshot is not None
            else _cached_or_download(config.model, "selected_tags.csv")
        )
        self.names, self.categories = self._load_labels(Path(labels_path))
        self._input_h, self._input_w, _, self._interp_mode, self._mean, self._std = (
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
        """WD-Tagger preprocessing: white-pad to square, bicubic resize, RGB->BGR, normalize."""
        target_h = self._input_h
        target_w = self._input_w
        mode = self._interp_mode
        non_blocking = self.device.type == "cuda"

        prepared: list[torch.Tensor] = []
        for item in images:
            chw = _to_rgb_chw_float(item, self.device, non_blocking=non_blocking)
            chw = _pad_square_and_resize(
                chw, target_h, target_w, pad_rgb=(1.0, 1.0, 1.0), mode=mode
            )
            prepared.append(chw)
            time.sleep(0)

        batch = torch.cat(prepared, dim=0)
        batch = batch[:, [2, 1, 0], :, :]
        mean = torch.tensor(self._mean, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._std, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
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
        cats = self.categories
        results: list[PredictedTags] = []
        for row_scores, row_indices in zip(scores_cpu, indices_cpu, strict=True):
            selected_tags: list[str] = []
            selected_scores: dict[str, float] = {}
            selected_cats: dict[str, int] = {}
            for score, idx in zip(row_scores, row_indices, strict=True):
                if score < threshold:
                    break
                name = names[idx]
                selected_tags.append(name)
                selected_scores[name] = round(float(score), 4)
                selected_cats[name] = int(cats[idx])
            results.append(
                PredictedTags(
                    tags=selected_tags,
                    scores=selected_scores,
                    categories=selected_cats,
                )
            )
        return results


class _CamieViTWrapper(nn.Module):
    """Exact ViT token/CLS extractor matching Camais03/camie-tagger-v2."""

    def __init__(self, vit_model: nn.Module) -> None:
        super().__init__()
        self.vit = vit_model
        self.embed_dim = int(vit_model.embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        x = self.vit.patch_embed(x)
        cls_tok = self.vit.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tok, x), dim=1)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed[:, : x.size(1), :]
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        cls_final = x[:, 0]
        patch_tokens = x[:, 1:]
        bsz, n_tok, dim = patch_tokens.shape
        side = int(n_tok**0.5)
        patch_features = patch_tokens.permute(0, 2, 1).reshape(bsz, dim, side, side)
        return patch_features, cls_final


class _CamieNet(nn.Module):
    """PyTorch implementation of Camie Tagger v2 (ViT + cross-attention tag refinement)."""

    def __init__(
        self,
        total_tags: int,
        *,
        backbone_name: str = "vit_base_patch16_384",
        img_size: int = 512,
        num_heads: int = 16,
        tag_context_size: int = 256,
    ) -> None:
        super().__init__()
        import timm

        vit_model = timm.create_model(
            backbone_name,
            pretrained=False,
            img_size=img_size,
            num_classes=0,
        )
        self.backbone = _CamieViTWrapper(vit_model)
        self.embedding_dim = self.backbone.embed_dim
        self.total_tags = int(total_tags)
        self.tag_context_size = int(tag_context_size)
        self.tag_embedding = nn.Embedding(self.total_tags, self.embedding_dim)
        self.tag_bias = nn.Parameter(torch.zeros(self.total_tags))
        self.image_token_proj = nn.Identity()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_map, cls_token = self.backbone(x)
        image_tokens = self.image_token_proj(patch_map).flatten(2).transpose(1, 2)
        global_features = 0.5 * (
            image_tokens.mean(dim=1, dtype=image_tokens.dtype) + cls_token
        )
        dtype = global_features.dtype
        tag_weights = self.tag_embedding.weight.to(dtype)
        tag_bias = self.tag_bias.to(dtype)
        initial_logits = global_features @ tag_weights.t() + tag_bias
        k = min(self.tag_context_size, self.total_tags)
        _, candidate_indices = torch.topk(
            torch.sigmoid(initial_logits.float()), k=k, dim=1, largest=True, sorted=True
        )
        tag_embeddings = self.tag_embedding(candidate_indices).to(dtype)
        attended, _ = self.cross_attention(
            query=tag_embeddings, key=image_tokens, value=image_tokens
        )
        attended = self.cross_norm(attended)
        cand_emb = self.tag_embedding(candidate_indices).to(dtype)
        candidate_logits = (attended * cand_emb).sum(dim=-1).to(initial_logits.dtype)
        refined = initial_logits.clone()
        refined.scatter_(1, candidate_indices, candidate_logits)
        return refined


class CamieTaggerV2:
    """Camie Tagger v2 in native PyTorch FP16 (512x512, 70k tags across 7 categories)."""

    model_versions: dict[str, int] = {"camie_v2": 1}
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, config: TaggerConfig, runtime: EmbeddingConfig):
        from safetensors.torch import load_file

        self.config = config
        self.device = resolve_device(runtime.device)
        self.dtype = resolve_dtype(runtime.dtype, self.device)
        configure_inference(self.device)

        repo_id = config.camie_model
        meta_path = _cached_or_download(repo_id, "camie-tagger-v2-metadata.json")
        weights_path = _cached_or_download(repo_id, "camie-tagger-v2.safetensors")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_info = meta.get("model_info", {})
        dataset_info = meta.get("dataset_info", {})
        mapping = dataset_info.get("tag_mapping", {})
        idx_to_tag = mapping.get("idx_to_tag") or {}
        tag_to_cat = mapping.get("tag_to_category") or {}
        if not idx_to_tag and mapping.get("tag_to_idx"):
            idx_to_tag = {str(v): k for k, v in mapping["tag_to_idx"].items()}

        total_tags = int(dataset_info.get("total_tags") or len(idx_to_tag))
        self.names = [str(idx_to_tag.get(str(i), f"tag_{i}")) for i in range(total_tags)]
        self.raw_categories = [
            str(tag_to_cat.get(name, "general")).casefold() for name in self.names
        ]
        self.categories = [
            CAMIE_CATEGORY_MAP.get(cat, CAT_GENERAL) for cat in self.raw_categories
        ]
        self._img_size = int(model_info.get("img_size", 512))

        net = _CamieNet(
            total_tags=total_tags,
            backbone_name=str(model_info.get("backbone", "vit_base_patch16_384")),
            img_size=self._img_size,
            num_heads=int(model_info.get("num_attention_heads", 16)),
            tag_context_size=int(model_info.get("tag_context_size", 256)),
        )
        state = load_file(str(weights_path), device="cpu")
        net.load_state_dict(state, strict=True)
        self.model = net.eval().to(device=self.device, dtype=self.dtype)

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            model.to("cpu")
        except Exception:
            pass
        self.model = None  # type: ignore[assignment]
        release_accelerator()

    def _preprocess_images_gpu(
        self, images: Sequence[Image.Image | np.ndarray]
    ) -> torch.Tensor:
        non_blocking = self.device.type == "cuda"
        prepared: list[torch.Tensor] = []
        for item in images:
            chw = _to_rgb_chw_float(item, self.device, non_blocking=non_blocking)
            chw = _pad_square_and_resize(
                chw,
                self._img_size,
                self._img_size,
                pad_rgb=self._MEAN,
                mode="bicubic",
            )
            prepared.append(chw)
            time.sleep(0)
        batch = torch.cat(prepared, dim=0)
        mean = torch.tensor(self._MEAN, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._STD, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        return ((batch - mean) / std).to(dtype=self.dtype)

    def predict(
        self, images: Sequence[Image.Image | np.ndarray]
    ) -> list[PredictedTags]:
        if not images:
            return []
        tensors = self._preprocess_images_gpu(images)
        with torch.inference_mode():
            logits = self.model(tensors)
            probs = torch.sigmoid(logits.float())
            top_k = min(256, probs.shape[-1])
            top_scores, top_indices = torch.topk(probs, k=top_k, dim=-1, largest=True, sorted=True)
            if top_scores.is_cuda:
                ev = torch.cuda.Event()
                ev.record()
                while not ev.query():
                    time.sleep(0.001)
            scores_cpu = top_scores.cpu().tolist()
            indices_cpu = top_indices.cpu().tolist()

        # Camie calibrated thresholds (~0.49 macro opt, ~0.61 micro opt)
        gen_th = max(0.45, float(self.config.general_threshold))
        char_th = max(0.55, float(self.config.character_threshold))
        copy_th = max(0.55, float(self.config.copyright_threshold))
        meta_th = max(0.45, float(self.config.meta_threshold))
        max_tags = max(1, int(self.config.max_tags))

        results: list[PredictedTags] = []
        for row_scores, row_indices in zip(scores_cpu, indices_cpu, strict=True):
            general_probs = [
                float(s)
                for s, idx in zip(row_scores, row_indices, strict=True)
                if self.categories[idx] == CAT_GENERAL and float(s) >= 0.30
            ]
            eff_gen_th = (
                mcut_threshold(general_probs, min_threshold=gen_th, max_threshold=0.65)
                if self.config.mcut_enabled and len(general_probs) >= 6
                else gen_th
            )

            best_rating: tuple[float, str] | None = None
            selected_tags: list[str] = []
            selected_scores: dict[str, float] = {}
            selected_cats: dict[str, int] = {}

            for score, idx in zip(row_scores, row_indices, strict=True):
                cat = self.categories[idx]
                raw_cat = self.raw_categories[idx]
                name = self.names[idx]
                prob = float(score)

                if cat == CAT_RATING:
                    canon = RATING_CANONICAL.get(name.casefold())
                    if canon and (best_rating is None or prob > best_rating[0]):
                        best_rating = (prob, canon)
                    continue
                if raw_cat == "year" or (
                    cat == CAT_CHARACTER and not self.config.include_character_tags
                ):
                    continue
                if cat == CAT_ARTIST_STYLE:
                    # Skip low-confidence artist guesses to avoid polluting artist provenance.
                    continue

                th = (
                    char_th
                    if cat == CAT_CHARACTER
                    else (
                        copy_th
                        if cat == CAT_COPYRIGHT
                        else (meta_th if cat == CAT_META else eff_gen_th)
                    )
                )
                if prob < th:
                    continue
                norm = normalize_tag(name)
                if not norm or norm in selected_scores:
                    continue
                selected_tags.append(norm)
                selected_scores[norm] = round(prob, 4)
                selected_cats[norm] = cat
                if len(selected_tags) >= max_tags:
                    break

            results.append(
                PredictedTags(
                    tags=selected_tags,
                    scores=selected_scores,
                    categories=selected_cats,
                    rating=best_rating[1] if best_rating else None,
                )
            )
        return results


class PixAITaggerV09:
    """PixAI Tagger v0.9 via ONNX Runtime CUDA (448x448, character + IP mapping)."""

    model_versions: dict[str, int] = {"pixai_v09": 1}

    def __init__(self, config: TaggerConfig, runtime: EmbeddingConfig):
        self.config = config
        self.device = resolve_device(runtime.device)
        repo_id = config.pixai09_model
        model_path = _cached_or_download(repo_id, "model.onnx")
        tags_path = _cached_or_download(repo_id, "selected_tags.csv")

        self.session = create_ort_session(model_path, self.device)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        self.names, self.categories, self.char_ips = self._load_tags(tags_path)

    @staticmethod
    def _load_tags(path: Path) -> tuple[list[str], list[int], list[tuple[str, ...]]]:
        names: list[str] = []
        categories: list[int] = []
        char_ips: list[tuple[str, ...]] = []
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                names.append(row["name"])
                categories.append(int(row["category"]))
                raw_ips = row.get("ips") or "[]"
                try:
                    parsed = json.loads(raw_ips)
                    ips = tuple(
                        normalize_tag(str(ip))
                        for ip in parsed
                        if normalize_tag(str(ip))
                    )
                except json.JSONDecodeError:
                    ips = ()
                char_ips.append(ips)
        return names, categories, char_ips

    def close(self) -> None:
        self.session = None

    @staticmethod
    def _preprocess_batch(images: Sequence[Image.Image | np.ndarray]) -> np.ndarray:
        tensors: list[np.ndarray] = []
        for item in images:
            if isinstance(item, np.ndarray):
                pil = Image.fromarray(item, mode="RGB")
            else:
                pil = composite_on_white(item)
            if pil.size != (448, 448):
                pil = pil.resize((448, 448), Image.Resampling.BILINEAR)
            arr = (np.asarray(pil, dtype=np.float32) / 255.0 - 0.5) / 0.5
            tensors.append(arr.transpose(2, 0, 1))
        return np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)

    def predict(
        self, images: Sequence[Image.Image | np.ndarray]
    ) -> list[PredictedTags]:
        if not images:
            return []
        batch = self._preprocess_batch(images)
        outputs = self.session.run([self._output_name], {self._input_name: batch})[0]
        # deepghs/pixai-tagger-v0.9-onnx outputs probabilities or logits; sigmoid if outside [0, 1]
        arr = np.asarray(outputs, dtype=np.float32)
        if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
            arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -30.0, 30.0)))

        gen_th = min(0.35, float(self.config.general_threshold))
        char_th = max(0.65, float(self.config.character_threshold))
        max_tags = max(1, int(self.config.max_tags))

        results: list[PredictedTags] = []
        for row in arr:
            order = np.argsort(-row)[: max_tags * 2]
            selected_tags: list[str] = []
            selected_scores: dict[str, float] = {}
            selected_cats: dict[str, int] = {}
            inferred_ips: dict[str, float] = {}

            for idx in order.tolist():
                prob = float(row[idx])
                cat = self.categories[idx]
                if cat == CAT_CHARACTER and not self.config.include_character_tags:
                    continue
                th = char_th if cat == CAT_CHARACTER else gen_th
                if prob < th:
                    continue
                name = normalize_tag(self.names[idx])
                if not name:
                    continue
                selected_tags.append(name)
                selected_scores[name] = round(prob, 4)
                selected_cats[name] = cat
                if cat == CAT_CHARACTER:
                    for ip in self.char_ips[idx]:
                        inferred_ips[ip] = max(inferred_ips.get(ip, 0.0), prob)
                if len(selected_tags) >= max_tags:
                    break

            for ip, ip_score in sorted(inferred_ips.items(), key=lambda x: -x[1]):
                if ip not in selected_scores:
                    selected_tags.insert(0, ip)
                    selected_scores[ip] = round(ip_score, 4)
                    selected_cats[ip] = CAT_COPYRIGHT

            results.append(
                PredictedTags(
                    tags=selected_tags[:max_tags],
                    scores=selected_scores,
                    categories=selected_cats,
                )
            )
        return results


class PixAITaggerV10:
    """Optional high-resolution PixAI Tagger v1.0 (1008x1008 SAM3, FP16 chunked)."""

    model_versions: dict[str, int] = {"pixai_v10": 1}

    def __init__(self, config: TaggerConfig, runtime: EmbeddingConfig):
        from transformers import AutoModel

        self.config = config
        self.device = resolve_device(runtime.device)
        self.dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else resolve_dtype(runtime.dtype, self.device)
        )
        configure_inference(self.device)
        repo_id = config.pixai10_model
        self.model = AutoModel.from_pretrained(
            repo_id,
            trust_remote_code=True,
            dtype=self.dtype,
        ).eval().to(self.device)
        cfg = self.model.config
        self.tags: list[str] = list(getattr(cfg, "tags", []))
        self.tags_split: list[tuple[str, int]] = [
            (str(c), int(n)) for c, n in getattr(cfg, "tags_split", [])
        ]
        self._img_size = int(getattr(cfg, "img_size", 1008))
        raw_th = getattr(cfg, "tags_best_threshold", None)
        if raw_th is not None:
            self._thresholds = [float(v) for v in raw_th]
        else:
            cat_th = getattr(cfg, "category_best_threshold", {}) or {}
            self._thresholds = []
            for cat, count in self.tags_split:
                self._thresholds.extend([float(cat_th.get(cat, 0.3))] * count)

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            model.to("cpu")
        except Exception:
            pass
        self.model = None
        release_accelerator()

    def _preprocess_one(self, item: Image.Image | np.ndarray) -> torch.Tensor:
        non_blocking = self.device.type == "cuda"
        chw = _to_rgb_chw_float(item, self.device, non_blocking=non_blocking)
        chw = _pad_square_and_resize(
            chw,
            self._img_size,
            self._img_size,
            pad_rgb=(0.0, 0.0, 0.0),
            mode="bicubic",
        )
        return ((chw - 0.5) / 0.5).to(dtype=self.dtype)

    def predict(
        self, images: Sequence[Image.Image | np.ndarray]
    ) -> list[PredictedTags]:
        if not images:
            return []
        results: list[PredictedTags] = []
        # Keep micro-batch <= 2 at 1008x1008 so 8 GB VRAM stays safe.
        for start in range(0, len(images), 2):
            chunk = images[start : start + 2]
            batch = torch.cat([self._preprocess_one(item) for item in chunk], dim=0)
            with torch.inference_mode():
                probs = torch.sigmoid(self.model(batch).float()).cpu().numpy()
            for row in probs:
                results.append(self._decode_row(row))
        return results

    def _decode_row(self, probs: np.ndarray) -> PredictedTags:
        selected: list[tuple[str, float, int]] = []
        best_rating: tuple[float, str] | None = None
        offset = 0
        for raw_cat, count in self.tags_split:
            cat_lower = raw_cat.casefold()
            mapped_cat = PIXAI10_CATEGORY_MAP.get(cat_lower, CAT_GENERAL)
            slice_probs = probs[offset : offset + count]
            for local_i, prob_val in enumerate(slice_probs.tolist()):
                idx = offset + local_i
                th = self._thresholds[idx] if idx < len(self._thresholds) else 0.3
                if prob_val < th:
                    continue
                tag = normalize_tag(self.tags[idx])
                if not tag:
                    continue
                if mapped_cat == CAT_RATING:
                    canon = RATING_CANONICAL.get(tag)
                    if canon and (best_rating is None or prob_val > best_rating[0]):
                        best_rating = (prob_val, canon)
                    continue
                if mapped_cat == CAT_CHARACTER and not self.config.include_character_tags:
                    continue
                selected.append((tag, prob_val, mapped_cat))
            offset += count

        selected.sort(key=lambda item: (-item[1], item[0]))
        selected = selected[: max(1, int(self.config.max_tags))]
        return PredictedTags(
            tags=[t for t, _, _ in selected],
            scores={t: round(float(s), 4) for t, s, _ in selected},
            categories={t: c for t, _, c in selected},
            rating=best_rating[1] if best_rating else None,
        )


class CompositeTagger:
    """Fuses multiple taggers (default: Camie Tagger v2 + PixAI Tagger v0.9)."""

    def __init__(self, taggers: Sequence[Any], *, max_tags: int = 80) -> None:
        self.taggers = list(taggers)
        self.max_tags = max(1, int(max_tags))
        versions: dict[str, int] = {}
        for tg in self.taggers:
            versions.update(getattr(tg, "model_versions", {}))
        self.model_versions = versions

    def predict(
        self, images: Sequence[Image.Image | np.ndarray]
    ) -> list[PredictedTags]:
        if not images:
            return []
        per_model = [tg.predict(images) for tg in self.taggers]
        out: list[PredictedTags] = []
        for idx in range(len(images)):
            preds = [model_res[idx] for model_res in per_model]
            out.append(merge_predicted_tags(preds, max_tags=self.max_tags))
        return out

    def close(self) -> None:
        for tg in self.taggers:
            closer = getattr(tg, "close", None)
            if callable(closer):
                closer()


def normalize_tagger_backend(backend: str | None) -> str:
    """Resolve user/config tagger backend aliases to a canonical identifier."""
    choice = str(backend or "camie+pixai0.9").strip().casefold()
    if choice in {"camie+pixai0.9", "default", "ensemble"}:
        return "camie+pixai0.9"
    if choice in {"camie", "camie_v2", "camiev2"}:
        return "camie"
    if choice in {"pixai0.9", "pixai_v09", "pixai09"}:
        return "pixai0.9"
    if choice in {"pixai1.0", "pixai_v10", "pixai10"}:
        return "pixai1.0"
    if choice in {"wd", "wd_v3", "wd3"}:
        return "wd_v3"
    raise ValueError(f"未知标签器后端：{choice}")


def model_versions_for_backend(backend: str | None) -> dict[str, int]:
    """Return expected `{model_key: version}` map without instantiating model weights."""
    canon = normalize_tagger_backend(backend)
    if canon == "camie+pixai0.9":
        return {"camie_v2": 1, "pixai_v09": 1}
    if canon == "camie":
        return {"camie_v2": 1}
    if canon == "pixai0.9":
        return {"pixai_v09": 1}
    if canon == "pixai1.0":
        return {"pixai_v10": 1}
    return {"wd_v3": 1}


def build_tagger(
    config: TaggerConfig,
    runtime: EmbeddingConfig,
    *,
    backend: str | None = None,
) -> Any | None:
    """Instantiate the configured tagger backend (or ensemble)."""
    if not config.enabled and backend is None:
        return None
    canon = normalize_tagger_backend(backend or config.backend)
    if canon == "camie+pixai0.9":
        return CompositeTagger(
            [
                CamieTaggerV2(config, runtime),
                PixAITaggerV09(config, runtime),
            ],
            max_tags=config.max_tags,
        )
    if canon == "camie":
        return CamieTaggerV2(config, runtime)
    if canon == "pixai0.9":
        return PixAITaggerV09(config, runtime)
    if canon == "pixai1.0":
        return PixAITaggerV10(config, runtime)
    return WDTagger(config, runtime)
