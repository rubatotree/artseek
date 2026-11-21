from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

PHASH_LENGTH = 16
PHASH_MAX_DISTANCE = 10
PHASH_ASPECT_MAX_RATIO = 1.25
_PHASH_WIDTH = 9
_PHASH_HEIGHT = 8
_PHASH_MIN_STD = 4.0

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".avif",
}


def canonical_path(path: Path | str) -> str:
    resolved = str(Path(path).expanduser().resolve())
    return os.path.normcase(resolved)


def point_id(path: Path | str) -> str:
    # Qdrant accepts UUIDs; a stable hash keeps rescans idempotent.
    digest = hashlib.sha256(canonical_path(path).encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def file_digest(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def perceptual_hash(source: Path | str | Image.Image) -> str:
    """64-bit difference hash. Empty string means the image is too flat to match."""
    image = source if isinstance(source, Image.Image) else load_rgb(source)
    gray = image.convert("L").resize(
        (_PHASH_WIDTH, _PHASH_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    pixels = np.asarray(gray, dtype=np.int16)
    if float(np.std(pixels)) < _PHASH_MIN_STD:
        return ""
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    value = 0
    for bit in bits.tolist():
        value = (value << 1) | int(bit)
    return f"{value:0{PHASH_LENGTH}x}"


def hamming_distance(left: str, right: str) -> int:
    if not left or not right:
        return 64
    if left == right:
        return 0
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 64


def aspect_ratio_compatible(
    width_a: int,
    height_a: int,
    width_b: int,
    height_b: int,
    *,
    max_ratio: float = PHASH_ASPECT_MAX_RATIO,
) -> bool:
    if min(width_a, height_a, width_b, height_b) <= 0:
        return True
    ratio_a = width_a / height_a
    ratio_b = width_b / height_b
    return max(ratio_a, ratio_b) / min(ratio_a, ratio_b) <= max_ratio


def scan_images(roots: Iterable[Path], recursive: bool = True) -> Iterator[tuple[Path, Path]]:
    seen: set[str] = set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"图片目录不存在：{root}")
        for path in _scan_root_images(root, recursive=recursive):
            key = canonical_path(path)
            if key not in seen:
                seen.add(key)
                yield root, path


def _scan_root_images(root: Path, *, recursive: bool) -> Iterator[Path]:
    """Yield images with scandir, avoiding Path.rglob's per-entry Path/stat overhead."""
    directories = [root]
    while directories:
        directory = directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        yield path
                    elif recursive and entry.is_dir(follow_symlinks=False):
                        directories.append(path)
        except OSError:
            # Match glob/rglob's practical behavior for directories that disappear mid-scan.
            continue


def composite_on_white(image: Image.Image) -> Image.Image:
    """Flatten alpha/palette transparency onto a solid white RGB canvas."""
    has_alpha = image.mode in {"RGBA", "LA", "PA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image if image.mode == "RGB" else image.convert("RGB")


def load_rgb(path: Path | str) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.seek(0)
            return composite_on_white(ImageOps.exif_transpose(image))
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"无法读取图片：{path}") from exc


def image_metadata(path: Path, image: Image.Image) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "path_key": canonical_path(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
        "width": image.width,
        "height": image.height,
    }


def _build_srgb_to_linear_lut() -> np.ndarray:
    u = np.arange(256, dtype=np.float32) / 255.0
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4).astype(np.float32)


_SRGB_TO_LINEAR_LUT = _build_srgb_to_linear_lut()


def _linear_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear, 0.0, 1.0)
    srgb = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(np.maximum(clipped, 0.0031308), 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(srgb * 255.0), 0.0, 255.0).astype(np.uint8)


def _resize_float_channel(
    channel: np.ndarray,
    size: tuple[int, int],
    *,
    pre_size: tuple[int, int] | None,
) -> np.ndarray:
    plane = Image.fromarray(channel, mode="F")
    if pre_size is not None:
        plane = plane.resize(pre_size, Image.Resampling.BOX)
    if plane.size != size:
        plane = plane.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(plane, dtype=np.float32)


def downscale_image(
    image: Image.Image,
    max_size: int | tuple[int, int],
    *,
    linear_light: bool = True,
) -> Image.Image:
    """Downscale an image with anti-aliased progressive linear-light resampling.

    Uses box pre-filtering for large downscale ratios (> 2.5x) to prevent
    Lanczos ringing on sharp line art, linear-light sRGB conversion for energy
    conservation, and premultiplied alpha for clean transparent edges.
    """
    max_w, max_h = (max_size, max_size) if isinstance(max_size, int) else max_size
    if max_w <= 0 or max_h <= 0:
        raise ValueError("max_size must be positive")

    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0 or (src_w <= max_w and src_h <= max_h):
        return image

    scale = min(max_w / src_w, max_h / src_h)
    target_w = min(max_w, max(1, round(src_w * scale)))
    target_h = min(max_h, max(1, round(src_h * scale)))
    target_size = (target_w, target_h)

    has_alpha = image.mode in {"RGBA", "LA", "PA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if image.mode not in ("RGB", "RGBA"):
        working = image.convert("RGBA" if has_alpha else "RGB")
    else:
        working = image

    ratio = max(src_w / target_w, src_h / target_h)
    pre_size: tuple[int, int] | None = None
    if ratio > 2.5:
        pre_w = min(src_w, target_w * 2)
        pre_h = min(src_h, target_h * 2)
        if (pre_w, pre_h) != (src_w, src_h) and (pre_w, pre_h) != target_size:
            pre_size = (pre_w, pre_h)

    if not linear_light:
        out = working.resize(pre_size, Image.Resampling.BOX) if pre_size else working
        return out.resize(target_size, Image.Resampling.LANCZOS) if out.size != target_size else out

    arr = np.asarray(working, dtype=np.uint8)
    if working.mode == "RGBA":
        alpha_f = (arr[:, :, 3].astype(np.float32)) / 255.0
        rgb_lin = _SRGB_TO_LINEAR_LUT[arr[:, :, :3]] * alpha_f[:, :, None]
        out_channels = [
            _resize_float_channel(rgb_lin[:, :, idx], target_size, pre_size=pre_size)
            for idx in range(3)
        ]
        out_alpha = np.clip(
            _resize_float_channel(alpha_f, target_size, pre_size=pre_size),
            0.0,
            1.0,
        )
        rgb_stack = np.stack(out_channels, axis=-1)
        safe_alpha = np.where(out_alpha > 1e-6, out_alpha, 1.0)[:, :, None]
        unpremul = np.where(out_alpha[:, :, None] > 1e-6, rgb_stack / safe_alpha, 0.0)
        rgb_u8 = _linear_to_srgb_u8(unpremul)
        alpha_u8 = np.clip(np.rint(out_alpha * 255.0), 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(np.dstack((rgb_u8, alpha_u8)), mode="RGBA")

    rgb_lin = _SRGB_TO_LINEAR_LUT[arr[:, :, :3]]
    out_channels = [
        _resize_float_channel(rgb_lin[:, :, idx], target_size, pre_size=pre_size)
        for idx in range(3)
    ]
    rgb_u8 = _linear_to_srgb_u8(np.stack(out_channels, axis=-1))
    return Image.fromarray(rgb_u8, mode="RGB")


def make_thumbnail(source: Path, destination: Path, size: int = 512) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = downscale_image(load_rgb(source), size)
    image.save(destination, "WEBP", quality=88, method=6)
    return destination


def make_search_image(source: Path, destination: Path, size: int = 1200) -> Path:
    """JPEG copy small enough for SauceNAO/ascii2d upload forms."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = downscale_image(load_rgb(source), size)
    image.save(destination, "JPEG", quality=90, optimize=True)
    return destination
