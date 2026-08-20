from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

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


def scan_images(roots: Iterable[Path], recursive: bool = True) -> Iterator[tuple[Path, Path]]:
    seen: set[str] = set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"图片目录不存在：{root}")
        iterator = root.rglob("*") if recursive else root.glob("*")
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = canonical_path(path)
            if key in seen:
                continue
            seen.add(key)
            yield root, path


def load_rgb(path: Path | str) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.seek(0)
            return ImageOps.exif_transpose(image).convert("RGB")
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


def make_thumbnail(source: Path, destination: Path, size: int = 512) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = load_rgb(source)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    image.save(destination, "WEBP", quality=88, method=6)
    return destination

