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


def file_digest(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


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
