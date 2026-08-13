"""Thumbnail generation, cached on disk keyed by file path + mtime so
re-runs against unchanged folders are fast.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from imagecompare.config import settings


def _cache_key(image_path: Path) -> str:
    stat = image_path.stat()
    raw = f"{image_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def get_or_create_thumbnail(image_path: str | Path, cache_dir: str | Path) -> Path:
    """Return the path to a cached thumbnail for `image_path`, generating
    it if it doesn't already exist."""
    image_path = Path(image_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(image_path)
    thumb_path = cache_dir / f"{key}.jpg"

    if thumb_path.exists():
        return thumb_path

    with Image.open(image_path) as raw_img:
        img = raw_img.convert("RGB")
        img.thumbnail(settings.thumbnail_size)
        img.save(thumb_path, "JPEG", quality=85)

    return thumb_path
