"""MD5-based exact-duplicate detection.

Exact byte-for-byte duplicates short-circuit the more expensive visual
comparison — if two files share an MD5, they're the same file and score
100% automatically.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def compute_md5(file_path: str | Path) -> str:
    """Compute the MD5 hex digest of a file's bytes, streaming so large
    files don't need to be loaded fully into memory."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            md5.update(chunk)
    return md5.hexdigest()
