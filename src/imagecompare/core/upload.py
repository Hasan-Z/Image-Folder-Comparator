"""Framework-agnostic helpers for the drag-and-drop folder upload
endpoint.

Browsers deliberately hide the real filesystem path of dragged/dropped
files (the same sandboxing reason a web page can't show a native "Open
Folder" dialog with real paths) -- so drag-and-drop has to actually
upload the folder's file contents to the server, which then treats the
resulting server-side directory exactly like any other scanned folder.

This module holds the two pieces of that flow that don't need FastAPI:
- sanitize_relative_path(): turns a browser-supplied relative path into
  a safe one, rejecting path traversal.
- cleanup_old_uploads(): prunes old upload directories so repeated
  drag-and-drop usage doesn't grow disk usage unbounded.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path


class InvalidRelativePathError(ValueError):
    """Raised when an uploaded file's claimed relative path is unsafe
    (e.g. contains '..' components) or empty."""


def sanitize_relative_path(raw: str) -> Path:
    """Turn a browser-supplied relative path (e.g. from
    `FormData.append(name, file, relativePath)`, which preserves forward
    slashes) into a safe, relative Path with no '..' traversal and no
    empty/'.' segments.

    Raises InvalidRelativePathError if the path is unsafe or empty.
    """
    normalized = raw.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p not in ("", ".")]
    if not parts:
        raise InvalidRelativePathError(f"Empty relative path: {raw!r}")
    if any(p == ".." for p in parts):
        raise InvalidRelativePathError(f"Path traversal not allowed: {raw!r}")
    return Path(*parts)


def cleanup_old_uploads(uploads_dir: str | Path, max_age_seconds: float) -> int:
    """Remove subdirectories of `uploads_dir` whose modification time is
    older than `max_age_seconds`. Returns the number removed. Safe to
    call even if `uploads_dir` doesn't exist yet."""
    uploads_dir = Path(uploads_dir)
    if not uploads_dir.exists():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for child in uploads_dir.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
