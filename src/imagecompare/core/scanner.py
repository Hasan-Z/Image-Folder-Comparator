"""Folder scanning: find candidate image files honoring recursion and
include/exclude glob filters.

Rules (see project spec):
- If recursive=True, nested folders are walked and merged into one flat set.
- include_patterns: if non-empty, only files matching at least one pattern
  are kept. If empty, all files with a supported image extension are kept.
- exclude_patterns: files matching any of these are dropped, regardless of
  the include result. Exclude always wins over include.
- Matching is case-insensitive and uses shell-style glob syntax (fnmatch),
  not regex.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

DEFAULT_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tiff",
    ".tif",
}


@dataclass(frozen=True)
class ScannedImage:
    """A single discovered image file."""

    path: Path
    relative_path: str  # path relative to the scanned root, for display

    @property
    def name(self) -> str:
        return self.path.name


def _normalize_patterns(patterns: str | list[str] | None) -> list[str]:
    """Accept either a space-separated string (as typed in the UI) or a
    list, and normalize to a lowercase list of glob patterns."""
    if not patterns:
        return []
    if isinstance(patterns, str):
        parts = patterns.split()
    else:
        parts = list(patterns)
    return [p.lower().strip() for p in parts if p.strip()]


def _matches_any(filename: str, patterns: list[str]) -> bool:
    lower = filename.lower()
    return any(fnmatch(lower, pat) for pat in patterns)


def scan_folder(
    root: str | Path,
    *,
    recursive: bool = False,
    include_patterns: str | list[str] | None = None,
    exclude_patterns: str | list[str] | None = None,
) -> list[ScannedImage]:
    """Scan a folder for image files.

    Raises FileNotFoundError / NotADirectoryError if the path is invalid,
    so callers (API layer) can turn that into a clean 4xx response.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    includes = _normalize_patterns(include_patterns)
    excludes = _normalize_patterns(exclude_patterns)

    results: list[ScannedImage] = []

    if recursive:
        walker: Iterable[tuple[str, list[str], list[str]]] = os.walk(root_path)
    else:
        walker = [(str(root_path), [], [p.name for p in root_path.iterdir() if p.is_file()])]

    for dirpath, _dirnames, filenames in walker:
        for fname in filenames:
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()
            if ext not in DEFAULT_IMAGE_EXTENSIONS:
                continue

            if includes and not _matches_any(fname, includes):
                continue
            if excludes and _matches_any(fname, excludes):
                continue

            results.append(
                ScannedImage(
                    path=fpath,
                    relative_path=str(fpath.relative_to(root_path)),
                )
            )

    results.sort(key=lambda im: im.relative_path.lower())
    return results


def count_matches(
    root: str | Path,
    *,
    recursive: bool = False,
    include_patterns: str | list[str] | None = None,
    exclude_patterns: str | list[str] | None = None,
) -> int:
    """Cheap helper for the UI's live preview counter."""
    return len(
        scan_folder(
            root,
            recursive=recursive,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    )
