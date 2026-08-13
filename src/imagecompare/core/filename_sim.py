"""Filename similarity scoring.

Compares filenames (ignoring extension) using difflib's SequenceMatcher,
returning a 0-100 percentage. This is an independent secondary signal
shown alongside visual similarity — not a filter by itself.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path


def _stem(filename: str) -> str:
    return Path(filename).stem.lower()


def filename_similarity(name_a: str, name_b: str) -> float:
    """Return similarity percentage (0-100) between two filenames,
    comparing stems only (extension ignored)."""
    a, b = _stem(name_a), _stem(name_b)
    if not a and not b:
        return 100.0
    ratio = SequenceMatcher(None, a, b).ratio()
    return round(ratio * 100, 2)
