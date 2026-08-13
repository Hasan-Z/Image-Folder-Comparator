"""Perceptual hashing utilities.

Provides:
- compute_hash(): a single perceptual hash (pHash by default) for an image.
- similarity_from_hashes(): hamming-distance-based similarity percentage.
- rotated_hashes(): precompute hashes at every rotation angle for the
  "Deep Rotation" feature, so each image is only rotated/hashed once
  regardless of how many candidates it's compared against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

HashAlgo = str  # "phash" | "dhash" | "ahash"

_ALGO_FUNCS = {
    "phash": imagehash.phash,
    "dhash": imagehash.dhash,
    "ahash": imagehash.average_hash,
}


def compute_hash(
    image_path: str | Path,
    *,
    algo: HashAlgo = "phash",
    hash_size: int = 16,
) -> imagehash.ImageHash:
    """Compute a perceptual hash for a single image file."""
    if algo not in _ALGO_FUNCS:
        raise ValueError(f"Unknown hash algorithm: {algo!r}")
    with Image.open(image_path) as raw_img:
        img = raw_img.convert("RGB")
        return _ALGO_FUNCS[algo](img, hash_size=hash_size)


def hash_from_image(
    img: Image.Image,
    *,
    algo: HashAlgo = "phash",
    hash_size: int = 16,
) -> imagehash.ImageHash:
    """Compute a perceptual hash for an already-open PIL image (avoids
    re-reading from disk during rotation sweeps)."""
    if algo not in _ALGO_FUNCS:
        raise ValueError(f"Unknown hash algorithm: {algo!r}")
    return _ALGO_FUNCS[algo](img.convert("RGB"), hash_size=hash_size)


def similarity_from_hashes(
    hash_a: imagehash.ImageHash,
    hash_b: imagehash.ImageHash,
) -> float:
    """Convert hamming distance between two hashes into a 0-100 similarity
    percentage. Distance of 0 -> 100%, distance == full bit length -> 0%.
    """
    distance = hash_a - hash_b
    max_distance = hash_a.hash.size  # number of bits in the hash
    if max_distance == 0:
        return 100.0
    similarity = (1 - (distance / max_distance)) * 100
    return max(0.0, min(100.0, similarity))


@dataclass(frozen=True)
class RotationSweepResult:
    best_similarity: float
    best_angle: int


def rotated_hashes(
    image_path: str | Path,
    *,
    algo: HashAlgo = "phash",
    hash_size: int = 16,
    step_degrees: int = 1,
) -> list[imagehash.ImageHash]:
    """Precompute a hash at every rotation angle from 0 up to (but not
    including) 360 degrees, in steps of `step_degrees`.

    Rotation uses a fixed canvas (expand=False) so every rotated hash has
    the same dimensions/comparability as the original.
    """
    if step_degrees <= 0:
        raise ValueError("step_degrees must be positive")

    hashes: list[imagehash.ImageHash] = []
    with Image.open(image_path) as raw_img:
        img = raw_img.convert("RGB")
        for angle in range(0, 360, step_degrees):
            rotated = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            hashes.append(hash_from_image(rotated, algo=algo, hash_size=hash_size))
    return hashes


def best_rotation_similarity(
    base_hash: imagehash.ImageHash,
    candidate_hashes: list[imagehash.ImageHash],
    *,
    step_degrees: int = 1,
) -> RotationSweepResult:
    """Compare a fixed base hash against every precomputed rotated hash of
    the other image, returning the best similarity found and the angle
    that produced it.
    """
    best_similarity = -1.0
    best_angle = 0
    for idx, cand in enumerate(candidate_hashes):
        sim = similarity_from_hashes(base_hash, cand)
        if sim > best_similarity:
            best_similarity = sim
            best_angle = idx * step_degrees
    return RotationSweepResult(best_similarity=best_similarity, best_angle=best_angle)
