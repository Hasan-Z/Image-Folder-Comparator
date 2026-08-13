"""Core comparison orchestration: turns two lists of scanned images into
a ranked list of similarity matches.

Pipeline (see project README for the full rationale):
  1. MD5 — exact duplicates short-circuit everything else (100% match).
  2. Base perceptual hash (or CLIP/DINOv2 embedding) per image, computed
     once, one image at a time -- each image's signature is computed and
     immediately reported via progress_cb/log_cb before moving to the
     next, so the progress bar advances in real time instead of jumping
     from "stuck" to "100%" once a whole folder's signatures are done.
  3. Optional Deep Rotation: pre-rotate every folder-B image through the
     full 360 degrees once (not per-pair — the expensive step happens
     once per image regardless of how many folder-A images it's compared
     against). Note this always runs for every image when enabled: an
     earlier version gated this behind a 0-degree similarity threshold,
     which was wrong — pHash is itself rotation-sensitive, so a rotated
     duplicate can easily score *below* that gate at 0 degrees, meaning
     the very thing Deep Rotation exists to catch was being filtered out
     before the rotation sweep ever ran.
  4. Full N x M similarity matrix: 0-degree comparison, plus the best
     angle from the rotation sweep when Deep Rotation is enabled.
  5. Mirror-pair deduplication: when the two folders overlap (most
     visibly when folder A and folder B are literally the same folder),
     the N x M matrix contains both (imageX from A, imageY from B) and
     (imageY from A, imageX from B) as separate grid cells -- the same
     underlying pair of files, just with the "which folder" role
     swapped. Since similarity is symmetric, these show up as duplicate-
     looking entries in the results. This step keeps only one direction,
     identified by each pair's resolved absolute file paths (so it's
     correct regardless of *why* the folders overlap).
  6. Filename similarity, computed for every surviving pair.
  7. Filter by visual_threshold, sort descending.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imagecompare.config import settings
from imagecompare.core import clip_embed, dino_embed, hashing, md5util
from imagecompare.core.filename_sim import filename_similarity
from imagecompare.core.scanner import ScannedImage

ProgressCallback = Callable[[int, int, str], None]  # (processed, total, phase_label)
CheckpointCallback = Callable[[], None]  # call frequently; may block (pause) or raise (cancel)
LogCallback = Callable[[str, str], None]  # (level, message)

_LOG_MILESTONE_STEP_PERCENT = 25


class ComparisonCancelled(Exception):
    """Raised internally when a job's cancel signal fires mid-comparison."""


def _noop_progress(processed: int, total: int, phase: str) -> None:
    return None


def _noop_checkpoint() -> None:
    return None


def _noop_log(level: str, message: str) -> None:
    return None


def _milestone_indices(total: int, step_percent: int = _LOG_MILESTONE_STEP_PERCENT) -> set[int]:
    """1-indexed positions at which to emit a periodic log milestone
    during a loop of `total` items -- e.g. step_percent=25 logs at
    roughly 25%, 50%, 75%, and 100% through, rather than flooding the
    (size-capped) activity log with one line per item."""
    if total <= 0:
        return set()
    return {max(1, (total * pct) // 100) for pct in range(step_percent, 101, step_percent)}


def _percentiles(
    values: list[float], points: tuple[int, ...] = (0, 25, 50, 75, 100)
) -> dict[int, float]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    result = {}
    for p in points:
        idx = min(n - 1, max(0, round((p / 100) * (n - 1))))
        result[p] = ordered[idx]
    return result


@dataclass
class MatchCandidate:
    image1: ScannedImage
    image2: ScannedImage
    visual_similarity: float
    filename_similarity: float
    is_exact_duplicate: bool
    best_rotation_angle: int | None = None
    image1_md5: str = ""
    image2_md5: str = ""


@dataclass(frozen=True)
class _EmbeddingBackend:
    """Groups together the pieces every embedding-based mode (CLIP,
    DINOv2, ...) needs, so adding a new one doesn't require duplicating
    the dispatch logic in compare_folders() below."""

    label: str
    module: Any
    baseline: Callable[[], float]
    unavailable_error: type[Exception]


_EMBEDDING_BACKENDS: dict[str, _EmbeddingBackend] = {
    "clip": _EmbeddingBackend(
        label="CLIP",
        module=clip_embed,
        baseline=lambda: settings.clip_similarity_baseline,
        unavailable_error=clip_embed.ClipUnavailableError,
    ),
    "dino": _EmbeddingBackend(
        label="DINOv2",
        module=dino_embed,
        baseline=lambda: settings.dino_similarity_baseline,
        unavailable_error=dino_embed.DinoUnavailableError,
    ),
}


def compare_folders(
    images1: Sequence[ScannedImage],
    images2: Sequence[ScannedImage],
    *,
    mode: str = "hash",
    visual_threshold: float = 70.0,
    deep_rotation: bool = False,
    hash_size: int = 16,
    rotation_step_degrees: int = 1,
    progress_cb: ProgressCallback = _noop_progress,
    checkpoint_cb: CheckpointCallback = _noop_checkpoint,
    log_cb: LogCallback = _noop_log,
) -> list[MatchCandidate]:
    if mode != "hash" and mode not in _EMBEDDING_BACKENDS:
        raise ValueError(f"Unknown mode: {mode!r}")

    backend = _EMBEDDING_BACKENDS.get(mode)
    if backend is not None:
        dependencies_ok = backend.module.is_available()
        log_cb(
            "info" if dependencies_ok else "error",
            f"{backend.label} dependency check: "
            f"{'installed' if dependencies_ok else 'NOT installed'}",
        )
        if not dependencies_ok:
            raise backend.unavailable_error(
                f"{backend.label} mode requires the optional '{mode}' extra, which isn't "
                f"installed. Install it with: {backend.module.INSTALL_HINT}"
            )

    n1, n2 = len(images1), len(images2)
    log_cb(
        "info",
        f"Comparing {n1} image(s) in folder A against {n2} image(s) in folder B "
        f"(mode={mode}, deep_rotation={deep_rotation})",
    )

    do_rotation = deep_rotation and mode == "hash"
    base_total = n1 + n2 + n1 + n2 + (n2 if do_rotation else 0) + (n1 * n2)
    processed = 0

    def tick(phase: str, total: int | None = None) -> None:
        nonlocal processed
        checkpoint_cb()
        processed += 1
        progress_cb(processed, total if total is not None else base_total, phase)

    milestones_1 = _milestone_indices(n1)
    milestones_2 = _milestone_indices(n2)

    # ---- Phase 1: MD5 (exact duplicates) ----
    log_cb("info", f"Computing MD5 hashes for folder A ({n1} image(s))...")
    md5_1: dict[int, str] = {}
    for idx, img in enumerate(images1, start=1):
        md5_1[idx - 1] = md5util.compute_md5(img.path)
        tick(f"hashing files (md5) - folder A ({idx}/{n1}): {img.name}")
        if idx in milestones_1:
            log_cb("info", f"MD5 hashing folder A: {idx}/{n1} done")

    log_cb("info", f"Computing MD5 hashes for folder B ({n2} image(s))...")
    md5_2: dict[int, str] = {}
    for idx, img in enumerate(images2, start=1):
        md5_2[idx - 1] = md5util.compute_md5(img.path)
        tick(f"hashing files (md5) - folder B ({idx}/{n2}): {img.name}")
        if idx in milestones_2:
            log_cb("info", f"MD5 hashing folder B: {idx}/{n2} done")

    # ---- Phase 2: base visual signature per image ----
    # Computed and reported one image at a time (not "compute the whole
    # list, then report progress after the fact") so the progress bar
    # and per-image log milestones reflect what's actually happening.
    sig_1: list[Any] = []
    sig_2: list[Any] = []
    if mode == "hash":
        log_cb("info", f"Computing perceptual hashes for folder A ({n1} image(s))...")
        for idx, img in enumerate(images1, start=1):
            sig_1.append(hashing.compute_hash(img.path, hash_size=hash_size))
            tick(f"computing perceptual hashes - folder A ({idx}/{n1}): {img.name}")
            if idx in milestones_1:
                log_cb("info", f"Perceptual hashing folder A: {idx}/{n1} done")

        log_cb("info", f"Computing perceptual hashes for folder B ({n2} image(s))...")
        for idx, img in enumerate(images2, start=1):
            sig_2.append(hashing.compute_hash(img.path, hash_size=hash_size))
            tick(f"computing perceptual hashes - folder B ({idx}/{n2}): {img.name}")
            if idx in milestones_2:
                log_cb("info", f"Perceptual hashing folder B: {idx}/{n2} done")
    else:
        assert backend is not None

        def _embedding_progress(msg: str) -> None:
            progress_cb(processed, base_total, msg)

        backend.module.ensure_model_ready(progress_cb=_embedding_progress, log_cb=log_cb)

        log_cb("info", f"Computing {backend.label} embeddings for folder A ({n1} image(s))...")
        for idx, img in enumerate(images1, start=1):
            sig_1.append(backend.module.compute_embedding(img.path))
            tick(f"computing {backend.label} embeddings - folder A ({idx}/{n1}): {img.name}")
            if idx in milestones_1:
                log_cb("info", f"{backend.label} embeddings folder A: {idx}/{n1} done")

        log_cb("info", f"Computing {backend.label} embeddings for folder B ({n2} image(s))...")
        for idx, img in enumerate(images2, start=1):
            sig_2.append(backend.module.compute_embedding(img.path))
            tick(f"computing {backend.label} embeddings - folder B ({idx}/{n2}): {img.name}")
            if idx in milestones_2:
                log_cb("info", f"{backend.label} embeddings folder B: {idx}/{n2} done")

    # ---- Phase 3: pre-rotate every folder-B image (Deep Rotation only) ----
    rotated_cache: dict[int, list] = {}
    if do_rotation:
        log_cb(
            "info",
            f"Deep Rotation enabled: sweeping all {n2} folder-B image(s) through "
            f"360 degrees in {rotation_step_degrees}-degree steps",
        )
        for idx, j in enumerate(range(n2), start=1):
            rotated_cache[j] = hashing.rotated_hashes(
                images2[j].path,
                hash_size=hash_size,
                step_degrees=rotation_step_degrees,
            )
            tick(f"deep rotation: pre-rotating folder B images ({idx}/{n2}): " f"{images2[j].name}")
            if idx in milestones_2:
                log_cb("info", f"Deep rotation pre-rotation: {idx}/{n2} done")

    # ---- Phase 4: full similarity matrix ----
    log_cb("info", f"Comparing {n1} x {n2} = {n1 * n2} image pair(s)...")
    pair_milestones = _milestone_indices(n1 * n2)
    raw_similarities: list[float] = []  # embedding modes only, for distribution logging below

    candidates: list[MatchCandidate] = []
    pair_idx = 0
    for i in range(n1):
        for j in range(n2):
            pair_idx += 1
            is_dup = md5_1[i] == md5_2[j]
            angle: int | None = None

            if is_dup:
                sim = 100.0
            elif mode == "hash":
                sim = hashing.similarity_from_hashes(sig_1[i], sig_2[j])
                if do_rotation:
                    result = hashing.best_rotation_similarity(
                        sig_1[i], rotated_cache[j], step_degrees=rotation_step_degrees
                    )
                    if result.best_similarity >= sim:
                        sim = result.best_similarity
                        angle = result.best_angle
                    else:
                        angle = 0
            else:
                assert backend is not None
                raw = backend.module.raw_cosine_similarity(sig_1[i], sig_2[j])
                raw_similarities.append(raw)
                sim = backend.module.calibrate_similarity(raw, baseline=backend.baseline())

            tick("comparing image pairs")
            if pair_idx in pair_milestones:
                log_cb("info", f"Comparing pairs: {pair_idx}/{n1 * n2} done")

            candidates.append(
                MatchCandidate(
                    image1=images1[i],
                    image2=images2[j],
                    visual_similarity=sim,
                    filename_similarity=0.0,
                    is_exact_duplicate=is_dup,
                    best_rotation_angle=angle,
                    image1_md5=md5_1[i],
                    image2_md5=md5_2[j],
                )
            )

    if raw_similarities and backend is not None:
        pct = _percentiles(raw_similarities)
        log_cb(
            "info",
            f"{backend.label} raw similarity distribution for this job: "
            f"min={pct[0]:.2f} p25={pct[25]:.2f} p50={pct[50]:.2f} p75={pct[75]:.2f} "
            f"max={pct[100]:.2f} (current baseline={backend.baseline():.2f})",
        )

    # ---- Phase 5: mirror-pair deduplication ----
    # When folder A and folder B overlap (most visibly: are the same
    # folder), the same underlying pair of files appears twice in the
    # N x M matrix -- once as (fileX from A, fileY from B), once as
    # (fileY from A, fileX from B). Keep only the first occurrence.
    resolved_path_cache: dict[Path, str] = {}

    def _resolved(p: Path) -> str:
        cached = resolved_path_cache.get(p)
        if cached is None:
            cached = str(p.resolve())
            resolved_path_cache[p] = cached
        return cached

    seen_pairs: set[frozenset[str]] = set()
    deduplicated: list[MatchCandidate] = []
    mirror_count = 0
    for c in candidates:
        key = frozenset({_resolved(c.image1.path), _resolved(c.image2.path)})
        if key in seen_pairs:
            mirror_count += 1
            continue
        seen_pairs.add(key)
        deduplicated.append(c)

    if mirror_count:
        log_cb(
            "info",
            f"Removed {mirror_count} mirror-duplicate pair(s) (folders overlap, so the "
            f"same file pair appeared twice with roles swapped)",
        )

    # ---- Phase 6: filename similarity + threshold filter ----
    final: list[MatchCandidate] = []
    for c in deduplicated:
        if c.visual_similarity < visual_threshold and not c.is_exact_duplicate:
            continue
        fname_sim = filename_similarity(c.image1.name, c.image2.name)
        final.append(
            MatchCandidate(
                image1=c.image1,
                image2=c.image2,
                visual_similarity=round(c.visual_similarity, 2),
                filename_similarity=fname_sim,
                is_exact_duplicate=c.is_exact_duplicate,
                best_rotation_angle=c.best_rotation_angle,
                image1_md5=c.image1_md5,
                image2_md5=c.image2_md5,
            )
        )

    final.sort(key=lambda c: c.visual_similarity, reverse=True)
    log_cb(
        "info",
        f"Comparison finished: {len(final)} match(es) at or above {visual_threshold}% "
        f"visual similarity",
    )
    return final
