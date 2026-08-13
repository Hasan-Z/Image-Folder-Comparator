from __future__ import annotations

from unittest.mock import patch

from imagecompare.core import hashing
from imagecompare.core.comparator import compare_folders
from imagecompare.core.scanner import scan_folder


def test_exact_duplicate_scores_100_and_flagged(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    matches = compare_folders(images1, images2, visual_threshold=0.0)

    dup = [m for m in matches if m.is_exact_duplicate]
    assert len(dup) == 1
    assert dup[0].visual_similarity == 100.0
    assert dup[0].image1.name == "circle_red.png"
    assert dup[0].image2.name == "circle_red_copy.png"


def test_md5_populated_for_every_match(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    matches = compare_folders(images1, images2, visual_threshold=0.0)
    for m in matches:
        assert m.image1_md5 != ""
        assert m.image2_md5 != ""


def test_exact_duplicate_pair_shares_identical_md5(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    matches = compare_folders(images1, images2, visual_threshold=0.0)
    dup = next(m for m in matches if m.is_exact_duplicate)
    assert dup.image1_md5 == dup.image2_md5


def test_non_duplicate_pairs_have_different_md5(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    matches = compare_folders(images1, images2, visual_threshold=0.0)
    non_dups = [m for m in matches if not m.is_exact_duplicate]
    assert len(non_dups) > 0
    for m in non_dups:
        assert m.image1_md5 != m.image2_md5


def test_threshold_filters_out_low_similarity_pairs(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    matches_loose = compare_folders(images1, images2, visual_threshold=0.0)
    matches_strict = compare_folders(images1, images2, visual_threshold=99.9)

    assert len(matches_strict) <= len(matches_loose)
    assert all(m.visual_similarity >= 99.9 or m.is_exact_duplicate for m in matches_strict)


def test_results_sorted_descending_by_similarity(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    matches = compare_folders(images1, images2, visual_threshold=0.0)
    scores = [m.visual_similarity for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_every_pair_present_at_zero_threshold(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    matches = compare_folders(images1, images2, visual_threshold=0.0)
    assert len(matches) == len(images1) * len(images2)


def test_filename_similarity_attached_to_every_match(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    matches = compare_folders(images1, images2, visual_threshold=0.0)
    for m in matches:
        assert 0.0 <= m.filename_similarity <= 100.0


def test_deep_rotation_does_not_lower_scores(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    without = {
        (m.image1.name, m.image2.name): m.visual_similarity
        for m in compare_folders(images1, images2, visual_threshold=0.0, deep_rotation=False)
    }
    with_rotation = {
        (m.image1.name, m.image2.name): m.visual_similarity
        for m in compare_folders(
            images1,
            images2,
            visual_threshold=0.0,
            deep_rotation=True,
            rotation_step_degrees=30,  # coarse for test speed
        )
    }

    for key, base_score in without.items():
        assert with_rotation[key] >= base_score - 0.01


def test_deep_rotation_detects_a_rotated_duplicate(tmp_path):
    """Regression test: an earlier implementation gated the rotation sweep
    behind the 0-degree pHash similarity, which is exactly backwards --
    pHash is itself rotation-sensitive, so a genuinely rotated duplicate
    can score low at 0 degrees and would never reach the sweep. Deep
    Rotation must catch this case.
    """
    from PIL import Image, ImageDraw

    base = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(base)
    # an asymmetric shape -- a rotated square/circle would look identical
    # at many angles and wouldn't actually exercise the rotation search
    draw.polygon([(100, 20), (40, 180), (160, 150)], fill=(200, 30, 30))

    f1 = tmp_path / "folder1"
    f2 = tmp_path / "folder2"
    f1.mkdir()
    f2.mkdir()

    base.save(f2 / "original.png")
    rotated = base.rotate(37, expand=False, fillcolor=(0, 0, 0))
    rotated.save(f1 / "rotated.png")

    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    without = compare_folders(images1, images2, visual_threshold=0.0, deep_rotation=False)[0]
    with_rotation = compare_folders(
        images1, images2, visual_threshold=0.0, deep_rotation=True, rotation_step_degrees=1
    )[0]

    # the un-rotated (0-degree) comparison should score meaningfully lower
    # than after searching for the actual rotation angle
    assert with_rotation.visual_similarity > without.visual_similarity
    # and it should actually find the (near-)correct alignment
    assert with_rotation.visual_similarity >= 90.0
    assert with_rotation.best_rotation_angle is not None


def test_progress_callback_reaches_full_total(folder_pair):
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    calls = []
    compare_folders(
        images1,
        images2,
        visual_threshold=0.0,
        progress_cb=lambda processed, total, phase: calls.append((processed, total, phase)),
    )

    assert len(calls) > 0
    last_processed, last_total, _ = calls[-1]
    assert last_processed == last_total


def test_progress_never_exceeds_total_with_deep_rotation_enabled(folder_pair):
    """Regression test: an earlier implementation's rotation-refinement
    phase called the progress callback with a smaller `total` than the
    already-accumulated `processed` count (since it only covered gated
    candidates), producing nonsense like '5486 / 3091 - 100%' in the UI.
    Deep Rotation now always runs (no gating), and progress must stay
    monotonic and never exceed the reported total at any point.
    """
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    calls = []
    compare_folders(
        images1,
        images2,
        visual_threshold=0.0,
        deep_rotation=True,
        rotation_step_degrees=45,  # coarse, for test speed
        progress_cb=lambda processed, total, phase: calls.append((processed, total, phase)),
    )

    assert len(calls) > 0
    for processed, total, phase in calls:
        assert processed <= total, f"processed ({processed}) exceeded total ({total}) at {phase!r}"

    last_processed, last_total, _ = calls[-1]
    assert last_processed == last_total


def test_invalid_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        compare_folders([], [], mode="not-a-real-mode")


def test_dino_mode_raises_clear_error_when_unavailable(folder_pair):
    import pytest

    from imagecompare.core.dino_embed import DinoUnavailableError

    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    with pytest.raises(DinoUnavailableError, match="pip install"):
        compare_folders(images1, images2, mode="dino", visual_threshold=0.0)


def test_dino_mode_logs_dependency_check(folder_pair):
    from imagecompare.core.dino_embed import DinoUnavailableError

    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    logs: list[tuple[str, str]] = []
    try:
        compare_folders(
            images1,
            images2,
            mode="dino",
            visual_threshold=0.0,
            log_cb=lambda level, msg: logs.append((level, msg)),
        )
    except DinoUnavailableError:
        pass

    assert any("DINOv2 dependency check" in msg for _level, msg in logs)


def test_dino_mode_uses_dino_similarity_baseline_setting(folder_pair, monkeypatch):
    """The comparator should read dino_similarity_baseline from settings
    (mirroring how CLIP mode reads clip_similarity_baseline), not a
    hardcoded value, so it stays tunable without code changes."""
    import numpy as np

    from imagecompare.config import settings
    from imagecompare.core import dino_embed

    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    monkeypatch.setattr(dino_embed, "is_available", lambda: True)
    monkeypatch.setattr(dino_embed, "ensure_model_ready", lambda **kwargs: None)
    monkeypatch.setattr(
        dino_embed, "compute_embedding", lambda path: np.array([1.0, 0.0], dtype=np.float32)
    )

    seen_baselines: list[float] = []
    original_calibrate = dino_embed.calibrate_similarity

    def spy_calibrate(raw, *, baseline=0.3):
        seen_baselines.append(baseline)
        return original_calibrate(raw, baseline=baseline)

    monkeypatch.setattr(dino_embed, "calibrate_similarity", spy_calibrate)
    monkeypatch.setattr(settings, "dino_similarity_baseline", 0.42)

    compare_folders(images1, images2, mode="dino", visual_threshold=0.0)

    assert len(seen_baselines) > 0
    assert all(b == 0.42 for b in seen_baselines)


def test_empty_folders_return_no_matches(tmp_path):
    matches = compare_folders([], [], visual_threshold=0.0)
    assert matches == []


def test_progress_reported_incrementally_not_in_a_burst_after_the_fact(tmp_path, tmp_image_factory):
    """Regression test: an earlier implementation computed a whole
    folder's signatures via a list comprehension first, and only
    afterward looped again purely to call tick() -- so the progress bar
    would appear frozen while the (potentially slow) computation ran,
    then suddenly jump from stuck to '100%' once it was already done.
    Each image's signature and its corresponding progress tick must be
    interleaved, not batched.
    """
    f1 = tmp_path / "f1"
    f2 = tmp_path / "f2"
    for i in range(4):
        tmp_image_factory(f1, f"a{i}.png", shape="circle", fg=(i * 10, 30, 30))
    for i in range(4):
        tmp_image_factory(f2, f"b{i}.png", shape="triangle", fg=(20, i * 10, 60))

    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    events: list[str] = []
    real_compute_hash = hashing.compute_hash

    def tracking_compute_hash(path, **kwargs):
        events.append(f"compute:{path.name}")
        return real_compute_hash(path, **kwargs)

    def tracking_progress(processed, total, phase):
        events.append(f"tick:{processed}")

    import imagecompare.core.comparator as comparator_module

    with patch.object(comparator_module.hashing, "compute_hash", tracking_compute_hash):
        compare_folders(images1, images2, visual_threshold=0.0, progress_cb=tracking_progress)

    # find the events for just the perceptual-hashing phase (8 computes,
    # 8 corresponding ticks -- ignore the MD5 phase's ticks which don't
    # call compute_hash)
    compute_events = [e for e in events if e.startswith("compute:")]
    assert len(compute_events) == 8

    # the critical assertion: every "compute:" event must be followed by
    # at least one "tick:" event before the NEXT "compute:" event -- i.e.
    # no two computes happen back-to-back without a progress update
    # between them (which is what the old buggy list-comprehension
    # pattern would produce).
    last_compute_idx = None
    for idx, event in enumerate(events):
        if event.startswith("compute:"):
            if last_compute_idx is not None:
                between = events[last_compute_idx + 1 : idx]
                assert any(e.startswith("tick:") for e in between), (
                    "two compute_hash() calls happened with no progress tick between them "
                    "-- progress is being batched instead of reported in real time"
                )
            last_compute_idx = idx


def test_progress_phase_text_includes_current_image_name(folder_pair):
    """The progress bar's phase text should name which image is
    currently being processed, not just a generic phase label, so the
    user can see real-time granular progress."""
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    phases: list[str] = []
    compare_folders(
        images1,
        images2,
        visual_threshold=0.0,
        progress_cb=lambda processed, total, phase: phases.append(phase),
    )

    hashing_phases = [p for p in phases if p.startswith("computing perceptual hashes")]
    assert len(hashing_phases) > 0
    # each should reference an actual filename from one of the folders
    all_names = {img.name for img in images1} | {img.name for img in images2}
    assert any(any(name in p for name in all_names) for p in hashing_phases)


def test_mirror_pairs_deduplicated_when_folders_are_identical(tmp_path, tmp_image_factory):
    """Regression test: comparing a folder against itself produced two
    scans (images1, images2) that reference the same files. The N x M
    matrix naturally contains both (fileA, fileB) and (fileB, fileA) as
    distinct grid cells when fileA != fileB but both live in the
    overlapping folders -- the same real-world pair, counted twice with
    roles swapped. This must be deduplicated.
    """
    folder = tmp_path / "photos"
    tmp_image_factory(folder, "sunset.png", shape="circle", fg=(200, 100, 20))
    tmp_image_factory(folder, "beach.png", shape="triangle", fg=(20, 150, 200))
    tmp_image_factory(folder, "mountains.png", shape="square", fg=(80, 80, 200))

    images1 = scan_folder(folder)
    images2 = scan_folder(folder)  # a second, independent scan of the SAME folder

    matches = compare_folders(images1, images2, visual_threshold=0.0)

    # build an unordered-pair identity for every match using resolved
    # absolute paths, and make sure no identity appears twice
    seen = set()
    for m in matches:
        key = frozenset({str(m.image1.path.resolve()), str(m.image2.path.resolve())})
        assert key not in seen, f"mirror-duplicate pair found: {m.image1.name} <-> {m.image2.name}"
        seen.add(key)


def test_mirror_dedup_keeps_self_pairs(tmp_path, tmp_image_factory):
    """The exact self-pair (same file matched against itself, e.g.
    sunset.png vs sunset.png) is a legitimate, meaningful result (it
    shows the file exists in both folders) and must NOT be removed by
    mirror deduplication -- only the (A,B)/(B,A) swapped-role
    duplicates should be removed."""
    folder = tmp_path / "photos"
    tmp_image_factory(folder, "sunset.png", shape="circle", fg=(200, 100, 20))
    tmp_image_factory(folder, "beach.png", shape="triangle", fg=(20, 150, 200))

    images1 = scan_folder(folder)
    images2 = scan_folder(folder)

    matches = compare_folders(images1, images2, visual_threshold=0.0)

    self_pairs = [m for m in matches if m.image1.name == m.image2.name]
    names = {m.image1.name for m in self_pairs}
    assert names == {"sunset.png", "beach.png"}


def test_mirror_dedup_does_not_affect_disjoint_folders(folder_pair):
    """When folder A and folder B don't overlap at all, dedup should be
    a no-op -- every pair is genuinely distinct."""
    f1, f2 = folder_pair
    images1 = scan_folder(f1)
    images2 = scan_folder(f2)

    without_threshold = compare_folders(images1, images2, visual_threshold=0.0)
    assert len(without_threshold) == len(images1) * len(images2)
