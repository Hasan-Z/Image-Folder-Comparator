from __future__ import annotations

import time

import numpy as np
import pytest

from imagecompare.core import clip_embed


def test_ensure_model_ready_calls_loader_exactly_once(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_loader():
        calls["n"] += 1

    monkeypatch.setattr(clip_embed, "_model", None)
    clip_embed.ensure_model_ready(cache_dir=tmp_path, _loader=fake_loader, heartbeat_interval=0.05)

    assert calls["n"] == 1


def test_ensure_model_ready_skips_loader_if_already_loaded(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_loader():
        calls["n"] += 1

    monkeypatch.setattr(clip_embed, "_model", object())  # pretend it's already loaded
    clip_embed.ensure_model_ready(cache_dir=tmp_path, _loader=fake_loader)

    assert calls["n"] == 0
    monkeypatch.setattr(clip_embed, "_model", None)  # reset for other tests


def test_ensure_model_ready_reports_progress_while_loading(monkeypatch, tmp_path):
    def slow_loader():
        time.sleep(0.25)

    monkeypatch.setattr(clip_embed, "_model", None)
    progress_messages: list[str] = []

    clip_embed.ensure_model_ready(
        cache_dir=tmp_path,
        _loader=slow_loader,
        progress_cb=progress_messages.append,
        heartbeat_interval=0.05,
    )

    assert len(progress_messages) > 0
    assert all("Loading CLIP model" in m for m in progress_messages)


def test_ensure_model_ready_reports_cache_dir_growth(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def growing_loader():
        # simulate a download writing bytes to the cache dir while it runs
        (cache_dir / "weights.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        time.sleep(0.2)

    monkeypatch.setattr(clip_embed, "_model", None)
    progress_messages: list[str] = []

    clip_embed.ensure_model_ready(
        cache_dir=cache_dir,
        _loader=growing_loader,
        progress_cb=progress_messages.append,
        heartbeat_interval=0.05,
    )

    assert any("MB downloaded so far" in m for m in progress_messages)


def test_ensure_model_ready_logs_start_and_completion(monkeypatch, tmp_path):
    def fake_loader():
        time.sleep(0.05)

    monkeypatch.setattr(clip_embed, "_model", None)
    log_entries: list[tuple[str, str]] = []

    clip_embed.ensure_model_ready(
        cache_dir=tmp_path,
        _loader=fake_loader,
        log_cb=lambda level, msg: log_entries.append((level, msg)),
        heartbeat_interval=0.05,
    )

    assert any("Loading CLIP model" in msg for _level, msg in log_entries)
    assert any("CLIP model ready" in msg for _level, msg in log_entries)


def test_ensure_model_ready_stops_heartbeat_thread_after_loader_returns(monkeypatch, tmp_path):
    def fake_loader():
        pass  # instant

    monkeypatch.setattr(clip_embed, "_model", None)
    calls = {"n": 0}

    def counting_progress(_msg: str) -> None:
        calls["n"] += 1

    clip_embed.ensure_model_ready(
        cache_dir=tmp_path,
        _loader=fake_loader,
        progress_cb=counting_progress,
        heartbeat_interval=0.05,
    )
    count_after_return = calls["n"]

    time.sleep(0.3)  # give a lingering thread time to misbehave, if there were one
    assert calls["n"] == count_after_return


def test_ensure_model_ready_propagates_loader_exception(monkeypatch, tmp_path):
    import pytest

    def failing_loader():
        raise RuntimeError("boom")

    monkeypatch.setattr(clip_embed, "_model", None)
    with pytest.raises(RuntimeError, match="boom"):
        clip_embed.ensure_model_ready(
            cache_dir=tmp_path, _loader=failing_loader, heartbeat_interval=0.05
        )


def test_default_weights_cache_dir_honors_hf_home(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/custom/hf/home")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert str(clip_embed.default_weights_cache_dir()) == "/custom/hf/home/hub"


def test_default_weights_cache_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    result = clip_embed.default_weights_cache_dir()
    assert ".cache" in str(result)
    assert "huggingface" in str(result)


def test_is_model_loaded_reflects_module_state(monkeypatch):
    monkeypatch.setattr(clip_embed, "_model", None)
    assert clip_embed.is_model_loaded() is False
    monkeypatch.setattr(clip_embed, "_model", object())
    assert clip_embed.is_model_loaded() is True
    monkeypatch.setattr(clip_embed, "_model", None)


def test_get_status_reflects_dependencies_not_installed():
    # torch/open_clip are not installed in this test environment (the
    # 'clip' extra is optional and not part of dev dependencies), so this
    # exercises the real "not available" path end-to-end.
    status = clip_embed.get_status()
    assert status.dependencies_installed is False
    assert status.install_hint == clip_embed.INSTALL_HINT
    assert status.model_name == "ViT-B-32"


def test_get_status_reflects_dependencies_installed(monkeypatch):
    monkeypatch.setattr(clip_embed, "is_available", lambda: True)
    status = clip_embed.get_status()
    assert status.dependencies_installed is True


def test_get_status_reflects_model_loaded_state(monkeypatch):
    monkeypatch.setattr(clip_embed, "_model", None)
    assert clip_embed.get_status().model_loaded is False
    monkeypatch.setattr(clip_embed, "_model", object())
    assert clip_embed.get_status().model_loaded is True
    monkeypatch.setattr(clip_embed, "_model", None)


def test_get_status_reports_zero_cache_size_when_dir_missing(monkeypatch, tmp_path):
    missing_dir = tmp_path / "does-not-exist"
    monkeypatch.setattr(clip_embed, "default_weights_cache_dir", lambda: missing_dir)
    status = clip_embed.get_status()
    assert status.weights_cache_dir_size_mb == 0.0
    assert status.weights_cache_dir == str(missing_dir)


def test_get_status_reports_nonzero_cache_size_when_files_present(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "weights.bin").write_bytes(b"x" * (3 * 1024 * 1024))
    monkeypatch.setattr(clip_embed, "default_weights_cache_dir", lambda: cache_dir)
    status = clip_embed.get_status()
    assert status.weights_cache_dir_size_mb == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------
# cosine_similarity() calibration
#
# Raw CLIP cosine similarity does not spread evenly across [-1, 1] for
# real images -- unrelated photos routinely score 0.5-0.8, not near 0
# (a well-documented property of CLIP's embedding space). These tests
# use plain synthetic numpy vectors, so they exercise the actual
# calibration math without needing torch/open_clip installed.
# ---------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_100_percent():
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert clip_embed.cosine_similarity(vec, vec, baseline=0.5) == pytest.approx(100.0)


def test_cosine_similarity_at_baseline_is_zero_percent():
    # two vectors whose raw cosine similarity is exactly the baseline
    # should map to 0%, not some positive number
    baseline = 0.5
    a = np.array([1.0, 0.0], dtype=np.float32)
    # cos(angle) = 0.5 -> angle = 60 degrees
    b = np.array([0.5, (3**0.5) / 2], dtype=np.float32)
    sim = clip_embed.cosine_similarity(a, b, baseline=baseline)
    assert sim == pytest.approx(0.0, abs=0.5)


def test_cosine_similarity_below_baseline_clamps_to_zero_not_negative():
    # orthogonal vectors: raw cosine similarity = 0, well below a 0.5
    # baseline -- must clamp to 0%, never go negative
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    sim = clip_embed.cosine_similarity(a, b, baseline=0.5)
    assert sim == 0.0


def test_cosine_similarity_opposite_vectors_clamps_to_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    sim = clip_embed.cosine_similarity(a, b, baseline=0.5)
    assert sim == 0.0


def test_cosine_similarity_midpoint_between_baseline_and_identical():
    # raw cosine similarity halfway between baseline (0.5) and 1.0 (i.e.
    # 0.75) should land at roughly 50%
    baseline = 0.5
    a = np.array([1.0, 0.0], dtype=np.float32)
    # construct b such that a . b == 0.75 (both unit vectors)
    import math

    angle = math.acos(0.75)
    b = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    sim = clip_embed.cosine_similarity(a, b, baseline=baseline)
    assert sim == pytest.approx(50.0, abs=1.0)


def test_cosine_similarity_never_exceeds_100_or_goes_below_0():
    import random

    random.seed(42)
    for _ in range(50):
        a = np.array([random.uniform(-1, 1) for _ in range(8)], dtype=np.float32)
        b = np.array([random.uniform(-1, 1) for _ in range(8)], dtype=np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        sim = clip_embed.cosine_similarity(a, b, baseline=0.5)
        assert 0.0 <= sim <= 100.0


def test_cosine_similarity_default_baseline_matches_documented_default():
    # calling without an explicit baseline should use the documented
    # default of 0.5, not some other value
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)  # raw cosine sim = 0.0
    assert clip_embed.cosine_similarity(a, b) == 0.0  # below default baseline -> clamped


def test_cosine_similarity_higher_baseline_is_more_conservative():
    # a stricter (higher) baseline should score the same pair lower,
    # since more of the raw similarity range is treated as "unrelated"
    a = np.array([1.0, 0.0], dtype=np.float32)
    import math

    angle = math.acos(0.8)
    b = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    lenient = clip_embed.cosine_similarity(a, b, baseline=0.3)
    strict = clip_embed.cosine_similarity(a, b, baseline=0.7)
    assert strict < lenient


def test_raw_cosine_similarity_returns_uncalibrated_value():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert clip_embed.raw_cosine_similarity(a, b) == pytest.approx(0.0)

    c = np.array([1.0, 0.0], dtype=np.float32)
    assert clip_embed.raw_cosine_similarity(a, c) == pytest.approx(1.0)


def test_raw_cosine_similarity_is_bounded():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    raw = clip_embed.raw_cosine_similarity(a, b)
    assert -1.0 <= raw <= 1.0
    assert raw == pytest.approx(-1.0)


def test_calibrate_similarity_matches_cosine_similarity():
    # calibrate_similarity(raw, baseline=x) should give the identical
    # result as cosine_similarity(a, b, baseline=x), since the latter is
    # implemented in terms of the former
    a = np.array([1.0, 0.0], dtype=np.float32)
    import math

    angle = math.acos(0.65)
    b = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    raw = clip_embed.raw_cosine_similarity(a, b)
    via_calibrate = clip_embed.calibrate_similarity(raw, baseline=0.5)
    via_cosine_similarity = clip_embed.cosine_similarity(a, b, baseline=0.5)
    assert via_calibrate == pytest.approx(via_cosine_similarity)
