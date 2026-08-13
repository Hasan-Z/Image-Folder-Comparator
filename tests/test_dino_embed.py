from __future__ import annotations

import time

import numpy as np
import pytest

from imagecompare.core import dino_embed


def test_ensure_model_ready_calls_loader_exactly_once(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_loader():
        calls["n"] += 1

    monkeypatch.setattr(dino_embed, "_model", None)
    dino_embed.ensure_model_ready(cache_dir=tmp_path, _loader=fake_loader, heartbeat_interval=0.05)

    assert calls["n"] == 1


def test_ensure_model_ready_skips_loader_if_already_loaded(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_loader():
        calls["n"] += 1

    monkeypatch.setattr(dino_embed, "_model", object())  # pretend it's already loaded
    dino_embed.ensure_model_ready(cache_dir=tmp_path, _loader=fake_loader)

    assert calls["n"] == 0
    monkeypatch.setattr(dino_embed, "_model", None)  # reset for other tests


def test_ensure_model_ready_reports_progress_while_loading(monkeypatch, tmp_path):
    def slow_loader():
        time.sleep(0.25)

    monkeypatch.setattr(dino_embed, "_model", None)
    progress_messages: list[str] = []

    dino_embed.ensure_model_ready(
        cache_dir=tmp_path,
        _loader=slow_loader,
        progress_cb=progress_messages.append,
        heartbeat_interval=0.05,
    )

    assert len(progress_messages) > 0
    assert all("Loading DINOv2 model" in m for m in progress_messages)


def test_ensure_model_ready_reports_cache_dir_growth(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def growing_loader():
        # simulate a download writing bytes to the cache dir while it runs
        (cache_dir / "weights.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        time.sleep(0.2)

    monkeypatch.setattr(dino_embed, "_model", None)
    progress_messages: list[str] = []

    dino_embed.ensure_model_ready(
        cache_dir=cache_dir,
        _loader=growing_loader,
        progress_cb=progress_messages.append,
        heartbeat_interval=0.05,
    )

    assert any("MB downloaded so far" in m for m in progress_messages)


def test_ensure_model_ready_logs_start_and_completion(monkeypatch, tmp_path):
    def fake_loader():
        time.sleep(0.05)

    monkeypatch.setattr(dino_embed, "_model", None)
    log_entries: list[tuple[str, str]] = []

    dino_embed.ensure_model_ready(
        cache_dir=tmp_path,
        _loader=fake_loader,
        log_cb=lambda level, msg: log_entries.append((level, msg)),
        heartbeat_interval=0.05,
    )

    assert any("Loading DINOv2 model" in msg for _level, msg in log_entries)
    assert any("DINOv2 model ready" in msg for _level, msg in log_entries)


def test_ensure_model_ready_stops_heartbeat_thread_after_loader_returns(monkeypatch, tmp_path):
    def fake_loader():
        pass  # instant

    monkeypatch.setattr(dino_embed, "_model", None)
    calls = {"n": 0}

    def counting_progress(_msg: str) -> None:
        calls["n"] += 1

    dino_embed.ensure_model_ready(
        cache_dir=tmp_path,
        _loader=fake_loader,
        progress_cb=counting_progress,
        heartbeat_interval=0.05,
    )
    count_after_return = calls["n"]

    time.sleep(0.3)  # give a lingering thread time to misbehave, if there were one
    assert calls["n"] == count_after_return


def test_ensure_model_ready_propagates_loader_exception(monkeypatch, tmp_path):
    def failing_loader():
        raise RuntimeError("boom")

    monkeypatch.setattr(dino_embed, "_model", None)
    with pytest.raises(RuntimeError, match="boom"):
        dino_embed.ensure_model_ready(
            cache_dir=tmp_path, _loader=failing_loader, heartbeat_interval=0.05
        )


def test_default_weights_cache_dir_honors_hf_home(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/custom/hf/home")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert str(dino_embed.default_weights_cache_dir()) == "/custom/hf/home/hub"


def test_default_weights_cache_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    result = dino_embed.default_weights_cache_dir()
    assert ".cache" in str(result)
    assert "huggingface" in str(result)


def test_is_model_loaded_reflects_module_state(monkeypatch):
    monkeypatch.setattr(dino_embed, "_model", None)
    assert dino_embed.is_model_loaded() is False
    monkeypatch.setattr(dino_embed, "_model", object())
    assert dino_embed.is_model_loaded() is True
    monkeypatch.setattr(dino_embed, "_model", None)


def test_get_status_reflects_dependencies_not_installed():
    # torch/transformers are not installed in this test environment (the
    # 'dino' extra is optional and not part of dev dependencies), so this
    # exercises the real "not available" path end-to-end.
    status = dino_embed.get_status()
    assert status.dependencies_installed is False
    assert status.install_hint == dino_embed.INSTALL_HINT
    assert status.model_name == "facebook/dinov2-small"


def test_get_status_reflects_dependencies_installed(monkeypatch):
    monkeypatch.setattr(dino_embed, "is_available", lambda: True)
    status = dino_embed.get_status()
    assert status.dependencies_installed is True


def test_get_status_reflects_model_loaded_state(monkeypatch):
    monkeypatch.setattr(dino_embed, "_model", None)
    assert dino_embed.get_status().model_loaded is False
    monkeypatch.setattr(dino_embed, "_model", object())
    assert dino_embed.get_status().model_loaded is True
    monkeypatch.setattr(dino_embed, "_model", None)


def test_get_status_reports_zero_cache_size_when_dir_missing(monkeypatch, tmp_path):
    missing_dir = tmp_path / "does-not-exist"
    monkeypatch.setattr(dino_embed, "default_weights_cache_dir", lambda: missing_dir)
    status = dino_embed.get_status()
    assert status.weights_cache_dir_size_mb == 0.0
    assert status.weights_cache_dir == str(missing_dir)


def test_get_status_reports_nonzero_cache_size_when_files_present(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "weights.bin").write_bytes(b"x" * (3 * 1024 * 1024))
    monkeypatch.setattr(dino_embed, "default_weights_cache_dir", lambda: cache_dir)
    status = dino_embed.get_status()
    assert status.weights_cache_dir_size_mb == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------
# cosine_similarity() calibration
#
# DINOv2 embeddings are better separated than CLIP's, but a small
# calibration baseline is still applied for consistency and because
# some residual clustering is common in most learned embedding spaces.
# These tests use plain synthetic numpy vectors, so they exercise the
# actual calibration math without needing torch/transformers installed.
# ---------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_100_percent():
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert dino_embed.cosine_similarity(vec, vec, baseline=0.3) == pytest.approx(100.0)


def test_cosine_similarity_below_baseline_clamps_to_zero_not_negative():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)  # raw cosine sim = 0.0, below 0.3 baseline
    sim = dino_embed.cosine_similarity(a, b, baseline=0.3)
    assert sim == 0.0


def test_cosine_similarity_opposite_vectors_clamps_to_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    sim = dino_embed.cosine_similarity(a, b, baseline=0.3)
    assert sim == 0.0


def test_cosine_similarity_midpoint_between_baseline_and_identical():
    # raw cosine similarity halfway between baseline (0.3) and 1.0 (i.e.
    # 0.65) should land at roughly 50%
    baseline = 0.3
    a = np.array([1.0, 0.0], dtype=np.float32)
    import math

    angle = math.acos(0.65)
    b = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    sim = dino_embed.cosine_similarity(a, b, baseline=baseline)
    assert sim == pytest.approx(50.0, abs=1.0)


def test_cosine_similarity_never_exceeds_100_or_goes_below_0():
    import random

    random.seed(7)
    for _ in range(50):
        a = np.array([random.uniform(-1, 1) for _ in range(8)], dtype=np.float32)
        b = np.array([random.uniform(-1, 1) for _ in range(8)], dtype=np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        sim = dino_embed.cosine_similarity(a, b, baseline=0.3)
        assert 0.0 <= sim <= 100.0


def test_cosine_similarity_default_baseline_matches_documented_default():
    # calling without an explicit baseline should use the documented
    # default of 0.3, not CLIP's 0.5
    a = np.array([1.0, 0.0], dtype=np.float32)
    import math

    angle = math.acos(0.4)  # between the two modules' default baselines
    b = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    # above DINOv2's default baseline (0.3) -> should be > 0
    assert dino_embed.cosine_similarity(a, b) > 0.0


def test_cosine_similarity_higher_baseline_is_more_conservative():
    a = np.array([1.0, 0.0], dtype=np.float32)
    import math

    angle = math.acos(0.6)
    b = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    lenient = dino_embed.cosine_similarity(a, b, baseline=0.1)
    strict = dino_embed.cosine_similarity(a, b, baseline=0.5)
    assert strict < lenient


def test_raw_cosine_similarity_returns_uncalibrated_value():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert dino_embed.raw_cosine_similarity(a, b) == pytest.approx(0.0)

    c = np.array([1.0, 0.0], dtype=np.float32)
    assert dino_embed.raw_cosine_similarity(a, c) == pytest.approx(1.0)


def test_calibrate_similarity_matches_cosine_similarity():
    a = np.array([1.0, 0.0], dtype=np.float32)
    import math

    angle = math.acos(0.5)
    b = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    raw = dino_embed.raw_cosine_similarity(a, b)
    via_calibrate = dino_embed.calibrate_similarity(raw, baseline=0.3)
    via_cosine_similarity = dino_embed.cosine_similarity(a, b, baseline=0.3)
    assert via_calibrate == pytest.approx(via_cosine_similarity)


def test_dino_default_baseline_is_lower_than_clip_default():
    # DINOv2's embeddings are documented as better separated than
    # CLIP's, so its default calibration baseline should be less
    # conservative (lower) than CLIP's.
    from imagecompare.core import clip_embed

    a = np.array([1.0, 0.0], dtype=np.float32)
    import math

    angle = math.acos(0.4)
    b = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    dino_score = dino_embed.cosine_similarity(a, b)  # default baseline 0.3
    clip_score = clip_embed.cosine_similarity(a, b)  # default baseline 0.5
    assert dino_score > clip_score
