from __future__ import annotations

import time

from imagecompare.jobs.manager import JobManager
from imagecompare.models import JobStatus


def _wait_for_status(manager, job_id, status, timeout=10.0):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        progress = manager.get_progress(job_id)
        if progress and progress.status == status:
            return progress
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not reach {status} in time")


def _wait_until_started(manager, job_id, timeout=5.0):
    """Wait for the worker thread to move past PENDING, so pause/cancel
    calls land on a checkpoint deterministically instead of racing the
    thread's startup."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        progress = manager.get_progress(job_id)
        if progress and progress.status != JobStatus.PENDING:
            return progress
        time.sleep(0.005)
    raise TimeoutError(f"Job {job_id} never left PENDING")


def test_job_reaches_done_status(tmp_path, folder_pair):
    f1, f2 = folder_pair
    manager = JobManager()
    thumb_dir = tmp_path / "thumbs"

    job_id = manager.create_job(
        folder1=str(f1),
        folder2=str(f2),
        mode="hash",
        recursive=False,
        include_patterns="",
        exclude_patterns="",
        visual_threshold=0.0,
        filename_threshold=90.0,
        deep_rotation=False,
        hash_size=16,
        thumb_cache_dir=str(thumb_dir),
    )

    progress = _wait_for_status(manager, job_id, JobStatus.DONE)
    assert progress.percent == 100.0
    assert progress.processed_items == progress.total_items


def test_job_results_available_after_completion(tmp_path, folder_pair):
    f1, f2 = folder_pair
    manager = JobManager()
    job_id = manager.create_job(
        folder1=str(f1),
        folder2=str(f2),
        mode="hash",
        recursive=False,
        include_patterns="",
        exclude_patterns="",
        visual_threshold=0.0,
        filename_threshold=90.0,
        deep_rotation=False,
        hash_size=16,
        thumb_cache_dir=str(tmp_path / "thumbs"),
    )
    _wait_for_status(manager, job_id, JobStatus.DONE)

    state = manager.get_state(job_id)
    assert state is not None
    assert len(state.results) > 0


def test_job_errors_on_missing_folder(tmp_path):
    manager = JobManager()
    job_id = manager.create_job(
        folder1=str(tmp_path / "does-not-exist"),
        folder2=str(tmp_path),
        mode="hash",
        recursive=False,
        include_patterns="",
        exclude_patterns="",
        visual_threshold=70.0,
        filename_threshold=90.0,
        deep_rotation=False,
        hash_size=16,
        thumb_cache_dir=str(tmp_path / "thumbs"),
    )
    progress = _wait_for_status(manager, job_id, JobStatus.ERROR)
    assert progress.error_message is not None


def test_unknown_job_id_returns_none():
    manager = JobManager()
    assert manager.get_progress("nonexistent") is None
    assert manager.get_state("nonexistent") is None


def _slow_folder_pair(tmp_path, tmp_image_factory, count=40):
    """A folder pair large/slow enough (bigger images, more pairs) that a
    pause/cancel issued right after the worker starts reliably lands
    mid-comparison instead of racing job completion."""
    f1 = tmp_path / "slow1"
    f2 = tmp_path / "slow2"
    for i in range(count):
        tmp_image_factory(f1, f"a{i}.png", shape="circle", fg=(i % 250, 30, 30), size=(500, 500))
    for i in range(count):
        tmp_image_factory(f2, f"b{i}.png", shape="triangle", fg=(20, i % 250, 60), size=(500, 500))
    return f1, f2


def test_cancel_job_stops_it_before_completion(tmp_path, tmp_image_factory):
    f1, f2 = _slow_folder_pair(tmp_path, tmp_image_factory)
    manager = JobManager()
    job_id = manager.create_job(
        folder1=str(f1),
        folder2=str(f2),
        mode="hash",
        recursive=False,
        include_patterns="",
        exclude_patterns="",
        visual_threshold=70.0,
        filename_threshold=90.0,
        deep_rotation=False,
        hash_size=16,
        thumb_cache_dir=str(tmp_path / "thumbs"),
    )

    # cancel once the worker has actually started (avoids racing thread
    # startup for a deterministic PENDING → CANCELLED landing point)
    _wait_until_started(manager, job_id)
    assert manager.cancel_job(job_id) is True
    progress = _wait_for_status(manager, job_id, JobStatus.CANCELLED)
    assert progress.status == JobStatus.CANCELLED


def test_cancel_unknown_job_returns_false():
    manager = JobManager()
    assert manager.cancel_job("nonexistent") is False


def test_cancel_already_done_job_returns_false(tmp_path, folder_pair):
    f1, f2 = folder_pair
    manager = JobManager()
    job_id = manager.create_job(
        folder1=str(f1),
        folder2=str(f2),
        mode="hash",
        recursive=False,
        include_patterns="",
        exclude_patterns="",
        visual_threshold=0.0,
        filename_threshold=90.0,
        deep_rotation=False,
        hash_size=16,
        thumb_cache_dir=str(tmp_path / "thumbs"),
    )
    _wait_for_status(manager, job_id, JobStatus.DONE)
    assert manager.cancel_job(job_id) is False


def test_pause_blocks_progress_until_resumed(tmp_path, tmp_image_factory):
    f1, f2 = _slow_folder_pair(tmp_path, tmp_image_factory)
    manager = JobManager()
    job_id = manager.create_job(
        folder1=str(f1),
        folder2=str(f2),
        mode="hash",
        recursive=False,
        include_patterns="",
        exclude_patterns="",
        visual_threshold=70.0,
        filename_threshold=90.0,
        deep_rotation=False,
        hash_size=16,
        thumb_cache_dir=str(tmp_path / "thumbs"),
    )

    # pause once the worker has actually started, so this lands on a real
    # checkpoint deterministically rather than racing thread startup.
    _wait_until_started(manager, job_id)
    assert manager.pause_job(job_id) is True
    paused_progress = _wait_for_status(manager, job_id, JobStatus.PAUSED)
    processed_at_pause = paused_progress.processed_items

    time.sleep(0.3)
    still_paused = manager.get_progress(job_id)
    assert still_paused.status == JobStatus.PAUSED
    assert still_paused.processed_items == processed_at_pause

    assert manager.resume_job(job_id) is True
    _wait_for_status(manager, job_id, JobStatus.DONE, timeout=20.0)


def test_pause_unknown_job_returns_false():
    manager = JobManager()
    assert manager.pause_job("nonexistent") is False


def test_resume_job_that_is_not_paused_returns_false(tmp_path, folder_pair):
    f1, f2 = folder_pair
    manager = JobManager()
    job_id = manager.create_job(
        folder1=str(f1),
        folder2=str(f2),
        mode="hash",
        recursive=False,
        include_patterns="",
        exclude_patterns="",
        visual_threshold=0.0,
        filename_threshold=90.0,
        deep_rotation=False,
        hash_size=16,
        thumb_cache_dir=str(tmp_path / "thumbs"),
    )
    _wait_for_status(manager, job_id, JobStatus.DONE)
    assert manager.resume_job(job_id) is False


def test_job_folders_are_resolved_absolute_paths(tmp_path, folder_pair):
    f1, f2 = folder_pair
    manager = JobManager()
    job_id = manager.create_job(
        folder1=str(f1),
        folder2=str(f2),
        mode="hash",
        recursive=False,
        include_patterns="",
        exclude_patterns="",
        visual_threshold=0.0,
        filename_threshold=90.0,
        deep_rotation=False,
        hash_size=16,
        thumb_cache_dir=str(tmp_path / "thumbs"),
    )
    state = manager.get_state(job_id)
    assert state is not None
    assert state.folder1 == str(f1.resolve())
    assert state.folder2 == str(f2.resolve())


def test_job_progress_never_exceeds_total_with_deep_rotation(tmp_path, folder_pair):
    """End-to-end guard for the regression where a job's reported
    processed_items outran total_items during Deep Rotation (e.g.
    '5486 / 3091 - 100%' in the UI). Poll a real deep-rotation job
    throughout its run and confirm the API-visible progress stays sane
    at every step."""
    f1, f2 = folder_pair
    manager = JobManager()
    job_id = manager.create_job(
        folder1=str(f1),
        folder2=str(f2),
        mode="hash",
        recursive=False,
        include_patterns="",
        exclude_patterns="",
        visual_threshold=0.0,
        filename_threshold=90.0,
        deep_rotation=True,
        hash_size=16,
        thumb_cache_dir=str(tmp_path / "thumbs"),
    )

    seen_done = False
    for _ in range(500):
        progress = manager.get_progress(job_id)
        assert progress.processed_items <= progress.total_items or progress.total_items == 0
        assert progress.percent <= 100.0
        if progress.status in (JobStatus.DONE, JobStatus.ERROR):
            seen_done = True
            break
        time.sleep(0.01)

    assert seen_done
