from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from imagecompare.core.upload import (
    InvalidRelativePathError,
    cleanup_old_uploads,
    sanitize_relative_path,
)


def test_sanitize_simple_relative_path():
    assert sanitize_relative_path("photo.jpg") == Path("photo.jpg")


def test_sanitize_nested_relative_path():
    assert sanitize_relative_path("sub/inner/photo.jpg") == Path("sub/inner/photo.jpg")


def test_sanitize_normalizes_backslashes():
    assert sanitize_relative_path("sub\\inner\\photo.jpg") == Path("sub/inner/photo.jpg")


def test_sanitize_strips_leading_slash_and_dot_segments():
    assert sanitize_relative_path("./sub/./photo.jpg") == Path("sub/photo.jpg")


def test_sanitize_rejects_parent_traversal():
    with pytest.raises(InvalidRelativePathError):
        sanitize_relative_path("../../etc/passwd")


def test_sanitize_rejects_traversal_in_middle():
    with pytest.raises(InvalidRelativePathError):
        sanitize_relative_path("sub/../../escape.jpg")


def test_sanitize_rejects_empty_string():
    with pytest.raises(InvalidRelativePathError):
        sanitize_relative_path("")


def test_sanitize_rejects_only_dots_and_slashes():
    with pytest.raises(InvalidRelativePathError):
        sanitize_relative_path("././.")


def test_cleanup_removes_old_directories(tmp_path):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    old_dir = uploads_dir / "old"
    old_dir.mkdir()
    (old_dir / "file.txt").write_text("x")
    old_time = time.time() - (100 * 24 * 60 * 60)  # 100 days ago
    os.utime(old_dir, (old_time, old_time))

    removed = cleanup_old_uploads(uploads_dir, max_age_seconds=60 * 60)
    assert removed == 1
    assert not old_dir.exists()


def test_cleanup_keeps_recent_directories(tmp_path):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    recent_dir = uploads_dir / "recent"
    recent_dir.mkdir()

    removed = cleanup_old_uploads(uploads_dir, max_age_seconds=60 * 60)
    assert removed == 0
    assert recent_dir.exists()


def test_cleanup_handles_missing_uploads_dir(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert cleanup_old_uploads(missing, max_age_seconds=60) == 0


def test_cleanup_ignores_files_directly_in_uploads_dir(tmp_path):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    stray_file = uploads_dir / "not-a-dir.txt"
    stray_file.write_text("x")
    old_time = time.time() - (100 * 24 * 60 * 60)
    os.utime(stray_file, (old_time, old_time))

    removed = cleanup_old_uploads(uploads_dir, max_age_seconds=60)
    assert removed == 0
    assert stray_file.exists()
