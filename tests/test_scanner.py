from __future__ import annotations

import pytest

from imagecompare.core.scanner import scan_folder


def test_scan_flat_folder(tmp_path, tmp_image_factory):
    tmp_image_factory(tmp_path, "a.png")
    tmp_image_factory(tmp_path, "b.png")
    (tmp_path / "notes.txt").write_text("not an image")

    results = scan_folder(tmp_path)
    names = {r.name for r in results}
    assert names == {"a.png", "b.png"}


def test_scan_non_recursive_ignores_subfolders(tmp_path, tmp_image_factory):
    tmp_image_factory(tmp_path, "top.png")
    nested = tmp_path / "nested"
    tmp_image_factory(nested, "inner.png")

    results = scan_folder(tmp_path, recursive=False)
    names = {r.name for r in results}
    assert names == {"top.png"}


def test_scan_recursive_includes_subfolders(tmp_path, tmp_image_factory):
    tmp_image_factory(tmp_path, "top.png")
    nested = tmp_path / "nested"
    tmp_image_factory(nested, "inner.png")
    deeper = nested / "deeper"
    tmp_image_factory(deeper, "deepest.png")

    results = scan_folder(tmp_path, recursive=True)
    names = {r.name for r in results}
    assert names == {"top.png", "inner.png", "deepest.png"}


def test_include_pattern_filters_to_matching_extension(tmp_path, tmp_image_factory):
    tmp_image_factory(tmp_path, "a.jpg", fmt="JPEG")
    tmp_image_factory(tmp_path, "b.png", fmt="PNG")

    results = scan_folder(tmp_path, include_patterns="*.jpg")
    names = {r.name for r in results}
    assert names == {"a.jpg"}


def test_exclude_pattern_removes_matching_files(tmp_path, tmp_image_factory):
    tmp_image_factory(tmp_path, "a.jpg", fmt="JPEG")
    tmp_image_factory(tmp_path, "b.png", fmt="PNG")

    results = scan_folder(tmp_path, exclude_patterns="*.jpg")
    names = {r.name for r in results}
    assert names == {"b.png"}


def test_exclude_wins_over_include_when_both_match(tmp_path, tmp_image_factory):
    tmp_image_factory(tmp_path, "a.jpg", fmt="JPEG")
    tmp_image_factory(tmp_path, "b.png", fmt="PNG")

    # include everything, but explicitly exclude jpg -> only png survives
    results = scan_folder(tmp_path, include_patterns="*.jpg *.png", exclude_patterns="*.jpg")
    names = {r.name for r in results}
    assert names == {"b.png"}


def test_patterns_are_case_insensitive(tmp_path, tmp_image_factory):
    tmp_image_factory(tmp_path, "A.JPG", fmt="JPEG")

    results = scan_folder(tmp_path, include_patterns="*.jpg")
    names = {r.name for r in results}
    assert names == {"A.JPG"}


def test_missing_folder_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_folder(tmp_path / "does-not-exist")


def test_file_instead_of_folder_raises(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        scan_folder(f)


def test_relative_path_preserved_for_recursive_scan(tmp_path, tmp_image_factory):
    nested = tmp_path / "sub"
    tmp_image_factory(nested, "x.png")

    results = scan_folder(tmp_path, recursive=True)
    assert results[0].relative_path in ("sub/x.png", "sub\\x.png")  # POSIX/Windows
