from __future__ import annotations

import shutil

from imagecompare.core.md5util import compute_md5


def test_identical_files_have_same_md5(tmp_path, tmp_image_factory):
    p1 = tmp_image_factory(tmp_path, "a.png", shape="circle")
    p2 = tmp_path / "a_copy.png"
    shutil.copyfile(p1, p2)

    assert compute_md5(p1) == compute_md5(p2)


def test_different_files_have_different_md5(tmp_path, tmp_image_factory):
    p1 = tmp_image_factory(tmp_path, "a.png", shape="circle", fg=(200, 30, 30))
    p2 = tmp_image_factory(tmp_path, "b.png", shape="triangle", fg=(20, 180, 60))

    assert compute_md5(p1) != compute_md5(p2)


def test_md5_is_deterministic(tmp_path, tmp_image_factory):
    p1 = tmp_image_factory(tmp_path, "a.png", shape="circle")
    assert compute_md5(p1) == compute_md5(p1)


def test_md5_matches_known_algorithm_output(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_bytes(b"hello world")
    # md5("hello world") is a well-known value
    assert compute_md5(p) == "5eb63bbbe01eeed093cb22bb8f5acdc3"
