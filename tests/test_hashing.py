from __future__ import annotations

from imagecompare.core import hashing


def test_identical_images_have_100_percent_similarity(tmp_path, tmp_image_factory):
    p1 = tmp_image_factory(tmp_path, "a.png", shape="circle")
    p2 = tmp_image_factory(tmp_path, "b.png", shape="circle")  # same params -> visually identical

    h1 = hashing.compute_hash(p1)
    h2 = hashing.compute_hash(p2)

    assert hashing.similarity_from_hashes(h1, h2) == 100.0


def test_different_shapes_are_less_similar_than_identical(tmp_path, tmp_image_factory):
    circle = tmp_image_factory(tmp_path, "circle.png", shape="circle", fg=(200, 30, 30))
    triangle = tmp_image_factory(tmp_path, "triangle.png", shape="triangle", fg=(20, 180, 60))
    circle2 = tmp_image_factory(tmp_path, "circle2.png", shape="circle", fg=(200, 30, 30))

    h_circle = hashing.compute_hash(circle)
    h_circle2 = hashing.compute_hash(circle2)
    h_triangle = hashing.compute_hash(triangle)

    same_shape_sim = hashing.similarity_from_hashes(h_circle, h_circle2)
    diff_shape_sim = hashing.similarity_from_hashes(h_circle, h_triangle)

    assert same_shape_sim > diff_shape_sim


def test_similarity_is_bounded_0_to_100(tmp_path, tmp_image_factory):
    a = tmp_image_factory(tmp_path, "a.png", shape="stripes", fg=(0, 0, 0), bg=(255, 255, 255))
    b = tmp_image_factory(tmp_path, "b.png", shape="circle", fg=(255, 255, 255), bg=(0, 0, 0))

    sim = hashing.similarity_from_hashes(hashing.compute_hash(a), hashing.compute_hash(b))
    assert 0.0 <= sim <= 100.0


def test_rotated_hashes_returns_expected_count():
    pass  # covered by rotated_hashes_length below with a real file


def test_rotated_hashes_length(tmp_path, tmp_image_factory):
    p = tmp_image_factory(tmp_path, "shape.png", shape="triangle")
    hashes = hashing.rotated_hashes(p, step_degrees=10)
    assert len(hashes) == 36  # 360 / 10


def test_rotated_hashes_step_1_full_sweep(tmp_path, tmp_image_factory):
    p = tmp_image_factory(tmp_path, "shape.png", shape="triangle")
    hashes = hashing.rotated_hashes(p, step_degrees=1)
    assert len(hashes) == 360


def test_best_rotation_similarity_finds_a_good_angle(tmp_path, tmp_image_factory):
    # An asymmetric shape (triangle) rotated should match itself best near
    # 0/360 degrees when compared against its own unrotated hash.
    p = tmp_image_factory(tmp_path, "triangle.png", shape="triangle")
    base_hash = hashing.compute_hash(p)
    candidate_hashes = hashing.rotated_hashes(p, step_degrees=15)

    result = hashing.best_rotation_similarity(base_hash, candidate_hashes, step_degrees=15)
    assert result.best_similarity == 100.0
    assert result.best_angle == 0


def test_invalid_algo_raises():
    import pytest

    with pytest.raises(ValueError):
        hashing.compute_hash(__file__, algo="not-a-real-algo")
