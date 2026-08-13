from __future__ import annotations

from imagecompare.core.filename_sim import filename_similarity


def test_identical_names_are_100_percent():
    assert filename_similarity("IMG_0231.jpg", "IMG_0231.jpg") == 100.0


def test_extension_is_ignored():
    assert filename_similarity("photo.jpg", "photo.png") == 100.0


def test_similar_names_score_high():
    score = filename_similarity("IMG_0231.jpg", "IMG_0231_edited.jpg")
    assert score > 65.0


def test_dissimilar_names_score_low():
    score = filename_similarity("vacation_beach.jpg", "xkq2093.png")
    assert score < 40.0


def test_empty_names_are_100_percent():
    assert filename_similarity("", "") == 100.0


def test_case_insensitive():
    assert filename_similarity("Photo.JPG", "photo.jpg") == 100.0


def test_unicode_filenames_do_not_raise():
    score = filename_similarity("café_été.jpg", "café_été2.jpg")
    assert 0.0 <= score <= 100.0


def test_score_is_symmetric():
    a = filename_similarity("sunset_beach.jpg", "beach_sunset.jpg")
    b = filename_similarity("beach_sunset.jpg", "sunset_beach.jpg")
    assert a == b
