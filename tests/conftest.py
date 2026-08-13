"""Shared pytest fixtures.

Test images are generated with PIL rather than checked into the repo, so
tests don't depend on binary fixtures and stay fast/deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw


def _draw_shape(size=(200, 200), bg=(255, 255, 255), shape="circle", fg=(200, 30, 30)):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse([40, 40, 160, 160], fill=fg)
    elif shape == "square":
        draw.rectangle([40, 40, 160, 160], fill=fg)
    elif shape == "triangle":
        draw.polygon([(100, 30), (30, 170), (170, 170)], fill=fg)
    elif shape == "stripes":
        for y in range(0, size[1], 20):
            draw.rectangle([0, y, size[0], y + 10], fill=fg)
    return img


@pytest.fixture
def tmp_image_factory(tmp_path):
    """Returns a callable that writes a generated image to disk and
    returns its Path."""

    counter = {"n": 0}

    def _make(
        directory: Path,
        name: str | None = None,
        *,
        shape: str = "circle",
        bg=(255, 255, 255),
        fg=(200, 30, 30),
        size=(200, 200),
        fmt: str = "PNG",
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        counter["n"] += 1
        if name is None:
            name = f"img_{counter['n']}.png"
        path = directory / name
        img = _draw_shape(size=size, bg=bg, shape=shape, fg=fg)
        img.save(path, fmt)
        return path

    return _make


@pytest.fixture
def folder_pair(tmp_path, tmp_image_factory):
    """A pre-populated pair of folders with some identical, some similar,
    and some unrelated images — used across comparator/API tests."""
    f1 = tmp_path / "folder1"
    f2 = tmp_path / "folder2"

    # exact duplicate (same bytes copied)
    a1 = tmp_image_factory(f1, "circle_red.png", shape="circle", fg=(200, 30, 30))
    import shutil

    a2 = f2 / "circle_red_copy.png"
    f2.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a1, a2)

    # visually similar but not identical (slightly different fill)
    tmp_image_factory(f1, "circle_orange.png", shape="circle", fg=(210, 60, 20))
    tmp_image_factory(f2, "circle_orange2.png", shape="circle", fg=(205, 55, 25))

    # unrelated
    tmp_image_factory(f1, "square_blue.png", shape="square", fg=(20, 40, 200))
    tmp_image_factory(f2, "triangle_green.png", shape="triangle", fg=(20, 180, 60))

    return f1, f2
