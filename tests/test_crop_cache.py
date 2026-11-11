"""Tests for the best-crop cache."""

from __future__ import annotations

import numpy as np

from src.pipeline import CropCache


def test_keeps_the_largest_crop():
    cache = CropCache()
    small = np.zeros((10, 20, 3), dtype=np.uint8)
    large = np.zeros((40, 30, 3), dtype=np.uint8)

    cache.update(1, small)
    cache.update(1, large)

    assert cache.get(1) is large
    assert len(cache) == 1


def test_smaller_crop_does_not_replace_best():
    cache = CropCache()
    large = np.zeros((40, 30, 3), dtype=np.uint8)
    small = np.zeros((10, 20, 3), dtype=np.uint8)

    cache.update(1, large)
    cache.update(1, small)

    assert cache.get(1) is large


def test_ignores_missing_and_empty_crops():
    cache = CropCache()
    cache.update(1, None)
    cache.update(1, np.zeros((0, 0, 3), dtype=np.uint8))
    assert len(cache) == 0


def test_pop_and_clear():
    cache = CropCache()
    cache.update(1, np.zeros((10, 10, 3), dtype=np.uint8))
    cache.update(2, np.zeros((20, 20, 3), dtype=np.uint8))

    assert cache.pop(1) is not None
    assert cache.get(1) is None
    assert len(cache) == 1

    cache.clear()
    assert len(cache) == 0
