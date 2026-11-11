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


def test_quality_crop_beats_a_larger_fallback():
    cache = CropCache()
    large_fallback = np.zeros((200, 200, 3), dtype=np.uint8)
    small_quality = np.zeros((60, 60, 3), dtype=np.uint8)

    cache.update(1, large_fallback)
    cache.update(1, small_quality, quality_score=0.8)

    assert cache.get(1) is small_quality

    better_quality = np.zeros((50, 50, 3), dtype=np.uint8)
    cache.update(1, better_quality, quality_score=0.9)
    assert cache.get(1) is better_quality

    worse_quality = np.zeros((80, 80, 3), dtype=np.uint8)
    cache.update(1, worse_quality, quality_score=0.7)
    assert cache.get(1) is better_quality


def test_pop_removes_quality_and_fallback():
    cache = CropCache()
    cache.update(1, np.zeros((20, 20, 3), dtype=np.uint8))
    cache.update(1, np.zeros((10, 10, 3), dtype=np.uint8), quality_score=0.9)

    assert cache.pop(1) is not None
    assert cache.get(1) is None
    assert len(cache) == 0
