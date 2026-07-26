"""Tests for label.strategies._base utilities."""

import numpy as np

from label.strategies._base import normalize_rewards, smooth


def test_smooth_identity_for_window_1() -> None:
    r = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(smooth(r, 1), r)


def test_smooth_preserves_length() -> None:
    r = np.random.default_rng(42).random(50)
    assert len(smooth(r, 10)) == 50


def test_smooth_causal() -> None:
    r = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    result = smooth(r, 3)
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 0.0
    assert result[3] > 0.0
    assert result[4] > 0.0


def test_smooth_window_zero_returns_original() -> None:
    r = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(smooth(r, 0), r)


def test_smooth_window_larger_than_array() -> None:
    r = np.array([1.0, 2.0, 3.0])
    result = smooth(r, 100)
    assert len(result) == 3
    assert np.all(np.isfinite(result))


def test_smooth_single_element() -> None:
    r = np.array([5.0])
    assert np.array_equal(smooth(r, 10), r)


def test_normalize_rewards_range() -> None:
    r = np.array([0.0, 0.5, 1.0])
    normed = normalize_rewards(r, min_r=2.0, max_r=8.0)
    assert np.isclose(normed.min(), 2.0)
    assert np.isclose(normed.max(), 8.0)


def test_normalize_rewards_constant() -> None:
    r = np.array([5.0, 5.0, 5.0])
    normed = normalize_rewards(r, min_r=0.0, max_r=6.0)
    assert np.allclose(normed, 3.0)


def test_normalize_rewards_negative() -> None:
    r = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    normed = normalize_rewards(r, min_r=0.0, max_r=1.0)
    assert np.isclose(normed[0], 0.0)
    assert np.isclose(normed[-1], 1.0)
