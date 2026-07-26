"""Tests for metrics_utils.ranking_accuracy."""

import numpy as np
import pytest

from metrics_utils import ranking_accuracy


def test_perfect_ranking() -> None:
    preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ranking_accuracy(preds, labels) == pytest.approx(1.0, abs=0.01)


def test_reversed_ranking() -> None:
    preds = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ranking_accuracy(preds, labels) == 0.0


def test_single_element_returns_zero() -> None:
    assert ranking_accuracy(np.array([1.0]), np.array([1.0])) == 0.0


def test_tied_labels_excluded() -> None:
    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.0, 1.0, 1.0])
    acc = ranking_accuracy(preds, labels)
    assert acc == 0.0


def test_deterministic_with_seed() -> None:
    rng = np.random.default_rng(123)
    preds = rng.random(50)
    labels = rng.random(50)
    a = ranking_accuracy(preds, labels, seed=42)
    b = ranking_accuracy(preds, labels, seed=42)
    assert a == b


def test_empty_arrays_return_zero() -> None:
    assert ranking_accuracy(np.array([]), np.array([])) == 0.0
