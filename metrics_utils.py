"""Shared evaluation metric utilities used by train, evaluate, and eval_suite."""

import numpy as np


def ranking_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    n_pairs: int = 5000,
    seed: int = 42,
) -> float:
    """Pairwise ranking accuracy via random sampling.

    Randomly draws ``n_pairs`` (i, j) index pairs and checks whether
    ``preds`` preserves the ordering of ``labels``.  Pairs with label
    difference below 1e-4 are excluded as ties.
    """
    if len(preds) < 2:
        return 0.0
    n_pairs = min(n_pairs, len(preds) ** 2)
    rng = np.random.default_rng(seed)
    idx_i = rng.choice(len(preds), n_pairs)
    idx_j = rng.choice(len(preds), n_pairs)
    valid = np.abs(labels[idx_i] - labels[idx_j]) > 1e-4
    correct = (preds[idx_i] - preds[idx_j]) * (labels[idx_i] - labels[idx_j]) > 0
    return float(np.sum(correct & valid) / (np.sum(valid) + 1e-8))
