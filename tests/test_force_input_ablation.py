import numpy as np
import pytest
import torch

from reward_model import build_proprio_input_mask
from scripts.run_force_input_ablation import aggregate_runs, reference_pra, split_episode_ids


def test_episode_split_is_deterministic_disjoint_and_15_5() -> None:
    train_a, val_a = split_episode_ids(20, seed=42)
    train_b, val_b = split_episode_ids(20, seed=42)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 15
    assert len(val_a) == 5
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a) | set(val_a) == set(range(20))


def test_episode_split_rejects_non_protocol_counts() -> None:
    with pytest.raises(ValueError, match="exactly 20"):
        split_episode_ids(19, seed=42)


def test_reference_pra_uses_all_pairs_and_counts_ties_incorrect() -> None:
    # Comparisons: 2>1 true, 2>2 false, 3>1 true, 3>2 true => 3/4.
    assert reference_pra([2.0, 3.0], [1.0, 2.0]) == 0.75


def test_force_mask_repeats_only_selected_coordinates() -> None:
    mask = build_proprio_input_mask(robot_dim=5, state_windows=3, masked_state_indices=[1, 2])
    expected = torch.tensor([1, 0, 0, 1, 1] * 3, dtype=torch.float32)
    torch.testing.assert_close(mask, expected)


def test_aggregate_reports_population_variance_and_keeps_runs() -> None:
    runs = [
        {"task": "button", "seed": 1, "condition": "full", "status": "ok", "accuracy": 0.5},
        {"task": "button", "seed": 2, "condition": "full", "status": "ok", "accuracy": 1.0},
        {"task": "button", "seed": 1, "condition": "no_force", "status": "ok", "accuracy": 0.25},
        {"task": "button", "seed": 2, "condition": "no_force", "status": "ok", "accuracy": 0.5},
    ]
    aggregates = aggregate_runs(runs)
    row = next(
        item
        for item in aggregates
        if item["task"] == "button" and item["condition"] == "full"
    )
    assert row["mean_accuracy"] == 0.75
    assert row["variance"] == np.var([0.5, 1.0], ddof=0)
    assert row["accuracies"] == [0.5, 1.0]
    delta = next(
        item
        for item in aggregates
        if item["task"] == "button" and item["condition"] == "full_minus_no_force"
    )
    assert delta["accuracies"] == [0.25, 0.5]
