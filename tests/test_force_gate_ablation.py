import numpy as np
import pytest

from label.auto_label import StageTarget, compute_dense_reward
from scripts.run_force_gate_ablation import evaluate_task


def _episode() -> list[dict]:
    forces = ([2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0])
    positions = ([0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0])
    episode = []
    for index, (force, position) in enumerate(zip(forces, positions, strict=True)):
        state = np.zeros(19, dtype=np.float64)
        state[0] = 0.5
        state[1:4] = force
        state[4:7] = position
        episode.append(
            {
                "observations": {"state": state},
                "actions": np.zeros(1),
                "rewards": 0.0,
                "dones": index == 2,
            }
        )
    return episode


def _target() -> StageTarget:
    return StageTarget(
        pos_mean=np.zeros(3),
        pos_cov_inv=np.eye(3),
        force_mean=np.zeros(3),
        force_cov_inv=np.eye(3),
        gripper_mean=0.5,
        gripper_std=0.1,
        is_contact=True,
        max_duration=0.0,
        dforce_mean=np.zeros(3),
        dforce_cov_inv=np.eye(3),
    )


def test_default_is_exactly_current_full_path() -> None:
    kwargs = {
        "check_gripper": True,
        "use_force_dynamics": True,
        "penalty_power": 2.0,
    }
    implicit = compute_dense_reward(_episode(), [_target()], np.zeros(3), **kwargs)
    explicit = compute_dense_reward(
        _episode(),
        [_target()],
        np.zeros(3),
        force_gate_enabled=True,
        **kwargs,
    )
    np.testing.assert_array_equal(implicit, explicit)


def test_ablation_changes_only_final_force_multiplier() -> None:
    episode = _episode()
    target = _target()
    state_sequence = np.zeros(3)
    kwargs = {
        "check_gripper": True,
        "use_force_dynamics": True,
        "penalty_power": 2.0,
    }
    full = compute_dense_reward(episode, [target], state_sequence, **kwargs)
    no_gate = compute_dense_reward(
        episode,
        [target],
        state_sequence,
        force_gate_enabled=False,
        **kwargs,
    )

    states = np.asarray([step["observations"]["state"] for step in episode])
    forces = states[:, 1:4]
    positions = states[:, 4:7]
    dforces = np.gradient(forces, axis=0)
    expected_force_multipliers = []
    for position, force, dforce in zip(positions, forces, dforces, strict=True):
        static_match = np.exp(-0.5 * np.linalg.norm(force))
        dynamics_match = np.exp(-0.5 * np.linalg.norm(dforce))
        force_match = 0.5 * static_match + 0.5 * dynamics_match
        spatial_confidence = np.exp(-20.0 * np.linalg.norm(position))
        validity = (1.0 - spatial_confidence) + spatial_confidence * force_match
        expected_force_multipliers.append(max(0.05, validity**2.0))

    np.testing.assert_allclose(
        full,
        no_gate * np.asarray(expected_force_multipliers),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(full < no_gate)


def test_force_free_task_is_rejected_before_data_loading() -> None:
    with pytest.raises(ValueError, match="state_force_slice=None"):
        evaluate_task("pushcube")
