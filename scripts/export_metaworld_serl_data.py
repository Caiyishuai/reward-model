#!/usr/bin/env python3
"""Export labeled MetaWorld episodes to SERL's stacked replay format."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.common import BASE_DIR  # noqa: E402

METAWORLD_TASKS = (
    "mw_button_press",
    "mw_window_open",
    "mw_reach_wall",
    "mw_plate_slide",
    "mw_push",
    "mw_coffee_push",
    "mw_stick_push",
    "mw_pick_place",
)

REWARD_COLUMNS = {
    "auto": "next.reward",
    "dense": "next.env_reward",
    "sparse": "next.sparse_reward",
}


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as file:
        return pickle.load(file)


def _validate(data: dict, reward_column: str, path: Path) -> None:
    required = {
        "observation.state",
        "previous_observation.state",
        "action",
        "episode_index",
        "next.done",
        reward_column,
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def export_task(task: str, prefix: str, reward_mode: str) -> Path:
    processed = Path(BASE_DIR) / task / f"{prefix}_processed"
    paths = [processed / "success_lerobot.pkl", processed / "fail_lerobot.pkl"]
    datasets = [_load(path) for path in paths]
    reward_column = REWARD_COLUMNS[reward_mode]
    for data, path in zip(datasets, paths, strict=True):
        _validate(data, reward_column, path)
    wrench_presence = ["observation.wrist_wrench" in data for data in datasets]
    if any(wrench_presence) and not all(wrench_presence):
        raise ValueError(f"{task}: wrist wrench columns must exist in both success and fail datasets")
    has_wrist_wrench = all(wrench_presence)

    columns: dict[str, list[np.ndarray]] = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "masks": [],
        "dones": [],
        "episode_index": [],
    }
    if has_wrist_wrench:
        columns["wrist_wrench"] = []
        columns["next_wrist_wrench"] = []
        columns["max_contact_force"] = []
    episode_offset = 0
    for data in datasets:
        done = np.asarray(data["next.done"], dtype=bool)
        episode_ids = np.asarray(data["episode_index"], dtype=np.int32)
        observations = np.asarray(data["previous_observation.state"], dtype=np.float32)
        next_observations = np.asarray(data["observation.state"], dtype=np.float32)
        if has_wrist_wrench:
            wrist_wrench = np.asarray(data["previous_observation.wrist_wrench"], dtype=np.float32)
            next_wrist_wrench = np.asarray(data["observation.wrist_wrench"], dtype=np.float32)
            observations = np.concatenate([observations, wrist_wrench], axis=1)
            next_observations = np.concatenate([next_observations, next_wrist_wrench], axis=1)
            columns["wrist_wrench"].append(wrist_wrench)
            columns["next_wrist_wrench"].append(next_wrist_wrench)
            columns["max_contact_force"].append(np.asarray(data["next.max_contact_force"], dtype=np.float32))
        columns["observations"].append(observations)
        columns["next_observations"].append(next_observations)
        columns["actions"].append(np.asarray(data["action"], dtype=np.float32))
        columns["rewards"].append(np.asarray(data[reward_column], dtype=np.float32))
        columns["masks"].append((~done).astype(np.float32))
        columns["dones"].append(done)
        columns["episode_index"].append(episode_ids + episode_offset)
        episode_offset += int(episode_ids.max(initial=-1)) + 1

    stacked = {key: np.concatenate(parts, axis=0) for key, parts in columns.items()}
    expected_observation_dim = 45 if has_wrist_wrench else 39
    if stacked["observations"].shape[1] != expected_observation_dim or stacked["actions"].shape[1] != 4:
        raise ValueError(
            f"{task}: expected obs_dim={expected_observation_dim}/action_dim=4, got "
            f"{stacked['observations'].shape}/{stacked['actions'].shape}"
        )

    output = Path(BASE_DIR) / task / f"serl_{reward_mode}.pkl"
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(stacked, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    print(
        f"{task:20s} mode={reward_mode:6s} N={len(stacked['rewards']):6d} "
        f"episodes={len(np.unique(stacked['episode_index'])):3d} "
        f"obs_dim={expected_observation_dim} -> {output}"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MetaWorld auto/dense/sparse rewards for SERL")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument(
        "--reward-modes", nargs="+", choices=sorted(REWARD_COLUMNS), default=["auto", "dense", "sparse"]
    )
    parser.add_argument("--prefix", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = list(METAWORLD_TASKS) if "all" in args.tasks else args.tasks
    unknown = sorted(set(tasks) - set(METAWORLD_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; expected {METAWORLD_TASKS}")
    for task in tasks:
        for reward_mode in args.reward_modes:
            export_task(task, args.prefix, reward_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
