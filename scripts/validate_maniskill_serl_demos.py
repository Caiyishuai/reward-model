#!/usr/bin/env python3
"""Validate ManiSkill normalized-dense demonstrations before SERL training."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--camera-keys", nargs="+", default=["base_camera", "hand_camera"])
    return parser.parse_args()


def validate(path: Path, expected_episodes: int, camera_keys: list[str]) -> dict[str, object]:
    with path.open("rb") as file:
        transitions = pickle.load(file)
    if not isinstance(transitions, list) or not transitions:
        raise ValueError(f"{path}: expected a non-empty list of transitions")

    required = {"observations", "next_observations", "actions", "rewards", "masks", "dones"}
    episode_count = 0
    rewards = []
    for index, transition in enumerate(transitions):
        missing = required - set(transition)
        if missing:
            raise ValueError(f"{path}: transition {index} missing {sorted(missing)}")
        for obs_field in ("observations", "next_observations"):
            obs = transition[obs_field]
            if np.asarray(obs["state"]).ndim != 1:
                raise ValueError(f"{path}: transition {index} {obs_field}.state must be 1-D")
            for camera in camera_keys:
                image = np.asarray(obs[camera])
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(
                        f"{path}: transition {index} {obs_field}.{camera} "
                        f"must be HxWx3 uint8, got {image.shape} {image.dtype}"
                    )
        action = np.asarray(transition["actions"])
        if action.shape != (7,):
            raise ValueError(f"{path}: transition {index} action must be (7,), got {action.shape}")
        if not np.isfinite(action).all() or float(np.max(np.abs(action))) > 1.0001:
            raise ValueError(f"{path}: transition {index} action is non-finite or outside [-1, 1]")
        reward = float(transition["rewards"])
        if not np.isfinite(reward):
            raise ValueError(f"{path}: transition {index} reward is non-finite")
        rewards.append(reward)
        if bool(transition["dones"]):
            episode_count += 1
            if float(transition["masks"]) != 0.0:
                raise ValueError(f"{path}: terminal transition {index} must have mask=0")

    if episode_count != expected_episodes:
        raise ValueError(f"{path}: expected {expected_episodes} episodes, found {episode_count}")
    reward_array = np.asarray(rewards, dtype=np.float32)
    if reward_array.min() < -1e-5 or reward_array.max() > 1.0001:
        raise ValueError(
            f"{path}: normalized_dense rewards expected in [0,1], "
            f"got [{reward_array.min():.4f}, {reward_array.max():.4f}]"
        )

    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{path}: missing metadata sidecar {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("reward_mode") != "normalized_dense":
        raise ValueError(f"{metadata_path}: reward_mode must be normalized_dense")
    if int(metadata.get("collected_episodes", -1)) != expected_episodes:
        raise ValueError(f"{metadata_path}: collected_episodes mismatch")

    return {
        "path": str(path),
        "episodes": episode_count,
        "transitions": len(transitions),
        "state_dim": int(np.asarray(transitions[0]["observations"]["state"]).size),
        "reward_min": float(reward_array.min()),
        "reward_max": float(reward_array.max()),
        "reward_mean": float(reward_array.mean()),
    }


def main() -> int:
    args = parse_args()
    summaries = [validate(path, args.expected_episodes, args.camera_keys) for path in args.datasets]
    print(json.dumps(summaries, indent=2))
    print(f"[OK] validated {len(summaries)} dataset(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
