#!/usr/bin/env python3
"""Strict validator for MetaWorld three-camera visual DrQ demos."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from metaworld_common import ROBOT_STATE_DIM, TASK_SPECS, VISUAL_CAMERA_SCHEMA, get_task_spec


def load_replay(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        return pickle.load(file)


def validate_task(task: str, root: Path, expected_episodes: int, image_size: int) -> dict:
    path = root / get_task_spec(task).rsync_name / "visual_drq" / "success_demos.pkl.gz"
    metadata_path = path.parent / "metadata.json"
    errors: list[str] = []
    if not path.exists():
        return {"episodes": 0, "transitions": 0, "errors": [f"missing {path}"]}
    if not metadata_path.exists():
        errors.append(f"missing {metadata_path}")
    else:
        metadata = json.loads(metadata_path.read_text())
        expected_metadata = {
            "task": task,
            "episodes": expected_episodes,
            "successful_episodes": expected_episodes,
            "image_size": image_size,
            "image_dtype": "uint8",
            "camera_schema": VISUAL_CAMERA_SCHEMA,
            "state_dim": ROBOT_STATE_DIM,
            "policy_reads_object_or_goal_state": False,
            "reward": "sparse info.success",
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                errors.append(f"metadata {key}={metadata.get(key)!r}, expected={expected!r}")
    data = load_replay(path)
    required = {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "masks",
        "dones",
        "episode_index",
        "episode_step",
        "episode_seed",
    }
    if missing := sorted(required - set(data)):
        return {"episodes": 0, "transitions": 0, "errors": [f"missing keys {missing}"]}
    count = len(data["rewards"])
    episode_index = np.asarray(data["episode_index"])
    episodes = np.unique(episode_index)
    if len(episodes) != expected_episodes or not np.array_equal(
        episodes, np.arange(expected_episodes)
    ):
        errors.append(f"expected episode ids 0..{expected_episodes - 1}, got {episodes.tolist()}")

    expected_keys = {"state", *VISUAL_CAMERA_SCHEMA}
    for obs_name in ("observations", "next_observations"):
        observation = data[obs_name]
        if set(observation) != expected_keys:
            errors.append(f"{obs_name} keys={sorted(observation)}, expected={sorted(expected_keys)}")
        state = np.asarray(observation.get("state", []))
        if state.shape != (count, ROBOT_STATE_DIM) or state.dtype != np.float32:
            errors.append(f"{obs_name}.state shape/dtype={state.shape}/{state.dtype}")
        elif not np.isfinite(state).all():
            errors.append(f"{obs_name}.state contains non-finite values")
        for key in VISUAL_CAMERA_SCHEMA:
            image = np.asarray(observation.get(key, []))
            expected_shape = (count, image_size, image_size, 3)
            if image.shape != expected_shape or image.dtype != np.uint8:
                errors.append(f"{obs_name}.{key} shape/dtype={image.shape}/{image.dtype}")
        if all(key in observation for key in VISUAL_CAMERA_SCHEMA):
            first_images = [np.asarray(observation[key][0]) for key in VISUAL_CAMERA_SCHEMA]
            if any(
                np.array_equal(first_images[left], first_images[right])
                for left in range(len(first_images))
                for right in range(left + 1, len(first_images))
            ):
                errors.append(f"{obs_name}: camera views are unexpectedly identical")

    expected_arrays = {
        "actions": ((count, 4), np.float32),
        "rewards": ((count,), np.float32),
        "masks": ((count,), np.float32),
        "dones": ((count,), np.bool_),
        "episode_index": ((count,), np.int32),
        "episode_step": ((count,), np.int32),
        "episode_seed": ((count,), np.int64),
    }
    for key, (shape, dtype) in expected_arrays.items():
        array = np.asarray(data[key])
        if array.shape != shape or array.dtype != dtype:
            errors.append(f"{key} shape/dtype={array.shape}/{array.dtype}, expected={shape}/{dtype}")
    if not np.isin(data["rewards"], [0.0, 1.0]).all():
        errors.append("rewards are not sparse binary values")
    if not np.isin(data["masks"], [0.0, 1.0]).all():
        errors.append("masks are not binary")

    for episode in episodes:
        indices = np.flatnonzero(episode_index == episode)
        if not np.array_equal(np.asarray(data["episode_step"])[indices], np.arange(len(indices))):
            errors.append(f"episode {episode}: non-contiguous episode_step")
        dones = np.asarray(data["dones"])[indices]
        rewards = np.asarray(data["rewards"])[indices]
        masks = np.asarray(data["masks"])[indices]
        if dones[:-1].any() or not dones[-1]:
            errors.append(f"episode {episode}: invalid done boundary")
        if masks[-1] != 0.0 or (masks[:-1] != 1.0).any():
            errors.append(f"episode {episode}: invalid masks")
        if rewards[-1] != 1.0 or rewards[:-1].any():
            errors.append(f"episode {episode}: expected exactly terminal sparse reward")
        seeds = np.unique(np.asarray(data["episode_seed"])[indices])
        if len(seeds) != 1:
            errors.append(f"episode {episode}: multiple seeds")

    # Consecutive transitions must preserve all observation modalities exactly.
    nonterminal = np.flatnonzero(~np.asarray(data["dones"]))[:100]
    for index in nonterminal:
        if episode_index[index] != episode_index[index + 1]:
            errors.append(f"transition {index}: nonterminal crosses episode")
            break
        for key in expected_keys:
            if not np.array_equal(
                data["next_observations"][key][index],
                data["observations"][key][index + 1],
            ):
                errors.append(f"transition {index}: {key} continuity failure")
                break
    return {"episodes": int(len(episodes)), "transitions": count, "errors": errors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("eval_results/metaworld_visual_demo_validation.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = list(TASK_SPECS) if "all" in args.tasks else args.tasks
    report = {}
    valid = True
    for task in tasks:
        result = validate_task(task, args.data_root, args.expected_episodes, args.image_size)
        report[task] = result
        valid &= not result["errors"]
        print(
            f"{task:14s} episodes={result['episodes']:2d} "
            f"transitions={result['transitions']:5d} errors={len(result['errors'])}"
        )
    payload = {"valid": valid, "expected_episodes": args.expected_episodes, "tasks": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
