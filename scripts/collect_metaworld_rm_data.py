#!/usr/bin/env python3
"""Collect MetaWorld demonstrations for both Rsync RM and SERL.

For each selected task this writes:

```
<reward-model-root>/data/<mw_task>/
  success_raw.pkl       # Rsync label input, RGB + state
  fail_raw.pkl
  serl_dense.pkl        # stacked state-only SERL transitions
  serl_sparse.pkl
  collection_meta.json
```

Successes use MetaWorld's task-specific scripted expert.  Failures use either a
random policy or a noisy expert and are accepted only when ``info["success"]``
never becomes true.  Every episode gets a unique reset seed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from metaworld_common import TASK_SPECS, get_task_spec, make_env, make_scripted_policy, render_rgb, success_from_info


@dataclass
class CollectionMetadata:
    task: str
    env_name: str
    rsync_name: str
    metaworld_version: str
    reward_function_version: str
    camera_key: str
    image_size: int
    max_episode_steps: int
    success_episodes: int
    fail_episodes: int
    failure_policy: str
    failure_noise_std: float
    seed: int
    success_attempts: int
    fail_attempts: int
    state_dim: int
    action_dim: int


def _atomic_pickle(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _rollout(
    task: str,
    seed: int,
    *,
    scripted: bool,
    failure_policy: str,
    failure_noise_std: float,
    image_size: int,
    max_episode_steps: int,
    reward_function_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    env = make_env(
        task,
        seed=seed,
        render=True,
        image_size=image_size,
        reward_function_version=reward_function_version,
    )
    policy = make_scripted_policy(task)
    rng = np.random.default_rng(seed)
    raw_steps: list[dict[str, Any]] = []
    serl_steps: list[dict[str, Any]] = []
    ever_succeeded = False

    try:
        observation, _ = env.reset(seed=seed)
        observation = np.asarray(observation, dtype=np.float32)

        for frame in range(max_episode_steps):
            if scripted or failure_policy == "noisy-expert":
                action = np.asarray(policy.get_action(observation), dtype=np.float32)
                if not scripted:
                    action += rng.normal(0.0, failure_noise_std, size=action.shape).astype(np.float32)
            else:
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            next_observation, dense_reward, terminated, truncated, info = env.step(action)
            next_observation = np.asarray(next_observation, dtype=np.float32)
            succeeded = success_from_info(info)
            ever_succeeded = ever_succeeded or succeeded
            horizon = frame + 1 >= max_episode_steps
            done = bool(terminated or truncated or horizon or succeeded)
            image = render_rgb(env)

            raw_steps.append(
                {
                    "observations": {
                        "state": next_observation.copy(),
                        "corner2": image,
                    },
                    "previous_observations": {"state": observation.copy()},
                    "actions": action.copy(),
                    "rewards": 0.0,
                    "env_rewards": np.float32(dense_reward),
                    "sparse_rewards": np.float32(succeeded),
                    "dones": int(done),
                    "infos": {
                        "succeed": bool(succeeded and done),
                        "episode_seed": seed,
                    },
                }
            )
            serl_steps.append(
                {
                    "observations": observation.copy(),
                    "next_observations": next_observation.copy(),
                    "actions": action.copy(),
                    "dense_rewards": np.float32(dense_reward),
                    "sparse_rewards": np.float32(succeeded),
                    "masks": np.float32(0.0 if done else 1.0),
                    "dones": bool(done),
                }
            )
            observation = next_observation
            if done:
                break
    finally:
        env.close()

    return raw_steps, serl_steps, ever_succeeded


def _stack_serl(episodes: list[list[dict[str, Any]]], reward_key: str) -> dict[str, np.ndarray]:
    flat = [step for episode in episodes for step in episode]
    if not flat:
        raise ValueError("Cannot stack an empty episode set")
    result = {
        "observations": np.stack([step["observations"] for step in flat]).astype(np.float32),
        "next_observations": np.stack([step["next_observations"] for step in flat]).astype(np.float32),
        "actions": np.stack([step["actions"] for step in flat]).astype(np.float32),
        "rewards": np.asarray([step[reward_key] for step in flat], dtype=np.float32),
        "masks": np.asarray([step["masks"] for step in flat], dtype=np.float32),
        "dones": np.asarray([step["dones"] for step in flat], dtype=bool),
    }
    episode_indices = []
    for episode_index, episode in enumerate(episodes):
        episode_indices.extend([episode_index] * len(episode))
    result["episode_index"] = np.asarray(episode_indices, dtype=np.int32)
    return result


def _collect_category(
    task: str,
    target_count: int,
    seed_start: int,
    *,
    want_success: bool,
    args: argparse.Namespace,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]], int]:
    raw_episodes = []
    serl_episodes = []
    attempts = 0
    while len(raw_episodes) < target_count and attempts < args.max_attempts_per_category:
        seed = seed_start + attempts
        raw, serl, succeeded = _rollout(
            task,
            seed,
            scripted=want_success,
            failure_policy=args.failure_policy,
            failure_noise_std=args.failure_noise_std,
            image_size=args.image_size,
            max_episode_steps=args.max_episode_steps,
            reward_function_version=args.reward_function_version,
        )
        attempts += 1
        if succeeded != want_success:
            continue
        raw_episodes.append(raw)
        serl_episodes.append(serl)
        label = "success" if want_success else "fail"
        print(f"[{task}] {label} {len(raw_episodes)}/{target_count} seed={seed} frames={len(raw)}")
    if len(raw_episodes) != target_count:
        label = "success" if want_success else "fail"
        raise RuntimeError(
            f"{task}: collected {len(raw_episodes)}/{target_count} {label} episodes "
            f"after {attempts} attempts"
        )
    return raw_episodes, serl_episodes, attempts


def collect_task(task: str, args: argparse.Namespace) -> None:
    spec = get_task_spec(task)
    output_dir = args.output_root / "data" / spec.rsync_name

    success_raw, success_serl, success_attempts = _collect_category(
        task,
        args.num_success,
        args.seed,
        want_success=True,
        args=args,
    )
    fail_raw, fail_serl, fail_attempts = _collect_category(
        task,
        args.num_fail,
        args.seed + 1_000_000,
        want_success=False,
        args=args,
    )

    _atomic_pickle([step for episode in success_raw for step in episode], output_dir / "success_raw.pkl")
    _atomic_pickle([step for episode in fail_raw for step in episode], output_dir / "fail_raw.pkl")

    combined_serl = success_serl + fail_serl
    _atomic_pickle(_stack_serl(combined_serl, "dense_rewards"), output_dir / "serl_dense.pkl")
    _atomic_pickle(_stack_serl(combined_serl, "sparse_rewards"), output_dir / "serl_sparse.pkl")

    sample = combined_serl[0][0]
    metadata = CollectionMetadata(
        task=task,
        env_name=spec.env_name,
        rsync_name=spec.rsync_name,
        metaworld_version=importlib.metadata.version("metaworld"),
        reward_function_version=args.reward_function_version,
        camera_key="corner2",
        image_size=args.image_size,
        max_episode_steps=args.max_episode_steps,
        success_episodes=len(success_raw),
        fail_episodes=len(fail_raw),
        failure_policy=args.failure_policy,
        failure_noise_std=args.failure_noise_std,
        seed=args.seed,
        success_attempts=success_attempts,
        fail_attempts=fail_attempts,
        state_dim=int(sample["observations"].shape[0]),
        action_dim=int(sample["actions"].shape[0]),
    )
    (output_dir / "collection_meta.json").write_text(json.dumps(asdict(metadata), indent=2))
    print(f"[OK] {task} -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Rsync + SERL data for eight MetaWorld tasks")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument(
        "--output-root",
        "--rsync-root",
        dest="output_root",
        type=Path,
        required=True,
        help="Reward-model repository root. --rsync-root remains as a compatibility alias.",
    )
    parser.add_argument("--num-success", type=int, default=20)
    parser.add_argument("--num-fail", type=int, default=20)
    parser.add_argument("--max-attempts-per-category", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--failure-policy", choices=["random", "noisy-expert"], default="random")
    parser.add_argument("--failure-noise-std", type=float, default=0.5)
    parser.add_argument("--reward-function-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tasks = list(TASK_SPECS) if "all" in args.tasks else args.tasks
    unknown = sorted(set(tasks) - set(TASK_SPECS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; available: {sorted(TASK_SPECS)}")
    if args.num_success < 1 or args.num_fail < 1:
        raise ValueError("--num-success and --num-fail must both be positive")
    for task in tasks:
        collect_task(task, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
