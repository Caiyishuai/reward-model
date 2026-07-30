#!/usr/bin/env python3
"""Collect successful three-camera, robot-only MetaWorld demonstrations."""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from metaworld_common import (
    ROBOT_STATE_DIM,
    TASK_SPECS,
    VISUAL_CAMERA_SCHEMA,
    WristWrenchSensor,
    get_task_spec,
    make_env,
    make_scripted_policy,
    render_visual_observation,
    robot_only_state,
    success_from_info,
)


def _atomic_gzip_pickle(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=3) as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _rollout(task: str, seed: int, args: argparse.Namespace) -> tuple[list[dict[str, Any]], bool]:
    env = make_env(task, seed=seed, render=True, image_size=args.image_size)
    policy = make_scripted_policy(task)
    sensor = WristWrenchSensor(
        env,
        filter_mode=args.force_filter,
        filter_alpha=args.wrench_filter_alpha,
        force_clip=args.wrench_force_clip,
        torque_clip=args.wrench_torque_clip,
    )
    transitions: list[dict[str, Any]] = []
    succeeded = False
    try:
        expert_observation, _ = env.reset(seed=seed)
        wrench = sensor.reset().wrist_wrench
        state = robot_only_state(env, wrench)
        images = render_visual_observation(env)
        for episode_step in range(args.max_episode_steps):
            action = np.clip(
                np.asarray(policy.get_action(expert_observation), dtype=np.float32),
                -1.0,
                1.0,
            )
            next_expert_observation, _, terminated, truncated, info = env.step(action)
            next_wrench = sensor.read().wrist_wrench
            next_state = robot_only_state(env, next_wrench)
            next_images = render_visual_observation(env)
            succeeded = succeeded or success_from_info(info)
            horizon = episode_step + 1 >= args.max_episode_steps
            done = bool(terminated or truncated or horizon or succeeded)
            transitions.append(
                {
                    "observations": {"state": state, **images},
                    "next_observations": {"state": next_state, **next_images},
                    "actions": action,
                    "rewards": np.float32(succeeded),
                    "masks": np.float32(0.0 if done else 1.0),
                    "dones": done,
                    "episode_step": episode_step,
                    "episode_seed": seed,
                }
            )
            expert_observation = next_expert_observation
            state, images = next_state, next_images
            if done:
                break
    finally:
        env.close()
    return transitions, succeeded


def _stack(episodes: list[list[dict[str, Any]]]) -> dict[str, Any]:
    flat = [transition for episode in episodes for transition in episode]
    observations = {
        key: np.stack([transition["observations"][key] for transition in flat])
        for key in ("state", *VISUAL_CAMERA_SCHEMA)
    }
    next_observations = {
        key: np.stack([transition["next_observations"][key] for transition in flat])
        for key in ("state", *VISUAL_CAMERA_SCHEMA)
    }
    episode_index = np.concatenate(
        [np.full(len(episode), index, dtype=np.int32) for index, episode in enumerate(episodes)]
    )
    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": np.stack([transition["actions"] for transition in flat]).astype(np.float32),
        "rewards": np.asarray([transition["rewards"] for transition in flat], dtype=np.float32),
        "masks": np.asarray([transition["masks"] for transition in flat], dtype=np.float32),
        "dones": np.asarray([transition["dones"] for transition in flat], dtype=bool),
        "episode_index": episode_index,
        "episode_step": np.asarray([transition["episode_step"] for transition in flat], dtype=np.int32),
        "episode_seed": np.asarray([transition["episode_seed"] for transition in flat], dtype=np.int64),
    }


def collect_task(task: str, args: argparse.Namespace) -> None:
    episodes: list[list[dict[str, Any]]] = []
    attempts = 0
    while len(episodes) < args.num_demos and attempts < args.max_attempts:
        seed = args.seed + attempts
        episode, succeeded = _rollout(task, seed, args)
        attempts += 1
        if not succeeded:
            continue
        episodes.append(episode)
        print(
            f"[{task}] demo {len(episodes)}/{args.num_demos} "
            f"seed={seed} transitions={len(episode)}",
            flush=True,
        )
    if len(episodes) != args.num_demos:
        raise RuntimeError(
            f"{task}: collected {len(episodes)}/{args.num_demos} successful demos "
            f"after {attempts} attempts"
        )

    spec = get_task_spec(task)
    output_dir = args.output_root / spec.rsync_name / "visual_drq"
    replay_path = output_dir / "success_demos.pkl.gz"
    stacked = _stack(episodes)
    _atomic_gzip_pickle(stacked, replay_path)
    metadata = {
        "schema_version": 1,
        "task": task,
        "env_name": spec.observable_env_name,
        "metaworld_version": importlib.metadata.version("metaworld"),
        "episodes": len(episodes),
        "transitions": int(len(stacked["rewards"])),
        "successful_episodes": len(episodes),
        "attempts": attempts,
        "seed_start": args.seed,
        "image_size": args.image_size,
        "image_dtype": "uint8",
        "camera_schema": VISUAL_CAMERA_SCHEMA,
        "state_dim": ROBOT_STATE_DIM,
        "state_layout": [
            "tcp_world_xyz",
            "tcp_world_intrinsic_xyz_euler",
            "tcp_world_linear_velocity",
            "tcp_world_angular_velocity",
            "wrist_frame_Fx_Fy_Fz_Tx_Ty_Tz",
            "gripper_tip_distance_m",
        ],
        "policy_reads_object_or_goal_state": False,
        "reward": "sparse info.success",
        "force_filter": args.force_filter,
        "wrench_filter_alpha": args.wrench_filter_alpha,
        "wrench_force_clip": args.wrench_force_clip,
        "wrench_torque_clip": args.wrench_torque_clip,
        "max_episode_steps": args.max_episode_steps,
        "replay_path": replay_path.name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[OK] {task}: episodes={len(episodes)} transitions={len(stacked['rewards'])} -> {replay_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--num-demos", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--force-filter", choices=["ema", "none"], default="ema")
    parser.add_argument("--wrench-filter-alpha", type=float, default=0.2)
    parser.add_argument("--wrench-force-clip", type=float, default=100.0)
    parser.add_argument("--wrench-torque-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = list(TASK_SPECS) if "all" in args.tasks else args.tasks
    unknown = sorted(set(tasks) - set(TASK_SPECS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    if args.num_demos < 1:
        raise ValueError("--num-demos must be positive")
    for task in tasks:
        collect_task(task, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
