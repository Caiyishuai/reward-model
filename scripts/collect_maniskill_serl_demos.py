#!/usr/bin/env python3
"""Collect successful ManiSkill demonstrations for visual SERL/RLPD.

This collector intentionally uses one environment and stores the exact
``normalized_dense`` reward returned by ManiSkill. The output is a flat list
of SERL replay-buffer transitions, while episode boundaries are retained via
``dones``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

TASKS = ("PushCube-v1", "PokeCube-v1", "PlaceSphere-v1", "StackCube-v1")


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class PPOAgent(nn.Module):
    """Network architecture used by maniskill-ws/rl/1_ppo_fast.py."""

    def __init__(self, obs_dim: int, action_dim: int, device: torch.device):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim, device=device), std=0.01 * math.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim, device=device))


@dataclass
class CollectionMetadata:
    env_id: str
    checkpoint: str
    reward_mode: str
    control_mode: str
    robot_uid: str
    sim_backend: str
    camera_keys: list[str]
    requested_episodes: int
    collected_episodes: int
    attempts: int
    transitions: int
    seed: int
    mani_skill_version: str


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value: Any) -> float:
    return float(to_numpy(value).reshape(-1)[0])


def strip_env_axis(value: Any) -> np.ndarray:
    array = to_numpy(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array


def policy_state(obs: dict[str, Any], device: torch.device) -> torch.Tensor:
    state = torch.as_tensor(obs["state"], dtype=torch.float32, device=device)
    return state.reshape(1, -1)


def serl_observation(obs: dict[str, Any], camera_keys: list[str]) -> dict[str, np.ndarray]:
    result = {"state": strip_env_axis(obs["state"]).reshape(-1).astype(np.float32)}
    sensor_data = obs.get("sensor_data", {})
    for camera in camera_keys:
        if camera not in sensor_data or "rgb" not in sensor_data[camera]:
            available = sorted(key for key, value in sensor_data.items() if "rgb" in value)
            raise KeyError(f"Missing RGB camera {camera!r}; available cameras: {available}")
        image = strip_env_axis(sensor_data[camera]["rgb"])
        if image.dtype != np.uint8:
            scale = 255.0 if image.size and float(image.max()) <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255).astype(np.uint8)
        result[camera] = image
    return result


def unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    for key in ("agent", "model", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise ValueError("Checkpoint contains no tensor state_dict")
    return state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True, choices=TASKS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--robot-uid", default="panda_wristcam")
    parser.add_argument("--sim-backend", default="physx_cuda")
    parser.add_argument("--camera-keys", nargs="+", default=["base_camera", "hand_camera"])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes < 1 or args.max_attempts < args.episodes:
        raise ValueError("Require episodes >= 1 and max-attempts >= episodes")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.sim_backend == "physx_cuda" and not torch.cuda.is_available():
        raise RuntimeError("physx_cuda collection requires a CUDA-enabled PyTorch build")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="rgb+state",
        render_mode="sensors",
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        reward_mode="normalized_dense",
        robot_uids=args.robot_uid,
        max_episode_steps=args.max_episode_steps,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(policy_state(obs, device).shape[-1])
    action_shape = tuple(int(dim) for dim in env.action_space.shape)
    action_dim = int(np.prod(action_shape[1:] if len(action_shape) > 1 and action_shape[0] == 1 else action_shape))
    if action_dim != 7:
        raise ValueError(f"Expected a 7-D action space, got {action_shape}")

    agent = PPOAgent(obs_dim, action_dim, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    agent.eval()

    episodes: list[list[dict[str, Any]]] = []
    attempts = 0
    try:
        while len(episodes) < args.episodes and attempts < args.max_attempts:
            obs, _ = env.reset(seed=args.seed + attempts)
            trajectory: list[dict[str, Any]] = []
            success_once = False
            horizon = args.max_episode_steps or getattr(env.spec, "max_episode_steps", 200) or 200

            for _ in range(horizon):
                current = serl_observation(obs, args.camera_keys)
                with torch.no_grad():
                    action = agent.actor_mean(policy_state(obs, device)).clamp(-1.0, 1.0)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = scalar(terminated) > 0.5 or scalar(truncated) > 0.5
                success_once = success_once or scalar(info.get("success", False)) > 0.5
                trajectory.append(
                    {
                        "observations": current,
                        "actions": to_numpy(action).reshape(-1).astype(np.float32),
                        "next_observations": serl_observation(next_obs, args.camera_keys),
                        "rewards": np.float32(scalar(reward)),
                        "masks": np.float32(0.0 if done else 1.0),
                        "dones": bool(done),
                    }
                )
                obs = next_obs
                if done:
                    break

            attempts += 1
            if success_once:
                trajectory[-1]["dones"] = True
                trajectory[-1]["masks"] = np.float32(0.0)
                episodes.append(trajectory)
                print(
                    f"[{args.env_id}] success={len(episodes)}/{args.episodes} "
                    f"attempt={attempts} steps={len(trajectory)} "
                    f"return={sum(float(step['rewards']) for step in trajectory):.3f}",
                    flush=True,
                )
    finally:
        env.close()

    transitions = [transition for episode in episodes for transition in episode]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(transitions, file, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = CollectionMetadata(
        env_id=args.env_id,
        checkpoint=str(args.checkpoint.resolve()),
        reward_mode="normalized_dense",
        control_mode=args.control_mode,
        robot_uid=args.robot_uid,
        sim_backend=args.sim_backend,
        camera_keys=args.camera_keys,
        requested_episodes=args.episodes,
        collected_episodes=len(episodes),
        attempts=attempts,
        transitions=len(transitions),
        seed=args.seed,
        mani_skill_version=importlib.metadata.version("mani-skill"),
    )
    args.output.with_suffix(".json").write_text(json.dumps(asdict(metadata), indent=2) + "\n")

    if len(episodes) != args.episodes:
        print(f"[INCOMPLETE] collected {len(episodes)}/{args.episodes} successful episodes")
        return 2
    print(f"[OK] wrote {len(episodes)} episodes / {len(transitions)} transitions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
