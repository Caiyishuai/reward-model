"""Unified ManiSkill3 environment creation.

Handles gym.make parameters, action space flattening, and GPU backend
selection in one place so that rm_eval and ppo_train share identical
environment setup.
"""

from __future__ import annotations

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from sim.task_configs import SimTaskConfig


def make_env(cfg: SimTaskConfig, num_envs: int = 1) -> gym.Env:
    """Create a ManiSkill3 environment from *cfg*.

    Args:
        cfg: Task-specific simulation configuration.
        num_envs: Number of parallel environments (GPU vectorised).

    Returns:
        A gymnasium environment ready for stepping.
    """
    env = gym.make(
        cfg.env_id,
        num_envs=num_envs,
        obs_mode=cfg.obs_mode,
        render_mode="rgb_array",
        sim_backend=cfg.sim_backend,
        robot_uids=cfg.robot_uid,
        control_mode=cfg.control_mode,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env
