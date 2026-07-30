#!/usr/bin/env python3
"""Run MetaWorld SERL with a learned six-axis wrist-wrench gate.

The underlying online training loop remains the tested implementation in the
external SERL checkout. This launcher replaces only:

1. the environment observation with ``39-D state + 6-D wrist_wrench``;
2. the actor/critic MLP backbones with force-gated variants.

The force representation follows hil-serl's proprioception convention:
``tcp_force(3) + tcp_torque(3)``. Unlike the current MetaWorld integration,
this policy is still state-based; RGB remains available to the reward model.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from metaworld_common import WristWrenchSensor

STATE_DIM = 39
WRENCH_DIM = 6
POLICY_OBSERVATION_DIM = STATE_DIM + WRENCH_DIM


class ForceGateFusion(nn.Module):
    """Learned soft gate over a normalized six-axis wrist wrench."""

    force_hidden_dim: int = 32
    force_scale: tuple[float, ...] = (30.0, 30.0, 30.0, 3.0, 3.0, 3.0)

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        state = observations[..., :STATE_DIM]
        wrench = observations[..., STATE_DIM:POLICY_OBSERVATION_DIM]
        scale = jnp.asarray(self.force_scale, dtype=wrench.dtype)
        normalized_wrench = jnp.clip(wrench / scale, -5.0, 5.0)

        force_features = nn.Dense(self.force_hidden_dim, name="force_projection")(normalized_wrench)
        force_features = nn.tanh(nn.LayerNorm(name="force_norm")(force_features))
        gate_inputs = jnp.concatenate([state, normalized_wrench], axis=-1)
        gate = nn.sigmoid(nn.Dense(self.force_hidden_dim, name="force_gate")(gate_inputs))
        return jnp.concatenate([state, gate * force_features], axis=-1)


class ForceGatedBackbone(nn.Module):
    """Actor or critic MLP with a dedicated gated force branch."""

    hidden_dims: tuple[int, ...] = (256, 256)
    critic: bool = False
    action_dim: int = 4

    @nn.compact
    def __call__(self, inputs: jax.Array, train: bool = False) -> jax.Array:
        del train
        observations = inputs[..., :POLICY_OBSERVATION_DIM]
        fused = ForceGateFusion(name="force_fusion")(observations)
        if self.critic:
            actions = inputs[..., POLICY_OBSERVATION_DIM : POLICY_OBSERVATION_DIM + self.action_dim]
            fused = jnp.concatenate([fused, actions], axis=-1)
        for index, width in enumerate(self.hidden_dims):
            fused = nn.Dense(width, name=f"dense_{index}")(fused)
            fused = nn.tanh(nn.LayerNorm(name=f"layer_norm_{index}")(fused))
        return fused


class WristWrenchObservationWrapper(gym.Wrapper):
    """Append filtered virtual wrist force/torque to MetaWorld's 39-D state."""

    def __init__(
        self,
        env: gym.Env,
        *,
        filter_alpha: float,
        force_clip: float,
        torque_clip: float,
    ):
        super().__init__(env)
        if env.observation_space.shape != (STATE_DIM,):
            raise ValueError(f"Expected MetaWorld state shape {(STATE_DIM,)}, got {env.observation_space.shape}")
        self.sensor = WristWrenchSensor(
            env,
            filter_alpha=filter_alpha,
            force_clip=force_clip,
            torque_clip=torque_clip,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(POLICY_OBSERVATION_DIM,),
            dtype=np.float32,
        )

    @staticmethod
    def _combine(observation: Any, wrench: np.ndarray) -> np.ndarray:
        state = np.asarray(observation, dtype=np.float32)
        return np.concatenate([state, wrench]).astype(np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        wrench = self.sensor.reset().wrist_wrench
        return self._combine(observation, wrench), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        wrench_observation = self.sensor.read()
        info = dict(info)
        info["wrist_wrench"] = wrench_observation.wrist_wrench.copy()
        info["max_contact_force"] = wrench_observation.max_contact_force
        info["contact_count"] = wrench_observation.contact_count
        return (
            self._combine(observation, wrench_observation.wrist_wrench),
            reward,
            terminated,
            truncated,
            info,
        )


def make_force_gated_agent(
    *,
    seed: int,
    sample_obs: np.ndarray,
    sample_action: np.ndarray,
    adaptive_tau_enabled: bool,
    critic_loss_threshold: float,
    tau_min: float,
    tau_max: float,
    tau_adjust_factor: float,
    tau_adjust_tolerance: float,
):
    from serl_launcher.agents.continuous.sac import SACAgent
    from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
    from serl_launcher.networks.lagrange import GeqLagrangeMultiplier

    if np.asarray(sample_obs).shape != (POLICY_OBSERVATION_DIM,):
        raise ValueError(
            f"Force-gated agent expects observation shape {(POLICY_OBSERVATION_DIM,)}, "
            f"got {np.asarray(sample_obs).shape}"
        )
    action_dim = int(np.asarray(sample_action).shape[-1])
    actor = Policy(
        encoder=None,
        network=ForceGatedBackbone(critic=False, action_dim=action_dim),
        action_dim=action_dim,
        tanh_squash_distribution=True,
        std_parameterization="exp",
        std_min=1e-5,
        std_max=5,
        name="actor",
    )
    critic_class = partial(
        Critic,
        encoder=None,
        network=ForceGatedBackbone(critic=True, action_dim=action_dim),
    )
    critic = ensemblize(critic_class, 10)(name="critic")
    temperature = GeqLagrangeMultiplier(
        init_value=1e-2,
        constraint_shape=(),
        constraint_type="geq",
        name="temperature",
    )
    return SACAgent.create(
        jax.random.PRNGKey(seed),
        jnp.asarray(sample_obs),
        jnp.asarray(sample_action),
        actor_def=actor,
        critic_def=critic,
        temperature_def=temperature,
        critic_ensemble_size=10,
        critic_subsample_size=2,
        discount=0.99,
        backup_entropy=False,
        adaptive_tau_enabled=adaptive_tau_enabled,
        critic_loss_threshold=critic_loss_threshold,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_adjust_factor=tau_adjust_factor,
        tau_adjust_tolerance=tau_adjust_tolerance,
    )


def parse_launcher_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--serl-root", type=Path, required=True)
    parser.add_argument("--wrench-filter-alpha", type=float, default=0.2)
    parser.add_argument("--wrench-force-clip", type=float, default=100.0)
    parser.add_argument("--wrench-torque-clip", type=float, default=10.0)
    return parser.parse_known_args()


def main() -> int:
    launcher_args, trainer_args = parse_launcher_args()
    serl_root = launcher_args.serl_root.resolve()
    sys.path.insert(0, str(serl_root / "auto_research" / "scripts"))
    sys.path.insert(0, str(serl_root / "serl_launcher"))

    import train_serl_metaworld as trainer

    original_make_env = trainer.make_env
    original_parse_args = trainer.parse_args

    def make_force_env(*args, **kwargs):
        env = original_make_env(*args, **kwargs)
        return WristWrenchObservationWrapper(
            env,
            filter_alpha=launcher_args.wrench_filter_alpha,
            force_clip=launcher_args.wrench_force_clip,
            torque_clip=launcher_args.wrench_torque_clip,
        )

    def parse_force_args():
        parsed = original_parse_args()
        if parsed.reward_mode not in {"dense", "sparse"}:
            raise ValueError("Force-gated MetaWorld trainer currently supports dense/sparse rewards only")
        parsed.policy_input = "state+wrist_wrench"
        parsed.state_dim = STATE_DIM
        parsed.wrist_wrench_dim = WRENCH_DIM
        parsed.force_gate = "learned_soft_gate"
        parsed.wrench_filter_alpha = launcher_args.wrench_filter_alpha
        parsed.wrench_force_clip = launcher_args.wrench_force_clip
        parsed.wrench_torque_clip = launcher_args.wrench_torque_clip
        return parsed

    trainer.make_env = make_force_env
    trainer.make_sac_agent = make_force_gated_agent
    trainer.parse_args = parse_force_args
    sys.argv = [str(Path(trainer.__file__)), *trainer_args]
    return int(trainer.main())


if __name__ == "__main__":
    raise SystemExit(main())
