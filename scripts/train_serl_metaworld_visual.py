#!/usr/bin/env python3
"""Online sparse MetaWorld DrQ with three cameras and robot-only force state."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.machinery
import importlib.metadata
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# The launcher lives in reward-model, but intentionally reuses the current SERL.
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--serl-root", type=Path, required=True)
_bootstrap_args, _ = _bootstrap.parse_known_args()
sys.path.insert(0, str(_bootstrap_args.serl_root.resolve() / "serl_launcher"))

# SERL only uses these optional packages for typing/networked data stores. Keep
# the lightweight local CPU environment independent of TensorFlow/AgentLace.
if "tensorflow" not in sys.modules:
    tensorflow = types.ModuleType("tensorflow")
    tensorflow.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    tensorflow.Tensor = type("Tensor", (), {})
    tensorflow_io = types.ModuleType("tensorflow.io")
    tensorflow_gfile = types.ModuleType("tensorflow.io.gfile")
    tensorflow_gfile.GFile = open
    tensorflow_gfile.exists = os.path.exists
    tensorflow_gfile.makedirs = lambda path: os.makedirs(path, exist_ok=True)
    tensorflow_gfile.remove = os.remove
    tensorflow_gfile.rename = lambda source, target, overwrite=False: os.replace(source, target)
    tensorflow_gfile.isdir = os.path.isdir
    tensorflow_gfile.listdir = os.listdir
    tensorflow_errors = types.ModuleType("tensorflow.errors")
    tensorflow_errors.NotFoundError = FileNotFoundError
    tensorflow_io.gfile = tensorflow_gfile
    tensorflow.io = tensorflow_io
    tensorflow.errors = tensorflow_errors
    sys.modules["tensorflow"] = tensorflow
    sys.modules["tensorflow.io"] = tensorflow_io
    sys.modules["tensorflow.io.gfile"] = tensorflow_gfile
    sys.modules["tensorflow.errors"] = tensorflow_errors
if "agentlace" not in sys.modules:
    agentlace = types.ModuleType("agentlace")
    agentlace_data = types.ModuleType("agentlace.data")
    agentlace_store = types.ModuleType("agentlace.data.data_store")

    class DataStoreBase:
        def __init__(self, capacity):
            self.capacity = capacity

    agentlace_store.DataStoreBase = DataStoreBase
    agentlace_data.data_store = agentlace_store
    agentlace.data = agentlace_data
    sys.modules["agentlace"] = agentlace
    sys.modules["agentlace.data"] = agentlace_data
    sys.modules["agentlace.data.data_store"] = agentlace_store

import flax.linen as nn  # noqa: E402
import gymnasium as gym  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import serialization, traverse_util  # noqa: E402
from metaworld_common import (  # noqa: E402
    ROBOT_STATE_DIM,
    TASK_SPECS,
    VISUAL_CAMERA_SCHEMA,
    WristWrenchSensor,
    make_env,
    render_visual_observation,
    robot_only_state,
    success_from_info,
)
from serl_launcher.agents.continuous import drq as drq_module  # noqa: E402
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore  # noqa: E402
from serl_launcher.utils.train_utils import concat_batches  # noqa: E402

IMAGE_KEYS = tuple(VISUAL_CAMERA_SCHEMA)
WRENCH_SLICE = slice(12, 18)
BASE_STATE_INDICES = (*range(12), 18)


class ForceAwareEncodingWrapper(nn.Module):
    """SERL-compatible multi-camera encoder with trainable force fusion."""

    encoder: Any
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: tuple[str, ...] = IMAGE_KEYS
    force_fusion: str = "learned_gate"
    force_hidden_dim: int = 32
    encoder_type: str = "small"

    @nn.compact
    def __call__(self, observations, train=False, stop_gradient=False, is_encoded=False):
        encoded_images = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded and self.enable_stacking:
                if image.ndim == 4:
                    image = jnp.transpose(image, (1, 2, 0, 3)).reshape(
                        image.shape[1], image.shape[2], image.shape[0] * image.shape[3]
                    )
                elif image.ndim == 5:
                    image = jnp.transpose(image, (0, 2, 3, 1, 4)).reshape(
                        image.shape[0],
                        image.shape[2],
                        image.shape[3],
                        image.shape[1] * image.shape[4],
                    )
            if self.encoder_type == "resnet-pretrained":
                image = self.encoder[image_key](image, train=train, encode=not is_encoded)
            else:
                if is_encoded:
                    raise ValueError(f"is_encoded is unsupported for {self.encoder_type}")
                image = self.encoder[image_key](image, train=train)
            if stop_gradient:
                image = jax.lax.stop_gradient(image)
            encoded_images.append(image)
        encoded = jnp.concatenate(encoded_images, axis=-1)
        if not self.use_proprio:
            return encoded

        state = observations["state"]
        # Only images are frame-stacked by MemoryEfficientReplayBuffer. State
        # remains (19,) or (B,19), so a 2-D state is a batch, not T x C.
        if state.shape[-1] != ROBOT_STATE_DIM:
            raise ValueError(f"Expected {ROBOT_STATE_DIM}-D state, got {state.shape}")
        base = state[..., jnp.asarray(BASE_STATE_INDICES)]
        wrench = state[..., WRENCH_SLICE]
        scale = jnp.asarray([30.0, 30.0, 30.0, 3.0, 3.0, 3.0], dtype=state.dtype)
        normalized_wrench = jnp.clip(wrench / scale, -5.0, 5.0)

        if self.force_fusion == "none":
            proprio = base
        elif self.force_fusion == "concat":
            proprio = jnp.concatenate([base, normalized_wrench], axis=-1)
        elif self.force_fusion == "learned_gate":
            force_features = nn.Dense(self.force_hidden_dim, name="force_projection")(
                normalized_wrench
            )
            force_features = nn.tanh(nn.LayerNorm(name="force_norm")(force_features))
            gate_input = jnp.concatenate([base, normalized_wrench], axis=-1)
            gate = nn.sigmoid(
                nn.Dense(self.force_hidden_dim, name="force_gate")(gate_input)
            )
            proprio = jnp.concatenate([base, gate * force_features], axis=-1)
        else:
            raise ValueError(self.force_fusion)
        proprio = nn.Dense(
            self.proprio_latent_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="proprio_projection",
        )(proprio)
        proprio = nn.tanh(nn.LayerNorm(name="proprio_norm")(proprio))
        return jnp.concatenate([encoded, proprio], axis=-1)


class VisualRobotObservationWrapper(gym.Wrapper):
    """Replace privileged MetaWorld state with images plus robot-only state."""

    def __init__(
        self,
        env: gym.Env,
        *,
        image_size: int,
        force_filter: str,
        filter_alpha: float,
        force_clip: float,
        torque_clip: float,
    ):
        super().__init__(env)
        self.sensor = WristWrenchSensor(
            env,
            filter_mode=force_filter,
            filter_alpha=filter_alpha,
            force_clip=force_clip,
            torque_clip=torque_clip,
        )
        spaces: dict[str, gym.Space] = {
            "state": gym.spaces.Box(-np.inf, np.inf, (ROBOT_STATE_DIM,), dtype=np.float32)
        }
        for key in IMAGE_KEYS:
            # A single frame-stack dimension lets SERL's memory-efficient replay
            # store each frame once. On-disk demos remain the requested HWC schema.
            spaces[key] = gym.spaces.Box(
                0, 255, (1, image_size, image_size, 3), dtype=np.uint8
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def _observation(self, wrench: np.ndarray) -> dict[str, np.ndarray]:
        images = render_visual_observation(self.env)
        return {
            "state": robot_only_state(self.env, wrench),
            **{key: value[None] for key, value in images.items()},
        }

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        wrench = self.sensor.reset().wrist_wrench
        return self._observation(wrench), info

    def step(self, action):
        _, dense_reward, terminated, truncated, info = self.env.step(action)
        wrench = self.sensor.read().wrist_wrench
        info = dict(info)
        info["wrist_wrench"] = wrench.copy()
        return self._observation(wrench), dense_reward, terminated, truncated, info


def _load_replay(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        return pickle.load(file)


def _insert_demo(path: Path, buffer: MemoryEfficientReplayBufferDataStore) -> tuple[int, int]:
    data = _load_replay(path)
    count = len(data["rewards"])
    episodes = int(len(np.unique(data["episode_index"])))
    if episodes != 20:
        raise ValueError(f"{path}: demo buffer must contain exactly 20 episodes, got {episodes}")
    for index in range(count):
        observation = {
            "state": np.asarray(data["observations"]["state"][index], dtype=np.float32),
            **{
                key: np.asarray(data["observations"][key][index], dtype=np.uint8)[None]
                for key in IMAGE_KEYS
            },
        }
        next_observation = {
            "state": np.asarray(data["next_observations"]["state"][index], dtype=np.float32),
            **{
                key: np.asarray(data["next_observations"][key][index], dtype=np.uint8)[None]
                for key in IMAGE_KEYS
            },
        }
        buffer.insert(
            {
                "observations": observation,
                "next_observations": next_observation,
                "actions": np.asarray(data["actions"][index], dtype=np.float32),
                "rewards": np.float32(data["rewards"][index]),
                "masks": np.float32(data["masks"][index]),
                "dones": bool(data["dones"][index]),
            }
        )
    return episodes, count


def _save_agent(agent, output_dir: Path, step: int) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"agent_step_{step:09d}.msgpack"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(serialization.to_bytes(agent))
    temporary.replace(path)
    return path


def _gate_parameter_paths(agent) -> list[str]:
    flat = traverse_util.flatten_dict(agent.state.params, sep="/")
    return sorted(str(path) for path in flat if "force_gate" in str(path))


def _gate_parameter_values(agent) -> dict[str, np.ndarray]:
    flat = traverse_util.flatten_dict(agent.state.params, sep="/")
    return {
        str(path): np.asarray(value).copy()
        for path, value in flat.items()
        if "force_gate" in str(path)
    }


def _make_visual_env(args: argparse.Namespace, seed: int) -> VisualRobotObservationWrapper:
    base_env = make_env(args.task, seed=seed, render=True, image_size=args.image_size)
    return VisualRobotObservationWrapper(
        base_env,
        image_size=args.image_size,
        force_filter=args.force_filter,
        filter_alpha=args.wrench_filter_alpha,
        force_clip=args.wrench_force_clip,
        torque_clip=args.wrench_torque_clip,
    )


def evaluate_visual_policy(
    agent,
    args: argparse.Namespace,
    *,
    episodes: int,
    seed: int,
) -> float:
    """Evaluate deterministic actions in an isolated visual-only environment."""
    eval_env = _make_visual_env(args, seed)
    successes: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = eval_env.reset(seed=seed + episode)
            succeeded = False
            for frame in range(args.max_episode_steps):
                action = np.asarray(
                    jax.device_get(
                        agent.sample_actions(
                            observations=jax.device_put(observation),
                            argmax=True,
                        )
                    ),
                    dtype=np.float32,
                )
                observation, _, terminated, truncated, info = eval_env.step(action)
                succeeded = succeeded or success_from_info(info)
                if terminated or truncated or succeeded or frame + 1 >= args.max_episode_steps:
                    break
            successes.append(float(succeeded))
    finally:
        eval_env.close()
    return float(np.mean(successes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serl-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--demo-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--encoder-type", choices=["small", "resnet", "resnet-pretrained"], default="resnet-pretrained"
    )
    parser.add_argument("--force-filter", choices=["ema", "none"], default="ema")
    parser.add_argument(
        "--force-fusion", choices=["learned_gate", "concat", "none"], default="learned_gate"
    )
    parser.add_argument("--wrench-filter-alpha", type=float, default=0.2)
    parser.add_argument("--wrench-force-clip", type=float, default=100.0)
    parser.add_argument("--wrench-torque-clip", type=float, default=10.0)
    parser.add_argument("--adaptive-tau", action="store_true")
    parser.add_argument("--critic-loss-threshold", type=float, default=0.05)
    parser.add_argument("--tau-min", type=float, default=0.001)
    parser.add_argument("--tau-max", type=float, default=0.05)
    parser.add_argument("--tau-adjust-factor", type=float, default=1.1)
    parser.add_argument("--tau-adjust-tolerance", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--random-steps", type=int, default=5_000)
    parser.add_argument("--training-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd-ratio", type=int, default=4)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-period", type=int, default=1_000)
    parser.add_argument("--eval-period", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size % 2:
        raise ValueError("--batch-size must be even")
    if args.eval_period < 0:
        raise ValueError("--eval-period must be non-negative")
    if args.eval_period > 0 and args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be positive when evaluation is enabled")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = _make_visual_env(args, args.seed)
    observation, _ = env.reset(seed=args.seed)
    sample_action = np.asarray(env.action_space.sample(), dtype=np.float32)

    original_encoding_wrapper = drq_module.EncodingWrapper
    drq_module.EncodingWrapper = lambda **kwargs: ForceAwareEncodingWrapper(
        **kwargs, force_fusion=args.force_fusion, encoder_type=args.encoder_type
    )
    try:
        agent = drq_module.DrQAgent.create_drq(
            jax.random.PRNGKey(args.seed),
            observation,
            sample_action,
            image_keys=IMAGE_KEYS,
            encoder_type=args.encoder_type,
            use_proprio=True,
            policy_kwargs={
                "tanh_squash_distribution": True,
                "std_parameterization": "exp",
                "std_min": 1e-5,
                "std_max": 5,
            },
            critic_network_kwargs={
                "activations": jax.nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            },
            policy_network_kwargs={
                "activations": jax.nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            },
            temperature_init=1e-2,
            discount=0.99,
            backup_entropy=False,
            critic_ensemble_size=10,
            critic_subsample_size=2,
            adaptive_tau_enabled=args.adaptive_tau,
            critic_loss_threshold=args.critic_loss_threshold,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            tau_adjust_factor=args.tau_adjust_factor,
            tau_adjust_tolerance=args.tau_adjust_tolerance,
        )
    finally:
        drq_module.EncodingWrapper = original_encoding_wrapper
    if args.encoder_type == "resnet-pretrained":
        from serl_launcher.utils.train_utils import load_resnet10_params

        agent = load_resnet10_params(agent, IMAGE_KEYS)

    gate_paths = _gate_parameter_paths(agent)
    if args.force_fusion == "learned_gate" and not gate_paths:
        raise RuntimeError("Learned gate missing from shared DrQ encoder parameter tree")
    initial_gate_values = _gate_parameter_values(agent)

    online_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=args.buffer_capacity,
        image_keys=IMAGE_KEYS,
    )
    demo_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=max(args.buffer_capacity, 10_000),
        image_keys=IMAGE_KEYS,
    )
    demo_episodes, demo_transitions = _insert_demo(args.demo_path, demo_buffer)
    print(f"[demo] episodes={demo_episodes} transitions={demo_transitions} path={args.demo_path}")
    print(f"[model] learned_gate_parameters={gate_paths}")

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "reward_mode": "sparse",
        "policy_schema": {"state": [ROBOT_STATE_DIM], **{key: [128, 128, 3] for key in IMAGE_KEYS}},
        "runtime_image_stack": 1,
        "camera_schema": VISUAL_CAMERA_SCHEMA,
        "policy_reads_object_or_goal_state": False,
        "demo_episodes": demo_episodes,
        "demo_transitions": demo_transitions,
        "gate_parameter_paths": gate_paths,
        "gate_scope": "shared DrQ image/proprio encoder used by actor and critic",
        "jax_devices": [str(device) for device in jax.devices()],
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("metaworld", "gymnasium", "mujoco", "jax")
        },
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    half_batch = args.batch_size // 2
    online_iterator = online_buffer.get_iterator(sample_args={"batch_size": half_batch})
    demo_iterator = demo_buffer.get_iterator(sample_args={"batch_size": half_batch})
    rng = jax.random.PRNGKey(args.seed)
    updates = 0
    episodes = 0
    episode_length = 0
    episode_success = False
    last_info: dict[str, Any] = {}
    batch_shapes: dict[str, Any] | None = None
    evaluation_runs = 0
    last_eval_success_rate = float("nan")
    started = time.time()
    metrics_path = args.output_dir / "metrics.csv"

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "step",
                "episodes",
                "updates",
                "train_success",
                "eval_success_rate",
                "critic_loss",
                "actor_loss",
                "tau",
            ],
        )
        writer.writeheader()
        for step in range(1, args.max_steps + 1):
            if step <= args.random_steps:
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
            else:
                rng, action_key = jax.random.split(rng)
                action = np.asarray(
                    jax.device_get(
                        agent.sample_actions(
                            observations=jax.device_put(observation),
                            seed=action_key,
                            argmax=False,
                        )
                    ),
                    dtype=np.float32,
                )
            next_observation, _, terminated, truncated, info = env.step(action)
            episode_length += 1
            episode_success = episode_success or success_from_info(info)
            done = bool(
                terminated
                or truncated
                or episode_success
                or episode_length >= args.max_episode_steps
            )
            online_buffer.insert(
                {
                    "observations": observation,
                    "next_observations": next_observation,
                    "actions": action,
                    "rewards": np.float32(episode_success),
                    "masks": np.float32(0.0 if done else 1.0),
                    "dones": done,
                }
            )
            observation = next_observation

            if step >= args.training_starts and len(online_buffer) >= half_batch + 1:
                online_batch = next(online_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(online_batch, demo_batch, axis=0)
                if batch_shapes is None:
                    batch_shapes = {
                        key: list(np.asarray(batch["observations"][key]).shape)
                        for key in ("state", *IMAGE_KEYS)
                    }
                    if set(batch["observations"]) != {"state", *IMAGE_KEYS}:
                        raise RuntimeError(f"Bad observation batch keys: {batch['observations'].keys()}")
                    print(f"[batch] observation_shapes={batch_shapes}")
                agent, last_info = agent.update_high_utd(batch, utd_ratio=args.utd_ratio)
                updates += 1

            completed_success: float | None = None
            if done:
                completed_success = float(episode_success)
                episodes += 1
                observation, _ = env.reset(seed=args.seed + episodes)
                episode_length = 0
                episode_success = False

            should_eval = args.eval_period > 0 and step % args.eval_period == 0
            eval_success_rate: float | None = None
            if should_eval:
                eval_success_rate = evaluate_visual_policy(
                    agent,
                    args,
                    episodes=args.eval_episodes,
                    seed=args.seed + 10_000_000 + step * args.eval_episodes,
                )
                evaluation_runs += 1
                last_eval_success_rate = eval_success_rate
                print(
                    f"[eval step {step}] episodes={args.eval_episodes} "
                    f"success_rate={eval_success_rate:.3f}"
                )

            should_log = (
                step == 1
                or step % args.log_period == 0
                or step == args.max_steps
                or completed_success is not None
                or should_eval
            )
            if should_log:
                critic_loss = (
                    float(np.asarray(last_info["critic"]["critic_loss"])) if last_info else float("nan")
                )
                actor_loss = (
                    float(np.asarray(last_info["actor"]["actor_loss"])) if last_info else float("nan")
                )
                tau = float(np.asarray(last_info["curr_tau"])) if "curr_tau" in last_info else float(agent.tau)
                writer.writerow(
                    {
                        "step": step,
                        "episodes": episodes,
                        "updates": updates,
                        "train_success": "" if completed_success is None else completed_success,
                        "eval_success_rate": (
                            "" if eval_success_rate is None else eval_success_rate
                        ),
                        "critic_loss": critic_loss,
                        "actor_loss": actor_loss,
                        "tau": tau,
                    }
                )
                metrics_file.flush()
                print(f"[step {step}] updates={updates} critic_loss={critic_loss:.5f} tau={tau:.6f}")
            if args.save_period > 0 and step % args.save_period == 0:
                print(f"[checkpoint] {_save_agent(agent, args.output_dir, step)}")

    final_checkpoint = _save_agent(agent, args.output_dir, step)
    final_gate_values = _gate_parameter_values(agent)
    gate_max_parameter_change = max(
        (
            float(np.max(np.abs(final_gate_values[path] - initial_gate_values[path])))
            for path in initial_gate_values
        ),
        default=0.0,
    )
    if args.force_fusion == "learned_gate" and gate_max_parameter_change <= 0.0:
        raise RuntimeError("Learned gate parameters did not change during real updates")
    summary = {
        "steps": step,
        "updates": updates,
        "episodes": episodes,
        "online_transitions": online_buffer.total_inserts,
        "evaluation_runs": evaluation_runs,
        "last_eval_success_rate": last_eval_success_rate,
        "demo_episodes": demo_episodes,
        "demo_transitions": demo_transitions,
        "batch_observation_shapes": batch_shapes,
        "gate_parameter_paths": gate_paths,
        "gate_max_parameter_change": gate_max_parameter_change,
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(final_checkpoint),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    env.close()
    if updates < 1:
        raise RuntimeError("Run completed without a real agent update")
    print(f"[done] steps={step} updates={updates} checkpoint={final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
