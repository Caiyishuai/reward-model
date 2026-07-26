"""Shared utilities for online (ManiSkill3) RM evaluation.

Both :mod:`sim.rm_eval` and :mod:`sim.traj_eval` run the same underlying
pipeline — spin up an env, load an RM, roll out a policy, compute alignment
metrics. This module owns the code that was previously duplicated across
the two files:

* :class:`EpisodeRecord` — per-episode RM/env reward record
* :func:`rollout_episode` — single-episode rollout with RM scoring
* :func:`compute_pra`, :func:`compute_monotonicity`, :func:`spearman_corr`
* :func:`make_ppo_policy`, :func:`make_random_policy`
* :func:`setup_online_eval` — cfg / device / RM / env / adapter boilerplate
* :func:`compute_step_metrics`, :func:`compute_return_metrics`
* :func:`save_eval_output`

The two entry-point modules keep their CLI and public ``evaluate*()``
function signatures unchanged; they just thin-wrap this module.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from checkpoint_io import load_rl_checkpoint
from reward_model import RewardModel
from sim.agents import PPOAgent
from sim.env_factory import make_env
from sim.obs_adapter import ObsAdapter, extract_full_state
from sim.task_configs import SimTaskConfig, get_sim_config

# ---------------------------------------------------------------------------
# Episode data container
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    """One-episode RM / env reward record (online rollout)."""

    rm_rewards: list[float] = field(default_factory=list)
    env_rewards: list[float] = field(default_factory=list)
    success: bool = False
    length: int = 0

    @property
    def rm_return(self) -> float:
        return sum(self.rm_rewards)

    @property
    def env_return(self) -> float:
        return sum(self.env_rewards)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_pra(succ_finals: list[float], fail_finals: list[float]) -> float:
    """Pairwise ranking accuracy: fraction of (s, f) pairs where s > f."""
    correct = sum(1 for s in succ_finals for f in fail_finals if s > f)
    total = len(succ_finals) * len(fail_finals)
    return correct / total if total > 0 else 0.0


def compute_monotonicity(rewards: list[float], window: int = 5) -> float:
    """Fraction of smoothed reward differences that are non-negative."""
    if len(rewards) < 2:
        return 1.0
    arr = np.array(rewards)
    window = min(window, len(arr))
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    diffs = np.diff(smoothed)
    if diffs.size == 0:
        return 1.0
    return float(np.mean(diffs >= -1e-6))


def spearman_corr(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation between two sequences (0.0 on <3 samples)."""
    from scipy.stats import spearmanr  # type: ignore[import-untyped]

    if len(a) < 3:
        return 0.0
    corr, _ = spearmanr(a, b)
    return float(corr) if not np.isnan(corr) else 0.0


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


PolicyFn = Callable[[Any], np.ndarray]


def rollout_episode(
    env,
    policy_fn: PolicyFn,
    rm: RewardModel,
    adapter: ObsAdapter,
    device: torch.device,
    max_steps: int,
) -> EpisodeRecord:
    """Run one episode, collecting RM and env rewards at each step."""
    del device  # kept in signature for backward-compat; rm already on device
    obs, _ = env.reset()
    adapter.reset(env, obs)
    record = EpisodeRecord()

    for _ in range(max_steps):
        images, proprio = adapter.get_rm_inputs()
        rm_reward = rm.get_reward(images, proprio).item()
        record.rm_rewards.append(rm_reward)

        action = policy_fn(obs)
        obs, env_reward, terminated, truncated, info = env.step(action)
        adapter.step(env, obs)

        r = env_reward.item() if hasattr(env_reward, "item") else float(env_reward)
        record.env_rewards.append(r)

        done = terminated or truncated
        if hasattr(done, "any"):
            done = bool(done.any())
        if done:
            success = info.get("success", False)
            if hasattr(success, "any"):
                success = bool(success.any())
            record.success = success
            break

    record.length = len(record.rm_rewards)
    return record


# ---------------------------------------------------------------------------
# Policy factories
# ---------------------------------------------------------------------------


def make_ppo_policy(ckpt_path: str, env, device: torch.device) -> PolicyFn:
    """Load a PPO checkpoint and return an ``obs -> action`` callable."""
    obs_test, _ = env.reset()
    full_state = extract_full_state(obs_test)
    n_obs = len(full_state)
    n_act = math.prod(env.action_space.shape)

    agent = PPOAgent(n_obs, n_act, device=device)
    state, _ = load_rl_checkpoint(ckpt_path, map_location=device)
    # ``state`` is either the unified ``{"agent": state_dict}`` or a legacy
    # flat state_dict (exposed by the fallback path in ``load_rl_checkpoint``).
    agent_sd = state.get("agent", state) if isinstance(state, dict) else state
    agent.load_state_dict(agent_sd)
    agent.eval()

    def policy_fn(obs: Any) -> np.ndarray:
        state = obs["state"] if isinstance(obs, dict) else obs
        t = (
            state.float().to(device)
            if isinstance(state, torch.Tensor)
            else torch.from_numpy(np.asarray(state)).float().to(device)
        )
        action = agent.act_deterministic(t).cpu().numpy()
        return action.squeeze(0) if action.ndim > 1 else action

    return policy_fn


def make_random_policy(env) -> PolicyFn:
    """Return a random-action callable for the given env."""

    def policy_fn(obs: Any) -> np.ndarray:
        del obs
        return env.action_space.sample()

    return policy_fn


# ---------------------------------------------------------------------------
# Setup boilerplate
# ---------------------------------------------------------------------------


def setup_online_eval(
    task_name: str,
    rm_checkpoint: str,
    device_str: str = "cuda",
) -> tuple[SimTaskConfig, torch.device, RewardModel, Any, ObsAdapter]:
    """Load cfg / device / RM / env / adapter used by both online evaluators."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    cfg = get_sim_config(task_name)
    rm = RewardModel.load(rm_checkpoint, device=str(device))
    env = make_env(cfg, num_envs=1)
    adapter = ObsAdapter(cfg, state_windows=rm.state_windows, device=device)
    return cfg, device, rm, env, adapter


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def compute_step_metrics(records: list[EpisodeRecord]) -> dict[str, Any]:
    """Step-level metrics shared by rm_eval and traj_eval.

    Returned keys:
        n_episodes, n_success, n_fail, success_rate,
        pra_final, gap_final_minmax, gap_final_mean,
        succ_rm_final_mean, fail_rm_final_mean,
        monotonicity, rm_env_step_spearman.
    """
    succ = [r for r in records if r.success]
    fail = [r for r in records if not r.success]

    succ_finals = [r.rm_rewards[-1] for r in succ if r.rm_rewards]
    fail_finals = [r.rm_rewards[-1] for r in fail if r.rm_rewards]

    pra = compute_pra(succ_finals, fail_finals)

    # Keep both gap definitions; ``gap_minmax`` aligns with label/metric.py
    # (pessimistic separation).
    if succ_finals and fail_finals:
        gap_minmax = float(np.min(succ_finals) - np.max(fail_finals))
        gap_mean = float(np.mean(succ_finals) - np.mean(fail_finals))
    else:
        gap_minmax = 0.0
        gap_mean = 0.0

    mono_scores = [compute_monotonicity(r.rm_rewards) for r in succ]
    mean_mono = float(np.mean(mono_scores)) if mono_scores else 0.0

    all_rm: list[float] = []
    all_env: list[float] = []
    for r in records:
        all_rm.extend(r.rm_rewards)
        all_env.extend(r.env_rewards)
    rm_env_corr = spearman_corr(all_rm, all_env)

    return {
        "n_episodes": len(records),
        "n_success": len(succ),
        "n_fail": len(fail),
        "success_rate": len(succ) / max(len(records), 1),
        "pra_final": pra,
        "gap_final_minmax": gap_minmax,
        "gap_final_mean": gap_mean,
        "succ_rm_final_mean": float(np.mean(succ_finals)) if succ_finals else 0.0,
        "fail_rm_final_mean": float(np.mean(fail_finals)) if fail_finals else 0.0,
        "monotonicity": mean_mono,
        "rm_env_step_spearman": rm_env_corr,
    }


def compute_return_metrics(records: list[EpisodeRecord]) -> dict[str, Any]:
    """Return-level (episode-integrated) metrics used by traj_eval."""
    succ = [r for r in records if r.success]
    fail = [r for r in records if not r.success]

    succ_returns = [r.rm_return for r in succ]
    fail_returns = [r.rm_return for r in fail]

    pra_ret = compute_pra(succ_returns, fail_returns)
    if succ_returns and fail_returns:
        gap_ret_minmax = float(np.min(succ_returns) - np.max(fail_returns))
        gap_ret_mean = float(np.mean(succ_returns) - np.mean(fail_returns))
    else:
        gap_ret_minmax = 0.0
        gap_ret_mean = 0.0

    traj_rm_returns = [r.rm_return for r in records]
    traj_env_returns = [r.env_return for r in records]
    return_corr = spearman_corr(traj_rm_returns, traj_env_returns)

    return {
        "pra_return": pra_ret,
        "gap_return_minmax": gap_ret_minmax,
        "gap_return_mean": gap_ret_mean,
        "succ_rm_return_mean": float(np.mean(succ_returns)) if succ_returns else 0.0,
        "fail_rm_return_mean": float(np.mean(fail_returns)) if fail_returns else 0.0,
        "rm_env_return_spearman": return_corr,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _validate_bare_filename(name: str, arg_name: str) -> None:
    """Reject names containing path separators, traversal or anchors.

    Callers may pass filenames built from user/config input. Accepting only
    bare filenames ensures the written file stays under ``output_dir`` and
    prevents traversal (``../etc/passwd.json``) or absolute paths.
    """
    if not name:
        raise ValueError(f"{arg_name} must not be empty")
    if name != Path(name).name:
        raise ValueError(f"{arg_name} must be a bare filename without path components, got {name!r}")


def save_eval_output(
    metrics: dict[str, Any],
    output_dir: str | Path,
    filename: str,
    extra_files: dict[str, Any] | None = None,
) -> Path:
    """Save metrics (and optional extra JSON payloads) under ``output_dir``.

    ``extra_files`` maps filename (under ``output_dir``) to a JSON-serialisable
    object, e.g. ``{"traj_curves_pushcube.json": per_traj}``. Keys must be bare
    filenames — any component that would escape ``output_dir`` is rejected.

    Returns the path of the primary metrics file for convenience.
    """
    _validate_bare_filename(filename, "filename")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    primary = out_path / filename
    with open(primary, "w") as f:
        json.dump(metrics, f, indent=2)

    if extra_files:
        for name, payload in extra_files.items():
            _validate_bare_filename(name, "extra_files key")
            with open(out_path / name, "w") as f:
                json.dump(payload, f)

    return primary


# ---------------------------------------------------------------------------
# Rollout loop with progress log
# ---------------------------------------------------------------------------


def run_episodes(
    env,
    policy_fn: PolicyFn,
    rm: RewardModel,
    adapter: ObsAdapter,
    device: torch.device,
    max_steps: int,
    n_episodes: int,
    *,
    tag: str = "Eval",
    extra_fields: Callable[[EpisodeRecord], str] | None = None,
) -> tuple[list[EpisodeRecord], float]:
    """Roll ``n_episodes`` and return the list of records plus wall time."""
    records: list[EpisodeRecord] = []
    t0 = time.time()
    for ep_idx in range(n_episodes):
        rec = rollout_episode(env, policy_fn, rm, adapter, device, max_steps)
        records.append(rec)
        label = "SUCCESS" if rec.success else "fail"
        rm_final = rec.rm_rewards[-1] if rec.rm_rewards else 0.0
        extra = f"  {extra_fields(rec)}" if extra_fields else ""
        print(
            f"  [{tag}] Ep {ep_idx + 1:3d}/{n_episodes} [{label:7s}] "
            f"len={rec.length:3d}  rm_final={rm_final:.4f}  "
            f"env_total={sum(rec.env_rewards):.3f}{extra}"
        )
    return records, time.time() - t0
