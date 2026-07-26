#!/usr/bin/env python3
"""Evaluate Rsync reward models on held-out ManiSkill episodes.

This benchmark distinguishes four signals that are often conflated:

* ``next.reward``: the G-HMM auto-label potential used as RM supervision.
* ``phi_hat``: the reward model's conservative (minimum ensemble) potential.
* ``next.env_reward``: ManiSkill's privileged normalized dense reward.
* ``pbrs_reward``: ``gamma * phi_hat(s_t) * (1-done_t) - phi_hat(s_{t-1})``.

Metrics are computed only on held-out *episodes*, using the same deterministic
90/10 episode split as reward-model training.  Frame-level random splitting is
not used because adjacent video frames would leak nearly identical samples.

This script measures signal quality; it does not replace the final policy-level
test.  A reward is only validated after an RM-only policy improves environment
success under a fixed interaction budget.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.common import BASE_DIR, IMG_SIZE_RM, get_task  # noqa: E402

if TYPE_CHECKING:
    from reward_model import RewardModel  # noqa: E402


PAPER_TASKS = ("pushcube", "pokecube", "placesphere", "stackcube")


@dataclass
class EpisodeResult:
    split: str
    episode_id: int
    potentials: np.ndarray
    uncertainties: np.ndarray
    labels: np.ndarray
    env_rewards: np.ndarray
    dones: np.ndarray


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[mask], np.asarray(b)[mask]
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[mask], np.asarray(b)[mask]
    if len(a) < 2:
        return float("nan")
    return _pearson(_rankdata(a), _rankdata(b))


def _pairwise_rank_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if len(predictions) < 2:
        return float("nan")
    target_delta = targets[:, None] - targets[None, :]
    pred_delta = predictions[:, None] - predictions[None, :]
    upper = np.triu(np.ones_like(target_delta, dtype=bool), k=1)
    valid = upper & (target_delta != 0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(target_delta[valid]) == np.sign(pred_delta[valid])))


def _binary_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0).sum() + 0.5 * (delta == 0).sum()) / delta.size)


def _held_out_episode_ids(data: dict, split_ratio: float, seed: int) -> set[int]:
    episode_ids = np.unique(np.asarray(data["episode_index"])).astype(int)
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)
    split_index = int(len(episode_ids) * split_ratio)
    return set(episode_ids[split_index:].tolist())


def _episode_frames(data: dict, episode_ids: set[int]) -> list[tuple[int, np.ndarray]]:
    all_ids = np.asarray(data["episode_index"]).astype(int)
    return [(episode_id, np.where(all_ids == episode_id)[0]) for episode_id in sorted(episode_ids)]


def _window_indices(frames: np.ndarray, end_local: int, window: int) -> np.ndarray:
    local = np.arange(end_local - window + 1, end_local + 1)
    return frames[np.clip(local, 0, len(frames) - 1)]


def _build_batch(
    data: dict,
    camera_keys: list[str],
    windows: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, window = windows.shape
    flat = windows.reshape(-1)

    cameras = []
    for key in camera_keys:
        camera_data = data[key]
        image_array = (
            np.asarray(camera_data[flat])
            if isinstance(camera_data, np.ndarray)
            else np.stack([camera_data[int(index)] for index in flat])
        )
        images = torch.from_numpy(image_array).float().div_(255.0).permute(0, 3, 1, 2)
        if images.shape[-2:] != (IMG_SIZE_RM, IMG_SIZE_RM):
            images = F.interpolate(images, size=(IMG_SIZE_RM, IMG_SIZE_RM), mode="bilinear", align_corners=False)
        cameras.append(images.reshape(batch_size, window * 3, IMG_SIZE_RM, IMG_SIZE_RM))

    image_batch = torch.cat(cameras, dim=1).to(device)
    states = np.asarray(data["observation.state"][flat], dtype=np.float32).reshape(batch_size, -1)
    state_batch = torch.from_numpy(states).to(device)
    return image_batch, state_batch


def _evaluate_split(
    model: RewardModel,
    data: dict,
    split_name: str,
    camera_keys: list[str],
    device: torch.device,
    batch_size: int,
    split_ratio: float,
    seed: int,
) -> list[EpisodeResult]:
    held_out = _held_out_episode_ids(data, split_ratio, seed)
    results: list[EpisodeResult] = []

    with torch.no_grad():
        for episode_id, frames in _episode_frames(data, held_out):
            potentials: list[float] = []
            uncertainties: list[float] = []
            for start in range(0, len(frames), batch_size):
                end_locals = np.arange(start, min(start + batch_size, len(frames)))
                windows = np.stack(
                    [_window_indices(frames, int(end), model.state_windows) for end in end_locals]
                )
                images, states = _build_batch(data, camera_keys, windows, device)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    normalized, _ = model(images, states)
                raw = model.unnormalize_reward(normalized)
                potentials.extend(raw.min(dim=1).values.float().cpu().numpy().tolist())
                uncertainties.extend(raw.std(dim=1).float().cpu().numpy().tolist())

            env_rewards = (
                np.asarray(data["next.env_reward"][frames], dtype=np.float32)
                if "next.env_reward" in data
                else np.full(len(frames), np.nan, dtype=np.float32)
            )
            results.append(
                EpisodeResult(
                    split=split_name,
                    episode_id=episode_id,
                    potentials=np.asarray(potentials, dtype=np.float32),
                    uncertainties=np.asarray(uncertainties, dtype=np.float32),
                    labels=np.asarray(data["next.reward"][frames], dtype=np.float32),
                    env_rewards=env_rewards,
                    dones=np.asarray(data["next.done"][frames], dtype=np.float32),
                )
            )
    return results


def _flatten(episodes: list[EpisodeResult], field: str) -> np.ndarray:
    return np.concatenate([getattr(episode, field) for episode in episodes]) if episodes else np.empty(0)


def _pbrs(potential: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    """PBRS for post-transition potentials; the unavailable first transition is NaN."""
    shaped = np.full_like(potential, np.nan, dtype=np.float32)
    if len(potential) > 1:
        shaped[1:] = gamma * potential[1:] * (1.0 - dones[1:]) - potential[:-1]
    return shaped


def _deltas(values: np.ndarray) -> np.ndarray:
    return np.diff(values) if len(values) > 1 else np.empty(0)


def _monotonicity(values: np.ndarray, tolerance: float = -0.01) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.mean(np.diff(values) >= tolerance))


def _task_metrics(success: list[EpisodeResult], fail: list[EpisodeResult], gamma: float) -> dict:
    all_episodes = success + fail
    potentials = _flatten(all_episodes, "potentials")
    labels = _flatten(all_episodes, "labels")
    env_rewards = _flatten(all_episodes, "env_rewards")
    uncertainties = _flatten(all_episodes, "uncertainties")

    absolute_error = np.abs(potentials - labels)
    terminal_success = np.asarray([episode.potentials[-1] for episode in success])
    terminal_fail = np.asarray([episode.potentials[-1] for episode in fail])
    env_returns = np.asarray([np.nansum(episode.env_rewards) for episode in all_episodes])
    pbrs_returns = np.asarray(
        [np.nansum(_pbrs(episode.potentials, episode.dones, gamma)) for episode in all_episodes]
    )

    progress_rm = np.concatenate([_deltas(episode.potentials) for episode in all_episodes])
    progress_env = np.concatenate([_deltas(episode.env_rewards) for episode in all_episodes])
    progress_mask = np.isfinite(progress_rm) & np.isfinite(progress_env)
    nontrivial = progress_mask & (np.abs(progress_env) > 1e-6)

    metrics = {
        "held_out_episodes": {
            "success": len(success),
            "fail": len(fail),
        },
        "supervision_fidelity": {
            "mse": float(np.mean((potentials - labels) ** 2)),
            "mae": float(np.mean(absolute_error)),
            "pearson_phi_vs_label": _pearson(potentials, labels),
            "spearman_phi_vs_label": _spearman(potentials, labels),
            "pairwise_rank_accuracy": _pairwise_rank_accuracy(potentials, labels),
        },
        "environment_alignment": {
            "pearson_phi_vs_dense": _pearson(potentials, env_rewards),
            "spearman_phi_vs_dense": _spearman(potentials, env_rewards),
            "spearman_delta_phi_vs_delta_dense": _spearman(progress_rm, progress_env),
            "progress_sign_agreement": (
                float(np.mean(np.sign(progress_rm[nontrivial]) == np.sign(progress_env[nontrivial])))
                if np.any(nontrivial)
                else float("nan")
            ),
            "spearman_pbrs_return_vs_dense_return": _spearman(pbrs_returns, env_returns),
        },
        "trajectory_discrimination": {
            "terminal_pra": _binary_auc(terminal_success, terminal_fail),
            "success_terminal_mean": float(np.mean(terminal_success)),
            "fail_terminal_mean": float(np.mean(terminal_fail)),
            "success_fail_gap": float(np.mean(terminal_success) - np.mean(terminal_fail)),
            "strict_terminal_gap": float(np.min(terminal_success) - np.max(terminal_fail)),
        },
        "temporal_quality": {
            "success_monotonicity_mean": float(
                np.nanmean([_monotonicity(episode.potentials) for episode in success])
            ),
            "mean_abs_potential_step": float(
                np.mean([np.mean(np.abs(np.diff(episode.potentials))) for episode in all_episodes])
            ),
        },
        "uncertainty": {
            "mean": float(np.mean(uncertainties)),
            "spearman_uncertainty_vs_abs_label_error": _spearman(uncertainties, absolute_error),
        },
    }
    diagnostic = {
        "label_spearman_at_least_0_7": metrics["supervision_fidelity"]["spearman_phi_vs_label"] >= 0.7,
        "terminal_pra_at_least_0_9": metrics["trajectory_discrimination"]["terminal_pra"] >= 0.9,
        "progress_sign_better_than_chance": metrics["environment_alignment"]["progress_sign_agreement"] >= 0.55,
    }
    diagnostic["all_pass"] = all(diagnostic.values())
    metrics["diagnostic_gates"] = diagnostic
    return metrics


def evaluate_task(
    task_name: str,
    checkpoint: Path,
    prefix: str,
    device: torch.device,
    batch_size: int,
    split_ratio: float,
    seed: int,
    gamma: float,
) -> dict:
    from reward_model import RewardModel

    task = get_task(task_name)
    data_dir = Path(BASE_DIR) / task_name / f"{prefix}_processed"
    success_path = data_dir / "success_lerobot.pkl"
    fail_path = data_dir / "fail_lerobot.pkl"
    for path in (checkpoint, success_path, fail_path):
        if not path.exists():
            raise FileNotFoundError(path)

    model = RewardModel.load(str(checkpoint), device=str(device))
    model.eval()
    camera_keys = [f"observation.images.{key}" for key in task.camera_keys]
    if model.num_cameras != len(camera_keys):
        raise ValueError(
            f"{task_name}: checkpoint expects {model.num_cameras} cameras, registry has {camera_keys}"
        )

    with success_path.open("rb") as file:
        success_data = pickle.load(file)
    with fail_path.open("rb") as file:
        fail_data = pickle.load(file)

    success = _evaluate_split(
        model, success_data, "success", camera_keys, device, batch_size, split_ratio, seed
    )
    fail = _evaluate_split(
        model, fail_data, "fail", camera_keys, device, batch_size, split_ratio, seed
    )
    return {
        "task": task_name,
        "checkpoint": str(checkpoint),
        "prefix": prefix,
        "split_ratio": split_ratio,
        "seed": seed,
        "gamma": gamma,
        "metrics": _task_metrics(success, fail, gamma),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Four-task ManiSkill RM quality benchmark")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--checkpoint-template", default="checkpoints/auto_{task}/best.pt")
    parser.add_argument("--prefix", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("eval_results/maniskill_rm_quality.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = list(PAPER_TASKS) if "all" in args.tasks else args.tasks
    unknown = sorted(set(tasks) - set(PAPER_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; expected {PAPER_TASKS}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    started = time.time()
    reports = []
    failures = {}
    for task in tasks:
        checkpoint = Path(args.checkpoint_template.format(task=task))
        try:
            report = evaluate_task(
                task, checkpoint, args.prefix, device, args.batch_size,
                args.split_ratio, args.seed, args.gamma,
            )
            reports.append(report)
            metrics = report["metrics"]
            print(
                f"{task:12s} "
                f"label_rho={metrics['supervision_fidelity']['spearman_phi_vs_label']:.3f} "
                f"env_rho={metrics['environment_alignment']['spearman_phi_vs_dense']:.3f} "
                f"terminal_pra={metrics['trajectory_discrimination']['terminal_pra']:.3f} "
                f"pass={metrics['diagnostic_gates']['all_pass']}"
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            failures[task] = str(error)
            print(f"{task:12s} ERROR: {error}")

    payload = {
        "tasks": reports,
        "failures": failures,
        "duration_s": time.time() - started,
        "interpretation": {
            "diagnostic_only": True,
            "final_acceptance": (
                "Train RM-only policies with a fixed interaction budget and compare environment "
                "success-AUC against sparse and privileged-dense baselines over multiple seeds."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
