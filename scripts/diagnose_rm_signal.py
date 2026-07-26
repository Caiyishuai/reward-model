#!/usr/bin/env python
"""P0 diagnostic for the AutoRM Reward Model (GOAL.md Phase 4b).

Runs the trained RM on the labelled demonstration pickles (success / fail) and
compares its output against the auto-label supervisor and the per-frame success
flag. Produces scale / smoothness / correlation / class-separation statistics
plus a few representative trajectory plots so we can decide whether the RM
signal itself is the bottleneck for pure-RM-reward SAC.

This script is read-only with respect to the RM and the data pipeline -- it
does NOT modify any training code (satisfies GOAL.md D1-D4).

Typical usage::

    uv run python scripts/diagnose_rm_signal.py

    uv run python scripts/diagnose_rm_signal.py \\
        --rm-checkpoint checkpoints/auto_pushcube/rm_pushcube_epoch_55_val_0.8286.pt \\
        --task pushcube --max-episodes 4 --batch-size 16 --no-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.common import BASE_DIR, IMG_SIZE_RM, get_task, load_pickle  # noqa: E402
from reward_model import RewardModel  # noqa: E402

logger = logging.getLogger("diagnose_rm")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class EpisodeStats:
    split: str
    episode_id: int
    length: int
    rm_rewards: np.ndarray
    rm_uncertainty: np.ndarray
    auto_labels: np.ndarray
    next_success: np.ndarray
    next_done: np.ndarray

    def to_row(self) -> dict[str, float | int | str]:
        """Aggregated per-episode scalar stats."""
        rm = self.rm_rewards
        al = self.auto_labels
        last_success_idx = int(np.argmax(self.next_success)) if self.next_success.any() else -1
        corr = _safe_pearson(rm, al)
        return {
            "split": self.split,
            "episode_id": self.episode_id,
            "length": self.length,
            "rm_mean": float(rm.mean()),
            "rm_std": float(rm.std()),
            "rm_min": float(rm.min()),
            "rm_max": float(rm.max()),
            "rm_terminal": float(rm[-1]),
            "rm_peak_step": int(rm.argmax()),
            "auto_mean": float(al.mean()),
            "auto_max": float(al.max()),
            "auto_terminal": float(al[-1]),
            "unc_mean": float(self.rm_uncertainty.mean()),
            "unc_max": float(self.rm_uncertainty.max()),
            "smoothness_abs_diff": float(np.abs(np.diff(rm)).mean()) if len(rm) > 1 else 0.0,
            "pearson_rm_auto": corr,
            "first_success_step": last_success_idx,
            "any_success_frame": bool(self.next_success.any()),
        }


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Inference core
# ---------------------------------------------------------------------------
def _split_episodes(data: dict) -> list[tuple[int, np.ndarray]]:
    """Return list of (episode_id, absolute_frame_indices) per episode."""
    ep_idx = np.asarray(data["episode_index"])
    unique = np.unique(ep_idx)
    out: list[tuple[int, np.ndarray]] = []
    for ep in unique:
        mask = ep_idx == ep
        frames = np.where(mask)[0]
        out.append((int(ep), frames))
    return out


def _window_indices(frames: np.ndarray, end_i: int, window: int) -> np.ndarray:
    """Return absolute-frame indices for a ``window``-step window ending at
    episode-local index ``end_i``. Clamps at the episode start (replicates frame
    0) so the RM never sees frames from a different episode, mirroring
    ``RMRelabeler._relabel_impl``."""
    local = np.arange(end_i - window + 1, end_i + 1)
    local = np.clip(local, 0, len(frames) - 1)
    return frames[local]


def _load_window_batch(
    data: dict,
    cam_keys: list[str],
    window_indices: np.ndarray,
    img_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (images, proprio) tensors for a batch of windows.

    ``window_indices`` has shape ``[B, T]`` of absolute frame positions.
    """
    B, T = window_indices.shape

    # Proprio: [B, T * robot_dim] -> flatten along time.
    state = data["observation.state"][window_indices.reshape(-1)]
    state = state.reshape(B, T, -1).astype(np.float32).reshape(B, -1)
    proprio_t = torch.from_numpy(state).to(device)

    # Images: per cam, stack [B, T, 3, H, W] -> [B, T*3, H, W] per cam, then
    # concat over cameras -> [B, cam*T*3, H, W].
    flat = window_indices.reshape(-1)
    cam_tensors: list[torch.Tensor] = []
    for cam_key in cam_keys:
        camera_data = data[cam_key]
        imgs_np = (
            camera_data[flat]
            if isinstance(camera_data, np.ndarray)
            else np.stack([camera_data[int(index)] for index in flat])
        )  # [B*T, H, W, 3] uint8
        imgs = torch.from_numpy(imgs_np).float().div_(255.0).permute(0, 3, 1, 2)
        if imgs.shape[-1] != img_size or imgs.shape[-2] != img_size:
            imgs = F.interpolate(imgs, size=(img_size, img_size), mode="bilinear", align_corners=False)
        imgs = imgs.reshape(B, T * 3, img_size, img_size)
        cam_tensors.append(imgs)
    images_t = torch.cat(cam_tensors, dim=1).to(device)

    return images_t, proprio_t


def run_rm_on_split(
    rm: RewardModel,
    data: dict,
    cam_keys: list[str],
    split: str,
    device: torch.device,
    batch_size: int,
    max_episodes: int | None,
) -> list[EpisodeStats]:
    """Run the RM on every frame of every episode in ``data`` and return
    per-episode results."""
    window = rm.state_windows
    episodes = _split_episodes(data)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    results: list[EpisodeStats] = []
    with torch.no_grad():
        for ep_id, frames in episodes:
            if len(frames) == 0:
                continue

            all_rm: list[float] = []
            all_unc: list[float] = []

            # Walk the episode in micro-batches.
            for batch_start in range(0, len(frames), batch_size):
                batch_end = min(batch_start + batch_size, len(frames))
                end_locals = np.arange(batch_start, batch_end)
                win_idx = np.stack(
                    [_window_indices(frames, int(e), window) for e in end_locals], axis=0
                )
                images_t, proprio_t = _load_window_batch(
                    data, cam_keys, win_idx, IMG_SIZE_RM, device
                )

                # We need uncertainty, so avoid the convenience get_reward() path
                # which only returns the conservative min. Go through forward().
                with torch.amp.autocast(device.type):
                    rewards_norm, _ = rm.forward(images_t, proprio_t)
                rewards_raw = rm.unnormalize_reward(rewards_norm)  # [B, ensemble]
                conservative = rewards_raw.min(dim=1, keepdim=True).values
                uncertainty = rewards_raw.std(dim=1, keepdim=True)

                all_rm.extend(conservative.squeeze(-1).float().cpu().numpy().tolist())
                all_unc.extend(uncertainty.squeeze(-1).float().cpu().numpy().tolist())

            auto_labels = np.asarray(data["next.reward"][frames], dtype=np.float32)
            next_success = np.asarray(data["next.success"][frames], dtype=bool)
            next_done = np.asarray(data["next.done"][frames], dtype=bool)

            results.append(
                EpisodeStats(
                    split=split,
                    episode_id=ep_id,
                    length=len(frames),
                    rm_rewards=np.asarray(all_rm, dtype=np.float32),
                    rm_uncertainty=np.asarray(all_unc, dtype=np.float32),
                    auto_labels=auto_labels,
                    next_success=next_success,
                    next_done=next_done,
                )
            )
            logger.info(
                "  [%s] ep=%d len=%d rm mean=%.3f min=%.3f max=%.3f unc mean=%.3f",
                split,
                ep_id,
                len(frames),
                float(results[-1].rm_rewards.mean()),
                float(results[-1].rm_rewards.min()),
                float(results[-1].rm_rewards.max()),
                float(results[-1].rm_uncertainty.mean()),
            )
    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def _flatten(eps: list[EpisodeStats], field: str) -> np.ndarray:
    return np.concatenate([getattr(e, field) for e in eps]) if eps else np.empty(0)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled < 1e-8:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def _pairwise_terminal_rank(success: list[EpisodeStats], fail: list[EpisodeStats]) -> float:
    """Fraction of (success_ep, fail_ep) pairs where RM at the terminal frame of
    the success ep exceeds RM at the terminal frame of the fail ep. Mirrors
    ``val_rank_acc`` but at trajectory level using the RM's own output."""
    if not success or not fail:
        return float("nan")
    s_term = np.array([float(e.rm_rewards[-1]) for e in success])
    f_term = np.array([float(e.rm_rewards[-1]) for e in fail])
    wins = (s_term[:, None] > f_term[None, :]).sum()
    total = s_term.size * f_term.size
    return float(wins / total)


def aggregate(
    success: list[EpisodeStats], fail: list[EpisodeStats]
) -> dict[str, object]:
    rm_s = _flatten(success, "rm_rewards")
    rm_f = _flatten(fail, "rm_rewards")
    auto_s = _flatten(success, "auto_labels")
    auto_f = _flatten(fail, "auto_labels")
    unc_s = _flatten(success, "rm_uncertainty")
    unc_f = _flatten(fail, "rm_uncertainty")

    def _per_split(name: str, rm: np.ndarray, auto: np.ndarray, unc: np.ndarray, eps: list[EpisodeStats]) -> dict[str, float | int]:
        if rm.size == 0:
            return {"name": name, "n_frames": 0}
        corr_per_ep = [e.to_row()["pearson_rm_auto"] for e in eps]
        corr_per_ep = [c for c in corr_per_ep if isinstance(c, float) and not math.isnan(c)]
        return {
            "name": name,
            "n_episodes": len(eps),
            "n_frames": int(rm.size),
            "rm_mean": float(rm.mean()),
            "rm_std": float(rm.std()),
            "rm_min": float(rm.min()),
            "rm_max": float(rm.max()),
            "auto_mean": float(auto.mean()),
            "auto_std": float(auto.std()),
            "auto_min": float(auto.min()),
            "auto_max": float(auto.max()),
            "unc_mean": float(unc.mean()),
            "unc_std": float(unc.std()),
            "smoothness_abs_diff_mean": float(
                np.mean([np.abs(np.diff(e.rm_rewards)).mean() for e in eps if e.length > 1])
            ),
            "pearson_rm_auto_global": _safe_pearson(rm, auto),
            "pearson_rm_auto_per_episode_mean": float(np.mean(corr_per_ep)) if corr_per_ep else float("nan"),
            "pearson_rm_auto_per_episode_std": float(np.std(corr_per_ep)) if corr_per_ep else float("nan"),
        }

    rm_range = float(max(rm_s.max() if rm_s.size else 0.0, rm_f.max() if rm_f.size else 0.0)
                     - min(rm_s.min() if rm_s.size else 0.0, rm_f.min() if rm_f.size else 0.0))
    agg = {
        "success": _per_split("success", rm_s, auto_s, unc_s, success),
        "fail": _per_split("fail", rm_f, auto_f, unc_f, fail),
        "separation": {
            "rm_success_mean_minus_fail_mean": float(rm_s.mean() - rm_f.mean()) if rm_s.size and rm_f.size else float("nan"),
            "cohens_d": _cohens_d(rm_s, rm_f),
            "pairwise_terminal_rank_acc": _pairwise_terminal_rank(success, fail),
            "rm_full_range": rm_range,
        },
    }
    return agg


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_per_frame_csv(path: Path, all_eps: list[EpisodeStats]) -> None:
    fieldnames = [
        "split", "episode_id", "step", "rm_reward", "rm_uncertainty",
        "auto_label", "next_success", "next_done",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for ep in all_eps:
            for t in range(ep.length):
                writer.writerow({
                    "split": ep.split,
                    "episode_id": ep.episode_id,
                    "step": t,
                    "rm_reward": float(ep.rm_rewards[t]),
                    "rm_uncertainty": float(ep.rm_uncertainty[t]),
                    "auto_label": float(ep.auto_labels[t]),
                    "next_success": int(bool(ep.next_success[t])),
                    "next_done": int(bool(ep.next_done[t])),
                })


def write_per_episode_csv(path: Path, all_eps: list[EpisodeStats]) -> None:
    if not all_eps:
        return
    rows = [ep.to_row() for ep in all_eps]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report_json(path: Path, report: dict[str, object]) -> None:
    with path.open("w") as fp:
        json.dump(report, fp, indent=2, default=float)


def write_summary_md(path: Path, report: dict[str, object]) -> None:
    agg = report["aggregate"]
    meta = report["meta"]
    sep = agg["separation"]

    def _fmt(v: object, digits: int = 4) -> str:
        if isinstance(v, (int, bool)):
            return str(v)
        if isinstance(v, float):
            return "NaN" if math.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    lines: list[str] = []
    lines.append("# RM Signal Diagnostic\n")
    lines.append(f"- Checkpoint: `{meta['checkpoint']}`")
    lines.append(f"- Task: `{meta['task']}`")
    lines.append(f"- Success episodes used: {meta['n_success_episodes']}")
    lines.append(f"- Fail episodes used:    {meta['n_fail_episodes']}")
    lines.append(f"- Device: `{meta['device']}`")
    lines.append(f"- Wall clock: {meta['duration_s']:.1f}s")
    lines.append(f"- RM reward training range (from checkpoint): [{meta['reward_min_val']:.4f}, {meta['reward_max_val']:.4f}]\n")

    lines.append("## Class separation (the punchline)\n")
    lines.append(f"- `rm_success_mean - rm_fail_mean` = **{_fmt(sep['rm_success_mean_minus_fail_mean'])}**")
    lines.append(f"- Cohen's d = **{_fmt(sep['cohens_d'])}** (|d|>0.8 = large, >0.5 = medium)")
    lines.append(f"- Pairwise terminal-rank acc = **{_fmt(sep['pairwise_terminal_rank_acc'])}** (target: close to val_rank_acc ~ 0.83)")
    lines.append(f"- Observed RM full range across both splits = {_fmt(sep['rm_full_range'])}\n")

    for split_key in ("success", "fail"):
        s = agg[split_key]
        lines.append(f"## {split_key.capitalize()} split (n_frames={s.get('n_frames', 0)})\n")
        if s.get("n_frames", 0) == 0:
            lines.append("_No frames processed._\n")
            continue
        lines.append(f"- RM: mean={_fmt(s['rm_mean'])} std={_fmt(s['rm_std'])} "
                     f"min={_fmt(s['rm_min'])} max={_fmt(s['rm_max'])}")
        lines.append(f"- Auto-label: mean={_fmt(s['auto_mean'])} std={_fmt(s['auto_std'])} "
                     f"min={_fmt(s['auto_min'])} max={_fmt(s['auto_max'])}")
        lines.append(f"- Uncertainty: mean={_fmt(s['unc_mean'])} std={_fmt(s['unc_std'])}")
        lines.append(f"- Smoothness (mean |Δrm|): {_fmt(s['smoothness_abs_diff_mean'])}")
        lines.append(f"- Pearson(rm, auto) global: {_fmt(s['pearson_rm_auto_global'])}")
        lines.append(f"- Pearson(rm, auto) per-episode: {_fmt(s['pearson_rm_auto_per_episode_mean'])} "
                     f"± {_fmt(s['pearson_rm_auto_per_episode_std'])}")
        lines.append("")

    lines.append("## Reading the numbers\n")
    lines.append("- **If `pairwise_terminal_rank_acc` is high (≥0.8) but `rm_mean` difference is tiny or "
                 "`Cohen's d` small**, the RM ranks classes correctly but gives a flat, low-contrast "
                 "signal -- SAC's critic has little gradient to learn from. Likely fix: bigger reward "
                 "range (retrain with wider `max_reward - min_reward`) and/or temporal smoothness loss.")
    lines.append("- **If `smoothness_abs_diff_mean` is a large fraction of `rm_full_range`**, the RM is "
                 "jittery step-to-step, which destabilises Q-learning. Likely fix: temporal-smoothness "
                 "regulariser during RM training or potential-based shaping form.")
    lines.append("- **If `pearson_rm_auto_global` is low**, the RM is diverging from its supervisor -- the "
                 "bottleneck is RM fitting, not the RL loop.")
    lines.append("- **If `rm_success_mean_minus_fail_mean` is negative or near zero**, the RM is not "
                 "separating success from fail at all at the frame level -- ranking-only training is "
                 "insufficient and preference-based or classification-aux losses should be tried.\n")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Plots (optional)
# ---------------------------------------------------------------------------
def try_plots(out_dir: Path, success: list[EpisodeStats], fail: list[EpisodeStats]) -> str | None:
    try:
        import matplotlib
    except ImportError:
        return "matplotlib not available; skipped plots"
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-episode trajectories (first 4 per split)
    for split_name, eps in (("success", success[:4]), ("fail", fail[:4])):
        if not eps:
            continue
        fig, axes = plt.subplots(len(eps), 1, figsize=(8, 2.0 * len(eps)), sharex=False)
        if len(eps) == 1:
            axes = [axes]
        for ax, ep in zip(axes, eps, strict=True):
            t = np.arange(ep.length)
            ax.plot(t, ep.rm_rewards, label="RM (conservative)", color="tab:blue")
            ax.fill_between(
                t,
                ep.rm_rewards - ep.rm_uncertainty,
                ep.rm_rewards + ep.rm_uncertainty,
                alpha=0.2,
                color="tab:blue",
                label="RM ± ensemble std",
            )
            ax.plot(t, ep.auto_labels, label="auto-label", color="tab:orange", linestyle="--")
            if ep.next_success.any():
                first = int(np.argmax(ep.next_success))
                ax.axvline(first, color="tab:green", linestyle=":", label="next.success=1")
            ax.set_title(f"{split_name} ep={ep.episode_id}")
            ax.legend(loc="best", fontsize=7)
            ax.set_xlabel("step")
            ax.set_ylabel("reward")
        fig.tight_layout()
        fig.savefig(plots_dir / f"curves_{split_name}.png", dpi=120)
        plt.close(fig)

    # 2. Global distribution histogram
    rm_s = _flatten(success, "rm_rewards")
    rm_f = _flatten(fail, "rm_rewards")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    if rm_s.size:
        axes[0].hist(rm_s, bins=30, alpha=0.6, label="success", color="tab:green")
    if rm_f.size:
        axes[0].hist(rm_f, bins=30, alpha=0.6, label="fail", color="tab:red")
    axes[0].set_title("RM reward distribution")
    axes[0].set_xlabel("rm_reward")
    axes[0].legend()

    auto_s = _flatten(success, "auto_labels")
    auto_f = _flatten(fail, "auto_labels")
    if auto_s.size:
        axes[1].scatter(auto_s, rm_s, s=3, alpha=0.3, color="tab:green", label="success")
    if auto_f.size:
        axes[1].scatter(auto_f, rm_f, s=3, alpha=0.3, color="tab:red", label="fail")
    axes[1].set_title("RM vs auto-label (per frame)")
    axes[1].set_xlabel("auto_label")
    axes[1].set_ylabel("rm_reward")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "distribution.png", dpi=120)
    plt.close(fig)
    return str(plots_dir)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 diagnostic of AutoRM reward signal.")
    parser.add_argument(
        "--rm-checkpoint",
        default="checkpoints/auto_pushcube/rm_pushcube_epoch_55_val_0.8286.pt",
        help="Path to RM checkpoint (.pt).",
    )
    parser.add_argument("--task", default="pushcube")
    parser.add_argument(
        "--data-prefix",
        default="auto",
        help="Subdirectory prefix: '{prefix}_processed/{success,fail}_lerobot.pkl'. "
        "Default 'auto' matches the data the RM was trained on.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: experiments/diagnose_rm/<timestamp>.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Optional cap on episodes per split (for smoke runs).")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but torch.cuda.is_available()==False. "
                     "Either install CUDA-enabled torch or pass --device cpu.")
        return 2

    out_dir = Path(args.output_dir) if args.output_dir else (
        _REPO_ROOT / "experiments" / "diagnose_rm" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing diagnostics to %s", out_dir)

    # Task + data paths (mirror train.py conventions).
    task = get_task(args.task)
    cam_keys = [f"observation.images.{k}" for k in task.camera_keys]
    data_dir = Path(BASE_DIR) / args.task / f"{args.data_prefix}_processed"
    success_path = data_dir / "success_lerobot.pkl"
    fail_path = data_dir / "fail_lerobot.pkl"
    for p in (success_path, fail_path):
        if not p.exists():
            logger.error("Missing data file: %s", p)
            return 3

    logger.info("Loading RM from %s", args.rm_checkpoint)
    device = torch.device(args.device)
    rm = RewardModel.load(args.rm_checkpoint, device=str(device))
    logger.info(
        "  num_cameras=%d state_windows=%d robot_dim=%d ensemble_size=%d reward_range=[%.4f, %.4f]",
        rm.num_cameras,
        rm.state_windows,
        rm.robot_dim,
        rm.ensemble_size,
        rm.reward_normalizer.min_val.item(),
        rm.reward_normalizer.max_val.item(),
    )
    if rm.num_cameras != len(cam_keys):
        logger.error(
            "Camera count mismatch: RM expects %d but task '%s' has %d keys (%s).",
            rm.num_cameras, args.task, len(cam_keys), cam_keys,
        )
        return 4

    t0 = time.time()
    logger.info("Loading success data: %s", success_path)
    success_data = load_pickle(success_path)
    logger.info("Loading fail data: %s", fail_path)
    fail_data = load_pickle(fail_path)

    logger.info("Running RM on success split...")
    success_eps = run_rm_on_split(
        rm, success_data, cam_keys, "success", device, args.batch_size, args.max_episodes,
    )
    logger.info("Running RM on fail split...")
    fail_eps = run_rm_on_split(
        rm, fail_data, cam_keys, "fail", device, args.batch_size, args.max_episodes,
    )
    duration = time.time() - t0

    aggregate_stats = aggregate(success_eps, fail_eps)
    meta = {
        "checkpoint": str(args.rm_checkpoint),
        "task": args.task,
        "data_prefix": args.data_prefix,
        "device": str(device),
        "n_success_episodes": len(success_eps),
        "n_fail_episodes": len(fail_eps),
        "reward_min_val": float(rm.reward_normalizer.min_val.item()),
        "reward_max_val": float(rm.reward_normalizer.max_val.item()),
        "duration_s": duration,
    }
    report = {"meta": meta, "aggregate": aggregate_stats}

    write_report_json(out_dir / "report.json", report)
    write_per_frame_csv(out_dir / "per_frame.csv", success_eps + fail_eps)
    write_per_episode_csv(out_dir / "per_episode.csv", success_eps + fail_eps)
    write_summary_md(out_dir / "SUMMARY.md", report)

    if not args.no_plots:
        note = try_plots(out_dir, success_eps, fail_eps)
        if note:
            logger.info(note)

    logger.info("=" * 70)
    logger.info("rm_success_mean - rm_fail_mean = %.4f",
                aggregate_stats["separation"]["rm_success_mean_minus_fail_mean"])
    logger.info("Cohen's d                      = %.4f",
                aggregate_stats["separation"]["cohens_d"])
    logger.info("Pairwise terminal-rank acc     = %.4f",
                aggregate_stats["separation"]["pairwise_terminal_rank_acc"])
    logger.info("Duration                       = %.1fs", duration)
    logger.info("=" * 70)
    logger.info("Report: %s", out_dir / "SUMMARY.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
