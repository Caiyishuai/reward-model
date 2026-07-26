"""Unified evaluation suite for the Reward Model.

Combines dataset-level metrics, trajectory-level visualization, and
structured JSON report generation. Can be run standalone or called
from train.py after training completes.

Usage:
    python eval_suite.py --checkpoint checkpoints/auto_button/best.pt --task button
    python eval_suite.py --checkpoint checkpoints/auto_button/best.pt --task button --prefix auto
"""

import logging
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec

from data.common import BASE_DIR, get_task, save_json
from eval_traj import TrajectoryEvalDataset, TrajectoryEvaluator
from evaluate import EvaluationDataset, RewardModelEvaluator
from log_utils import setup_logging
from metrics_utils import ranking_accuracy
from reward_model import RewardModel

logger = logging.getLogger(__name__)

sns.set_theme("paper", style="whitegrid")


def run_evaluation(
    checkpoint_path: str,
    task_name: str,
    prefix: str = "auto",
    output_dir: str | None = None,
    device: str = "cuda",
    num_traj_episodes: int = 5,
    batch_size: int = 16,
    seed: int = 42,
) -> dict:
    """Run full evaluation pipeline and generate structured report.

    Returns:
        Dictionary with all computed metrics.
    """
    if output_dir is None:
        output_dir = f"eval_results/{prefix}_{task_name}"
    os.makedirs(output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        device = "cpu"

    task = get_task(task_name)
    camera_keys = [f"observation.images.{k}" for k in task.camera_keys]
    data_dir = BASE_DIR / task_name / f"{prefix}_processed"
    success_path = str(data_dir / "success_lerobot.pkl")
    fail_path = str(data_dir / "fail_lerobot.pkl")

    setup_logging()
    logger.info("Evaluation Suite: %s (%s)", task_name, prefix)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Output: %s", output_dir)

    logger.info("[1/4] Loading model...")
    model = RewardModel.load(checkpoint_path, device=device)
    model.eval()

    report: dict = {
        "task": task_name,
        "prefix": prefix,
        "checkpoint": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
        "device": device,
    }

    # Phase 1: Dataset-level evaluation
    logger.info("[2/4] Dataset-level evaluation...")
    data_paths = {}
    if os.path.exists(success_path):
        data_paths["success"] = success_path
    if os.path.exists(fail_path):
        data_paths["fail"] = fail_path

    if data_paths:
        eval_dataset = EvaluationDataset(
            data_paths=data_paths,
            camera_keys=camera_keys,
            split="val",
            window_size=model.state_windows,
            seed=seed,
        )

        evaluator = RewardModelEvaluator(model, device=device)
        raw_results = evaluator.evaluate(eval_dataset, batch_size=batch_size, save_dir=output_dir)

        preds = np.array(raw_results["preds"])
        labels = np.array(raw_results["labels"])
        types = np.array(raw_results["types"])
        uncertainties = np.array(raw_results["uncertainties"])

        mse = float(np.mean((preds - labels) ** 2))

        rank_acc = ranking_accuracy(preds, labels, seed=seed)

        succ_preds = preds[types == "success"]
        fail_preds = preds[types == "fail"]

        report["dataset_metrics"] = {
            "mse": mse,
            "rank_acc": rank_acc,
            "n_samples": len(preds),
            "mean_uncertainty": float(np.mean(uncertainties)),
            "success_mean": float(np.mean(succ_preds)) if len(succ_preds) > 0 else None,
            "success_std": float(np.std(succ_preds)) if len(succ_preds) > 0 else None,
            "fail_mean": float(np.mean(fail_preds)) if len(fail_preds) > 0 else None,
            "fail_std": float(np.std(fail_preds)) if len(fail_preds) > 0 else None,
            "gap": float(np.mean(succ_preds) - np.mean(fail_preds))
            if len(succ_preds) > 0 and len(fail_preds) > 0
            else None,
        }

        _plot_eval_summary(preds, labels, types, uncertainties, task_name, output_dir)
    else:
        warnings.warn(f"No data files found: {success_path}, {fail_path}", stacklevel=2)
        report["dataset_metrics"] = None

    # Phase 2: Trajectory-level evaluation
    logger.info("[3/4] Trajectory-level evaluation...")
    traj_done = False
    for split_name, path in [("success", success_path), ("fail", fail_path)]:
        if not os.path.exists(path):
            continue
        try:
            traj_dataset = TrajectoryEvalDataset(
                data_path=path,
                window_size=model.state_windows,
                camera_keys=camera_keys,
            )
            traj_evaluator = TrajectoryEvaluator(model, device=device)
            traj_evaluator.evaluate_trajectories(
                traj_dataset,
                num_episodes=num_traj_episodes,
                save_dir=os.path.join(output_dir, f"traj_{split_name}"),
            )
            traj_done = True
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.warning("Trajectory eval failed for %s: %s", split_name, e)

    report["trajectory_eval"] = traj_done

    # Phase 3: Save report
    logger.info("[4/4] Saving report...")
    report_path = os.path.join(output_dir, "eval_report.json")
    save_json(report, report_path)
    logger.info("Report saved: %s", report_path)

    _print_summary(report)
    return report


def _plot_eval_summary(
    preds: np.ndarray,
    labels: np.ndarray,
    types: np.ndarray,
    uncertainties: np.ndarray,
    task_name: str,
    output_dir: str,
) -> None:
    """Generate comprehensive evaluation summary plot."""
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # Pred vs Label scatter
    ax1 = fig.add_subplot(gs[0, 0])
    succ_mask = types == "success"
    fail_mask = types == "fail"
    if np.any(succ_mask):
        ax1.scatter(labels[succ_mask], preds[succ_mask], color="green", alpha=0.3, s=10, label="Success")
    if np.any(fail_mask):
        ax1.scatter(labels[fail_mask], preds[fail_mask], color="red", alpha=0.3, s=10, label="Fail")
    lim = [min(labels.min(), preds.min()) - 0.1, max(labels.max(), preds.max()) + 0.1]
    ax1.plot(lim, lim, "k--", alpha=0.5, linewidth=1)
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Prediction")
    ax1.set_title("Prediction vs Ground Truth")
    ax1.legend()

    # Distribution comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if np.any(succ_mask):
        sns.histplot(preds[succ_mask], ax=ax2, color="green", alpha=0.5, label="Success", kde=True)
    if np.any(fail_mask):
        sns.histplot(preds[fail_mask], ax=ax2, color="red", alpha=0.5, label="Fail", kde=True)
    ax2.set_title("Prediction Distribution")
    ax2.legend()

    # Uncertainty distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(uncertainties, ax=ax3, color="steelblue", kde=True)
    ax3.set_title("Ensemble Uncertainty")
    ax3.set_xlabel("Std across ensemble heads")

    plt.suptitle(f"Evaluation: {task_name}", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(output_dir, "eval_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _print_summary(report: dict) -> None:
    """Log formatted evaluation summary."""
    logger.info("EVALUATION SUMMARY: %s", report["task"])

    dm = report.get("dataset_metrics")
    if dm:
        logger.info("  MSE:          %.4f", dm["mse"])
        logger.info("  Rank Acc:     %.2f%%", dm["rank_acc"] * 100)
        logger.info("  Uncertainty:  %.4f", dm["mean_uncertainty"])
        if dm.get("success_mean") is not None:
            logger.info("  Success:      %.4f +/- %.4f", dm["success_mean"], dm["success_std"])
        if dm.get("fail_mean") is not None:
            logger.info("  Fail:         %.4f +/- %.4f", dm["fail_mean"], dm["fail_std"])
        if dm.get("gap") is not None:
            logger.info("  Gap:          %.4f", dm["gap"])
    else:
        logger.info("  [No dataset metrics]")

    logger.info("  Traj plots:   %s", "Yes" if report.get("trajectory_eval") else "No")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Reward Model evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--task", type=str, required=True, help="Task name (from registry)")
    parser.add_argument("--prefix", type=str, default="auto", help="Data prefix: auto or manual")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory (default: eval_results/<prefix>_<task>)"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_traj", type=int, default=5, help="Number of trajectory episodes to plot")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        task_name=args.task,
        prefix=args.prefix,
        output_dir=args.output,
        device=args.device,
        num_traj_episodes=args.num_traj,
        batch_size=args.batch_size,
    )
