#!/usr/bin/env python3
"""Summarize auto-label quality from exported MetaWorld SERL replay files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.common import BASE_DIR, load_pickle  # noqa: E402
from scripts.export_metaworld_serl_data import METAWORLD_TASKS  # noqa: E402


def _pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0).sum() + 0.5 * (delta == 0).sum()) / delta.size)


def evaluate_task(task: str) -> dict:
    task_dir = Path(BASE_DIR) / task
    metadata = json.loads((task_dir / "collection_meta.json").read_text())
    data = load_pickle(task_dir / "serl_auto.pkl")
    episode_ids = np.asarray(data["episode_index"], dtype=np.int32)
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    unique_ids = np.unique(episode_ids)
    success_count = int(metadata["success_episodes"])
    fail_count = int(metadata["fail_episodes"])
    expected_count = success_count + fail_count
    if len(unique_ids) != expected_count:
        raise ValueError(f"{task}: expected {expected_count} episodes, found {len(unique_ids)}")

    curves = [rewards[episode_ids == episode_id] for episode_id in unique_ids]
    success_curves = curves[:success_count]
    fail_curves = curves[success_count:]
    success_terminal = np.asarray([curve[-1] for curve in success_curves])
    fail_terminal = np.asarray([curve[-1] for curve in fail_curves])
    monotonicity = [
        float(np.mean(np.diff(curve) >= -0.01)) if len(curve) > 1 else float("nan")
        for curve in success_curves
    ]

    return {
        "episodes": {"success": success_count, "fail": fail_count},
        "transitions": int(len(rewards)),
        "terminal_pra": _pairwise_auc(success_terminal, fail_terminal),
        "mean_terminal_gap": float(success_terminal.mean() - fail_terminal.mean()),
        "strict_terminal_gap": float(success_terminal.min() - fail_terminal.max()),
        "success_terminal_mean": float(success_terminal.mean()),
        "fail_terminal_mean": float(fail_terminal.mean()),
        "success_monotonicity": float(np.nanmean(monotonicity)),
        "success_initial_to_terminal_gain": float(
            np.mean([curve[-1] - curve[0] for curve in success_curves])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate eight-task MetaWorld automatic labels")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/metaworld_auto_label_quality.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = list(METAWORLD_TASKS) if "all" in args.tasks else args.tasks
    unknown = sorted(set(tasks) - set(METAWORLD_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; expected {METAWORLD_TASKS}")

    reports = {}
    for task in tasks:
        report = evaluate_task(task)
        reports[task] = report
        print(
            f"{task:20s} PRA={report['terminal_pra']:.3f} "
            f"mean_gap={report['mean_terminal_gap']:.3f} "
            f"strict_gap={report['strict_terminal_gap']:.3f} "
            f"mono={report['success_monotonicity']:.3f}"
        )

    payload = {
        "tasks": reports,
        "warning": (
            "These are in-sample pseudo-label diagnostics. Final reward quality requires held-out "
            "RM evaluation and online policy utility."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
