"""Offline two-condition ablation for the HMM labeler's final force gate.

This script never trains a reward model or policy. It evaluates real-robot
tasks with force observations under identical leave-one-out folds:

    full:          current reward construction
    no_force_gate: identical construction, except force_mult is fixed to 1.0

Outputs are machine-readable JSON/CSV. Tasks without a configured force slice
are loudly skipped and never included in aggregate metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from data.common import TaskEntry, get_task
from label.auto_label import StageDiscovery, compute_dense_reward, load_episodes
from label.metric import _align_curve, compute_step_auc

CONDITIONS = {
    "full": True,
    "no_force_gate": False,
}
DEFAULT_REAL_FORCE_TASKS = [
    "pickup",
    "plug_insert",
    "iphone_insert",
    "button",
    "usb",
    "key",
    "op_dr",
    "pk_toy",
    "pl_toy",
]
METRIC_KEYS = ("pra", "strict_gap", "continuous_monotonicity", "loo_sauc")


def _new_discovery(cfg: TaskEntry) -> StageDiscovery:
    return StageDiscovery(
        use_force_dynamics=cfg.use_force_dynamics,
        n_restarts=cfg.n_restarts,
        n_iter=cfg.hmm_n_iter,
        use_prototype_stages=cfg.use_prototype_stages,
        pos_slice=cfg.state_pos_slice,
        force_slice=cfg.state_force_slice,
        gripper_idx=cfg.state_gripper_idx,
    )


def _fit_fold(train_success: list[list[dict]], cfg: TaskEntry) -> tuple[StageDiscovery, list[Any]]:
    discovery = _new_discovery(cfg)
    discovery.fit(train_success, n_stages=cfg.n_stages, max_stages=cfg.max_stages)
    targets = discovery.compute_stage_targets(train_success, cfg.contact_force_threshold)
    if not targets:
        raise RuntimeError("stage discovery produced no reward targets")
    return discovery, targets


def _label_held_out(
    episode: list[dict],
    discovery: StageDiscovery,
    targets: list[Any],
    cfg: TaskEntry,
    *,
    force_gate_enabled: bool,
) -> np.ndarray:
    _, state_sequence = discovery.decode(episode)
    return compute_dense_reward(
        episode,
        targets,
        state_sequence,
        force_gate_enabled=force_gate_enabled,
        check_gripper=cfg.check_gripper,
        use_force_dynamics=cfg.use_force_dynamics,
        penalty_power=cfg.penalty_power,
        pos_slice=cfg.state_pos_slice,
        force_slice=cfg.state_force_slice,
        gripper_idx=cfg.state_gripper_idx,
    )


def _metrics(success: list[np.ndarray], failure: list[np.ndarray]) -> dict[str, float]:
    success_finals = np.asarray([curve[-1] for curve in success], dtype=np.float64)
    failure_finals = np.asarray([curve[-1] for curve in failure], dtype=np.float64)
    differences = success_finals[:, None] - failure_finals[None, :]
    pra = float(np.mean(differences > 0))
    strict_gap = float(success_finals.min() - failure_finals.max())

    fractions = [
        float(np.mean(np.diff(curve) >= 0.0))
        for curve in success
        if len(curve) >= 2
    ]
    continuous_monotonicity = float(np.mean(fractions)) if fractions else 0.0

    success_aligned = np.asarray([_align_curve(curve) for curve in success])
    failure_aligned = np.asarray([_align_curve(curve) for curve in failure])
    loo_sauc = compute_step_auc(success_aligned, failure_aligned)
    return {
        "pra": pra,
        "strict_gap": strict_gap,
        "continuous_monotonicity": continuous_monotonicity,
        "loo_sauc": loo_sauc,
    }


def evaluate_task(task_name: str) -> dict[str, dict[str, Any]]:
    cfg = get_task(task_name)
    if cfg.state_force_slice is None:
        raise ValueError(
            f"{task_name}: state_force_slice=None; force-gate ablation is meaningless "
            "for force-free/simulation tasks"
        )

    success = load_episodes(cfg.success_path)
    failure = load_episodes(cfg.fail_path)
    if len(success) < 2 or not failure:
        raise ValueError(
            f"{task_name}: LOO requires at least 2 success and 1 failure episodes; "
            f"found {len(success)} success and {len(failure)} failure"
        )

    curves: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {"success": [], "failure": []} for condition in CONDITIONS
    }

    # Success LOO: each success trajectory is labeled by a model that did not
    # see it. Both conditions reuse the exact same fitted fold.
    for held_out in range(len(success)):
        train_success = [episode for index, episode in enumerate(success) if index != held_out]
        discovery, targets = _fit_fold(train_success, cfg)
        for condition, enabled in CONDITIONS.items():
            curves[condition]["success"].append(
                _label_held_out(
                    success[held_out],
                    discovery,
                    targets,
                    cfg,
                    force_gate_enabled=enabled,
                )
            )

    # Failure demonstrations are never used by StageDiscovery, so one model
    # fitted on all successes is valid for every held-out failure trajectory.
    discovery, targets = _fit_fold(success, cfg)
    for episode in failure:
        for condition, enabled in CONDITIONS.items():
            curves[condition]["failure"].append(
                _label_held_out(
                    episode,
                    discovery,
                    targets,
                    cfg,
                    force_gate_enabled=enabled,
                )
            )

    task_config = {
        "task": task_name,
        "success_path": cfg.success_path,
        "fail_path": cfg.fail_path,
        "n_success": len(success),
        "n_failure": len(failure),
        "protocol": "leave-one-success-out; failures never used for fitting",
        "hmm_restart_seeds": list(range(42, 42 + cfg.n_restarts)),
        "task_config": asdict(cfg),
    }
    return {
        condition: {
            "status": "ok",
            "condition": condition,
            "force_gate_enabled": enabled,
            "force_multiplier": "computed by current static/dynamics gate" if enabled else 1.0,
            **task_config,
            "metrics": _metrics(condition_curves["success"], condition_curves["failure"]),
        }
        for condition, enabled in CONDITIONS.items()
        for condition_curves in [curves[condition]]
    }


def _aggregate(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    valid = [row for row in rows if row["condition"] == condition and row["status"] == "ok"]
    metrics = {
        key: float(np.mean([row["metrics"][key] for row in valid])) if valid else None
        for key in METRIC_KEYS
    }
    return {
        "condition": condition,
        "force_gate_enabled": CONDITIONS[condition],
        "n_tasks": len(valid),
        "metrics_mean_over_tasks": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline Full vs no-force-gate LOO label ablation"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_REAL_FORCE_TASKS,
        help="Registered real-robot tasks with state_force_slice configured",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results/force_gate_ablation"),
        help="Root output directory (full/ and no_force_gate/ are created below it)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for task_name in args.tasks:
        try:
            result = evaluate_task(task_name)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            message = str(error)
            print(f"[SKIP] {task_name}: {message}")
            skipped.append({"task": task_name, "reason": message})
            continue

        for condition, payload in result.items():
            condition_dir = args.output_dir / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            with (condition_dir / f"{task_name}.json").open("w") as handle:
                json.dump(payload, handle, indent=2, default=str)
            rows.append(payload)
        print(f"[OK] {task_name}: completed identical LOO folds for both conditions")

    aggregates = [_aggregate(rows, condition) for condition in CONDITIONS]
    manifest = {
        "experiment": "force_gate_ablation",
        "conditions": {
            "full": "current label construction",
            "no_force_gate": "only final force_mult is fixed to 1.0",
        },
        "tasks_requested": args.tasks,
        "tasks_skipped": skipped,
        "protocol": "offline real-force demonstrations; identical LOO folds and HMM seeds",
        "aggregates": aggregates,
        "task_results": rows,
    }
    with (args.output_dir / "aggregate.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    csv_fields = ["condition", "task", "status", *METRIC_KEYS]
    with (args.output_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "condition": row["condition"],
                    "task": row["task"],
                    "status": row["status"],
                    **row["metrics"],
                }
            )

    if not rows:
        print("[ERROR] No eligible task completed; aggregate files contain no results.")
        return 2
    print(f"[DONE] Wrote JSON/CSV under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
