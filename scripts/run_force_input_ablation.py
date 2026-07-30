"""Train/evaluate reward models with and without the force input channel.

All 20 success and 20 failure trajectories participate.  Within every
trajectory, candidate sample-ending frames are split 80/20 for reward-model
training and evaluation.  The default is a fixed seeded random split, with
temporal and strided alternatives available from the CLI.

The primary ``heldout_frame_pra`` compares every held-out success-frame
prediction with every held-out failure-frame prediction and counts only strict
``success > failure`` comparisons as correct.  This preserves the pairwise
rule of ``label.metric`` PRA, but is deliberately named differently because
the reference metric compares one final frame per trajectory.

The two conditions share labels, frame split, seed, parameter initialization,
and all hyperparameters.  ``no_force`` only zeros normalized force coordinates
before the proprio encoder.  This protocol measures held-out-frame
generalization within seen trajectories, not unseen-trajectory generalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.common import TaskEntry, get_task
from data.dataset import FRAME_SPLIT_STRATEGIES, BalancedLeRobotDataset
from label.auto_label import StageDiscovery, compute_dense_reward, export_lerobot, load_episodes
from reward_model import RewardModel
from train import TrainConfig, train

DEFAULT_TASKS = [
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
DEFAULT_SEEDS = list(range(42, 52))
CONDITIONS = ("full", "no_force")
EXPECTED_PER_CLASS = 20
TRAIN_FRAME_RATIO = 0.8


def validate_episode_count(n_episodes: int) -> None:
    """Require the protocol's 20 trajectories for one outcome class."""
    if n_episodes != EXPECTED_PER_CLASS:
        raise ValueError(
            f"protocol requires exactly {EXPECTED_PER_CLASS} episodes per class; "
            f"found {n_episodes}"
        )


def force_indices(cfg: TaskEntry, state_dim: int) -> list[int]:
    """Expand the configured force slice into validated single-step indices."""
    if cfg.state_force_slice is None:
        raise ValueError(f"{cfg.name}: state_force_slice=None, so no force input can be ablated")
    start, stop, step = cfg.state_force_slice.indices(state_dim)
    indices = list(range(start, stop, step))
    if not indices:
        raise ValueError(f"{cfg.name}: configured force slice selects no state coordinates")
    return indices


def heldout_frame_pra(success_frames: list[float], failure_frames: list[float]) -> float:
    """All-pairs held-out-frame PRA; strict ties follow reference PRA."""
    if not success_frames or not failure_frames:
        return 0.0
    success = np.asarray(success_frames, dtype=np.float64)
    failure = np.asarray(failure_frames, dtype=np.float64)
    return float(np.mean(success[:, None] > failure[None, :]))


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


def _label_episode(
    episode: list[dict],
    discovery: StageDiscovery,
    targets: list[Any],
    cfg: TaskEntry,
) -> None:
    _, states = discovery.decode(episode)
    rewards = compute_dense_reward(
        episode,
        targets,
        states,
        check_gripper=cfg.check_gripper,
        use_force_dynamics=cfg.use_force_dynamics,
        penalty_power=cfg.penalty_power,
        pos_slice=cfg.state_pos_slice,
        force_slice=cfg.state_force_slice,
        gripper_idx=cfg.state_gripper_idx,
    )
    for step, reward in zip(episode, rewards, strict=True):
        step["rewards"] = float(reward)


def prepare_seed_data(
    task_name: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit labels using all success trajectories and export all 40 trajectories."""
    cfg = get_task(task_name)
    success = load_episodes(cfg.success_path)
    failure = load_episodes(cfg.fail_path)
    validate_episode_count(len(success))
    validate_episode_count(len(failure))

    discovery = _new_discovery(cfg)
    discovery.fit(
        success,
        n_stages=cfg.n_stages,
        max_stages=cfg.max_stages,
    )
    targets = discovery.compute_stage_targets(
        success,
        cfg.contact_force_threshold,
    )
    if not targets:
        raise RuntimeError(f"{task_name}: stage discovery produced no reward targets")

    for episode in success:
        _label_episode(episode, discovery, targets, cfg)
    for episode in failure:
        try:
            _label_episode(episode, discovery, targets, cfg)
        except (ValueError, np.linalg.LinAlgError):
            for step in episode:
                step["rewards"] = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "success_lerobot.pkl"
    failure_path = output_dir / "fail_lerobot.pkl"
    export_lerobot(success, success_path, cfg.camera_keys)
    export_lerobot(failure, failure_path, cfg.camera_keys)
    return {
        "success_path": str(success_path),
        "fail_path": str(failure_path),
        "label_fit_success_episode_ids": list(range(EXPECTED_PER_CLASS)),
        "frame_split_seed": seed,
        "n_stages": discovery.n_components,
    }


def predict_heldout_frames(
    model: RewardModel,
    dataset: BalancedLeRobotDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, list[float]]]:
    """Predict every held-out sample-ending frame with both ensemble rules."""
    outputs = {
        "ensemble_min": {"success": [], "fail": []},
        "ensemble_mean": {"success": [], "fail": []},
    }
    positions = list(range(len(dataset)))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            items = [dataset[index] for index in batch_positions]
            frames = torch.stack([item[0] for item in items]).to(device)
            proprio = torch.stack([item[1] for item in items]).to(device)
            normalized, _ = model(frames, proprio)
            pred_min = model.unnormalize_reward(normalized.min(dim=1).values).cpu().tolist()
            pred_mean = model.unnormalize_reward(normalized.mean(dim=1)).cpu().tolist()
            for dataset_position, minimum, mean in zip(
                batch_positions, pred_min, pred_mean, strict=True
            ):
                category = dataset.samples[dataset_position][0]
                outputs["ensemble_min"][category].append(float(minimum))
                outputs["ensemble_mean"][category].append(float(mean))
    return outputs


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate completed runs; ``variance`` is population variance (ddof=0)."""
    rows: list[dict[str, Any]] = []
    tasks = sorted({run["task"] for run in runs if run.get("status") == "ok"})
    for task in tasks:
        for condition in CONDITIONS:
            values = [
                float(run["heldout_frame_pra"])
                for run in runs
                if run.get("status") == "ok"
                and run["task"] == task
                and run["condition"] == condition
            ]
            if not values:
                continue
            rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "n_runs": len(values),
                    "mean_heldout_frame_pra": float(np.mean(values)),
                    "variance": float(np.var(values, ddof=0)),
                    "sample_variance": float(np.var(values, ddof=1)) if len(values) > 1 else None,
                    "heldout_frame_pras": values,
                }
            )
        paired: dict[int, dict[str, float]] = {}
        for run in runs:
            if run.get("status") == "ok" and run["task"] == task:
                paired.setdefault(int(run["seed"]), {})[run["condition"]] = float(
                    run["heldout_frame_pra"]
                )
        deltas = [
            values["full"] - values["no_force"]
            for _, values in sorted(paired.items())
            if all(condition in values for condition in CONDITIONS)
        ]
        if deltas:
            rows.append(
                {
                    "task": task,
                    "condition": "full_minus_no_force",
                    "n_runs": len(deltas),
                    "mean_heldout_frame_pra": float(np.mean(deltas)),
                    "variance": float(np.var(deltas, ddof=0)),
                    "sample_variance": float(np.var(deltas, ddof=1)) if len(deltas) > 1 else None,
                    "heldout_frame_pras": deltas,
                }
            )
    for condition in CONDITIONS:
        per_seed: dict[int, list[float]] = {}
        for run in runs:
            if run.get("status") == "ok" and run["condition"] == condition:
                per_seed.setdefault(int(run["seed"]), []).append(float(run["heldout_frame_pra"]))
        macro = [float(np.mean(values)) for _, values in sorted(per_seed.items())]
        if macro:
            rows.append(
                {
                    "task": "__macro_over_tasks__",
                    "condition": condition,
                    "n_runs": len(macro),
                    "mean_heldout_frame_pra": float(np.mean(macro)),
                    "variance": float(np.var(macro, ddof=0)),
                    "sample_variance": float(np.var(macro, ddof=1)) if len(macro) > 1 else None,
                    "heldout_frame_pras": macro,
                }
            )
    macro_by_seed: dict[int, dict[str, list[float]]] = {}
    for run in runs:
        if run.get("status") == "ok":
            by_condition = macro_by_seed.setdefault(int(run["seed"]), {})
            by_condition.setdefault(run["condition"], []).append(float(run["heldout_frame_pra"]))
    macro_deltas = [
        float(np.mean(values["full"]) - np.mean(values["no_force"]))
        for _, values in sorted(macro_by_seed.items())
        if all(condition in values for condition in CONDITIONS)
    ]
    if macro_deltas:
        rows.append(
            {
                "task": "__macro_over_tasks__",
                "condition": "full_minus_no_force",
                "n_runs": len(macro_deltas),
                "mean_heldout_frame_pra": float(np.mean(macro_deltas)),
                "variance": float(np.var(macro_deltas, ddof=0)),
                "sample_variance": (
                    float(np.var(macro_deltas, ddof=1)) if len(macro_deltas) > 1 else None
                ),
                "heldout_frame_pras": macro_deltas,
            }
        )
    return rows


def _write_results(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    with (output_dir / "runs.csv").open("w", newline="") as handle:
        fields = [
            "task",
            "seed",
            "condition",
            "status",
            "heldout_frame_pra",
            "heldout_frame_pra_ensemble_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in manifest["runs"]:
            writer.writerow({key: run.get(key) for key in fields})
    with (output_dir / "aggregate.csv").open("w", newline="") as handle:
        fields = [
            "task",
            "condition",
            "n_runs",
            "mean_heldout_frame_pra",
            "variance",
            "sample_variance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manifest["aggregates"]:
            writer.writerow({key: row.get(key) for key in fields})


def _inspect_task(task_name: str) -> dict[str, Any]:
    cfg = get_task(task_name)
    success = load_episodes(cfg.success_path)
    failure = load_episodes(cfg.fail_path)
    if not success or not failure:
        raise ValueError(f"{task_name}: empty success or failure data")
    state = np.asarray(success[0][0]["observations"]["state"]).reshape(-1)
    indices = force_indices(cfg, len(state))
    validate_episode_count(len(success))
    validate_episode_count(len(failure))
    return {
        "task": task_name,
        "n_success": len(success),
        "n_failure": len(failure),
        "state_dim": len(state),
        "force_indices": indices,
        "camera_keys": cfg.camera_keys,
        "status": "ready",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Full vs force-input-masked reward-model ablation")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/force_input_ablation"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--split-strategy",
        choices=FRAME_SPLIT_STRATEGIES,
        default="random",
        help=(
            "Per-trajectory 80/20 frame-endpoint split: seeded random (default), "
            "temporal final-20%% holdout, or uniformly distributed strided holdout"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-prepared", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    device_name = (
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    protocol = {
        "primary_metric": "heldout_frame_pra",
        "accuracy_definition": (
            "heldout_frame_pra = mean over every held-out success frame s and held-out "
            "failure frame f of [predicted_reward(s) > predicted_reward(f)]; ties are incorrect"
        ),
        "reference_metric_difference": (
            "label.metric PRA uses one final frame per trajectory; this experiment instead uses "
            "all held-out frame endpoints and therefore does not report the reference final-step PRA"
        ),
        "primary_ensemble_rule": "minimum across ensemble heads (production RewardModel inference)",
        "split_unit": "sample-ending frame within every trajectory",
        "split_strategy": args.split_strategy,
        "train_frame_ratio": TRAIN_FRAME_RATIO,
        "per_class": {"trajectories": 20, "all_trajectories_participate": True},
        "generalization_scope": (
            "held-out-frame generalization within seen trajectories, not unseen-trajectory generalization"
        ),
        "conditions": {
            "full": "all proprioceptive inputs",
            "no_force": (
                "same model/data/labels/seed; configured force coordinates are zeroed after "
                "normalization and before the proprio encoder"
            ),
        },
        "variance": "population variance across repetition accuracies (numpy.var, ddof=0)",
    }
    inspections: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for task_name in args.tasks:
        try:
            inspections.append(_inspect_task(task_name))
        except (FileNotFoundError, KeyError, ValueError) as error:
            skipped.append({"task": task_name, "reason": str(error)})

    if args.dry_run:
        manifest = {
            "experiment": "force_input_ablation",
            "status": "dry_run",
            "protocol": protocol,
            "tasks": inspections,
            "skipped": skipped,
            "seeds": args.seeds,
            "runs": [],
            "aggregates": [],
        }
        _write_results(args.output_dir, manifest)
        print(json.dumps(manifest, indent=2))
        return 0 if inspections else 2

    runs: list[dict[str, Any]] = []
    ready_tasks = {inspection["task"]: inspection for inspection in inspections}
    for task_name in args.tasks:
        if task_name not in ready_tasks:
            continue
        cfg = get_task(task_name)
        for seed in args.seeds:
            prepared_dir = args.output_dir / "prepared" / task_name / f"seed_{seed}"
            try:
                prepared = prepare_seed_data(task_name, seed, prepared_dir)
                state_dim = ready_tasks[task_name]["state_dim"]
                masked_indices = force_indices(cfg, state_dim)
                paired_train_kwargs = {
                    "fail_path": prepared["fail_path"],
                    "success_path": prepared["success_path"],
                    "camera_keys": [f"observation.images.{key}" for key in cfg.camera_keys],
                    "split_ratio": TRAIN_FRAME_RATIO,
                    "frame_split_strategy": args.split_strategy,
                    "samples_per_epoch": args.samples_per_epoch,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "n_epoch_every_eval": args.eval_every,
                    "num_workers": args.num_workers,
                    "seed": seed,
                    "run_post_train_eval": False,
                }
                for condition in CONDITIONS:
                    run_dir = args.output_dir / "checkpoints" / task_name / f"seed_{seed}" / condition
                    train_cfg = TrainConfig(
                        task_name=f"{task_name}_{condition}_seed_{seed}",
                        save_dir=str(run_dir),
                        masked_state_indices=masked_indices if condition == "no_force" else [],
                        **paired_train_kwargs,
                    )
                    train(train_cfg)
                    checkpoint = run_dir / "best.pt"
                    if not checkpoint.exists():
                        raise FileNotFoundError(f"training produced no best checkpoint at {checkpoint}")
                    model = RewardModel.load(str(checkpoint), device=device_name)
                    val_dataset = BalancedLeRobotDataset(
                        fail_path=prepared["fail_path"],
                        success_path=prepared["success_path"],
                        camera_keys=train_cfg.camera_keys,
                        split="val",
                        split_ratio=TRAIN_FRAME_RATIO,
                        epoch_size=1,
                        window_size=model.state_windows,
                        img_size=train_cfg.img_size,
                        seed=seed,
                        max_reward=model.max_reward,
                        min_reward=model.min_reward,
                        frame_split_strategy=args.split_strategy,
                    )
                    heldout_trajectory_counts = {
                        category: len(
                            {
                                int(val_dataset.data_store[category]["raw"]["episode_index"][end_index])
                                for sample_category, end_index in val_dataset.samples
                                if sample_category == category
                            }
                        )
                        for category in ("success", "fail")
                    }
                    if any(
                        count != EXPECTED_PER_CLASS
                        for count in heldout_trajectory_counts.values()
                    ):
                        raise RuntimeError(
                            f"{task_name} seed {seed}: held-out frames do not cover all trajectories: "
                            f"{heldout_trajectory_counts}"
                        )
                    predictions = predict_heldout_frames(model, val_dataset, device, args.batch_size)
                    primary = predictions["ensemble_min"]
                    diagnostic = predictions["ensemble_mean"]
                    if not primary["success"] or not primary["fail"]:
                        raise RuntimeError(
                            f"{task_name} seed {seed}: empty held-out success or failure frame set"
                        )
                    run = {
                        "task": task_name,
                        "seed": seed,
                        "condition": condition,
                        "status": "ok",
                        "heldout_frame_pra": heldout_frame_pra(
                            primary["success"], primary["fail"]
                        ),
                        "heldout_frame_pra_ensemble_mean": heldout_frame_pra(
                            diagnostic["success"], diagnostic["fail"]
                        ),
                        "heldout_frame_predictions": predictions,
                        "heldout_frame_counts": {
                            "success": len(primary["success"]),
                            "fail": len(primary["fail"]),
                        },
                        "heldout_trajectory_counts": heldout_trajectory_counts,
                        "prepared_data": prepared,
                        "frame_split_strategy": args.split_strategy,
                        "frame_train_ratio": TRAIN_FRAME_RATIO,
                        "checkpoint": str(checkpoint),
                        "masked_state_indices": (
                            masked_indices if condition == "no_force" else []
                        ),
                    }
                    runs.append(run)
                    print(
                        f"[OK] {task_name} seed={seed} {condition}: "
                        f"held-out-frame PRA={run['heldout_frame_pra'] * 100:.1f}%"
                    )
            except Exception as error:  # keep the long batch auditable
                failed = {
                    "task": task_name,
                    "seed": seed,
                    "condition": "seed_setup_or_pending_condition",
                    "status": "error",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                runs.append(failed)
                print(f"[ERROR] {task_name} seed={seed}: {error}")
                if args.fail_fast:
                    raise
            finally:
                if not args.keep_prepared:
                    shutil.rmtree(prepared_dir, ignore_errors=True)

    manifest = {
        "experiment": "force_input_ablation",
        "status": "completed",
        "protocol": protocol,
        "tasks": inspections,
        "skipped": skipped,
        "seeds": args.seeds,
        "runs": runs,
        "aggregates": aggregate_runs(runs),
    }
    _write_results(args.output_dir, manifest)
    completed = sum(run.get("status") == "ok" for run in runs)
    expected = len(inspections) * len(args.seeds) * len(CONDITIONS)
    print(f"[DONE] completed {completed}/{expected} condition-runs; results in {args.output_dir}")
    return 0 if completed == expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
