#!/usr/bin/env python3
"""Export the fitted auto-reward (HMM potential) model per MetaWorld task.

Re-fits ``StageDiscovery`` + per-stage targets on the 20 success demos. For the
MetaWorld tasks this is *deterministic* (prototype-stage fit = progress-binned
means + deterministic DP decode), so the exported model reproduces exactly the
auto rewards baked into ``serl_auto.pkl``.

The output is a **self-contained numpy-only bundle** so the online SERL trainer
(in the ``serl`` conda env) can load and score states without importing
hmmlearn / scikit-learn / the reward-model package.

Run with the ``mwrm`` interpreter (has hmmlearn + sklearn)::

    /opt/miniconda3/envs/mwrm/bin/python scripts/export_auto_reward_model.py \
        --out-dir /apdcephfs_hzlf/share_1227201/yishuaicai/experiments/mw_serl_final/auto_models
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.common import get_task  # noqa: E402
from label.auto_label import StageDiscovery, load_episodes  # noqa: E402

MW_TASKS = [
    "mw_button_press",
    "mw_window_open",
    "mw_reach_wall",
    "mw_plate_slide",
    "mw_push",
    "mw_coffee_push",
    "mw_stick_push",
    "mw_pick_place",
]


def _slice_pair(s):
    if s is None:
        return None
    return [int(s.start), int(s.stop)]


def _target_to_dict(t):
    return {
        "pos_mean": np.asarray(t.pos_mean, dtype=np.float64),
        "pos_cov_inv": np.asarray(t.pos_cov_inv, dtype=np.float64),
        "force_mean": np.asarray(t.force_mean, dtype=np.float64),
        "force_cov_inv": np.asarray(t.force_cov_inv, dtype=np.float64),
        "gripper_mean": float(t.gripper_mean),
        "gripper_std": float(t.gripper_std),
        "is_contact": bool(t.is_contact),
        "max_duration": float(t.max_duration),
        "dforce_mean": None if t.dforce_mean is None else np.asarray(t.dforce_mean, dtype=np.float64),
        "dforce_cov_inv": None if t.dforce_cov_inv is None else np.asarray(t.dforce_cov_inv, dtype=np.float64),
    }


def export_one(task_name: str, out_dir: Path) -> Path:
    cfg = get_task(task_name)
    succ_eps = load_episodes(cfg.success_path)

    disc = StageDiscovery(
        use_force_dynamics=cfg.use_force_dynamics,
        n_restarts=cfg.n_restarts,
        n_iter=cfg.hmm_n_iter,
        use_prototype_stages=cfg.use_prototype_stages,
        pos_slice=cfg.state_pos_slice,
        force_slice=cfg.state_force_slice,
        gripper_idx=cfg.state_gripper_idx,
    )
    disc.fit(succ_eps, n_stages=cfg.n_stages, max_stages=cfg.max_stages)
    targets = disc.compute_stage_targets(succ_eps, cfg.contact_force_threshold)

    if disc.scaler is None or disc.feature_mask is None:
        raise RuntimeError(f"{task_name}: fit did not populate scaler/feature_mask")
    if disc.prototype_centers is None:
        raise RuntimeError(
            f"{task_name}: prototype_centers is None; online causal scorer only "
            "supports the prototype-stage path used by the MetaWorld tasks"
        )

    bundle = {
        "task": task_name,
        "pos_slice": _slice_pair(cfg.state_pos_slice),
        "force_slice": _slice_pair(cfg.state_force_slice),
        "gripper_idx": int(cfg.state_gripper_idx),
        "use_force_dynamics": bool(cfg.use_force_dynamics),
        "use_prototype_stages": bool(cfg.use_prototype_stages),
        "check_gripper": bool(cfg.check_gripper),
        "penalty_power": float(cfg.penalty_power),
        "contact_threshold": float(cfg.contact_force_threshold),
        "n_components": int(disc.n_components),
        "feature_mask": np.asarray(disc.feature_mask, dtype=bool),
        "scaler_mean": np.asarray(disc.scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(disc.scaler.scale_, dtype=np.float64),
        "prototype_centers": np.asarray(disc.prototype_centers, dtype=np.float64),
        "targets": [_target_to_dict(t) for t in targets],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(bundle, f)

    print(
        f"[export] {task_name}: n_components={disc.n_components} "
        f"n_targets={len(targets)} feat={int(disc.feature_mask.sum())}/"
        f"{disc.feature_mask.shape[0]} -> {out_path}"
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Export MetaWorld auto-reward models")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tasks", nargs="*", default=MW_TASKS)
    args = ap.parse_args()
    for task in args.tasks:
        export_one(task, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
