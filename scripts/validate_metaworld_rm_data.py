#!/usr/bin/env python3
"""Validate raw 20+20 MetaWorld RGB/state datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.common import get_episodes, get_task, load_pickle  # noqa: E402
from scripts.export_metaworld_serl_data import METAWORLD_TASKS  # noqa: E402


def validate_file(path: str, category: str, minimum: int) -> dict:
    errors: list[str] = []
    file_path = Path(path)
    if not file_path.exists():
        return {"episodes": 0, "transitions": 0, "errors": [f"missing: {path}"]}
    raw = load_pickle(path)
    if not isinstance(raw, list):
        return {"episodes": 0, "transitions": 0, "errors": ["raw pickle is not a list"]}
    episodes = get_episodes(raw)
    if len(episodes) < minimum:
        errors.append(f"requires >= {minimum} episodes, found {len(episodes)}")

    seeds = []
    for episode_index, episode in enumerate(episodes):
        if not episode:
            errors.append(f"episode {episode_index}: empty")
            continue
        expected_success = category == "success"
        actual_success = bool(episode[-1].get("infos", {}).get("succeed", False))
        if actual_success != expected_success:
            errors.append(f"episode {episode_index}: succeed={actual_success}, expected={expected_success}")
        if not bool(episode[-1].get("dones", False)):
            errors.append(f"episode {episode_index}: final done is false")
        seeds.append(episode[-1].get("infos", {}).get("episode_seed"))

        for frame, step in enumerate(episode):
            prefix = f"episode {episode_index} frame {frame}"
            observation = step.get("observations", {})
            previous = step.get("previous_observations", {})
            if np.asarray(observation.get("state", [])).shape != (39,):
                errors.append(f"{prefix}: next state is not (39,)")
            if np.asarray(previous.get("state", [])).shape != (39,):
                errors.append(f"{prefix}: previous state is not (39,)")
            wrist_wrench = np.asarray(observation.get("wrist_wrench", []), dtype=np.float32)
            previous_wrench = np.asarray(previous.get("wrist_wrench", []), dtype=np.float32)
            if wrist_wrench.shape != (6,) or not np.isfinite(wrist_wrench).all():
                errors.append(f"{prefix}: wrist_wrench must be finite (6,)")
            if previous_wrench.shape != (6,) or not np.isfinite(previous_wrench).all():
                errors.append(f"{prefix}: previous wrist_wrench must be finite (6,)")
            if not np.array_equal(wrist_wrench, np.asarray(step.get("wrist_wrench", []))):
                errors.append(f"{prefix}: top-level and observation wrist_wrench differ")
            contact_force = np.asarray(step.get("contact_force", []), dtype=np.float32)
            if contact_force.ndim != 2 or contact_force.shape[1:] != (3,) or not np.isfinite(contact_force).all():
                errors.append(f"{prefix}: contact_force must be finite (N,3)")
            max_contact_force = float(step.get("max_contact_force", np.nan))
            if not np.isfinite(max_contact_force) or max_contact_force < 0.0:
                errors.append(f"{prefix}: max_contact_force must be finite and non-negative")
            if np.asarray(step.get("actions", [])).shape != (4,):
                errors.append(f"{prefix}: action is not (4,)")
            image = np.asarray(observation.get("corner2", []))
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                errors.append(f"{prefix}: corner2 must be uint8 (H,W,3), got {image.shape}/{image.dtype}")
            for key in ("env_rewards", "sparse_rewards"):
                if key not in step or not np.isfinite(float(step.get(key, np.nan))):
                    errors.append(f"{prefix}: invalid {key}")
            if len(errors) >= 100:
                errors.append("stopped after 100 errors")
                break
        if len(errors) >= 100:
            break
    known_seeds = [seed for seed in seeds if seed is not None]
    if len(known_seeds) != len(set(known_seeds)):
        errors.append("duplicate episode seeds")
    return {"episodes": len(episodes), "transitions": len(raw), "errors": errors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate eight MetaWorld RM datasets")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--minimum-episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("eval_results/metaworld_data_validation.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = list(METAWORLD_TASKS) if "all" in args.tasks else args.tasks
    report = {}
    valid = True
    for task_name in tasks:
        task = get_task(task_name)
        result = {
            "success": validate_file(task.success_path, "success", args.minimum_episodes),
            "fail": validate_file(task.fail_path, "fail", args.minimum_episodes),
        }
        errors = sum(len(category["errors"]) for category in result.values())
        valid = valid and errors == 0
        report[task_name] = result
        print(
            f"{task_name:20s} success={result['success']['episodes']:3d} "
            f"fail={result['fail']['episodes']:3d} errors={errors}"
        )
    payload = {"valid": valid, "minimum_episodes": args.minimum_episodes, "tasks": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
