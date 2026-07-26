#!/usr/bin/env python3
"""Validate four-task raw ManiSkill data before auto-labeling or RM training."""

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


PAPER_TASKS = ("pushcube", "pokecube", "placesphere", "stackcube")


def validate_category(task_name: str, category: str, path: str, camera_keys: list[str], minimum: int) -> dict:
    errors: list[str] = []
    if not Path(path).exists():
        return {"episodes": 0, "transitions": 0, "errors": [f"missing file: {path}"]}

    raw = load_pickle(path)
    if not isinstance(raw, list):
        return {"episodes": 0, "transitions": 0, "errors": [f"{path} is not a transition list"]}
    episodes = get_episodes(raw)
    if len(episodes) < minimum:
        errors.append(f"requires >= {minimum} episodes, found {len(episodes)}")

    episode_seeds = []
    for episode_index, episode in enumerate(episodes):
        if not episode:
            errors.append(f"episode {episode_index} is empty")
            continue
        if not bool(episode[-1].get("dones", False)):
            errors.append(f"episode {episode_index} has no terminal done")
        if any(bool(step.get("dones", False)) for step in episode[:-1]):
            errors.append(f"episode {episode_index} has an early done")

        final_success = bool(episode[-1].get("infos", {}).get("succeed", False))
        expected_success = category == "success"
        if final_success != expected_success:
            errors.append(
                f"episode {episode_index} terminal success={final_success}, expected {expected_success}"
            )
        episode_seeds.append(episode[-1].get("infos", {}).get("episode_seed"))

        for frame_index, step in enumerate(episode):
            prefix = f"episode {episode_index} frame {frame_index}"
            obs = step.get("observations", {})
            state = np.asarray(obs.get("state", []))
            action = np.asarray(step.get("actions", []))
            if state.shape != (17,):
                errors.append(f"{prefix}: state shape {state.shape}, expected (17,)")
            if action.shape != (7,):
                errors.append(f"{prefix}: action shape {action.shape}, expected (7,)")
            if "env_rewards" not in step or not np.isfinite(float(step.get("env_rewards", np.nan))):
                errors.append(f"{prefix}: missing/non-finite env_rewards")
            for camera_key in camera_keys:
                image = np.asarray(obs.get(camera_key, []))
                if image.ndim != 3 or image.shape[-1] != 3:
                    errors.append(f"{prefix}: {camera_key} shape {image.shape}, expected (H,W,3)")
                elif image.dtype != np.uint8:
                    errors.append(f"{prefix}: {camera_key} dtype {image.dtype}, expected uint8")
            if len(errors) >= 100:
                errors.append("stopped after 100 errors")
                break
        if len(errors) >= 100:
            break

    known_seeds = [seed for seed in episode_seeds if seed is not None]
    if known_seeds and len(set(known_seeds)) != len(known_seeds):
        errors.append("duplicate episode_seed values detected")

    return {
        "episodes": len(episodes),
        "transitions": len(raw),
        "state_dim": 17,
        "action_dim": 7,
        "camera_keys": camera_keys,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw four-task RM datasets")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--minimum-episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("eval_results/maniskill_rm_data_validation.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = list(PAPER_TASKS) if "all" in args.tasks else args.tasks
    unknown = sorted(set(tasks) - set(PAPER_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; expected {PAPER_TASKS}")

    report: dict[str, dict] = {}
    has_errors = False
    for task_name in tasks:
        task = get_task(task_name)
        task_report = {
            "success": validate_category(
                task_name, "success", task.success_path, task.camera_keys, args.minimum_episodes
            ),
            "fail": validate_category(
                task_name, "fail", task.fail_path, task.camera_keys, args.minimum_episodes
            ),
        }
        report[task_name] = task_report
        task_errors = sum(len(result["errors"]) for result in task_report.values())
        has_errors = has_errors or task_errors > 0
        print(
            f"{task_name:12s} success={task_report['success']['episodes']:3d} "
            f"fail={task_report['fail']['episodes']:3d} errors={task_errors}"
        )

    payload = {
        "minimum_episodes_per_category": args.minimum_episodes,
        "valid": not has_errors,
        "tasks": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")
    return 2 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
