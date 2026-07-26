"""Dataset statistics and analysis for raw transition pickle files.

Usage:
    python -m data.info --task button --split success
    python -m data.info --task button --split all
    python -m data.info --tasks all --axis z_max
"""

import os

import numpy as np

from data.common import STATE_INDICES, TaskEntry, get_episodes, get_task, list_tasks, load_pickle


def get_axis_pos(
    episodes: list,
    axis: int = 2,
    direction: str = "max",
    pos_slice: slice = STATE_INDICES["pos"],
) -> np.ndarray:
    """Find TCP position at extremal axis value per episode."""
    op_func = np.argmax if direction == "max" else np.argmin
    results = []

    for ep in episodes:
        states = []
        for step in ep:
            if "observations" in step and "state" in step["observations"]:
                s = step["observations"]["state"]
                states.append(s.flatten() if isinstance(s, np.ndarray) else s)
        if not states:
            continue
        states = np.stack(states)
        pos_traj = states[:, pos_slice]
        idx = op_func(pos_traj[:, min(axis, pos_traj.shape[1] - 1)])
        results.append(pos_traj[idx])

    pos_dim = (pos_slice.stop or 0) - (pos_slice.start or 0)
    return np.array(results) if results else np.empty((0, pos_dim))


def analyze_dataset(
    file_path: str,
    axis_dir: str | None = None,
    task: TaskEntry | None = None,
) -> None:
    """Print comprehensive statistics report for a dataset file."""
    print(f"\n{'#' * 70}")
    print(f"ANALYSIS: {os.path.basename(file_path)}")
    print(f"{'#' * 70}")

    data = load_pickle(file_path)
    episodes = get_episodes(data)
    lengths = [len(ep) for ep in episodes]

    if not lengths:
        print("No episodes found.")
        return

    print(f"\n1. Overview: {len(data)} transitions, {len(episodes)} episodes")
    print(f"   Length: min={np.min(lengths)}, max={np.max(lengths)}, mean={np.mean(lengths):.1f}")

    success_count = sum(
        1 for ep in episodes if ep and isinstance(ep[-1].get("infos"), dict) and ep[-1]["infos"].get("succeed", False)
    )
    print(f"   Success: {success_count}/{len(episodes)} ({success_count / len(episodes) * 100:.1f}%)")

    all_states = np.stack(
        [x["observations"]["state"].flatten() for x in data if "observations" in x and "state" in x["observations"]]
    )

    pos_slice = task.state_pos_slice if task else STATE_INDICES["pos"]
    force_slice = task.state_force_slice if task else STATE_INDICES["force"]

    pos = all_states[:, pos_slice]
    pos_labels = list("XYZ")[: pos.shape[1]]
    print("\n2. TCP Position:")
    for i, ax in enumerate(pos_labels):
        print(f"   {ax}: [{np.min(pos[:, i]):.4f}, {np.max(pos[:, i]):.4f}] span={np.ptp(pos[:, i]):.4f}")

    if force_slice is not None:
        force = all_states[:, force_slice]
        for i, ax in enumerate(list("XYZ")[: force.shape[1]]):
            print(f"   Force {ax}: [{np.min(force[:, i]):.4f}, {np.max(force[:, i]):.4f}]")
    else:
        print("   Force: N/A (no force data for this task)")

    state_dim = all_states.shape[1]
    vel_slice = STATE_INDICES.get("vel_lin")
    if vel_slice and vel_slice.stop <= state_dim:
        vel = all_states[:, vel_slice]
        vel_mag = np.linalg.norm(vel, axis=1)
        print(f"   Speed: max={np.max(vel_mag):.4f}, avg={np.mean(vel_mag):.4f}")

    all_rewards = np.array([x["rewards"] for x in data if "rewards" in x])
    if len(all_rewards) > 0:
        print(
            f"\n3. Rewards: mean={np.mean(all_rewards):.4f}, range=[{np.min(all_rewards):.4f}, {np.max(all_rewards):.4f}]"
        )

    if axis_dir:
        axis_map = {"x": 0, "y": 1, "z": 2}
        parts = axis_dir.split("_")
        if len(parts) != 2 or parts[0] not in axis_map or parts[1] not in ("min", "max"):
            print(f"  Invalid --axis format '{axis_dir}'. Expected: x_min, y_max, z_min, etc.")
            return
        axis_name, direction = parts
        points = get_axis_pos(episodes, axis=axis_map[axis_name], direction=direction, pos_slice=pos_slice)
        if len(points) > 0:
            print(f"\n4. {axis_name.upper()}-{direction}: mean={np.mean(points, axis=0)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset statistics analysis")
    parser.add_argument("--tasks", nargs="+", default=["button"], help="Task names or 'all'")
    parser.add_argument(
        "--split", choices=["success", "fail", "all"], default="all", help="Which data split to analyze"
    )
    parser.add_argument("--axis", type=str, default=None, help="Axis analysis, e.g. 'z_max' or 'x_min'")
    args = parser.parse_args()

    tasks = list_tasks() if "all" in args.tasks else args.tasks

    for task_name in tasks:
        task = get_task(task_name)
        paths = []
        if args.split in ("success", "all"):
            paths.append(("success", task.success_path))
        if args.split in ("fail", "all"):
            paths.append(("fail", task.fail_path))

        for label, path in paths:
            if os.path.exists(path):
                analyze_dataset(path, axis_dir=args.axis, task=task)
            else:
                print(f"[SKIP] {task_name}/{label}: {path} not found")
