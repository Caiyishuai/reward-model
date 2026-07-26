"""Potential-field based dense reward computation for robot manipulation tasks.

Computes rewards based on TCP distance to task-specific keypoint targets,
with phase progression and success bonuses.
"""

import numpy as np

from data.common import TaskEntry, get_episodes, get_task, load_pickle


def reward_function(state: np.ndarray, task: TaskEntry, phase_idx: int, ep_idx: int) -> tuple[float, int]:
    """Compute potential-energy reward based on TCP distance to phase target."""
    if task.target_positions is None:
        raise ValueError(f"Task '{task.name}' has no target_positions for manual labeling")

    key = f"k{phase_idx}"
    if key not in task.target_positions:
        raise ValueError(f"No target '{key}' for task '{task.name}' (available: {list(task.target_positions)})")
    target_pos_list = task.target_positions[key]
    if len(target_pos_list) == 1:
        target_pos = target_pos_list[0]
    elif ep_idx < len(target_pos_list):
        target_pos = target_pos_list[ep_idx]
    else:
        target_pos = target_pos_list[ep_idx % len(target_pos_list)]
    target_dim = len(target_pos)

    pos_start = task.state_pos_slice.start or 0
    pos_sl = slice(pos_start, pos_start + target_dim)
    pos = state[pos_sl] if state.ndim == 1 else state[:, pos_sl]

    dist = np.linalg.norm(pos - target_pos)
    max_dist = 0.1
    potential = (phase_idx + 1) * np.exp(-5.0 * (dist / max_dist)) + 2 * (2**phase_idx - 1)

    if dist < 0.0001:
        phase_idx += 1

    return float(potential), phase_idx


def modify_data(data: list, task_name: str, max_episodes: int | None = None) -> list:
    """Assign dense rewards to transitions in-place using potential-field method."""
    flat = []
    for item in data:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    data = flat

    task = get_task(task_name)
    phase_idx, ep_idx = 0, 0
    for x in data:
        curr_state = x["observations"]["state"]
        curr_rew, phase_idx = reward_function(curr_state, task, phase_idx, ep_idx)
        x["rewards"] = curr_rew

        if bool(np.asarray(x["dones"]).item()):
            if x.get("infos", {}).get("succeed", False):
                x["rewards"] += 5
            phase_idx = 0
            ep_idx += 1

        if max_episodes is not None and ep_idx >= max_episodes:
            break

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual reward preprocessing")
    parser.add_argument("--task", type=str, default="button")
    args = parser.parse_args()

    task = get_task(args.task)
    success_data = load_pickle(task.success_path)
    fail_data = load_pickle(task.fail_path)
    if success_data and fail_data:
        modify_data(success_data, args.task)
        modify_data(fail_data, args.task)
        episodes = get_episodes(fail_data)
        for ep in episodes[:1]:
            print([float(f["rewards"]) for f in ep])
