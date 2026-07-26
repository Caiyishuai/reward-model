"""Multi-panel dataset probing visualization.

Usage:
    python -m data.probe_dataset --tasks button pickup --output analysis/
    python -m data.probe_dataset --tasks all --split success
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from data.common import STATE_INDICES, TaskEntry, get_episodes, get_task, list_tasks, load_pickle

PENALTY_STEP = 0.05


def visualize_dataset_probing(
    data_path: str,
    output_dir: str = "analysis_plots",
    task: TaskEntry | None = None,
) -> None:
    """Generate comprehensive multi-panel analysis of a raw dataset."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = os.path.basename(data_path).replace(".pkl", "")

    pos_slice = task.state_pos_slice if task else STATE_INDICES["pos"]
    force_slice = task.state_force_slice if task else STATE_INDICES["force"]

    data = load_pickle(data_path)
    episodes = get_episodes(data)

    all_forces, all_positions, all_velocities = [], [], []
    episode_lengths, final_positions = [], []
    force_profiles, reward_profiles = [], []

    for ep in episodes:
        episode_lengths.append(len(ep))
        ep_forces, ep_pos, ep_vel, ep_rewards = [], [], [], []

        for frame in ep:
            if "observations" in frame and "state" in frame["observations"]:
                s = frame["observations"]["state"].flatten()
                if force_slice is not None:
                    ep_forces.append(s[force_slice])
                ep_pos.append(s[pos_slice])
                state_dim = len(s)
                vel_sl = STATE_INDICES.get("vel_lin")
                if vel_sl and vel_sl.stop <= state_dim:
                    ep_vel.append(s[vel_sl])
            if "rewards" in frame:
                ep_rewards.append(frame["rewards"])

        if not ep_pos:
            continue

        ep_pos = np.array(ep_pos)
        all_positions.append(ep_pos)
        final_positions.append(ep_pos[-1])

        if ep_forces:
            ep_forces = np.array(ep_forces)
            all_forces.append(ep_forces)
            force_profiles.append(np.linalg.norm(ep_forces, axis=1))

        if ep_vel:
            ep_vel = np.array(ep_vel)
            all_velocities.append(ep_vel)

        if ep_rewards:
            reward_profiles.append(np.cumsum(np.array(ep_rewards) - PENALTY_STEP))
        else:
            reward_profiles.append(np.zeros(len(ep_pos)))

    if not all_positions:
        print(f"[SKIP] No valid data in {data_path}")
        return

    has_force = len(all_forces) > 0
    has_vel = len(all_velocities) > 0
    flat_positions = np.concatenate(all_positions, axis=0)
    flat_forces = np.concatenate(all_forces, axis=0) if has_force else None
    flat_velocities = np.concatenate(all_velocities, axis=0) if has_vel else None
    final_positions_arr = np.array(final_positions)

    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(episode_lengths, kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Episode Lengths")

    ax2 = fig.add_subplot(gs[0, 1])
    if flat_forces is not None:
        sns.histplot(np.linalg.norm(flat_forces, axis=1), kde=True, ax=ax2, color="salmon", log_scale=(False, True))
        ax2.set_title("Force Magnitude (log Y)")
    else:
        ax2.set_title("Force: N/A")

    ax3 = fig.add_subplot(gs[0, 2])
    if flat_velocities is not None:
        sns.histplot(np.linalg.norm(flat_velocities, axis=1), kde=True, ax=ax3, color="lightgreen")
        ax3.set_title("Velocity Magnitude")
    else:
        ax3.set_title("Velocity: N/A")

    ax4 = fig.add_subplot(gs[0, 3])
    if flat_forces is not None:
        n_force_dims = flat_forces.shape[1]
        labels = [f"F{i}" for i in range(n_force_dims)]
        ax4.boxplot([flat_forces[:, i] for i in range(n_force_dims)], tick_labels=labels)
        ax4.set_title("Force Components")
    else:
        ax4.set_title("Force Components: N/A")

    ax5 = fig.add_subplot(gs[1, 0:2])
    h = ax5.hist2d(flat_positions[:, 0], flat_positions[:, 1], bins=50, cmap="Blues", cmin=1)
    fig.colorbar(h[3], ax=ax5)
    ax5.set_title("Workspace XY")

    ax6 = fig.add_subplot(gs[1, 2:4])
    h2 = ax6.hist2d(flat_positions[:, 0], flat_positions[:, 2], bins=50, cmap="Purples", cmin=1)
    fig.colorbar(h2[3], ax=ax6)
    ax6.set_title("Workspace XZ")

    sample_idx = list(range(min(3, len(force_profiles))))

    ax7 = fig.add_subplot(gs[2, 0])
    if force_profiles:
        for idx in sample_idx:
            if idx < len(force_profiles):
                ax7.plot(force_profiles[idx], alpha=0.7, label=f"Ep {idx}")
        ax7.set_title("Force Profiles")
    else:
        ax7.set_title("Force Profiles: N/A")

    ax8 = fig.add_subplot(gs[2, 1])
    if all_velocities:
        for idx in sample_idx:
            if idx < len(all_velocities):
                ax8.plot(np.linalg.norm(all_velocities[idx], axis=1), alpha=0.7)
        ax8.set_title("Velocity Profiles")
    else:
        ax8.set_title("Velocity Profiles: N/A")

    ax_rew = fig.add_subplot(gs[2, 2:4])
    for idx in sample_idx:
        ax_rew.plot(reward_profiles[idx], alpha=0.8, linewidth=2, label=f"Ep {idx}")
    ax_rew.set_title("Cumulative Reward")
    ax_rew.legend()

    ax9 = fig.add_subplot(gs[3, 0])
    ax9.scatter(final_positions_arr[:, 0], final_positions_arr[:, 1], alpha=0.6, c="crimson", edgecolors="k")
    ax9.set_title("Final XY")

    ax10 = fig.add_subplot(gs[3, 1])
    sns.histplot(final_positions_arr[:, 2], kde=True, ax=ax10, color="orange")
    ax10.set_title("Final Z Height")

    final_forces_mag = [fp[-1] for fp in force_profiles] if force_profiles else None
    ax11 = fig.add_subplot(gs[3, 2])
    if final_forces_mag:
        sns.histplot(final_forces_mag, kde=True, ax=ax11, color="brown")
        ax11.set_title("Final Force")
    else:
        ax11.set_title("Final Force: N/A")

    ax12 = fig.add_subplot(gs[3, 3], projection="3d")
    pos_dim = final_positions_arr.shape[1]
    if pos_dim >= 3:
        c = final_forces_mag if final_forces_mag else "steelblue"
        ax12.scatter(
            final_positions_arr[:, 0], final_positions_arr[:, 1], final_positions_arr[:, 2], c=c, cmap="viridis"
        )
    ax12.set_title("Final XYZ")

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{dataset_name}_probing.png")
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset probing visualization")
    parser.add_argument("--tasks", nargs="+", default=["button"], help="Task names or 'all'")
    parser.add_argument(
        "--split", choices=["success", "fail", "all"], default="all", help="Which data split to visualize"
    )
    parser.add_argument("--output", type=str, default="analysis_plots", help="Output directory for plots")
    args = parser.parse_args()

    tasks = list_tasks() if "all" in args.tasks else args.tasks

    for task_name in tasks:
        task = get_task(task_name)
        paths = []
        if args.split in ("success", "all"):
            paths.append(task.success_path)
        if args.split in ("fail", "all"):
            paths.append(task.fail_path)

        for path in paths:
            if os.path.exists(path):
                visualize_dataset_probing(path, output_dir=args.output, task=task)
            else:
                print(f"[SKIP] {path} not found")
