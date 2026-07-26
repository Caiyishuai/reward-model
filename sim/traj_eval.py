"""Evaluate Reward Model on real ManiSkill3 trajectories.

Collects trajectories in the ManiSkill3 environment using a policy (PPO
checkpoint or random), scores each step with the trained RM, and reports
alignment metrics.  This is a separate evaluation axis from
RM-guided training — it directly measures whether the RM assigns higher
scores to successful trajectories.

Reported metrics:

* **PRA** — pairwise ranking accuracy (success final RM > fail final RM)
* **Gap** — mean success RM − mean fail RM at final step
* **Monotonicity** — within successful trajectories, RM reward should increase
* **RM-Env Spearman** — rank correlation between RM reward and env reward
* **Per-trajectory RM curve** — saved for visualization

Implementation detail: rollout and shared metrics live in
:mod:`sim.eval_common`. This module owns the CLI, the trajectory-level
report (return-based metrics + per-trajectory curves) and policy-type
selection.

Usage::

    python -m sim.traj_eval \\
        --task pushcube \\
        --rm_checkpoint checkpoints/auto_pushcube/rm_pushcube_epoch_55_val_0.8286.pt \\
        --policy_checkpoint rl_ckpt/pushcube_ckpt_301.pt \\
        --policy_type ppo \\
        --n_episodes 100
"""

from __future__ import annotations

import argparse
from typing import Any

from sim.eval_common import (
    EpisodeRecord as Trajectory,  # noqa: F401  — kept for API compatibility
)
from sim.eval_common import (
    compute_return_metrics,
    compute_step_metrics,
    make_ppo_policy,
    make_random_policy,
    run_episodes,
    save_eval_output,
    setup_online_eval,
)
from sim.task_configs import get_sim_config

__all__ = ["Trajectory", "evaluate_trajectories"]


def evaluate_trajectories(
    task_name: str,
    rm_checkpoint: str,
    policy_checkpoint: str | None = None,
    policy_type: str = "ppo",
    n_episodes: int = 100,
    output_dir: str = "eval_results",
    device_str: str = "cuda",
) -> dict[str, Any]:
    """Collect trajectories and evaluate RM alignment.

    Returns a dict with both step-level and return-level metrics.
    """
    cfg, device, rm, env, adapter = setup_online_eval(task_name, rm_checkpoint, device_str)

    print(f"[Traj-Eval] Task: {cfg.env_id}, Episodes: {n_episodes}")
    print(f"[Traj-Eval] RM: {rm_checkpoint}")
    print(f"[Traj-Eval] Policy: {policy_checkpoint or 'random'} (type={policy_type})")

    if policy_checkpoint and policy_type == "ppo":
        policy_fn = make_ppo_policy(policy_checkpoint, env, device)
    else:
        policy_fn = make_random_policy(env)
        if policy_checkpoint:
            print(f"[Traj-Eval] Warning: policy_type='{policy_type}' not yet supported for loading, using random")

    records, elapsed = run_episodes(
        env,
        policy_fn,
        rm,
        adapter,
        device,
        cfg.max_episode_steps,
        n_episodes,
        tag="Traj-Eval",
        extra_fields=lambda rec: f"rm_return={rec.rm_return:.3f}  env_return={rec.env_return:.3f}",
    )
    env.close()

    step = compute_step_metrics(records)
    ret = compute_return_metrics(records)

    metrics: dict[str, Any] = {
        "task": task_name,
        "env_id": cfg.env_id,
        "policy_type": policy_type,
        "policy_checkpoint": policy_checkpoint,
        **step,
        **ret,
        # gap_final / gap_return keep the (min-max) definition shared with
        # label/metric.py so offline and online reports are commensurable.
        "gap_final": step["gap_final_minmax"],
        "gap_return": ret["gap_return_minmax"],
        "elapsed_seconds": elapsed,
    }

    per_traj = [
        {
            "episode": i,
            "success": r.success,
            "length": r.length,
            "rm_rewards": r.rm_rewards,
            "env_rewards": r.env_rewards,
        }
        for i, r in enumerate(records)
    ]

    _print_summary(cfg.env_id, metrics)
    main_path = save_eval_output(
        metrics,
        output_dir,
        filename=f"traj_eval_{task_name}.json",
        extra_files={f"traj_curves_{task_name}.json": per_traj},
    )
    print(f"\n  Metrics  saved to {main_path}")
    print(f"  Curves   saved to {main_path.parent / f'traj_curves_{task_name}.json'}")

    return metrics


def _print_summary(env_id: str, m: dict[str, Any]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  RM Trajectory Evaluation — {env_id}")
    print(f"{'=' * 70}")
    print(f"  Episodes       : {m['n_episodes']} ({m['n_success']} success, {m['n_fail']} fail)")
    print(f"  Success rate   : {m['success_rate'] * 100:.1f}%")
    print(f"  PRA (final)    : {m['pra_final'] * 100:.1f}%")
    print(f"  PRA (return)   : {m['pra_return'] * 100:.1f}%")
    print(f"  Gap (min-max)  : {m['gap_final_minmax']:.4f}")
    print(f"  Gap (mean)     : {m['gap_final_mean']:.4f}")
    print(f"  Succ RM final  : {m['succ_rm_final_mean']:.4f}")
    print(f"  Fail RM final  : {m['fail_rm_final_mean']:.4f}")
    print(f"  Monotonicity   : {m['monotonicity'] * 100:.1f}%")
    print(f"  Step Spearman  : {m['rm_env_step_spearman']:.4f}")
    print(f"  Return Spearman: {m['rm_env_return_spearman']:.4f}")
    print(f"  Time           : {m['elapsed_seconds']:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RM on real ManiSkill3 trajectories")
    parser.add_argument("--task", type=str, default="pushcube")
    parser.add_argument("--rm_checkpoint", type=str, default=None)
    parser.add_argument("--policy_checkpoint", type=str, default=None)
    parser.add_argument(
        "--policy_type",
        type=str,
        default="ppo",
        choices=["ppo", "random"],
        help="Policy type: ppo (load checkpoint) or random",
    )
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sim_cfg = get_sim_config(args.task)
    rm_ckpt = args.rm_checkpoint or sim_cfg.rm_checkpoint
    policy_ckpt = args.policy_checkpoint or sim_cfg.ppo_checkpoint

    if rm_ckpt is None:
        raise ValueError("No RM checkpoint specified (--rm_checkpoint or in task config)")

    evaluate_trajectories(
        task_name=args.task,
        rm_checkpoint=rm_ckpt,
        policy_checkpoint=policy_ckpt,
        policy_type=args.policy_type,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
