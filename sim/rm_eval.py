"""RM online evaluation in ManiSkill3 environments.

Rolls out a policy (PPO checkpoint or random) in simulation, scores every
step with the trained Reward Model, and reports alignment metrics:

* **RM-Success Correlation** — success trajectories should score higher
* **PRA** — pairwise ranking accuracy between success / fail final rewards
* **Monotonicity** — RM reward should increase along successful trajectories
* **RM-Env Correlation** — Spearman rank correlation with env reward

Implementation detail: the shared rollout / metric code now lives in
:mod:`sim.eval_common`; this module is a thin wrapper that owns the CLI and
the step-level report layout.

Usage::

    python -m sim.rm_eval \\
        --task pushcube \\
        --rm_checkpoint checkpoints/auto_pushcube/rm_pushcube_epoch_55_val_0.8286.pt \\
        --ppo_checkpoint rl_ckpt/pushcube_ckpt_301.pt \\
        --n_episodes 50
"""

from __future__ import annotations

import argparse
from typing import Any

from sim.eval_common import (
    EpisodeRecord,
    compute_monotonicity,
    compute_pra,
    compute_step_metrics,
    make_ppo_policy,
    make_random_policy,
    rollout_episode,
    run_episodes,
    save_eval_output,
    setup_online_eval,
    spearman_corr,
)
from sim.task_configs import get_sim_config

__all__ = [
    "EpisodeRecord",
    "compute_monotonicity",
    "compute_pra",
    "evaluate",
    "rollout_episode",
    "spearman_corr",
]


def evaluate(
    task_name: str,
    rm_checkpoint: str,
    ppo_checkpoint: str | None = None,
    n_episodes: int = 50,
    output_dir: str = "eval_results",
    device_str: str = "cuda",
) -> dict[str, Any]:
    """Run full RM online evaluation and return metrics dict."""
    cfg, device, rm, env, adapter = setup_online_eval(task_name, rm_checkpoint, device_str)

    print(f"[RM-Eval] Task: {cfg.env_id}, Episodes: {n_episodes}")
    print(f"[RM-Eval] RM: {rm_checkpoint}")
    print(f"[RM-Eval] PPO: {ppo_checkpoint or 'random policy'}")

    policy_fn = make_ppo_policy(ppo_checkpoint, env, device) if ppo_checkpoint else make_random_policy(env)

    records, elapsed = run_episodes(
        env, policy_fn, rm, adapter, device, cfg.max_episode_steps, n_episodes, tag="RM-Eval"
    )
    env.close()

    step_metrics = compute_step_metrics(records)

    metrics: dict[str, Any] = {
        "task": task_name,
        "env_id": cfg.env_id,
        **step_metrics,
        # Legacy field names preserved for downstream consumers that read
        # ``rm_eval_<task>.json``; their values come from ``gap_final_*``.
        "pra": step_metrics["pra_final"],
        "gap": step_metrics["gap_final_minmax"],
        "gap_minmax": step_metrics["gap_final_minmax"],
        "gap_mean": step_metrics["gap_final_mean"],
        "succ_rm_mean": step_metrics["succ_rm_final_mean"],
        "fail_rm_mean": step_metrics["fail_rm_final_mean"],
        "rm_env_spearman": step_metrics["rm_env_step_spearman"],
        "elapsed_seconds": elapsed,
    }

    _print_summary(cfg.env_id, metrics)
    path = save_eval_output(metrics, output_dir, f"rm_eval_{task_name}.json")
    print(f"\n  Report saved to {path}")

    return metrics


def _print_summary(env_id: str, m: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  RM Online Evaluation — {env_id}")
    print(f"{'=' * 60}")
    print(f"  Episodes     : {m['n_episodes']} ({m['n_success']} success, {m['n_fail']} fail)")
    print(f"  Success rate : {m['success_rate'] * 100:.1f}%")
    print(f"  PRA          : {m['pra_final'] * 100:.1f}%")
    print(f"  Gap (min-max): {m['gap_final_minmax']:.4f}")
    print(f"  Gap (mean)   : {m['gap_final_mean']:.4f}")
    print(f"  Succ RM mean : {m['succ_rm_final_mean']:.4f}")
    print(f"  Fail RM mean : {m['fail_rm_final_mean']:.4f}")
    print(f"  Monotonicity : {m['monotonicity'] * 100:.1f}%")
    print(f"  RM-Env Corr  : {m['rm_env_step_spearman']:.4f}")
    print(f"  Time         : {m['elapsed_seconds']:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="RM online evaluation in ManiSkill3")
    parser.add_argument("--task", type=str, default="pushcube")
    parser.add_argument("--rm_checkpoint", type=str, default=None)
    parser.add_argument("--ppo_checkpoint", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = get_sim_config(args.task)
    rm_ckpt = args.rm_checkpoint or cfg.rm_checkpoint
    ppo_ckpt = args.ppo_checkpoint or cfg.ppo_checkpoint

    if rm_ckpt is None:
        raise ValueError("No RM checkpoint specified (--rm_checkpoint or in task config)")

    evaluate(
        task_name=args.task,
        rm_checkpoint=rm_ckpt,
        ppo_checkpoint=ppo_ckpt,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
