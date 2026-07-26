#!/usr/bin/env python
"""PushCube: SAC v2 + Reward Model training presets (GOAL phase 4).

Presets:
  smoke                  — short pipeline smoke-test (12k steps, small buffer, no compile)
  mixed                  — 200k, env_reward + 0.5*rm_reward  (对照组，4a)
  rm-only                — 200k, pure rm_reward, env_reward=0 (4b 默认)
  rm-only-no-norm        — 同 rm-only，但 ``rm_normalize=False``（auto-exp: 去归一化以稳定尺度）
  rm-only-nounc          — 同 rm-only，关闭 uncertainty weighting（auto-exp: RM 集成方差过大时去权放信号）
  rm-only-potential      — 同 rm-only，启用 ``γ·RM(s')−RM(s)`` potential-based shaping
                           (auto-exp, Ng/Harada/Russell: 保持最优策略不变同时移除 RM 偏移)

AutoRM 核心命题验证：只有 ``rm-only`` 成功（eval success 接近纯 env baseline ~94%）才能
证明 auto labelling → RM 链路可以独立替代手工 reward。

Requires optional sim deps (ManiSkill3: PyPI package ``mani-skill`` 3.x). Prefer Python 3.10
(see ``.python-version``): dependency ``toppra`` often has no wheel for Python 3.11 on Linux.::

    uv sync --extra sim
    uv run --extra sim python scripts/run_pushcube_sac_v2_rm.py --preset smoke
    uv run --extra sim python scripts/run_pushcube_sac_v2_rm.py --preset mixed
    uv run --extra sim python scripts/run_pushcube_sac_v2_rm.py --preset rm-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sim.sac_train_v2 import Args, train  # noqa: E402
from sim.task_configs import get_sim_config  # noqa: E402


def _resolve_rm(repo: Path, path_str: str | None) -> str:
    if not path_str:
        raise SystemExit("SimTaskConfig.rm_checkpoint is missing for pushcube")
    p = Path(path_str)
    if not p.is_absolute():
        p = repo / p
    if not p.is_file():
        raise SystemExit(f"RM checkpoint not found: {p}")
    return str(p.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="PushCube SAC v2 + RM training presets")
    parser.add_argument(
        "--preset",
        choices=(
            "smoke",
            "mixed",
            "rm-only",
            "rm-only-no-norm",
            "rm-only-nounc",
            "rm-only-potential",
        ),
        default="rm-only",
        help=(
            "smoke: 12k pipeline smoke-test | "
            "mixed: 200k env+rm reward (4a, 对照组) | "
            "rm-only: 200k pure RM, env=0 | "
            "rm-only-no-norm: rm-only + rm_normalize=False | "
            "rm-only-nounc: rm-only + rm_uncertainty_weight=False | "
            "rm-only-potential: rm-only + potential-based shaping γ·RM(s')−RM(s)"
        ),
    )
    parser.add_argument(
        "--rm-checkpoint",
        type=str,
        default=None,
        help="Override RM weights path (default: pushcube entry in task_configs)",
    )
    args_ns = parser.parse_args()

    cfg = get_sim_config("pushcube")
    rm_path = _resolve_rm(_REPO_ROOT, args_ns.rm_checkpoint or cfg.rm_checkpoint)

    # Shared base kwargs for all 200k full runs
    _full_base = dict(
        env_id=cfg.env_id,
        robot_uids=cfg.robot_uid,
        num_envs=32,
        num_eval_envs=8,
        total_timesteps=200_000,
        buffer_size=100_000,  # ~2.3 GB GPU; fits alongside 32-env sim + RM backbone
        compile=True,
        amp=True,
        training_freq=128,
        camera_width=64,
        camera_height=64,
        gamma=0.8,
        tau=0.01,
        rm_checkpoint=rm_path,
    )

    if args_ns.preset == "smoke":
        # Pipeline smoke-test: 12k steps, no compile, small buffer
        run_args = Args(
            exp_name="sac_pushcube_rm_smoke",
            env_id=cfg.env_id,
            robot_uids=cfg.robot_uid,
            num_envs=8,
            num_eval_envs=4,
            total_timesteps=12_000,
            learning_starts=4_000,
            training_freq=64,
            buffer_size=20_000,
            capture_video=False,
            compile=False,
            amp=True,
            camera_width=64,
            camera_height=64,
            gamma=0.8,
            tau=0.01,
            rm_checkpoint=rm_path,
            # smoke 也走 rm-only 验证流水线
            rm_alpha=1.0,
            env_reward_scale=0.0,
        )

    elif args_ns.preset == "mixed":
        # 对照组 (4a)：env_reward + 0.5 * rm_reward
        run_args = Args(
            exp_name="sac_pushcube_200k_mixed",
            rm_alpha=0.5,
            env_reward_scale=1.0,
            **_full_base,
        )

    elif args_ns.preset == "rm-only-no-norm":
        # Auto-exp hypothesis: running mean/std normalization may collapse signal early
        run_args = Args(
            exp_name="sac_pushcube_200k_rm_only_no_norm",
            rm_alpha=1.0,
            env_reward_scale=0.0,
            rm_normalize=False,
            **_full_base,
        )

    elif args_ns.preset == "rm-only-nounc":
        # Auto-exp hypothesis: uncertainty weighting ``1/(1+unc/std)`` may
        # collapse RM magnitude when ensemble variance is high; removing it
        # lets the signal reach the critic unattenuated.
        run_args = Args(
            exp_name="sac_pushcube_200k_rm_only_nounc",
            rm_alpha=1.0,
            env_reward_scale=0.0,
            rm_uncertainty_weight=False,
            **_full_base,
        )

    elif args_ns.preset == "rm-only-potential":
        # Auto-exp hypothesis: the raw RM(s) carries a task-independent offset
        # that dominates Q-learning. Potential-based shaping
        # F(s,a,s') = γ·RM(s') − RM(s) keeps the optimal policy (Ng, Harada,
        # Russell 1999) but re-centers reward scale to differentials which
        # are closer to TD-target magnitude.
        run_args = Args(
            exp_name="sac_pushcube_200k_rm_only_potential",
            rm_alpha=1.0,
            env_reward_scale=0.0,
            rm_potential_shaping=True,
            **_full_base,
        )

    else:  # rm-only — 核心验证实验 (4b)
        # env_reward 完全归零，policy 只靠 RM reward 学习
        # eval 仍用 env episode stats（success_once / return）
        # → 高 eval success 证明 auto labelling → RM 链路有效
        run_args = Args(
            exp_name="sac_pushcube_200k_rm_only",
            rm_alpha=1.0,
            env_reward_scale=0.0,
            **_full_base,
        )

    train(run_args)


if __name__ == "__main__":
    main()
