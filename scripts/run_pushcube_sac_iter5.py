#!/usr/bin/env python
"""PushCube SAC long-run preset for Iteration 5: GT-only vs RM-shaped reward.

Runs ``sim.sac_train.train`` with a **fixed** ``run_name`` so outputs and
``log.json`` land in a predictable directory, then appends one line to
``exp_ledger.jsonl`` for comparison tracking.

Example::

    # Environment reward only (no RM), 200k steps
    python scripts/run_pushcube_sac_iter5.py --variant gt

    # Default RM checkpoint from sim.task_configs (PushCube)
    python scripts/run_pushcube_sac_iter5.py --variant rm

    # Dry-run: print resolved config paths only
    python scripts/run_pushcube_sac_iter5.py --variant rm --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("RM_DATA_DIR", str(_REPO_ROOT / "data"))

from sim.sac_train import SACConfig, train  # noqa: E402
from sim.task_configs import get_sim_config  # noqa: E402

logger = logging.getLogger(__name__)


def _git_short_hash() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        out = subprocess.run(  # noqa: S603 — argv from shutil.which("git"), not user input
            [git, "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def _next_ledger_id(path: Path) -> int:
    if not path.is_file():
        return 0
    best = -1
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            best = max(best, int(json.loads(line).get("id", -1)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return best + 1


def _resolve_rm_checkpoint(path_str: str | None, repo: Path) -> str | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = repo / p
    return str(p.resolve()) if p.is_file() else str(p)


def main() -> None:
    p = argparse.ArgumentParser(description="Iteration 5 PushCube SAC long-run (GT vs RM)")
    p.add_argument("--variant", choices=("gt", "rm"), required=True)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--learning_starts", type=int, default=4_000)
    p.add_argument("--eval_freq", type=int, default=10_000)
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--save_freq", type=int, default=100_000)
    p.add_argument("--relabel_interval", type=int, default=500)
    p.add_argument("--relabel_batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="runs/iter5_pushcube_sac")
    p.add_argument(
        "--rm_checkpoint",
        default=None,
        help="Override RM checkpoint path (rm variant only); default from task registry",
    )
    p.add_argument("--ledger", type=Path, default=_REPO_ROOT / "exp_ledger.jsonl")
    p.add_argument("--no-ledger", action="store_true", help="Do not append exp_ledger.jsonl")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    sim_cfg = get_sim_config("pushcube")
    run_name = f"{args.variant}_s{args.seed}_t{args.total_timesteps}"

    if args.variant == "gt":
        rm_ckpt: str | None = None
        alpha = 0.0
    else:
        raw = args.rm_checkpoint if args.rm_checkpoint is not None else sim_cfg.rm_checkpoint
        rm_ckpt = _resolve_rm_checkpoint(raw, _REPO_ROOT)
        alpha = sim_cfg.reward_shaping_weight

    cfg = SACConfig(
        task="pushcube",
        rm_checkpoint=rm_ckpt,
        reward_shaping_weight=alpha,
        relabel_interval=args.relabel_interval,
        relabel_batch=args.relabel_batch,
        total_timesteps=args.total_timesteps,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        output_dir=args.output_dir,
        run_name=run_name,
        seed=args.seed,
        device=args.device,
    )

    run_dir = Path(cfg.output_dir) / run_name
    if args.dry_run:
        print("variant:", args.variant)
        print("run_dir:", run_dir)
        print("rm_checkpoint:", rm_ckpt)
        print("reward_shaping_weight:", alpha)
        print("total_timesteps:", cfg.total_timesteps)
        return

    if args.variant == "rm" and rm_ckpt is not None and not Path(rm_ckpt).is_file():
        logger.error("RM checkpoint not found: %s", rm_ckpt)
        sys.exit(1)

    train(cfg)

    log_path = run_dir / "log.json"
    if not log_path.is_file():
        logger.warning("Missing %s; skip ledger append", log_path)
        return

    log_data = json.loads(log_path.read_text(encoding="utf-8"))
    best_sr = max((float(x.get("success_rate", 0.0)) for x in log_data), default=0.0)
    last = log_data[-1] if log_data else {}

    if args.no_ledger:
        return

    entry = {
        "id": _next_ledger_id(args.ledger),
        "experiment": "iter5_pushcube_sac_long",
        "variant": args.variant,
        "commit": _git_short_hash(),
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "reward_shaping_weight": alpha,
        "rm_checkpoint": rm_ckpt,
        "run_dir": str(run_dir.resolve()),
        "best_success_rate": best_sr,
        "last_eval_step": last.get("step"),
        "last_success_rate": last.get("success_rate"),
        "last_mean_reward": last.get("mean_reward"),
    }
    with args.ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Appended ledger id=%s -> %s", entry["id"], args.ledger)


if __name__ == "__main__":
    main()
