#!/usr/bin/env python3
"""Run held-out RM quality evaluation for the eight MetaWorld tasks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_metaworld_serl_data import METAWORLD_TASKS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eight-task MetaWorld RM quality benchmark")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--checkpoint-template", default="checkpoints/auto_{task}/best.pt")
    parser.add_argument("--prefix", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--output", type=Path, default=Path("eval_results/metaworld_rm_quality.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from scripts.evaluate_maniskill_rm_quality import evaluate_task

    tasks = list(METAWORLD_TASKS) if "all" in args.tasks else args.tasks
    unknown = sorted(set(tasks) - set(METAWORLD_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; expected {METAWORLD_TASKS}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    started = time.time()
    device = torch.device(args.device)
    reports, failures = [], {}
    for task in tasks:
        try:
            report = evaluate_task(
                task_name=task,
                checkpoint=Path(args.checkpoint_template.format(task=task)),
                prefix=args.prefix,
                device=device,
                batch_size=args.batch_size,
                split_ratio=args.split_ratio,
                seed=args.seed,
                gamma=args.gamma,
            )
            reports.append(report)
            metrics = report["metrics"]
            print(
                f"{task:20s} "
                f"label_rho={metrics['supervision_fidelity']['spearman_phi_vs_label']:.3f} "
                f"env_rho={metrics['environment_alignment']['spearman_phi_vs_dense']:.3f} "
                f"terminal_pra={metrics['trajectory_discrimination']['terminal_pra']:.3f} "
                f"pass={metrics['diagnostic_gates']['all_pass']}"
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            failures[task] = str(error)
            print(f"{task:20s} ERROR: {error}")

    payload = {
        "tasks": reports,
        "failures": failures,
        "duration_s": time.time() - started,
        "policy_utility_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
