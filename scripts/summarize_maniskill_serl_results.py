#!/usr/bin/env python3
"""Summarize locally logged ManiSkill SERL evaluation success rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PREFIX = "SERL_EVAL_JSON "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def success_value(metrics: dict[str, object]) -> float:
    for key in ("success", "episode_success_rate", "eval_success", "success_once"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"No episode success metric in evaluation keys: {sorted(metrics)}")


def main() -> int:
    args = parse_args()
    summaries = []
    for actor_log in sorted(args.runs_root.glob("*/actor.log")):
        evaluations = []
        for line in actor_log.read_text(errors="replace").splitlines():
            marker = line.find(PREFIX)
            if marker < 0:
                continue
            payload = json.loads(line[marker + len(PREFIX) :])
            evaluations.append(
                {
                    "step": int(payload["step"]),
                    "success_rate": success_value(payload["eval"]),
                    "metrics": payload["eval"],
                }
            )
        if not evaluations:
            raise ValueError(f"No evaluation records found in {actor_log}")

        parts = actor_log.parent.name.split("__")
        summaries.append(
            {
                "run": actor_log.parent.name,
                "task": parts[0],
                "reward_mode": parts[1],
                "tau_mode": parts[2],
                "seed": int(parts[3].removeprefix("seed")),
                "num_evaluations": len(evaluations),
                "best_success_rate": max(item["success_rate"] for item in evaluations),
                "final_success_rate": evaluations[-1]["success_rate"],
                "evaluations": evaluations,
            }
        )

    if not summaries:
        raise FileNotFoundError(f"No actor logs found below {args.runs_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2) + "\n")
    for summary in summaries:
        print(
            f"{summary['task']:12s} {summary['tau_mode']:8s} "
            f"best={summary['best_success_rate']:.3f} "
            f"final={summary['final_success_rate']:.3f}"
        )
    print(f"[OK] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
