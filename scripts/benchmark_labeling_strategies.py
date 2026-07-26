"""Benchmark every labeling strategy on a single task and rank by LOO-sAUC.

P1 action for the auto-label bottleneck: swap the HMM-only baseline in
``label.auto_label.run_pipeline`` for whichever pluggable strategy wins here.

Example:
    uv run python scripts/benchmark_labeling_strategies.py --task pushcube --no-loo
    uv run python scripts/benchmark_labeling_strategies.py --task pushcube

Outputs JSON + Markdown into ``exp_logs/strategy_benchmark/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.common import get_episodes, get_task, load_pickle  # noqa: E402
from label.metric import evaluate_strategy_on_task  # noqa: E402
from label.strategies import (  # noqa: E402
    ContrastiveDistanceStrategy,
    DiscriminativeStrategy,
    EnsembleStrategy,
    GloballyConsistentStrategy,
    HMMBaselineStrategy,
    HMMContrastiveHybridStrategy,
    OptimalTransportStrategy,
    PotentialBasedStrategy,
    ProgressContrastiveHybridStrategy,
    ProgressEstimatorStrategy,
    ReturnDecompositionStrategy,
    TemporalContrastiveStrategy,
)

# Mirror label.strategies.registry.build_strategy_pool, but as (label, cls, kwargs)
# triples so we can forward them to evaluate_strategy_on_task which needs class+kwargs.
STRATEGY_CONFIGS: list[tuple[str, type, dict]] = [
    ("HMM-baseline", HMMBaselineStrategy, {}),
    ("Contrast-k10-d0.3-t0.9", ContrastiveDistanceStrategy,
     {"n_clusters": 10, "distance_weight": 0.3, "temporal_weight": 0.9}),
    ("Contrast-k20-d0.5-t0.9", ContrastiveDistanceStrategy,
     {"n_clusters": 20, "distance_weight": 0.5, "temporal_weight": 0.9}),
    ("Contrast-k15-d0.5-t0.95", ContrastiveDistanceStrategy,
     {"n_clusters": 15, "distance_weight": 0.5, "temporal_weight": 0.95}),
    ("TempContrast-td0.9-n3", TemporalContrastiveStrategy,
     {"time_decay": 0.9, "n_neighbors": 3}),
    ("TempContrast-td0.95-n5", TemporalContrastiveStrategy,
     {"time_decay": 0.95, "n_neighbors": 5}),
    ("TempContrast-td0.99-n10", TemporalContrastiveStrategy,
     {"time_decay": 0.99, "n_neighbors": 10}),
    ("Progress-fc0.3", ProgressEstimatorStrategy, {"fail_ceiling": 0.3}),
    ("Progress-fc0.5", ProgressEstimatorStrategy, {"fail_ceiling": 0.5}),
    ("ReturnDecomp-h32-i200", ReturnDecompositionStrategy,
     {"hidden_dim": 32, "n_iters": 200}),
    ("ReturnDecomp-h64-i300", ReturnDecompositionStrategy,
     {"hidden_dim": 64, "n_iters": 300}),
    ("HMMContrastHyb-w0.3", HMMContrastiveHybridStrategy,
     {"hmm_weight": 0.3, "n_clusters": 15}),
    ("HMMContrastHyb-w0.5", HMMContrastiveHybridStrategy,
     {"hmm_weight": 0.5, "n_clusters": 15}),
    ("HMMContrastHyb-w0.7", HMMContrastiveHybridStrategy,
     {"hmm_weight": 0.7, "n_clusters": 15}),
    ("ProgContrastHyb-w0.4", ProgressContrastiveHybridStrategy,
     {"progress_weight": 0.4}),
    ("ProgContrastHyb-w0.6", ProgressContrastiveHybridStrategy,
     {"progress_weight": 0.6}),
    ("Potential-g0.99-n10", PotentialBasedStrategy,
     {"gamma": 0.99, "n_neighbors": 10}),
    ("Potential-g0.95-n5", PotentialBasedStrategy,
     {"gamma": 0.95, "n_neighbors": 5}),
    ("Discrim-w5", DiscriminativeStrategy, {"window": 5}),
    ("Discrim-w10", DiscriminativeStrategy, {"window": 10}),
    ("OT-b10", OptimalTransportStrategy, {"n_time_bins": 10}),
    ("OT-b20", OptimalTransportStrategy, {"n_time_bins": 20}),
    ("Ensemble", EnsembleStrategy, {}),
    ("GlobalCons-td0.9-m1.0", GloballyConsistentStrategy,
     {"time_decay": 0.9, "margin": 1.0}),
    ("GlobalCons-td0.95-m2.0", GloballyConsistentStrategy,
     {"time_decay": 0.95, "margin": 2.0}),
    ("GlobalCons-td0.85-m1.5", GloballyConsistentStrategy,
     {"time_decay": 0.85, "margin": 1.5}),
]


def _flatten(data: object) -> list:
    if not isinstance(data, list):
        data = list(data)
    flat: list = []
    for item in data:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="pushcube")
    parser.add_argument("--no-loo", action="store_true",
                        help="Skip LOO-sAUC (much faster, but less honest)")
    parser.add_argument("--output-dir", default="exp_logs/strategy_benchmark")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Subset of strategy labels to run (default: all)")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="Strategy labels to skip")
    args = parser.parse_args()

    cfg = get_task(args.task)
    if not (os.path.exists(cfg.success_path) and os.path.exists(cfg.fail_path)):
        raise SystemExit(
            f"Raw data not found for {args.task}: "
            f"{cfg.success_path} or {cfg.fail_path} missing"
        )

    raw_s = load_pickle(cfg.success_path)
    raw_f = load_pickle(cfg.fail_path)
    succ_eps = get_episodes(_flatten(raw_s))
    fail_eps = get_episodes(_flatten(raw_f))
    print(f"[{args.task}] loaded {len(succ_eps)} success, {len(fail_eps)} fail episodes",
          flush=True)

    configs = STRATEGY_CONFIGS
    if args.strategies:
        configs = [c for c in configs if c[0] in args.strategies]
    if args.skip:
        configs = [c for c in configs if c[0] not in args.skip]

    results: list[dict] = []
    t_total = time.time()
    for i, (label, cls, kwargs) in enumerate(configs, 1):
        t0 = time.time()
        try:
            r = evaluate_strategy_on_task(
                cls, kwargs, succ_eps, fail_eps,
                task_name=args.task,
                compute_loo=not args.no_loo,
            )
            r_dict = asdict(r)
        except Exception as e:  # noqa: BLE001
            r_dict = {
                "task": args.task,
                "pra": 0.0, "gap": -9.0, "monotonicity": 0.0,
                "step_auc": 0.5, "loo_step_auc": 0.5, "win_rank": 0.5,
                "coherence": 0.0, "succ_mean": 0.0, "fail_mean": 0.0,
                "duration_s": time.time() - t0,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            }
        dt = time.time() - t0
        r_dict["strategy"] = label
        results.append(r_dict)

        err = f" ERR={r_dict['error'].splitlines()[0]}" if r_dict.get("error") else ""
        print(
            f"  [{i:>2}/{len(configs)}] {label:28s} "
            f"PRA={r_dict['pra'] * 100:5.1f}% Gap={r_dict['gap']:+5.2f} "
            f"Mono={r_dict['monotonicity'] * 100:4.0f}% "
            f"sAUC={r_dict['step_auc']:5.3f} LOO={r_dict['loo_step_auc']:5.3f} "
            f"WinR={r_dict['win_rank']:5.3f} T={dt:5.1f}s{err}",
            flush=True,
        )

    total = time.time() - t_total

    # Rank primarily by LOO-sAUC (honest), break ties by win_rank then pra.
    results.sort(
        key=lambda r: (r["loo_step_auc"], r["win_rank"], r["pra"]),
        reverse=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "noloo" if args.no_loo else "loo"
    json_path = os.path.join(args.output_dir, f"{args.task}_{suffix}.json")
    md_path = os.path.join(args.output_dir, f"{args.task}_{suffix}.md")

    with open(json_path, "w") as f:
        json.dump(
            {
                "task": args.task,
                "n_success": len(succ_eps),
                "n_fail": len(fail_eps),
                "compute_loo": not args.no_loo,
                "total_duration_s": total,
                "ranked_by": "loo_step_auc desc, win_rank desc, pra desc",
                "results": results,
            },
            f,
            indent=2,
        )

    with open(md_path, "w") as f:
        f.write(f"# Labeling strategy benchmark — `{args.task}` ({suffix})\n\n")
        f.write(f"- success episodes: **{len(succ_eps)}**\n")
        f.write(f"- fail episodes: **{len(fail_eps)}**\n")
        f.write(f"- compute_loo: `{not args.no_loo}`\n")
        f.write(f"- total wall time: **{total:.1f}s**\n\n")
        f.write(
            "Ranked by **LOO-sAUC** (higher = better generalization). "
            "`sAUC` is the in-sample variant and can overfit when the strategy "
            "sees success/fail labels during fit.\n\n"
        )
        f.write(
            "| # | Strategy | PRA | Gap | Mono | sAUC | LOO | WinR | Coh | T(s) | err |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
        )
        for i, r in enumerate(results, 1):
            err_mark = "!" if r.get("error") else ""
            f.write(
                f"| {i} | `{r['strategy']}` | "
                f"{r['pra'] * 100:.1f}% | {r['gap']:+.2f} | "
                f"{r['monotonicity'] * 100:.0f}% | {r['step_auc']:.3f} | "
                f"{r['loo_step_auc']:.3f} | {r['win_rank']:.3f} | "
                f"{r['coherence']:.3f} | {r['duration_s']:.1f} | {err_mark} |\n"
            )
        err_rows = [r for r in results if r.get("error")]
        if err_rows:
            f.write("\n## Errors\n\n")
            for r in err_rows:
                f.write(f"### `{r['strategy']}`\n\n```\n{r['error']}\n```\n\n")

    print(f"\ntotal wall time: {total:.1f}s", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"Markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
