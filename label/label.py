"""Unified labeling entry point for auto and manual reward labeling.

Usage:
    python -m label.label --task button --method auto
    python -m label.label --task all --method auto
    python -m label.label --task button usb --method manual
"""

import argparse

from data.common import list_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified reward labeling pipeline")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Task names to label, or 'all' for every registered task",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "manual"],
        default="auto",
        help="Labeling method: 'auto' (HMM) or 'manual' (potential-field)",
    )
    args = parser.parse_args()

    tasks = list_tasks() if "all" in args.tasks else args.tasks

    if args.method == "auto":
        from label.auto_label import run_pipeline as auto_run

        all_metrics = {}
        for task in tasks:
            result = auto_run(task)
            if result:
                all_metrics[task] = result

        if all_metrics:
            print(f"\n{'=' * 60}")
            print("  Summary")
            print(f"{'=' * 60}")
            print(f"{'Task':<18} {'PRA':>6} {'Gap':>8} {'Succ':>8} {'Fail':>8} {'Mono':>6}")
            print("-" * 60)
            for name, m in all_metrics.items():
                print(
                    f"{name:<18} {m.pra * 100:>5.1f}% {m.gap:>8.3f} "
                    f"{m.succ_mean:>8.3f} {m.fail_mean:>8.3f} {m.succ_monotonicity * 100:>5.0f}%"
                )
    else:
        from label.manual_label import run_pipeline as manual_run

        for task in tasks:
            manual_run(task)


if __name__ == "__main__":
    main()
