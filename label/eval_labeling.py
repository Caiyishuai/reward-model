"""Multi-task labeling evaluation with honest generalization metrics.

Thin wrapper around label.metric — kept for backward compatibility.

Usage:
    python -m label.eval_labeling
"""

import os

from data.common import get_episodes, get_task, load_pickle
from label.metric import (
    AggregateMetrics as AggregateResult,
)
from label.metric import (
    TaskMetrics as TaskResult,
)
from label.metric import (
    _deep_copy_episodes as deep_copy_episodes,
)
from label.metric import (
    aggregate as _aggregate,
)
from label.metric import (
    evaluate_labeled_episodes as evaluate,
)
from label.metric import (
    evaluate_strategy_on_task as _run_single_task_impl,
)
from label.metric import (
    print_metrics as print_result,
)

__all__ = [
    "TaskResult",
    "AggregateResult",
    "EVAL_TASKS",
    "load_task_episodes",
    "deep_copy_episodes",
    "evaluate",
    "evaluate_strategy_multi_task",
    "print_result",
    "result_to_dict",
]

EVAL_TASKS = [
    "button",
    "pickup",
    "plug_insert",
    "iphone_insert",
    "usb",
    "op_dr",
    "pk_toy",
    "pl_toy",
]


def load_task_episodes(task_name: str) -> tuple[list[list[dict]], list[list[dict]]] | None:
    """Load and split episodes for a task. Returns None if data is missing."""
    cfg = get_task(task_name)
    if not os.path.exists(cfg.success_path) or not os.path.exists(cfg.fail_path):
        return None

    succ_raw = load_pickle(cfg.success_path)
    fail_raw = load_pickle(cfg.fail_path)

    def flatten(data: object) -> list[dict]:
        if not isinstance(data, list):
            data = list(data)
        flat: list[dict] = []
        for item in data:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    return get_episodes(flatten(succ_raw)), get_episodes(flatten(fail_raw))


def evaluate_strategy_multi_task(
    strategy_cls: type,
    strategy_kwargs: dict,
    tasks: list[str] | None = None,
) -> AggregateResult:
    """Evaluate a strategy across multiple tasks with LOO-sAUC."""
    if tasks is None:
        tasks = EVAL_TASKS

    results: list[TaskResult] = []
    for t in tasks:
        data = load_task_episodes(t)
        if data is None:
            continue
        succ_eps, fail_eps = data
        r = _run_single_task_impl(
            strategy_cls,
            strategy_kwargs,
            succ_eps,
            fail_eps,
            task_name=t,
            compute_loo=True,
        )
        results.append(r)

    return _aggregate(results)


def result_to_dict(r: AggregateResult) -> dict:
    """Serialize aggregate result to a plain dict."""
    return {
        "composite": r.composite,
        "mean_loo_auc": r.mean_loo_auc,
        "mean_win_rank": r.mean_win_rank,
        "mean_pra": r.mean_pra,
        "mean_mono": r.mean_mono,
        "mean_gap": r.mean_gap,
        "mean_step_auc": r.mean_step_auc,
        "mean_coherence": r.mean_coherence,
        "min_pra": r.min_pra,
        "min_loo_auc": r.min_loo_auc,
        "per_task": {
            t.task: {"pra": t.pra, "loo": t.loo_step_auc, "wr": t.win_rank, "mono": t.monotonicity, "gap": t.gap}
            for t in r.task_results
        },
    }


if __name__ == "__main__":
    from label.strategies import GloballyConsistentStrategy

    r = evaluate_strategy_multi_task(
        GloballyConsistentStrategy,
        {"time_decay": 0.95, "margin": 2.0},
    )
    print_result(r, "GloballyConsistent (LOO + WinRank)")
