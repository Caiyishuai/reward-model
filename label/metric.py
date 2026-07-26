"""Reward labeling quality metrics extracted from eval_labeling.py.

Metrics (all with unified normalization):

  PRA          — final-step pairwise ranking accuracy (guard ≥95%)
  sAUC         — stepwise AUC across time-aligned bins
  LOO-sAUC     — leave-one-out stepwise AUC: true generalization quality
  WinRank      — random-window reward-sum ranking accuracy: RM-training proxy
  Mono         — success curve monotonicity (guard ≥90%)
  Gap          — min(succ_final) - max(fail_final)
  Coherence    — intra-class reward curve similarity
  Composite    — 0.40*LOO_sAUC + 0.30*WinRank + 0.15*PRA + 0.15*Mono

Usage:
    python -m label.metric                         # all tasks with data
    python -m label.metric --tasks button pushcube
    python -m label.metric --tasks all --no-loo     # skip LOO (faster)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

N_ALIGN_BINS = 20
N_WINDOW_SAMPLES = 2000
WINDOW_SIZES = [5, 10, 20]


# ------------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------------
@dataclass
class TaskMetrics:
    """Per-task evaluation metrics."""

    task: str
    pra: float
    gap: float
    monotonicity: float
    step_auc: float
    loo_step_auc: float
    win_rank: float
    coherence: float
    succ_mean: float
    fail_mean: float
    duration_s: float
    error: str | None = None


@dataclass
class AggregateMetrics:
    """Cross-task aggregated metrics."""

    task_results: list[TaskMetrics] = field(default_factory=list)
    mean_pra: float = 0.0
    mean_gap: float = 0.0
    mean_mono: float = 0.0
    mean_step_auc: float = 0.5
    mean_loo_auc: float = 0.5
    mean_win_rank: float = 0.5
    mean_coherence: float = 0.0
    min_pra: float = 0.0
    min_loo_auc: float = 0.5
    n_tasks: int = 0
    composite: float = 0.0
    total_duration_s: float = 0.0


# ------------------------------------------------------------------
# Core Metric Computation
# ------------------------------------------------------------------
def _get_reward_curve(ep: list[dict]) -> np.ndarray:
    return np.array([step["rewards"] for step in ep], dtype=np.float64)


def _align_curve(curve: np.ndarray, n_bins: int = N_ALIGN_BINS) -> np.ndarray:
    T = len(curve)
    if T < 2:
        return np.full(n_bins, curve[0] if T == 1 else 0.0)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, n_bins)
    return interp1d(x_old, curve, kind="linear")(x_new)


def compute_step_auc(
    succ_aligned: np.ndarray,
    fail_aligned: np.ndarray,
) -> float:
    """Per-time-bin AUC between success and fail curves, then average.

    Vectorized: broadcasts [n_s, 1, B] - [1, n_f, B] → [n_s, n_f, B].
    """
    total = succ_aligned.shape[0] * fail_aligned.shape[0]
    if total == 0:
        return 0.5
    diff = succ_aligned[:, None, :] - fail_aligned[None, :, :]
    correct = (diff > 0).sum(axis=(0, 1))
    ties = (diff == 0).sum(axis=(0, 1))
    aucs = (correct + 0.5 * ties) / total
    return float(aucs.mean())


def compute_coherence(curves: np.ndarray) -> float:
    """Mean pairwise cosine similarity of aligned reward curves."""
    n = len(curves)
    if n < 2:
        return 1.0
    norms = np.linalg.norm(curves, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = curves / norms
    sim_matrix = normed @ normed.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim_matrix[mask].mean())


def compute_window_ranking(
    succ_curves: list[np.ndarray],
    fail_curves: list[np.ndarray],
    rng: np.random.Generator | None = None,
) -> float:
    """Fraction of random windows where success reward sum > fail reward sum."""
    if rng is None:
        rng = np.random.default_rng(42)
    correct = 0
    total = 0
    for w in WINDOW_SIZES:
        for _ in range(N_WINDOW_SAMPLES // len(WINDOW_SIZES)):
            si = rng.integers(0, len(succ_curves))
            fi = rng.integers(0, len(fail_curves))
            sc, fc = succ_curves[si], fail_curves[fi]
            if len(sc) < w or len(fc) < w:
                continue
            s_start = rng.integers(0, len(sc) - w + 1)
            f_start = rng.integers(0, len(fc) - w + 1)
            s_sum = sc[s_start : s_start + w].sum()
            f_sum = fc[f_start : f_start + w].sum()
            total += 1
            if s_sum > f_sum:
                correct += 1
    return correct / total if total > 0 else 0.5


def success_curve_monotonicity(
    curves: list[np.ndarray] | list[list[float]],
    *,
    min_length: int = 2,
    smooth_threshold: float = -0.01,
    smooth_ratio: float = 0.8,
) -> float:
    """Single source of truth for success-trajectory monotonicity.

    Short episodes (``len(curve) < min_length``) are *excluded* from both the
    numerator and the denominator. Historically two call sites diverged on the
    short-episode rule — see review C3. Centralising here keeps dashboards and
    training logs comparable.

    Returns fraction of qualifying curves whose smoothed first-difference is
    ≥ ``smooth_threshold`` at least ``smooth_ratio`` of the time.
    """
    mono_count = 0
    n_eligible = 0
    for raw in curves:
        c = np.asarray(raw, dtype=np.float64)
        if len(c) < min_length:
            continue
        n_eligible += 1
        w = max(1, len(c) // 10)
        sm = np.convolve(c, np.ones(w) / w, mode="valid")
        diffs = np.diff(sm)
        if diffs.size == 0:
            continue
        if np.mean(diffs >= smooth_threshold) > smooth_ratio:
            mono_count += 1
    return mono_count / n_eligible if n_eligible else 0.0


def evaluate_labeled_episodes(
    succ_eps: list[list[dict]],
    fail_eps: list[list[dict]],
) -> dict[str, float]:
    """Compute all metrics from already-labeled success/fail episodes.

    Episodes must have 'rewards' populated in each step dict.
    Returns dict with keys: pra, gap, monotonicity, step_auc, coherence,
                            win_rank, succ_mean, fail_mean.
    """
    succ_raw = [_get_reward_curve(ep) for ep in succ_eps if len(ep) > 0]
    fail_raw = [_get_reward_curve(ep) for ep in fail_eps if len(ep) > 0]

    if not succ_raw or not fail_raw:
        return {
            k: 0.0
            for k in [
                "pra",
                "gap",
                "monotonicity",
                "step_auc",
                "coherence",
                "win_rank",
                "succ_mean",
                "fail_mean",
            ]
        }

    all_flat = np.concatenate([np.concatenate(succ_raw), np.concatenate(fail_raw)])
    vmin, vmax = all_flat.min(), all_flat.max()
    if vmax - vmin < 1e-8:  # noqa: SIM108
        sc = lambda x: np.full_like(x, 3.0)  # noqa: E731
    else:
        sc = lambda x: (x - vmin) / (vmax - vmin) * 6.0  # noqa: E731

    succ_scaled = [sc(c) for c in succ_raw]
    fail_scaled = [sc(c) for c in fail_raw]

    succ_finals = np.array([c[-1] for c in succ_scaled])
    fail_finals = np.array([c[-1] for c in fail_scaled])
    total = len(succ_finals) * len(fail_finals)
    if total > 0:
        correct = int((succ_finals[:, None] > fail_finals[None, :]).sum())
        pra = correct / total
    else:
        pra = 0.0
    gap = float(succ_finals.min() - fail_finals.max()) if total > 0 else 0.0

    mono_count = 0
    for c in succ_scaled:
        if len(c) < 3:
            mono_count += 1
            continue
        w = max(1, len(c) // 10)
        sm = np.convolve(c, np.ones(w) / w, mode="valid")
        mono_count += int(np.mean(np.diff(sm) >= -0.01) > 0.8)
    mono = mono_count / len(succ_scaled)

    succ_aligned = np.array([_align_curve(c) for c in succ_scaled])
    fail_aligned = np.array([_align_curve(c) for c in fail_scaled])
    step_auc = compute_step_auc(succ_aligned, fail_aligned)

    succ_coh = compute_coherence(succ_aligned)
    fail_coh = compute_coherence(fail_aligned)
    coherence = 0.6 * succ_coh + 0.4 * fail_coh

    rng = np.random.default_rng(42)
    win_rank = compute_window_ranking(succ_scaled, fail_scaled, rng)

    return {
        "pra": pra,
        "gap": gap,
        "monotonicity": mono,
        "step_auc": step_auc,
        "coherence": coherence,
        "win_rank": win_rank,
        "succ_mean": float(succ_finals.mean()),
        "fail_mean": float(fail_finals.mean()),
    }


# ------------------------------------------------------------------
# Strategy Evaluation Pipeline
# ------------------------------------------------------------------
def _deep_copy_episodes(episodes: list[list[dict]]) -> list[list[dict]]:
    """Copy episodes so that label mutations don't corrupt originals.

    Observations/actions arrays are shared (read-only during labeling);
    only scalar fields that get overwritten (rewards) are fresh copies.
    """
    copied = []
    for ep in episodes:
        ep_copy = []
        for step in ep:
            obs_copy = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in step["observations"].items()}
            act = step["actions"]
            ep_copy.append(
                {
                    "observations": obs_copy,
                    "actions": act.copy() if isinstance(act, np.ndarray) else act,
                    "rewards": step.get("rewards", 0.0),
                    "dones": step["dones"],
                    "infos": step.get("infos", {}),
                    **({"masks": step["masks"]} if "masks" in step else {}),
                }
            )
        copied.append(ep_copy)
    return copied


def _label_episode_list(strategy: object, episodes: list[list[dict]], is_success: bool) -> None:
    for ep in episodes:
        rewards = strategy.label(ep, is_success=is_success)  # type: ignore[attr-defined]
        for j, r in enumerate(rewards):
            ep[j]["rewards"] = float(r)


def evaluate_strategy_on_task(
    strategy_cls: type,
    strategy_kwargs: dict,
    succ_eps: list[list[dict]],
    fail_eps: list[list[dict]],
    task_name: str = "",
    compute_loo: bool = True,
) -> TaskMetrics:
    """Evaluate a labeling strategy on one task's episodes.

    Args:
        strategy_cls: Strategy class (must have fit/label/label_all).
        strategy_kwargs: Kwargs passed to strategy constructor after config.
        succ_eps: Raw success episodes.
        fail_eps: Raw failure episodes.
        task_name: Name for reporting.
        compute_loo: Whether to compute LOO-sAUC (slow but honest).
    """
    from label.strategies import StrategyConfig

    cfg = StrategyConfig(normalize_output=False)
    t0 = time.time()

    try:
        succ_copy = _deep_copy_episodes(succ_eps)
        fail_copy = _deep_copy_episodes(fail_eps)
        strategy = strategy_cls(cfg, **strategy_kwargs)
        strategy.fit(succ_copy, fail_copy)
        strategy.label_all(succ_copy, fail_copy)
        m = evaluate_labeled_episodes(succ_copy, fail_copy)

        loo_auc = 0.5
        if compute_loo:
            loo_auc = _compute_loo_auc(strategy_cls, strategy_kwargs, cfg, succ_eps, fail_eps)

        dt = time.time() - t0
        return TaskMetrics(
            task=task_name,
            pra=m["pra"],
            gap=m["gap"],
            monotonicity=m["monotonicity"],
            step_auc=m["step_auc"],
            loo_step_auc=loo_auc,
            win_rank=m["win_rank"],
            coherence=m["coherence"],
            succ_mean=m["succ_mean"],
            fail_mean=m["fail_mean"],
            duration_s=dt,
        )
    except (ValueError, RuntimeError, KeyError, AssertionError, np.linalg.LinAlgError, FileNotFoundError) as e:
        dt = time.time() - t0
        logging.warning("evaluate_strategy_on_task(%s) failed: %s", task_name, e)
        return TaskMetrics(
            task=task_name,
            pra=0,
            gap=-6,
            monotonicity=0,
            step_auc=0.5,
            loo_step_auc=0.5,
            win_rank=0.5,
            coherence=0,
            succ_mean=0,
            fail_mean=0,
            duration_s=dt,
            error=str(e),
        )


def _compute_loo_auc(
    strategy_cls: type,
    strategy_kwargs: dict,
    cfg: object,
    succ_eps: list[list[dict]],
    fail_eps: list[list[dict]],
) -> float:
    n_s, n_f = len(succ_eps), len(fail_eps)
    loo_succ_curves: list[np.ndarray] = []
    loo_fail_curves: list[np.ndarray] = []

    for i in range(n_s):
        s_train = [ep for j, ep in enumerate(succ_eps) if j != i]
        s_test = _deep_copy_episodes([succ_eps[i]])
        strat = strategy_cls(cfg, **strategy_kwargs)
        strat.fit(_deep_copy_episodes(s_train), _deep_copy_episodes(fail_eps))
        _label_episode_list(strat, s_test, is_success=True)
        loo_succ_curves.append(_get_reward_curve(s_test[0]))

    for i in range(n_f):
        f_train = [ep for j, ep in enumerate(fail_eps) if j != i]
        f_test = _deep_copy_episodes([fail_eps[i]])
        strat = strategy_cls(cfg, **strategy_kwargs)
        strat.fit(_deep_copy_episodes(succ_eps), _deep_copy_episodes(f_train))
        _label_episode_list(strat, f_test, is_success=False)
        loo_fail_curves.append(_get_reward_curve(f_test[0]))

    if not loo_succ_curves and not loo_fail_curves:
        return 0.5
    all_loo = np.concatenate(loo_succ_curves + loo_fail_curves)
    vmin, vmax = all_loo.min(), all_loo.max()
    if vmax - vmin < 1e-8:  # noqa: SIM108
        sc = lambda x: np.full_like(x, 3.0)  # noqa: E731
    else:
        sc = lambda x: (x - vmin) / (vmax - vmin) * 6.0  # noqa: E731

    loo_s_aligned = np.array([_align_curve(sc(c)) for c in loo_succ_curves])
    loo_f_aligned = np.array([_align_curve(sc(c)) for c in loo_fail_curves])
    return compute_step_auc(loo_s_aligned, loo_f_aligned)


def aggregate(results: list[TaskMetrics]) -> AggregateMetrics:
    """Aggregate per-task metrics into cross-task summary."""
    if not results:
        return AggregateMetrics()

    pras = [r.pra for r in results]
    monos = [r.monotonicity for r in results]
    loo_aucs = [r.loo_step_auc for r in results]
    win_ranks = [r.win_rank for r in results]

    mean_pra = float(np.mean(pras))
    mean_mono = float(np.mean(monos))
    mean_loo = float(np.mean(loo_aucs))
    mean_wr = float(np.mean(win_ranks))

    composite = 0.40 * np.clip((mean_loo - 0.5) * 2, 0, 1) + 0.30 * mean_wr + 0.15 * mean_pra + 0.15 * mean_mono

    return AggregateMetrics(
        task_results=results,
        mean_pra=mean_pra,
        mean_gap=float(np.mean([r.gap for r in results])),
        mean_mono=mean_mono,
        mean_step_auc=float(np.mean([r.step_auc for r in results])),
        mean_loo_auc=mean_loo,
        mean_win_rank=mean_wr,
        mean_coherence=float(np.mean([r.coherence for r in results])),
        min_pra=float(np.min(pras)),
        min_loo_auc=float(np.min(loo_aucs)),
        n_tasks=len(results),
        composite=float(composite),
        total_duration_s=sum(r.duration_s for r in results),
    )


def print_metrics(result: AggregateMetrics, label: str = "") -> None:
    """Pretty-print evaluation results."""
    header = f"  {label}" if label else "  EVALUATION"
    print(f"\n{'─' * 100}")
    print(header)
    print(f"{'─' * 100}")
    print(
        f"  {'Task':16s} {'PRA':>6s} {'Gap':>6s} {'Mono':>5s} "
        f"{'sAUC':>5s} {'LOO':>5s} {'WinR':>5s} {'Coh':>5s} {'T':>5s}"
    )
    print(f"  {'─' * 88}")
    for r in result.task_results:
        err = " !" if r.error else ""
        print(
            f"  {r.task:16s} {r.pra * 100:5.1f}% {r.gap:+5.1f} {r.monotonicity * 100:4.0f}% "
            f"{r.step_auc:5.3f} {r.loo_step_auc:5.3f} {r.win_rank:5.3f} "
            f"{r.coherence:5.3f} {r.duration_s:4.1f}s{err}"
        )
    print(f"  {'─' * 88}")
    print(
        f"  {'AGG':16s} {result.mean_pra * 100:5.1f}% {result.mean_gap:+5.1f} "
        f"{result.mean_mono * 100:4.0f}% {result.mean_step_auc:5.3f} "
        f"{result.mean_loo_auc:5.3f} {result.mean_win_rank:5.3f} "
        f"{result.mean_coherence:5.3f}"
    )
    print(
        f"  MinPRA={result.min_pra * 100:.1f}% MinLOO={result.min_loo_auc:.3f} "
        f"Composite={result.composite:.4f} Time={result.total_duration_s:.1f}s"
    )
    print(f"{'─' * 100}")


# ------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os

    from data.common import get_episodes, get_task, list_tasks, load_pickle
    from label.strategies import GloballyConsistentStrategy

    parser = argparse.ArgumentParser(description="Evaluate labeling quality on all tasks")
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Task names or 'all'")
    parser.add_argument("--no-loo", action="store_true", help="Skip LOO-sAUC (much faster)")
    args = parser.parse_args()

    tasks = list_tasks() if "all" in args.tasks else args.tasks

    results: list[TaskMetrics] = []
    for task_name in tasks:
        cfg = get_task(task_name)
        if not os.path.exists(cfg.success_path) or not os.path.exists(cfg.fail_path):
            print(f"[SKIP] {task_name}: data not found")
            continue

        print(f"Evaluating {task_name}...")
        raw_s = load_pickle(cfg.success_path)
        raw_f = load_pickle(cfg.fail_path)

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

        succ_eps = get_episodes(flatten(raw_s))
        fail_eps = get_episodes(flatten(raw_f))

        r = evaluate_strategy_on_task(
            GloballyConsistentStrategy,
            {"time_decay": 0.95, "margin": 2.0},
            succ_eps,
            fail_eps,
            task_name=task_name,
            compute_loo=not args.no_loo,
        )
        results.append(r)

    agg = aggregate(results)
    print_metrics(agg, "GloballyConsistentStrategy (all tasks)")
