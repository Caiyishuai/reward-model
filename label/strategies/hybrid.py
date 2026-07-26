"""Hybrid and ensemble strategies (D1, D2, D3)."""

import logging

import numpy as np

from label.strategies._base import LabelingStrategy, StrategyConfig, normalize_rewards
from label.strategies.attribution import ProgressEstimatorStrategy
from label.strategies.contrastive import ContrastiveDistanceStrategy
from label.strategies.hmm import HMMBaselineStrategy

logger = logging.getLogger(__name__)


class HMMContrastiveHybridStrategy(LabelingStrategy):
    """Hybrid: HMM provides stage structure, contrastive provides within-stage density."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        hmm_weight: float = 0.5,
        n_clusters: int = 10,
    ):
        self.config = config or StrategyConfig()
        self.hmm_weight = hmm_weight
        self.n_clusters = n_clusters
        self._hmm = HMMBaselineStrategy(config)
        self._contrastive = ContrastiveDistanceStrategy(config, n_clusters=n_clusters)

    @property
    def name(self) -> str:
        return f"hmm_contrastive_hybrid_w{self.hmm_weight}_k{self.n_clusters}"

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        self._hmm.fit(success_episodes, fail_episodes)
        self._contrastive.fit(success_episodes, fail_episodes)

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        hmm_r = self._hmm.label(episode, is_success)
        cont_r = self._contrastive.label(episode, is_success)
        combined = self.hmm_weight * hmm_r + (1 - self.hmm_weight) * cont_r
        if self.config.normalize_output:
            combined = normalize_rewards(combined, self.config.min_reward, self.config.max_reward)
        return combined


class ProgressContrastiveHybridStrategy(LabelingStrategy):
    """Hybrid: progress provides monotonic structure, contrastive provides discriminative signal."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        progress_weight: float = 0.6,
        n_clusters: int = 15,
    ):
        self.config = config or StrategyConfig()
        self.progress_weight = progress_weight
        self._progress = ProgressEstimatorStrategy(config)
        self._contrastive = ContrastiveDistanceStrategy(config, n_clusters=n_clusters)

    @property
    def name(self) -> str:
        return f"progress_contrastive_w{self.progress_weight}"

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        self._progress.fit(success_episodes, fail_episodes)
        self._contrastive.fit(success_episodes, fail_episodes)

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        prog_r = self._progress.label(episode, is_success)
        cont_r = self._contrastive.label(episode, is_success)
        combined = self.progress_weight * prog_r + (1 - self.progress_weight) * cont_r
        if self.config.normalize_output:
            combined = normalize_rewards(combined, self.config.min_reward, self.config.max_reward)
        return combined


class EnsembleStrategy(LabelingStrategy):
    """Ensemble of base strategies with learned PRA-based weights."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        strategies: list[LabelingStrategy] | None = None,
    ):
        self.config = config or StrategyConfig()
        if strategies is not None:
            self._strategies = strategies
        else:
            from label.strategies.attribution import ReturnDecompositionStrategy
            from label.strategies.contrastive import TemporalContrastiveStrategy

            self._strategies = [
                ContrastiveDistanceStrategy(config),
                TemporalContrastiveStrategy(config),
                ProgressEstimatorStrategy(config),
                ReturnDecompositionStrategy(config),
            ]
        self._weights: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "ensemble_" + "_".join(s.name[:10] for s in self._strategies)

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        n_val = max(1, len(success_episodes) // 5)
        n_val = min(n_val, len(success_episodes) - 1) if len(success_episodes) > 1 else 0
        if n_val == 0:
            succ_train = success_episodes
            succ_val = success_episodes
        else:
            succ_train = success_episodes[:-n_val]
            succ_val = success_episodes[-n_val:]
        if len(fail_episodes) > n_val + 1:
            fail_train = fail_episodes[:-n_val]
            fail_val = fail_episodes[-n_val:]
        else:
            fail_train = fail_episodes
            fail_val = fail_episodes[-1:]

        scores = []
        for strat in self._strategies:
            try:
                strat.fit(succ_train, fail_train)
                succ_finals = [strat.label(ep, is_success=True)[-1] for ep in succ_val]
                fail_finals = [strat.label(ep, is_success=False)[-1] for ep in fail_val]
                correct = sum(1 for s in succ_finals for f in fail_finals if s > f)
                total = len(succ_finals) * len(fail_finals)
                scores.append(correct / total if total > 0 else 0.5)
            # Hybrid ensemble tolerates any single-strategy failure (bad fit,
            # numerical issues, shape mismatches, ...). We catch broadly *by
            # design* and zero that strategy's weight instead of aborting the
            # whole ensemble build.
            except Exception as e:  # noqa: BLE001 — intentional ensemble fallback
                logger.warning("Strategy %s failed: %s", strat.name, e)
                scores.append(0.0)

        for i, strat in enumerate(self._strategies):
            try:
                strat.fit(success_episodes, fail_episodes)
            except Exception as e:  # noqa: BLE001 — see rationale above
                logger.warning("Strategy %s full-data refit failed: %s, zeroing weight", strat.name, e)
                scores[i] = 0.0

        scores_arr = np.maximum(np.array(scores) - 0.5, 0)
        total = scores_arr.sum()
        self._weights = scores_arr / total if total > 0 else np.ones(len(scores_arr)) / len(scores_arr)
        logger.info("Ensemble weights: %s", dict(zip([s.name for s in self._strategies], self._weights, strict=True)))

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        all_rewards = []
        for strat in self._strategies:
            try:
                all_rewards.append(strat.label(episode, is_success))
            except Exception as e:  # noqa: BLE001 — see fit() rationale
                logger.warning("Strategy %s label failed: %s", strat.name, e)
                all_rewards.append(np.zeros(len(episode)))
        combined = sum(w * r for w, r in zip(self._weights, all_rewards, strict=True))
        if self.config.normalize_output:
            combined = normalize_rewards(combined, self.config.min_reward, self.config.max_reward)
        return combined
