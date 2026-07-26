"""HMM-based labeling strategy (baseline)."""

import numpy as np

from label.strategies._base import LabelingStrategy, StrategyConfig, normalize_rewards


class HMMBaselineStrategy(LabelingStrategy):
    """Current HMM-based labeling from auto_label.py as baseline."""

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()
        self._discovery = None
        self._targets = None

    @property
    def name(self) -> str:
        return "hmm_baseline"

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        from label.auto_label import StageDiscovery

        self._discovery = StageDiscovery(use_force_dynamics=False, n_restarts=10)
        self._discovery.fit(success_episodes)
        self._targets = self._discovery.compute_stage_targets(success_episodes)

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        from label.auto_label import compute_dense_reward

        _, state_seq = self._discovery.decode(episode)
        rewards = compute_dense_reward(episode, self._targets, state_seq)
        if self.config.normalize_output:
            rewards = normalize_rewards(
                rewards,
                self.config.min_reward,
                self.config.max_reward,
            )
        return rewards
