"""Base classes and shared utilities for labeling strategies."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Shared configuration for labeling strategies."""

    success_bonus: float = 1.0
    fail_penalty: float = 0.0
    smooth_window: int = 15
    normalize_output: bool = True
    max_reward: float = 6.0
    min_reward: float = 0.0


class LabelingStrategy(ABC):
    """Base interface for all reward labeling strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this strategy."""

    @abstractmethod
    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        """Learn from demonstration data."""

    @abstractmethod
    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        """Assign per-step rewards to one episode. Returns array of shape [T]."""

    def label_all(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> tuple[list[list[dict]], list[list[dict]]]:
        """Label all episodes in-place and return them."""
        for ep in success_episodes:
            rewards = self.label(ep, is_success=True)
            for j, r in enumerate(rewards):
                ep[j]["rewards"] = float(r)

        for ep in fail_episodes:
            try:
                rewards = self.label(ep, is_success=False)
            # Failure labeling is best-effort; a broken episode should not
            # kill the full pipeline. Catch broadly on purpose.
            except Exception as e:  # noqa: BLE001 — intentional pipeline tolerance
                logger.warning("Labeling failed for fail episode: %s, assigning 0", e)
                rewards = np.zeros(len(ep))
            for j, r in enumerate(rewards):
                ep[j]["rewards"] = float(r)

        return success_episodes, fail_episodes


def extract_states(episode: list[dict]) -> np.ndarray:
    """Extract [T, D] state array from episode."""
    states = np.array([step["observations"]["state"] for step in episode])
    if states.ndim > 2:
        states = states.reshape(len(episode), -1)
    return states


def extract_actions(episode: list[dict]) -> np.ndarray:
    """Extract [T, A] action array from episode."""
    return np.array([step["actions"] for step in episode])


def smooth(rewards: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average smoothing. Output length always matches input."""
    if window <= 1 or len(rewards) < 2:
        return rewards
    n = len(rewards)
    cumsum = np.concatenate([[0.0], np.cumsum(rewards)])
    indices = np.arange(1, n + 1)
    starts = np.maximum(indices - window, 0)
    divisors = indices - starts
    return (cumsum[indices] - cumsum[starts]) / divisors


def normalize_rewards(
    rewards: np.ndarray,
    min_r: float = 0.0,
    max_r: float = 6.0,
) -> np.ndarray:
    """Scale rewards to [min_r, max_r] range."""
    r_min, r_max = rewards.min(), rewards.max()
    if r_max - r_min < 1e-8:
        return np.full_like(rewards, (min_r + max_r) / 2)
    normalized = (rewards - r_min) / (r_max - r_min)
    return normalized * (max_r - min_r) + min_r
