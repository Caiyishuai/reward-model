"""Attribution-family labeling strategies (C1: Progress, C2: Return Decomposition)."""

import logging

import numpy as np
from sklearn.preprocessing import StandardScaler

from label.strategies._base import (
    LabelingStrategy,
    StrategyConfig,
    extract_states,
    normalize_rewards,
    smooth,
)

logger = logging.getLogger(__name__)


class ProgressEstimatorStrategy(LabelingStrategy):
    """Per-step reward based on learned progress toward task completion."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        fail_ceiling: float = 0.4,
    ):
        self.config = config or StrategyConfig()
        self.fail_ceiling = fail_ceiling
        self._scaler: StandardScaler | None = None
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0

    @property
    def name(self) -> str:
        return f"progress_estimator_fc{self.fail_ceiling}"

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        all_states = []
        all_progress = []

        rng = np.random.default_rng(42)
        for ep in success_episodes:
            states = extract_states(ep)
            T = len(states)
            progress = np.linspace(0.1, 1.0, T)
            all_states.append(states)
            all_progress.append(progress)

        for ep in fail_episodes:
            states = extract_states(ep)
            T = len(states)
            ceiling = self.fail_ceiling * (0.5 + rng.random())
            t_peak = int(T * (0.3 + 0.3 * rng.random()))
            progress = np.zeros(T)
            progress[:t_peak] = np.linspace(0.05, ceiling, t_peak)
            progress[t_peak:] = np.linspace(ceiling, ceiling * 0.8, T - t_peak)
            all_states.append(states)
            all_progress.append(progress)

        X = np.vstack(all_states)
        y = np.concatenate(all_progress)

        self._scaler = StandardScaler()
        X_norm = self._scaler.fit_transform(X)

        from sklearn.linear_model import Ridge

        reg = Ridge(alpha=1.0)
        reg.fit(X_norm, y)
        self._weights = reg.coef_
        self._bias = reg.intercept_

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        states = extract_states(episode)
        norm_states = self._scaler.transform(states)
        progress = norm_states @ self._weights + self._bias
        progress = np.clip(progress, 0, 1)

        T = len(progress)
        time_bonus = np.linspace(0, 0.3, T) if is_success else np.linspace(0, 0.05, T)
        raw_reward = progress + time_bonus

        if is_success:
            raw_reward = raw_reward * self.config.max_reward
        else:
            raw_reward = raw_reward * self.config.max_reward * self.fail_ceiling

        rewards = smooth(raw_reward, self.config.smooth_window)
        if self.config.normalize_output:
            rewards = normalize_rewards(
                rewards,
                self.config.min_reward,
                self.config.max_reward,
            )
        return rewards


class ReturnDecompositionStrategy(LabelingStrategy):
    """RUDDER-inspired: train sequence model to predict episode return,
    then use prediction changes as per-step reward."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        hidden_dim: int = 32,
        lr: float = 0.01,
        n_iters: int = 200,
        bptt_len: int = 30,
    ):
        self.config = config or StrategyConfig()
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_iters = n_iters
        self.bptt_len = bptt_len
        self._scaler: StandardScaler | None = None
        self._W_ih: np.ndarray | None = None
        self._W_hh: np.ndarray | None = None
        self._W_ho: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"return_decomp_h{self.hidden_dim}_i{self.n_iters}"

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        all_states = []
        for ep in success_episodes + fail_episodes:
            all_states.append(extract_states(ep))

        self._scaler = StandardScaler()
        self._scaler.fit(np.vstack(all_states))

        input_dim = np.vstack(all_states).shape[1]
        h = self.hidden_dim

        rng = np.random.default_rng(42)
        scale = 0.01
        self._W_ih = rng.normal(0, scale, (input_dim, h))
        self._W_hh = rng.normal(0, scale, (h, h)) * 0.1
        self._W_ho = rng.normal(0, scale, (h, 1))

        episodes = []
        returns = []
        for ep in success_episodes:
            episodes.append(self._scaler.transform(extract_states(ep)))
            returns.append(1.0)
        for ep in fail_episodes:
            episodes.append(self._scaler.transform(extract_states(ep)))
            returns.append(0.0)

        for iteration in range(self.n_iters):
            total_loss = 0.0
            for ep_states, ep_return in zip(episodes, returns, strict=True):
                pred = self._forward_sequence(ep_states)
                error = pred - ep_return
                total_loss += error**2

                T = len(ep_states)
                h_states = np.zeros((T + 1, h))
                for t in range(T):
                    h_states[t + 1] = np.tanh(ep_states[t] @ self._W_ih + h_states[t] @ self._W_hh)

                d_out = 2 * error * self._W_ho.T
                d_h = d_out * (1 - h_states[-1] ** 2)

                grad_W_ho = h_states[-1].reshape(-1, 1) * (2 * error)
                grad_W_ih = np.zeros_like(self._W_ih)
                grad_W_hh = np.zeros_like(self._W_hh)

                for t in range(T - 1, max(T - self.bptt_len, -1), -1):
                    deriv = 1 - h_states[t + 1] ** 2
                    grad_W_ih += np.outer(ep_states[t], d_h.flatten() * deriv)
                    grad_W_hh += np.outer(h_states[t], d_h.flatten() * deriv)
                    d_h = (d_h.flatten() * deriv) @ self._W_hh.T

                clip = 1.0
                for grad in [grad_W_ih, grad_W_hh, grad_W_ho]:
                    norm = np.linalg.norm(grad)
                    if norm > clip:
                        grad *= clip / norm

                self._W_ih -= self.lr * grad_W_ih / len(episodes)
                self._W_hh -= self.lr * grad_W_hh / len(episodes)
                self._W_ho -= self.lr * grad_W_ho / len(episodes)

            if iteration % 50 == 0:
                logger.info(
                    "ReturnDecomp iter %d, loss=%.4f",
                    iteration,
                    total_loss / len(episodes),
                )

    def _forward_sequence(self, states: np.ndarray) -> float:
        """Run RNN forward and return final prediction."""
        h = np.zeros(self.hidden_dim)
        for t in range(len(states)):
            h = np.tanh(states[t] @ self._W_ih + h @ self._W_hh)
        return float((h @ self._W_ho).item())

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        states = self._scaler.transform(extract_states(episode))
        T = len(states)

        h = np.zeros(self.hidden_dim)
        predictions = np.zeros(T)

        for t in range(T):
            h = np.tanh(states[t] @ self._W_ih + h @ self._W_hh)
            predictions[t] = float((h @ self._W_ho).item())

        step_contributions = np.diff(predictions, prepend=0)
        cumulative_reward = np.cumsum(np.abs(step_contributions)) * np.sign(predictions[-1] + 1e-8)

        time_progress = np.linspace(0, 1, T)
        if is_success:
            rewards = 0.6 * cumulative_reward + 0.4 * time_progress
        else:
            rewards = 0.6 * cumulative_reward + 0.4 * (time_progress * 0.3)

        rewards = smooth(rewards, self.config.smooth_window)
        if self.config.normalize_output:
            rewards = normalize_rewards(
                rewards,
                self.config.min_reward,
                self.config.max_reward,
            )
        return rewards
