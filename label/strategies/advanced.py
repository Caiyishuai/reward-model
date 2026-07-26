"""Advanced strategies: Potential-Based, Discriminative, Optimal Transport (E1-E3)."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from label.strategies._base import (
    LabelingStrategy,
    StrategyConfig,
    extract_states,
    normalize_rewards,
    smooth,
)


class PotentialBasedStrategy(LabelingStrategy):
    """Potential-based reward shaping with learned potential function (PBRS theory)."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        gamma: float = 0.99,
        n_neighbors: int = 10,
    ):
        self.config = config or StrategyConfig()
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self._scaler: StandardScaler | None = None
        self._all_states: np.ndarray | None = None
        self._potentials: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"potential_based_g{self.gamma}_k{self.n_neighbors}"

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        all_states = []
        all_values = []

        for ep in success_episodes:
            states = extract_states(ep)
            T = len(states)
            values = np.array([self.gamma ** (T - 1 - t) for t in range(T)])
            all_states.append(states)
            all_values.append(values)

        for ep in fail_episodes:
            states = extract_states(ep)
            T = len(states)
            values = np.array([-0.5 * self.gamma ** (T - 1 - t) for t in range(T)])
            all_states.append(states)
            all_values.append(values)

        from scipy.spatial import cKDTree

        self._scaler = StandardScaler()
        X = np.vstack(all_states)
        self._all_states = self._scaler.fit_transform(X)
        self._potentials = np.concatenate(all_values)
        self._tree = cKDTree(self._all_states)

    def _estimate_potential(self, state_norm: np.ndarray) -> float:
        k = min(self.n_neighbors, len(self._potentials))
        dists, idx = self._tree.query(state_norm, k=k)
        if k == 1:
            return float(self._potentials[idx])
        weights = np.exp(-dists)
        weights /= weights.sum() + 1e-8
        return float(np.sum(weights * self._potentials[idx]))

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        states = extract_states(episode)
        norm_states = self._scaler.transform(states)
        T = len(states)

        potentials = np.array([self._estimate_potential(s) for s in norm_states])

        shaped = np.zeros(T)
        for t in range(1, T):
            shaped[t] = self.gamma * potentials[t] - potentials[t - 1]

        cumulative = np.cumsum(shaped)
        rewards = smooth(cumulative, self.config.smooth_window)
        if self.config.normalize_output:
            rewards = normalize_rewards(rewards, self.config.min_reward, self.config.max_reward)
        return rewards


class DiscriminativeStrategy(LabelingStrategy):
    """Binary classifier confidence as reward signal."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        window: int = 5,
    ):
        self.config = config or StrategyConfig()
        self.window = window
        self._scaler: StandardScaler | None = None
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0

    @property
    def name(self) -> str:
        return f"discriminative_w{self.window}"

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        X_list = []
        y_list = []

        for ep, lbl in [(e, 1.0) for e in success_episodes] + [(e, 0.0) for e in fail_episodes]:
            states = extract_states(ep)
            state_dim = states.shape[1]
            for t in range(len(states)):
                start = max(0, t - self.window + 1)
                window_states = states[start : t + 1]
                features = np.concatenate(
                    [
                        np.mean(window_states, axis=0),
                        np.std(window_states, axis=0) if len(window_states) > 1 else np.zeros(state_dim),
                        [t / len(states)],
                    ]
                )
                X_list.append(features)
                y_list.append(lbl)

        X = np.array(X_list)
        y = np.array(y_list)

        self._scaler = StandardScaler()
        X_norm = self._scaler.fit_transform(X)

        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_norm, y)
        self._weights = clf.coef_[0]
        self._bias = clf.intercept_[0]

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        states = extract_states(episode)
        T = len(states)
        state_dim = states.shape[1]
        rewards = np.zeros(T)

        for t in range(T):
            start = max(0, t - self.window + 1)
            window_states = states[start : t + 1]
            features = np.concatenate(
                [
                    np.mean(window_states, axis=0),
                    np.std(window_states, axis=0) if len(window_states) > 1 else np.zeros(state_dim),
                    [t / T],
                ]
            )
            norm_feat = self._scaler.transform(features.reshape(1, -1))
            logit = float(norm_feat @ self._weights + self._bias)
            prob = 1.0 / (1.0 + np.exp(-logit))
            rewards[t] = prob

        rewards = smooth(rewards, self.config.smooth_window)
        if self.config.normalize_output:
            rewards = normalize_rewards(rewards, self.config.min_reward, self.config.max_reward)
        return rewards


class OptimalTransportStrategy(LabelingStrategy):
    """Reward based on Wasserstein distance to success trajectory distribution."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        n_time_bins: int = 20,
    ):
        self.config = config or StrategyConfig()
        self.n_time_bins = n_time_bins
        self._scaler: StandardScaler | None = None
        self._succ_bin_means: list[np.ndarray] | None = None
        self._succ_bin_stds: list[np.ndarray] | None = None
        self._fail_bin_means: list[np.ndarray] | None = None

    @property
    def name(self) -> str:
        return f"optimal_transport_b{self.n_time_bins}"

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        all_states = []
        for ep in success_episodes + fail_episodes:
            all_states.append(extract_states(ep))
        all_stacked = np.vstack(all_states)
        self._scaler = StandardScaler()
        self._scaler.fit(all_stacked)
        state_dim = all_stacked.shape[1]

        self._succ_bin_means = []
        self._succ_bin_stds = []
        for b in range(self.n_time_bins):
            bin_states = []
            for ep in success_episodes:
                states = self._scaler.transform(extract_states(ep))
                T = len(states)
                t_start = int(b / self.n_time_bins * T)
                t_end = int((b + 1) / self.n_time_bins * T)
                if t_start < T:
                    bin_states.append(states[t_start : max(t_end, t_start + 1)])
            if bin_states:
                combined = np.vstack(bin_states)
                self._succ_bin_means.append(np.mean(combined, axis=0))
                self._succ_bin_stds.append(np.std(combined, axis=0) + 1e-6)
            else:
                self._succ_bin_means.append(np.zeros(state_dim))
                self._succ_bin_stds.append(np.ones(state_dim))

        self._fail_bin_means = []
        for b in range(self.n_time_bins):
            bin_states = []
            for ep in fail_episodes:
                states = self._scaler.transform(extract_states(ep))
                T = len(states)
                t_start = int(b / self.n_time_bins * T)
                t_end = int((b + 1) / self.n_time_bins * T)
                if t_start < T:
                    bin_states.append(states[t_start : max(t_end, t_start + 1)])
            if bin_states:
                self._fail_bin_means.append(np.mean(np.vstack(bin_states), axis=0))
            else:
                self._fail_bin_means.append(np.zeros(state_dim))

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        states = self._scaler.transform(extract_states(episode))
        T = len(states)
        rewards = np.zeros(T)

        for t in range(T):
            b = min(int(t / T * self.n_time_bins), self.n_time_bins - 1)
            dist_succ = np.linalg.norm((states[t] - self._succ_bin_means[b]) / self._succ_bin_stds[b])
            dist_fail = np.linalg.norm((states[t] - self._fail_bin_means[b]) / self._succ_bin_stds[b])
            rewards[t] = np.exp(-0.3 * dist_succ) - 0.5 * np.exp(-0.3 * dist_fail)

        cumulative = np.cumsum(np.maximum(np.diff(rewards, prepend=rewards[0] - 0.01), 0))
        combined = 0.4 * rewards + 0.6 * cumulative

        combined = smooth(combined, self.config.smooth_window)
        if self.config.normalize_output:
            combined = normalize_rewards(combined, self.config.min_reward, self.config.max_reward)
        return combined
