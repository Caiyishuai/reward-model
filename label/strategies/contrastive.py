"""Contrastive-family labeling strategies (B1, B2)."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from label.strategies._base import (
    LabelingStrategy,
    StrategyConfig,
    extract_states,
    normalize_rewards,
    smooth,
)


class ContrastiveDistanceStrategy(LabelingStrategy):
    """Reward = proximity to success end-states - proximity to fail end-states.

    Uses temporal weighting: later states in success trajectories and
    late states in fail trajectories receive higher importance as cluster seeds.
    Cumulative reward ensures monotonic growth for well-progressing trajectories.
    """

    def __init__(
        self,
        config: StrategyConfig | None = None,
        n_clusters: int = 20,
        distance_weight: float = 0.5,
        temporal_weight: float = 0.9,
    ):
        self.config = config or StrategyConfig()
        self.n_clusters = n_clusters
        self.distance_weight = distance_weight
        self.temporal_weight = temporal_weight
        self._scaler: StandardScaler | None = None
        self._succ_end_states: np.ndarray | None = None
        self._fail_end_states: np.ndarray | None = None
        self._succ_centroids: np.ndarray | None = None
        self._fail_centroids: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"contrastive_dist_k{self.n_clusters}_w{self.distance_weight}_t{self.temporal_weight}"

    def fit(
        self,
        success_episodes: list[list[dict]],
        fail_episodes: list[list[dict]],
    ) -> None:
        if not success_episodes or not fail_episodes:
            raise ValueError("ContrastiveDistanceStrategy requires both success and fail episodes")
        from sklearn.cluster import MiniBatchKMeans

        succ_states = np.vstack([extract_states(ep) for ep in success_episodes])
        fail_states = np.vstack([extract_states(ep) for ep in fail_episodes])

        self._scaler = StandardScaler()
        all_states = np.vstack([succ_states, fail_states])
        self._scaler.fit(all_states)

        succ_weighted = []
        for ep in success_episodes:
            s = extract_states(ep)
            T = len(s)
            weights = np.array([self.temporal_weight ** (T - 1 - t) for t in range(T)])
            for t in range(T):
                n_copies = max(1, int(weights[t] * 5))
                succ_weighted.extend([s[t]] * n_copies)
        succ_weighted = np.array(succ_weighted)

        fail_weighted = []
        for ep in fail_episodes:
            s = extract_states(ep)
            T = len(s)
            weights = np.array([self.temporal_weight ** (T - 1 - t) for t in range(T)])
            for t in range(T):
                n_copies = max(1, int(weights[t] * 5))
                fail_weighted.extend([s[t]] * n_copies)
        fail_weighted = np.array(fail_weighted)

        succ_norm = self._scaler.transform(succ_weighted)
        fail_norm = self._scaler.transform(fail_weighted)

        k_succ = max(1, min(self.n_clusters, len(succ_norm) // 3))
        k_fail = max(1, min(self.n_clusters, len(fail_norm) // 3))

        km_succ = MiniBatchKMeans(n_clusters=k_succ, random_state=42, n_init=3)
        km_fail = MiniBatchKMeans(n_clusters=k_fail, random_state=42, n_init=3)
        km_succ.fit(succ_norm)
        km_fail.fit(fail_norm)

        self._succ_centroids = km_succ.cluster_centers_
        self._fail_centroids = km_fail.cluster_centers_

        self._succ_end_states = self._scaler.transform(np.array([extract_states(ep)[-1] for ep in success_episodes]))
        self._fail_end_states = self._scaler.transform(np.array([extract_states(ep)[-1] for ep in fail_episodes]))

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        states = extract_states(episode)
        norm_states = self._scaler.transform(states)

        dist_to_succ = np.min(
            np.linalg.norm(
                norm_states[:, None, :] - self._succ_centroids[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        dist_to_fail = np.min(
            np.linalg.norm(
                norm_states[:, None, :] - self._fail_centroids[None, :, :],
                axis=2,
            ),
            axis=1,
        )

        succ_sim = np.exp(-self.distance_weight * dist_to_succ)
        fail_sim = np.exp(-self.distance_weight * dist_to_fail)
        contrastive = succ_sim - fail_sim

        dist_to_goal = np.mean(
            np.linalg.norm(
                norm_states[:, None, :] - self._succ_end_states[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        goal_proximity = np.exp(-0.3 * dist_to_goal)

        raw_reward = 0.6 * contrastive + 0.4 * goal_proximity
        cumulative = np.cumsum(np.maximum(np.diff(raw_reward, prepend=raw_reward[0] - 0.01), 0))
        combined = 0.5 * raw_reward + 0.5 * cumulative

        rewards = smooth(combined, self.config.smooth_window)
        if self.config.normalize_output:
            rewards = normalize_rewards(
                rewards,
                self.config.min_reward,
                self.config.max_reward,
            )
        return rewards


class TemporalContrastiveStrategy(LabelingStrategy):
    """Time-weighted contrastive reward inspired by TW-CRL."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        time_decay: float = 0.95,
        n_neighbors: int = 5,
    ):
        self.config = config or StrategyConfig()
        self.time_decay = time_decay
        self.n_neighbors = n_neighbors
        self._scaler: StandardScaler | None = None
        self._succ_weighted_states: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._fail_weighted_states: list[tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def name(self) -> str:
        return f"temporal_contrastive_d{self.time_decay}_k{self.n_neighbors}"

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

        self._succ_weighted_states = []
        for ep in success_episodes:
            states = self._scaler.transform(extract_states(ep))
            T = len(states)
            weights = np.array([self.time_decay ** (T - 1 - t) for t in range(T)])
            weights /= weights.sum() + 1e-8
            self._succ_weighted_states.append((states, weights))

        self._fail_weighted_states = []
        for ep in fail_episodes:
            states = self._scaler.transform(extract_states(ep))
            T = len(states)
            weights = np.array([self.time_decay ** (T - 1 - t) for t in range(T)])
            weights /= weights.sum() + 1e-8
            self._fail_weighted_states.append((states, weights))

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        states = self._scaler.transform(extract_states(episode))
        T = len(states)
        rewards = np.zeros(T)

        for t in range(T):
            succ_scores = []
            for s_states, s_weights in self._succ_weighted_states:
                dists = np.linalg.norm(s_states - states[t], axis=1)
                top_k = min(self.n_neighbors, len(dists))
                idx = (
                    np.arange(len(dists))
                    if top_k >= len(dists)
                    else np.argpartition(dists, top_k)[:top_k]
                )
                weighted_sim = np.sum(s_weights[idx] * np.exp(-dists[idx]))
                succ_scores.append(weighted_sim)

            fail_scores = []
            for f_states, f_weights in self._fail_weighted_states:
                dists = np.linalg.norm(f_states - states[t], axis=1)
                top_k = min(self.n_neighbors, len(dists))
                idx = (
                    np.arange(len(dists))
                    if top_k >= len(dists)
                    else np.argpartition(dists, top_k)[:top_k]
                )
                weighted_sim = np.sum(f_weights[idx] * np.exp(-dists[idx]))
                fail_scores.append(weighted_sim)

            rewards[t] = np.mean(succ_scores) - np.mean(fail_scores)

        rewards = smooth(rewards, self.config.smooth_window)
        if self.config.normalize_output:
            rewards = normalize_rewards(
                rewards,
                self.config.min_reward,
                self.config.max_reward,
            )
        return rewards
