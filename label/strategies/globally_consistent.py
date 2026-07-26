"""GloballyConsistentStrategy: label-free dense reward via time-aware classifier."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from label.strategies._base import (
    LabelingStrategy,
    StrategyConfig,
    extract_actions,
    extract_states,
    normalize_rewards,
    smooth,
)


class GloballyConsistentStrategy(LabelingStrategy):
    """Label-free dense reward via a single time-aware classifier.

    Architecture:
    1. Features: [state, velocity, acceleration, relative_time, cum_displacement]
    2. A single HistGradientBoosting classifier: P(success | features)
    3. Reward = weighted_cumulative_mean(P) * (0.7 + 0.3 * t/T)

    No is_success branching in label() — discrimination is purely data-driven.
    """

    def __init__(
        self,
        config: StrategyConfig | None = None,
        time_decay: float = 0.9,
        margin: float = 1.0,
        **_kw: object,
    ):
        self.config = config or StrategyConfig()
        self.time_decay = time_decay
        self.margin = margin
        self._scaler: StandardScaler | None = None

    @property
    def name(self) -> str:
        return f"globally_consistent_d{self.time_decay}_m{self.margin}"

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

        from sklearn.ensemble import HistGradientBoostingClassifier

        self._feat_scaler = StandardScaler()

        X_list, y_list = [], []
        traj_X, traj_y = [], []
        for ep, lbl in [(e, 1) for e in success_episodes] + [(e, 0) for e in fail_episodes]:
            feats = self._build_step_features(ep)
            X_list.append(feats)
            y_list.extend([lbl] * len(feats))
            traj_X.append(self._trajectory_summary(feats))
            traj_y.append(lbl)

        X_all = np.vstack(X_list)
        self._feat_scaler.fit(X_all)
        X_norm = self._feat_scaler.transform(X_all)
        y_arr = np.array(y_list)

        self._classifier = HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=5,
            learning_rate=0.08,
            min_samples_leaf=15,
            l2_regularization=5.0,
            random_state=42,
        )
        self._classifier.fit(X_norm, y_arr)

        self._traj_scaler = StandardScaler()
        traj_X_arr = np.vstack(traj_X)
        self._traj_scaler.fit(traj_X_arr)
        traj_X_norm = self._traj_scaler.transform(traj_X_arr)
        self._traj_clf = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            l2_regularization=3.0,
            random_state=42,
        )
        self._traj_clf.fit(traj_X_norm, np.array(traj_y))

        from scipy.spatial import cKDTree

        self._knn_tree = cKDTree(X_norm)
        self._knn_labels = y_arr
        self._knn_k = min(5, len(X_norm))

    def _build_step_features(self, episode: list[dict]) -> np.ndarray:
        """Per-step features: [state, vel, acc, action, act_vel, rel_time, cum_disp]."""
        states = self._scaler.transform(extract_states(episode))
        actions = extract_actions(episode)
        T = len(states)
        rel_t = np.linspace(0, 1, T).reshape(-1, 1)
        if T < 2:
            sz = np.zeros_like(states)
            az = np.zeros_like(actions)
            return np.hstack([states, sz, sz, actions, az, rel_t, np.zeros((T, 1))])
        vel = np.diff(states, axis=0, prepend=states[0:1])
        acc = np.diff(vel, axis=0, prepend=vel[0:1])
        act_vel = np.diff(actions, axis=0, prepend=actions[0:1])
        step_disp = np.linalg.norm(vel, axis=1)
        cum_disp = (np.cumsum(step_disp) / (step_disp.sum() + 1e-8)).reshape(-1, 1)
        return np.hstack([states, vel, acc, actions, act_vel, rel_t, cum_disp])

    @staticmethod
    def _trajectory_summary(step_feats: np.ndarray) -> np.ndarray:
        """Compress a trajectory into a fixed-length summary vector."""
        T = len(step_feats)
        diffs = np.diff(step_feats, axis=0) if T > 1 else np.zeros((1, step_feats.shape[1]))
        speed = np.linalg.norm(diffs, axis=1)
        q1 = T // 4
        q3 = 3 * T // 4
        return np.concatenate(
            [
                step_feats.mean(axis=0),
                step_feats.std(axis=0),
                step_feats[-1] - step_feats[0],
                step_feats[len(step_feats) // 2],
                step_feats[q1],
                step_feats[q3],
                [
                    speed.mean(),
                    speed.std(),
                    speed.max(),
                    float(np.sum(np.diff(np.sign(diffs.sum(axis=1))) != 0)),
                    float(T),
                ],
            ]
        )

    def label(self, episode: list[dict], is_success: bool) -> np.ndarray:
        feats = self._build_step_features(episode)
        feats_norm = self._feat_scaler.transform(feats)
        T = len(feats)

        hgbc_prob = self._classifier.predict_proba(feats_norm)[:, 1]

        dists, idxs = self._knn_tree.query(feats_norm, k=self._knn_k)
        knn_prob = np.empty(T)
        for t in range(T):
            w = 1.0 / (dists[t] + 1e-8)
            knn_prob[t] = np.sum(w * self._knn_labels[idxs[t]]) / np.sum(w)

        traj_feat = self._trajectory_summary(feats)
        traj_norm = self._traj_scaler.transform(traj_feat.reshape(1, -1))
        traj_prior = self._traj_clf.predict_proba(traj_norm)[0, 1]

        prob_succ = 0.3 * hgbc_prob + 0.2 * knn_prob + 0.5 * traj_prior

        weights = np.linspace(0.5, 1.5, T)
        w_cum_mean = np.cumsum(prob_succ * weights) / np.cumsum(weights)
        relative_time = np.linspace(0, 1, T)

        rewards = w_cum_mean * (0.7 + 0.3 * relative_time)

        rewards = smooth(rewards, self.config.smooth_window)
        if self.config.normalize_output:
            rewards = normalize_rewards(rewards, self.config.min_reward, self.config.max_reward)
        return rewards
