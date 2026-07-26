"""Automatic reward labeling via Gaussian HMM stage discovery.

Pipeline: raw demos -> HMM phase segmentation -> Mahalanobis-distance dense reward
-> quality evaluation -> LeRobot export.

Key features:
- No hand-crafted keypoint targets required
- Automatically discovers task stages from success demonstrations
- Adaptive stage count via BIC model selection
- Robust HMM fitting with multiple random restarts
- Multi-signal reward: position + force + force dynamics + gripper + temporal penalty
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hmmlearn import hmm
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler

from data.common import (
    BASE_DIR,
    get_episodes,
    get_task,
    load_pickle,
    save_pickle,
)

sns.set_theme("paper", style="whitegrid")


# ==========================================
# Stage Discovery (HMM)
# ==========================================
@dataclass
class StageTarget:
    """Canonical target statistics for one task stage."""

    pos_mean: np.ndarray
    pos_cov_inv: np.ndarray
    force_mean: np.ndarray
    force_cov_inv: np.ndarray
    gripper_mean: float
    gripper_std: float
    is_contact: bool
    max_duration: float
    dforce_mean: np.ndarray | None = None
    dforce_cov_inv: np.ndarray | None = None


def _safe_inv(cov: np.ndarray) -> np.ndarray:
    """Numerically stable matrix inversion with adaptive regularization."""
    trace_scale = max(1e-5, 0.01 * np.trace(cov) / cov.shape[0])
    cov = cov + np.eye(cov.shape[0]) * trace_scale
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov)


def _safe_cov_inv(data: np.ndarray) -> np.ndarray:
    """Compute inverse covariance; fall back to identity for few samples."""
    d = data.shape[1] if data.ndim == 2 else 1
    if len(data) < d + 1:
        logging.warning("Too few samples (%d) for %dD covariance, using Euclidean", len(data), d)
        return np.eye(d)
    return _safe_inv(np.cov(data, rowvar=False))


def _extract_features(
    episode: list[dict],
    use_force_dynamics: bool,
    force_slice: slice | None = slice(1, 4),
) -> np.ndarray:
    """Extract state features, optionally with force gradient."""
    states = np.array([step["observations"]["state"] for step in episode])
    if states.ndim > 2:
        states = states.reshape(len(episode), -1)
    if not use_force_dynamics or force_slice is None:
        return states
    forces = states[:, force_slice]
    d_force = np.gradient(forces, axis=0)
    return np.hstack([states, d_force])


def _build_left_right_hmm(
    n_components: int,
    random_state: int,
    covariance_type: str = "full",
    n_iter: int = 200,
) -> hmm.GaussianHMM:
    """Create a left-to-right constrained GaussianHMM.

    ``params="mc"`` (no ``t``) keeps EM from learning arbitrary transitions
    and destroying the hand-crafted left-to-right sparsity pattern, which
    would otherwise let the model go backwards across stages.
    """
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=1e-4,
        min_covar=1e-3,
        random_state=random_state,
        init_params="",
        params="mc",
    )

    model.startprob_ = np.zeros(n_components)
    model.startprob_[0] = 1.0

    trans = np.zeros((n_components, n_components))
    for k in range(n_components):
        if k == n_components - 1:
            trans[k, k] = 1.0
        else:
            trans[k, k] = 0.5
            trans[k, k + 1] = 0.5
    model.transmat_ = trans

    return model


class StageDiscovery:
    """HMM-based task stage segmentation with optional auto stage count via BIC."""

    def __init__(
        self,
        use_force_dynamics: bool = False,
        n_restarts: int = 10,
        n_iter: int = 200,
        use_prototype_stages: bool = False,
        pos_slice: slice = slice(4, 7),
        force_slice: slice | None = slice(1, 4),
        gripper_idx: int = 0,
    ):
        self.use_force_dynamics = use_force_dynamics
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.use_prototype_stages = use_prototype_stages
        self.pos_slice = pos_slice
        self.force_slice = force_slice
        self.gripper_idx = gripper_idx
        self.scaler: StandardScaler | None = None
        self.model: hmm.GaussianHMM | None = None
        self.n_components: int = 0
        self.feature_mask: np.ndarray | None = None
        self.covariance_type = "full"
        self.prototype_centers: np.ndarray | None = None

    def fit(self, episodes: list[list[dict]], n_stages: int | None = None, max_stages: int = 5) -> None:
        """Train HMM on success episodes.

        If n_stages is None, select best model via BIC over [2, max_stages].
        """
        all_features = []
        lengths = []
        for ep in episodes:
            feat = _extract_features(ep, self.use_force_dynamics, self.force_slice)
            all_features.append(feat)
            lengths.append(len(feat))

        # hmmlearn's sufficient-statistics matmuls can overflow in float32
        # even for standardized long trajectories; use float64 for EM.
        training_data = np.asarray(np.vstack(all_features), dtype=np.float64)
        original_feature_count = training_data.shape[1]
        feature_variance = np.var(training_data, axis=0)
        self.feature_mask = np.isfinite(feature_variance) & (feature_variance > 1e-10)
        if not np.any(self.feature_mask):
            raise ValueError("All HMM features are constant or non-finite")
        dropped = int((~self.feature_mask).sum())
        if dropped:
            print(f"[HMM] Dropping {dropped}/{len(self.feature_mask)} constant features")
        training_data = training_data[:, self.feature_mask]
        # Full covariance is fragile and cubic in feature count.  MetaWorld's
        # 39-D observation contains strongly correlated current/previous state
        # blocks; diagonal covariance is the stable production choice there.
        self.covariance_type = (
            "diag" if original_feature_count > 32 or training_data.shape[1] > 16 else "full"
        )
        print(f"[HMM] Features={training_data.shape[1]}, covariance={self.covariance_type}")

        self.scaler = StandardScaler()
        norm_data = self.scaler.fit_transform(training_data)

        if self.use_prototype_stages:
            self.n_components = n_stages or max_stages
            assignments = np.concatenate(
                [
                    np.minimum(
                        np.arange(length, dtype=np.int64) * self.n_components // max(length, 1),
                        self.n_components - 1,
                    )
                    for length in lengths
                ]
            )
            self.prototype_centers = np.stack(
                [norm_data[assignments == stage].mean(axis=0) for stage in range(self.n_components)]
            )
            print(f"[Prototype] Final model: {self.n_components} left-to-right stages")
            return

        if n_stages is not None:
            self.model = self._fit_single(norm_data, lengths, n_stages)
            self.n_components = n_stages
        else:
            self.model, self.n_components = self._fit_bic(norm_data, lengths, max_stages)

        print(f"[HMM] Final model: {self.n_components} stages, LogL={self.model.monitor_.history[-1]:.1f}")

    def _fit_single(self, data: np.ndarray, lengths: list[int], n_components: int) -> hmm.GaussianHMM:
        """Fit one HMM with multiple random restarts, return best."""
        best_model = None
        best_score = -np.inf

        for i in range(self.n_restarts):
            candidate = _build_left_right_hmm(
                n_components,
                random_state=i + 42,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
            )
            try:
                # Deterministic progress-bin initialization avoids KMeans
                # degeneracy on trajectories with repeated/constant state
                # blocks (notably MetaWorld goal-observable observations).
                assignments = np.concatenate(
                    [
                        np.minimum(
                            np.arange(length, dtype=np.int64) * n_components // max(length, 1),
                            n_components - 1,
                        )
                        for length in lengths
                    ]
                )
                rng = np.random.default_rng(i + 42)
                means = []
                covariances = []
                for stage in range(n_components):
                    stage_data = data[assignments == stage]
                    if len(stage_data) == 0:
                        stage_data = data
                    means.append(stage_data.mean(axis=0) + rng.normal(0.0, 1e-4, data.shape[1]))
                    if self.covariance_type == "diag":
                        covariances.append(np.maximum(stage_data.var(axis=0), 1e-3))
                    else:
                        covariance = np.atleast_2d(np.cov(stage_data, rowvar=False))
                        covariance += np.eye(data.shape[1]) * 1e-3
                        covariances.append(covariance)
                candidate.means_ = np.asarray(means)
                candidate.covars_ = np.asarray(covariances)
                candidate.fit(data, lengths)
                score = candidate.monitor_.history[-1]
                if score > best_score:
                    best_score = score
                    best_model = candidate
            except (ValueError, np.linalg.LinAlgError) as e:
                logging.debug("HMM restart %d failed: %s", i, e)
                continue

        if best_model is None:
            raise RuntimeError(f"HMM fitting failed for all {self.n_restarts} restarts (k={n_components})")
        return best_model

    def _fit_bic(self, data: np.ndarray, lengths: list[int], max_stages: int) -> tuple[hmm.GaussianHMM, int]:
        """Select stage count by BIC over [2, max_stages]."""
        best_model = None
        best_bic = np.inf
        best_k = 2
        n_samples = len(data)

        print(f"[HMM] Auto-selecting stages (2..{max_stages}) via BIC...")

        for k in range(2, max_stages + 1):
            try:
                model = self._fit_single(data, lengths, k)
                log_likelihood = model.score(data, lengths)
                d = data.shape[1]
                covariance_params = k * d if self.covariance_type == "diag" else k * d * (d + 1) // 2
                n_params = (k - 1) + k * d + covariance_params
                bic = -2 * log_likelihood + n_params * np.log(n_samples)
                print(f"  k={k}: BIC={bic:.1f}, LogL={log_likelihood:.1f}")

                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_k = k
            except RuntimeError:
                print(f"  k={k}: fitting failed, skipping")
                continue

        if best_model is None:
            raise RuntimeError("BIC model selection failed for all k values")

        print(f"[HMM] Selected k={best_k} (BIC={best_bic:.1f})")
        return best_model, best_k

    def decode(self, episode: list[dict]) -> tuple[list[int], np.ndarray]:
        """Decode episode into keyframe indices and state sequence."""
        if self.scaler is None or self.feature_mask is None:
            raise RuntimeError("Call fit() before decode()")
        feat = _extract_features(episode, self.use_force_dynamics, self.force_slice)
        feat = np.asarray(feat[:, self.feature_mask], dtype=np.float64)
        norm_feat = self.scaler.transform(feat)
        if self.use_prototype_stages:
            if self.prototype_centers is None:
                raise RuntimeError("Prototype centers are unavailable")
            costs = np.sum((norm_feat[:, None, :] - self.prototype_centers[None, :, :]) ** 2, axis=2)
            steps, stages = costs.shape
            dynamic = np.full((steps, stages), np.inf)
            back = np.zeros((steps, stages), dtype=np.int64)
            dynamic[0, 0] = costs[0, 0]
            for t in range(1, steps):
                dynamic[t, 0] = dynamic[t - 1, 0] + costs[t, 0]
                for stage in range(1, stages):
                    previous = stage if dynamic[t - 1, stage] <= dynamic[t - 1, stage - 1] else stage - 1
                    dynamic[t, stage] = dynamic[t - 1, previous] + costs[t, stage]
                    back[t, stage] = previous
            state_seq = np.zeros(steps, dtype=np.int64)
            state_seq[-1] = int(np.argmin(dynamic[-1]))
            for t in range(steps - 1, 0, -1):
                state_seq[t - 1] = back[t, state_seq[t]]
        else:
            if self.model is None:
                raise RuntimeError("HMM model is unavailable")
            _, state_seq = self.model.decode(norm_feat)

        change_pts = np.where(state_seq[1:] != state_seq[:-1])[0] + 1
        keyframes = sorted(set([0] + list(change_pts) + [len(episode) - 1]))
        return keyframes, state_seq

    def compute_stage_targets(self, episodes: list[list[dict]], contact_threshold: float = 1.0) -> list[StageTarget]:
        """Extract canonical per-stage target distributions from success demos."""
        from collections import Counter

        all_keyframes = []
        all_state_seqs = []
        for ep in episodes:
            kf, ss = self.decode(ep)
            all_keyframes.append(kf)
            all_state_seqs.append(ss)

        # Precompute central-difference force gradients per episode so that
        # stage target statistics match compute_dense_reward, which also uses
        # np.gradient. The previous forward-difference implementation produced
        # a distribution mismatch that biased the Mahalanobis distance.
        ep_dforce_traces: list[np.ndarray | None] = [None] * len(episodes)
        if self.use_force_dynamics and self.force_slice is not None:
            for idx, ep in enumerate(episodes):
                ep_states = np.array([step["observations"]["state"] for step in ep])
                if ep_states.ndim > 2:
                    ep_states = ep_states.reshape(len(ep), -1)
                ep_forces = ep_states[:, self.force_slice]
                ep_dforce_traces[idx] = np.gradient(ep_forces, axis=0)

        # Tie-break deterministically: when two keyframe counts are equally
        # frequent, prefer the larger one (more stages). `max(set(...))` used
        # set iteration order, which is not guaranteed to be stable.
        kf_lengths = [len(k) for k in all_keyframes]
        len_counts = Counter(kf_lengths)
        common_len = max(len_counts, key=lambda x: (len_counts[x], x))

        stage_durations: dict[int, list[int]] = {i: [] for i in range(self.n_components)}
        for ss in all_state_seqs:
            unique, counts = np.unique(ss, return_counts=True)
            for state, count in zip(unique, counts, strict=True):
                if state < self.n_components:
                    stage_durations[state].append(int(count))

        targets: list[StageTarget] = []
        for stage_idx in range(1, common_len):
            pos_pts, force_pts, grip_pts, dforce_pts = [], [], [], []
            segment_hmm_states: list[int] = []

            for i, kf in enumerate(all_keyframes):
                if len(kf) != common_len:
                    continue
                k = kf[stage_idx]
                state = episodes[i][k]["observations"]["state"].squeeze()
                pos_pts.append(state[self.pos_slice])
                if self.force_slice is not None:
                    force_pts.append(state[self.force_slice])
                grip_pts.append(float(state[self.gripper_idx]))
                segment_hmm_states.append(int(all_state_seqs[i][k]))

                if ep_dforce_traces[i] is not None:
                    dforce_pts.append(ep_dforce_traces[i][k])

            if not pos_pts:
                continue

            pos_arr = np.array(pos_pts)
            grip_arr = np.array(grip_pts)

            hmm_state_id = int(np.median(segment_hmm_states)) if segment_hmm_states else stage_idx
            durs = stage_durations.get(hmm_state_id, [])
            max_dur = float(np.percentile(durs, 95)) if durs else 1000.0

            dforce_mean = None
            dforce_cov_inv = None

            if force_pts:
                force_arr = np.array(force_pts)
                force_mean = np.mean(force_arr, axis=0)
                force_cov_inv = _safe_cov_inv(force_arr)
                is_contact = float(np.mean(np.linalg.norm(force_arr, axis=1))) > contact_threshold

                if self.use_force_dynamics and dforce_pts:
                    df_arr = np.array(dforce_pts)
                    dforce_mean = np.mean(df_arr, axis=0)
                    dforce_cov_inv = _safe_cov_inv(df_arr)
            else:
                pos_dim = pos_arr.shape[1]
                force_mean = np.zeros(pos_dim)
                force_cov_inv = np.eye(pos_dim)
                is_contact = False

            targets.append(
                StageTarget(
                    pos_mean=np.mean(pos_arr, axis=0),
                    pos_cov_inv=_safe_cov_inv(pos_arr),
                    force_mean=force_mean,
                    force_cov_inv=force_cov_inv,
                    gripper_mean=float(np.mean(grip_arr)),
                    gripper_std=float(np.std(grip_arr)),
                    is_contact=is_contact,
                    max_duration=max_dur,
                    dforce_mean=dforce_mean,
                    dforce_cov_inv=dforce_cov_inv,
                )
            )

        return targets


# ==========================================
# Dense Reward Computation
# ==========================================
def compute_dense_reward(
    episode: list[dict],
    targets: list[StageTarget],
    state_sequence: np.ndarray,
    *,
    check_gripper: bool = False,
    use_force_dynamics: bool = False,
    penalty_power: float = 1.0,
    pos_slice: slice = slice(4, 7),
    force_slice: slice | None = slice(1, 4),
    gripper_idx: int = 0,
) -> np.ndarray:
    """Compute per-step dense reward from HMM state sequence and stage targets.

    Reward = stage_base + progress * force_mult * gripper_mult * time_mult
    """
    states = np.array([x["observations"]["state"] for x in episode])
    if states.ndim > 2:
        states = states.reshape(len(episode), -1)

    positions = states[:, pos_slice]
    forces = states[:, force_slice] if force_slice is not None else None
    grippers = states[:, gripper_idx]

    d_forces = np.gradient(forces, axis=0) if use_force_dynamics and forces is not None else None

    max_target_idx = len(targets) - 1
    rewards = np.zeros(len(states))

    current_state = -1
    duration = 0

    for t in range(len(states)):
        hmm_state = int(state_sequence[t])

        if hmm_state != current_state:
            current_state = hmm_state
            duration = 0
        else:
            duration += 1

        tidx = min(hmm_state, max_target_idx)
        tgt = targets[tidx]

        diff = positions[t] - tgt.pos_mean
        dist_mah = np.sqrt(np.clip(diff.T @ tgt.pos_cov_inv @ diff, 0.0, None))
        progress = np.exp(-0.5 * dist_mah)

        force_mult = 1.0
        if tgt.is_contact and forces is not None:
            f_diff = forces[t] - tgt.force_mean
            f_dist = np.sqrt(np.clip(f_diff.T @ tgt.force_cov_inv @ f_diff, 0.0, None))
            force_match = np.exp(-0.5 * f_dist)

            if use_force_dynamics and tgt.dforce_mean is not None and d_forces is not None:
                df_diff = d_forces[t] - tgt.dforce_mean
                df_dist = np.sqrt(np.clip(df_diff.T @ tgt.dforce_cov_inv @ df_diff, 0.0, None))
                dforce_match = np.exp(-0.5 * df_dist)
                force_match = 0.5 * force_match + 0.5 * dforce_match

            spatial_conf = np.exp(-20.0 * np.linalg.norm(diff))
            validity = (1.0 - spatial_conf) + spatial_conf * force_match
            force_mult = max(0.05, validity**penalty_power)

        gripper_mult = 1.0
        if check_gripper and tidx == max_target_idx:
            if grippers[t] < (tgt.gripper_mean - 3 * tgt.gripper_std) or grippers[t] < 0.005:
                gripper_mult = 0.05
            gripper_mult = gripper_mult**penalty_power

        time_mult = 1.0
        if duration > tgt.max_duration * 1.5:
            excess = duration - tgt.max_duration * 1.5
            time_mult = max(0.1, np.exp(-0.01 * excess))

        rewards[t] = float(hmm_state) + progress * force_mult * gripper_mult * time_mult

    return rewards


# ==========================================
# Quality Evaluation
# ==========================================
@dataclass
class LabelingMetrics:
    """Quality metrics for reward labeling."""

    pra: float
    gap: float
    succ_mean: float
    fail_mean: float
    succ_std: float
    fail_std: float
    succ_monotonicity: float
    n_stages: int


def evaluate_labeling(
    succ_eps: list[list[dict]],
    fail_eps: list[list[dict]],
    n_stages: int,
) -> LabelingMetrics:
    """Compute comprehensive labeling quality metrics."""
    succ_finals = [ep[-1]["rewards"] for ep in succ_eps]
    fail_finals = [ep[-1]["rewards"] for ep in fail_eps]

    correct = sum(1 for s in succ_finals for f in fail_finals if s > f)
    total = len(succ_finals) * len(fail_finals)
    pra = correct / total if total > 0 else 0.0

    gap = (min(succ_finals) - max(fail_finals)) if succ_finals and fail_finals else 0.0

    mono_count = 0
    for ep in succ_eps:
        rews = [step["rewards"] for step in ep]
        if len(rews) < 2:
            continue
        window = max(1, len(rews) // 10)
        smoothed = np.convolve(rews, np.ones(window) / window, mode="valid")
        diffs = np.diff(smoothed)
        mono_count += int(np.mean(diffs >= -0.01) > 0.8)
    mono_frac = mono_count / max(len(succ_eps), 1)

    return LabelingMetrics(
        pra=pra,
        gap=gap,
        succ_mean=float(np.mean(succ_finals)),
        fail_mean=float(np.mean(fail_finals)),
        succ_std=float(np.std(succ_finals)),
        fail_std=float(np.std(fail_finals)),
        succ_monotonicity=mono_frac,
        n_stages=n_stages,
    )


# ==========================================
# Dashboard Visualization
# ==========================================
def plot_dashboard(
    task_name: str,
    succ_eps: list[list[dict]],
    fail_eps: list[list[dict]],
    metrics: LabelingMetrics,
    output_dir: Path,
) -> None:
    """Generate comprehensive 2x3 evaluation dashboard."""
    fig = plt.figure(figsize=(24, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    for ep in succ_eps[:15]:
        ax1.plot([x["rewards"] for x in ep], color="green", alpha=0.25, linewidth=0.8)
    for ep in fail_eps[:15]:
        ax1.plot([x["rewards"] for x in ep], color="red", alpha=0.25, linewidth=0.8)
    ax1.set_title("Reward Curves (green=success, red=fail)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Reward")

    ax2 = fig.add_subplot(gs[0, 1])
    succ_finals = [ep[-1]["rewards"] for ep in succ_eps]
    fail_finals = [ep[-1]["rewards"] for ep in fail_eps]
    if succ_finals:
        sns.histplot(succ_finals, ax=ax2, color="green", alpha=0.5, label="Success", kde=True)
    if fail_finals:
        sns.histplot(fail_finals, ax=ax2, color="red", alpha=0.5, label="Fail", kde=True)
    ax2.set_title("Final Reward Distribution")
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    text = (
        f"Task: {task_name}\n"
        f"Stages: {metrics.n_stages}\n\n"
        f"PRA: {metrics.pra * 100:.1f}%\n"
        f"Gap: {metrics.gap:.3f}\n"
        f"Succ mean: {metrics.succ_mean:.3f} +/- {metrics.succ_std:.3f}\n"
        f"Fail mean: {metrics.fail_mean:.3f} +/- {metrics.fail_std:.3f}\n"
        f"Monotonicity: {metrics.succ_monotonicity * 100:.0f}%"
    )
    ax3.text(
        0.1,
        0.5,
        text,
        transform=ax3.transAxes,
        fontsize=13,
        verticalalignment="center",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightyellow"},
    )
    ax3.set_title("Quality Metrics")

    ax4 = fig.add_subplot(gs[1, 0])
    s_x = list(range(len(succ_finals)))
    f_x = list(range(len(fail_finals)))
    ax4.scatter(s_x, succ_finals, color="green", label="Success", s=40, zorder=3)
    ax4.scatter(f_x, fail_finals, color="red", label="Fail", s=40, zorder=3)
    ax4.axhline(y=np.mean(succ_finals) if succ_finals else 0, color="green", linestyle="--", alpha=0.5)
    ax4.axhline(y=np.mean(fail_finals) if fail_finals else 0, color="red", linestyle="--", alpha=0.5)
    ax4.set_title("Per-Episode Final Score")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Final Reward")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    if succ_eps:
        max_len = max(len(ep) for ep in succ_eps[:20])
        reward_matrix = np.full((min(20, len(succ_eps)), max_len), np.nan)
        for i, ep in enumerate(succ_eps[:20]):
            rews = [x["rewards"] for x in ep]
            reward_matrix[i, : len(rews)] = rews
        sns.heatmap(reward_matrix, ax=ax5, cmap="YlGn", cbar_kws={"label": "Reward"})
        ax5.set_title("Success Reward Heatmap")
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Episode")

    ax6 = fig.add_subplot(gs[1, 2])
    box_data = []
    box_labels = []
    if succ_finals:
        box_data.append(succ_finals)
        box_labels.append("Success")
    if fail_finals:
        box_data.append(fail_finals)
        box_labels.append("Fail")
    if box_data:
        bp = ax6.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        colors = ["lightgreen", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])], strict=False):
            patch.set_facecolor(color)
    ax6.set_title("Score Comparison")
    ax6.set_ylabel("Final Reward")

    plt.suptitle(f"Auto-Label Dashboard: {task_name}", fontsize=16, fontweight="bold")
    plt.savefig(output_dir / f"{task_name}_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


# ==========================================
# LeRobot Export
# ==========================================
def export_lerobot(episodes: list[list[dict]], path: Path, camera_keys: list[str]) -> None:
    """Export episode list to LeRobot-format pickle."""
    has_env_reward = any("env_rewards" in step for episode in episodes for step in episode)
    has_sparse_reward = any("sparse_rewards" in step for episode in episodes for step in episode)
    has_previous_state = any("previous_observations" in step for episode in episodes for step in episode)
    converted: dict[str, list] = {
        "observation.state": [],
        "action": [],
        "episode_index": [],
        "frame_index": [],
        "next.reward": [],
        "next.done": [],
        "next.success": [],
        "index": [],
    }
    if has_env_reward:
        converted["next.env_reward"] = []
    if has_sparse_reward:
        converted["next.sparse_reward"] = []
    if has_previous_state:
        converted["previous_observation.state"] = []
    for k in camera_keys:
        converted[f"observation.images.{k}"] = []

    idx = 0
    for ep_i, ep in enumerate(episodes):
        for fr_i, step in enumerate(ep):
            obs = step["observations"]
            converted["observation.state"].append(obs["state"].squeeze())
            converted["action"].append(step["actions"])
            converted["episode_index"].append(ep_i)
            converted["frame_index"].append(fr_i)
            converted["index"].append(idx)
            converted["next.reward"].append(step["rewards"])
            converted["next.done"].append(step["dones"])
            converted["next.success"].append(step.get("infos", {}).get("succeed", False))
            if has_env_reward:
                converted["next.env_reward"].append(float(step.get("env_rewards", np.nan)))
            if has_sparse_reward:
                converted["next.sparse_reward"].append(float(step.get("sparse_rewards", 0.0)))
            if has_previous_state:
                previous = step.get("previous_observations", {}).get("state", obs["state"])
                converted["previous_observation.state"].append(np.asarray(previous).squeeze())
            for k in camera_keys:
                converted[f"observation.images.{k}"].append(obs[k].squeeze())
            idx += 1

    final = {}
    for k, v in converted.items():
        # Keeping images as a list of per-frame arrays avoids a multi-gigabyte
        # peak allocation when np.stack duplicates large RGB datasets.  The
        # training/evaluation loaders support both lists and stacked arrays.
        if k.startswith("observation.images."):
            final[k] = v
            continue
        try:
            final[k] = np.stack(v)
        except ValueError as e:
            logging.warning("Cannot stack '%s': %s, keeping as list", k, e)
            final[k] = v

    save_pickle(final, path)


# ==========================================
# Pipeline
# ==========================================
def load_episodes(path: str) -> list[list[dict]]:
    """Load raw pickle and split into episodes."""
    data = load_pickle(path)
    flat: list[dict] = []
    for x in data:
        if isinstance(x, list):
            flat.extend(x)
        else:
            flat.append(x)
    return get_episodes(flat)


def run_pipeline(task_name: str) -> LabelingMetrics | None:
    """Full auto-labeling pipeline for one task."""
    cfg = get_task(task_name)
    out_dir = BASE_DIR / task_name / "auto_processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Auto-Labeling: {task_name}")
    print(f"{'=' * 60}")

    succ_eps = load_episodes(cfg.success_path)
    fail_eps = load_episodes(cfg.fail_path)
    print(f"Loaded {len(succ_eps)} success, {len(fail_eps)} fail episodes")

    reward_kw = {
        "pos_slice": cfg.state_pos_slice,
        "force_slice": cfg.state_force_slice,
        "gripper_idx": cfg.state_gripper_idx,
    }

    discovery = StageDiscovery(
        use_force_dynamics=cfg.use_force_dynamics,
        n_restarts=cfg.n_restarts,
        n_iter=cfg.hmm_n_iter,
        use_prototype_stages=cfg.use_prototype_stages,
        pos_slice=cfg.state_pos_slice,
        force_slice=cfg.state_force_slice,
        gripper_idx=cfg.state_gripper_idx,
    )
    discovery.fit(succ_eps, n_stages=cfg.n_stages, max_stages=cfg.max_stages)
    targets = discovery.compute_stage_targets(succ_eps, cfg.contact_force_threshold)

    print("Labeling success episodes...")
    for ep in succ_eps:
        _, ss = discovery.decode(ep)
        rews = compute_dense_reward(
            ep,
            targets,
            ss,
            check_gripper=cfg.check_gripper,
            use_force_dynamics=cfg.use_force_dynamics,
            penalty_power=cfg.penalty_power,
            **reward_kw,
        )
        for j, r in enumerate(rews):
            ep[j]["rewards"] = float(r)

    print("Labeling fail episodes...")
    for ep in fail_eps:
        try:
            _, ss = discovery.decode(ep)
            rews = compute_dense_reward(
                ep,
                targets,
                ss,
                check_gripper=cfg.check_gripper,
                use_force_dynamics=cfg.use_force_dynamics,
                penalty_power=cfg.penalty_power,
                **reward_kw,
            )
            for j, r in enumerate(rews):
                ep[j]["rewards"] = float(r)
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"  Warning: decode failed for one fail episode ({e}), assigning 0 reward")
            for step in ep:
                step["rewards"] = 0.0

    metrics = evaluate_labeling(succ_eps, fail_eps, discovery.n_components)
    print(f"\n  PRA:          {metrics.pra * 100:.1f}%")
    print(f"  Gap:          {metrics.gap:.4f}")
    print(f"  Succ mean:    {metrics.succ_mean:.3f} +/- {metrics.succ_std:.3f}")
    print(f"  Fail mean:    {metrics.fail_mean:.3f} +/- {metrics.fail_std:.3f}")
    print(f"  Monotonicity: {metrics.succ_monotonicity * 100:.0f}%")

    if cfg.use_prototype_stages:
        # Matplotlib/seaborn dashboard rendering has a pathological multi-GB
        # allocation with these larger MetaWorld episode objects.  Metrics are
        # still printed and preserved by downstream evaluation.
        print("Skipping dashboard for prototype-stage dataset.")
    else:
        print("Generating labeling dashboard...")
        plot_dashboard(task_name, succ_eps, fail_eps, metrics, out_dir)

    print("Exporting success episodes...")
    export_lerobot(succ_eps, out_dir / "success_lerobot.pkl", cfg.camera_keys)
    print("Exporting fail episodes...")
    export_lerobot(fail_eps, out_dir / "fail_lerobot.pkl", cfg.camera_keys)
    print(f"Exported to {out_dir}")

    return metrics
