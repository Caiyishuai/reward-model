"""RM-guided vision-based SAC training with **asynchronous reward relabeling**.

The Reward Model (DINOv2 backbone) is too expensive to run every env step.
Instead, the rollout loop stores only ``env_reward`` and raw RM-input data
(per-camera images + robot proprioceptive state) into the replay buffer.
A periodic **relabel pass** (every ``relabel_interval`` steps) batch-scores
unlabeled transitions with the RM and writes ``rm_reward`` back into the
buffer.  SAC training samples ``reward = env_reward + alpha * rm_reward``.

This decouples environment throughput from RM inference cost:
    - Rollout SPS: ~200  (bottleneck = CPU sim, no RM)
    - Relabel: batched GPU RM inference on buffer data

Usage::

    python -m sim.sac_train \\
        --task pushcube \\
        --total_timesteps 200000 \\
        --relabel_interval 500 --relabel_batch 64
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from checkpoint_io import save_rl_checkpoint
from data.common import IMG_SIZE_RM, IMG_SIZE_SAC, save_json
from log_utils import setup_logging
from reward_model import RewardModel
from sim.env_factory import make_env
from sim.obs_adapter import extract_full_state, extract_rgb, extract_state
from sim.rm_reward import combine_env_and_rm
from sim.task_configs import get_sim_config

logger = logging.getLogger(__name__)

LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ---------------------------------------------------------------------------
# Visual encoder (lightweight CNN for the SAC policy, NOT the RM backbone)
# ---------------------------------------------------------------------------


class PlainCNN(nn.Module):
    """Lightweight CNN: (B, 3, H, W) -> (B, out_dim)."""

    def __init__(self, in_channels: int = 3, out_dim: int = 256, img_size: int = IMG_SIZE_SAC):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.cnn(torch.zeros(1, in_channels, img_size, img_size)).shape[-1]
        self.fc = nn.Linear(flat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(x))


class MultiCameraEncoder(nn.Module):
    """Encode N cameras -> single feature vector."""

    def __init__(self, n_cameras: int, per_cam_dim: int = 256, img_size: int = IMG_SIZE_SAC):
        super().__init__()
        self.encoders = nn.ModuleList([PlainCNN(out_dim=per_cam_dim, img_size=img_size) for _ in range(n_cameras)])
        self.out_dim = per_cam_dim * n_cameras

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        for enc, (_, img) in zip(self.encoders, sorted(images.items()), strict=True):
            feats.append(enc(img))
        return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------------------
# Actor / Critic
# ---------------------------------------------------------------------------


class SoftQNetwork(nn.Module):
    def __init__(self, vis_dim: int, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        d = vis_dim + state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, vis: torch.Tensor, st: torch.Tensor, a: torch.Tensor):
        x = torch.cat([vis, st, a], -1)
        return self.q1(x), self.q2(x)


class Actor(nn.Module):
    def __init__(self, vis_dim: int, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vis_dim + state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, vis: torch.Tensor, st: torch.Tensor):
        h = self.net(torch.cat([vis, st], -1))
        return self.mu(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def get_action(self, vis: torch.Tensor, st: torch.Tensor):
        mu, log_std = self.forward(vis, st)
        dist = Normal(mu, log_std.exp())
        x_t = dist.rsample()
        a = torch.tanh(x_t)
        lp = (dist.log_prob(x_t) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, lp


# ---------------------------------------------------------------------------
# Replay buffer with separate env / rm reward columns
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Ring-buffer with split reward storage for async RM relabeling.

    Stores per-camera 128×128 uint8 images for the SAC policy, plus
    per-camera 224×224 uint8 images and 17-D robot state for RM scoring.
    ``rm_rewards`` is initially 0 and filled in by :meth:`relabel`.
    """

    def __init__(
        self,
        capacity: int,
        cam_keys: list[str],
        img_h: int,
        img_w: int,
        state_dim: int,
        action_dim: int,
        rm_state_dim: int,
        rm_img_size: int = IMG_SIZE_RM,
        seed: int | None = None,
    ):
        self.capacity = capacity
        self.cam_keys = cam_keys
        self.ptr = 0
        self.size = 0

        self.images = {k: np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8) for k in cam_keys}
        self.next_images = {k: np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8) for k in cam_keys}
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.env_rewards = np.zeros(capacity, dtype=np.float32)
        self.rm_rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.labeled = np.zeros(capacity, dtype=np.bool_)

        self.rm_img_size = rm_img_size
        self.rm_images = {k: np.zeros((capacity, 3, rm_img_size, rm_img_size), dtype=np.uint8) for k in cam_keys}
        self.rm_states = np.zeros((capacity, rm_state_dim), dtype=np.float32)
        self.ep_ids = np.full(capacity, -1, dtype=np.int64)
        self._rng = np.random.default_rng(seed)

        mem_gb = (
            sum(a.nbytes for a in self.images.values())
            + sum(a.nbytes for a in self.next_images.values())
            + sum(a.nbytes for a in self.rm_images.values())
            + self.states.nbytes
            + self.next_states.nbytes
            + self.rm_states.nbytes
        ) / (1024**3)
        logger.info("Buffer capacity=%d, est. memory=%.1f GB", capacity, mem_gb)

    def add(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        action: np.ndarray,
        env_reward: float,
        next_images: dict[str, np.ndarray],
        next_state: np.ndarray,
        done: bool,
        rm_images: dict[str, np.ndarray],
        rm_state: np.ndarray,
        ep_id: int,
    ) -> None:
        i = self.ptr
        for k in self.cam_keys:
            self.images[k][i] = images[k]
            self.next_images[k][i] = next_images[k]
            self.rm_images[k][i] = rm_images[k]
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.env_rewards[i] = env_reward
        self.rm_rewards[i] = 0.0
        self.dones[i] = float(done)
        self.labeled[i] = False
        self.rm_states[i] = rm_state
        self.ep_ids[i] = ep_id
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device, rm_alpha: float) -> dict:
        idxs = self._rng.integers(0, self.size, size=batch_size)
        # Fallback for unlabeled samples: substitute the running mean of
        # already-labeled rm_rewards so freshly-added transitions don't bias
        # SAC Q-learning toward rm_reward=0. Matches sac_train_v2.py behaviour.
        rm_r = self.rm_rewards[idxs].copy()
        unlabeled = ~self.labeled[idxs]
        if unlabeled.any() and self.labeled[: self.size].any():
            mean_rm = float(self.rm_rewards[: self.size][self.labeled[: self.size]].mean())
            rm_r[unlabeled] = mean_rm
        combined = combine_env_and_rm(self.env_rewards[idxs], rm_r, alpha=rm_alpha)
        return {
            "images": {k: torch.from_numpy(self.images[k][idxs]).float().div_(255.0).to(device) for k in self.cam_keys},
            "next_images": {
                k: torch.from_numpy(self.next_images[k][idxs]).float().div_(255.0).to(device) for k in self.cam_keys
            },
            "states": torch.from_numpy(self.states[idxs]).to(device),
            "next_states": torch.from_numpy(self.next_states[idxs]).to(device),
            "actions": torch.from_numpy(self.actions[idxs]).to(device),
            "rewards": torch.from_numpy(combined).to(device),
            "dones": torch.from_numpy(self.dones[idxs]).to(device),
        }

    def relabel(
        self,
        rm: RewardModel,
        n_cameras: int,
        state_windows: int,
        rm_state_dim: int,
        device: torch.device,
        batch_size: int = 64,
    ) -> int:
        """Batch-score unlabeled transitions and write back ``rm_rewards``.

        Builds temporal windows from consecutive same-episode steps.
        Returns the number of newly labeled transitions.
        """
        unlabeled = np.where(~self.labeled[: self.size])[0]
        if len(unlabeled) == 0:
            return 0

        cam_order = sorted(self.cam_keys)
        labeled_count = 0

        for start in range(0, len(unlabeled), batch_size):
            batch_idxs = unlabeled[start : start + batch_size]
            imgs_list: list[torch.Tensor] = []
            proprio_list: list[torch.Tensor] = []

            for idx in batch_idxs:
                ep = self.ep_ids[idx]
                window_idxs = self._get_temporal_window(idx, ep, state_windows)

                cam_frames: list[list[np.ndarray]] = [[] for _ in cam_order]
                state_frames: list[np.ndarray] = []
                for wi in window_idxs:
                    for ci, k in enumerate(cam_order):
                        cam_frames[ci].append(self.rm_images[k][wi])
                    state_frames.append(self.rm_states[wi])

                all_cam: list[np.ndarray] = []
                for per_cam in cam_frames:
                    stacked = np.stack(per_cam)  # (T, 3, H, W) uint8
                    all_cam.append(stacked.reshape(-1, self.rm_img_size, self.rm_img_size))
                img_tensor = torch.from_numpy(np.concatenate(all_cam, axis=0)).float().div_(255.0)
                imgs_list.append(img_tensor)

                proprio_np = np.concatenate(state_frames)
                proprio_list.append(torch.from_numpy(proprio_np).float())

            images_batch = torch.stack(imgs_list).to(device)
            proprio_batch = torch.stack(proprio_list).to(device)

            with torch.no_grad():
                rewards = rm.get_reward(images_batch, proprio_batch)

            rewards_np = rewards.squeeze(-1).cpu().numpy()
            for i, idx in enumerate(batch_idxs):
                self.rm_rewards[idx] = float(rewards_np[i])
                self.labeled[idx] = True
            labeled_count += len(batch_idxs)

        return labeled_count

    def _get_temporal_window(self, idx: int, ep_id: int, window_size: int) -> list[int]:
        """Gather ``window_size`` consecutive indices from the same episode.

        Handles the ring-buffer wrap: once ``size == capacity`` the logical
        predecessor of position 0 lives at ``capacity - 1``. Without this we
        silently pad the window with the current frame whenever a transition
        sits near pos 0 in a full buffer, biasing the RM reward.
        """
        result = [idx]
        cur = idx
        wrapped = self.size >= self.capacity
        for _ in range(window_size - 1):
            prev = cur - 1
            if prev < 0 and wrapped:
                prev = self.capacity - 1
            if 0 <= prev < self.size and self.ep_ids[prev] == ep_id:
                result.insert(0, prev)
                cur = prev
            else:
                result.insert(0, result[0])
        return result

    @property
    def label_ratio(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.labeled[: self.size].sum()) / self.size


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _extract_sac_obs(
    obs: dict, cam_order: list[str], target_h: int, target_w: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-camera uint8 (3,H,W) + flat state, resized to SAC resolution."""
    imgs: dict[str, np.ndarray] = {}
    for cam in cam_order:
        rgb = extract_rgb(obs, cam)
        chw = rgb.transpose(2, 0, 1)
        if chw.shape[1] != target_h or chw.shape[2] != target_w:
            t = torch.from_numpy(chw).float().unsqueeze(0)
            t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
            chw = t.squeeze(0).clamp_(0, 255).byte().numpy()
        imgs[cam] = chw
    return imgs, extract_full_state(obs)


def _extract_rm_images(obs: dict, cam_order: list[str], rm_size: int) -> dict[str, np.ndarray]:
    """Per-camera uint8 (3, rm_size, rm_size) for RM relabeling storage."""
    result: dict[str, np.ndarray] = {}
    for cam in cam_order:
        rgb = extract_rgb(obs, cam)
        chw = rgb.transpose(2, 0, 1)
        if chw.shape[1] != rm_size or chw.shape[2] != rm_size:
            t = torch.from_numpy(chw).float().unsqueeze(0)
            t = F.interpolate(t, size=(rm_size, rm_size), mode="bilinear", align_corners=False)
            chw = t.squeeze(0).clamp_(0, 255).byte().numpy()
        result[cam] = chw
    return result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SACConfig:
    task: str = "pushcube"
    rm_checkpoint: str | None = None
    reward_shaping_weight: float = 0.5

    relabel_interval: int = 500
    relabel_batch: int = 64

    total_timesteps: int = 500_000
    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 4_000
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha_init: float = 0.2
    autotune_alpha: bool = True
    utd_ratio: float = 0.5

    img_width: int = 128
    img_height: int = 128

    eval_freq: int = 10_000
    eval_episodes: int = 20
    save_freq: int = 50_000
    output_dir: str = "runs"
    # When set, run directory is output_dir / run_name (stable path for experiment scripts).
    run_name: str | None = None
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.utd_ratio <= 0:
            raise ValueError(f"utd_ratio must be positive, got {self.utd_ratio}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(
    env,
    vis_encoder: MultiCameraEncoder,
    actor: Actor,
    cam_order: list[str],
    device: torch.device,
    n_episodes: int,
    max_steps: int,
    img_h: int,
    img_w: int,
) -> dict[str, float]:
    vis_encoder.eval()
    actor.eval()
    successes = 0
    ep_rewards: list[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        for _ in range(max_steps):
            imgs, st = _extract_sac_obs(obs, cam_order, img_h, img_w)
            with torch.no_grad():
                img_t = {k: torch.from_numpy(v).float().div_(255.0).unsqueeze(0).to(device) for k, v in imgs.items()}
                mu, _ = actor(vis_encoder(img_t), torch.from_numpy(st).float().unsqueeze(0).to(device))
                action = torch.tanh(mu)
            obs, r, term, trunc, info = env.step(action.squeeze(0).cpu().numpy())
            total_r += r.item() if hasattr(r, "item") else float(r)
            done = term or trunc
            if hasattr(done, "any"):
                done = bool(done.any())
            if done:
                s = info.get("success", False)
                if hasattr(s, "any"):
                    s = bool(s.any())
                if s:
                    successes += 1
                break
        ep_rewards.append(total_r)

    vis_encoder.train()
    actor.train()
    return {"success_rate": successes / max(n_episodes, 1), "mean_reward": float(np.mean(ep_rewards))}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: SACConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    sim_cfg = get_sim_config(cfg.task)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002
    torch.manual_seed(cfg.seed)

    run_name = cfg.run_name or f"sac_{cfg.task}_{cfg.seed}_{int(time.time())}"
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=run_dir / "train.log")
    logger.info("Config: %s", cfg)

    env = make_env(sim_cfg, num_envs=1)
    eval_env = env  # reuse same env for eval to avoid SAPIEN multi-instance segfault

    obs_test, _ = env.reset()
    state_dim = len(extract_full_state(obs_test))
    action_dim = math.prod(env.action_space.shape)
    cam_order = sorted(sim_cfg.camera_map.keys())
    n_cameras = len(cam_order)

    logger.info("Task=%s  state=%d  act=%d  cams=%s", sim_cfg.env_id, state_dim, action_dim, cam_order)
    logger.info("RM=%s  alpha=%.3f", cfg.rm_checkpoint or "none", cfg.reward_shaping_weight)
    logger.info("Relabel every %d steps, batch=%d", cfg.relabel_interval, cfg.relabel_batch)
    logger.info("Output: %s", run_dir)

    vis_enc = MultiCameraEncoder(n_cameras, per_cam_dim=256, img_size=cfg.img_width).to(device)
    actor = Actor(vis_enc.out_dim, state_dim, action_dim).to(device)
    qf = SoftQNetwork(vis_enc.out_dim, state_dim, action_dim).to(device)
    qf_tgt = SoftQNetwork(vis_enc.out_dim, state_dim, action_dim).to(device)
    qf_tgt.load_state_dict(qf.state_dict())
    vis_enc_tgt = MultiCameraEncoder(n_cameras, per_cam_dim=256, img_size=cfg.img_width).to(device)
    vis_enc_tgt.load_state_dict(vis_enc.state_dict())
    for p in qf_tgt.parameters():
        p.requires_grad_(False)
    for p in vis_enc_tgt.parameters():
        p.requires_grad_(False)

    q_opt = optim.Adam(list(qf.parameters()) + list(vis_enc.parameters()), lr=cfg.lr)
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.lr)

    if cfg.autotune_alpha:
        target_entropy = -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_opt = optim.Adam([log_alpha], lr=cfg.lr)
        alpha = log_alpha.exp().item()
    else:
        alpha = cfg.alpha_init

    rm: RewardModel | None = None
    state_windows = 3
    if cfg.rm_checkpoint:
        rm = RewardModel.load(cfg.rm_checkpoint, device=str(device))
        state_windows = rm.state_windows
        logger.info("RM loaded (state_windows=%d)", state_windows)

    replay = ReplayBuffer(
        capacity=cfg.buffer_size,
        cam_keys=cam_order,
        img_h=cfg.img_height,
        img_w=cfg.img_width,
        state_dim=state_dim,
        action_dim=action_dim,
        rm_state_dim=sim_cfg.rm_state_dim,
        rm_img_size=IMG_SIZE_RM,
        seed=cfg.seed,
    )

    obs, _ = env.reset()
    cur_imgs, cur_state = _extract_sac_obs(obs, cam_order, cfg.img_height, cfg.img_width)

    best_sr = 0.0
    ep_count = 0
    ep_env_r = 0.0
    ep_len = 0
    ep_id = 0
    log_data: list[dict] = []
    start_time = time.time()
    # Latest update losses; may remain None on a logging tick when UTD<1 skips
    # updates on that particular step.
    qf_loss: torch.Tensor | None = None
    actor_loss: torch.Tensor | None = None

    logger.info("Training for %d steps (async relabel)", cfg.total_timesteps)

    for step in range(1, cfg.total_timesteps + 1):
        # --- action selection ---
        if step < cfg.learning_starts:
            a_np = env.action_space.sample()
            if hasattr(a_np, "squeeze"):
                a_np = a_np.squeeze(0) if a_np.ndim > 1 else a_np
        else:
            with torch.no_grad():
                img_t = {
                    k: torch.from_numpy(v).float().div_(255.0).unsqueeze(0).to(device) for k, v in cur_imgs.items()
                }
                a_t, _ = actor.get_action(
                    vis_enc(img_t),
                    torch.from_numpy(cur_state).float().unsqueeze(0).to(device),
                )
                a_np = a_t.squeeze(0).cpu().numpy()

        # --- env step ---
        obs, env_r, term, trunc, info = env.step(a_np)
        r_env = env_r.item() if hasattr(env_r, "item") else float(env_r)
        ep_env_r += r_env
        ep_len += 1

        done = term or trunc
        if hasattr(done, "any"):
            done = bool(done.any())

        next_imgs, next_state = _extract_sac_obs(obs, cam_order, cfg.img_height, cfg.img_width)

        rm_imgs = _extract_rm_images(obs, cam_order, 224)
        rm_state = extract_state(env.unwrapped, sim_cfg.state_mode)

        replay.add(
            cur_imgs,
            cur_state,
            a_np,
            r_env,
            next_imgs,
            next_state,
            done,
            rm_imgs,
            rm_state,
            ep_id,
        )

        if done:
            ep_count += 1
            success = info.get("success", False)
            if hasattr(success, "any"):
                success = bool(success.any())
            if ep_count % 10 == 0:
                tag = "OK" if success else "  "
                logger.info(
                    "Ep %4d [%s] len=%3d env_r=%.2f labeled=%.0f%%",
                    ep_count,
                    tag,
                    ep_len,
                    ep_env_r,
                    replay.label_ratio * 100,
                )
            ep_env_r = 0.0
            ep_len = 0
            ep_id += 1
            obs, _ = env.reset()
            cur_imgs, cur_state = _extract_sac_obs(obs, cam_order, cfg.img_height, cfg.img_width)
        else:
            cur_imgs, cur_state = next_imgs, next_state

        # --- async RM relabeling ---
        if rm is not None and step % cfg.relabel_interval == 0 and replay.size > 0:
            t0 = time.time()
            n_labeled = replay.relabel(
                rm,
                n_cameras,
                state_windows,
                sim_cfg.rm_state_dim,
                device,
                batch_size=cfg.relabel_batch,
            )
            dt = time.time() - t0
            if n_labeled > 0:
                logger.info(
                    "RELABEL step=%d scored=%d labeled=%.0f%% time=%.1fs", step, n_labeled, replay.label_ratio * 100, dt
                )

        # --- SAC gradient updates ---
        if step < cfg.learning_starts:
            continue

        # Support fractional UTD: utd_ratio=0.5 means one update every 2 env steps.
        # The naive `int(utd_ratio)` truncates to 0 for 0<utd<1 and then max(1, 0)=1
        # silently forces utd=1 — the opposite of the intent.
        if cfg.utd_ratio >= 1.0:
            n_updates = int(cfg.utd_ratio)
        else:
            period = max(1, int(round(1.0 / cfg.utd_ratio)))
            n_updates = 1 if step % period == 0 else 0

        for _ in range(n_updates):
            batch = replay.sample(cfg.batch_size, device, cfg.reward_shaping_weight)

            with torch.no_grad():
                vf_next = vis_enc_tgt(batch["next_images"])
                na, nlp = actor.get_action(vf_next, batch["next_states"])
                q1n, q2n = qf_tgt(vf_next, batch["next_states"], na)
                td_target = batch["rewards"].unsqueeze(1) + (1 - batch["dones"].unsqueeze(1)) * cfg.gamma * (
                    torch.min(q1n, q2n) - alpha * nlp
                )

            vf_cur = vis_enc(batch["images"])
            q1, q2 = qf(vf_cur, batch["states"], batch["actions"])
            qf_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            q_opt.zero_grad()
            qf_loss.backward()
            q_opt.step()

            vf_a = vf_cur.detach()
            new_a, lp = actor.get_action(vf_a, batch["states"])
            q1a, q2a = qf(vf_a, batch["states"], new_a)
            actor_loss = (alpha * lp - torch.min(q1a, q2a)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            if cfg.autotune_alpha:
                al = -(log_alpha.exp() * (lp.detach() + target_entropy)).mean()
                alpha_opt.zero_grad()
                al.backward()
                alpha_opt.step()
                alpha = log_alpha.exp().item()

            with torch.no_grad():
                for p, pt in zip(qf.parameters(), qf_tgt.parameters(), strict=True):
                    pt.data.lerp_(p.data, cfg.tau)
                for p, pt in zip(vis_enc.parameters(), vis_enc_tgt.parameters(), strict=True):
                    pt.data.lerp_(p.data, cfg.tau)

        # --- logging ---
        if step % 2000 == 0:
            sps = int(step / (time.time() - start_time))
            qf_val = qf_loss.item() if qf_loss is not None else float("nan")
            pi_val = actor_loss.item() if actor_loss is not None else float("nan")
            logger.info(
                "Step %7d/%d | qf=%.3f pi=%.3f α=%.3f | buf=%d labeled=%.0f%% | SPS=%d",
                step,
                cfg.total_timesteps,
                qf_val,
                pi_val,
                alpha,
                replay.size,
                replay.label_ratio * 100,
                sps,
            )

        if step % cfg.eval_freq == 0:
            m = evaluate_policy(
                eval_env,
                vis_enc,
                actor,
                cam_order,
                device,
                cfg.eval_episodes,
                sim_cfg.max_episode_steps,
                cfg.img_height,
                cfg.img_width,
            )
            log_data.append({"step": step, **m})
            logger.info("EVAL step=%d  success=%.1f%%  reward=%.3f", step, m["success_rate"] * 100, m["mean_reward"])
            if m["success_rate"] > best_sr:
                best_sr = m["success_rate"]
                _save(run_dir / "best.pt", vis_enc, actor, qf, cfg=cfg, step=step, stage="best")
                logger.info("EVAL New best -> %s", run_dir / "best.pt")

        if step % cfg.save_freq == 0:
            _save(run_dir / f"ckpt_{step}.pt", vis_enc, actor, qf, cfg=cfg, step=step, stage="checkpoint")

    env.close()
    _save(
        run_dir / "final.pt",
        vis_enc,
        actor,
        qf,
        cfg=cfg,
        step=cfg.total_timesteps,
        stage="final",
    )
    save_json(log_data, run_dir / "log.json")
    logger.info("Done. Best success=%.1f%%  Output: %s", best_sr * 100, run_dir)


def _save(
    path: Path,
    ve: nn.Module,
    ac: nn.Module,
    qf: nn.Module,
    *,
    cfg: SACConfig,
    step: int,
    stage: str,
) -> None:
    save_rl_checkpoint(
        path,
        state={"vis_encoder": ve.state_dict(), "actor": ac.state_dict(), "qf": qf.state_dict()},
        meta={
            "agent": "sac_v1",
            "stage": stage,
            "task": cfg.task,
            "env_id": get_sim_config(cfg.task).env_id,
            "total_timesteps": cfg.total_timesteps,
            "global_step": int(step),
            "rm_checkpoint": cfg.rm_checkpoint,
            "rm_alpha": cfg.reward_shaping_weight,
            "seed": cfg.seed,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="RM-guided SAC with async relabeling")
    p.add_argument("--task", default="pushcube")
    p.add_argument("--rm_checkpoint", default=None, help="RM checkpoint path; pass 'none' to disable RM")
    p.add_argument("--reward_shaping_weight", type=float, default=0.5)
    p.add_argument("--relabel_interval", type=int, default=500)
    p.add_argument("--relabel_batch", type=int, default=64)
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_starts", type=int, default=4_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--utd_ratio", type=float, default=0.5)
    p.add_argument("--img_width", type=int, default=128)
    p.add_argument("--img_height", type=int, default=128)
    p.add_argument("--eval_freq", type=int, default=10_000)
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--save_freq", type=int, default=50_000)
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--run_name", default=None, help="Fixed subdirectory under output_dir (default: timestamped)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    sim_cfg = get_sim_config(args.task)
    rm_ckpt = args.rm_checkpoint
    if rm_ckpt is None:
        rm_ckpt = sim_cfg.rm_checkpoint
    elif rm_ckpt.lower() == "none":
        rm_ckpt = None
    cfg = SACConfig(
        task=args.task,
        rm_checkpoint=rm_ckpt,
        reward_shaping_weight=args.reward_shaping_weight,
        relabel_interval=args.relabel_interval,
        relabel_batch=args.relabel_batch,
        total_timesteps=args.total_timesteps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        utd_ratio=args.utd_ratio,
        img_width=args.img_width,
        img_height=args.img_height,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
