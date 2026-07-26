"""Vision-based SAC for ManiSkill3 with optional RM reward shaping.

Based on ManiSkill3 official ``sac_rgbd.py`` baseline and LeanRL patterns.

Architecture:
- PlainConv CNN shared between Actor and Critic (ManiSkill3 pattern)
- GPU replay buffer via DictArray for high throughput
- GPU-vectorized parallel environments (16+)
- Optional RM relabeling: CPU-side observation storage + periodic batch scoring

Usage::

    # Pure SAC
    python -m sim.sac_train_v2 --env-id PushCube-v1 --num-envs 32

    # SAC + RM reward shaping
    python -m sim.sac_train_v2 --env-id PushCube-v1 --num-envs 32 \\
        --robot-uids panda_wristcam \\
        --rm-checkpoint checkpoints/auto_pushcube/rm_pushcube.pt \\
        --rm-alpha 0.5

    # From script
    from sim.sac_train_v2 import Args, train
    train(Args(env_id="PushCube-v1", total_timesteps=500_000))
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter

from checkpoint_io import load_rl_checkpoint, save_rl_checkpoint
from data.common import IMG_SIZE_RM
from sim.rm_reward import combine_env_and_rm, effective_alpha, postprocess_rm_reward

# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """SAC training arguments (parsed by tyro)."""

    # -- experiment --
    exp_name: str | None = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    """Enable Weights & Biases logging."""
    wandb_project_name: str = "AutoRM"
    wandb_entity: str | None = None
    wandb_group: str = "SAC"
    capture_video: bool = True
    save_model: bool = True
    evaluate: bool = False
    checkpoint: str | None = None
    log_freq: int = 1_000

    # -- environment --
    env_id: str = "PushCube-v1"
    obs_mode: str = "rgb"
    include_state: bool = True
    num_envs: int = 16
    num_eval_envs: int = 16
    num_steps: int = 50
    num_eval_steps: int = 50
    control_mode: str | None = "pd_ee_delta_pos"
    render_mode: str = "all"
    camera_width: int | None = 64
    camera_height: int | None = 64
    partial_reset: bool = False
    eval_partial_reset: bool = False
    reconfiguration_freq: int | None = None
    eval_reconfiguration_freq: int | None = 1
    eval_freq: int = 25
    """Evaluate every this many iterations (1 iteration = training_freq env steps)."""
    save_train_video_freq: int | None = None
    robot_uids: str | None = None
    """Robot UID (e.g. 'panda_wristcam' for dual-camera RM setup)."""
    sim_backend: str = "gpu"
    """ManiSkill3 sim backend. Default ``gpu`` matches GPU training; use ``cpu`` only with ``--no-cuda`` (debug)."""

    # -- algorithm --
    total_timesteps: int = 1_000_000
    buffer_size: int = 500_000
    buffer_device: str = "cuda"
    gamma: float = 0.8
    tau: float = 0.01
    batch_size: int = 512
    learning_starts: int = 4_000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    training_freq: int = 64
    utd: float = 0.25
    bootstrap_at_done: str = "always"

    # -- acceleration (LeanRL-style) --
    compile: bool = False
    """torch.compile actor and critic for ~1.5-2x speedup (requires PyTorch 2.0+)."""
    amp: bool = False
    """Enable automatic mixed precision (fp16) for training forward/backward passes."""

    # -- RM relabeling (Phase B) --
    rm_checkpoint: str | None = None
    """RewardModel checkpoint. None = env reward only."""
    rm_alpha: float = 0.5
    """Reward shaping weight: total = env_reward_scale * env_reward + rm_alpha * rm_reward."""
    env_reward_scale: float = 1.0
    """Multiplier applied to env reward before combining. Set 0.0 to train on pure RM reward only (key ablation)."""
    relabel_interval: int = 500
    """Re-score unlabeled transitions every this many env steps."""
    relabel_batch: int = 64
    """Batch size for RM inference during relabeling."""
    rm_img_size: int = IMG_SIZE_RM
    """Image resolution for RM (DINOv2 expects 224)."""
    rm_alpha_warmup: int = 0
    """Linear warmup steps for rm_alpha. 0 = no warmup."""
    rm_clip: float = 0.0
    """Clip RM rewards to [-rm_clip, rm_clip]. 0 = no clipping."""
    rm_normalize: bool = True
    """Normalize RM rewards using running mean/std."""
    rm_uncertainty_weight: bool = True
    """Weight RM rewards by ensemble confidence (1 / (1 + ensemble_std))."""
    rm_potential_shaping: bool = False
    """Use potential-based shaping F(s,a,s') = γ·RM(s') − RM(s) instead of raw RM(s).
    Preserves the optimal policy (Ng/Harada/Russell 1999) while removing the raw
    offset of the RM and re-centering reward magnitude to TD-target scale."""

    # -- computed --
    grad_steps_per_iteration: int = 0
    steps_per_env: int = 0


# ---------------------------------------------------------------------------
# GPU observation storage
# ---------------------------------------------------------------------------


class DictArray:
    """Tensor-backed nested dict storage for GPU replay buffers."""

    def __init__(self, buffer_shape, element_space=None, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict is not None:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.Dict)
            self.data: dict = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    if np.issubdtype(v.dtype, np.floating):
                        dtype = torch.float32
                    elif v.dtype == np.uint8:
                        dtype = torch.uint8
                    elif np.issubdtype(v.dtype, np.signedinteger):
                        dtype = torch.int32
                    else:
                        dtype = torch.float32
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return list(self.data.keys())

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
            return
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class ReplayBufferSample:
    obs: dict
    next_obs: dict
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    batch_inds: torch.Tensor
    env_inds: torch.Tensor


class ReplayBuffer:
    """GPU-backed replay buffer for vectorized environments."""

    def __init__(self, env, num_envs, buffer_size, storage_device, sample_device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs

        self.obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
        self.next_obs = DictArray(
            (self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device
        )
        self.actions = torch.zeros(
            (self.per_env_buffer_size, num_envs) + env.single_action_space.shape, device=storage_device
        )
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

    def add(self, obs, next_obs, action, reward, done):
        if self.storage_device == torch.device("cpu"):
            obs = {k: v.cpu() for k, v in obs.items()}
            next_obs = {k: v.cpu() for k, v in next_obs.items()}
            action, reward, done = action.cpu(), reward.cpu(), done.cpu()
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        max_idx = self.per_env_buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, max_idx, size=(batch_size,), device=self.storage_device)
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,), device=self.storage_device)
        obs_s = {k: v.to(self.sample_device) for k, v in self.obs[batch_inds, env_inds].items()}
        nobs_s = {k: v.to(self.sample_device) for k, v in self.next_obs[batch_inds, env_inds].items()}
        return ReplayBufferSample(
            obs=obs_s,
            next_obs=nobs_s,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
            batch_inds=batch_inds.cpu(),
            env_inds=env_inds.cpu(),
        )


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    layers = []
    c_in = in_channels
    for idx, c_out in enumerate(mlp_channels):
        layers.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            layers.append(act_builder())
        c_in = c_out
    return nn.Sequential(*layers)


class PlainConv(nn.Module):
    """Lightweight CNN: (B, C, H, W) -> (B, out_dim)."""

    def __init__(self, in_channels=3, out_dim=256, image_size=(128, 128)):
        super().__init__()
        self.out_dim = out_dim
        first_pool = nn.MaxPool2d(4, 4) if image_size[0] >= 128 else nn.MaxPool2d(2, 2)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            first_pool,
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fc = make_mlp(64 * 4 * 4, [out_dim], last_act=True)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, image):
        return self.fc(self.cnn(image).flatten(1))


class EncoderObsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs):
        parts = []
        if "rgb" in obs:
            parts.append(obs["rgb"].float() / 255.0)
        if "depth" in obs:
            parts.append(obs["depth"].float())
        img = torch.cat(parts, dim=3).permute(0, 3, 1, 2)
        return self.encoder(img)


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, envs, encoder):
        super().__init__()
        self.encoder = encoder
        action_dim = int(np.prod(envs.single_action_space.shape))
        state_dim = envs.single_observation_space["state"].shape[0]
        self.mlp = make_mlp(encoder.encoder.out_dim + action_dim + state_dim, [512, 256, 1], last_act=False)

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        return self.mlp(torch.cat([visual_feature, obs["state"], action], dim=1))


class Actor(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        action_dim = int(np.prod(envs.single_action_space.shape))
        state_dim = envs.single_observation_space["state"].shape[0]
        in_channels, image_size = 0, (64, 64)
        if "rgb" in sample_obs:
            in_channels += sample_obs["rgb"].shape[-1]
            image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])
        if "depth" in sample_obs:
            in_channels += sample_obs["depth"].shape[-1]
            image_size = (sample_obs["depth"].shape[1], sample_obs["depth"].shape[2])
        self.encoder = EncoderObsWrapper(PlainConv(in_channels=in_channels, out_dim=256, image_size=image_size))
        self.mlp = make_mlp(self.encoder.encoder.out_dim + state_dim, [512, 256], last_act=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        h, lo = envs.single_action_space.high, envs.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - lo) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + lo) / 2.0, dtype=torch.float32))

    def _get_feature(self, obs, detach_encoder=False):
        vf = self.encoder(obs)
        if detach_encoder:
            vf = vf.detach()
        return self.mlp(torch.cat([vf, obs["state"]], dim=1)), vf

    def forward(self, obs, detach_encoder=False):
        x, vf = self._get_feature(obs, detach_encoder)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std, vf

    def get_eval_action(self, obs):
        mean, _, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, obs, detach_encoder=False):
        mean, log_std, vf = self(obs, detach_encoder)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * self.action_scale + self.action_bias, vf


# ---------------------------------------------------------------------------
# RM Relabeler (Phase B)
# ---------------------------------------------------------------------------


def _batch_quat2axisangle(quat):
    """Batch quaternion [qw, qx, qy, qz] -> axis-angle (B, 3)."""
    qw = quat[:, 0].clamp(-1.0, 1.0)
    den = torch.sqrt((1.0 - qw * qw).clamp(min=0.0))
    angle = 2.0 * torch.acos(qw)
    safe = den > 1e-7
    axis = torch.zeros(quat.shape[0], 3, device=quat.device, dtype=quat.dtype)
    axis[safe] = quat[safe, 1:4] * (angle[safe] / den[safe]).unsqueeze(1)
    return axis


def _extract_rm_state_batch(envs):
    """Extract 17-D RM state (8-D EEF + 9-D joints) from vectorized env.

    Keeps computation on GPU, single .cpu() at the end.
    """
    raw = envs._env
    tcp = raw.agent.tcp.pose.raw_pose
    qpos = raw.agent.robot.get_qpos()
    pos = tcp[:, :3]
    axis_angle = _batch_quat2axisangle(tcp[:, 3:7])
    fingers = qpos[:, -2:]
    eef = torch.cat([pos, axis_angle, fingers], dim=1)
    return torch.cat([eef, qpos], dim=1).float().cpu().numpy()


class RMRelabeler:
    """CPU-side RM observation storage with DINOv2 feature caching.

    Key optimizations:
    - Feature cache: DINOv2 backbone runs once per unique frame, features are
      cached and reused across temporal windows (shared frames between adjacent
      transitions avoid redundant backbone inference)
    - Lazy resize: images stored at env resolution, resized only for backbone
    - GPU rewards: rm_rewards mirrored on GPU for zero-copy training access
    - Async relabeling: background thread overlaps with gradient updates
    - Pending queue: O(1) new-transition tracking
    """

    def __init__(self, rm_checkpoint, per_env_capacity, num_envs, num_cameras, rm_img_size, device):
        from reward_model import RewardModel

        self.rm = RewardModel.load(rm_checkpoint, device=device)
        self.rm.eval()
        self.state_windows = self.rm.state_windows
        self.robot_dim = self.rm.robot_dim
        self.feat_dim = self.rm.vision_feat_dim
        self.rm_device = torch.device(device)
        self.num_cameras = num_cameras
        self.rm_img_size = rm_img_size
        self.per_env_cap = per_env_capacity
        self.num_envs = num_envs
        self._relabel_batch_size = 256
        self._backbone_batch_size = 512

        self._store_img_size: tuple[int, int] | None = None
        self.rm_images: list[np.ndarray] | None = None
        self.rm_states = np.zeros((per_env_capacity, num_envs, self.robot_dim), dtype=np.float32)
        self.rm_rewards = np.zeros((per_env_capacity, num_envs), dtype=np.float32)
        self.rm_uncertainty = np.ones((per_env_capacity, num_envs), dtype=np.float32)
        self.labeled = np.zeros((per_env_capacity, num_envs), dtype=np.bool_)
        self.ep_ids = np.full((per_env_capacity, num_envs), -1, dtype=np.int64)

        self.feat_cache = np.zeros(
            (per_env_capacity, num_envs, num_cameras, self.feat_dim),
            dtype=np.float16,
        )
        self.feat_valid = np.zeros((per_env_capacity, num_envs), dtype=np.bool_)

        self._pending: list[tuple[int, int]] = []
        self._pending_lock = threading.Lock()
        self._gpu_rewards: torch.Tensor | None = None
        self._gpu_uncertainty: torch.Tensor | None = None
        self._gpu_labeled: torch.Tensor | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._relabel_future: Future | None = None

        # float64 accumulators: summing millions of float32 RM outputs can overflow
        # float32 range (~3e38) when running_sum / sq_sum are updated every relabel batch.
        self._rm_running_sum = np.float64(0.0)
        self._rm_running_sq_sum = np.float64(0.0)
        self._rm_running_count = 0

        feat_mb = self.feat_cache.nbytes / (1024**2)
        print(
            f"[RMRelabeler] cap={per_env_capacity * num_envs}, cams={num_cameras}, "
            f"sw={self.state_windows}, feat_cache={feat_mb:.0f}MB, async=on"
        )

    def _init_image_storage(self, img_h: int, img_w: int) -> None:
        # Keep (h, w) separately so non-square inputs store correctly and the
        # backbone receives the right aspect ratio instead of being squashed.
        self._store_img_size = (img_h, img_w)
        self.rm_images = [
            np.zeros((self.per_env_cap, self.num_envs, 3, img_h, img_w), dtype=np.uint8)
            for _ in range(self.num_cameras)
        ]
        mem_gb = sum(a.nbytes for a in self.rm_images) / (1024**3)
        print(f"[RMRelabeler] Image storage: {img_h}x{img_w}, {mem_gb:.1f}GB")

    def store(self, buf_pos, obs_rgb, rm_state, ep_ids):
        # obs_rgb from ManiSkill is uint8 [B, H, W, C]; avoid the fp32 copy
        # (4x memory / slower) — we end up casting back to uint8 anyway.
        rgb_t = obs_rgb.permute(0, 3, 1, 2).contiguous()
        if rgb_t.dtype != torch.uint8:
            rgb_t = rgb_t.clamp(0, 255).to(torch.uint8)
        per_cam = rgb_t.shape[1] // self.num_cameras
        img_h, img_w = rgb_t.shape[2], rgb_t.shape[3]

        if self.rm_images is None:
            self._init_image_storage(img_h, img_w)

        rgb_np = rgb_t.cpu().numpy()
        for ci in range(self.num_cameras):
            self.rm_images[ci][buf_pos] = rgb_np[:, ci * per_cam : (ci + 1) * per_cam]

        self.rm_states[buf_pos] = rm_state
        self.ep_ids[buf_pos] = ep_ids
        self.labeled[buf_pos] = False
        self.feat_valid[buf_pos] = False
        self.rm_rewards[buf_pos] = 0.0

        with self._pending_lock:
            for e in range(self.num_envs):
                self._pending.append((buf_pos, e))

    def _compute_backbone_features(self, positions: np.ndarray, envs: np.ndarray) -> None:
        """Run DINOv2 backbone on frames that lack cached features."""
        need_mask = ~self.feat_valid[positions, envs]
        if not np.any(need_mask):
            return

        unique_pe = set()
        for p, e in zip(positions[need_mask], envs[need_mask], strict=False):
            unique_pe.add((int(p), int(e)))
        if not unique_pe:
            return

        u_pos = np.array([p for p, _ in unique_pe], dtype=np.intp)
        u_env = np.array([e for _, e in unique_pe], dtype=np.intp)

        assert self._store_img_size is not None, "Image storage must be initialised before feature extraction"
        store_h, store_w = self._store_img_size
        for batch_start in range(0, len(u_pos), self._backbone_batch_size):
            batch_end = min(batch_start + self._backbone_batch_size, len(u_pos))
            bp = u_pos[batch_start:batch_end]
            be = u_env[batch_start:batch_end]
            bs = len(bp)

            for ci in range(self.num_cameras):
                frames_np = np.empty((bs, 3, store_h, store_w), dtype=np.float32)
                for i in range(bs):
                    frames_np[i] = self.rm_images[ci][bp[i], be[i]]
                frames_t = torch.from_numpy(frames_np).div_(255.0)
                if (store_h, store_w) != (self.rm_img_size, self.rm_img_size):
                    frames_t = F.interpolate(
                        frames_t,
                        size=(self.rm_img_size, self.rm_img_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                frames_t = frames_t.to(self.rm_device)
                with torch.no_grad(), torch.amp.autocast(self.rm_device.type):
                    feats = self.rm.encode_frames(frames_t)
                self.feat_cache[bp, be, ci] = feats.half().cpu().numpy()

            self.feat_valid[bp, be] = True

    def relabel_async(self, buf_size):
        if self._relabel_future is not None and not self._relabel_future.done():
            return
        with self._pending_lock:
            if not self._pending:
                return
            pending = self._pending
            self._pending = []
        self._relabel_future = self._executor.submit(self._relabel_impl, buf_size, pending)

    def wait_relabel(self) -> int:
        if self._relabel_future is None:
            return 0
        n = self._relabel_future.result()
        self._relabel_future = None
        if n > 0:
            self._sync_gpu_rewards()
        return n

    def _sync_gpu_rewards(self):
        t = torch.from_numpy(self.rm_rewards)
        u = torch.from_numpy(self.rm_uncertainty)
        lb = torch.from_numpy(self.labeled)
        if self._gpu_rewards is None:
            self._gpu_rewards = t.to(self.rm_device)
            self._gpu_uncertainty = u.to(self.rm_device)
            self._gpu_labeled = lb.to(self.rm_device)
        else:
            self._gpu_rewards.copy_(t)
            self._gpu_uncertainty.copy_(u)
            self._gpu_labeled.copy_(lb)

    def relabel(self, buf_size):
        with self._pending_lock:
            if not self._pending:
                return 0
            pending = self._pending
            self._pending = []
        n = self._relabel_impl(buf_size, pending)
        if n > 0:
            self._sync_gpu_rewards()
        return n

    def _relabel_impl(self, buf_size, pending):
        labeled_count = 0
        for start in range(0, len(pending), self._relabel_batch_size):
            end = min(start + self._relabel_batch_size, len(pending))
            batch = pending[start:end]
            b_pos = np.array([p for p, _ in batch], dtype=np.intp)
            b_env = np.array([e for _, e in batch], dtype=np.intp)
            bs = len(b_pos)
            sw = self.state_windows

            # When the replay buffer is full the per-env storage is a ring:
            # position 0's logical predecessor is position per_env_cap-1. If we
            # ignore wrap-around the window silently pads with the current
            # frame whenever a transition sits near pos 0, biasing the reward.
            wrapped = buf_size >= self.per_env_cap
            win_pos = np.empty((bs, sw), dtype=np.intp)
            for i in range(bs):
                p, e = int(b_pos[i]), int(b_env[i])
                ep = self.ep_ids[p, e]
                indices = [p]
                c = p
                for _ in range(sw - 1):
                    prev = c - 1
                    if prev < 0 and wrapped:
                        prev = self.per_env_cap - 1
                    if 0 <= prev < buf_size and self.ep_ids[prev, e] == ep:
                        indices.insert(0, prev)
                        c = prev
                    else:
                        indices.insert(0, indices[0])
                win_pos[i] = indices

            all_frame_pos = win_pos.ravel()
            all_frame_env = np.repeat(b_env, sw)
            self._compute_backbone_features(all_frame_pos, all_frame_env)

            cam_features = np.empty(
                (bs, self.num_cameras, sw, self.feat_dim),
                dtype=np.float32,
            )
            for i in range(bs):
                e = int(b_env[i])
                cam_features[i] = self.feat_cache[win_pos[i], e].transpose(1, 0, 2)

            cam_features_t = torch.from_numpy(cam_features).to(self.rm_device)

            state_flat = np.empty((bs, sw * self.robot_dim), dtype=np.float32)
            for i in range(bs):
                state_flat[i] = self.rm_states[win_pos[i], int(b_env[i])].ravel()
            proprio_t = torch.from_numpy(state_flat).to(self.rm_device)

            with torch.no_grad(), torch.amp.autocast(self.rm_device.type):
                rewards, uncertainty = self.rm.get_reward_from_features(
                    cam_features_t,
                    proprio_t,
                    return_uncertainty=True,
                )
            rewards_np = rewards.squeeze(-1).cpu().numpy()
            unc_np = uncertainty.squeeze(-1).cpu().numpy()

            # AMP/fp16 RM backbone can emit inf on OOD frames (observed:
            # reward/rm_mean -> inf at ~65k steps corrupts running stats and
            # propagates NaN through qf_loss/actor_loss). Clamp non-finite
            # values to the running mean (safe fallback) and flag them as
            # maximum uncertainty so downstream confidence weighting can
            # still down-weight them if enabled. Always gate by np.isfinite
            # rather than np.isnan — inf poisons sums the same way NaN does.
            bad = ~np.isfinite(rewards_np)
            if bad.any():
                running_mean_fallback = float(self.rm_running_mean)
                rewards_np = np.where(bad, running_mean_fallback, rewards_np)
                unc_np = np.where(bad, 1.0, unc_np)

            self.rm_rewards[b_pos, b_env] = rewards_np
            self.rm_uncertainty[b_pos, b_env] = unc_np
            self.labeled[b_pos, b_env] = True
            labeled_count += bs

            rs = np.sum(rewards_np, dtype=np.float64)
            rq = np.sum(rewards_np.astype(np.float64) ** 2, dtype=np.float64)
            self._rm_running_sum += rs
            self._rm_running_sq_sum += rq
            self._rm_running_count += len(rewards_np)

        return labeled_count

    def get_rewards(self, batch_inds, env_inds):
        # PyTorch advanced indexing requires index tensors to live on the same
        # device as the indexed tensor. batch_inds/env_inds are stored on CPU
        # (see ReplayBuffer.sample), so move them to the RM device when the
        # reward buffer is on GPU.
        if self._gpu_rewards is not None:
            return self._gpu_rewards[
                batch_inds.to(self.rm_device).long(),
                env_inds.to(self.rm_device).long(),
            ]
        return torch.from_numpy(self.rm_rewards[batch_inds.numpy(), env_inds.numpy()].astype(np.float32))

    def get_labeled_mask(self, batch_inds, env_inds):
        """Return a float mask (1.0 if labeled, 0.0 otherwise) matching get_rewards device."""
        if self._gpu_labeled is not None:
            return self._gpu_labeled[
                batch_inds.to(self.rm_device).long(),
                env_inds.to(self.rm_device).long(),
            ].float()
        return torch.from_numpy(self.labeled[batch_inds.numpy(), env_inds.numpy()].astype(np.float32))

    def get_uncertainty(self, batch_inds, env_inds):
        if self._gpu_uncertainty is not None:
            return self._gpu_uncertainty[
                batch_inds.to(self.rm_device).long(),
                env_inds.to(self.rm_device).long(),
            ]
        return torch.from_numpy(self.rm_uncertainty[batch_inds.numpy(), env_inds.numpy()].astype(np.float32))

    @property
    def rm_running_mean(self) -> float:
        return float(self._rm_running_sum / max(self._rm_running_count, 1))

    @property
    def rm_running_std(self) -> float:
        if self._rm_running_count < 2:
            return 1.0
        m = float(self._rm_running_sum / max(self._rm_running_count, 1))
        var = float(self._rm_running_sq_sum / self._rm_running_count) - m * m
        return max(var, 0.0) ** 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Logger:
    def __init__(self, log_wandb=False, tensorboard=None):
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb

            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def _compute_done_masks(terminations, truncations, strategy):
    if strategy == "never":
        return torch.ones_like(terminations, dtype=torch.bool), truncations | terminations
    elif strategy == "always":
        return truncations | terminations, torch.zeros_like(terminations, dtype=torch.bool)
    else:
        return truncations & (~terminations), terminations


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _torch_cuda_usable() -> bool:
    """True only if allocating a CUDA tensor works (driver/runtime mismatch can still report 'available')."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except RuntimeError:
        return False


def _build_rl_meta(args: Args, global_step: int, *, stage: str) -> dict[str, object]:
    """Return the meta dict embedded into every v2 RL checkpoint."""
    return {
        "agent": "sac_v2",
        "stage": stage,
        "env_id": args.env_id,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "global_step": int(global_step),
        "rm_checkpoint": args.rm_checkpoint,
        "rm_alpha": args.rm_alpha,
        "rm_normalize": args.rm_normalize,
        "rm_clip": args.rm_clip,
        "rm_uncertainty_weight": args.rm_uncertainty_weight,
        "rm_potential_shaping": args.rm_potential_shaping,
        "seed": args.seed,
        "autotune": args.autotune,
    }


def _make_envs(args: Args, run_name: str):
    """Create train and eval ManiSkill3 environments."""
    env_kwargs: dict = dict(
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        sensor_configs={},
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids
    if args.camera_width is not None:
        env_kwargs["sensor_configs"]["width"] = args.camera_width
    if args.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = args.camera_height

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )

    use_rgb = args.obs_mode in ("rgb", "rgbd")
    use_depth = args.obs_mode in ("depth", "rgbd")
    envs = FlattenRGBDObservationWrapper(envs, rgb=use_rgb, depth=use_depth, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=use_rgb, depth=use_depth, state=args.include_state)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate and args.checkpoint:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        if args.save_train_video_freq is not None:
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=lambda x: (x // args.num_steps) % args.save_train_video_freq == 0,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=False,
            save_video=True,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    return envs, eval_envs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_eval(
    actor: Actor,
    eval_envs,
    num_steps: int,
    logger: Logger | None,
    global_step: int,
) -> dict[str, float]:
    """Run evaluation and return mean metrics."""
    actor.eval()
    eval_obs, _ = eval_envs.reset()
    eval_metrics: dict[str, list] = defaultdict(list)
    for _ in range(num_steps):
        with torch.no_grad():
            eval_obs, _, _, _, eval_infos = eval_envs.step(actor.get_eval_action(eval_obs))
        if "final_info" in eval_infos:
            for k, v in eval_infos["final_info"]["episode"].items():
                eval_metrics[k].append(v)
    eval_means: dict[str, float] = {}
    for k, v in eval_metrics.items():
        m = torch.stack(v).float().mean().item()
        eval_means[k] = m
        if logger:
            logger.add_scalar(f"eval/{k}", m, global_step)
    actor.train()
    return eval_means


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args: Args) -> dict[str, float]:
    """Run SAC training and return final eval metrics."""
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Default training path is GPU-only: env (gpu sim), replay buffer, policy, and RM relabel all expect CUDA.
    if args.cuda:
        if not _torch_cuda_usable():
            raise RuntimeError(
                "GPU training requires working CUDA (driver + PyTorch runtime). Allocation on cuda:0 failed. "
                "Typical fix: upgrade NVIDIA drivers, or install a PyTorch build matching your driver — "
                "https://pytorch.org/get-started/locally/"
            )
        if args.sim_backend != "gpu":
            raise ValueError(
                f"With cuda=True, use sim_backend='gpu' for GPU simulation (got {args.sim_backend!r}). "
                "Use --no-cuda and sim_backend=cpu only for CPU debug."
            )
        if args.buffer_device != "cuda":
            raise ValueError(f"With cuda=True, use buffer_device='cuda' for GPU replay (got {args.buffer_device!r}).")
        device = torch.device("cuda")
    else:
        if args.sim_backend != "cpu":
            raise ValueError("With cuda=False (--no-cuda), use sim_backend='cpu' (gpu sim still needs CUDA).")
        if args.buffer_device != "cpu":
            raise ValueError("With cuda=False, use buffer_device='cpu'.")
        device = torch.device("cpu")

    envs, eval_envs = _make_envs(args, run_name)

    # -- logging --
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["sac", "vision"],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n{}".format("\n".join(f"|{k}|{v}|" for k, v in vars(args).items())),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # -- replay buffer --
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
    )

    # -- networks --
    obs, info = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    actor = Actor(envs, sample_obs=obs).to(device)
    qf1 = SoftQNetwork(envs, actor.encoder).to(device)
    qf2 = SoftQNetwork(envs, actor.encoder).to(device)
    qf1_target = SoftQNetwork(envs, actor.encoder).to(device)
    qf2_target = SoftQNetwork(envs, actor.encoder).to(device)
    if args.checkpoint is not None:
        ckpt, ckpt_meta = load_rl_checkpoint(args.checkpoint, map_location=device)
        if ckpt_meta:
            print(f"[ckpt] Loaded artifact meta: {ckpt_meta}")
        actor.load_state_dict(ckpt["actor"])
        # Backward compat: pre-fix checkpoints only stored *_target weights (bug).
        # New format stores both online (qf{1,2}) and target (qf{1,2}_target) weights.
        has_online = "qf1" in ckpt and "qf2" in ckpt
        has_target = "qf1_target" in ckpt and "qf2_target" in ckpt
        if has_online:
            qf1.load_state_dict(ckpt["qf1"])
            qf2.load_state_dict(ckpt["qf2"])
            if has_target:
                qf1_target.load_state_dict(ckpt["qf1_target"])
                qf2_target.load_state_dict(ckpt["qf2_target"])
            else:
                qf1_target.load_state_dict(qf1.state_dict())
                qf2_target.load_state_dict(qf2.state_dict())
        elif has_target:
            # Legacy checkpoint: seed both online and target from saved target weights.
            print("[ckpt] Legacy checkpoint detected (no qf1/qf2 keys); seeding online networks from target weights.")
            qf1.load_state_dict(ckpt["qf1_target"])
            qf2.load_state_dict(ckpt["qf2_target"])
            qf1_target.load_state_dict(ckpt["qf1_target"])
            qf2_target.load_state_dict(ckpt["qf2_target"])
        else:
            raise KeyError(f"Checkpoint {args.checkpoint} missing both online and target Q-networks")
    else:
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

    if args.compile:
        print("[Compile] Compiling actor and critic networks with torch.compile...")
        actor = torch.compile(actor)
        qf1 = torch.compile(qf1)
        qf2 = torch.compile(qf2)
        qf1_target = torch.compile(qf1_target)
        qf2_target = torch.compile(qf2_target)

    # SAC-AE style: the visual encoder is shared between actor and critics. We
    # assign its parameters **only** to the critic optimiser so q-loss is the
    # single signal that trains the encoder (the established recipe). Including
    # the encoder in both optimisers would cause actor_optimizer.zero_grad() to
    # wipe gradients accumulated by q_optimizer.backward() and vice versa,
    # making one of the two updates silently no-op on encoder weights.
    encoder_params = list(actor.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    actor_only_params = [p for p in actor.parameters() if id(p) not in encoder_param_ids]
    q_optimizer = optim.Adam(
        list(qf1.mlp.parameters()) + list(qf2.mlp.parameters()) + encoder_params,
        lr=args.q_lr,
    )
    actor_optimizer = optim.Adam(actor_only_params, lr=args.policy_lr)

    log_alpha = torch.zeros(1, device=device)
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)

    # -- RM relabeler --
    rm_relabeler = None
    if args.rm_checkpoint is not None:
        rgb_shape = obs.get("rgb")
        num_cameras = rgb_shape.shape[-1] // 3 if rgb_shape is not None else 1
        rm_relabeler = RMRelabeler(
            rm_checkpoint=args.rm_checkpoint,
            per_env_capacity=rb.per_env_buffer_size,
            num_envs=args.num_envs,
            num_cameras=num_cameras,
            rm_img_size=args.rm_img_size,
            device=str(device),
        )
        print(f"[RM] Loaded from {args.rm_checkpoint}, alpha={args.rm_alpha}")

    ep_ids = np.zeros(args.num_envs, dtype=np.int64)
    ep_counter = 0

    # -- training loop --
    global_step = 0
    global_update = 0
    iteration = 0
    learning_has_started = False
    actor_loss: torch.Tensor | float = 0.0
    last_eval: dict[str, float] = {}
    # Edge-triggered counter for RM relabeling. `global_step` advances by
    # `num_envs * steps_per_env` each iteration, so `global_step %
    # relabel_interval == 0` may never be satisfied (e.g. step=64 vs
    # interval=500). Track the last trigger and fire whenever the gap reaches
    # relabel_interval.
    last_relabel_step = 0
    pbar = tqdm.tqdm(range(args.total_timesteps), desc="SAC training")
    cumulative_times: dict[str, float] = defaultdict(float)
    start_time = time.perf_counter()

    while global_step < args.total_timesteps:
        iteration += 1

        # -- eval --
        if args.eval_freq > 0 and iteration % args.eval_freq == 1:
            last_eval = run_eval(actor, eval_envs, args.num_eval_steps, logger, global_step)
            pbar.set_description(
                f"success: {last_eval.get('success_once', 0):.2f}, ret: {last_eval.get('return', 0):.2f}"
            )
            if args.evaluate:
                break
            if args.save_model:
                save_rl_checkpoint(
                    f"runs/{run_name}/ckpt_{global_step}.pt",
                    state={
                        "actor": actor.state_dict(),
                        "qf1": qf1.state_dict(),
                        "qf2": qf2.state_dict(),
                        "qf1_target": qf1_target.state_dict(),
                        "qf2_target": qf2_target.state_dict(),
                        "log_alpha": log_alpha if args.autotune else None,
                    },
                    meta=_build_rl_meta(args, global_step, stage="checkpoint"),
                )

        # -- rollout --
        rollout_time = time.perf_counter()
        for _ in range(args.steps_per_env):
            global_step += args.num_envs
            if not learning_has_started:
                actions = 2 * torch.rand(envs.action_space.shape, dtype=torch.float32, device=device) - 1
            else:
                actions, _, _, _ = actor.get_action(obs)
                actions = actions.detach()

            # Capture pre-step proprio (s_t) BEFORE envs.step so it is time-aligned
            # with obs["rgb"] (also s_t). RM training expects same-timestep (img, proprio)
            # pairing — see data/dataset.py:__getitem__ where indices are shared.
            pre_step_rm_state = _extract_rm_state_batch(envs) if rm_relabeler is not None else None

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            need_final_obs, stop_bootstrap = _compute_done_masks(terminations, truncations, args.bootstrap_at_done)

            # Snapshot ep_ids BEFORE done handling: the transition being stored
            # is (s_t, a_t, r_t) where s_t belongs to the *current* episode, so
            # the RM temporal window must be keyed by the pre-done episode id.
            # Incrementing ep_ids first would label terminal frames as the new
            # episode and break RMRelabeler._get_temporal_window matching.
            ep_ids_for_store = ep_ids.copy() if rm_relabeler is not None else None

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                real_next_obs = {k: v.clone() for k, v in next_obs.items()}
                for k in real_next_obs:
                    real_next_obs[k][need_final_obs] = infos["final_observation"][k][need_final_obs]
                if logger:
                    for k, v in infos["final_info"]["episode"].items():
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                done_envs = torch.where(done_mask)[0]
                for env_i in done_envs.cpu().tolist():
                    ep_counter += 1
                    ep_ids[env_i] = ep_counter
            else:
                real_next_obs = next_obs

            buf_pos = rb.pos
            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)

            if rm_relabeler is not None:
                rm_relabeler.store(buf_pos, obs["rgb"].cpu(), pre_step_rm_state, ep_ids_for_store)

            obs = next_obs

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # -- RM relabeling (async: launch background, overlap with gradient updates) --
        if rm_relabeler is not None and global_step - last_relabel_step >= args.relabel_interval:
            n_prev = rm_relabeler.wait_relabel()
            if n_prev > 0 and logger:
                logger.add_scalar("rm/labeled_count", n_prev, global_step)
            buf_size = rb.per_env_buffer_size if rb.full else rb.pos
            if buf_size > 0:
                rm_relabeler.relabel_async(buf_size)
            last_relabel_step = global_step

        # -- gradient updates --
        if global_step < args.learning_starts:
            continue
        update_time = time.perf_counter()
        learning_has_started = True

        prefetch_data = rb.sample(args.batch_size)

        for _ in range(args.grad_steps_per_iteration):
            global_update += 1
            data = prefetch_data
            prefetch_data = rb.sample(args.batch_size)

            env_rewards = data.rewards.flatten()
            if rm_relabeler is not None:
                rm_r = rm_relabeler.get_rewards(data.batch_inds, data.env_inds)
                if rm_r.device != device:
                    rm_r = rm_r.to(device)

                # Fallback for unlabeled transitions: use running mean instead of 0.
                # Storing 0 for unlabeled samples (see RMRelabeler.store) and feeding
                # them to SAC biases Q-learning against the most recent rollouts
                # (which are the ones still waiting for async relabel).
                labeled_mask = rm_relabeler.get_labeled_mask(data.batch_inds, data.env_inds)
                if labeled_mask.device != device:
                    labeled_mask = labeled_mask.to(device)

                unc = (
                    rm_relabeler.get_uncertainty(data.batch_inds, data.env_inds) if args.rm_uncertainty_weight else None
                )
                if unc is not None and unc.device != device:
                    unc = unc.to(device)

                rm_r = postprocess_rm_reward(
                    rm_r,
                    labeled_mask=labeled_mask,
                    running_mean=rm_relabeler.rm_running_mean,
                    running_std=rm_relabeler.rm_running_std,
                    normalize=args.rm_normalize,
                    clip=args.rm_clip,
                    uncertainty=unc,
                )
                alpha_eff = effective_alpha(global_step, args.rm_alpha_warmup, args.rm_alpha)

                if args.rm_potential_shaping:
                    # Potential-based shaping (Ng, Harada, Russell 1999):
                    #   F(s, a, s') = γ·Φ(s') − Φ(s), with Φ(s) := RM(s).
                    # Preserves optimal policy in γ-discounted MDPs and removes
                    # the raw RM offset. next_batch_inds wraps via
                    # per_env_buffer_size; slots at rb.pos that are pending
                    # relabel fall back to running_mean via labeled_mask_next,
                    # which is the best unbiased estimate before the async
                    # relabel thread catches up. Terminal transitions use
                    # Φ(s')=0 so shaping vanishes at episode boundaries.
                    next_batch_inds = (data.batch_inds + 1) % rb.per_env_buffer_size
                    rm_r_next = rm_relabeler.get_rewards(next_batch_inds, data.env_inds)
                    if rm_r_next.device != device:
                        rm_r_next = rm_r_next.to(device)
                    labeled_mask_next = rm_relabeler.get_labeled_mask(next_batch_inds, data.env_inds)
                    if labeled_mask_next.device != device:
                        labeled_mask_next = labeled_mask_next.to(device)
                    unc_next = (
                        rm_relabeler.get_uncertainty(next_batch_inds, data.env_inds)
                        if args.rm_uncertainty_weight
                        else None
                    )
                    if unc_next is not None and unc_next.device != device:
                        unc_next = unc_next.to(device)
                    rm_r_next = postprocess_rm_reward(
                        rm_r_next,
                        labeled_mask=labeled_mask_next,
                        running_mean=rm_relabeler.rm_running_mean,
                        running_std=rm_relabeler.rm_running_std,
                        normalize=args.rm_normalize,
                        clip=args.rm_clip,
                        uncertainty=unc_next,
                    )
                    done_flat = data.dones.flatten()
                    shaped_rm = args.gamma * rm_r_next * (1.0 - done_flat) - rm_r
                    train_rewards = combine_env_and_rm(
                        env_rewards, shaped_rm, alpha=alpha_eff, env_scale=args.env_reward_scale
                    )
                    # keep rm_r pointing to the effective shaping term for
                    # logging consistency (reward/rm_mean reflects what the
                    # critic actually sees under shaping)
                    rm_r = shaped_rm
                else:
                    # env_reward_scale=0.0 → pure RM reward (key ablation for AutoRM validation)
                    train_rewards = combine_env_and_rm(
                        env_rewards, rm_r, alpha=alpha_eff, env_scale=args.env_reward_scale
                    )
            else:
                rm_r = torch.zeros_like(env_rewards)
                train_rewards = env_rewards * args.env_reward_scale

            with torch.no_grad(), amp_ctx:
                na, nlp, _, vfn = actor.get_action(data.next_obs)
                q1n = qf1_target(data.next_obs, na, vfn)
                q2n = qf2_target(data.next_obs, na, vfn)
                next_q = train_rewards + (1 - data.dones.flatten()) * args.gamma * (
                    torch.min(q1n, q2n) - alpha * nlp
                ).view(-1)

            with amp_ctx:
                vf = actor.encoder(data.obs)
                q1v = qf1(data.obs, data.actions, vf).view(-1)
                q2v = qf2(data.obs, data.actions, vf).view(-1)
                qf_loss = F.mse_loss(q1v, next_q) + F.mse_loss(q2v, next_q)
            q_optimizer.zero_grad()
            scaler.scale(qf_loss).backward()
            scaler.step(q_optimizer)

            if global_update % args.policy_frequency == 0:
                with amp_ctx:
                    # detach_encoder=True: the encoder is owned by q_optimizer
                    # (see SAC-AE note above). Keeping gradients flowing through
                    # actor.encoder here would waste memory/compute and leave
                    # stale grads on encoder weights.
                    pi, lp, _, vfp = actor.get_action(data.obs, detach_encoder=True)
                    actor_loss = (
                        (alpha * lp)
                        - torch.min(
                            qf1(data.obs, pi, vfp, detach_encoder=True),
                            qf2(data.obs, pi, vfp, detach_encoder=True),
                        ).view(-1)
                    ).mean()
                actor_optimizer.zero_grad()
                scaler.scale(actor_loss).backward()
                scaler.step(actor_optimizer)

                if args.autotune:
                    alpha_loss = (-log_alpha.exp() * (lp.detach() + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            scaler.update()

            if global_update % args.target_network_frequency == 0:
                for p, tp in zip(qf1.parameters(), qf1_target.parameters(), strict=False):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                for p, tp in zip(qf2.parameters(), qf2_target.parameters(), strict=False):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        if logger and (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            elapsed = time.perf_counter() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar(
                "losses/actor_loss",
                actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
                global_step,
            )
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("time/rollout_fps", args.num_envs * args.steps_per_env / rollout_time, global_step)
            logger.add_scalar("reward/env_mean", env_rewards.mean().item(), global_step)
            logger.add_scalar("reward/rm_mean", rm_r.mean().item(), global_step)
            logger.add_scalar("reward/combined_mean", train_rewards.mean().item(), global_step)

    # -- save final model --
    if not args.evaluate and args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path = f"runs/{run_name}/final_ckpt.pt"
        save_rl_checkpoint(
            model_path,
            state={
                "actor": actor.state_dict(),
                "qf1": qf1.state_dict(),
                "qf2": qf2.state_dict(),
                "qf1_target": qf1_target.state_dict(),
                "qf2_target": qf2_target.state_dict(),
                "log_alpha": log_alpha if args.autotune else None,
            },
            meta=_build_rl_meta(args, global_step, stage="final"),
        )
        print(f"model saved to {model_path}")

    if logger:
        logger.close()
    envs.close()
    eval_envs.close()

    total_time = time.perf_counter() - start_time
    print(f"Training complete: {global_step} steps in {total_time:.0f}s ({int(global_step / total_time)} SPS)")
    return last_eval


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    train(tyro.cli(Args))
