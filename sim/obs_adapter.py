"""Adapt ManiSkill3 observations into RewardModel input tensors.

Maintains a sliding window of recent frames so that the RM receives the
same ``(images, proprio)`` layout it was trained on.

Image layout expected by RM:
    ``[B, N_CAM * T * 3, H, W]``  (cameras are outer blocks, time steps inner)

Proprio layout:
    ``[B, robot_dim * T]``  (T consecutive state vectors flattened)
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from data.common import IMG_SIZE_RM, STATE_WINDOWS_DEFAULT
from sim.task_configs import SimTaskConfig

# ---------------------------------------------------------------------------
# State extraction helpers (adapted from 1_collect_dp_data.py)
# ---------------------------------------------------------------------------


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion ``[qw, qx, qy, qz]`` to axis-angle (3-D)."""
    qw = float(np.clip(quat[0], -1.0, 1.0))
    den = np.sqrt(max(1.0 - qw * qw, 0.0))
    if den < 1e-7:
        return np.zeros(3, dtype=np.float32)
    return (quat[1:4] * 2.0 * math.acos(qw) / den).astype(np.float32)


def extract_eef_state(env) -> np.ndarray:
    """8-D EEF state: ``[xyz, axis-angle, finger_left, finger_right]``."""
    tcp_pose = env.agent.tcp.pose.raw_pose.squeeze(0).cpu().numpy()
    eef_pos = tcp_pose[:3]
    eef_rot = _quat2axisangle(tcp_pose[3:7])
    qpos = env.agent.robot.get_qpos().squeeze(0).cpu().numpy()
    fingers = qpos[-2:].astype(np.float32)
    return np.concatenate([eef_pos, eef_rot, fingers]).astype(np.float32)


def extract_joint_state(env) -> np.ndarray:
    """9-D joint state: ``[joint1..7, finger_left, finger_right]``."""
    return env.agent.robot.get_qpos().squeeze(0).cpu().numpy().astype(np.float32)


def extract_state(env, mode: str = "eef+joint") -> np.ndarray:
    """Extract robot state vector matching the RM training format.

    ``eef+joint`` yields the same 17-D vector used by the PushCube RM.
    """
    if mode == "eef+joint":
        return np.concatenate([extract_eef_state(env), extract_joint_state(env)])
    if mode == "eef":
        return extract_eef_state(env)
    if mode == "joint":
        return extract_joint_state(env)
    raise ValueError(f"Unknown state_mode: {mode}")


def extract_rgb(obs: dict, cam_ms_name: str) -> np.ndarray:
    """``(H, W, 3)`` uint8 image from ManiSkill obs dict."""
    img = obs["sensor_data"][cam_ms_name]["rgb"]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img.squeeze(0)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and (img.max() <= 1.0 or img.max() == 0.0):
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def extract_full_state(obs: dict) -> np.ndarray:
    """Flatten ManiSkill ``state`` obs for PPO (includes task-specific extras)."""
    state = obs["state"] if isinstance(obs, dict) and "state" in obs else obs
    if isinstance(state, torch.Tensor):
        return state.squeeze(0).cpu().float().numpy().astype(np.float32)
    return np.asarray(state, dtype=np.float32).flatten()


# ---------------------------------------------------------------------------
# Observation adapter with sliding window
# ---------------------------------------------------------------------------


class ObsAdapter:
    """Converts ManiSkill3 observations into RM input tensors.

    Call :meth:`reset` at episode start, then :meth:`step` after each
    ``env.step``.  The adapter maintains a deque of the last
    ``state_windows`` observations and builds the correctly-shaped tensors
    for ``RewardModel.get_reward(images, proprio)``.
    """

    def __init__(
        self,
        cfg: SimTaskConfig,
        state_windows: int = STATE_WINDOWS_DEFAULT,
        img_size: int = IMG_SIZE_RM,
        device: torch.device | str = "cuda",
    ):
        self.cfg = cfg
        self.state_windows = state_windows
        self.img_size = img_size
        self.device = torch.device(device)

        self._cam_order = sorted(cfg.camera_map.keys())

        self._img_buf: deque[list[torch.Tensor]] = deque(maxlen=state_windows)
        self._state_buf: deque[np.ndarray] = deque(maxlen=state_windows)

    def reset(self, env, obs: dict) -> None:
        """Initialise buffers with the first observation (replicated T times)."""
        state = extract_state(env, self.cfg.state_mode)
        imgs = self._extract_cam_tensors(obs)

        self._state_buf.clear()
        self._img_buf.clear()
        for _ in range(self.state_windows):
            self._state_buf.append(state)
            self._img_buf.append(imgs)

    def step(self, env, obs: dict) -> None:
        """Push new observation into the sliding window."""
        self._state_buf.append(extract_state(env, self.cfg.state_mode))
        self._img_buf.append(self._extract_cam_tensors(obs))

    def get_rm_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(images, proprio)`` tensors for ``RewardModel.get_reward``.

        Returns:
            images: ``[1, N_CAM * T * 3, H, W]``
            proprio: ``[1, robot_dim * T]``
        """
        cam_frames: list[list[torch.Tensor]] = [[] for _ in self._cam_order]
        for frame_imgs in self._img_buf:
            for cam_idx, img_t in enumerate(frame_imgs):
                cam_frames[cam_idx].append(img_t)

        all_cam_flat: list[torch.Tensor] = []
        for per_cam in cam_frames:
            stacked = torch.stack(per_cam)  # [T, 3, H, W]
            all_cam_flat.append(stacked.reshape(-1, self.img_size, self.img_size))  # [T*3, H, W]

        images = torch.cat(all_cam_flat, dim=0).unsqueeze(0).to(self.device)  # [1, N_CAM*T*3, H, W]

        proprio_np = np.concatenate(list(self._state_buf))  # [robot_dim * T]
        proprio = torch.from_numpy(proprio_np).float().unsqueeze(0).to(self.device)  # [1, robot_dim*T]

        return images, proprio

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_cam_tensors(self, obs: dict) -> list[torch.Tensor]:
        """Extract per-camera ``[3, H, W]`` float tensors in canonical order."""
        tensors = []
        for cam_ms in self._cam_order:
            rgb = extract_rgb(obs, cam_ms)  # (H, W, 3) uint8
            t = torch.from_numpy(rgb).float().div_(255.0)  # (H, W, 3) [0,1]
            t = t.permute(2, 0, 1)  # (3, H, W)
            if t.shape[-1] != self.img_size or t.shape[-2] != self.img_size:
                t = F.interpolate(
                    t.unsqueeze(0),
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            tensors.append(t)
        return tensors
