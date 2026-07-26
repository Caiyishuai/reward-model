"""High-performance inference wrapper for the Reward Model.

Features: automatic history management, FP16, torch.compile, multi-camera.
"""

from collections import deque

import numpy as np
import torch
import torchvision.transforms.functional as F

from reward_model import RewardModel


class RewardModelInference:
    """Online inference wrapper with ring-buffer history management."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda:0",
        img_size: int = 224,
        use_fp16: bool = True,
        compile_model: bool = False,
    ):
        self.device = torch.device(device)
        self.img_size = img_size
        self.use_fp16 = use_fp16

        self.model = RewardModel.load(ckpt_path, device=device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.num_cameras = self.model.num_cameras
        self.window_size = self.model.state_windows

        if use_fp16:
            self.model.half()

        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except RuntimeError as e:
                print(f"[Inference] Compilation failed, using eager mode: {e}")

        self.img_buffers: list[deque] = [deque(maxlen=self.window_size) for _ in range(self.num_cameras)]
        self.prop_buffer: deque = deque(maxlen=self.window_size)

    def reset(self) -> None:
        """Clear history buffers at episode start."""
        for buf in self.img_buffers:
            buf.clear()
        self.prop_buffer.clear()

    def update(
        self,
        obs_imgs: np.ndarray | list[np.ndarray] | torch.Tensor | list[torch.Tensor],
        obs_prop: np.ndarray | torch.Tensor,
    ) -> float:
        """Push observation, return current reward.

        Args:
            obs_imgs: single [H,W,3] or list of [H,W,3] per camera
            obs_prop: [D] proprioceptive state
        """
        if not isinstance(obs_imgs, list):
            obs_imgs = [obs_imgs]

        if len(obs_imgs) != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} images, got {len(obs_imgs)}")

        processed = [self._process_image(img) for img in obs_imgs]
        prop_t = self._process_proprio(obs_prop)

        for i in range(self.num_cameras):
            self.img_buffers[i].append(processed[i])
        self.prop_buffer.append(prop_t)

        if len(self.prop_buffer) < self.window_size:
            pad_count = self.window_size - len(self.prop_buffer)
            first_prop = self.prop_buffer[0]
            first_imgs = [self.img_buffers[i][0] for i in range(self.num_cameras)]
            for _ in range(pad_count):
                for i in range(self.num_cameras):
                    self.img_buffers[i].appendleft(first_imgs[i])
                self.prop_buffer.appendleft(first_prop)

        all_cam = [torch.cat(list(self.img_buffers[i]), dim=1) for i in range(self.num_cameras)]
        final_img = torch.cat(all_cam, dim=1)

        props = torch.cat(list(self.prop_buffer), dim=1).flatten().unsqueeze(0)

        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    return self._compute_reward(final_img, props)
            return self._compute_reward(final_img, props)

    def _compute_reward(self, imgs: torch.Tensor, props: torch.Tensor) -> float:
        out = self.model(imgs, props)
        raw_rewards = out[0] if isinstance(out, tuple) else out
        min_reward, _ = torch.min(raw_rewards, dim=1)
        return self.model.unnormalize_reward(min_reward).item()

    def _process_image(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        elif isinstance(img, torch.Tensor):
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            if img.ndim == 3 and img.shape[0] != 3:
                img = img.permute(2, 0, 1)

        if img.shape[-1] != self.img_size or img.shape[-2] != self.img_size:
            img = F.resize(img, [self.img_size, self.img_size], antialias=True)

        img = img.unsqueeze(0).to(self.device)
        if self.use_fp16:
            img = img.half()
        return img

    def _process_proprio(self, prop: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(prop, np.ndarray):
            prop = torch.from_numpy(prop).float()
        prop = prop.unsqueeze(0).to(self.device)
        if self.use_fp16:
            prop = prop.half()
        return prop
