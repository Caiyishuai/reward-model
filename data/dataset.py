"""Balanced training dataset and augmentation pipeline for the Reward Model."""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop as _RandomResizedCrop

from data.common import IMG_SIZE_RM, STATE_WINDOWS_DEFAULT, load_pickle


FRAME_SPLIT_STRATEGIES = ("random", "temporal", "strided")


def split_frame_end_indices(
    indices: list[int],
    split_ratio: float,
    strategy: str,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split one trajectory's candidate sample-ending frames into train/test.

    The returned lists partition ``indices`` exactly.  ``random`` is the
    default seeded permutation, ``temporal`` holds out the final frames, and
    ``strided`` spreads held-out frames approximately uniformly over time.
    """
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1); got {split_ratio}")
    if strategy not in FRAME_SPLIT_STRATEGIES:
        raise ValueError(
            f"frame_split_strategy must be one of {FRAME_SPLIT_STRATEGIES}; got {strategy!r}"
        )
    if len(indices) < 2:
        raise ValueError("each trajectory needs at least two candidate frame endpoints")

    n_train = min(max(int(len(indices) * split_ratio), 1), len(indices) - 1)
    ordered = np.asarray(indices, dtype=np.int64)
    if strategy == "random":
        permutation = np.random.default_rng(seed).permutation(len(ordered))
        train_positions = set(permutation[:n_train].tolist())
    elif strategy == "temporal":
        train_positions = set(range(n_train))
    else:
        n_test = len(ordered) - n_train
        test_positions = set(
            np.floor((np.arange(n_test) + 0.5) * len(ordered) / n_test)
            .astype(np.int64)
            .tolist()
        )
        train_positions = set(range(len(ordered))) - test_positions

    train = [int(value) for position, value in enumerate(ordered) if position in train_positions]
    test = [int(value) for position, value in enumerate(ordered) if position not in train_positions]
    return train, test


class BalancedLeRobotDataset(Dataset):
    """Balanced multi-camera dataset with episode- or within-episode splits."""

    def __init__(
        self,
        fail_path: str,
        success_path: str,
        camera_keys: list[str],
        split: str = "train",
        split_ratio: float = 0.9,
        target_ratio: float = 1.0,
        epoch_size: int = 4000,
        window_size: int = STATE_WINDOWS_DEFAULT,
        img_size: int = IMG_SIZE_RM,
        transform: nn.Module | None = None,
        seed: int = 42,
        max_reward: float = 6.0,
        min_reward: float = 0.0,
        future_steps: int = 3,
        frame_split_strategy: str | None = None,
    ):
        self.window_size = window_size
        self.future_steps = future_steps
        self.transform = transform
        self.img_size = img_size
        self.split = split
        self.epoch_size = epoch_size
        self.target_ratio = target_ratio
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.camera_keys = camera_keys
        self.frame_split_strategy = frame_split_strategy

        self.num_success = int(epoch_size / (1 + target_ratio))
        self.num_fail = epoch_size - self.num_success

        is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
        if is_main:
            print(f"[Dataset] {split.upper()}: {self.num_success} success, {self.num_fail} fail per epoch")

        self.data_store: dict[str, dict] = {}
        self.reward_stats = {"min": float("inf"), "max": float("-inf")}

        for name, path in [("fail", fail_path), ("success", success_path)]:
            if os.path.exists(path):
                raw = load_pickle(path)
                episode_indices = raw["episode_index"]
                unique_eps = np.unique(episode_indices)

                rewards = raw["next.reward"]
                valid_indices = []
                if frame_split_strategy is None:
                    rng = np.random.default_rng(seed)
                    rng.shuffle(unique_eps)
                    split_idx = int(len(unique_eps) * split_ratio)
                    target_eps = unique_eps[:split_idx] if split == "train" else unique_eps[split_idx:]
                    target_eps_set = set(target_eps)
                    for i in range(window_size - 1, len(episode_indices)):
                        if episode_indices[i] in target_eps_set:
                            start = i - window_size + 1
                            if episode_indices[start] == episode_indices[i]:
                                valid_indices.append(i)
                    split_description = f"{len(target_eps)} in {split}"
                else:
                    if split not in {"train", "val", "test"}:
                        raise ValueError(f"frame-level splitting does not support split={split!r}")
                    for episode_id in unique_eps:
                        episode_candidates = np.flatnonzero(episode_indices == episode_id).tolist()
                        episode_candidates = episode_candidates[window_size - 1 :]
                        episode_seed = int(
                            np.random.SeedSequence([seed, int(episode_id)]).generate_state(1)[0]
                        )
                        train_indices, test_indices = split_frame_end_indices(
                            episode_candidates,
                            split_ratio,
                            frame_split_strategy,
                            episode_seed,
                        )
                        valid_indices.extend(train_indices if split == "train" else test_indices)
                    split_description = (
                        f"all {len(unique_eps)} eps, {len(valid_indices)} {split} frame endpoints "
                        f"({frame_split_strategy})"
                    )

                for i in valid_indices:
                    r = rewards[i]
                    self.reward_stats["min"] = min(self.reward_stats["min"], r)
                    self.reward_stats["max"] = max(self.reward_stats["max"], r)

                if is_main:
                    print(f"  {name}: {len(unique_eps)} eps total, {split_description}")

                self.data_store[name] = {"raw": raw, "valid_indices": valid_indices}
            else:
                self.data_store[name] = {"raw": None, "valid_indices": []}

        self._compute_stats()
        self._resample(seed)

    def _resample(self, seed: int) -> None:
        self.samples: list[tuple[str, int]] = []
        rng = np.random.default_rng(seed)

        for name, count in [("success", self.num_success), ("fail", self.num_fail)]:
            pool = self.data_store[name]["valid_indices"]
            if not pool:
                continue
            # Validation / test splits must iterate the entire unique pool
            # each evaluation. Sub-sampling here would (a) make val metrics
            # non-comparable across runs/epochs when count < len(pool), and
            # (b) introduce duplicates via replace=True when count > len(pool),
            # artificially lowering the measured variance. Always use the full
            # pool so val/test stays an unbiased generalisation estimate.
            chosen = rng.choice(pool, size=count, replace=True) if self.split == "train" else pool
            self.samples.extend((name, idx) for idx in chosen)

        if self.split == "train":
            rng.shuffle(self.samples)

    def _compute_stats(self) -> None:
        all_states = []
        for cat in ("fail", "success"):
            if self.data_store[cat]["raw"] and self.data_store[cat]["valid_indices"]:
                raw = self.data_store[cat]["raw"]
                indices = self.data_store[cat]["valid_indices"]
                all_states.append(raw["observation.state"][indices])

        if all_states:
            full = np.concatenate(all_states, axis=0)
            self.state_dim = full.shape[1]
            self.mean = torch.tensor(np.mean(full, axis=0), dtype=torch.float32)
            self.std = torch.tensor(np.std(full, axis=0), dtype=torch.float32)
            self.std[self.std < 1e-5] = 1.0
        else:
            self.state_dim = self._infer_state_dim()
            self.mean = torch.zeros(self.state_dim)
            self.std = torch.ones(self.state_dim)
        self.action_dim = self._infer_action_dim()

    def _infer_state_dim(self) -> int:
        """Infer single-step state dimension from loaded data."""
        for cat in ("success", "fail"):
            raw = self.data_store[cat].get("raw")
            if raw and "observation.state" in raw and len(raw["observation.state"]) > 0:
                return raw["observation.state"].shape[1]
        return 19

    def _infer_action_dim(self) -> int:
        """Infer action dimension so the auxiliary dynamics head matches data."""
        for cat in ("success", "fail"):
            raw = self.data_store[cat].get("raw")
            if raw and "action" in raw and len(raw["action"]) > 0:
                return int(raw["action"].shape[1])
        return 7

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        cat, end_idx = self.samples[idx]
        data = self.data_store[cat]["raw"]
        indices = np.arange(end_idx - self.window_size + 1, end_idx + 1)

        all_cam_flat: list[Tensor] = []
        for cam_key in self.camera_keys:
            if cam_key in data:
                camera_data = data[cam_key]
                imgs_np = (
                    camera_data[indices]
                    if isinstance(camera_data, np.ndarray)
                    else np.stack([camera_data[int(index)] for index in indices])
                )
            else:
                imgs_np = np.zeros((self.window_size, self.img_size, self.img_size, 3), dtype=np.uint8)

            imgs = torch.from_numpy(imgs_np).float().div_(255.0).permute(0, 3, 1, 2)

            if imgs.shape[-1] != self.img_size:
                imgs = F.interpolate(imgs, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

            if self.transform:
                imgs = self.transform(imgs)

            all_cam_flat.append(imgs.reshape(-1, imgs.shape[2], imgs.shape[3]))

        final_img = torch.cat(all_cam_flat, dim=0)

        proprio = torch.from_numpy(data["observation.state"][indices]).float().flatten()

        raw_reward = data["next.reward"][end_idx]
        norm_reward = self._normalize_reward(raw_reward)
        label = torch.tensor([norm_reward], dtype=torch.float32)

        action = torch.from_numpy(data["action"][end_idx]).float() if "action" in data else torch.zeros(7)

        ep_id = data["episode_index"][end_idx]
        future_idx = end_idx + self.future_steps
        max_idx = len(data["observation.state"]) - 1
        valid_index_set = self.data_store[cat].setdefault(
            "valid_index_set", set(self.data_store[cat]["valid_indices"])
        )
        if (
            future_idx > max_idx
            or data["episode_index"][min(future_idx, max_idx)] != ep_id
            or future_idx not in valid_index_set
        ):
            future_idx = end_idx
        future_state = torch.from_numpy(data["observation.state"][future_idx]).float()

        type_int = torch.tensor([1 if cat == "success" else 0], dtype=torch.long)
        return final_img, proprio, label, type_int, action, future_state

    def _normalize_reward(self, raw: float) -> float:
        rng = self.max_reward - self.min_reward
        if rng < 1e-8:
            return 0.0
        norm = 2.0 * (raw - self.min_reward) / rng - 1.0
        return float(np.clip(norm, -1.0, 1.0))


class AugmentationPipeline(nn.Module):
    """Temporally consistent augmentation preserving motion dynamics."""

    def __init__(self, size: int = 224, p_blur: float = 0.2, p_noise: float = 0.1):
        super().__init__()
        self.output_size = (size, size)
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.scale_range = (0.85, 1.0)
        self.ratio_range = (0.95, 1.05)
        self.brightness_delta = 0.1
        self.contrast_range = (0.9, 1.1)
        self.saturation_range = (0.9, 1.1)
        self.hue_delta = 0.02

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            return images

        i, j, h, w = _RandomResizedCrop.get_params(images[0], scale=self.scale_range, ratio=self.ratio_range)

        do_color = random.random() < 0.8  # noqa: S311
        fn_idx = torch.randperm(4) if do_color else []
        b_f = float(torch.empty(1).uniform_(1 - self.brightness_delta, 1 + self.brightness_delta))
        c_f = float(torch.empty(1).uniform_(*self.contrast_range))
        s_f = float(torch.empty(1).uniform_(*self.saturation_range))
        h_f = float(torch.empty(1).uniform_(-self.hue_delta, self.hue_delta))
        do_blur = random.random() < self.p_blur  # noqa: S311
        sigma = float(torch.empty(1).uniform_(0.1, 1.5))
        do_noise = random.random() < self.p_noise  # noqa: S311
        noise_std = float(torch.empty(1).uniform_(0.01, 0.05))

        color_fns = {
            0: lambda x: TF.adjust_brightness(x, b_f),
            1: lambda x: TF.adjust_contrast(x, c_f),
            2: lambda x: TF.adjust_saturation(x, s_f),
            3: lambda x: TF.adjust_hue(x, h_f),
        }

        augmented = []
        for t in range(images.shape[0]):
            img = TF.resized_crop(images[t], i, j, h, w, self.output_size)
            if do_color:
                for fn_id in fn_idx:
                    img = color_fns[fn_id.item()](img)
            if do_blur:
                img = TF.gaussian_blur(img, kernel_size=5, sigma=sigma)
            if do_noise:
                img = torch.clamp(img + torch.randn_like(img) * noise_std, 0.0, 1.0)
            augmented.append(img)

        return torch.stack(augmented)
