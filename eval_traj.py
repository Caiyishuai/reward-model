"""Trajectory-level evaluation with per-episode reward curve visualization."""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from data.common import IMG_SIZE_RM, STATE_WINDOWS_DEFAULT, load_pickle
from reward_model import RewardModel


class TrajectoryEvalDataset(Dataset):
    """Groups data by episode for trajectory-level plotting."""

    def __init__(
        self,
        data_path: str,
        window_size: int = STATE_WINDOWS_DEFAULT,
        img_size: int = IMG_SIZE_RM,
        camera_keys: list[str] | None = None,
    ):
        self.window_size = window_size
        self.img_size = img_size

        self.data = load_pickle(data_path)
        self.episode_indices = self.data["episode_index"]
        self.unique_episodes = np.unique(self.episode_indices)
        self.states = self.data["observation.state"]
        self.rewards = self.data["next.reward"]

        if camera_keys is None:
            candidates = ["observation.images.wrist_1", "observation.images.wrist_2", "observation.images.side_policy"]
            self.camera_keys = [k for k in candidates if k in self.data and len(self.data[k]) > 0]
        else:
            self.camera_keys = camera_keys

        if not self.camera_keys:
            self.camera_keys = [k for k in self.data if "observation.images" in k and len(self.data[k]) > 0]

        if not self.camera_keys:
            raise ValueError(f"No image data found. Keys: {list(self.data.keys())}")

    def get_episode_indices(self, episode_idx: int) -> list[int]:
        """Valid frame indices for a given episode."""
        all_idx = np.where(self.episode_indices == episode_idx)[0]
        if len(all_idx) == 0:
            return []
        return [i for i in all_idx if i - self.window_size + 1 >= all_idx[0]]

    def get_batch(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Build batched tensors for a list of frame indices."""
        batch_frames, batch_proprio, batch_labels = [], [], []

        for end_idx in indices:
            idx_range = np.arange(end_idx - self.window_size + 1, end_idx + 1)

            all_cam_flat: list[torch.Tensor] = []
            for cam_key in self.camera_keys:
                raw = self.data[cam_key]
                imgs_np = np.stack([raw[i] for i in idx_range]) if isinstance(raw, list) else raw[idx_range]
                imgs = torch.from_numpy(imgs_np).float().div_(255.0).permute(0, 3, 1, 2)
                if imgs.shape[-2] != self.img_size or imgs.shape[-1] != self.img_size:
                    imgs = torch.nn.functional.interpolate(
                        imgs, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
                    )
                all_cam_flat.append(imgs.reshape(-1, self.img_size, self.img_size))

            batch_frames.append(torch.cat(all_cam_flat, dim=0))
            batch_proprio.append(torch.from_numpy(self.states[idx_range]).float().flatten())
            batch_labels.append(self.rewards[end_idx])

        return torch.stack(batch_frames), torch.stack(batch_proprio), np.array(batch_labels)


class TrajectoryEvaluator:
    """Plots predicted vs ground-truth reward curves per episode."""

    def __init__(self, model: RewardModel, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def evaluate_trajectories(
        self,
        dataset: TrajectoryEvalDataset,
        num_episodes: int = 3,
        save_dir: str = "eval_results",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        rng = random.Random(42)  # noqa: S311 — non-crypto sampling of eval episodes
        selected = rng.sample(list(dataset.unique_episodes), min(num_episodes, len(dataset.unique_episodes)))

        for ep_idx in selected:
            indices = dataset.get_episode_indices(ep_idx)
            if not indices:
                continue

            preds, gts, uncertainties = [], [], []
            batch_size = 32

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]
                frames, proprio, labels = dataset.get_batch(batch_idx)
                frames, proprio = frames.to(self.device), proprio.to(self.device)

                with torch.no_grad():
                    raw_preds, _ = self.model(frames, proprio)
                    real_preds = self.model.unnormalize_reward(raw_preds)
                    preds.extend(real_preds.mean(dim=1).cpu().numpy())
                    uncertainties.extend(real_preds.std(dim=1).cpu().numpy())
                    gts.extend(labels)

            preds_arr = np.array(preds)
            uncs_arr = np.array(uncertainties)
            steps = np.arange(len(preds_arr))

            plt.figure(figsize=(12, 5))
            plt.plot(steps, gts, label="Ground Truth", color="black", linewidth=2, linestyle="--")
            plt.plot(steps, preds_arr, label="Prediction", color="blue", linewidth=2)
            plt.fill_between(
                steps, preds_arr - uncs_arr, preds_arr + uncs_arr, color="blue", alpha=0.2, label="Uncertainty"
            )
            plt.title(f"Episode {ep_idx}")
            plt.xlabel("Time Step")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"traj_ep_{ep_idx}.png"))
            plt.close()
            print(f"[Eval] Saved: traj_ep_{ep_idx}.png")
