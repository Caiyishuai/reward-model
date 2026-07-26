"""Dataset-level evaluation: MSE, ranking accuracy, success/fail distribution."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.common import IMG_SIZE_RM, STATE_WINDOWS_DEFAULT, load_pickle
from metrics_utils import ranking_accuracy
from reward_model import RewardModel


class EvaluationDataset(Dataset):
    """Episode-split evaluation dataset with configurable camera keys."""

    def __init__(
        self,
        data_paths: dict[str, str],
        camera_keys: list[str],
        split_ratio: float = 0.9,
        split: str = "val",
        window_size: int = STATE_WINDOWS_DEFAULT,
        img_size: int = IMG_SIZE_RM,
        seed: int = 42,
    ):
        self.window_size = window_size
        self.img_size = img_size
        self.camera_keys = camera_keys
        self.samples: list[dict] = []

        for label_type, path in data_paths.items():
            if not os.path.exists(path):
                continue

            data = load_pickle(path)
            episode_indices = data["episode_index"]
            unique_eps = np.unique(episode_indices)

            rng = np.random.default_rng(seed)
            rng.shuffle(unique_eps)

            split_idx = int(len(unique_eps) * split_ratio)
            target_eps = set(unique_eps[:split_idx] if split == "train" else unique_eps[split_idx:])

            count = 0
            for i in range(window_size - 1, len(episode_indices)):
                if episode_indices[i] in target_eps:
                    start = i - window_size + 1
                    if episode_indices[start] == episode_indices[i]:
                        indices = np.arange(start, i + 1)

                        all_cam_imgs = []
                        for cam_key in camera_keys:
                            if cam_key in data:
                                camera_data = data[cam_key]
                                all_cam_imgs.append(
                                    camera_data[indices]
                                    if isinstance(camera_data, np.ndarray)
                                    else np.stack([camera_data[int(index)] for index in indices])
                                )

                        if not all_cam_imgs:
                            continue

                        self.samples.append(
                            {
                                "images": all_cam_imgs,
                                "state": data["observation.state"][indices],
                                "reward": data["next.reward"][i],
                                "type": label_type,
                            }
                        )
                        count += 1
            print(f"[Eval] {label_type} ({split}): {count} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]

        all_cam_flat: list[torch.Tensor] = []
        for imgs_np in sample["images"]:
            imgs = torch.from_numpy(imgs_np).float().div_(255.0).permute(0, 3, 1, 2)
            if imgs.shape[-2] != self.img_size or imgs.shape[-1] != self.img_size:
                imgs = torch.nn.functional.interpolate(
                    imgs, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
                )
            all_cam_flat.append(imgs.reshape(-1, self.img_size, self.img_size))

        frames = torch.cat(all_cam_flat, dim=0)
        proprio = torch.from_numpy(sample["state"]).float().flatten()
        label = torch.tensor([sample["reward"]], dtype=torch.float32)

        return frames, proprio, label, sample["type"]


class RewardModelEvaluator:
    """Evaluator computing MSE, ranking accuracy, and distribution plots."""

    def __init__(self, model: RewardModel, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def evaluate(self, dataset: Dataset, batch_size: int = 16, save_dir: str = "eval_results") -> dict:
        os.makedirs(save_dir, exist_ok=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        results: dict[str, list] = {"preds": [], "labels": [], "types": [], "uncertainties": []}

        with torch.no_grad():
            for frames, proprio, labels, types in tqdm(loader, desc="Evaluating"):
                frames, proprio = frames.to(self.device), proprio.to(self.device)
                ensemble_preds, _ = self.model(frames, proprio)

                mean_pred = ensemble_preds.mean(dim=1, keepdim=True)
                pred_raw = self.model.unnormalize_reward(mean_pred).squeeze(1)
                results["preds"].extend(pred_raw.cpu().numpy())
                results["uncertainties"].extend(ensemble_preds.std(dim=1).cpu().numpy())
                results["labels"].extend(labels.flatten().numpy())
                results["types"].extend(types)

        self._analyze(results, save_dir)
        return results

    def _analyze(self, results: dict, save_dir: str) -> None:
        preds = np.array(results["preds"])
        labels = np.array(results["labels"])
        types = np.array(results["types"])

        mse = float(np.mean((preds - labels) ** 2))
        print(f"[Eval] MSE: {mse:.4f}")

        if len(preds) > 1:
            rank_acc = ranking_accuracy(preds, labels)
            print(f"[Eval] Ranking Accuracy: {rank_acc * 100:.2f}%")

        succ_preds = preds[types == "success"]
        fail_preds = preds[types == "fail"]

        if len(succ_preds) > 0 and len(fail_preds) > 0:
            print(f"[Eval] Success: {np.mean(succ_preds):.4f} +/- {np.std(succ_preds):.4f}")
            print(f"[Eval] Fail:    {np.mean(fail_preds):.4f} +/- {np.std(fail_preds):.4f}")

            plt.figure(figsize=(10, 6))
            sns.histplot(succ_preds, color="green", label="Success", kde=True, alpha=0.5)
            sns.histplot(fail_preds, color="red", label="Fail", kde=True, alpha=0.5)
            plt.title(f"Reward Distribution | MSE: {mse:.2f}")
            plt.legend()
            plt.savefig(os.path.join(save_dir, "test_dist.png"))
            plt.close()
