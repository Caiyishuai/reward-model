"""Retrieval-based (non-parametric) reward using cosine similarity to success embeddings."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reward_model import RewardModel


class TrajectoryRewarder:
    """KNN reward: embed query, compare to success support set via cosine similarity.

    Score = exp(-(1 - cosine_sim) / temperature), averaged over top-K neighbors.
    """

    def __init__(self, model: RewardModel, device: str = "cuda", temperature: float = 0.1):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.model = model
        self.device = device
        self.temperature = temperature
        self.support_embeddings: torch.Tensor | None = None
        self.num_support_samples = 0

    def build_support_set(self, data_loader: DataLoader) -> None:
        """Encode success trajectories into a support embedding bank."""
        self.model.eval()
        self.model.to(self.device)
        embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in data_loader:
                frames, proprio, _, type_int, _, _ = batch
                success_mask = type_int.view(-1) == 1
                if not success_mask.any():
                    continue

                feats = self.model.extract_features(
                    frames[success_mask].to(self.device),
                    proprio[success_mask].to(self.device),
                )
                embeddings.append(F.normalize(feats, p=2, dim=1).cpu())

        if not embeddings:
            raise RuntimeError("No success trajectories found in loader")

        self.support_embeddings = torch.cat(embeddings, dim=0).to(self.device)
        self.num_support_samples = self.support_embeddings.size(0)

    def get_reward(
        self,
        frames: torch.Tensor,
        proprio: torch.Tensor,
        k: int = 5,
        return_sims: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute retrieval reward in [0, 1].

        Args:
            frames: [B, ...]
            proprio: [B, ...]
            k: number of nearest neighbors
        """
        if self.support_embeddings is None:
            raise RuntimeError("Call build_support_set() first")

        self.model.eval()

        with torch.no_grad():
            query = F.normalize(
                self.model.extract_features(frames.to(self.device), proprio.to(self.device)),
                p=2,
                dim=1,
            )
            sim_matrix = torch.matmul(query, self.support_embeddings.t())
            topk_vals, _ = torch.topk(sim_matrix, k=min(k, sim_matrix.size(1)), dim=1)

            dist = 1.0 - topk_vals
            reward = torch.exp(-dist / self.temperature).mean(dim=1, keepdim=True)

            if return_sims:
                return reward, topk_vals
            return reward
