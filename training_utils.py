"""Training utilities: EMA, early stopping, data prefetching, checkpoint management."""

import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def _is_main() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap DDP wrapper to access the raw model."""
    return model.module if isinstance(model, DDP) else model


class EMAModel:
    """Exponential Moving Average of model parameters for stable evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        self._init_shadow(model)

    def _init_shadow(self, model: nn.Module) -> None:
        raw = _unwrap(model)
        for name, param in raw.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        raw = _unwrap(model)
        for name, param in raw.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> dict[str, Tensor]:
        """Apply EMA weights, return backup of originals."""
        raw = _unwrap(model)
        backup = {}
        for name, param in raw.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict[str, Tensor]) -> None:
        """Restore original weights from backup."""
        raw = _unwrap(model)
        for name, param in raw.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


class EarlyStopping:
    """Early stopping with patience tracking."""

    def __init__(self, patience: int, higher_is_better: bool = True):
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.best_score: float | None = None
        self.counter = 0

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score) if self.higher_is_better else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class DataPrefetcher:
    """CUDA prefetching iterator for DataLoader to overlap data transfer.

    Falls back to synchronous transfer on non-CUDA devices.
    """

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self._use_cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self._use_cuda else None
        self._iter = iter(loader)
        self._next_batch: tuple | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        if self._use_cuda:
            with torch.cuda.stream(self.stream):
                self._next_batch = tuple(
                    t.to(self.device, non_blocking=True) if isinstance(t, Tensor) else t for t in batch
                )
        else:
            self._next_batch = tuple(t.to(self.device) if isinstance(t, Tensor) else t for t in batch)

    def __iter__(self) -> "DataPrefetcher":
        return self

    def __next__(self) -> tuple:
        if self._use_cuda:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        if self._next_batch is None:
            raise StopIteration
        batch = self._next_batch
        if self._use_cuda:
            cur = torch.cuda.current_stream(self.device)
            for t in batch:
                if isinstance(t, Tensor) and t.is_cuda:
                    t.record_stream(cur)
        self._preload()
        return batch

    def __len__(self) -> int:
        return len(self.loader)


class CheckpointManager:
    """Maintains top-K checkpoints by monitored metric."""

    # Explicit direction table avoids fragile substring matching
    # (e.g. `val_rank_loss` contains "rank" but is lower-is-better).
    _DIRECTION_TABLE: dict[str, bool] = {
        "val_rank_acc": True,
        "val_rank_acc_min": True,
        "val_accuracy": True,
        "val_pra": True,
        "val_gap": True,
        "val_monotonicity": True,
        "val_spearman": True,
        "val_loss": False,
        "val_mse": False,
        "val_mse_min": False,
        "val_mae": False,
        "val_rank_loss": False,
        "val_dynamics_loss": False,
    }

    def __init__(
        self,
        save_dir: str | None,
        task: str,
        monitor_metric: str = "val_rank_acc",
        top_k: int = 3,
        higher_is_better: bool | None = None,
    ):
        if save_dir is None:
            save_dir = os.path.join("checkpoints", task)
        self.save_dir = save_dir
        self.monitor_metric = monitor_metric
        self.top_k = top_k
        self.task = task
        if higher_is_better is not None:
            self.is_higher_better = higher_is_better
        elif monitor_metric in self._DIRECTION_TABLE:
            self.is_higher_better = self._DIRECTION_TABLE[monitor_metric]
        else:
            lowered = monitor_metric.lower()
            if any(tok in lowered for tok in ("loss", "mse", "mae", "err")):
                self.is_higher_better = False
            elif any(tok in lowered for tok in ("acc", "pra", "auc", "gap", "spearman", "monoton")):
                self.is_higher_better = True
            else:
                raise ValueError(
                    f"Cannot infer direction for monitor_metric='{monitor_metric}'. "
                    "Pass higher_is_better=... explicitly or add to CheckpointManager._DIRECTION_TABLE."
                )
        self.ckpts: list[tuple[float, str]] = []
        os.makedirs(save_dir, exist_ok=True)

    def update(self, model: nn.Module, metric_value: float, epoch: int) -> None:
        if not _is_main():
            return

        filename = f"rm_{self.task}_epoch_{epoch}_val_{metric_value:.4f}.pt"
        path = os.path.join(self.save_dir, filename)

        raw_model = _unwrap(model)
        raw_model.save(path)

        self.ckpts.append((metric_value, path))
        self.ckpts.sort(key=lambda x: x[0], reverse=self.is_higher_better)
        if self.ckpts[0][1] == path:
            # Stable path for evaluation, simulation configs, and batch jobs.
            # RewardModel.save also writes the matching best.json config.
            raw_model.save(os.path.join(self.save_dir, "best.pt"))

        if len(self.ckpts) > self.top_k:
            worst = self.ckpts.pop()
            if os.path.exists(worst[1]):
                os.remove(worst[1])
            json_path = worst[1].rsplit(".", 1)[0] + ".json"
            if os.path.exists(json_path):
                os.remove(json_path)

    def best_checkpoint(self) -> str | None:
        """Return path to the best checkpoint, or None if empty."""
        return self.ckpts[0][1] if self.ckpts else None
