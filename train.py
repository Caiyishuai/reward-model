"""Training script for the Reward Model.

Supports DDP, AMP, balanced sampling, differential LR, warmup+cosine schedule,
ensemble heads, ranking loss, auxiliary dynamics prediction, EMA, early stopping,
and data prefetching.
"""

import argparse
import datetime
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.common import BASE_DIR, IMG_SIZE_RM, ROBOT_DIM, STATE_WINDOWS_DEFAULT, get_task
from data.dataset import AugmentationPipeline, BalancedLeRobotDataset
from metrics_utils import ranking_accuracy
from reward_model import RewardModel
from training_utils import (
    CheckpointManager,
    DataPrefetcher,
    EarlyStopping,
    EMAModel,
    _is_main,
    _unwrap,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# ==========================================
# Configuration
# ==========================================
@dataclass
class TrainConfig:
    """Training hyperparameters and paths."""

    task_name: str = ""
    base_dir: str = str(BASE_DIR)
    fail_path: str | None = None
    success_path: str | None = None
    save_dir: str | None = None
    camera_keys: list[str] = field(default_factory=lambda: ["observation.images.wrist_1", "observation.images.wrist_2"])

    split_ratio: float = 0.9
    frame_split_strategy: str | None = None
    fail_ratio: float = 1.0
    samples_per_epoch: int = 4000

    epochs: int = 50
    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    backbone_lr: float = 3e-5
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    lambda_rank: float = 0.5
    lambda_aux: float = 0.1
    margin: float = 0.1

    n_epoch_every_eval: int = 5
    save_top_k: int = 3
    monitor_metric: str = "val_rank_acc"

    robot_dim: int = ROBOT_DIM
    state_windows: int = STATE_WINDOWS_DEFAULT
    img_size: int = IMG_SIZE_RM
    ensemble_size: int = 3
    dropout: float = 0.3
    min_reward: float = 0.0
    max_reward: float = 6.0
    unfreeze_last_n_layers: int = 2
    use_film: bool = False
    use_gradient_checkpointing: bool = False
    use_patch_pooling: bool = False
    masked_state_indices: list[int] = field(default_factory=list)

    ema_decay: float = 0.999
    use_ema: bool = True
    early_stop_patience: int = 15

    seed: int = 42
    num_workers: int = 16
    local_rank: int = -1
    use_amp: bool = True
    use_compile: bool = False
    compile_mode: str = "default"
    run_post_train_eval: bool = True


# ==========================================
# Loss & Utils
# ==========================================
def ranking_loss(rewards: Tensor, labels: Tensor, margin: float = 0.1) -> Tensor:
    """Pairwise ranking loss encouraging correct ordering."""
    r_diff = rewards - rewards.t()
    y_diff = labels - labels.t()
    indicator = torch.sign(y_diff)
    valid = (torch.abs(y_diff) > 1e-4).float()
    loss = torch.relu(-indicator * r_diff + margin) * valid
    return loss.sum() / (valid.sum() + 1e-8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================
# Evaluation
# ==========================================
def evaluate_epoch(model: nn.Module, loader: DataLoader, device: torch.device, seed: int = 42) -> dict[str, float]:
    """Comprehensive validation: MSE, ranking accuracy, success/fail separation.

    Reports two ensemble-aggregation metrics side by side so the training
    signal stays aligned with the conservative (min) reward actually used at
    inference by ``get_reward`` / ``get_reward_from_features``:

    * ``val_rank_acc`` / ``val_mse``              — ensemble **mean** (training diagnostic)
    * ``val_rank_acc_min`` / ``val_mse_min``      — ensemble **min**  (matches online SAC reward)
    """
    model.eval()
    raw_model = _unwrap(model)

    all_preds_mean_norm: list[Tensor] = []
    all_preds_min_norm: list[Tensor] = []
    all_labels_norm: list[Tensor] = []
    all_types: list[Tensor] = []

    with torch.no_grad():
        for frames, proprio, labels, types, _, _ in loader:
            frames, proprio, labels = frames.to(device), proprio.to(device), labels.to(device)
            ensemble_preds, _ = model(frames, proprio)
            all_preds_mean_norm.append(ensemble_preds.mean(dim=1).cpu())
            all_preds_min_norm.append(ensemble_preds.min(dim=1).values.cpu())
            all_labels_norm.append(labels.flatten().cpu())
            all_types.append(types.flatten().cpu())

    preds_mean_norm = torch.cat(all_preds_mean_norm)
    preds_min_norm = torch.cat(all_preds_min_norm)
    labels_norm = torch.cat(all_labels_norm)
    types = torch.cat(all_types).numpy()

    norm_device = raw_model.reward_normalizer.min_val.device
    preds_mean_real = raw_model.unnormalize_reward(preds_mean_norm.to(norm_device)).cpu().numpy()
    preds_min_real = raw_model.unnormalize_reward(preds_min_norm.to(norm_device)).cpu().numpy()
    labels_real = raw_model.unnormalize_reward(labels_norm.to(norm_device)).cpu().numpy()

    mse_mean = float(np.mean((preds_mean_real - labels_real) ** 2))
    mse_min = float(np.mean((preds_min_real - labels_real) ** 2))

    rank_acc_mean = ranking_accuracy(preds_mean_real, labels_real, seed=seed)
    rank_acc_min = ranking_accuracy(preds_min_real, labels_real, seed=seed)

    succ_mask = types == 1
    fail_mask = types == 0
    succ_count = int(succ_mask.sum())
    fail_count = int(fail_mask.sum())
    succ_sum = float(preds_mean_real[succ_mask].sum()) if succ_count else 0.0
    fail_sum = float(preds_mean_real[fail_mask].sum()) if fail_count else 0.0
    succ_mean = succ_sum / succ_count if succ_count else 0.0
    fail_mean = fail_sum / fail_count if fail_count else 0.0

    # Trailing _* fields are sufficient statistics for DDP reduction in train()
    # so aggregation does not rely on len(val_loader.dataset) (which can be
    # wrong under DistributedSampler drop_last / padding).
    return {
        "val_mse": mse_mean,
        "val_mse_min": mse_min,
        "val_rank_acc": rank_acc_mean,
        "val_rank_acc_min": rank_acc_min,
        "val_succ_mean": succ_mean,
        "val_fail_mean": fail_mean,
        "_val_n_samples": int(labels_real.shape[0]),
        "_val_succ_count": succ_count,
        "_val_fail_count": fail_count,
        "_val_succ_sum": succ_sum,
        "_val_fail_sum": fail_sum,
    }


# ==========================================
# Training Loop
# ==========================================
def train(cfg: TrainConfig) -> None:
    """Main training function with EMA, early stopping, and prefetching."""
    if cfg.local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(cfg.local_rank)
        device = torch.device(f"cuda:{cfg.local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed + (cfg.local_rank if cfg.local_rank != -1 else 0))

    try:
        _train_inner(cfg, device)
    finally:
        if cfg.local_rank != -1:
            dist.destroy_process_group()


def _train_inner(cfg: TrainConfig, device: torch.device) -> None:
    """Inner training loop, separated for try/finally DDP cleanup."""
    ckpt_manager = CheckpointManager(cfg.save_dir, cfg.task_name, cfg.monitor_metric, cfg.save_top_k)

    # Datasets
    train_ds = BalancedLeRobotDataset(
        fail_path=cfg.fail_path,
        success_path=cfg.success_path,
        camera_keys=cfg.camera_keys,
        split="train",
        split_ratio=cfg.split_ratio,
        target_ratio=cfg.fail_ratio,
        epoch_size=cfg.samples_per_epoch,
        window_size=cfg.state_windows,
        img_size=cfg.img_size,
        transform=AugmentationPipeline(cfg.img_size),
        seed=cfg.seed,
        frame_split_strategy=cfg.frame_split_strategy,
    )

    cfg.min_reward = float(train_ds.reward_stats["min"])
    cfg.max_reward = float(train_ds.reward_stats["max"])
    cfg.robot_dim = train_ds.state_dim
    train_ds.max_reward = cfg.max_reward
    train_ds.min_reward = cfg.min_reward

    if _is_main():
        print(f"[Config] Reward range: [{cfg.min_reward:.4f}, {cfg.max_reward:.4f}]")
        print(f"[Config] Robot dim: {cfg.robot_dim} (inferred from data)")
        print(f"[Config] Action dim: {train_ds.action_dim} (inferred from data)")

    train_sampler = DistributedSampler(train_ds) if cfg.local_rank != -1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    val_ds = BalancedLeRobotDataset(
        fail_path=cfg.fail_path,
        success_path=cfg.success_path,
        camera_keys=cfg.camera_keys,
        split="val",
        split_ratio=cfg.split_ratio,
        target_ratio=cfg.fail_ratio,
        epoch_size=1000,
        window_size=cfg.state_windows,
        img_size=cfg.img_size,
        seed=cfg.seed,
        max_reward=cfg.max_reward,
        min_reward=cfg.min_reward,
        frame_split_strategy=cfg.frame_split_strategy,
    )
    val_sampler = DistributedSampler(val_ds, shuffle=False) if cfg.local_rank != -1 else None
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # Model
    robot_mean = train_ds.mean.repeat(cfg.state_windows)
    robot_std = train_ds.std.repeat(cfg.state_windows)

    model = RewardModel(
        robot_dim=cfg.robot_dim,
        state_windows=cfg.state_windows,
        ensemble_size=cfg.ensemble_size,
        max_reward=cfg.max_reward,
        min_reward=cfg.min_reward,
        action_dim=train_ds.action_dim,
        normalizer_stats={"mean": robot_mean, "std": robot_std},
        unfreeze_last_n_layers=cfg.unfreeze_last_n_layers,
        num_cameras=len(cfg.camera_keys),
        dropout=cfg.dropout,
        use_film=cfg.use_film,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        use_patch_pooling=cfg.use_patch_pooling,
        masked_state_indices=cfg.masked_state_indices,
    ).to(device)

    if cfg.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg.compile_mode)

    if cfg.local_rank != -1:
        model = DDP(model, device_ids=[cfg.local_rank], find_unused_parameters=False)

    # EMA
    ema: EMAModel | None = None
    if cfg.use_ema:
        ema = EMAModel(model, decay=cfg.ema_decay)

    # Optimizer with differential LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (backbone_params if "backbone" in name else head_params).append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.min_lr),
        ],
        milestones=[cfg.warmup_epochs],
    )

    mse_crit = nn.MSELoss()
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.use_amp)

    # Early stopping
    higher_is_better = "rank" in cfg.monitor_metric or "acc" in cfg.monitor_metric
    early_stopper = EarlyStopping(cfg.early_stop_patience, higher_is_better)

    if _is_main():
        print(
            f"[Train] Start. Monitor: {cfg.monitor_metric}. AMP: {cfg.use_amp}. "
            f"EMA: {cfg.use_ema}. Patience: {cfg.early_stop_patience}"
        )
        start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_ds._resample(cfg.seed + epoch)

        model.train()
        accum_mse = accum_rank = accum_aux = 0.0

        prefetcher = DataPrefetcher(train_loader, device)
        step_count = 0

        for batch in prefetcher:
            frames, proprio, labels, _, action, future_state = batch

            with torch.amp.autocast(device.type, enabled=cfg.use_amp):
                preds, pred_next_state = model(frames, proprio, action)

                reward_mse = sum(mse_crit(preds[:, k : k + 1], labels) for k in range(cfg.ensemble_size))
                reward_rank = sum(
                    ranking_loss(preds[:, k : k + 1], labels, cfg.margin) for k in range(cfg.ensemble_size)
                )

                raw_normalizer = _unwrap(model).normalizer
                mean_single = raw_normalizer.mean[: cfg.robot_dim]
                std_single = raw_normalizer.std[: cfg.robot_dim]
                target_norm = (future_state - mean_single) / (std_single + 1e-8)
                aux_loss = mse_crit(pred_next_state.float(), target_norm.float())

                loss = (reward_mse + cfg.lambda_rank * reward_rank + cfg.lambda_aux * aux_loss) / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            step_count += 1
            if step_count % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(model)

            accum_mse += reward_mse.item() / cfg.ensemble_size
            accum_rank += reward_rank.item() / cfg.ensemble_size
            accum_aux += aux_loss.item()

        # Flush remaining gradients for incomplete accumulation
        if step_count % cfg.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        n_batches = max(step_count, 1)
        avg_mse = accum_mse / n_batches
        avg_rank = accum_rank / n_batches
        avg_aux = accum_aux / n_batches

        if cfg.local_rank != -1:
            metrics_t = torch.tensor([avg_mse, avg_rank, avg_aux], device=device)
            dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)
            metrics_t /= dist.get_world_size()
            avg_mse, avg_rank, avg_aux = metrics_t.tolist()

        total_loss = avg_mse + cfg.lambda_rank * avg_rank + cfg.lambda_aux * avg_aux
        scheduler.step()

        if _is_main():
            elapsed = time.time() - start_time
            eta = elapsed / epoch * (cfg.epochs - epoch)
            print(
                f"Epoch {epoch:02d} | Loss={total_loss:.4f} "
                f"(MSE={avg_mse:.4f} Rank={avg_rank:.4f} Aux={avg_aux:.4f}) | "
                f"Time: {datetime.timedelta(seconds=int(elapsed))} "
                f"ETA: {datetime.timedelta(seconds=int(eta))}"
            )

        if epoch % cfg.n_epoch_every_eval == 0 or epoch == cfg.epochs:
            # Use EMA weights for evaluation if available
            backup = None
            if ema is not None:
                backup = ema.apply(model)

            local_metrics = evaluate_epoch(model, val_loader, device, seed=cfg.seed)

            if cfg.local_rank != -1:
                # Aggregate sufficient statistics using per-rank sample counts
                # (not len(val_loader.dataset)) so drop_last / uneven shards in
                # DistributedSampler do not bias the global metric. Layout:
                #   [sum_sq_err_mean, sum_sq_err_min,
                #    sum_rank_acc_mean, sum_rank_acc_min,
                #    n_samples,
                #    succ_sum, succ_count,
                #    fail_sum, fail_count]
                n_local = float(local_metrics["_val_n_samples"])
                vec = torch.tensor(
                    [
                        local_metrics["val_mse"] * n_local,
                        local_metrics["val_mse_min"] * n_local,
                        local_metrics["val_rank_acc"] * n_local,
                        local_metrics["val_rank_acc_min"] * n_local,
                        n_local,
                        float(local_metrics["_val_succ_sum"]),
                        float(local_metrics["_val_succ_count"]),
                        float(local_metrics["_val_fail_sum"]),
                        float(local_metrics["_val_fail_count"]),
                    ],
                    device=device,
                )
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)
                denom_n = max(vec[4].item(), 1.0)
                denom_succ = max(vec[6].item(), 1.0)
                denom_fail = max(vec[8].item(), 1.0)
                global_metrics = {
                    "val_mse": vec[0].item() / denom_n,
                    "val_mse_min": vec[1].item() / denom_n,
                    "val_rank_acc": vec[2].item() / denom_n,
                    "val_rank_acc_min": vec[3].item() / denom_n,
                    "val_succ_mean": vec[5].item() / denom_succ,
                    "val_fail_mean": vec[7].item() / denom_fail,
                }
            else:
                # Strip internal sufficient-statistics keys from single-GPU output.
                global_metrics = {k: v for k, v in local_metrics.items() if not k.startswith("_")}

            if _is_main():
                print(
                    f"  [EVAL] RankAcc(mean): {global_metrics['val_rank_acc'] * 100:.2f}% | "
                    f"RankAcc(min): {global_metrics['val_rank_acc_min'] * 100:.2f}% | "
                    f"MSE(mean): {global_metrics['val_mse']:.4f} | "
                    f"MSE(min): {global_metrics['val_mse_min']:.4f} | "
                    f"Succ: {global_metrics['val_succ_mean']:.3f} | "
                    f"Fail: {global_metrics['val_fail_mean']:.3f}"
                )
                if cfg.monitor_metric not in global_metrics:
                    print(
                        f"  [WARN] monitor_metric={cfg.monitor_metric!r} not found in eval output; "
                        f"available keys: {sorted(global_metrics)}"
                    )
                ckpt_manager.update(model, global_metrics.get(cfg.monitor_metric, 0.0), epoch)

            if ema is not None and backup is not None:
                ema.restore(model, backup)

            monitor_val = global_metrics.get(cfg.monitor_metric, 0.0)
            if early_stopper.step(monitor_val):
                if _is_main():
                    print(f"[EarlyStop] No improvement for {cfg.early_stop_patience} evals. Stopping.")
                break

    if _is_main() and cfg.run_post_train_eval:
        best_ckpt = ckpt_manager.best_checkpoint()
        if best_ckpt:
            print(f"\n[PostTrain] Running evaluation on best checkpoint: {best_ckpt}")
            try:
                from eval_suite import run_evaluation

                if cfg.save_dir:
                    eval_prefix = cfg.task_name if cfg.task_name else os.path.basename(cfg.save_dir).split("_")[0]
                    eval_output = os.path.join(cfg.save_dir, "eval_report")
                else:
                    eval_prefix = cfg.task_name or "auto"
                    eval_output = os.path.join("checkpoints", f"eval_report_{eval_prefix}")

                run_evaluation(
                    checkpoint_path=best_ckpt,
                    task_name=cfg.task_name,
                    prefix=eval_prefix,
                    output_dir=eval_output,
                    device=str(device),
                    num_traj_episodes=3,
                    batch_size=cfg.batch_size,
                )
            except (ImportError, FileNotFoundError) as e:
                print(f"[PostTrain] Evaluation skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument("--task_name", type=str, default="button")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefix", type=str, default="auto")
    parser.add_argument("--use_film", action="store_true")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_patch_pooling", action="store_true")
    args = parser.parse_args()

    task = get_task(args.task_name)
    data_dir = BASE_DIR / args.task_name / f"{args.prefix}_processed"
    success_path = str(data_dir / "success_lerobot.pkl")
    fail_path = str(data_dir / "fail_lerobot.pkl")
    save_dir = os.path.join("checkpoints", f"{args.prefix}_{args.task_name}")

    camera_keys = [f"observation.images.{k}" for k in task.camera_keys]

    cfg = TrainConfig(
        task_name=args.task_name,
        epochs=args.epochs,
        num_workers=args.num_workers,
        success_path=success_path,
        fail_path=fail_path,
        save_dir=save_dir,
        camera_keys=camera_keys,
        use_film=args.use_film,
        use_ema=args.use_ema and not args.no_ema,
        early_stop_patience=args.patience,
        grad_accum_steps=args.grad_accum,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_patch_pooling=args.use_patch_pooling,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )
    train(cfg)
