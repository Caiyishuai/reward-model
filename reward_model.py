"""Unified Reward Model with DINOv2 backbone for robot manipulation tasks.

Supports optional FiLM conditioning, temporal modeling, backbone fine-tuning,
ensemble prediction with auxiliary dynamics head, gradient checkpointing,
batched multi-camera backbone forward, and multi-scale feature fusion.
"""

import json
import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from data.common import (
    MAX_SEQ_LEN_MIN,
    MAX_SEQ_LEN_SLACK,
    ROBOT_DIM,
    STATE_WINDOWS_DEFAULT,
)

logger = logging.getLogger(__name__)

DEFAULT_BACKBONE = "facebook/dinov2-small"


def build_proprio_input_mask(
    robot_dim: int,
    state_windows: int,
    masked_state_indices: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Build a repeated binary mask for normalized proprioceptive inputs."""
    indices = sorted(set(masked_state_indices or []))
    if any(index < 0 or index >= robot_dim for index in indices):
        raise ValueError(f"masked_state_indices must be within [0, {robot_dim}); got {indices}")
    single_step = torch.ones(robot_dim)
    if indices:
        single_step[indices] = 0.0
    return single_step.repeat(state_windows)


class ProprioNormalizer(nn.Module):
    """Buffer-based normalizer for robot proprioceptive states.

    Saved/loaded automatically with the model state_dict.
    """

    def __init__(self, dim: int, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        super().__init__()
        self.dim = dim
        self.register_buffer("mean", torch.zeros(dim) if mean is None else mean)
        self.register_buffer("std", torch.ones(dim) if std is None else std)

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean.to(self.mean.device))
        self.std.copy_(std.to(self.std.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)


class MinMaxNormalizer(nn.Module):
    """Buffer-based normalizer mapping [min_val, max_val] <-> [-1, 1]."""

    def __init__(self, min_val: float = 0.0, max_val: float = 6.0):
        super().__init__()
        self.register_buffer("min_val", torch.tensor(min_val))
        self.register_buffer("max_val", torch.tensor(max_val))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """[min, max] -> [-1, 1].

        Invertible via unnormalize() when x is in [min_val, max_val].
        Out-of-range inputs are clamped; roundtrip is lossy for those.
        """
        range_val = self.max_val - self.min_val
        norm = 2.0 * (x - self.min_val) / (range_val + 1e-8) - 1.0
        return torch.clamp(norm, -1.0, 1.0)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> [min, max].

        Inverse of normalize(). Clamped inputs are not exactly invertible.
        """
        x = torch.clamp(x, -1.0, 1.0)
        range_val = self.max_val - self.min_val
        return (x + 1.0) / 2.0 * range_val + self.min_val


class TemporalAdapter(nn.Module):
    """Transformer-based temporal aggregation: [B, T, D] -> [B, D]."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 10,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, input_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        max_len = self.pos_embed.size(1)
        if max_len < T:
            raise ValueError(
                f"TemporalAdapter received sequence length T={T}, but pos_embed only "
                f"supports up to max_seq_len={max_len}. Increase max_seq_len when "
                f"constructing TemporalAdapter (must be >= state_windows)."
            )
        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.out_proj(x)


class FiLMGenerator(nn.Module):
    """Feature-wise Linear Modulation: proprioception conditions vision features.

    Zero-initialized output for identity modulation at init.
    """

    def __init__(self, cond_dim: int, target_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, target_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(cond)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class PatchAttentionPool(nn.Module):
    """Learnable attention pooling over patch tokens.

    Uses cross-attention with a learned query to aggregate spatial
    information from all patch tokens into a single vector.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Pool patch tokens via cross-attention.

        Args:
            patches: [B, N_patches, D]

        Returns:
            [B, D]
        """
        q = self.query.expand(patches.size(0), -1, -1)
        out, _ = self.attn(q, patches, patches)
        return self.norm(out.squeeze(1))


class RewardModel(nn.Module):
    """DINOv2-based reward model with ensemble heads.

    Architecture: Multi-camera DINOv2 -> TemporalAdapter -> Fusion -> Ensemble Heads.
    Optional FiLM conditioning of vision features by proprioception.
    Optional auxiliary dynamics head for forward prediction.
    Optional multi-scale feature fusion via PatchAttentionPool.

    Performance features:
        - Batched backbone forward: all cameras x time steps in one pass
        - Gradient checkpointing for backbone to reduce VRAM
        - Frozen backbone runs in no_grad by default

    Range strategy:
        - Network output: Tanh -> [-1, 1]
        - Training target: Normalized to [-1, 1] via MinMaxNormalizer
        - Inference output: Unnormalized back to [min_reward, max_reward]
    """

    def __init__(
        self,
        robot_dim: int = ROBOT_DIM,
        state_windows: int = STATE_WINDOWS_DEFAULT,
        ensemble_size: int = 3,
        dropout: float = 0.1,
        unfreeze_last_n_layers: int = 0,
        backbone_type: str = DEFAULT_BACKBONE,
        max_reward: float = 6.0,
        min_reward: float = 0.0,
        action_dim: int = 7,
        normalizer_stats: dict[str, torch.Tensor] | None = None,
        num_cameras: int = 1,
        use_film: bool = False,
        use_gradient_checkpointing: bool = False,
        use_patch_pooling: bool = False,
        masked_state_indices: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()

        self.state_windows = state_windows
        self.ensemble_size = ensemble_size
        self.backbone_type = backbone_type
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.robot_dim = robot_dim
        self.action_dim = action_dim
        self.num_cameras = num_cameras
        self.use_film = use_film
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_patch_pooling = use_patch_pooling
        self.masked_state_indices = sorted(set(masked_state_indices or []))

        # Normalizers
        self.proprio_dim_flat = robot_dim * state_windows
        self.normalizer = ProprioNormalizer(self.proprio_dim_flat)
        self.register_buffer(
            "proprio_input_mask",
            build_proprio_input_mask(robot_dim, state_windows, self.masked_state_indices),
        )
        self.reward_normalizer = MinMaxNormalizer(min_val=min_reward, max_val=max_reward)

        if normalizer_stats is not None:
            self.normalizer.set_stats(normalizer_stats["mean"], normalizer_stats["std"])

        # Vision encoder
        self.backbone_config = AutoConfig.from_pretrained(backbone_type)
        self.backbone = AutoModel.from_pretrained(backbone_type)
        self.vision_feat_dim = self.backbone_config.hidden_size

        self.dropout = dropout
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self._configure_backbone_freezing(unfreeze_last_n_layers)
        self._backbone_frozen = unfreeze_last_n_layers <= 0

        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # Multi-scale patch pooling (optional)
        self.patch_pool: PatchAttentionPool | None = None
        if use_patch_pooling:
            self.patch_pool = PatchAttentionPool(self.vision_feat_dim, num_heads=4)

        # Proprioception encoder
        proprio_feat_dim = 128
        self.proprio_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim_flat, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, proprio_feat_dim),
            nn.LayerNorm(proprio_feat_dim),
            nn.ReLU(),
        )

        # Temporal adapter (shared across cameras). max_seq_len must cover
        # state_windows; +2 slack so callers can experiment with longer windows
        # without silently truncating positional embeddings.
        self.temporal_adapter = TemporalAdapter(
            input_dim=self.vision_feat_dim,
            num_layers=2,
            nhead=8,
            dropout=dropout,
            max_seq_len=max(MAX_SEQ_LEN_MIN, state_windows + MAX_SEQ_LEN_SLACK),
        )

        # FiLM (optional)
        vision_out_dim = self.vision_feat_dim * self.num_cameras
        self.film_generator: FiLMGenerator | None = None
        if use_film:
            self.film_generator = FiLMGenerator(cond_dim=proprio_feat_dim, target_dim=vision_out_dim)

        # Fusion
        fusion_dim = vision_out_dim + proprio_feat_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Ensemble reward heads: Tanh -> [-1, 1]
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1),
                    nn.Tanh(),
                )
                for _ in range(ensemble_size)
            ]
        )

        # Auxiliary dynamics head
        self.dynamics_head = nn.Sequential(
            nn.Linear(512 + action_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, robot_dim),
        )

        self._init_heads()

    def train(self, mode: bool = True) -> "RewardModel":
        """Override to keep frozen backbone in eval mode."""
        super().train(mode)
        if self._backbone_frozen and mode:
            self.backbone.eval()
        return self

    def _configure_backbone_freezing(self, unfreeze_last_n: int) -> None:
        """Freeze backbone, optionally unfreeze last N encoder layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n <= 0:
            self.backbone.eval()
            return

        layers = None
        for attr in ("encoder.layer", "layer", "layers"):
            obj = self.backbone
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                layers = obj
                break

        if layers is not None:
            total = len(layers)
            start = max(0, total - unfreeze_last_n)
            for i in range(start, total):
                for param in layers[i].parameters():
                    param.requires_grad = True

        for attr in ("layernorm", "norm"):
            norm_layer = getattr(self.backbone, attr, None)
            if norm_layer is not None:
                for param in norm_layer.parameters():
                    param.requires_grad = True
                break

        self.backbone.train()

    def _init_heads(self) -> None:
        """Xavier initialization for prediction heads."""
        for module in [self.heads, self.dynamics_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode multi-camera temporal images to a single feature vector.

        Uses batched backbone forward: all cameras × time steps in one pass
        instead of sequential per-camera loops.

        Args:
            images: [B, N_CAM * T * 3, H, W]

        Returns:
            [B, vision_feat_dim * N_CAM]
        """
        B = images.size(0)
        T = self.state_windows
        N = self.num_cameras

        camera_chunks = torch.chunk(images, chunks=N, dim=1)

        all_frames = []
        for cam_imgs in camera_chunks:
            frames = torch.chunk(cam_imgs, chunks=T, dim=1)
            all_frames.extend(frames)

        stacked = torch.cat(all_frames, dim=0)  # [B * N * T, 3, H, W]
        if stacked.shape[0] != B * N * T:
            raise ValueError(f"Expected batch {B * N * T}, got {stacked.shape[0]}")

        if self._backbone_frozen:
            with torch.no_grad():
                outputs = self.backbone(stacked)
        else:
            outputs = self.backbone(stacked)

        hidden = outputs.last_hidden_state  # [B*N*T, 1+N_patches, D]
        cls_tokens = hidden[:, 0, :]  # [B*N*T, D]

        if self.patch_pool is not None:
            patch_tokens = hidden[:, 1:, :]  # [B*N*T, N_patches, D]
            spatial_feat = self.patch_pool(patch_tokens)  # [B*N*T, D]
            cls_tokens = cls_tokens + spatial_feat

        cls_tokens = cls_tokens.view(N, T, B, self.vision_feat_dim)

        cam_features = []
        for cam_idx in range(N):
            per_cam = cls_tokens[cam_idx].permute(1, 0, 2)  # [B, T, D]
            cam_features.append(self.temporal_adapter(per_cam))  # [B, D]

        return torch.cat(cam_features, dim=1)

    def extract_features(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Extract fused features (Vision + Proprio) without reward heads.

        Returns: [B, 512]
        """
        proprio = self.normalizer(proprio) * self.proprio_input_mask
        vision_feat = self._encode_vision(images)
        proprio_feat = self.proprio_encoder(proprio)

        if self.film_generator is not None:
            gamma, beta = self.film_generator(proprio_feat)
            vision_feat = vision_feat * (1 + gamma) + beta

        fused = torch.cat([vision_feat, proprio_feat], dim=1)
        return self.fusion_proj(fused)

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            images: [B, num_cameras * T * 3, H, W]
            proprio: [B, robot_dim * T]
            action: [B, action_dim] optional, for auxiliary dynamics prediction

        Returns:
            rewards: [B, ensemble_size] in [-1, 1]
            pred_next_state: [B, robot_dim] or None
        """
        fused = self.extract_features(images, proprio)

        rewards = torch.cat([head(fused) for head in self.heads], dim=1)

        pred_next_state = None
        if action is not None:
            dyn_input = torch.cat([fused, action], dim=1)
            pred_next_state = self.dynamics_head(dyn_input)

        return rewards, pred_next_state

    def normalize_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        """Map [min_reward, max_reward] -> [-1, 1]."""
        return self.reward_normalizer.normalize(raw_reward)

    def unnormalize_reward(self, norm_reward: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] -> [min_reward, max_reward]."""
        return self.reward_normalizer.unnormalize(norm_reward)

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        """Encode raw images through backbone only. For feature caching.

        Args:
            images: [B, 3, H, W] — single frames (any camera, any timestep)

        Returns:
            [B, vision_feat_dim] — CLS tokens (+ optional patch pool)
        """
        if self._backbone_frozen:
            with torch.no_grad():
                outputs = self.backbone(images)
        else:
            outputs = self.backbone(images)
        hidden = outputs.last_hidden_state
        cls_tokens = hidden[:, 0, :]
        if self.patch_pool is not None:
            spatial_feat = self.patch_pool(hidden[:, 1:, :])
            cls_tokens = cls_tokens + spatial_feat
        return cls_tokens

    def get_reward_from_features(
        self,
        cam_features: torch.Tensor,
        proprio: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute reward from pre-cached backbone features (skip DINOv2).

        Args:
            cam_features: [B, N_CAM, T, vision_feat_dim] — cached CLS tokens
            proprio: [B, robot_dim * T]
            return_uncertainty: if True, also return ensemble std as [B, 1]

        Returns:
            reward: [B, 1] — conservative (min ensemble) reward in original scale
            uncertainty: [B, 1] — ensemble std (only if return_uncertainty=True)
        """
        B, N, T, D = cam_features.shape
        cam_results = []
        for cam_idx in range(N):
            cam_results.append(self.temporal_adapter(cam_features[:, cam_idx]))
        vision_feat = torch.cat(cam_results, dim=1)

        proprio_norm = self.normalizer(proprio) * self.proprio_input_mask
        proprio_feat = self.proprio_encoder(proprio_norm)

        if self.film_generator is not None:
            gamma, beta = self.film_generator(proprio_feat)
            vision_feat = vision_feat * (1 + gamma) + beta

        fused = self.fusion_proj(torch.cat([vision_feat, proprio_feat], dim=1))
        rewards = torch.cat([head(fused) for head in self.heads], dim=1)
        min_reward_norm, _ = torch.min(rewards, dim=1, keepdim=True)
        reward = self.unnormalize_reward(min_reward_norm)

        if return_uncertainty:
            rewards_real = self.unnormalize_reward(rewards)
            uncertainty = rewards_real.std(dim=1, keepdim=True)
            return reward, uncertainty
        return reward

    def get_reward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Inference: conservative (min ensemble) reward in original scale.

        Temporarily sets eval mode if model is training, restores afterwards.
        """
        was_training = self.training
        self.eval()
        try:
            device_type = next(self.parameters()).device.type
            with torch.no_grad(), torch.amp.autocast(device_type):
                rewards_norm, _ = self.forward(images, proprio)
                min_reward_norm, _ = torch.min(rewards_norm, dim=1, keepdim=True)
                return self.unnormalize_reward(min_reward_norm)
        finally:
            if was_training:
                self.train()

    def save(self, path: str) -> None:
        """Save model: state_dict as .pt, config as sibling .json."""
        config = {
            "robot_dim": self.robot_dim,
            "state_windows": self.state_windows,
            "ensemble_size": self.ensemble_size,
            "backbone_type": self.backbone_type,
            "max_reward": self.max_reward,
            "min_reward": self.min_reward,
            "num_cameras": self.num_cameras,
            "use_film": self.use_film,
            "use_patch_pooling": self.use_patch_pooling,
            "masked_state_indices": self.masked_state_indices,
            "action_dim": self.action_dim,
            "dropout": self.dropout,
            "unfreeze_last_n_layers": self.unfreeze_last_n_layers,
            "robot_mean": self.normalizer.mean.cpu().tolist(),
            "robot_std": self.normalizer.std.cpu().tolist(),
            "reward_min_val": self.reward_normalizer.min_val.item(),
            "reward_max_val": self.reward_normalizer.max_val.item(),
        }
        torch.save(self.state_dict(), path)
        config_path = path.rsplit(".", 1)[0] + ".json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RewardModel":
        """Load model from state_dict .pt + sibling .json config."""
        config_path = path.rsplit(".", 1)[0] + ".json"
        with open(config_path) as f:
            config = json.load(f)

        stats = {
            "mean": torch.tensor(config["robot_mean"]),
            "std": torch.tensor(config["robot_std"]),
        }

        model = cls(
            robot_dim=config["robot_dim"],
            state_windows=config["state_windows"],
            ensemble_size=config["ensemble_size"],
            backbone_type=config.get("backbone_type", DEFAULT_BACKBONE),
            max_reward=config.get("max_reward", 6.0),
            min_reward=config.get("min_reward", 0.0),
            action_dim=config.get("action_dim", 7),
            normalizer_stats=stats,
            num_cameras=config.get("num_cameras", 1),
            use_film=config.get("use_film", False),
            use_patch_pooling=config.get("use_patch_pooling", False),
            masked_state_indices=config.get("masked_state_indices", []),
            dropout=config.get("dropout", 0.1),
            unfreeze_last_n_layers=config.get("unfreeze_last_n_layers", 0),
        )

        state_dict = torch.load(path, map_location=device, weights_only=True)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            logger.warning("Missing keys: %s", result.missing_keys)
            logger.warning("Unexpected keys: %s", result.unexpected_keys)
        critical_prefixes = ("heads", "fusion_proj", "normalizer", "reward_normalizer", "proprio_encoder", "temporal_adapter")
        critical_missing = [k for k in result.missing_keys if k.startswith(critical_prefixes)]
        if critical_missing:
            raise RuntimeError(f"Critical weight keys missing: {critical_missing}")

        model.to(device)
        model.eval()
        return model


if __name__ == "__main__":
    print("--- RewardModel Sanity Checks ---")

    configs = [
        {"use_film": False, "use_patch_pooling": False, "label": "Standard"},
        {"use_film": True, "use_patch_pooling": False, "label": "FiLM"},
        {"use_film": False, "use_patch_pooling": True, "label": "PatchPool"},
        {"use_film": True, "use_patch_pooling": True, "label": "FiLM+PatchPool"},
    ]

    for cfg in configs:
        print(f"\n[{cfg['label']}] Testing with 2 cameras...")

        model = RewardModel(
            robot_dim=ROBOT_DIM,
            state_windows=STATE_WINDOWS_DEFAULT,
            max_reward=6.0,
            num_cameras=2,
            use_film=cfg["use_film"],
            use_patch_pooling=cfg["use_patch_pooling"],
        )

        raw_vals = torch.tensor([0.0, 3.0, 6.0])
        norm_vals = model.normalize_reward(raw_vals)
        recon_vals = model.unnormalize_reward(norm_vals)
        assert torch.allclose(raw_vals, recon_vals), "Normalization roundtrip failed!"
        print("  PASS: Normalization roundtrip")

        B = 2
        dummy_img = torch.randn(B, 2 * 3 * 3, 224, 224)
        dummy_prop = torch.randn(B, 19 * 3)
        dummy_action = torch.randn(B, 7)

        rewards, pred_state = model(dummy_img, dummy_prop, dummy_action)
        assert rewards.shape == (B, 3), f"Expected (2,3), got {rewards.shape}"
        assert pred_state is not None and pred_state.shape == (B, 19)
        assert rewards.min() >= -1.0 and rewards.max() <= 1.0
        print("  PASS: Forward shape and range")

        inf_out = model.get_reward(dummy_img, dummy_prop)
        assert inf_out.shape == (B, 1)
        print("  PASS: Inference output")

    print("\nAll checks passed.")
