"""Reward-postprocessing helpers shared by ``sim/sac_train.py`` (v1),
``sim/sac_train_v2.py`` (v2) and ``sim/ppo_train.py``.

Every RL loop that uses the RewardModel combines the env reward with the RM
reward and optionally normalises / clips / confidence-weights the latter.
Keeping these operations in one place ensures that:

* all training scripts apply the *same* formula;
* behaviour changes (e.g. numerical stability, new knobs) propagate
  consistently;
* unit tests cover the post-processing regardless of which agent uses it.

Intentional non-goals:

* This module does **not** maintain running statistics of the RM reward
  stream — that lives with the relabeler (see
  :class:`AsyncRMRelabeler` in ``sim/sac_train_v2.py``), which is the
  natural owner because it is the component that actually writes new
  ``rm_rewards`` into the buffer.
* No device / dtype conversion beyond what is strictly required; the
  caller owns tensor placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch

ArrayLike = TypeVar("ArrayLike", torch.Tensor, np.ndarray)


@dataclass
class RMRewardConfig:
    """Configuration for the RM reward post-processing pipeline.

    Mirrors (and is the single source of truth for) the equivalent fields
    currently embedded in ``sac_train_v2.Args``. Attributes match the CLI
    names used by the training scripts so a ``Config -> RMRewardConfig``
    conversion is a simple struct copy.
    """

    alpha: float = 0.5
    """Shaping weight: ``total = env_scale * env + alpha * rm`` after warmup."""

    alpha_warmup_steps: int = 0
    """Linear warmup steps for ``alpha``. 0 disables warmup."""

    env_scale: float = 1.0
    """Multiplier applied to env reward. Set 0.0 for pure RM reward."""

    normalize: bool = False
    """Standardise ``rm_r`` with running mean/std before weighting."""

    clip: float = 0.0
    """Clip normalised ``rm_r`` to ``[-clip, +clip]``. 0 disables clipping."""

    use_uncertainty_weight: bool = False
    """Down-weight ``rm_r`` by ``1 / (1 + unc / running_std)``."""


def effective_alpha(step: int, warmup_steps: int, alpha_max: float) -> float:
    """Compute the warmup-scaled alpha.

    Matches ``min(1.0, step / warmup) * alpha_max`` from
    ``sim/sac_train_v2.py``. Returns ``alpha_max`` immediately when warmup
    is disabled or step is beyond it.
    """
    if warmup_steps <= 0:
        return alpha_max
    return min(1.0, step / warmup_steps) * alpha_max


def _fill_unlabeled_with_running_mean(
    rm_r: torch.Tensor,
    labeled_mask: torch.Tensor | None,
    running_mean: float,
) -> torch.Tensor:
    """Replace unlabeled entries with ``running_mean`` so fresh transitions
    do not bias SAC Q-learning toward zero rewards.

    ``labeled_mask`` is expected to be a bool / float tensor broadcastable
    to ``rm_r``. ``None`` disables the fallback (caller is responsible).
    """
    if labeled_mask is None:
        return rm_r
    mask = labeled_mask.bool() if labeled_mask.dtype != torch.bool else labeled_mask
    if mask.device != rm_r.device:
        mask = mask.to(rm_r.device)
    return torch.where(mask, rm_r, torch.full_like(rm_r, float(running_mean)))


def postprocess_rm_reward(
    rm_r: torch.Tensor,
    *,
    labeled_mask: torch.Tensor | None = None,
    running_mean: float = 0.0,
    running_std: float = 1.0,
    normalize: bool = False,
    clip: float = 0.0,
    uncertainty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the full RM reward post-processing pipeline.

    Steps (each is a no-op unless its knob is engaged):
        1. Substitute unlabeled entries with ``running_mean``.
        2. Normalise: ``(rm_r - running_mean) / (running_std + eps)``.
        3. Clip to ``[-clip, +clip]``.
        4. Confidence weight: ``rm_r * 1 / (1 + unc / running_std)``.

    Returns a tensor of the same shape / dtype as ``rm_r``.
    """
    rm_r = _fill_unlabeled_with_running_mean(rm_r, labeled_mask, running_mean)

    if normalize and running_std > 1e-6:
        rm_r = (rm_r - running_mean) / (running_std + 1e-8)

    if clip > 0.0:
        rm_r = rm_r.clamp(-clip, clip)

    if uncertainty is not None:
        if uncertainty.device != rm_r.device:
            uncertainty = uncertainty.to(rm_r.device)
        unc_normalised = uncertainty / (running_std + 1e-8) if running_std > 1e-6 else uncertainty
        confidence = 1.0 / (1.0 + unc_normalised)
        rm_r = rm_r * confidence

    return rm_r


def combine_env_and_rm(
    env_reward: ArrayLike,
    rm_reward: ArrayLike,
    *,
    alpha: float,
    env_scale: float = 1.0,
) -> ArrayLike:
    """Return the shaped reward actually used by the RL update.

    ``total = env_scale * env_reward + alpha * rm_reward``

    Works element-wise on both ``torch.Tensor`` and ``np.ndarray`` inputs
    (they must be the same type). Use :func:`effective_alpha` to get the
    warmup-scaled alpha.
    """
    return env_scale * env_reward + alpha * rm_reward
