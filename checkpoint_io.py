"""Unified checkpoint artifact format for RL (and optionally RM) training.

All RL training scripts (``sim/sac_train.py`` v1, ``sim/sac_train_v2.py``
v2, ``sim/ppo_train.py``) now route their saves and loads through
:func:`save_rl_checkpoint` and :func:`load_rl_checkpoint`. This gives us:

* A single ``schema`` and ``version`` field so loaders can detect format
  drift instead of silently KeyError-ing.
* A ``meta`` slot for training context (env, seed, global step, RM
  checkpoint path, alpha, …) that downstream tooling (eval, release
  notes, regression tests) can read.
* A ``created_at`` ISO-timestamp for provenance.

Wire format (written by :func:`save_rl_checkpoint`)::

    {
        "schema": "autorm_rl_ckpt",
        "version": 1,
        "created_at": "2026-04-18T...",
        "meta":  { ...arbitrary JSON-serialisable... },
        "state": { ...module name -> state_dict (or tensor)... },
    }

Loading honours PyTorch's ``weights_only=True`` path by default; pass
``weights_only=False`` only for trusted checkpoints. Legacy flat
checkpoints (``{actor, qf1, qf2, ...}`` with no ``schema``) are detected
by the absence of the ``schema`` key and returned as-is via the ``state``
slot with an empty ``meta``.

The :class:`RewardModel`'s ``save``/``load`` remain on their own
state_dict + sibling JSON format — RM checkpoints are a distinct artifact
because they need the full model config to reconstruct the class at load
time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

SCHEMA: str = "autorm_rl_ckpt"
VERSION: int = 1


def save_rl_checkpoint(
    path: str | Path,
    *,
    state: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> None:
    """Write an RL checkpoint in the unified schema.

    Args:
        path: Destination path (``.pt``). Parent dirs must already exist.
        state: Mapping ``name -> state_dict | tensor``. Typically includes
            ``actor``, ``qf1``, ``qf2``, ``qf*_target``, ``log_alpha``, …
        meta: Optional JSON-serialisable dict with training provenance
            (env id, seed, global step, RM checkpoint path, etc.).

    Raises:
        TypeError: if ``state`` contains entries that are neither a
            ``state_dict`` nor a ``torch.Tensor``.
    """
    _validate_state(state)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "meta": dict(meta) if meta else {},
        "state": state,
    }
    torch.save(payload, path)


def load_rl_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    weights_only: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load an RL checkpoint and return ``(state, meta)``.

    * New-format checkpoints are unwrapped: returns ``(payload["state"],
      payload.get("meta", {}))``.
    * Legacy flat checkpoints (no ``schema``) are returned as
      ``(raw_dict, {})`` so callers need only migrate their save site.

    Args:
        path: Path to checkpoint ``.pt``.
        map_location: Passed through to ``torch.load``.
        weights_only: Passed through to ``torch.load``. Default ``True``
            (safe for any checkpoint written by :func:`save_rl_checkpoint`).

    Raises:
        ValueError: on future-schema checkpoints we do not know how to
            read.
    """
    raw = torch.load(path, map_location=map_location, weights_only=weights_only)

    if not isinstance(raw, dict) or "schema" not in raw:
        return raw if isinstance(raw, dict) else {"state_dict": raw}, {}

    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"Unknown checkpoint schema {schema!r} at {path}")

    version = raw.get("version", 0)
    if version > VERSION:
        raise ValueError(
            f"Checkpoint {path} was written with schema version {version}, "
            f"but this installation understands up to {VERSION}. Upgrade the code."
        )

    state = raw.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} missing 'state' dict (got {type(state).__name__})")

    meta = raw.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    return state, meta


def _validate_state(state: dict[str, Any]) -> None:
    """Sanity-check that ``state`` is shaped as expected.

    We accept either ``state_dict``-style mappings (``dict`` of
    ``str -> Tensor``) or a single ``Tensor`` value (for scalars like
    ``log_alpha``). Anything else is almost certainly a programming error
    and would break ``weights_only=True`` loading, so we reject eagerly.
    """
    for key, value in state.items():
        if value is None:
            continue  # Some trainers store `None` slots (e.g. log_alpha when not autotuned)
        if isinstance(value, torch.Tensor):
            continue
        if isinstance(value, dict):
            continue
        raise TypeError(f"state['{key}'] must be a state_dict or tensor, got {type(value).__name__}")


__all__ = ["SCHEMA", "VERSION", "save_rl_checkpoint", "load_rl_checkpoint"]
