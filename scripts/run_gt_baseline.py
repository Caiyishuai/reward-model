#!/usr/bin/env python
"""SAC GT-reward baseline: verify SAC can learn with sim reward only."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RM_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from sim.sac_train import SACConfig, train  # noqa: E402

c = SACConfig(
    task="pushcube",
    rm_checkpoint=None,
    reward_shaping_weight=0.0,
    total_timesteps=50_000,
    learning_starts=2_000,
    eval_freq=5_000,
    eval_episodes=20,
    save_freq=25_000,
    buffer_size=50_000,
    batch_size=256,
    lr=3e-4,
    output_dir="runs/gt_baseline",
    device="cuda",
    seed=42,
)
train(c)
