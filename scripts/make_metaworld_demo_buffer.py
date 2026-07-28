#!/usr/bin/env python3
"""Extract the N success episodes from an exported SERL replay file into a
compact demo-buffer file (serl_demos_<mode>.pkl).

The exported serl_<mode>.pkl files contain 20 success + 20 fail episodes
(episode_index 0..num_success-1 are the successes). SERL / RLPD expects the
demo buffer to hold successful demonstrations only, so this tool slices out the
first ``success_episodes`` trajectories (read from collection_meta.json).
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent / "data"

KEYS = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "masks",
    "dones",
    "episode_index",
)


def make_demo(task_dir: Path, mode: str) -> Path:
    src = task_dir / f"serl_{mode}.pkl"
    meta = json.loads((task_dir / "collection_meta.json").read_text())
    n_success = int(meta["success_episodes"])

    with open(src, "rb") as f:
        data = pickle.load(f)

    ep = np.asarray(data["episode_index"])
    keep = ep < n_success  # success episodes are indexed first

    out = {k: np.asarray(data[k])[keep] for k in KEYS if k in data}
    n_eps = int(np.unique(out["episode_index"]).size)

    dst = task_dir / f"serl_demos_{mode}.pkl"
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    print(
        f"{task_dir.name:20s} mode={mode:6s} demos={n_eps:3d} "
        f"transitions={out['actions'].shape[0]:5d} -> {dst.name}"
    )
    return dst


def main():
    ap = argparse.ArgumentParser(description="Build SERL demo buffer (success-only)")
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument(
        "--reward-modes", nargs="+", default=["dense", "sparse", "auto"],
        choices=["auto", "dense", "sparse"],
    )
    args = ap.parse_args()

    if args.tasks == ["all"]:
        task_dirs = sorted(p for p in BASE_DIR.glob("mw_*") if p.is_dir())
    else:
        task_dirs = [BASE_DIR / t for t in args.tasks]

    for task_dir in task_dirs:
        if not (task_dir / "collection_meta.json").exists():
            print(f"[skip] {task_dir.name}: no collection_meta.json")
            continue
        for mode in args.reward_modes:
            if (task_dir / f"serl_{mode}.pkl").exists():
                make_demo(task_dir, mode)
            else:
                print(f"[skip] {task_dir.name}: serl_{mode}.pkl missing")


if __name__ == "__main__":
    main()
