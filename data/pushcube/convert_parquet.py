"""Convert PushCube LeRobot v3 parquet datasets to raw transition pkl format.

Reads parquet + meta from datasets/PushCube-v1 and PushCube-v1_failed,
outputs raw list[dict] pkl files matching the project's standard format
(see data/data_structure.txt).

Usage:
    python -m data.pushcube.convert_parquet
"""

import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from data.common import save_pickle

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
OUTPUT_DIR = Path(__file__).resolve().parent

CAMERA_KEYS = ["wrist", "third"]

SOURCES = [
    {
        "parquet_dir": DATASETS_DIR / "PushCube-v1",
        "output_name": "pushcube_success.pkl",
        "is_success": True,
    },
    {
        "parquet_dir": DATASETS_DIR / "PushCube-v1_failed",
        "output_name": "pushcube_fail.pkl",
        "is_success": False,
    },
]


def _decode_image(raw: dict) -> np.ndarray:
    """Decode PNG bytes from parquet image column to uint8 array (H,W,3)."""
    img = Image.open(BytesIO(raw["bytes"]))
    return np.array(img, dtype=np.uint8)


def _build_state(eef: list, joint: list) -> np.ndarray:
    """Concatenate EEF (8) + joint (9) into a 1-D (17,) state vector."""
    return np.concatenate(
        [
            np.array(eef, dtype=np.float32),
            np.array(joint, dtype=np.float32),
        ]
    )


def convert_parquet_to_raw(parquet_dir: Path, is_success: bool) -> list[dict]:
    """Convert a LeRobot v3 parquet dataset to raw transition list."""
    data_files = sorted((parquet_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir / 'data'}")
    tables = [pq.read_table(f) for f in data_files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    n = len(table)

    ep_col = table.column("episode_index").to_pylist()
    frame_col = table.column("frame_index").to_pylist()

    ep_meta_files = sorted((parquet_dir / "meta" / "episodes").rglob("*.parquet"))
    ep_tables = [pq.read_table(f) for f in ep_meta_files]
    ep_table = pa.concat_tables(ep_tables) if len(ep_tables) > 1 else ep_tables[0]
    ep_lengths = dict(
        zip(
            ep_table.column("episode_index").to_pylist(),
            ep_table.column("length").to_pylist(),
            strict=True,
        )
    )

    transitions: list[dict] = []
    for i in range(n):
        ep_idx = ep_col[i]
        frame_idx = frame_col[i]
        ep_len = ep_lengths.get(ep_idx)
        if ep_len is None:
            logging.warning("episode_index %d not found in meta, skipping", ep_idx)
            continue
        is_last_frame = frame_idx == ep_len - 1

        eef = table.column("observation.state_eef")[i].as_py()
        joint = table.column("observation.state_joint")[i].as_py()
        state = _build_state(eef, joint)

        obs: dict = {"state": state}
        for cam_key in CAMERA_KEYS:
            col_name = f"observation.images.{cam_key}"
            raw_img = table.column(col_name)[i].as_py()
            obs[cam_key] = _decode_image(raw_img)[np.newaxis, ...]  # (1, H, W, 3)

        action = np.array(table.column("action")[i].as_py(), dtype=np.float64)

        done = 1 if is_last_frame else 0
        succeed = is_success if is_last_frame else False

        step = {
            "observations": obs,
            "actions": action,
            "rewards": 0,
            "dones": done,
            "infos": {"succeed": succeed},
        }
        transitions.append(step)

    return transitions


def main() -> None:
    for src in SOURCES:
        parquet_dir = src["parquet_dir"]
        output_path = OUTPUT_DIR / src["output_name"]
        is_success = src["is_success"]

        if not parquet_dir.exists():
            print(f"[SKIP] {parquet_dir} not found")
            continue

        print(f"Converting {parquet_dir.name} ...")
        data = convert_parquet_to_raw(parquet_dir, is_success=is_success)

        save_pickle(data, output_path)

        from data.common import get_episodes

        episodes = get_episodes(data)
        success_count = sum(1 for ep in episodes if ep and ep[-1].get("infos", {}).get("succeed", False))
        print(f"  -> {output_path.name}: {len(data)} transitions, {len(episodes)} episodes, {success_count} success")

    print("Done.")


if __name__ == "__main__":
    main()
