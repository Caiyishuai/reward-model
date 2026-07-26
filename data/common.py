"""Shared constants, utilities, and unified task registry for the data pipeline.

Single source of truth for state indices, episode parsing, data I/O,
and per-task configuration used by auto_label, manual_label, preprocess,
train, and data analysis tools.
"""

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

# ==========================================
# Robot State Layout
# ==========================================
STATE_INDICES = {
    "gripper": slice(0, 1),
    "force": slice(1, 4),
    "pose": slice(4, 10),
    "pos": slice(4, 7),
    "rot": slice(7, 10),
    "torque": slice(10, 13),
    "vel_lin": slice(13, 16),
    "vel_ang": slice(16, 19),
}

ROBOT_DIM = 19

# ==========================================
# Image / Temporal Defaults (single source of truth)
# ==========================================
# DINOv2 backbone input size; used by RM training, dataset loaders, online
# inference and the RM relabeling path in SAC/PPO.
IMG_SIZE_RM = 224
# SAC policy-side CNN input size (smaller for throughput, independent of RM).
IMG_SIZE_SAC = 128
# Default temporal window length used by the RewardModel and its callers.
# The TemporalAdapter's positional embedding is sized to
# ``max(MAX_SEQ_LEN_MIN, STATE_WINDOWS_DEFAULT + MAX_SEQ_LEN_SLACK)``.
STATE_WINDOWS_DEFAULT = 3
# Floor for TemporalAdapter.max_seq_len so short windows still leave slack for
# CLS / special tokens that external callers may append.
MAX_SEQ_LEN_MIN = 10
# Extra positional-embedding slots beyond state_windows (buffer for longer
# windows without re-training, see reward_model.TemporalAdapter).
MAX_SEQ_LEN_SLACK = 2

_default_data_dir = os.environ.get("RM_DATA_DIR")
if _default_data_dir is None:
    _default_data_dir = str(Path(__file__).resolve().parent.parent / "data")
BASE_DIR = Path(_default_data_dir)


# ==========================================
# Task Registry
# ==========================================
@dataclass
class TaskEntry:
    """Unified per-task configuration for labeling, training, and analysis.

    Fields are grouped by subsystem:
    - Common: paths and camera config
    - Auto-label: HMM stage discovery parameters
    - Manual-label: potential-field keypoint targets
    """

    # Common
    name: str
    success_path: str
    fail_path: str
    camera_keys: list[str] = field(default_factory=lambda: ["wrist_1", "wrist_2"])

    # State index layout (defaults match 19-dim real-robot state)
    state_pos_slice: slice = field(default_factory=lambda: slice(4, 7))
    state_force_slice: slice | None = field(default_factory=lambda: slice(1, 4))
    state_gripper_idx: int = 0

    # Auto-label config
    n_stages: int | None = None
    max_stages: int = 5
    check_gripper: bool = False
    use_force_dynamics: bool = False
    penalty_power: float = 1.0
    n_restarts: int = 10
    hmm_n_iter: int = 200
    use_prototype_stages: bool = False
    contact_force_threshold: float = 1.0

    # Manual-label config (potential-field keypoint targets)
    target_positions: dict[str, list[list[float]]] | None = None


def _build_registry() -> dict[str, TaskEntry]:
    """Build the canonical task registry. All task paths and configs live here."""
    return {
        "pickup": TaskEntry(
            name="pickup",
            success_path=f"{BASE_DIR}/pickup/pickup_success_20_demos_2026-01-10_14-51-39.pkl",
            fail_path=f"{BASE_DIR}/pickup/pickup_failure_20_demos_2026-01-10_14-51-41.pkl",
            check_gripper=True,
            use_force_dynamics=True,
            penalty_power=3.0,
            target_positions={
                "k0": [
                    [0.00640625, -0.00620482, 0.03940177],
                    [-0.01185292, -0.0147341, 0.0413675],
                    [0.00651142, 0.00071618, 0.04027406],
                    [0.00033376, 0.01761943, 0.03754395],
                    [-0.02816998, 0.00208793, 0.03585935],
                    [-0.00140552, -0.00963043, 0.04034723],
                    [-0.03535765, 0.00238707, 0.04341285],
                    [-0.0093194, 0.01507648, 0.03875196],
                    [-0.00333431, 0.00660923, 0.03932335],
                    [0.02080778, -0.00911316, 0.0404017],
                    [0.00754955, -0.00526306, 0.03825398],
                    [-0.01628396, -0.00063944, 0.03927459],
                    [-0.01230491, -0.01655152, 0.04022428],
                    [-0.01335198, 0.02405355, 0.03747394],
                    [0.0315534, 0.01283737, 0.03681832],
                    [-0.00962507, -0.02112815, 0.04038769],
                    [-0.01435191, -0.01291251, 0.04096404],
                    [-0.00141499, -0.00821233, 0.03880323],
                    [0.01632789, -0.01229154, 0.04156873],
                    [-0.00367197, 0.02268588, 0.0381972],
                ],
                "k1": [[0.00217855, 0.0096429, -0.0225036]],
            },
        ),
        "pickup_failure": TaskEntry(
            name="pickup_failure",
            success_path=f"{BASE_DIR}/pickup/pickup_success_20_demos_2026-01-10_14-51-39.pkl",
            fail_path=f"{BASE_DIR}/pickup/pickup_failure_20_demos_2026-01-10_14-51-41.pkl",
            target_positions={
                "k0": [
                    [-0.01590205, -0.01290739, 0.04391432],
                    [-0.04136332, 0.01263248, 0.03562428],
                    [-0.00667803, -0.00785904, 0.04026861],
                    [-0.01913033, 0.01393042, 0.03814394],
                    [-0.00833596, -0.00794096, 0.03666535],
                    [0.0003232, -0.0086476, 0.04052001],
                ],
                "k1": [[0.00217855, 0.0096429, -0.0225036]],
            },
        ),
        "plug_insert": TaskEntry(
            name="plug_insert",
            success_path=f"{BASE_DIR}/plug_insert/plug_insert_20_demos_2026-01-11_12-52-48.pkl",
            fail_path=f"{BASE_DIR}/plug_insert/plug_insert_20_fail_demos_2026-01-11_12-58-44.pkl",
            target_positions={
                "k0": [[-1.1337542e-02, 7.3714342e-05, 5.3255804e-02]],
            },
        ),
        "iphone_insert": TaskEntry(
            name="iphone_insert",
            success_path=f"{BASE_DIR}/iphone_insert/iphone_insert_20_demos_2026-01-12_15-58-47.pkl",
            fail_path=f"{BASE_DIR}/iphone_insert/iphone_insert_20_fail_demos_2026-01-12_16-04-53.pkl",
            target_positions={
                "k0": [[1.6812701e-06, -3.2935578e-03, 4.6115994e-02]],
            },
        ),
        "button": TaskEntry(
            name="button",
            success_path=f"{BASE_DIR}/button/push_button_success_20_demos_2026-01-13_05-50-37.pkl",
            fail_path=f"{BASE_DIR}/button/push_button_failure_20_demos_2026-01-13_06-05-00.pkl",
            use_force_dynamics=True,
            penalty_power=3.0,
            target_positions={
                "k0": [[-0.01114031, -0.0031739, 0.05255952]],
            },
        ),
        "usb": TaskEntry(
            name="usb",
            success_path=f"{BASE_DIR}/usb/usb_insert_20_demos_2026-01-13_01-51-36.pkl",
            fail_path=f"{BASE_DIR}/usb/usb_insert_failure_20_demos_2026-01-13_02-10-57.pkl",
            target_positions={
                "k0": [[0.00051164, 0.00428889, 0.07199589, 0.06110301, 0.06311922, -0.00010049]],
            },
        ),
        "key": TaskEntry(
            name="key",
            success_path=f"{BASE_DIR}/key/key_success_20_demos_2026-01-26_07-21-06.pkl",
            fail_path=f"{BASE_DIR}/key/key_failure_20_demos_2026-01-18_04-05-34.pkl",
            target_positions={
                "k0": [[0.00185677, 0.00219854, 0.03579868, 0.01498105, 0.00430112, 0.00252195]],
            },
        ),
        "op_dr": TaskEntry(
            name="op_dr",
            success_path=f"{BASE_DIR}/op_dr/open_success_20_demos_2026-01-18_04-53-21.pkl",
            fail_path=f"{BASE_DIR}/op_dr/open_failure_20_demos_2026-01-18_04-53-22.pkl",
            target_positions={
                "k0": [[0.01425169, -0.01492897, 0.01922264, 0.01957859, 0.00308654, -0.00549553]],
                "k1": [[-0.03674399, 0.00895164, -0.01820534, -0.00542592, 0.00724713, -0.00142918]],
            },
        ),
        "pk_toy": TaskEntry(
            name="pk_toy",
            success_path=f"{BASE_DIR}/pk_toy/pickup_toy_success_20_demos_2026-01-18_05-41-22.pkl",
            fail_path=f"{BASE_DIR}/pk_toy/pickup_toy_failure_20_demos_2026-01-18_05-41-23.pkl",
            target_positions={
                "k0": [[-0.01956423, -0.00225834, 0.05161842]],
                "k1": [[-0.01417353, -0.0040108, -0.00893513]],
            },
        ),
        "pl_toy": TaskEntry(
            name="pl_toy",
            success_path=f"{BASE_DIR}/pl_toy/place_success_20_demos_2026-01-18_06-35-12.pkl",
            fail_path=f"{BASE_DIR}/pl_toy/place_failure_20_demos_2026-01-18_06-35-13.pkl",
            target_positions={
                "k0": [[-0.03902494, -0.01136104, 0.06752809, 0.00957903, -0.00039864, 0.01348135]],
            },
        ),
        "pushcube": TaskEntry(
            name="pushcube",
            success_path=f"{BASE_DIR}/pushcube/success_raw.pkl",
            fail_path=f"{BASE_DIR}/pushcube/fail_raw.pkl",
            camera_keys=["hand_camera", "base_camera"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=6,
        ),
        "pokecube": TaskEntry(
            name="pokecube",
            success_path=f"{BASE_DIR}/pokecube/success_raw.pkl",
            fail_path=f"{BASE_DIR}/pokecube/fail_raw.pkl",
            camera_keys=["hand_camera", "base_camera"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=6,
        ),
        "placesphere": TaskEntry(
            name="placesphere",
            success_path=f"{BASE_DIR}/placesphere/success_raw.pkl",
            fail_path=f"{BASE_DIR}/placesphere/fail_raw.pkl",
            camera_keys=["hand_camera", "base_camera"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=6,
            check_gripper=True,
        ),
        "stackcube": TaskEntry(
            name="stackcube",
            success_path=f"{BASE_DIR}/stackcube/success_raw.pkl",
            fail_path=f"{BASE_DIR}/stackcube/fail_raw.pkl",
            camera_keys=["hand_camera", "base_camera"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=6,
            check_gripper=True,
        ),
        "mw_button_press": TaskEntry(
            name="mw_button_press",
            success_path=f"{BASE_DIR}/mw_button_press/success_raw.pkl",
            fail_path=f"{BASE_DIR}/mw_button_press/fail_raw.pkl",
            camera_keys=["corner2"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=3,
            n_restarts=2,
            hmm_n_iter=50,
            use_prototype_stages=True,
        ),
        "mw_window_open": TaskEntry(
            name="mw_window_open",
            success_path=f"{BASE_DIR}/mw_window_open/success_raw.pkl",
            fail_path=f"{BASE_DIR}/mw_window_open/fail_raw.pkl",
            camera_keys=["corner2"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=3,
            n_restarts=2,
            hmm_n_iter=50,
            use_prototype_stages=True,
        ),
        "mw_reach_wall": TaskEntry(
            name="mw_reach_wall",
            success_path=f"{BASE_DIR}/mw_reach_wall/success_raw.pkl",
            fail_path=f"{BASE_DIR}/mw_reach_wall/fail_raw.pkl",
            camera_keys=["corner2"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=3,
            n_restarts=2,
            hmm_n_iter=50,
            use_prototype_stages=True,
        ),
        "mw_plate_slide": TaskEntry(
            name="mw_plate_slide",
            success_path=f"{BASE_DIR}/mw_plate_slide/success_raw.pkl",
            fail_path=f"{BASE_DIR}/mw_plate_slide/fail_raw.pkl",
            camera_keys=["corner2"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=3,
            n_restarts=2,
            hmm_n_iter=50,
            use_prototype_stages=True,
        ),
        "mw_push": TaskEntry(
            name="mw_push",
            success_path=f"{BASE_DIR}/mw_push/success_raw.pkl",
            fail_path=f"{BASE_DIR}/mw_push/fail_raw.pkl",
            camera_keys=["corner2"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=3,
            n_restarts=2,
            hmm_n_iter=50,
            use_prototype_stages=True,
        ),
        "mw_coffee_push": TaskEntry(
            name="mw_coffee_push",
            success_path=f"{BASE_DIR}/mw_coffee_push/success_raw.pkl",
            fail_path=f"{BASE_DIR}/mw_coffee_push/fail_raw.pkl",
            camera_keys=["corner2"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=3,
            n_restarts=2,
            hmm_n_iter=50,
            use_prototype_stages=True,
        ),
        "mw_stick_push": TaskEntry(
            name="mw_stick_push",
            success_path=f"{BASE_DIR}/mw_stick_push/success_raw.pkl",
            fail_path=f"{BASE_DIR}/mw_stick_push/fail_raw.pkl",
            camera_keys=["corner2"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=3,
            n_restarts=2,
            hmm_n_iter=50,
            use_prototype_stages=True,
        ),
        "mw_pick_place": TaskEntry(
            name="mw_pick_place",
            success_path=f"{BASE_DIR}/mw_pick_place/success_raw.pkl",
            fail_path=f"{BASE_DIR}/mw_pick_place/fail_raw.pkl",
            camera_keys=["corner2"],
            state_pos_slice=slice(0, 3),
            state_force_slice=None,
            state_gripper_idx=3,
            n_restarts=2,
            hmm_n_iter=50,
            use_prototype_stages=True,
        ),
    }


TASK_REGISTRY: dict[str, TaskEntry] = _build_registry()


def get_task(name: str) -> TaskEntry:
    """Look up a task by name, raising KeyError with available tasks on miss."""
    if name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    return TASK_REGISTRY[name]


def list_tasks() -> list[str]:
    """Return sorted list of registered task names."""
    return sorted(TASK_REGISTRY.keys())


# ==========================================
# Episode Parsing
# ==========================================
def get_episodes(data: list[dict]) -> list[list[dict]]:
    """Split flat transition list into episodes by done flags."""
    episodes: list[list[dict]] = []
    current: list[dict] = []
    for x in data:
        current.append(x)
        done = x.get("dones", False)
        if hasattr(done, "item"):
            done = done.item() == 1
        elif isinstance(done, (int, float)):
            done = done == 1
        if done:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


# ==========================================
# Data I/O
# ==========================================
def load_pickle(path: str | Path) -> object:
    """Load a pickle file from a trusted local path.

    WARNING: pickle deserialization can execute arbitrary code.
    Only call on paths within the project data directory.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    # Use Path.is_relative_to (3.9+) instead of string startswith: the latter
    # accepts prefix-siblings (e.g. "/data-evil" matches "/data"), which would
    # bypass the root whitelist.
    allowed_roots = (BASE_DIR.resolve(), Path.cwd().resolve())
    if not any(path.is_relative_to(root) for root in allowed_roots):
        raise ValueError(f"Refusing to load pickle outside allowed roots: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted local data only


def save_pickle(data: object, path: str | Path) -> None:
    """Save data to pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_json(path: str | Path) -> object:
    """Load a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path) as f:
        return json.load(f)


def save_json(data: object, path: str | Path, *, indent: int = 2) -> None:
    """Write *data* to a JSON file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
