"""ManiSkill3 task configuration registry.

Defines per-task simulation parameters (env creation, obs extraction, camera
mapping) and links them to the offline ``data.common.TaskEntry`` for RM
checkpoint resolution.

Extend ``SIM_REGISTRY`` to add new ManiSkill3 tasks.
"""

from dataclasses import dataclass, field


@dataclass
class SimTaskConfig:
    """Configuration for one ManiSkill3 task."""

    env_id: str
    control_mode: str = "pd_ee_delta_pose"
    obs_mode: str = "state+rgb"
    robot_uid: str = "panda_wristcam"
    max_episode_steps: int = 100
    sim_backend: str = "physx_cuda"

    camera_map: dict[str, str] = field(
        default_factory=lambda: {"hand_camera": "wrist", "base_camera": "third"},
    )

    state_mode: str = "eef+joint"
    rm_state_dim: int = 17
    action_dim: int = 7

    rm_checkpoint: str | None = None
    ppo_checkpoint: str | None = None

    reward_shaping_weight: float = 0.5


def _build_registry() -> dict[str, SimTaskConfig]:
    return {
        "pushcube": SimTaskConfig(
            env_id="PushCube-v1",
            robot_uid="panda_wristcam",
            sim_backend="gpu",
            rm_state_dim=17,
            action_dim=7,
            rm_checkpoint="checkpoints/auto_pushcube/best.pt",
            ppo_checkpoint=None,
        ),
        "pokecube": SimTaskConfig(
            env_id="PokeCube-v1",
            robot_uid="panda_wristcam",
            sim_backend="gpu",
            rm_state_dim=17,
            action_dim=7,
            rm_checkpoint="checkpoints/auto_pokecube/best.pt",
            ppo_checkpoint=None,
        ),
        "placesphere": SimTaskConfig(
            env_id="PlaceSphere-v1",
            robot_uid="panda_wristcam",
            sim_backend="gpu",
            rm_state_dim=17,
            action_dim=7,
            rm_checkpoint="checkpoints/auto_placesphere/best.pt",
            ppo_checkpoint=None,
        ),
        "stackcube": SimTaskConfig(
            env_id="StackCube-v1",
            robot_uid="panda_wristcam",
            sim_backend="gpu",
            rm_state_dim=17,
            action_dim=7,
            rm_checkpoint="checkpoints/auto_stackcube/best.pt",
            ppo_checkpoint=None,
        ),
    }


SIM_REGISTRY: dict[str, SimTaskConfig] = _build_registry()


def get_sim_config(task_name: str) -> SimTaskConfig:
    """Look up simulation config by task name."""
    if task_name not in SIM_REGISTRY:
        available = ", ".join(sorted(SIM_REGISTRY))
        raise KeyError(f"Unknown sim task '{task_name}'. Available: {available}")
    return SIM_REGISTRY[task_name]
