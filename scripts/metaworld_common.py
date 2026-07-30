"""Shared MetaWorld task registry and environment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import metaworld
import metaworld.policies as policies
import mujoco
import numpy as np
from PIL import Image

VISUAL_CAMERA_SCHEMA = {
    "wrist_1": "behindGripper",
    "wrist_2": "gripperPOV",
    "front": "corner2",
}
ROBOT_STATE_DIM = 19


@dataclass(frozen=True)
class MetaWorldTaskSpec:
    key: str
    env_name: str
    rsync_name: str
    policy_class: type

    @property
    def observable_env_name(self) -> str:
        return f"{self.env_name}-goal-observable"


TASK_SPECS: dict[str, MetaWorldTaskSpec] = {
    "button-press": MetaWorldTaskSpec(
        "button-press", "button-press-v3", "mw_button_press", policies.SawyerButtonPressV3Policy
    ),
    "window-open": MetaWorldTaskSpec(
        "window-open", "window-open-v3", "mw_window_open", policies.SawyerWindowOpenV3Policy
    ),
    "reach-wall": MetaWorldTaskSpec("reach-wall", "reach-wall-v3", "mw_reach_wall", policies.SawyerReachWallV3Policy),
    "plate-slide": MetaWorldTaskSpec(
        "plate-slide", "plate-slide-v3", "mw_plate_slide", policies.SawyerPlateSlideV3Policy
    ),
    "push": MetaWorldTaskSpec("push", "push-v3", "mw_push", policies.SawyerPushV3Policy),
    "coffee-push": MetaWorldTaskSpec(
        "coffee-push", "coffee-push-v3", "mw_coffee_push", policies.SawyerCoffeePushV3Policy
    ),
    "stick-push": MetaWorldTaskSpec("stick-push", "stick-push-v3", "mw_stick_push", policies.SawyerStickPushV3Policy),
    "pick-place": MetaWorldTaskSpec("pick-place", "pick-place-v3", "mw_pick_place", policies.SawyerPickPlaceV3Policy),
}


@dataclass(frozen=True)
class WrenchObservation:
    """Contact wrench acting on the end-effector subtree, in the wrist frame.

    MuJoCo's ``mj_contactForce`` returns force/torque in each contact frame.
    We transform every external contact involving the robot wrist subtree to
    world coordinates, shift torque to the wrist site, sum, and finally rotate
    the result into the wrist-site frame.
    """

    wrist_wrench: np.ndarray
    contact_force: np.ndarray
    max_contact_force: float
    contact_count: int


class WristWrenchSensor:
    """Stable six-axis virtual wrist force/torque sensor for MetaWorld.

    MetaWorld's Sawyer XML has no native MuJoCo force/torque sensors. This
    class synthesizes the equivalent signal from solved contact constraints.
    ``right_l6`` is the subtree above the hand and gripper, while
    ``endEffector`` defines the reported wrench frame and reference point.
    """

    def __init__(
        self,
        env: Any,
        *,
        subtree_root: str = "right_l6",
        wrist_site: str = "endEffector",
        filter_mode: str = "ema",
        filter_alpha: float = 0.2,
        force_clip: float = 100.0,
        torque_clip: float = 10.0,
    ):
        if filter_mode not in {"ema", "none"}:
            raise ValueError(f"filter_mode must be 'ema' or 'none', got {filter_mode!r}")
        if not 0.0 < filter_alpha <= 1.0:
            raise ValueError(f"filter_alpha must be in (0, 1], got {filter_alpha}")
        self.env = env.unwrapped
        self.model = self.env.model
        self.data = self.env.data
        self.filter_mode = filter_mode
        self.filter_alpha = float(filter_alpha)
        self.force_clip = float(force_clip)
        self.torque_clip = float(torque_clip)
        self.root_body_id = int(self.model.body(subtree_root).id)
        self.wrist_site_id = int(self.model.site(wrist_site).id)
        self.subtree_body_ids = self._descendants(self.root_body_id)
        self._filtered = np.zeros(6, dtype=np.float32)

    def _descendants(self, root_body_id: int) -> set[int]:
        descendants = set()
        for body_id in range(self.model.nbody):
            current = body_id
            while current > 0:
                if current == root_body_id:
                    descendants.add(body_id)
                    break
                current = int(self.model.body_parentid[current])
        return descendants

    def reset(self) -> WrenchObservation:
        self._filtered.fill(0.0)
        return self.read()

    def read(self) -> WrenchObservation:
        wrist_position = np.asarray(self.data.site_xpos[self.wrist_site_id], dtype=np.float64)
        wrist_rotation = np.asarray(self.data.site_xmat[self.wrist_site_id], dtype=np.float64).reshape(3, 3)
        world_force = np.zeros(3, dtype=np.float64)
        world_torque = np.zeros(3, dtype=np.float64)
        contact_force_vectors = []

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            body0 = int(self.model.geom_bodyid[contact.geom[0]])
            body1 = int(self.model.geom_bodyid[contact.geom[1]])
            robot0 = body0 in self.subtree_body_ids
            robot1 = body1 in self.subtree_body_ids
            if robot0 == robot1:
                continue

            contact_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_id, contact_wrench)
            # MuJoCo reports the wrench acting on geom[1]. Reverse it when the
            # robot subtree is geom[0].
            sign = 1.0 if robot1 else -1.0
            contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            force = sign * (contact_frame.T @ contact_wrench[:3])
            torque_at_contact = sign * (contact_frame.T @ contact_wrench[3:])
            contact_position = np.asarray(contact.pos, dtype=np.float64)

            world_force += force
            world_torque += torque_at_contact + np.cross(contact_position - wrist_position, force)
            contact_force_vectors.append(force)

        wrench = np.concatenate(
            [
                wrist_rotation.T @ world_force,
                wrist_rotation.T @ world_torque,
            ]
        ).astype(np.float32)
        wrench[:3] = np.clip(wrench[:3], -self.force_clip, self.force_clip)
        wrench[3:] = np.clip(wrench[3:], -self.torque_clip, self.torque_clip)
        if self.filter_mode == "ema":
            self._filtered = (
                self.filter_alpha * wrench + (1.0 - self.filter_alpha) * self._filtered
            ).astype(np.float32)
        else:
            self._filtered = wrench

        if contact_force_vectors:
            contact_forces = np.stack(contact_force_vectors).astype(np.float32)
            max_contact_force = float(np.linalg.norm(contact_forces, axis=1).max())
        else:
            contact_forces = np.zeros((0, 3), dtype=np.float32)
            max_contact_force = 0.0
        return WrenchObservation(
            wrist_wrench=self._filtered.copy(),
            contact_force=contact_forces,
            max_contact_force=max_contact_force,
            contact_count=len(contact_force_vectors),
        )


def get_task_spec(task: str) -> MetaWorldTaskSpec:
    try:
        return TASK_SPECS[task]
    except KeyError as error:
        raise KeyError(f"Unknown MetaWorld task {task!r}; available: {sorted(TASK_SPECS)}") from error


def make_env(
    task: str,
    *,
    seed: int,
    render: bool = False,
    image_size: int = 128,
    reward_function_version: str = "v2",
    camera_name: str = "corner2",
) -> Any:
    """Create a goal-observable MetaWorld v3 environment.

    MetaWorld 3.1's generated goal-observable constructors expose only
    ``seed`` and ``render_mode``.  Rendering dimensions and reward version are
    therefore set on the base environment immediately after construction.
    """
    spec = get_task_spec(task)
    env_class = metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[spec.observable_env_name]
    env = env_class(seed=seed, render_mode="rgb_array" if render else None)
    env.width = image_size
    env.height = image_size
    env._rsync_image_size = image_size
    env.reward_function_version = reward_function_version
    if render:
        try:
            camera_id = int(env.model.camera(camera_name).id)
        except KeyError as error:
            available = [env.model.camera(index).name for index in range(env.model.ncam)]
            raise ValueError(f"Unknown camera {camera_name!r}; available: {available}") from error
        env.mujoco_renderer.camera_id = camera_id
        env.mujoco_renderer.width = image_size
        env.mujoco_renderer.height = image_size
        env._rsync_camera_name = camera_name
    return env


def render_rgb(env: Any) -> np.ndarray:
    image = np.asarray(env.render())
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max(initial=0) <= 1.0 else image, 0, 255).astype(np.uint8)
    target_size = int(getattr(env, "_rsync_image_size", image.shape[0]))
    if image.shape[:2] != (target_size, target_size):
        image = np.asarray(Image.fromarray(image).resize((target_size, target_size), Image.Resampling.BILINEAR))
    return image


def render_camera(env: Any, camera_name: str) -> np.ndarray:
    """Render one named camera without changing the environment state."""
    base_env = env.unwrapped
    try:
        camera_id = int(base_env.model.camera(camera_name).id)
    except KeyError as error:
        available = [base_env.model.camera(index).name for index in range(base_env.model.ncam)]
        raise ValueError(f"Unknown camera {camera_name!r}; available: {available}") from error
    base_env.mujoco_renderer.camera_id = camera_id
    return render_rgb(base_env)


def render_visual_observation(env: Any) -> dict[str, np.ndarray]:
    """Render the stable three-camera visual-policy schema."""
    return {
        schema_key: render_camera(env, camera_name)
        for schema_key, camera_name in VISUAL_CAMERA_SCHEMA.items()
    }


def _rotation_matrix_to_xyz_euler(rotation: np.ndarray) -> np.ndarray:
    """Return intrinsic XYZ (roll, pitch, yaw) angles from a rotation matrix."""
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if sy > 1e-6:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def robot_only_state(env: Any, wrist_wrench: np.ndarray) -> np.ndarray:
    """Build a task-invariant 19-D robot-only proprioceptive observation.

    Layout: world-frame TCP xyz + intrinsic XYZ Euler, world-frame TCP linear
    velocity + angular velocity, wrist-frame wrench, and gripper opening in
    metres. No object or goal qpos is read.
    """
    base_env = env.unwrapped
    site_id = int(base_env.model.site("endEffector").id)
    position = np.asarray(base_env.data.site_xpos[site_id], dtype=np.float32)
    rotation = np.asarray(base_env.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    pose = np.concatenate([position, _rotation_matrix_to_xyz_euler(rotation)])

    spatial_velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        base_env.model,
        base_env.data,
        mujoco.mjtObj.mjOBJ_SITE,
        site_id,
        spatial_velocity,
        0,
    )
    # MuJoCo cvel order is angular, linear; policy convention is linear, angular.
    velocity = np.concatenate([spatial_velocity[3:], spatial_velocity[:3]]).astype(np.float32)

    right = np.asarray(base_env.data.site_xpos[base_env.model.site("rightEndEffector").id])
    left = np.asarray(base_env.data.site_xpos[base_env.model.site("leftEndEffector").id])
    gripper_opening = np.asarray([np.linalg.norm(right - left)], dtype=np.float32)
    state = np.concatenate(
        [pose, velocity, np.asarray(wrist_wrench, dtype=np.float32), gripper_opening]
    ).astype(np.float32)
    if state.shape != (ROBOT_STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid robot-only state {state.shape}: {state}")
    return state


def success_from_info(info: dict[str, Any]) -> bool:
    return bool(float(info.get("success", 0.0)) > 0.5)


def make_scripted_policy(task: str) -> Any:
    return get_task_spec(task).policy_class()
