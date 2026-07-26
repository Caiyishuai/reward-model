"""Shared MetaWorld task registry and environment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import metaworld
import metaworld.policies as policies
import numpy as np
from PIL import Image


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
    "reach-wall": MetaWorldTaskSpec(
        "reach-wall", "reach-wall-v3", "mw_reach_wall", policies.SawyerReachWallV3Policy
    ),
    "plate-slide": MetaWorldTaskSpec(
        "plate-slide", "plate-slide-v3", "mw_plate_slide", policies.SawyerPlateSlideV3Policy
    ),
    "push": MetaWorldTaskSpec("push", "push-v3", "mw_push", policies.SawyerPushV3Policy),
    "coffee-push": MetaWorldTaskSpec(
        "coffee-push", "coffee-push-v3", "mw_coffee_push", policies.SawyerCoffeePushV3Policy
    ),
    "stick-push": MetaWorldTaskSpec(
        "stick-push", "stick-push-v3", "mw_stick_push", policies.SawyerStickPushV3Policy
    ),
    "pick-place": MetaWorldTaskSpec(
        "pick-place", "pick-place-v3", "mw_pick_place", policies.SawyerPickPlaceV3Policy
    ),
}


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


def success_from_info(info: dict[str, Any]) -> bool:
    return bool(float(info.get("success", 0.0)) > 0.5)


def make_scripted_policy(task: str) -> Any:
    return get_task_spec(task).policy_class()
