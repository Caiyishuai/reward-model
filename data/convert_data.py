"""Convert raw transition lists to LeRobot-style numpy dictionary format."""

import logging

import numpy as np

from data.common import BASE_DIR, load_pickle, save_pickle


def convert_to_lerobot_format(data: list[dict], camera_keys: list[str]) -> dict[str, np.ndarray]:
    """Convert step dictionaries to LeRobot-like flat arrays.

    Output keys: observation.state, observation.images.*, action,
    episode_index, frame_index, next.reward, next.done, next.success, index.
    """
    converted: dict[str, list] = {
        "observation.state": [],
        "action": [],
        "episode_index": [],
        "frame_index": [],
        "next.reward": [],
        "next.done": [],
        "next.success": [],
        "index": [],
    }
    for k in camera_keys:
        converted[f"observation.images.{k}"] = []

    episode_idx = 0
    frame_idx = 0

    for i, step in enumerate(data):
        obs = step["observations"]

        state = obs["state"]
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        converted["observation.state"].append(state)

        for key in camera_keys:
            img = obs[key]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            converted[f"observation.images.{key}"].append(img)

        converted["action"].append(step["actions"])
        converted["episode_index"].append(episode_idx)
        converted["frame_index"].append(frame_idx)
        converted["index"].append(i)
        converted["next.reward"].append(step["rewards"])
        converted["next.done"].append(step["dones"])
        converted["next.success"].append(step.get("infos", {}).get("succeed", False))

        done = step["dones"]
        if hasattr(done, "item"):
            done = done.item() == 1
        elif isinstance(done, (int, float)):
            done = done == 1
        if done:
            episode_idx += 1
            frame_idx = 0
        else:
            frame_idx += 1

    final = {}
    for k, v in converted.items():
        try:
            final[k] = np.stack(v)
        except ValueError as e:
            logging.warning("Cannot stack '%s': %s, keeping as list", k, e)
            final[k] = v

    return final


if __name__ == "__main__":
    task_name = "button"
    camera_keys = ["wrist_1", "wrist_2"]
    for input_file in ["success_data.pkl", "fail_data.pkl"]:
        input_path = BASE_DIR / task_name / input_file
        output_path = BASE_DIR / task_name / f"{input_file.replace('.pkl', '_lerobot_format.pkl')}"
        data = load_pickle(input_path)
        new_data = convert_to_lerobot_format(data, camera_keys)
        save_pickle(new_data, output_path)
        print(f"Saved: {output_path}")
