"""Manual labeling pipeline: preprocess -> reward assignment -> LeRobot conversion."""

from data import convert_data, preprocess
from data.common import BASE_DIR, get_task, save_pickle


def run_pipeline(task_name: str) -> None:
    """Run full manual labeling pipeline for a task."""
    task = get_task(task_name)

    if task.target_positions is None:
        print(f"[SKIP] Task '{task_name}' has no target_positions for manual labeling")
        return

    try:
        success_data = preprocess.load_data(task.success_path)
        fail_data = preprocess.load_data(task.fail_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    labeled_success = preprocess.modify_data(success_data, task_name)
    labeled_fail = preprocess.modify_data(fail_data, task_name)

    camera_keys = task.camera_keys
    if labeled_success:
        sample_obs = labeled_success[0]["observations"]
        detected = [k for k in camera_keys if k in sample_obs]
        if detected:
            camera_keys = detected

    lerobot_success = convert_data.convert_to_lerobot_format(labeled_success, camera_keys)
    lerobot_fail = convert_data.convert_to_lerobot_format(labeled_fail, camera_keys)

    output_dir = BASE_DIR / task_name / "manual_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_pickle(lerobot_success, output_dir / "success_lerobot.pkl")
    save_pickle(lerobot_fail, output_dir / "fail_lerobot.pkl")
    print(f"[OK] Pipeline done for {task_name}")
