#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TASKS=(
  mw_button_press
  mw_window_open
  mw_reach_wall
  mw_plate_slide
  mw_push
  mw_coffee_push
  mw_stick_push
  mw_pick_place
)

"${PYTHON_BIN}" scripts/validate_metaworld_rm_data.py --tasks all --minimum-episodes 20
"${PYTHON_BIN}" -m label.label --tasks "${TASKS[@]}" --method auto
"${PYTHON_BIN}" scripts/export_metaworld_serl_data.py \
  --tasks all \
  --reward-modes auto dense sparse
"${PYTHON_BIN}" scripts/evaluate_metaworld_auto_labels.py \
  --tasks all \
  --output eval_results/metaworld_auto_label_quality.json

for task in "${TASKS[@]}"; do
  if [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
    "${PYTHON_BIN}" train.py \
      --task_name "${task}" \
      --prefix auto \
      --epochs "${EPOCHS}" \
      --num_workers "${NUM_WORKERS}" \
      --use_gradient_checkpointing
  else
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${NPROC_PER_NODE}" \
      train.py \
      --task_name "${task}" \
      --prefix auto \
      --epochs "${EPOCHS}" \
      --num_workers "${NUM_WORKERS}" \
      --use_gradient_checkpointing
  fi
done

"${PYTHON_BIN}" scripts/evaluate_metaworld_rm_quality.py \
  --tasks all \
  --device cuda \
  --gamma 0.99 \
  --output eval_results/metaworld_rm_quality.json
