#!/usr/bin/env bash
set -euo pipefail

# Full four-task Rsync reward-model job. Run from the Rsync repository root on
# Linux/NVIDIA after collect_rm_episodes.py has produced 20 success + 20 fail
# episodes under data/{pushcube,pokecube,placesphere,stackcube}.

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TASKS=(pushcube pokecube placesphere stackcube)

for task in "${TASKS[@]}"; do
  success_path="data/${task}/success_raw.pkl"
  fail_path="data/${task}/fail_raw.pkl"
  if [[ ! -f "${success_path}" || ! -f "${fail_path}" ]]; then
    echo "Missing ${success_path} or ${fail_path}; collect the 20+20 dataset first." >&2
    exit 2
  fi
done

"${PYTHON_BIN}" scripts/validate_maniskill_rm_data.py --tasks all --minimum-episodes 20

"${PYTHON_BIN}" -m label.label --tasks "${TASKS[@]}" --method auto

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

"${PYTHON_BIN}" scripts/evaluate_maniskill_rm_quality.py \
  --tasks all \
  --device cuda \
  --output eval_results/maniskill_rm_quality.json
