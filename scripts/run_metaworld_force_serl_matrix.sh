#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERL_ROOT="${SERL_ROOT:-$(cd "$ROOT/../serl" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$SERL_ROOT/auto_research/venv_serl/bin/python}"
MAX_STEPS="${MAX_STEPS:-1000000}"
SEEDS="${SEEDS:-0 1 2}"
REWARD_MODES="${REWARD_MODES:-sparse}"
TAU_MODES="${TAU_MODES:-fixed adaptive}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
RANDOM_STEPS="${RANDOM_STEPS:-5000}"
TRAINING_STARTS="${TRAINING_STARTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
UTD_RATIO="${UTD_RATIO:-4}"
EVAL_PERIOD="${EVAL_PERIOD:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
SAVE_PERIOD="${SAVE_PERIOD:-100000}"
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs/metaworld_force_serl}"
mkdir -p "$OUTPUT_ROOT"

TASKS=(
  button-press
  window-open
  reach-wall
  plate-slide
  push
  coffee-push
  stick-push
  pick-place
)

task_data_name() {
  echo "mw_${1//-/_}"
}

for seed in $SEEDS; do
  for task in "${TASKS[@]}"; do
    task_data="$(task_data_name "$task")"
    for reward_mode in $REWARD_MODES; do
      demo="$ROOT/data/$task_data/serl_${reward_mode}.pkl"
      if [[ ! -f "$demo" ]]; then
        echo "Missing force-aware demo: $demo" >&2
        echo "Run scripts/collect_metaworld_rm_data.py first." >&2
        exit 2
      fi
      for tau_mode in $TAU_MODES; do
        adaptive_arg=""
        if [[ "$tau_mode" == "adaptive" ]]; then
          adaptive_arg="--adaptive-tau"
        elif [[ "$tau_mode" != "fixed" ]]; then
          echo "Unknown tau mode: $tau_mode" >&2
          exit 2
        fi

        output="$OUTPUT_ROOT/${task}__${reward_mode}__${tau_mode}__seed${seed}"
        echo "[RUN] task=$task reward=$reward_mode tau=$tau_mode seed=$seed"
        "$PYTHON_BIN" "$ROOT/scripts/train_serl_metaworld_force.py" \
          --serl-root "$SERL_ROOT" \
          --task "$task" \
          --reward-mode "$reward_mode" \
          --demo-path "$demo" \
          --seed "$seed" \
          --max-steps "$MAX_STEPS" \
          --max-episode-steps "$MAX_EPISODE_STEPS" \
          --random-steps "$RANDOM_STEPS" \
          --training-starts "$TRAINING_STARTS" \
          --batch-size "$BATCH_SIZE" \
          --utd-ratio "$UTD_RATIO" \
          --time-limit-min "$TIME_LIMIT_MIN" \
          --eval-period "$EVAL_PERIOD" \
          --eval-episodes "$EVAL_EPISODES" \
          --save-period "$SAVE_PERIOD" \
          --critic-loss-threshold 0.05 \
          --tau-min 0.001 \
          --tau-max 0.05 \
          --tau-adjust-factor 1.1 \
          --tau-adjust-tolerance 0.2 \
          --output-dir "$output" \
          ${adaptive_arg:+$adaptive_arg} \
          2>&1 | tee "${output}.log"
      done
    done
  done
done
