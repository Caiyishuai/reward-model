#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERL_ROOT="${SERL_ROOT:-$(cd "$ROOT/../serl" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$SERL_ROOT/auto_research/venv_serl/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs/metaworld_visual_drq}"
MAX_STEPS="${MAX_STEPS:-1000000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
RANDOM_STEPS="${RANDOM_STEPS:-5000}"
TRAINING_STARTS="${TRAINING_STARTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
UTD_RATIO="${UTD_RATIO:-4}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-100000}"
ENCODER_TYPE="${ENCODER_TYPE:-resnet-pretrained}"
FORCE_FILTER="${FORCE_FILTER:-ema}"
FORCE_FUSION="${FORCE_FUSION:-learned_gate}"
TAU_MODES="${TAU_MODES:-fixed adaptive}"
SEEDS="${SEEDS:-0 1 2}"
LOG_PERIOD="${LOG_PERIOD:-1000}"
EVAL_PERIOD="${EVAL_PERIOD:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
SAVE_PERIOD="${SAVE_PERIOD:-100000}"

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

mkdir -p "$OUTPUT_ROOT"
for seed in $SEEDS; do
  for task in "${TASKS[@]}"; do
    task_data="mw_${task//-/_}"
    demo="$DATA_ROOT/$task_data/visual_drq/success_demos.pkl.gz"
    if [[ ! -f "$demo" ]]; then
      echo "Missing visual demo buffer: $demo" >&2
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
      output="$OUTPUT_ROOT/${task}__${tau_mode}__${FORCE_FILTER}__${FORCE_FUSION}__seed${seed}"
      echo "[RUN] task=$task tau=$tau_mode filter=$FORCE_FILTER fusion=$FORCE_FUSION seed=$seed"
      "$PYTHON_BIN" "$ROOT/scripts/train_serl_metaworld_visual.py" \
        --serl-root "$SERL_ROOT" \
        --task "$task" \
        --demo-path "$demo" \
        --output-dir "$output" \
        --encoder-type "$ENCODER_TYPE" \
        --force-filter "$FORCE_FILTER" \
        --force-fusion "$FORCE_FUSION" \
        --max-steps "$MAX_STEPS" \
        --max-episode-steps "$MAX_EPISODE_STEPS" \
        --random-steps "$RANDOM_STEPS" \
        --training-starts "$TRAINING_STARTS" \
        --batch-size "$BATCH_SIZE" \
        --utd-ratio "$UTD_RATIO" \
        --buffer-capacity "$BUFFER_CAPACITY" \
        --seed "$seed" \
        --log-period "$LOG_PERIOD" \
        --eval-period "$EVAL_PERIOD" \
        --eval-episodes "$EVAL_EPISODES" \
        --save-period "$SAVE_PERIOD" \
        ${adaptive_arg:+$adaptive_arg} \
        2>&1 | tee "${output}.log"
    done
  done
done
