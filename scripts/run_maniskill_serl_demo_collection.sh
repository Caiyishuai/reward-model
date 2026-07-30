#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANISKILL_ROOT="${MANISKILL_ROOT:-$(cd "$ROOT/../maniskill-ws" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$MANISKILL_ROOT/auto_research/venv_maniskill/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data/maniskill_serl_demos}"
EPISODES="${EPISODES:-20}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
SEED="${SEED:-42}"

: "${PUSHCUBE_CKPT:?Set PUSHCUBE_CKPT to a 7-D pd_ee_delta_pose PPO checkpoint}"
: "${POKECUBE_CKPT:?Set POKECUBE_CKPT to a 7-D pd_ee_delta_pose PPO checkpoint}"
: "${PLACESPHERE_CKPT:?Set PLACESPHERE_CKPT to a 7-D pd_ee_delta_pose PPO checkpoint}"
: "${STACKCUBE_CKPT:?Set STACKCUBE_CKPT to a 7-D pd_ee_delta_pose PPO checkpoint}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

collect() {
  local env_id="$1"
  local checkpoint="$2"
  local slug="$3"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found for $env_id: $checkpoint" >&2
    exit 1
  fi
  "$PYTHON_BIN" "$ROOT/scripts/collect_maniskill_serl_demos.py" \
    --env-id "$env_id" \
    --checkpoint "$checkpoint" \
    --output "$OUTPUT_DIR/${slug}_normalized_dense_20.pkl" \
    --episodes "$EPISODES" \
    --max-attempts "$MAX_ATTEMPTS" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --seed "$SEED"
}

collect "PushCube-v1" "$PUSHCUBE_CKPT" "pushcube"
collect "PokeCube-v1" "$POKECUBE_CKPT" "pokecube"
collect "PlaceSphere-v1" "$PLACESPHERE_CKPT" "placesphere"
collect "StackCube-v1" "$STACKCUBE_CKPT" "stackcube"

"$PYTHON_BIN" "$ROOT/scripts/validate_maniskill_serl_demos.py" \
  "$OUTPUT_DIR/pushcube_normalized_dense_20.pkl" \
  "$OUTPUT_DIR/pokecube_normalized_dense_20.pkl" \
  "$OUTPUT_DIR/placesphere_normalized_dense_20.pkl" \
  "$OUTPUT_DIR/stackcube_normalized_dense_20.pkl" \
  --expected-episodes "$EPISODES"

echo "[OK] four-task normalized_dense SERL demonstrations are ready in $OUTPUT_DIR"
