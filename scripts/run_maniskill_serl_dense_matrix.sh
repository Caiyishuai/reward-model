#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERL_ROOT="${SERL_ROOT:-$(cd "$ROOT/../serl" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$SERL_ROOT/auto_research/venv_serl/bin/python}"
TRAINER="${SERL_TRAINER:-$SERL_ROOT/examples/async_drq_sim/async_drq_sim_maniskill_ty.py}"
DEMO_DIR="${DEMO_DIR:-$ROOT/data/maniskill_serl_demos}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs/maniskill_serl_dense}"

ACTOR_STEPS="${ACTOR_STEPS:-1000000}"
LEARNER_UPDATES="${LEARNER_UPDATES:-125000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
RANDOM_STEPS="${RANDOM_STEPS:-1000}"
TRAINING_STARTS="${TRAINING_STARTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEMO_BATCH_SIZE="${DEMO_BATCH_SIZE:-64}"
EVAL_PERIOD="${EVAL_PERIOD:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-10000}"
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-24}"
SEED="${SEED:-42}"
TAU_MODES="${TAU_MODES:-fixed adaptive}"
BASE_PORT="${BASE_PORT:-5488}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "SERL Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TRAINER" ]]; then
  echo "SERL trainer not found: $TRAINER" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "import jax, torch, mani_skill.envs, serl_launcher" >/dev/null 2>&1; then
  echo "PYTHON_BIN must contain JAX, CUDA PyTorch, ManiSkill, and the installed serl_launcher package." >&2
  echo "Use one combined GPU environment for the online actor/learner run." >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required (Ubuntu: apt-get install coreutils)." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

declare -a TASKS=(
  "PushCube-v1:pushcube"
  "PokeCube-v1:pokecube"
  "PlaceSphere-v1:placesphere"
  "StackCube-v1:stackcube"
)

run_index=0
for task_spec in "${TASKS[@]}"; do
  env_id="${task_spec%%:*}"
  slug="${task_spec##*:}"
  demo="$DEMO_DIR/${slug}_normalized_dense_20.pkl"
  if [[ ! -f "$demo" ]]; then
    echo "Missing validated demo dataset: $demo" >&2
    exit 1
  fi

  for tau_mode in $TAU_MODES; do
    adaptive="False"
    if [[ "$tau_mode" == "adaptive" ]]; then
      adaptive="True"
    elif [[ "$tau_mode" != "fixed" ]]; then
      echo "Unsupported TAU_MODES entry: $tau_mode" >&2
      exit 1
    fi

    server_port=$((BASE_PORT + run_index * 2))
    broadcast_port=$((server_port + 1))
    run_dir="$OUTPUT_ROOT/${slug}__normalized_dense__${tau_mode}__seed${SEED}"
    checkpoint_dir="$run_dir/checkpoints"
    eval_video_dir="$run_dir/eval_videos"
    mkdir -p "$checkpoint_dir" "$eval_video_dir"
    echo "[RUN] env=$env_id reward=normalized_dense tau=$tau_mode seed=$SEED"

    common_args=(
      --env "$env_id"
      --obs_mode rgb+state
      --control_mode pd_ee_delta_pose
      --robot_uids panda_wristcam
      --reward_mode normalized_dense
      --potential_reward_shaping=False
      --max_episode_steps "$MAX_EPISODE_STEPS"
      --encoder_type resnet-pretrained
      --seed "$SEED"
      --server_port "$server_port"
      --broadcast_port "$broadcast_port"
      --adaptive_tau_enabled="$adaptive"
      --critic_loss_threshold 0.05
      --tau_min 0.001
      --tau_max 0.05
      --tau_adjust_factor 1.1
      --tau_adjust_tolerance 0.2
      --debug=True
    )

    timeout "${TIME_LIMIT_HOURS}h" "$PYTHON_BIN" \
      "$ROOT/scripts/run_serl_trainer_sanitized.py" \
      --trainer "$TRAINER" \
      "${common_args[@]}" \
      --learner \
      --exp_name "${slug}_normalized_dense_${tau_mode}_seed${SEED}" \
      --max_steps "$LEARNER_UPDATES" \
      --training_starts "$TRAINING_STARTS" \
      --batch_size "$BATCH_SIZE" \
      --demo_batch_size "$DEMO_BATCH_SIZE" \
      --demo_path "$demo" \
      --checkpoint_period "$CHECKPOINT_PERIOD" \
      --checkpoint_path "$checkpoint_dir" \
      >"$run_dir/learner.log" 2>&1 &
    learner_pid=$!

    cleanup() {
      kill "$learner_pid" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    sleep 10
    if ! kill -0 "$learner_pid" 2>/dev/null; then
      echo "Learner exited before actor startup; inspect $run_dir/learner.log" >&2
      wait "$learner_pid" || true
      exit 1
    fi

    set +e
    timeout "${TIME_LIMIT_HOURS}h" "$PYTHON_BIN" \
      "$ROOT/scripts/run_serl_trainer_sanitized.py" \
      --trainer "$TRAINER" \
      "${common_args[@]}" \
      --actor \
      --ip localhost \
      --max_steps "$ACTOR_STEPS" \
      --random_steps "$RANDOM_STEPS" \
      --eval_period "$EVAL_PERIOD" \
      --eval_n_trajs "$EVAL_EPISODES" \
      --save_eval_video=True \
      --eval_video_dir "$eval_video_dir" \
      >"$run_dir/actor.log" 2>&1
    actor_rc=$?
    wait "$learner_pid"
    learner_rc=$?
    set -e
    trap - EXIT INT TERM

    if [[ "$actor_rc" -ne 0 || "$learner_rc" -ne 0 ]]; then
      echo "[FAIL] actor=$actor_rc learner=$learner_rc logs=$run_dir" >&2
      exit 1
    fi
    echo "[OK] $env_id $tau_mode logs=$run_dir"
    run_index=$((run_index + 1))
  done
done

"$PYTHON_BIN" "$ROOT/scripts/summarize_maniskill_serl_results.py" \
  --runs-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.json"
