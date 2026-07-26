#!/bin/bash
set -euo pipefail

# ==========================================
# Configuration
# ==========================================
TASK_LIST=(button iphone_insert plug_insert pickup usb key)
PREFIX=${PREFIX:-auto}        # auto | manual (env override or default)
EPOCHS=${EPOCHS:-100}
NUM_WORKERS=${NUM_WORKERS:-12}
PATIENCE=${PATIENCE:-15}
GRAD_ACCUM=${GRAD_ACCUM:-1}

# GPU selection: auto-detect available GPUs or use explicit list
if [ -n "${CUDA_LIST:-}" ]; then
    IFS=',' read -ra GPU_IDS <<< "$CUDA_LIST"
else
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[ERROR] nvidia-smi not found. Install NVIDIA drivers or set CUDA_LIST explicitly."
        exit 1
    fi
    GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' '))
    if [ ${#GPU_IDS[@]} -eq 0 ]; then
        echo "[ERROR] No GPUs detected. Set CUDA_LIST=0,1,2 explicitly."
        exit 1
    fi
fi

NUM_GPUS=${#GPU_IDS[@]}
LOG_ROOT="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ==========================================
# Filter tasks (optional CLI args)
# ==========================================
if [ $# -gt 0 ]; then
    TASK_LIST=("$@")
fi

# ==========================================
# Launch
# ==========================================
echo "============================================"
echo "  Reward Model Training Launcher"
echo "============================================"
echo "  Tasks:     ${TASK_LIST[*]}"
echo "  Prefix:    ${PREFIX}"
echo "  GPUs:      ${GPU_IDS[*]} (${NUM_GPUS} total)"
echo "  Epochs:    ${EPOCHS}"
echo "  Workers:   ${NUM_WORKERS}"
echo "  Patience:  ${PATIENCE}"
echo "  Timestamp: ${TIMESTAMP}"
echo "============================================"

if [ ${#TASK_LIST[@]} -gt "$NUM_GPUS" ]; then
    echo "[WARN] ${#TASK_LIST[@]} tasks > ${NUM_GPUS} GPUs. Multiple tasks will share GPUs."
fi

PIDS=()

cleanup() {
    echo "[CLEANUP] Terminating background training processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

for ((i=0; i<${#TASK_LIST[@]}; i++)); do
    TASK=${TASK_LIST[$i]}
    GPU_IDX=$((i % NUM_GPUS))
    GPU_ID=${GPU_IDS[$GPU_IDX]}

    LOG_DIR="${LOG_ROOT}/${PREFIX}_${TASK}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"

    ARGS=()
    [ -n "${USE_FILM:-}" ]          && ARGS+=(--use_film)
    [ -n "${USE_EMA_OFF:-}" ]       && ARGS+=(--no_ema)
    [ -n "${USE_PATCH_POOLING:-}" ] && ARGS+=(--use_patch_pooling)

    echo "[LAUNCH] task=${TASK} gpu=${GPU_ID} log=${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
        --task_name "$TASK" \
        --epochs "$EPOCHS" \
        --num_workers "$NUM_WORKERS" \
        --prefix "$PREFIX" \
        --patience "$PATIENCE" \
        --grad_accum "$GRAD_ACCUM" \
        "${ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    sleep 1
done

# ==========================================
# Wait and report
# ==========================================
echo ""
echo "[INFO] Waiting for ${#PIDS[@]} tasks to complete..."

FAILED=0
for ((i=0; i<${#PIDS[@]}; i++)); do
    TASK=${TASK_LIST[$i]}
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "[DONE] ${TASK} (pid=${PID}) succeeded"
    else
        echo "[FAIL] ${TASK} (pid=${PID}) exited with error"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "[SUCCESS] All ${#TASK_LIST[@]} tasks completed successfully."
else
    echo "[WARNING] ${FAILED}/${#TASK_LIST[@]} tasks failed. Check logs in ${LOG_ROOT}/."
    exit 1
fi
