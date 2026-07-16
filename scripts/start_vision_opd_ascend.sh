#!/usr/bin/env bash

# The single public entry for the complete Vision-OPD Ascend lifecycle:
# configuration -> optional Python setup -> optional data preparation ->
# preflight -> training/checkpointing -> optional HuggingFace model merge.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VOPD_CONFIG_FILE="${VOPD_CONFIG_FILE:-${PROJECT_ROOT}/vision_opd_ascend.env}"

# Export every value loaded from the config file. Entries in the checked-in
# config use ${VAR:-default}, so variables injected by the platform win.
if [[ ! -f "$VOPD_CONFIG_FILE" ]]; then
    echo "Vision-OPD config file does not exist: $VOPD_CONFIG_FILE" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$VOPD_CONFIG_FILE"
set +a

# ModelArts can invoke a rank-table boot file once per NPU. Only global rank 0
# owns this orchestration lifecycle; the verl driver later creates Ray workers.
if [[ "${VOPD_SINGLE_DRIVER_GUARD:-1}" == "1" ]]; then
    if [[ -n "${RANK_ID:-}" && "${RANK_ID}" != "0" ]]; then
        echo "Vision-OPD lifecycle is owned by RANK_ID=0; rank ${RANK_ID} exits normally."
        exit 0
    fi
    if [[ -z "${RANK_ID:-}" && -n "${ASCEND_DEVICE_ID:-}" && "${ASCEND_DEVICE_ID}" != "0" ]]; then
        echo "Vision-OPD lifecycle is owned by ASCEND_DEVICE_ID=0; device ${ASCEND_DEVICE_ID} exits normally."
        exit 0
    fi
fi

if [[ "${MA_RUN_METHOD:-}" == "torchrun" ]]; then
    echo "MA_RUN_METHOD=torchrun is incompatible with the single verl/Ray driver." >&2
    echo "Use the ModelArts rank-table/custom boot mode." >&2
    exit 2
fi

export VOPD_GPUS_PER_NODE="${VOPD_GPUS_PER_NODE:-${MA_NUM_GPUS:-${RANK_SIZE:-8}}}"
export VOPD_NNODES="${VOPD_NNODES:-${MA_NUM_HOSTS:-1}}"
export TRAINER_N_GPUS_PER_NODE="$VOPD_GPUS_PER_NODE"
export TRAINER_NNODES="$VOPD_NNODES"

if [[ "$VOPD_NNODES" != "1" ]]; then
    echo "The unified entry currently supports one node; got VOPD_NNODES=$VOPD_NNODES." >&2
    echo "Multi-node training requires a ModelArts Ray head/worker bootstrap." >&2
    exit 2
fi

export VOPD_LOG_DIR="${VOPD_LOG_DIR:-${MA_LOG_DIR:-${PROJECT_ROOT}/logs}}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${VOPD_LOG_DIR}/tensorboard}"
mkdir -p \
    "$VOPD_OUTPUT_DIR" \
    "$VOPD_ROLLOUT_DIR" \
    "$VOPD_LOG_DIR" \
    "$TENSORBOARD_DIR" \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$VLLM_CACHE_ROOT" \
    "$TORCH_HOME"

echo "[1/6] Loading the Ascend runtime environment..."
# shellcheck source=ascend_env.sh
source "$PROJECT_ROOT/scripts/ascend_env.sh"

echo "[2/6] Resolving the Python environment (mode: $VOPD_INSTALL_MODE)..."
case "$VOPD_INSTALL_MODE" in
    always)
        bash "$PROJECT_ROOT/scripts/install_ascend.sh"
        ;;
    auto)
        if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/check_ascend_env.py" \
            --dependencies-only >/dev/null 2>&1; then
            echo "The Python environment is incomplete or version-mismatched; installing all pinned dependencies."
            bash "$PROJECT_ROOT/scripts/install_ascend.sh"
        else
            echo "Reusing the complete pinned Vision-OPD environment."
        fi
        ;;
    never)
        echo "Dependency installation disabled by VOPD_INSTALL_MODE=never."
        ;;
    *)
        echo "VOPD_INSTALL_MODE must be auto, always, or never; got $VOPD_INSTALL_MODE" >&2
        exit 2
        ;;
esac

echo "[3/6] Resolving the training dataset..."
if [[ ! -f "$VOPD_TRAIN_FILE" ]]; then
    if [[ "$VOPD_PREPARE_DATA_IF_MISSING" == "1" ]]; then
        "$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_data.py" --data-dir "$VOPD_DATA_DIR"
    else
        echo "Training data is missing: $VOPD_TRAIN_FILE" >&2
        echo "Mount it and set VOPD_TRAIN_FILE, or set VOPD_PREPARE_DATA_IF_MISSING=1." >&2
        exit 1
    fi
fi
if [[ ! -f "$VOPD_TRAIN_FILE" ]]; then
    echo "Dataset preparation did not create: $VOPD_TRAIN_FILE" >&2
    exit 1
fi

echo "[4/6] Running Ascend and configuration preflight checks..."
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/check_ascend_env.py" --min-npus "$VOPD_GPUS_PER_NODE"

echo "Vision-OPD resolved configuration"
echo "  config_file:     $VOPD_CONFIG_FILE"
echo "  job_id:          ${MA_VJ_NAME:-${JOB_ID:-unknown}}"
echo "  model:           $VOPD_MODEL_PATH"
echo "  train_file:      $VOPD_TRAIN_FILE"
echo "  cache_dir:       $VOPD_CACHE_DIR"
echo "  output_dir:      $VOPD_OUTPUT_DIR"
echo "  npu_per_node:    $VOPD_GPUS_PER_NODE"
echo "  learning_rate:   $VOPD_LR"
echo "  train_batch:     $VOPD_TRAIN_BATCH_SIZE"
echo "  rollout_n:       $VOPD_ROLLOUT_N"
echo "  training_steps:  $VOPD_TOTAL_TRAINING_STEPS"
echo "  resume_mode:     $VOPD_RESUME_MODE"
echo "  prompt/response: $VOPD_MAX_PROMPT_LENGTH/$VOPD_MAX_RESPONSE_LENGTH"
echo "  save_frequency:  $VOPD_SAVE_FREQ"

echo "[5/6] Starting Vision-OPD training and checkpoint lifecycle..."
export VOPD_SKIP_PREFLIGHT=1
bash "$PROJECT_ROOT/scripts/run_vision_opd_ascend.sh" "$@"

echo "[6/6] Finalizing saved model artifacts..."
if [[ "$VOPD_AUTO_MERGE" == "1" ]]; then
    mapfile -t _vopd_checkpoints < <(
        find "$VOPD_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print | sort -V
    )
    if [[ ${#_vopd_checkpoints[@]} -eq 0 ]]; then
        echo "Training completed but no global_step_* checkpoint exists in $VOPD_OUTPUT_DIR" >&2
        echo "The trainer should always save its final step; inspect the training save logs." >&2
        exit 1
    fi
    _vopd_latest_checkpoint="${_vopd_checkpoints[-1]}"
    echo "Merging latest checkpoint: $_vopd_latest_checkpoint"
    TARGET_DIR="$VOPD_MERGED_MODEL_DIR" PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/merge_checkpoint.sh" "$_vopd_latest_checkpoint"
    echo "Merged HuggingFace model: $VOPD_MERGED_MODEL_DIR"
else
    echo "Checkpoint merge disabled by VOPD_AUTO_MERGE=0."
fi

echo "Vision-OPD Ascend lifecycle completed successfully."
