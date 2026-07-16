#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export TRAINER_N_GPUS_PER_NODE="${VOPD_GPUS_PER_NODE:-${TRAINER_N_GPUS_PER_NODE:-8}}"
export TRAINER_NNODES="${VOPD_NNODES:-${TRAINER_NNODES:-1}}"

# shellcheck source=ascend_env.sh
source "$PROJECT_ROOT/scripts/ascend_env.sh"

TRAIN_FILE="${VOPD_TRAIN_FILE:-${VOPD_DATA_DIR:-${PROJECT_ROOT}/data}/train.parquet}"
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Training data is missing: $TRAIN_FILE" >&2
    echo "Set VOPD_TRAIN_FILE to a prepared train.parquet file." >&2
    exit 1
fi

export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
export PYTHON_BIN

if [[ "${VOPD_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/check_ascend_env.py" \
        --min-npus "$TRAINER_N_GPUS_PER_NODE"
fi

exec bash "$PROJECT_ROOT/scripts/run_vision_opd.sh" "$@"
