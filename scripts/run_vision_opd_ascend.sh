#!/usr/bin/env bash

# Huawei CANN/NNAL environment scripts are not nounset-safe.
set +u
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOPD_CONFIG_FILE="${VOPD_CONFIG_FILE:-${PROJECT_ROOT}/vision_opd_ascend.env}"
if [[ ! -f "$VOPD_CONFIG_FILE" ]]; then
    echo "Vision-OPD config file does not exist: $VOPD_CONFIG_FILE" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$VOPD_CONFIG_FILE"
set +a

if [[ -z "${PYTHON_BIN:-}" && "${VOPD_RUNTIME_PROFILE_READY:-0}" != "1" ]]; then
    # Direct invocations still resolve the same ABI-specific environment as
    # the public lifecycle; normal start-script invocations already exported it.
    # shellcheck source=resolve_ascend_runtime.sh
    source "$PROJECT_ROOT/scripts/resolve_ascend_runtime.sh"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ "$VOPD_VENV_DIR" == /* ]]; then
        PYTHON_BIN="${VOPD_VENV_DIR}/bin/python"
    else
        PYTHON_BIN="${PROJECT_ROOT}/${VOPD_VENV_DIR}/bin/python"
    fi
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Vision-OPD runtime Python does not exist: $PYTHON_BIN" >&2
    echo "Run bash scripts/start_vision_opd_ascend.sh first." >&2
    exit 2
fi
export PATH="$(dirname "$PYTHON_BIN"):${PATH}"

export TRAINER_N_GPUS_PER_NODE="${VOPD_GPUS_PER_NODE:-${TRAINER_N_GPUS_PER_NODE:-8}}"
export TRAINER_NNODES="${VOPD_NNODES:-${TRAINER_NNODES:-1}}"

if [[ "${VOPD_ASCEND_ENV_READY:-0}" != "1" ]]; then
    # shellcheck source=ascend_env.sh
    source "$PROJECT_ROOT/scripts/ascend_env.sh"
fi

# prompt.txt changes blocking mode from 0 to 1 immediately before training.
export ASCEND_LAUNCH_BLOCKING=1
export TARGET_DEVICE=ascend

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

exec bash "$PROJECT_ROOT/scripts/run_vision_opd.sh" "$@"
