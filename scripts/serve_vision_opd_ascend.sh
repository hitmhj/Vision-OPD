#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/serve_vision_opd_ascend.sh <merged_model_path> [vllm arguments...]" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="$1"
shift

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-$TENSOR_PARALLEL_SIZE}"

# shellcheck source=ascend_env.sh
source "$PROJECT_ROOT/scripts/ascend_env.sh"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Vision-OPD-4B}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/check_ascend_env.py" \
    --min-npus "$TENSOR_PARALLEL_SIZE"

exec "$VLLM_BIN" serve "$MODEL_PATH" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    "$@"
