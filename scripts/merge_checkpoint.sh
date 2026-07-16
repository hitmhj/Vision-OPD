#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_BASE_DIR="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/"
BASE_DIR="${BASE_DIR:-${1:-${DEFAULT_BASE_DIR}}}"
BASE_DIR="${BASE_DIR%/}"
ACTOR_DIR="${BASE_DIR}/actor"
TARGET_DIR="${TARGET_DIR:-${2:-${BASE_DIR}}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "${ACTOR_DIR}" ]; then
  echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
  exit 1
fi

echo "Merging ${ACTOR_DIR} -> ${TARGET_DIR}"

# Preserve actor shards. When the historical in-place target is requested,
# remove only previously merged top-level files; a separate lifecycle target
# can be recreated without touching the checkpoint directory.
if [[ "${TARGET_DIR}" == "${BASE_DIR}" ]]; then
  find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type f -print -delete
else
  mkdir -p "${TARGET_DIR}"
fi

"${PYTHON_BIN}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TARGET_DIR}"

echo "Merge completed."
