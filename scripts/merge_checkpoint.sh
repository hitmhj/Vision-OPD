#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="${BASE_DIR:-${1:-}}"
if [[ -z "${BASE_DIR}" ]]; then
  echo "Usage: $0 <global_step_checkpoint_dir> [merged_target_dir]" >&2
  exit 2
fi
BASE_DIR="${BASE_DIR%/}"
ACTOR_DIR="${BASE_DIR}/actor"
TARGET_DIR="${TARGET_DIR:-${2:-${PROJECT_ROOT}/output/merged/$(basename "${BASE_DIR}")}}"

if [ ! -d "${ACTOR_DIR}" ]; then
  echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
  exit 1
fi

BASE_RESOLVED="$(cd "${BASE_DIR}" && pwd)"
mkdir -p "${TARGET_DIR}"
TARGET_RESOLVED="$(cd "${TARGET_DIR}" && pwd)"
case "${TARGET_RESOLVED}" in
  "${BASE_RESOLVED}"|"${BASE_RESOLVED}"/*)
    echo "Merged target must be outside the sharded checkpoint directory." >&2
    exit 2
    ;;
esac

echo "Merging ${ACTOR_DIR} -> ${TARGET_DIR}"

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TARGET_DIR}"

echo "Merge completed."
