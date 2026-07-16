#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export TRAINER_N_GPUS_PER_NODE="${ASCEND_INSTALL_MIN_NPUS:-1}"

# CANN/NNAL must already exist on the host or in the base container.
# shellcheck source=ascend_env.sh
source "$PROJECT_ROOT/scripts/ascend_env.sh"

"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 12), "Python 3.10 or 3.11 is required"'
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements-ascend.txt"
"$PYTHON_BIN" -m pip install -e "$PROJECT_ROOT" --no-deps
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/check_ascend_env.py" \
    --min-npus "$TRAINER_N_GPUS_PER_NODE"

echo "Ascend Python environment is ready."
echo "Unified job entry: bash scripts/start_vision_opd_ascend.sh"
