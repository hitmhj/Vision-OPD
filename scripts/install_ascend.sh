#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VOPD_CONFIG_FILE="${VOPD_CONFIG_FILE:-${PROJECT_ROOT}/vision_opd_ascend.env}"

if [[ -f "$VOPD_CONFIG_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$VOPD_CONFIG_FILE"
    set +a
fi

SETUP_DIR="${VOPD_DEPENDENCY_SETUP_DIR:-$PROJECT_ROOT}"
if [[ "$SETUP_DIR" != /* ]]; then
    SETUP_DIR="${PROJECT_ROOT}/${SETUP_DIR}"
fi
SETUP_SCRIPT="${VOPD_DEPENDENCY_SETUP_SCRIPT:-train_scripts/init_env_qwen35vl_speedup_local.sh}"
if [[ "$SETUP_SCRIPT" != /* ]]; then
    SETUP_SCRIPT="${SETUP_DIR}/${SETUP_SCRIPT}"
fi

if [[ ! -d "$SETUP_DIR" ]]; then
    echo "Dependency setup directory does not exist: $SETUP_DIR" >&2
    exit 2
fi
if [[ ! -f "$SETUP_SCRIPT" ]]; then
    echo "Huawei dependency setup script does not exist: $SETUP_SCRIPT" >&2
    exit 2
fi

echo "Using Huawei dependency initializer: $SETUP_SCRIPT"
pushd "$SETUP_DIR" >/dev/null
bash "$SETUP_SCRIPT"
popd >/dev/null

if [[ "${VOPD_INSTALL_SAMPLE_ACCELERATE:-1}" == "1" ]]; then
    SAMPLE_ACCELERATE_VERSION="${VOPD_SAMPLE_ACCELERATE_VERSION:-1.11.0}"
    echo "Applying sample dependency: accelerate==$SAMPLE_ACCELERATE_VERSION"
    "$PYTHON_BIN" -m pip install "accelerate==$SAMPLE_ACCELERATE_VERSION"
fi

echo "Huawei-provided Python dependency setup completed."
