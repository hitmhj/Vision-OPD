#!/usr/bin/env bash

# Match the Huawei sample bootstrap. Vendor initialization scripts are not
# nounset-safe (ATB reads ZSH_VERSION directly), so this stage must not use -u.
set +u
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
# The Huawei initializer expects its own directory as cwd, exactly like the
# sample's `cd /opt/.../llama_factory...`. pushd/popd provides that cwd without
# losing the relocatable Vision-OPD project root used after installation.
pushd "$SETUP_DIR" >/dev/null
bash "$SETUP_SCRIPT"
if [[ "${VOPD_INSTALL_SAMPLE_ACCELERATE:-1}" == "1" ]]; then
    SAMPLE_ACCELERATE_VERSION="${VOPD_SAMPLE_ACCELERATE_VERSION:-1.11.0}"
    echo "Applying sample dependency: accelerate==$SAMPLE_ACCELERATE_VERSION"
    pip install "accelerate==$SAMPLE_ACCELERATE_VERSION"
fi
popd >/dev/null

echo "Huawei-provided Python dependency setup completed."
