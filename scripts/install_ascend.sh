#!/usr/bin/env bash

# Build an isolated Vision-OPD environment on the actual NPU worker.  WebStudio
# may have a different CPU architecture and Python version, so dependency
# compatibility is intentionally decided here rather than on the editing host.
set +u
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOPD_CONFIG_FILE="${VOPD_CONFIG_FILE:-${PROJECT_ROOT}/vision_opd_ascend.env}"

if [[ -f "$VOPD_CONFIG_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$VOPD_CONFIG_FILE"
    set +a
fi

_vopd_resolve_path() {
    local value="$1"
    if [[ "$value" == /* ]]; then
        printf '%s\n' "$value"
    else
        printf '%s\n' "${PROJECT_ROOT}/${value}"
    fi
}

REQUIREMENTS_FILE="$(_vopd_resolve_path "${VOPD_ASCEND_REQUIREMENTS:-requirements-ascend.txt}")"
CORE_REQUIREMENTS_FILE="$(_vopd_resolve_path "${VOPD_ASCEND_CORE_REQUIREMENTS:-requirements-ascend-core.txt}")"
PLUGIN_REQUIREMENTS_FILE="$(_vopd_resolve_path "${VOPD_ASCEND_PLUGIN_REQUIREMENTS:-requirements-ascend-plugins.txt}")"
INSTALL_MODE="${VOPD_INSTALL_MODE:-auto}"

for _vopd_lock_file in "$REQUIREMENTS_FILE" "$CORE_REQUIREMENTS_FILE" "$PLUGIN_REQUIREMENTS_FILE"; do
    if [[ ! -f "$_vopd_lock_file" ]]; then
        echo "Ascend requirements file does not exist: $_vopd_lock_file" >&2
        exit 2
    fi
done

MACHINE="$(uname -m)"
if [[ "$MACHINE" != "aarch64" ]]; then
    echo "WARNING: worker architecture is $MACHINE; Atlas 910B normally uses aarch64. Continuing." >&2
fi

# Resolve Python, ABI-specific wheelhouse and versioned venv as one unit. The
# public entry normally sources this first; sourcing it again is guarded.
# shellcheck source=resolve_ascend_runtime.sh
source "$PROJECT_ROOT/scripts/resolve_ascend_runtime.sh"
BOOTSTRAP_PYTHON="$VOPD_BOOTSTRAP_PYTHON"
VENV_DIR="$VOPD_VENV_DIR"

_vopd_glibc_version="$(ldd --version 2>/dev/null | head -n 1 | grep -Eo '[0-9]+\.[0-9]+' | tail -n 1 || true)"

echo "Vision-OPD dependency target"
echo "  worker_arch:      $MACHINE"
echo "  bootstrap_python: $BOOTSTRAP_PYTHON ($($BOOTSTRAP_PYTHON --version 2>&1))"
echo "  python_target:    $VOPD_PYTHON_VERSION ($VOPD_PYTHON_TAG)"
echo "  venv:             $VENV_DIR"
echo "  wheelhouse:       $VOPD_LOCAL_WHEEL_DIR"
echo "  stack_profile:    $VOPD_STACK_PROFILE"
echo "  glibc:            ${_vopd_glibc_version:-unknown}"
echo "  runtime_lock:     $REQUIREMENTS_FILE"
echo "  npu_core_lock:    $CORE_REQUIREMENTS_FILE"
echo "  plugin_lock:      $PLUGIN_REQUIREMENTS_FILE"

if [[ "$INSTALL_MODE" == "never" ]]; then
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        echo "VOPD_INSTALL_MODE=never but the runtime environment is missing: $VENV_DIR" >&2
        exit 2
    fi
    echo "Dependency installation disabled; reusing $VENV_DIR."
    exit 0
fi
if [[ "$INSTALL_MODE" != "auto" && "$INSTALL_MODE" != "always" ]]; then
    echo "VOPD_INSTALL_MODE must be auto, always, or never; got $INSTALL_MODE" >&2
    exit 2
fi

REQUIREMENTS_SHA="$($BOOTSTRAP_PYTHON -c '
import hashlib, pathlib, sys
digest = hashlib.sha256()
for name in sys.argv[1:]:
    path = pathlib.Path(name)
    digest.update(path.name.encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
' "$REQUIREMENTS_FILE" "$CORE_REQUIREMENTS_FILE" "$PLUGIN_REQUIREMENTS_FILE" "$0")"
INSTALL_FINGERPRINT="schema=6;requirements=${REQUIREMENTS_SHA};python=${VOPD_PYTHON_VERSION};arch=${MACHINE};profile=${VOPD_STACK_PROFILE};cann=${VOPD_EXPECTED_CANN_VERSION:-unset}"
MARKER_FILE="$VENV_DIR/.vision_opd_requirements.sha256"
if [[ "$INSTALL_MODE" == "auto" && -x "$VENV_DIR/bin/python" && -f "$MARKER_FILE" ]]; then
    if [[ "$(<"$MARKER_FILE")" == "$INSTALL_FINGERPRINT" ]]; then
        echo "Ascend environment already matches requirements; skipping pip installation."
        exit 0
    fi
fi

# vLLM-Ascend wheels are normally prebuilt, but some platform mirrors may
# provide a source distribution. Make CANN headers and libraries visible to
# that build process. This source occurs in the installer subprocess; the
# parent training shell loads the same runtime again after installation.
# shellcheck source=ascend_env.sh
source "$PROJECT_ROOT/scripts/ascend_env.sh"

mkdir -p "$(dirname "$VENV_DIR")"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

RUNTIME_PYTHON="$VENV_DIR/bin/python"

PIP_ARGS=(
    --disable-pip-version-check
    --no-input
    --timeout "${VOPD_PIP_TIMEOUT:-120}"
    --retries "${VOPD_PIP_RETRIES:-5}"
)
# Do not inherit a base image's pip extra-index/find-links settings. All package
# sources for this lifecycle are declared explicitly below.
export PIP_CONFIG_FILE="${VOPD_PIP_CONFIG_FILE:-/dev/null}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX
if [[ "${VOPD_PIP_NO_INDEX:-0}" == "1" ]]; then
    PIP_ARGS+=(--no-index)
    if [[ -z "${VOPD_LOCAL_WHEEL_DIR:-}" ]]; then
        echo "VOPD_PIP_NO_INDEX=1 requires a complete VOPD_LOCAL_WHEEL_DIR." >&2
        exit 2
    fi
fi
if [[ "${VOPD_PIP_NO_INDEX:-0}" != "1" && -n "${VOPD_PIP_INDEX_URL:-}" ]]; then
    PIP_ARGS+=(--index-url "$VOPD_PIP_INDEX_URL")
fi
if [[ "${VOPD_PIP_NO_INDEX:-0}" != "1" && -n "${VOPD_PIP_EXTRA_INDEX_URL:-}" ]]; then
    PIP_ARGS+=(--extra-index-url "$VOPD_PIP_EXTRA_INDEX_URL")
fi
if [[ -n "${VOPD_PIP_TRUSTED_HOST:-}" ]]; then
    PIP_ARGS+=(--trusted-host "$VOPD_PIP_TRUSTED_HOST")
fi
if [[ -n "${VOPD_LOCAL_WHEEL_DIR:-}" ]]; then
    LOCAL_WHEEL_DIR="$(_vopd_resolve_path "$VOPD_LOCAL_WHEEL_DIR")"
    if [[ -d "$LOCAL_WHEEL_DIR" ]]; then
        PIP_ARGS+=(--find-links "$LOCAL_WHEEL_DIR")
    else
        echo "WARNING: configured wheel directory does not exist: $LOCAL_WHEEL_DIR; pip will continue with the remaining sources." >&2
    fi
fi

# Upgrade the isolated installer from the declared source as well. In offline
# mode pip/setuptools/wheel must therefore be present in the wheelhouse.
"$RUNTIME_PYTHON" -m pip install \
    "${PIP_ARGS[@]}" \
    --upgrade \
    "pip>=23.3,<26" setuptools wheel

echo "[install 1/4] Installing the ${VOPD_STACK_PROFILE} / Python ${VOPD_PYTHON_VERSION} compatibility unit..."
echo "  primary_index:    ${VOPD_PIP_INDEX_URL:-pip default}"
echo "  extra_index:      ${VOPD_PIP_EXTRA_INDEX_URL:-disabled}"
echo "  local_wheels:     ${VOPD_LOCAL_WHEEL_DIR:-disabled}"
echo "  pip_config:       $PIP_CONFIG_FILE"
"$RUNTIME_PYTHON" -m pip install \
    "${PIP_ARGS[@]}" \
    --upgrade \
    --upgrade-strategy only-if-needed \
    -r "$CORE_REQUIREMENTS_FILE"

echo "[install 2/4] Installing Vision-OPD and generic inference dependencies..."
"$RUNTIME_PYTHON" -m pip install \
    "${PIP_ARGS[@]}" \
    --upgrade \
    --upgrade-strategy only-if-needed \
    --constraint "$CORE_REQUIREMENTS_FILE" \
    -r "$REQUIREMENTS_FILE"

echo "[install 3/4] Installing prebuilt vLLM and vLLM-Ascend wheels..."
# vLLM 0.18's wheel metadata describes its CUDA/PyTorch dependency set, while
# the official Ascend 0.18 Atlas A2 matrix uses the special torch-npu build in
# requirements-ascend-core.txt.
# Installing only these two wheel payloads after their complete curated runtime
# prevents pip from replacing the working NPU ABI with CUDA or another CANN line.
_vopd_install_empty_vllm() {
    echo "Building vLLM's hardware-neutral payload (VLLM_TARGET_DEVICE=empty)..."
    # The PyPI source archive avoids a GitHub dependency and also supports
    # worker images older than the upstream manylinux_2_31 aarch64 wheel.
    VLLM_TARGET_DEVICE=empty "$RUNTIME_PYTHON" -m pip install \
        "${PIP_ARGS[@]}" \
        --upgrade \
        --force-reinstall \
        --no-deps \
        --no-build-isolation \
        --no-binary=vllm \
        "vllm==0.18.0"
    "$RUNTIME_PYTHON" -m pip install \
        "${PIP_ARGS[@]}" \
        --upgrade \
        --only-binary=:all: \
        --no-deps \
        "vllm-ascend==0.18.0"
}

if "$RUNTIME_PYTHON" -m pip install \
    "${PIP_ARGS[@]}" \
    --upgrade \
    --only-binary=:all: \
    --no-deps \
    -r "$PLUGIN_REQUIREMENTS_FILE"; then
    echo "Prebuilt vLLM and vLLM-Ascend packages installed; runtime loading is left to training."
else
    if [[ "${VOPD_VLLM_ALLOW_SOURCE_FALLBACK:-0}" != "1" ]]; then
        echo "Prebuilt vLLM/vLLM-Ascend installation failed and source fallback is disabled." >&2
        echo "Add compatible ${VOPD_PYTHON_TAG}/aarch64 wheels to VOPD_LOCAL_WHEEL_DIR." >&2
        exit 1
    fi
    echo "Prebuilt vLLM wheel is incompatible or unavailable; using the source fallback..."
    _vopd_install_empty_vllm
fi

echo "[install 4/4] Installing the local Vision-OPD package..."
"$RUNTIME_PYTHON" -m pip install "${PIP_ARGS[@]}" --no-build-isolation --no-deps --editable "$PROJECT_ROOT"

printf '%s\n' "$INSTALL_FINGERPRINT" > "$MARKER_FILE"

echo "Vision-OPD Ascend Python environment is ready: $RUNTIME_PYTHON"
