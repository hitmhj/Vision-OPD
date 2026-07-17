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

VENV_DIR="$(_vopd_resolve_path "${VOPD_VENV_DIR:-.venv-ascend}")"
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

_vopd_python_supported() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' \
        >/dev/null 2>&1
}

_vopd_select_bootstrap_python() {
    local candidate
    if [[ -n "${VOPD_BOOTSTRAP_PYTHON:-}" ]]; then
        if [[ ! -x "$VOPD_BOOTSTRAP_PYTHON" ]] || ! _vopd_python_supported "$VOPD_BOOTSTRAP_PYTHON"; then
            echo "VOPD_BOOTSTRAP_PYTHON must be an executable Python 3.10: $VOPD_BOOTSTRAP_PYTHON" >&2
            return 1
        fi
        printf '%s\n' "$VOPD_BOOTSTRAP_PYTHON"
        return 0
    fi

    for candidate in python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && _vopd_python_supported "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done

    echo "No supported Python was found on the NPU worker." >&2
    echo "Use the ModelArts Python 3.10 worker environment, or inject VOPD_BOOTSTRAP_PYTHON." >&2
    return 1
}

BOOTSTRAP_PYTHON="$(_vopd_select_bootstrap_python)"
MACHINE="$(uname -m)"
if [[ "$MACHINE" != "aarch64" ]]; then
    echo "Unsupported training worker architecture: $MACHINE (expected aarch64 for Atlas 910B)." >&2
    echo "Do not build the NPU environment in the x86_64 WebStudio session." >&2
    exit 2
fi

echo "Vision-OPD dependency target"
echo "  worker_arch:      $MACHINE"
echo "  bootstrap_python: $BOOTSTRAP_PYTHON ($($BOOTSTRAP_PYTHON --version 2>&1))"
echo "  venv:             $VENV_DIR"
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
INSTALL_FINGERPRINT="schema=5;requirements=${REQUIREMENTS_SHA};python=$($BOOTSTRAP_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")');arch=${MACHINE}"
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
if ! _vopd_python_supported "$RUNTIME_PYTHON"; then
    echo "Existing venv uses an unsupported Python: $($RUNTIME_PYTHON --version 2>&1)" >&2
    echo "Delete $VENV_DIR or point VOPD_VENV_DIR to a new directory." >&2
    exit 2
fi

PIP_ARGS=(
    --disable-pip-version-check
    --no-input
    --timeout "${VOPD_PIP_TIMEOUT:-120}"
    --retries "${VOPD_PIP_RETRIES:-5}"
)
# Do not inherit a base image's pip extra-index/find-links settings. All package
# sources for this lifecycle are declared explicitly below.
export PIP_CONFIG_FILE="${VOPD_PIP_CONFIG_FILE:-/dev/null}"
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
    if [[ ! -d "$LOCAL_WHEEL_DIR" ]]; then
        echo "Configured VOPD_LOCAL_WHEEL_DIR does not exist: $LOCAL_WHEEL_DIR" >&2
        exit 2
    fi
    PIP_ARGS+=(--find-links "$LOCAL_WHEEL_DIR")
fi

if [[ "${VOPD_PIP_NO_INDEX:-0}" == "1" ]]; then
    "$BOOTSTRAP_PYTHON" "$PROJECT_ROOT/scripts/check_ascend_assets.py" \
        --project-root "$PROJECT_ROOT" \
        --wheel-dir "$LOCAL_WHEEL_DIR"
fi

# Upgrade the isolated installer from the declared source as well. In offline
# mode pip/setuptools/wheel must therefore be present in the wheelhouse.
"$RUNTIME_PYTHON" -m pip install \
    "${PIP_ARGS[@]}" \
    --upgrade \
    "pip>=23.3,<26" setuptools wheel

# Resolve the clean environment before mutating its NPU stack. This catches an
# incomplete offline wheelhouse at the beginning of the task rather than after
# half of the packages have been installed. Hardware plugins are intentionally
# excluded because they are installed --no-deps below.
echo "[install 0/4] Resolving the complete worker environment without installation..."
"$RUNTIME_PYTHON" -m pip install \
    "${PIP_ARGS[@]}" \
    --dry-run \
    --ignore-installed \
    --constraint "$CORE_REQUIREMENTS_FILE" \
    -r "$CORE_REQUIREMENTS_FILE" \
    -r "$REQUIREMENTS_FILE"

echo "[install 1/4] Installing the official CANN 8.5.1 / PyTorch 2.9 compatibility unit..."
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

_vopd_vllm_importable() {
    VLLM_PLUGINS=ascend "$RUNTIME_PYTHON" -c '
import vllm
import vllm_ascend
from vllm.platforms import current_platform
if getattr(current_platform, "device_type", None) != "npu":
    raise RuntimeError(f"vLLM platform is {type(current_platform).__name__}, not Ascend NPU")
print(f"vLLM import check: {vllm.__version__}; {type(current_platform).__name__}")
'
}

if "$RUNTIME_PYTHON" -m pip install \
    "${PIP_ARGS[@]}" \
    --upgrade \
    --only-binary=:all: \
    --no-deps \
    -r "$PLUGIN_REQUIREMENTS_FILE"; then
    if ! _vopd_vllm_importable; then
        if [[ "${VOPD_VLLM_ALLOW_SOURCE_FALLBACK:-0}" != "1" ]]; then
            echo "The prebuilt vLLM wheel cannot load with the NPU stack and source fallback is disabled." >&2
            exit 1
        fi
        echo "The prebuilt vLLM wheel cannot load with the selected NPU ABI; rebuilding it..."
        _vopd_install_empty_vllm
    fi
else
    if [[ "${VOPD_VLLM_ALLOW_SOURCE_FALLBACK:-0}" != "1" ]]; then
        echo "Prebuilt vLLM/vLLM-Ascend installation failed and source fallback is disabled." >&2
        echo "Add compatible cp310/aarch64 wheels to VOPD_LOCAL_WHEEL_DIR." >&2
        exit 1
    fi
    echo "Prebuilt vLLM wheel is incompatible or unavailable; using the source fallback..."
    _vopd_install_empty_vllm
fi
_vopd_vllm_importable

echo "[install 4/4] Installing the local Vision-OPD package..."
"$RUNTIME_PYTHON" -m pip install "${PIP_ARGS[@]}" --no-build-isolation --no-deps --editable "$PROJECT_ROOT"

# Validate every direct pin, every import used by the active training path and
# all dependency metadata except the documented accelerator/plugin divergences.
"$RUNTIME_PYTHON" "$PROJECT_ROOT/scripts/check_ascend_env.py" --dependencies-only
printf '%s\n' "$INSTALL_FINGERPRINT" > "$MARKER_FILE"

echo "Vision-OPD Ascend Python environment is ready: $RUNTIME_PYTHON"
