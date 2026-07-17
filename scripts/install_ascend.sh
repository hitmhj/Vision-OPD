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
INSTALL_MODE="${VOPD_INSTALL_MODE:-auto}"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "Ascend requirements file does not exist: $REQUIREMENTS_FILE" >&2
    exit 2
fi

_vopd_python_supported() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 12) else 1)' \
        >/dev/null 2>&1
}

_vopd_select_bootstrap_python() {
    local candidate
    if [[ -n "${VOPD_BOOTSTRAP_PYTHON:-}" ]]; then
        if [[ ! -x "$VOPD_BOOTSTRAP_PYTHON" ]] || ! _vopd_python_supported "$VOPD_BOOTSTRAP_PYTHON"; then
            echo "VOPD_BOOTSTRAP_PYTHON must be an executable Python 3.10 or 3.11: $VOPD_BOOTSTRAP_PYTHON" >&2
            return 1
        fi
        printf '%s\n' "$VOPD_BOOTSTRAP_PYTHON"
        return 0
    fi

    for candidate in python3.11 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && _vopd_python_supported "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done

    echo "No supported Python was found on the NPU worker." >&2
    echo "Use a ModelArts image with Python 3.10/3.11, or inject VOPD_BOOTSTRAP_PYTHON." >&2
    return 1
}

BOOTSTRAP_PYTHON="$(_vopd_select_bootstrap_python)"
MACHINE="$(uname -m)"
case "$MACHINE" in
    x86_64|aarch64) ;;
    *)
        echo "Unsupported worker architecture: $MACHINE (expected x86_64 or aarch64)" >&2
        exit 2
        ;;
esac

echo "Vision-OPD dependency target"
echo "  worker_arch:      $MACHINE"
echo "  bootstrap_python: $BOOTSTRAP_PYTHON ($($BOOTSTRAP_PYTHON --version 2>&1))"
echo "  venv:             $VENV_DIR"
echo "  requirements:     $REQUIREMENTS_FILE"

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

REQUIREMENTS_SHA="$($BOOTSTRAP_PYTHON -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$REQUIREMENTS_FILE")"
INSTALL_FINGERPRINT="schema=2;requirements=${REQUIREMENTS_SHA};python=$($BOOTSTRAP_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")');arch=${MACHINE}"
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

PIP_ARGS=(--disable-pip-version-check --no-input)
if [[ "${VOPD_PIP_NO_INDEX:-0}" == "1" ]]; then
    PIP_ARGS+=(--no-index)
    if [[ -z "${VOPD_LOCAL_WHEEL_DIR:-}" ]]; then
        echo "VOPD_PIP_NO_INDEX=1 requires a complete VOPD_LOCAL_WHEEL_DIR." >&2
        exit 2
    fi
fi
if [[ -n "${VOPD_PIP_INDEX_URL:-}" ]]; then
    PIP_ARGS+=(--index-url "$VOPD_PIP_INDEX_URL")
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
    "$RUNTIME_PYTHON" -m pip --version
else
    "$RUNTIME_PYTHON" -m pip install "${PIP_ARGS[@]}" --upgrade "pip>=23.3,<26" setuptools wheel
fi
"$RUNTIME_PYTHON" -m pip install "${PIP_ARGS[@]}" --upgrade --upgrade-strategy only-if-needed -r "$REQUIREMENTS_FILE"
"$RUNTIME_PYTHON" -m pip install "${PIP_ARGS[@]}" --no-build-isolation --no-deps --editable "$PROJECT_ROOT"

# Validate resolver consistency before marking the environment reusable.
"$RUNTIME_PYTHON" -m pip check
printf '%s\n' "$INSTALL_FINGERPRINT" > "$MARKER_FILE"

echo "Vision-OPD Ascend Python environment is ready: $RUNTIME_PYTHON"
