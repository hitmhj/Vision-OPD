#!/usr/bin/env bash

# Resolve one coherent Python/wheelhouse/venv unit for the NPU worker. This
# file is sourced by the public lifecycle and the installer so every stage
# uses the same interpreter and ABI tag.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file instead: source scripts/resolve_ascend_runtime.sh" >&2
    exit 2
fi

set +u

if [[ "${VOPD_RUNTIME_PROFILE_READY:-0}" == "1" ]]; then
    return 0
fi

_vopd_profile_resolve_path() {
    local value="$1"
    if [[ "$value" == /* ]]; then
        printf '%s\n' "$value"
    else
        printf '%s\n' "${PROJECT_ROOT}/${value}"
    fi
}

_vopd_profile_python_version() {
    "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
        2>/dev/null
}

_vopd_profile_tag() {
    local version="$1"
    printf 'cp%s\n' "${version/./}"
}

_vopd_profile_default_wheel_dir() {
    local tag="$1"
    _vopd_profile_resolve_path "${VOPD_WHEEL_ROOT:-envs/wheels}/${tag}-aarch64"
}

_vopd_profile_has_manifest() {
    local version="$1"
    local tag wheel_dir
    tag="$(_vopd_profile_tag "$version")"
    if [[ -n "${VOPD_LOCAL_WHEEL_DIR:-}" ]]; then
        wheel_dir="$(_vopd_profile_resolve_path "$VOPD_LOCAL_WHEEL_DIR")"
    else
        wheel_dir="$(_vopd_profile_default_wheel_dir "$tag")"
    fi
    [[ -f "$wheel_dir/.vision_opd_wheelhouse_manifest.json" ]]
}

_vopd_profile_has_runtime() {
    local version="$1"
    local tag venv_dir
    tag="$(_vopd_profile_tag "$version")"
    if [[ -n "${VOPD_VENV_DIR:-}" ]]; then
        venv_dir="$(_vopd_profile_resolve_path "$VOPD_VENV_DIR")"
    else
        venv_dir="$(_vopd_profile_resolve_path "${VOPD_RUNTIME_ROOT:-envs/runtime}/.venv-ascend-${tag}")"
    fi
    [[ -x "$venv_dir/bin/python" && -f "$venv_dir/.vision_opd_requirements.sha256" ]]
}

_vopd_profile_candidate_path() {
    local name="$1"
    command -v "$name" 2>/dev/null || true
}

_vopd_profile_select_python() {
    local requested="${VOPD_TARGET_PYTHON:-auto}"
    local explicit="${VOPD_BOOTSTRAP_PYTHON:-}"
    local candidate path version phase
    local -a candidates=()

    if [[ -n "$explicit" ]]; then
        if [[ ! -x "$explicit" ]]; then
            echo "VOPD_BOOTSTRAP_PYTHON is not executable: $explicit" >&2
            return 1
        fi
        version="$(_vopd_profile_python_version "$explicit")"
        printf '%s\n' "$explicit"
        return 0
    fi

    if [[ "$requested" == "auto" ]]; then
        for version in ${VOPD_PYTHON_PREFERENCE:-3.11 3.10}; do
            candidates+=("python${version}")
        done
        candidates+=(python3 python)
    else
        candidates+=("python${requested}" python3 python)
    fi

    # Prefer an existing runtime or prepared wheelhouse, then use the first
    # working Python and let pip/the real training imports decide compatibility.
    for phase in with_runtime with_manifest any_python; do
        for candidate in "${candidates[@]}"; do
            path="$(_vopd_profile_candidate_path "$candidate")"
            [[ -n "$path" && -x "$path" ]] || continue
            version="$(_vopd_profile_python_version "$path")"
            [[ -n "$version" ]] || continue
            if [[ "$phase" != "any_python" && "$requested" != "auto" && "$version" != "$requested" ]]; then
                continue
            fi
            case "$phase" in
                with_runtime) _vopd_profile_has_runtime "$version" || continue ;;
                with_manifest) _vopd_profile_has_manifest "$version" || continue ;;
            esac
            printf '%s\n' "$path"
            return 0
        done
    done

    echo "No working Python interpreter was found on the worker." >&2
    echo "Detected candidates:" >&2
    for candidate in python3.11 python3.10 python3 python; do
        path="$(_vopd_profile_candidate_path "$candidate")"
        [[ -n "$path" ]] || continue
        echo "  $candidate -> $path ($($path --version 2>&1 || true))" >&2
    done
    echo "Inject VOPD_BOOTSTRAP_PYTHON with the worker's Python executable." >&2
    return 1
}

VOPD_BOOTSTRAP_PYTHON="$(_vopd_profile_select_python)" || return $?
VOPD_PYTHON_VERSION="$(_vopd_profile_python_version "$VOPD_BOOTSTRAP_PYTHON")"
if [[ "${VOPD_TARGET_PYTHON:-auto}" != "auto" && \
      "$VOPD_PYTHON_VERSION" != "$VOPD_TARGET_PYTHON" ]]; then
    echo "WARNING: requested Python $VOPD_TARGET_PYTHON is unavailable; using Python $VOPD_PYTHON_VERSION." >&2
fi
VOPD_PYTHON_TAG="$(_vopd_profile_tag "$VOPD_PYTHON_VERSION")"

if [[ -n "${VOPD_LOCAL_WHEEL_DIR:-}" ]]; then
    VOPD_LOCAL_WHEEL_DIR="$(_vopd_profile_resolve_path "$VOPD_LOCAL_WHEEL_DIR")"
else
    VOPD_LOCAL_WHEEL_DIR="$(_vopd_profile_default_wheel_dir "$VOPD_PYTHON_TAG")"
fi
if [[ -n "${VOPD_VENV_DIR:-}" ]]; then
    VOPD_VENV_DIR="$(_vopd_profile_resolve_path "$VOPD_VENV_DIR")"
else
    VOPD_VENV_DIR="$(_vopd_profile_resolve_path "${VOPD_RUNTIME_ROOT:-envs/runtime}/.venv-ascend-${VOPD_PYTHON_TAG}")"
fi

export VOPD_BOOTSTRAP_PYTHON VOPD_PYTHON_VERSION VOPD_PYTHON_TAG
export VOPD_LOCAL_WHEEL_DIR VOPD_VENV_DIR
export PYTHONNOUSERSITE=1
export VOPD_RUNTIME_PROFILE_READY=1
