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

case "${VOPD_STACK_PROFILE:-vllm-ascend-0.18-cann8.5.1}" in
    vllm-ascend-0.18-cann8.5.1) ;;
    *)
        echo "Unsupported VOPD_STACK_PROFILE: ${VOPD_STACK_PROFILE}" >&2
        echo "Supported profile: vllm-ascend-0.18-cann8.5.1" >&2
        return 2
        ;;
esac

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

_vopd_profile_supported() {
    local actual="$1"
    local supported
    for supported in ${VOPD_SUPPORTED_PYTHONS:-3.11 3.10}; do
        [[ "$actual" == "$supported" ]] && return 0
    done
    return 1
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
        if ! _vopd_profile_supported "$version"; then
            echo "VOPD_BOOTSTRAP_PYTHON is Python ${version:-unknown}; supported: ${VOPD_SUPPORTED_PYTHONS:-3.11 3.10}" >&2
            return 1
        fi
        if [[ "$requested" != "auto" && "$version" != "$requested" ]]; then
            echo "VOPD_BOOTSTRAP_PYTHON is Python $version, but VOPD_TARGET_PYTHON=$requested." >&2
            return 1
        fi
        printf '%s\n' "$explicit"
        return 0
    fi

    if [[ "$requested" != "auto" ]] && ! _vopd_profile_supported "$requested"; then
        echo "VOPD_TARGET_PYTHON must be auto or one of: ${VOPD_SUPPORTED_PYTHONS:-3.11 3.10}; got $requested" >&2
        return 1
    fi

    if [[ "$requested" == "auto" ]]; then
        for version in ${VOPD_SUPPORTED_PYTHONS:-3.11 3.10}; do
            candidates+=("python${version}")
        done
        candidates+=(python3 python)
    else
        candidates+=("python${requested}" python3 python)
    fi

    # Reuse an already validated runtime first, then prefer an interpreter whose
    # offline wheelhouse was completely resolved. The last phase still selects
    # a supported interpreter so a missing-assets error names the exact target.
    for phase in with_runtime with_manifest any_supported; do
        for candidate in "${candidates[@]}"; do
            path="$(_vopd_profile_candidate_path "$candidate")"
            [[ -n "$path" && -x "$path" ]] || continue
            version="$(_vopd_profile_python_version "$path")"
            _vopd_profile_supported "$version" || continue
            [[ "$requested" == "auto" || "$version" == "$requested" ]] || continue
            case "$phase" in
                with_runtime) _vopd_profile_has_runtime "$version" || continue ;;
                with_manifest) _vopd_profile_has_manifest "$version" || continue ;;
            esac
            printf '%s\n' "$path"
            return 0
        done
    done

    echo "No supported Python was found on the NPU worker." >&2
    echo "Detected candidates:" >&2
    for candidate in python3.11 python3.10 python3 python; do
        path="$(_vopd_profile_candidate_path "$candidate")"
        [[ -n "$path" ]] || continue
        echo "  $candidate -> $path ($($path --version 2>&1 || true))" >&2
    done
    echo "Use a Python 3.10/3.11 worker image or inject VOPD_BOOTSTRAP_PYTHON." >&2
    return 1
}

VOPD_BOOTSTRAP_PYTHON="$(_vopd_profile_select_python)" || return $?
VOPD_PYTHON_VERSION="$(_vopd_profile_python_version "$VOPD_BOOTSTRAP_PYTHON")"
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
