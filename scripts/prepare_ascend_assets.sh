#!/usr/bin/env bash

# Prepare portable Vision-OPD model and wheel assets. This entry never starts
# training and stays offline unless --online (or VOPD_PREPARE_ONLINE=1) is set.
set +u
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOPD_CONFIG_FILE="${VOPD_CONFIG_FILE:-${PROJECT_ROOT}/vision_opd_ascend.env}"

if [[ ! -f "$VOPD_CONFIG_FILE" ]]; then
    echo "Vision-OPD config file does not exist: $VOPD_CONFIG_FILE" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$VOPD_CONFIG_FILE"
set +a

_vopd_resolve_path() {
    local value="$1"
    if [[ "$value" == /* ]]; then
        printf '%s\n' "$value"
    else
        printf '%s\n' "${PROJECT_ROOT}/${value}"
    fi
}

_vopd_usage() {
    cat <<'EOF'
Usage: bash scripts/prepare_ascend_assets.sh [options]

Options:
  --online                    Permit model and wheel downloads from configured sources.
  --check-only                Validate existing assets without copying or downloading.
  --internal-wheel-dir DIR    Add a platform-internal wheel source; may be repeated.
  --model-source-dir DIR      Copy an already downloaded model snapshot into envs/.
  --skip-model                Prepare/check only the Python wheelhouse.
  --skip-wheels               Prepare/check only the model snapshot.
  -h, --help                  Show this message.

The default is offline. VOPD_INTERNAL_WHEEL_DIRS may contain colon-separated
directories. Relative paths are resolved from the repository root.
EOF
}

ONLINE="${VOPD_PREPARE_ONLINE:-0}"
CHECK_ONLY=0
SKIP_MODEL=0
SKIP_WHEELS=0
MODEL_SOURCE_DIR="${VOPD_MODEL_SOURCE_DIR:-}"
declare -a INTERNAL_WHEEL_DIRS=()

if [[ -n "${VOPD_INTERNAL_WHEEL_DIRS:-}" ]]; then
    IFS=':' read -r -a INTERNAL_WHEEL_DIRS <<< "$VOPD_INTERNAL_WHEEL_DIRS"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --online)
            ONLINE=1
            shift
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --internal-wheel-dir)
            [[ $# -ge 2 ]] || { echo "--internal-wheel-dir requires a path" >&2; exit 2; }
            INTERNAL_WHEEL_DIRS+=("$2")
            shift 2
            ;;
        --model-source-dir)
            [[ $# -ge 2 ]] || { echo "--model-source-dir requires a path" >&2; exit 2; }
            MODEL_SOURCE_DIR="$2"
            shift 2
            ;;
        --skip-model)
            SKIP_MODEL=1
            shift
            ;;
        --skip-wheels)
            SKIP_WHEELS=1
            shift
            ;;
        -h|--help)
            _vopd_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            _vopd_usage >&2
            exit 2
            ;;
    esac
done

if [[ "$ONLINE" != "0" && "$ONLINE" != "1" ]]; then
    echo "VOPD_PREPARE_ONLINE must be 0 or 1; got $ONLINE" >&2
    exit 2
fi

# The preparation phase declares its own sources. Do not let WebStudio or a
# base image silently inject an unrelated pip index, find-links directory or
# offline mode. The training entry remains offline independently of this flag.
export PIP_CONFIG_FILE="${VOPD_PIP_CONFIG_FILE:-/dev/null}"
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX
if [[ "$ONLINE" == "1" ]]; then
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
fi
if [[ "$SKIP_MODEL" == "1" && "$SKIP_WHEELS" == "1" ]]; then
    echo "--skip-model and --skip-wheels cannot be used together." >&2
    exit 2
fi

ASSET_ROOT="$(_vopd_resolve_path "${VOPD_ASSET_ROOT:-envs}")"
MODEL_DIR="$(_vopd_resolve_path "${VOPD_MODEL_PATH:-envs/models/Qwen3.5-4B}")"
WHEEL_DIR="$(_vopd_resolve_path "${VOPD_LOCAL_WHEEL_DIR:-envs/wheels/cp310-aarch64}")"
PREPARE_RUNTIME_DIR="${ASSET_ROOT}/runtime/asset-preparer"
MODEL_MANIFEST="${MODEL_DIR}/.vision_opd_model_manifest.json"

mkdir -p \
    "$ASSET_ROOT" \
    "$MODEL_DIR" \
    "$WHEEL_DIR" \
    "${ASSET_ROOT}/cache/huggingface" \
    "${ASSET_ROOT}/cache/vllm" \
    "${ASSET_ROOT}/cache/torch" \
    "${ASSET_ROOT}/cache/pip" \
    "${ASSET_ROOT}/runtime"

_vopd_select_python() {
    local candidate candidate_path
    if [[ -n "${VOPD_BOOTSTRAP_PYTHON:-}" ]]; then
        if [[ ! -x "$VOPD_BOOTSTRAP_PYTHON" ]] || \
            ! "$VOPD_BOOTSTRAP_PYTHON" -c 'import sys' >/dev/null 2>&1; then
            echo "VOPD_BOOTSTRAP_PYTHON is not a working Python: $VOPD_BOOTSTRAP_PYTHON" >&2
            return 1
        fi
        printf '%s\n' "$VOPD_BOOTSTRAP_PYTHON"
        return 0
    fi
    for candidate in python3.10 python3 python; do
        candidate_path="$(command -v "$candidate" 2>/dev/null || true)"
        [[ -n "$candidate_path" ]] || continue
        if "$candidate_path" -c 'import sys' >/dev/null 2>&1; then
            printf '%s\n' "$candidate_path"
            return 0
        fi
    done
    echo "Python is required to prepare and validate Ascend assets." >&2
    return 1
}

PREPARE_PYTHON="$(_vopd_select_python)"

_vopd_write_model_manifest() {
    local resolved_revision="$1"
    local source_kind="$2"
    "$PREPARE_PYTHON" - \
        "$MODEL_MANIFEST" \
        "$VOPD_MODEL_REPO_ID" \
        "$VOPD_MODEL_REVISION" \
        "$resolved_revision" \
        "$source_kind" <<'PY'
import json
import os
import pathlib
import sys

path, repo_id, requested, resolved, source = sys.argv[1:]
manifest_path = pathlib.Path(path)
temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
payload = {
    "schema": 1,
    "repo_id": repo_id,
    "requested_revision": requested,
    "resolved_revision": resolved,
    "source": source,
}
temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary_path, manifest_path)
PY
}

_vopd_require_cp310_aarch64() {
    if ! "$PREPARE_PYTHON" -c \
        'import platform, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) and platform.machine().lower() in {"aarch64", "arm64"} else 1)'; then
        echo "Wheel preparation must run with Python 3.10 on aarch64." >&2
        echo "WebStudio x86_64/Python 3.9 cannot resolve the Atlas 910B wheelhouse." >&2
        return 1
    fi
}

echo "Vision-OPD portable asset preparation"
echo "  project_root:       $PROJECT_ROOT"
echo "  asset_root:         $ASSET_ROOT"
echo "  model_dir:          $MODEL_DIR"
echo "  wheel_dir:          $WHEEL_DIR"
echo "  python:             $PREPARE_PYTHON ($($PREPARE_PYTHON --version 2>&1))"
echo "  network_enabled:    $ONLINE"
echo "  check_only:         $CHECK_ONLY"

# Fail before a potentially large model download when this is the wrong host.
if [[ "$SKIP_WHEELS" != "1" && "$CHECK_ONLY" != "1" ]]; then
    _vopd_require_cp310_aarch64
fi

if [[ "$SKIP_MODEL" != "1" && "$CHECK_ONLY" != "1" ]]; then
    if [[ -n "$MODEL_SOURCE_DIR" ]]; then
        MODEL_SOURCE_DIR="$(_vopd_resolve_path "$MODEL_SOURCE_DIR")"
        "$PREPARE_PYTHON" "$PROJECT_ROOT/scripts/check_ascend_assets.py" \
            --project-root "$PROJECT_ROOT" \
            --model-dir "$MODEL_SOURCE_DIR"
        if [[ "$(cd "$MODEL_SOURCE_DIR" && pwd)" != "$(cd "$MODEL_DIR" && pwd)" ]]; then
            echo "Copying the validated model snapshot from $MODEL_SOURCE_DIR ..."
            cp -a "$MODEL_SOURCE_DIR"/. "$MODEL_DIR"/
        fi
        _vopd_write_model_manifest "$VOPD_MODEL_REVISION" "local-copy"
    elif "$PREPARE_PYTHON" "$PROJECT_ROOT/scripts/check_ascend_assets.py" \
        --project-root "$PROJECT_ROOT" \
        --model-dir "$MODEL_DIR" \
        --expected-model-repo-id "$VOPD_MODEL_REPO_ID" \
        --expected-model-revision "$VOPD_MODEL_REVISION" \
        --require-manifests >/dev/null 2>&1; then
        echo "Complete model snapshot already exists; download skipped."
    elif [[ -f "$MODEL_MANIFEST" ]]; then
        "$PREPARE_PYTHON" "$PROJECT_ROOT/scripts/check_ascend_assets.py" \
            --project-root "$PROJECT_ROOT" \
            --model-dir "$MODEL_DIR" \
            --expected-model-repo-id "$VOPD_MODEL_REPO_ID" \
            --expected-model-revision "$VOPD_MODEL_REVISION" \
            --require-manifests
        exit 1
    elif "$PREPARE_PYTHON" "$PROJECT_ROOT/scripts/check_ascend_assets.py" \
        --project-root "$PROJECT_ROOT" --model-dir "$MODEL_DIR" >/dev/null 2>&1; then
        echo "Recording the configured revision for the existing model snapshot..."
        _vopd_write_model_manifest "$VOPD_MODEL_REVISION" "existing-local-snapshot"
    elif [[ "$ONLINE" == "1" ]]; then
        HUB_PYTHON="$PREPARE_PYTHON"
        if ! "$HUB_PYTHON" -c 'import huggingface_hub' >/dev/null 2>&1; then
            echo "Creating the isolated Hugging Face asset-preparer environment..."
            if [[ ! -x "$PREPARE_RUNTIME_DIR/bin/python" ]]; then
                "$PREPARE_PYTHON" -m venv "$PREPARE_RUNTIME_DIR"
            fi
            HUB_PYTHON="$PREPARE_RUNTIME_DIR/bin/python"
            _vopd_hub_pip_args=(
                --disable-pip-version-check
                --no-input
                --index-url "${VOPD_PREPARE_PIP_INDEX_URL}"
            )
            if [[ -n "${VOPD_PREPARE_PIP_EXTRA_INDEX_URL:-}" ]]; then
                _vopd_hub_pip_args+=(--extra-index-url "$VOPD_PREPARE_PIP_EXTRA_INDEX_URL")
            fi
            if [[ -n "${VOPD_PIP_TRUSTED_HOST:-}" ]]; then
                _vopd_hub_pip_args+=(--trusted-host "$VOPD_PIP_TRUSTED_HOST")
            fi
            "$HUB_PYTHON" -m pip install "${_vopd_hub_pip_args[@]}" \
                'huggingface_hub>=0.32,<2'
        fi
        echo "Downloading ${VOPD_MODEL_REPO_ID}@${VOPD_MODEL_REVISION} ..."
        "$HUB_PYTHON" - "$VOPD_MODEL_REPO_ID" "$VOPD_MODEL_REVISION" "$MODEL_DIR" <<'PY'
import json
import os
import pathlib
import sys
from huggingface_hub import HfApi, snapshot_download

repo_id, revision, local_dir = sys.argv[1:]
resolved_revision = HfApi(token=os.environ.get("HF_TOKEN") or None).model_info(
    repo_id=repo_id,
    revision=revision,
).sha
path = snapshot_download(
    repo_id=repo_id,
    revision=resolved_revision,
    local_dir=local_dir,
    token=os.environ.get("HF_TOKEN") or None,
)
manifest_path = pathlib.Path(local_dir) / ".vision_opd_model_manifest.json"
temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
temporary_path.write_text(
    json.dumps(
        {
            "schema": 1,
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "source": "huggingface-snapshot-download",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
os.replace(temporary_path, manifest_path)
print(f"Model snapshot ready: {path}")
print(f"Resolved model revision: {resolved_revision}")
PY
    fi
fi

_vopd_download_file() {
    local url="$1"
    local destination="$2"
    [[ -s "$destination" ]] && return 0
    echo "Downloading $(basename "$destination") ..."
    "$PREPARE_PYTHON" - "$url" "$destination" "${VOPD_PIP_TIMEOUT:-120}" <<'PY'
import os
import pathlib
import shutil
import sys
import urllib.request

url, destination, timeout = sys.argv[1], pathlib.Path(sys.argv[2]), float(sys.argv[3])
temporary = destination.with_name(destination.name + ".part")
destination.parent.mkdir(parents=True, exist_ok=True)
try:
    with urllib.request.urlopen(url, timeout=timeout) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size == 0:
        raise RuntimeError(f"empty response from {url}")
    os.replace(temporary, destination)
finally:
    if temporary.exists():
        temporary.unlink()
PY
}

if [[ "$SKIP_WHEELS" != "1" && "$CHECK_ONLY" != "1" ]]; then
    declare -a RESOLVED_INTERNAL_WHEEL_DIRS=()
    for _vopd_internal_dir in "${INTERNAL_WHEEL_DIRS[@]}"; do
        [[ -n "$_vopd_internal_dir" ]] || continue
        _vopd_internal_dir="$(_vopd_resolve_path "$_vopd_internal_dir")"
        if [[ ! -d "$_vopd_internal_dir" ]]; then
            echo "Platform-internal wheel directory does not exist: $_vopd_internal_dir" >&2
            exit 2
        fi
        RESOLVED_INTERNAL_WHEEL_DIRS+=("$_vopd_internal_dir")
    done

    if [[ "$ONLINE" == "1" ]]; then
        _vopd_download_file \
            'https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/torch_npu-2.9.0.post1%2Bgit4c901a4-cp310-cp310-manylinux_2_28_aarch64.whl' \
            "$WHEEL_DIR/torch_npu-2.9.0.post1+git4c901a4-cp310-cp310-manylinux_2_28_aarch64.whl"
        _vopd_download_file \
            'https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/triton_ascend-3.2.0.dev20260322-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl' \
            "$WHEEL_DIR/triton_ascend-3.2.0.dev20260322-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
    fi

    _vopd_download_args=(
        --disable-pip-version-check
        --no-input
        --timeout "${VOPD_PIP_TIMEOUT:-120}"
        --retries "${VOPD_PIP_RETRIES:-5}"
        --dest "$WHEEL_DIR"
        --only-binary=:all:
        --find-links "$WHEEL_DIR"
    )
    for _vopd_internal_dir in "${RESOLVED_INTERNAL_WHEEL_DIRS[@]}"; do
        _vopd_download_args+=(--find-links "$_vopd_internal_dir")
    done
    if [[ "$ONLINE" == "1" ]]; then
        _vopd_download_args+=(--index-url "$VOPD_PREPARE_PIP_INDEX_URL")
        if [[ -n "${VOPD_PREPARE_PIP_EXTRA_INDEX_URL:-}" ]]; then
            _vopd_download_args+=(--extra-index-url "$VOPD_PREPARE_PIP_EXTRA_INDEX_URL")
        fi
        if [[ -n "${VOPD_PIP_TRUSTED_HOST:-}" ]]; then
            _vopd_download_args+=(--trusted-host "$VOPD_PIP_TRUSTED_HOST")
        fi
    else
        _vopd_download_args+=(--no-index)
    fi

    echo "Resolving and collecting the complete Python 3.10/aarch64 wheelhouse..."
    "$PREPARE_PYTHON" -m pip download \
        "${_vopd_download_args[@]}" \
        --constraint "$PROJECT_ROOT/requirements-ascend-core.txt" \
        -r "$PROJECT_ROOT/requirements-ascend-core.txt" \
        -r "$PROJECT_ROOT/requirements-ascend.txt" \
        'pip>=23.3,<26' setuptools wheel

    "$PREPARE_PYTHON" -m pip download \
        "${_vopd_download_args[@]}" \
        --no-deps \
        -r "$PROJECT_ROOT/requirements-ascend-plugins.txt"

    # This manifest is written only after pip has successfully resolved the
    # full generic/core dependency closure and collected both hardware plugins.
    "$PREPARE_PYTHON" "$PROJECT_ROOT/scripts/check_ascend_assets.py" \
        --project-root "$PROJECT_ROOT" \
        --wheel-dir "$WHEEL_DIR" \
        --write-wheel-manifest
fi

_vopd_check_args=(--project-root "$PROJECT_ROOT")
if [[ "$SKIP_MODEL" != "1" ]]; then
    _vopd_check_args+=(
        --model-dir "$MODEL_DIR"
        --expected-model-repo-id "$VOPD_MODEL_REPO_ID"
        --expected-model-revision "$VOPD_MODEL_REVISION"
    )
fi
if [[ "$SKIP_WHEELS" != "1" ]]; then
    _vopd_check_args+=(--wheel-dir "$WHEEL_DIR")
fi
_vopd_check_args+=(--require-manifests)
"$PREPARE_PYTHON" "$PROJECT_ROOT/scripts/check_ascend_assets.py" "${_vopd_check_args[@]}"

echo "Vision-OPD portable assets are ready."
echo "Next command on the Atlas 910B worker:"
echo "  bash scripts/start_vision_opd_ascend.sh"
