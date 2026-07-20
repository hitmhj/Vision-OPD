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
directories. Relative paths are resolved from the repository root. On an
x86_64/Python 3.9 WebStudio host, pip is automatically placed in cross-target
mode for the configured CPython 3.10/3.11 aarch64 target; the generated venv is still created only later
on the real NPU worker.
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
TARGET_PYTHON="${VOPD_PREPARE_TARGET_PYTHON:-3.11}"
case "$TARGET_PYTHON" in
    3.10|3.11) ;;
    *)
        echo "VOPD_PREPARE_TARGET_PYTHON must be 3.10 or 3.11; got $TARGET_PYTHON" >&2
        exit 2
        ;;
esac
TARGET_PYTHON_COMPACT="${TARGET_PYTHON/./}"
TARGET_PYTHON_TAG="cp${TARGET_PYTHON_COMPACT}"
if [[ -n "${VOPD_LOCAL_WHEEL_DIR:-}" ]]; then
    WHEEL_DIR="$(_vopd_resolve_path "$VOPD_LOCAL_WHEEL_DIR")"
else
    WHEEL_DIR="$(_vopd_resolve_path "${VOPD_WHEEL_ROOT:-envs/wheels}/${TARGET_PYTHON_TAG}-aarch64")"
fi
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
    for candidate in python3.11 python3.10 python3 python; do
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

PREPARE_HOST_ARCH="$($PREPARE_PYTHON -c 'import platform; print(platform.machine().lower())')"
PREPARE_HOST_PYTHON="$($PREPARE_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PREPARE_MODE="cross"
declare -a PIP_TARGET_ARGS=(
    --platform manylinux_2_17_aarch64
    --platform manylinux2014_aarch64
    --platform manylinux_2_24_aarch64
    --platform manylinux_2_27_aarch64
    --platform manylinux_2_28_aarch64
    --platform manylinux_2_31_aarch64
    --platform linux_aarch64
    --python-version "$TARGET_PYTHON"
    --implementation cp
    --abi "$TARGET_PYTHON_TAG"
)
if [[ "$PREPARE_HOST_PYTHON" == "$TARGET_PYTHON" && \
      ("$PREPARE_HOST_ARCH" == "aarch64" || "$PREPARE_HOST_ARCH" == "arm64") ]]; then
    PREPARE_MODE="native"
    PIP_TARGET_ARGS=()
fi

if [[ "$SKIP_WHEELS" != "1" && "$CHECK_ONLY" != "1" ]]; then
    if ! "$PREPARE_PYTHON" -m pip --version >/dev/null 2>&1; then
        echo "pip is required to resolve the portable wheelhouse." >&2
        exit 2
    fi
    if ! "$PREPARE_PYTHON" -c '
import importlib.metadata
import re
version = importlib.metadata.version("pip")
parts = tuple(int(part) for part in re.match(r"\d+(?:\.\d+)*", version).group().split("."))
raise SystemExit(0 if parts >= (23, 3) else 1)
'; then
        echo "Asset preparation requires pip>=23.3; found $($PREPARE_PYTHON -m pip --version)." >&2
        exit 2
    fi
fi

echo "Vision-OPD portable asset preparation"
echo "  project_root:       $PROJECT_ROOT"
echo "  asset_root:         $ASSET_ROOT"
echo "  model_dir:          $MODEL_DIR"
echo "  wheel_dir:          $WHEEL_DIR"
echo "  python:             $PREPARE_PYTHON ($($PREPARE_PYTHON --version 2>&1))"
echo "  host_arch:          $PREPARE_HOST_ARCH"
echo "  wheel_target:       CPython ${TARGET_PYTHON}/aarch64 (${TARGET_PYTHON_TAG})"
echo "  resolution_mode:    $PREPARE_MODE"
echo "  network_enabled:    $ONLINE"
echo "  check_only:         $CHECK_ONLY"

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
    "$PREPARE_PYTHON" - "$url" "$destination" "${VOPD_PIP_TIMEOUT:-120}" "${VOPD_PIP_RETRIES:-5}" <<'PY'
import os
import pathlib
import shutil
import sys
import time
import urllib.request

url = sys.argv[1]
destination = pathlib.Path(sys.argv[2])
timeout = float(sys.argv[3])
retries = int(sys.argv[4])
temporary = destination.with_name(destination.name + ".part")
destination.parent.mkdir(parents=True, exist_ok=True)
last_error = None
for attempt in range(retries + 1):
    try:
        existing = temporary.stat().st_size if temporary.exists() else 0
        request = urllib.request.Request(url)
        if existing:
            request.add_header("Range", f"bytes={existing}-")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            resumed = existing > 0 and getattr(response, "status", None) == 206
            mode = "ab" if resumed else "wb"
            with temporary.open(mode) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"empty response from {url}")
        os.replace(temporary, destination)
        break
    except Exception as exc:
        last_error = exc
        if attempt >= retries:
            raise RuntimeError(
                f"download failed after {retries + 1} attempts; partial file kept at {temporary}: {exc}"
            ) from exc
        delay = min(2 ** attempt, 15)
        print(f"Download attempt {attempt + 1} failed: {exc}; retrying in {delay}s", file=sys.stderr)
        time.sleep(delay)
else:
    raise RuntimeError(last_error)
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

    # Hydra 1.3.2 and OmegaConf 2.3.0 require ANTLR 4.9.x. Upstream publishes
    # antlr4-python3-runtime 4.9.3 only as an sdist, so a binary-only
    # cross-targeted pip download cannot select it. Build this architecture-
    # independent wheel once on the online preparation host; the NPU worker
    # still installs exclusively from the completed offline wheelhouse.
    if [[ -z "$(find "$WHEEL_DIR" -maxdepth 1 -type f \
            -iname 'antlr4_python3_runtime-4.9.3-py3-none-any.whl' \
            -print -quit)" && "$ONLINE" == "1" ]]; then
        echo "Building the pure-Python ANTLR 4.9.3 runtime wheel..."
        _vopd_antlr_wheel_args=(
            --disable-pip-version-check
            --no-input
            --timeout "${VOPD_PIP_TIMEOUT:-120}"
            --retries "${VOPD_PIP_RETRIES:-5}"
            --wheel-dir "$WHEEL_DIR"
            --no-deps
            --index-url "$VOPD_PREPARE_PIP_INDEX_URL"
        )
        if [[ -n "${VOPD_PREPARE_PIP_EXTRA_INDEX_URL:-}" ]]; then
            _vopd_antlr_wheel_args+=(--extra-index-url "$VOPD_PREPARE_PIP_EXTRA_INDEX_URL")
        fi
        if [[ -n "${VOPD_PIP_TRUSTED_HOST:-}" ]]; then
            _vopd_antlr_wheel_args+=(--trusted-host "$VOPD_PIP_TRUSTED_HOST")
        fi
        "$PREPARE_PYTHON" -m pip wheel \
            "${_vopd_antlr_wheel_args[@]}" \
            'antlr4-python3-runtime==4.9.3'
    fi

    if [[ "$ONLINE" == "1" ]]; then
        _vopd_torch_npu_wheel="torch_npu-2.9.0.post1+git4c901a4-${TARGET_PYTHON_TAG}-${TARGET_PYTHON_TAG}-manylinux_2_28_aarch64.whl"
        _vopd_triton_wheel="triton_ascend-3.2.0.dev20260322-${TARGET_PYTHON_TAG}-${TARGET_PYTHON_TAG}-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
        _vopd_download_file \
            "https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/${_vopd_torch_npu_wheel/+/%2B}" \
            "$WHEEL_DIR/${_vopd_torch_npu_wheel}"
        _vopd_download_file \
            "https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/${_vopd_triton_wheel}" \
            "$WHEEL_DIR/${_vopd_triton_wheel}"
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
    _vopd_download_args+=("${PIP_TARGET_ARGS[@]}")
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
        if [[ ${#RESOLVED_INTERNAL_WHEEL_DIRS[@]} -eq 0 && \
              -z "$(find "$WHEEL_DIR" -maxdepth 1 -type f -name '*.whl' -print -quit)" ]]; then
            echo "Offline preparation has no wheel source." >&2
            echo "Set VOPD_INTERNAL_WHEEL_DIRS or copy ${TARGET_PYTHON_TAG}/aarch64 wheels into $WHEEL_DIR." >&2
            exit 2
        fi
    fi

    echo "Resolving and collecting the complete Python ${TARGET_PYTHON}/aarch64 wheelhouse..."
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
        --python-version "$TARGET_PYTHON" \
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
    _vopd_check_args+=(--wheel-dir "$WHEEL_DIR" --python-version "$TARGET_PYTHON")
fi
_vopd_check_args+=(--require-manifests)
"$PREPARE_PYTHON" "$PROJECT_ROOT/scripts/check_ascend_assets.py" "${_vopd_check_args[@]}"

echo "Vision-OPD portable assets are ready."
echo "Next command on the Atlas 910B worker:"
echo "  bash scripts/start_vision_opd_ascend.sh"
