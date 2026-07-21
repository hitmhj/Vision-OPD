#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT VOPD_PROJECT_ROOT="${PROJECT_ROOT}"
# shellcheck source=../vision_opd_ascend.env
source "${PROJECT_ROOT}/vision_opd_ascend.env"
cd "${PROJECT_ROOT}"

set +u
export ZSH_VERSION="${ZSH_VERSION-}"
for candidate in \
    "${ASCEND_TOOLKIT_HOME:+${ASCEND_TOOLKIT_HOME}/set_env.sh}" \
    "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
    "/usr/local/Ascend/latest/set_env.sh" \
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
        # shellcheck disable=SC1090
        source "${candidate}"
        break
    fi
done

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${VOPD_CACHE_DIR}/huggingface"

MODEL_DIR="${VOPD_INFER_MODEL_DIR:-}"
if [[ -z "${MODEL_DIR}" ]]; then
    latest_step=-1
    for candidate in "${VOPD_MERGED_DIR}"/global_step_*; do
        [[ -d "${candidate}" ]] || continue
        step="${candidate##*_}"
        if [[ "${step}" =~ ^[0-9]+$ && "${step}" -gt "${latest_step}" ]]; then
            latest_step="${step}"
            MODEL_DIR="${candidate}"
        fi
    done
fi
if [[ -z "${MODEL_DIR}" ]]; then
    echo "No merged checkpoint found; set VOPD_INFER_MODEL_DIR explicitly." >&2
    exit 1
fi
exec "${VOPD_PYTHON}" "${SCRIPT_DIR}/infer_vision_opd_ascend.py" --model "${MODEL_DIR}" "$@"
