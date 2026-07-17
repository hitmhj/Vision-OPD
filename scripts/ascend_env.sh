#!/usr/bin/env bash

# This file must be sourced. It configures process-level Ascend variables but
# never installs or modifies the system driver/CANN installation.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file instead: source scripts/ascend_env.sh" >&2
    exit 2
fi

_vision_opd_source_if_present() {
    local env_file="$1"
    if [[ -f "$env_file" ]]; then
        # shellcheck disable=SC1090
        source "$env_file"
        return 0
    fi
    return 1
}

_vision_opd_source_vendor_script() {
    local env_file="$1"
    shift
    local restore_nounset=0
    if [[ "$-" == *u* ]]; then
        restore_nounset=1
        set +u
    fi
    # shellcheck disable=SC1090
    source "$env_file" "$@"
    local source_status=$?
    if [[ "$restore_nounset" == "1" ]]; then
        set -u
    fi
    return "$source_status"
}

if [[ -n "${CANN_ENV_SCRIPT:-}" ]]; then
    if [[ ! -f "$CANN_ENV_SCRIPT" ]]; then
        echo "CANN_ENV_SCRIPT does not exist: $CANN_ENV_SCRIPT" >&2
        return 1
    fi
    _vision_opd_source_vendor_script "$CANN_ENV_SCRIPT"
elif [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    _vision_opd_source_if_present /usr/local/Ascend/ascend-toolkit/set_env.sh \
        || _vision_opd_source_if_present /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
        || {
            echo "CANN environment was not found. Set CANN_ENV_SCRIPT or source CANN manually." >&2
            return 1
        }
fi

if [[ -n "${NNAL_ENV_SCRIPT:-}" ]]; then
    if [[ ! -f "$NNAL_ENV_SCRIPT" ]]; then
        echo "NNAL_ENV_SCRIPT does not exist: $NNAL_ENV_SCRIPT" >&2
        return 1
    fi
    _vision_opd_source_vendor_script "$NNAL_ENV_SCRIPT" --cxx_abi=0
else
    _vision_opd_source_if_present /usr/local/Ascend/nnal/atb/set_env.sh || true
fi

if [[ -n "${ASDSIP_ENV_SCRIPT:-}" ]]; then
    if [[ ! -f "$ASDSIP_ENV_SCRIPT" ]]; then
        echo "ASDSIP_ENV_SCRIPT does not exist: $ASDSIP_ENV_SCRIPT" >&2
        return 1
    fi
    _vision_opd_source_vendor_script "$ASDSIP_ENV_SCRIPT"
fi

export TARGET_DEVICE=ascend
export VLLM_USE_V1=1
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export TASK_QUEUE="${TASK_QUEUE:-0}"
export COMBINED_ENABLE="${COMBINED_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-64000}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-18000}"
export HCCL_EXEC_TIMEOT="${HCCL_EXEC_TIMEOT:-3600}"
export HCCL_CONNECT_TIMEOT="${HCCL_CONNECT_TIMEOT:-3600}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}"
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

unset CUDA_VISIBLE_DEVICES
unset VLLM_ATTENTION_BACKEND

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    _vision_opd_npu_count="${TRAINER_N_GPUS_PER_NODE:-8}"
    if ! [[ "$_vision_opd_npu_count" =~ ^[1-9][0-9]*$ ]]; then
        echo "TRAINER_N_GPUS_PER_NODE must be a positive integer." >&2
        return 1
    fi
    ASCEND_RT_VISIBLE_DEVICES="$(seq -s, 0 $((_vision_opd_npu_count - 1)))"
    export ASCEND_RT_VISIBLE_DEVICES
fi

unset -f _vision_opd_source_if_present
unset -f _vision_opd_source_vendor_script
unset _vision_opd_npu_count
