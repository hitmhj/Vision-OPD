#!/usr/bin/env bash

# This file must be sourced. Keep the vendor initialization semantics aligned
# with prompt.txt: Huawei set_env.sh files run without Bash nounset, in the
# explicit CANN -> ATB -> ASDSIP order, and exactly once per lifecycle.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file instead: source scripts/ascend_env.sh" >&2
    exit 2
fi

if [[ "${VOPD_ASCEND_ENV_READY:-0}" == "1" ]]; then
    return 0
fi

# Huawei's ATB script reads variables such as ZSH_VERSION without a default.
# prompt.txt uses `set -e`, not nounset; preserve that contract even when this
# loader is called from a stricter parent shell. Do not restore nounset here:
# restoring it before a nested vendor source completes recreates the failure.
set +u

# Some Huawei NNAL releases read ZSH_VERSION directly even while running under
# Bash. Defining it is stronger than relying on nounset being disabled and also
# protects us if a nested vendor script enables nounset internally.
export ZSH_VERSION="${ZSH_VERSION-}"

if [[ -z "${CANN_ENV_SCRIPT:-}" || ! -f "$CANN_ENV_SCRIPT" ]]; then
    echo "Configured CANN environment script does not exist: ${CANN_ENV_SCRIPT:-unset}" >&2
    return 1
fi
if [[ -z "${NNAL_ENV_SCRIPT:-}" || ! -f "$NNAL_ENV_SCRIPT" ]]; then
    echo "Configured NNAL/ATB environment script does not exist: ${NNAL_ENV_SCRIPT:-unset}" >&2
    return 1
fi
if [[ "${VOPD_REQUIRE_ASDSIP:-0}" == "1" ]]; then
    if [[ -z "${ASDSIP_ENV_SCRIPT:-}" || ! -f "$ASDSIP_ENV_SCRIPT" ]]; then
        echo "Required ASDSIP environment script does not exist: ${ASDSIP_ENV_SCRIPT:-unset}" >&2
        return 1
    fi
fi

# These are the three active source commands from prompt.txt. There is
# intentionally no /usr/local fallback: silently switching stacks can mix CANN
# and NNAL versions and was the path that triggered the ZSH_VERSION failure.
# shellcheck disable=SC1090
source "$CANN_ENV_SCRIPT"
set +u
# shellcheck disable=SC1090
source "$NNAL_ENV_SCRIPT" --cxx_abi=0
set +u
if [[ -n "${ASDSIP_ENV_SCRIPT:-}" && -f "$ASDSIP_ENV_SCRIPT" ]]; then
    # shellcheck disable=SC1090
    source "$ASDSIP_ENV_SCRIPT"
    set +u
else
    echo "ASDSIP environment script not found; optional component skipped."
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export TASK_QUEUE=0
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1

export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=64000
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=18000
export HCCL_EXEC_TIMEOT=3600
export HCCL_CONNECT_TIMEOT=3600

export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VOPD_ASCEND_ENV_READY=1
