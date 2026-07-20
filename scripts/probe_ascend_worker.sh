#!/usr/bin/env bash

# Best-effort, non-mutating worker fingerprint. It deliberately does not source
# NNAL/ATB, install packages, create a venv, or require a particular Python.
set +u
set +e

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

_vopd_probe_value() {
    local label="$1"
    shift
    local value
    value="$("$@" 2>&1)"
    [[ -n "$value" ]] || value="unavailable"
    printf '  %-22s %s\n' "${label}:" "$value"
}

_vopd_probe_cann_version() {
    local candidate detected
    for candidate in \
        "${ASCEND_HOME_PATH:-}/version.cfg" \
        "${ASCEND_HOME_PATH:-}/../ascend_toolkit_install.info" \
        "$(dirname "${CANN_ENV_SCRIPT:-/nonexistent}")/version.cfg" \
        "$(dirname "$(dirname "${CANN_ENV_SCRIPT:-/nonexistent}")")/version.cfg" \
        /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
        "/usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info" \
        /etc/Ascend/ascend_cann_install.info; do
        [[ -f "$candidate" ]] || continue
        detected="$(grep -Eio '[0-9]+\.[0-9]+([.][0-9]+|[.]?RC[0-9]+)' "$candidate" | head -n 1)"
        if [[ -n "$detected" ]]; then
            printf '%s (%s)\n' "$detected" "$candidate"
            return 0
        fi
    done
    printf 'unknown\n'
}

echo "Vision-OPD Ascend worker fingerprint"
_vopd_probe_value "timestamp" date '+%Y-%m-%d %H:%M:%S %z'
_vopd_probe_value "project_root" printf '%s' "$PROJECT_ROOT"
_vopd_probe_value "kernel" uname -sr
_vopd_probe_value "architecture" uname -m
_vopd_probe_value "glibc" sh -c 'ldd --version 2>/dev/null | head -n 1'
_vopd_probe_value "CANN detected" _vopd_probe_cann_version
printf '  %-22s %s\n' "CANN set_env:" "${CANN_ENV_SCRIPT:-unset} ($( [[ -f "${CANN_ENV_SCRIPT:-}" ]] && echo present || echo missing ))"
printf '  %-22s %s\n' "NNAL/ATB set_env:" "${NNAL_ENV_SCRIPT:-unset} ($( [[ -f "${NNAL_ENV_SCRIPT:-}" ]] && echo present || echo missing ))"
printf '  %-22s %s\n' "ASDSIP set_env:" "${ASDSIP_ENV_SCRIPT:-unset} ($( [[ -f "${ASDSIP_ENV_SCRIPT:-}" ]] && echo present || echo optional/missing ))"

echo "  Python candidates:"
_vopd_seen_pythons=""
for _vopd_name in python3.11 python3.10 python3 python; do
    _vopd_path="$(command -v "$_vopd_name" 2>/dev/null)"
    [[ -n "$_vopd_path" ]] || continue
    case ":$_vopd_seen_pythons:" in *":$_vopd_path:"*) continue ;; esac
    _vopd_seen_pythons="${_vopd_seen_pythons}:${_vopd_path}"
    _vopd_python_description="$("$_vopd_path" --version 2>&1)"
    if [[ $? -ne 0 ]]; then
        _vopd_python_description="unusable: ${_vopd_python_description}"
    fi
    printf '    %-12s %-55s %s\n' "$_vopd_name" "$_vopd_path" "$_vopd_python_description"
done

printf '  Platform ranks:\n'
for _vopd_var in RANK_ID RANK_SIZE ASCEND_DEVICE_ID ASCEND_RT_VISIBLE_DEVICES \
    MA_NUM_GPUS MA_NUM_HOSTS VC_TASK_INDEX GROUP_RANK; do
    printf '    %-28s %s\n' "${_vopd_var}=" "${!_vopd_var:-unset}"
done

_vopd_probe_value "memory" sh -c 'free -h 2>/dev/null | awk '\''NR==2 {print $2 " total, " $7 " available"}'\'''
_vopd_probe_value "/dev/shm" sh -c 'df -h /dev/shm 2>/dev/null | awk '\''NR==2 {print $2 " total, " $4 " available"}'\'''
_vopd_probe_value "project disk" sh -c 'df -h "$1" 2>/dev/null | awk '\''NR==2 {print $2 " total, " $4 " available"}'\''' sh "$PROJECT_ROOT"
if command -v npu-smi >/dev/null 2>&1; then
    echo "  npu-smi summary:"
    npu-smi info 2>&1 | sed 's/^/    /'
else
    echo "  npu-smi summary:      unavailable"
fi

exit 0
