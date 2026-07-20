#!/usr/bin/env bash

# Run exactly one complete Vision-OPD optimization step. Unlike the lightweight
# preflight, this loads Qwen3.5, executes multimodal rollout, teacher JSD/EMA,
# FSDP backward, optimizer update and FSDP-to-vLLM weight synchronization.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export VOPD_RUN_MODE=smoke

exec bash "$PROJECT_ROOT/scripts/start_vision_opd_ascend.sh" "$@"
