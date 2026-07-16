#!/usr/bin/env bash

# Run exactly one complete Vision-OPD optimization step. Unlike the lightweight
# preflight, this loads Qwen3.5, executes multimodal rollout, teacher JSD/EMA,
# FSDP backward, optimizer update and FSDP-to-vLLM weight synchronization.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export VOPD_TRAIN_BATCH_SIZE="${VOPD_TRAIN_BATCH_SIZE:-8}"
export VOPD_PPO_MINI_BATCH_SIZE="${VOPD_PPO_MINI_BATCH_SIZE:-8}"
export VOPD_ROLLOUT_N="${VOPD_ROLLOUT_N:-2}"
export VOPD_MAX_PROMPT_LENGTH="${VOPD_MAX_PROMPT_LENGTH:-4096}"
export VOPD_MAX_RESPONSE_LENGTH="${VOPD_MAX_RESPONSE_LENGTH:-512}"
export VOPD_ROLLOUT_MEMORY_UTILIZATION="${VOPD_ROLLOUT_MEMORY_UTILIZATION:-0.4}"
export VOPD_SAVE_FREQ=-1
export VOPD_TOTAL_TRAINING_STEPS=1
export VOPD_RESUME_MODE=disable

exec bash "$PROJECT_ROOT/scripts/start_vision_opd_ascend.sh" "$@"
