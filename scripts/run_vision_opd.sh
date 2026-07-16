#!/bin/bash

set -euo pipefail

# =============================================================================
# Vision-OPD Training Script
# Paper: Vision-OPD: Learning to See Fine Details for Multimodal LLMs
#        via On-Policy Self-Distillation
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_NAME="${VOPD_CONFIG_NAME:-${CONFIG_NAME:-vopd}}"
MODEL_PATH="${VOPD_MODEL_PATH:-${MODEL_PATH:-Qwen/Qwen3.5-4B}}"
TARGET_DEVICE="${TARGET_DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEACHER_MODEL_SOURCE="${VOPD_TEACHER_MODEL_SOURCE:-legacy}"
TEACHER_REGULARIZATION="${VOPD_TEACHER_REGULARIZATION:-ema}"
TEACHER_UPDATE_RATE="${VOPD_TEACHER_UPDATE_RATE:-0.05}"

TRAIN_BATCH_SIZE="${VOPD_TRAIN_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-96}}"
PPO_MIMI_BATCH_SIZE="${VOPD_PPO_MINI_BATCH_SIZE:-${PPO_MIMI_BATCH_SIZE:-96}}"
ROLLOUT_N="${VOPD_ROLLOUT_N:-${ROLLOUT_N:-8}}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${VOPD_ROLLOUT_TP_SIZE:-${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}}"
LR="${VOPD_LR:-${LR:-2e-6}}"
DONT_REPROMPT_ON_SELF_SUCCESS="${VOPD_DONT_REPROMPT_ON_SELF_SUCCESS:-True}"
ALPHA="${VOPD_DISTILL_ALPHA:-0.5}"
MAX_PROMPT_LENGTH="${VOPD_MAX_PROMPT_LENGTH:-${MAX_PROMPT_LENGTH:-8192}}"
MAX_RESPONSE_LENGTH="${VOPD_MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LENGTH:-1024}}"
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${VOPD_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${VOPD_ROLLOUT_MEMORY_UTILIZATION:-${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}}"
ACTOR_USE_DYNAMIC_BSZ="${VOPD_USE_DYNAMIC_BATCH:-${ACTOR_USE_DYNAMIC_BSZ:-True}}"
PPO_MAX_TOKEN_LEN_PER_GPU="${VOPD_MAX_TOKENS_PER_NPU:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${VOPD_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-1}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${VOPD_REF_LOGPROB_MICRO_BATCH_SIZE:-1}"
ACTOR_PARAM_OFFLOAD="${VOPD_ACTOR_PARAM_OFFLOAD:-${ACTOR_PARAM_OFFLOAD:-True}}"
ACTOR_OPTIMIZER_OFFLOAD="${VOPD_ACTOR_OPTIMIZER_OFFLOAD:-${ACTOR_OPTIMIZER_OFFLOAD:-True}}"
REF_PARAM_OFFLOAD="${VOPD_TEACHER_PARAM_OFFLOAD:-${REF_PARAM_OFFLOAD:-True}}"
TRAINER_N_GPUS_PER_NODE="${VOPD_GPUS_PER_NODE:-${TRAINER_N_GPUS_PER_NODE:-8}}"
# WORLD_SIZE normally means the total process count, not the node count. Use
# TRAINER_NNODES explicitly for multi-node jobs and default to a single node.
TRAINER_NNODES="${VOPD_NNODES:-${TRAINER_NNODES:-1}}"
TRAINER_SAVE_FREQ="${VOPD_SAVE_FREQ:--1}"
TRAINER_TOTAL_EPOCHS="${VOPD_TOTAL_EPOCHS:-1}"
TRAINER_TOTAL_TRAINING_STEPS="${VOPD_TOTAL_TRAINING_STEPS:-null}"
TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${VOPD_MAX_CHECKPOINTS:-null}"
TRAINER_RESUME_MODE="${VOPD_RESUME_MODE:-auto}"
TRAINER_RESUME_FROM_PATH="${VOPD_RESUME_FROM_PATH:-null}"
TRAINER_LOGGER="${VOPD_LOGGER:-[\"console\",\"tensorboard\"]}"
ROLLOUT_AGENT_NUM_WORKERS="${VOPD_ROLLOUT_WORKERS:-${ROLLOUT_AGENT_NUM_WORKERS:-8}}"
DATA_DATALOADER_NUM_WORKERS="${VOPD_DATALOADER_WORKERS:-${DATA_DATALOADER_NUM_WORKERS:-8}}"
CUSTOM_CHAT_TEMPLATE_FILE="${VOPD_CHAT_TEMPLATE:-${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja}"

SELF_DISTILL_TOPK="${VOPD_DISTILL_TOPK:-100}"
SELF_DISTILL_MAX_REPROMPT_LEN="${VOPD_MAX_REPROMPT_LENGTH:-10240}"
SELF_DISTILL_CLIP="${VOPD_DISTILL_CLIP:-2.0}"
ROLLOUT_IS_THRESHOLD="${VOPD_ROLLOUT_IS_THRESHOLD:-2.0}"
ACTOR_CLIP_RATIO_HIGH="${VOPD_CLIP_RATIO_HIGH:-0.3}"
ACTOR_CLIP_RATIO_LOW="${VOPD_CLIP_RATIO_LOW:-0.2}"
LR_WARMUP_STEPS="${VOPD_LR_WARMUP_STEPS:-10}"

# --- Data Paths ---
DATA_DIR="${VOPD_DATA_DIR:-${PROJECT_ROOT}/data}"
TASK_TRAIN_FILE="${VOPD_TRAIN_FILE:-${DATA_DIR}/train.parquet}"

MODEL_NAME=$(basename "$MODEL_PATH")
EXPERIMENT_NAME="${VOPD_EXPERIMENT_NAME:-Vision-OPD-${MODEL_NAME}}"
PROJECT_NAME="${VOPD_PROJECT_NAME:-Vision-OPD}"
TRAINER_DEFAULT_LOCAL_DIR="${VOPD_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
TRAINER_ROLLOUT_DATA_DIR="${VOPD_ROLLOUT_DIR:-${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}}"
mkdir -p "$TRAINER_ROLLOUT_DATA_DIR"

EXTRA_ARGS=("$@")

# =============================================================================
# ENVIRONMENT
# =============================================================================
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONBUFFERED=1
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
ulimit -c 0

DEVICE_ARGS=()
ROLLOUT_ENGINE_ARGS=()
MODEL_USE_REMOVE_PADDING=True

case "${TARGET_DEVICE,,}" in
    ascend|npu)
        # Conservative bring-up settings for Ascend 910B. Optimized graph,
        # fused-kernel and remove-padding paths can be enabled after the target
        # CANN/torch_npu stack passes model-level forward/backward validation.
        export VLLM_USE_V1=1
        export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"
        MODEL_USE_REMOVE_PADDING=False
        DEVICE_ARGS+=(
            trainer.device=npu
            actor_rollout_ref.actor.use_torch_compile=False
            actor_rollout_ref.ref.use_torch_compile=False
            actor_rollout_ref.model.use_fused_kernels=False
            +actor_rollout_ref.model.override_config.attn_implementation=sdpa
            actor_rollout_ref.rollout.enforce_eager=True
            actor_rollout_ref.rollout.load_format=safetensors
        )
        ;;
    cuda|gpu)
        unset VLLM_ATTENTION_BACKEND
        export VLLM_USE_V1=1
        ROLLOUT_ENGINE_ARGS+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False
            +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False
        )
        ;;
    *)
        echo "Unsupported TARGET_DEVICE=${TARGET_DEVICE}; expected cuda, ascend, or npu." >&2
        exit 2
        ;;
esac

CHAT_TEMPLATE_ARGS=()
if [[ -n "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
    if [[ ! -f "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
        echo "Custom chat template file not found: ${CUSTOM_CHAT_TEMPLATE_FILE}" >&2
        exit 1
    fi
    CHAT_TEMPLATE_ARGS+=(actor_rollout_ref.model.custom_chat_template_file="$CUSTOM_CHAT_TEMPLATE_FILE")
fi

echo "Running: $EXPERIMENT_NAME"
echo "Teacher model source: $TEACHER_MODEL_SOURCE"
echo "Teacher regularization: $TEACHER_REGULARIZATION"
echo "Teacher update rate: $TEACHER_UPDATE_RATE"
echo "Target device: $TARGET_DEVICE"

"$PYTHON_BIN" -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    data.train_files="[\"$TASK_TRAIN_FILE\"]" \
    data.val_files="[]" \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=$DATA_DATALOADER_NUM_WORKERS \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=$MODEL_USE_REMOVE_PADDING \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MIMI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.clip_ratio_high=$ACTOR_CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.clip_ratio_low=$ACTOR_CLIP_RATIO_LOW \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=$SELF_DISTILL_TOPK \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=$SELF_DISTILL_MAX_REPROMPT_LEN \
    actor_rollout_ref.actor.self_distillation.is_clip=$SELF_DISTILL_CLIP \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REGULARIZATION \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=$ROLLOUT_IS_THRESHOLD \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    reward_model.enable=False \
    critic.model.path=$MODEL_PATH \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    trainer.project_name=$PROJECT_NAME \
    trainer.group_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$TRAINER_SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.total_training_steps=$TRAINER_TOTAL_TRAINING_STEPS \
    trainer.resume_mode=$TRAINER_RESUME_MODE \
    trainer.resume_from_path=$TRAINER_RESUME_FROM_PATH \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${DEVICE_ARGS[@]}" \
    "${ROLLOUT_ENGINE_ARGS[@]}" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
