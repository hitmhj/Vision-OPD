#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
export VOPD_PROJECT_ROOT="${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

# shellcheck source=../vision_opd_ascend.env
source "${PROJECT_ROOT}/vision_opd_ascend.env"

# Several Huawei environment scripts read ZSH_VERSION even under bash.  Keep
# nounset disabled while sourcing vendor scripts and define the variable first.
set +u
export ZSH_VERSION="${ZSH_VERSION-}"

source_first_existing() {
    local label="$1"
    shift
    local candidate
    for candidate in "$@"; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            echo "[env] sourcing ${label}: ${candidate}"
            # shellcheck disable=SC1090
            source "${candidate}"
            return 0
        fi
    done
    echo "[env] warning: ${label} set_env.sh not found; continuing with image environment"
    return 0
}

source_first_existing "CANN" \
    "${ASCEND_TOOLKIT_HOME:+${ASCEND_TOOLKIT_HOME}/set_env.sh}" \
    "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
    "/usr/local/Ascend/latest/set_env.sh" \
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
source_first_existing "NNAL/ATB" \
    "${ATB_HOME_PATH:+${ATB_HOME_PATH}/set_env.sh}" \
    "/usr/local/Ascend/nnal/atb/set_env.sh" \
    "/usr/local/Ascend/atb/set_env.sh"
source_first_existing "ASDSIP" \
    "${ASDSIP_HOME_PATH:+${ASDSIP_HOME_PATH}/set_env.sh}" \
    "/usr/local/Ascend/nnal/asdsip/set_env.sh" \
    "/usr/local/Ascend/asdsip/set_env.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${VOPD_CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${VOPD_CACHE_DIR}/datasets"
export TRANSFORMERS_CACHE="${VOPD_CACHE_DIR}/transformers"
export XDG_CACHE_HOME="${VOPD_CACHE_DIR}"
export PIP_NO_INDEX=1
export RAY_TMPDIR="${VOPD_RAY_TMPDIR}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export COMBINED_ENABLE="${COMBINED_ENABLE:-1}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-18000}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-64000}"
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${VOPD_CACHE_DIR}" "${VOPD_RUNTIME_DIR}" "${VOPD_CHECKPOINT_DIR}" \
    "${VOPD_ROLLOUT_DIR}" "${VOPD_MERGED_DIR}" "${VOPD_LOG_DIR}" "${RAY_TMPDIR}"

echo "[env] project=${PROJECT_ROOT}"
echo "[env] image=${MA_CONTAINER_IMAGE_URI:-unknown}"
echo "[env] python=$(${VOPD_PYTHON} --version 2>&1)"
echo "[env] ray_tmpdir=${RAY_TMPDIR}"
echo "[env] npu_asd_config=${NPU_ASD_CONFIG}"
command -v npu-smi >/dev/null 2>&1 && npu-smi info || true

if [[ "${VOPD_INSTALL_DEPS}" == "1" && "${VOPD_DRY_RUN}" != "1" ]]; then
    echo "[deps] installing non-core packages from ${VOPD_WHEELHOUSE}"
    "${VOPD_PYTHON}" -m pip install --no-index --find-links "${VOPD_WHEELHOUSE}" \
        -r "${PROJECT_ROOT}/requirements-ascend.txt"
fi

if [[ "${VOPD_DRY_RUN}" != "1" ]]; then
    "${VOPD_PYTHON}" - <<'PY'
import importlib.metadata as metadata
import platform
import re
import sys


def dist_version(*names):
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "not-installed"


def major_minor(value):
    match = re.match(r"^(\d+)\.(\d+)", str(value))
    return match.groups() if match else None


def version_warning(condition, message):
    if condition:
        print(f"[WARNING] {message}", file=sys.stderr)


torch_version = dist_version("torch")
torch_npu_version = dist_version("torch-npu", "torch_npu")
transformers_version = dist_version("transformers")
ray_version = dist_version("ray")

print(f"[runtime] platform={platform.platform()} machine={platform.machine()}")
print(f"[runtime] torch={torch_version} torch_npu={torch_npu_version}")
print(f"[runtime] transformers={transformers_version} ray={ray_version}")
print("[runtime] native torch/torch_npu import is deferred to the real training process")
version_warning(sys.version_info[:2] != (3, 11), f"validated Python is 3.11, imported {platform.python_version()}; continuing")
version_warning(major_minor(torch_version) != ("2", "6"), f"target image declares torch 2.6, installed {torch_version}; continuing")
version_warning(torch_npu_version == "not-installed", "torch-npu distribution metadata was not found; continuing")
version_warning(
    major_minor(torch_version) is not None
    and major_minor(torch_npu_version) is not None
    and major_minor(torch_version) != major_minor(torch_npu_version),
    f"torch {torch_version} and torch-npu {torch_npu_version} have different major/minor versions; continuing",
)
version_warning(
    major_minor(transformers_version) != ("5", "5"),
    f"Qwen3.5 path was validated with transformers 5.5, installed {transformers_version}; continuing",
)
version_warning(ray_version != "2.53.0", f"Ray was validated at 2.53.0, installed {ray_version}; continuing")
PY
fi

if [[ ! -f "${VOPD_TRAIN_FILE}" ]]; then
    if [[ -f "${VOPD_DATA_DIR}/train.jsonl" ]]; then
        if [[ "${VOPD_DRY_RUN}" == "1" ]]; then
            echo "[dry-run] would convert ${VOPD_DATA_DIR}/train.jsonl to parquet without downloading"
        else
            "${VOPD_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_data.py" \
                --data-dir "${VOPD_DATA_DIR}" --skip-download
        fi
    elif [[ "${VOPD_DRY_RUN}" == "1" ]]; then
        echo "[dry-run] warning: training data is not staged at ${VOPD_TRAIN_FILE}"
    else
        echo "Training data not found: ${VOPD_TRAIN_FILE}" >&2
        echo "Stage train.jsonl and images under ${VOPD_DATA_DIR}; runtime download is disabled." >&2
        exit 1
    fi
fi

if [[ ! -d "${VOPD_MODEL_DIR}" ]]; then
    if [[ "${VOPD_DRY_RUN}" == "1" ]]; then
        echo "[dry-run] warning: local model is not staged at ${VOPD_MODEL_DIR}"
    else
        echo "Local Qwen3.5 model directory not found: ${VOPD_MODEL_DIR}" >&2
        exit 1
    fi
fi

MASTER_ADDR="${VOPD_MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_OVERRIDE=()

if [[ "${VOPD_DRY_RUN}" != "1" && -n "${VOPD_RAY_ADDRESS}" ]]; then
    if [[ "${VOPD_NODE_RANK}" != "0" ]]; then
        echo "[ray] external cluster supplied; rank ${VOPD_NODE_RANK} has no driver to launch"
        exit 0
    fi
    export RAY_ADDRESS="${VOPD_RAY_ADDRESS}"
    RAY_OVERRIDE+=("+ray_kwargs.ray_init.address=auto")
elif [[ "${VOPD_DRY_RUN}" != "1" && "${VOPD_NNODES}" -gt 1 ]]; then
    RAY_CLUSTER_ADDRESS="${MASTER_ADDR}:${VOPD_MASTER_PORT}"
    if [[ "${VOPD_NODE_RANK}" == "0" ]]; then
        echo "[ray] starting head at ${RAY_CLUSTER_ADDRESS}"
        ray start --head --node-ip-address "${MASTER_ADDR}" --port "${VOPD_MASTER_PORT}" \
            --temp-dir "${RAY_TMPDIR}" --disable-usage-stats
        export RAY_ADDRESS="${RAY_CLUSTER_ADDRESS}"
        RAY_OVERRIDE+=("+ray_kwargs.ray_init.address=auto")
    else
        echo "[ray] rank ${VOPD_NODE_RANK} joining ${RAY_CLUSTER_ADDRESS}; the Ray head owns the only driver"
        exec ray start --address "${RAY_CLUSTER_ADDRESS}" --disable-usage-stats --block
    fi
fi

MAX_MODEL_LEN=$((VOPD_MAX_PROMPT_LENGTH + VOPD_MAX_RESPONSE_LENGTH))
CHAT_TEMPLATE="${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja"
LOG_FILE="${VOPD_LOG_DIR}/train-$(date +%Y%m%d-%H%M%S).log"

TRAIN_CMD=(
    "${VOPD_PYTHON}" -m verl.trainer.main_ppo --config-name vopd
    "data.train_files=[\"${VOPD_TRAIN_FILE}\"]"
    "data.val_files=[]"
    data.filter_overlong_prompts=False
    "data.max_prompt_length=${VOPD_MAX_PROMPT_LENGTH}"
    "data.max_response_length=${VOPD_MAX_RESPONSE_LENGTH}"
    data.truncation=error
    data.shuffle=True
    data.trust_remote_code=False
    data.return_multi_modal_inputs=True
    data.image_key=images
    "data.train_batch_size=${VOPD_TRAIN_BATCH_SIZE}"
    "data.dataloader_num_workers=${VOPD_DATALOADER_WORKERS}"
    "actor_rollout_ref.model.path=${VOPD_MODEL_DIR}"
    actor_rollout_ref.model.trust_remote_code=False
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    "+actor_rollout_ref.model.override_config.attn_implementation=sdpa"
    "actor_rollout_ref.model.custom_chat_template_file=${CHAT_TEMPLATE}"
    actor_rollout_ref.actor.use_torch_compile=False
    "actor_rollout_ref.actor.optim.lr=${VOPD_LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${VOPD_PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.use_dynamic_bsz=False
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LEN}"
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.clip_ratio_high=0.3
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd
    actor_rollout_ref.actor.calculate_entropy=False
    actor_rollout_ref.actor.self_distillation.distillation_topk=100
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240
    actor_rollout_ref.actor.self_distillation.is_clip=2.0
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True
    actor_rollout_ref.actor.self_distillation.teacher_model_source=legacy
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema
    "actor_rollout_ref.actor.self_distillation.teacher_update_rate=${VOPD_TEACHER_UPDATE_RATE}"
    actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True
    actor_rollout_ref.actor.self_distillation.alpha=0.5
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    "actor_rollout_ref.rollout.name=${VOPD_ROLLOUT_BACKEND}"
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    "actor_rollout_ref.rollout.n=${VOPD_ROLLOUT_N}"
    "actor_rollout_ref.rollout.response_length=${VOPD_MAX_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.free_cache_engine=False
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.agent.num_workers=${VOPD_AGENT_WORKERS}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.rollout_is_threshold=2.0
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    reward_model.enable=False
    reward_model.use_reward_loop=False
    "critic.model.path=${VOPD_MODEL_DIR}"
    custom_reward_function.path=null
    trainer.device=npu
    trainer.project_name=Vision-OPD
    trainer.group_name=Vision-OPD-Qwen3.5-4B-Ascend
    trainer.experiment_name=Vision-OPD-Qwen3.5-4B-Ascend
    'trainer.logger=["console"]'
    "trainer.n_gpus_per_node=${VOPD_NPUS_PER_NODE}"
    "trainer.nnodes=${VOPD_NNODES}"
    "trainer.save_freq=${VOPD_SAVE_FREQ}"
    trainer.test_freq=-1
    "trainer.max_actor_ckpt_to_keep=${VOPD_MAX_CKPTS}"
    "trainer.total_epochs=${VOPD_TOTAL_EPOCHS}"
    trainer.val_before_train=False
    "trainer.default_local_dir=${VOPD_CHECKPOINT_DIR}"
    "trainer.rollout_data_dir=${VOPD_ROLLOUT_DIR}"
    "${RAY_OVERRIDE[@]}"
)

if [[ "${VOPD_TOTAL_TRAINING_STEPS}" -gt 0 ]]; then
    TRAIN_CMD+=("trainer.total_training_steps=${VOPD_TOTAL_TRAINING_STEPS}")
fi

IGNORED_PLATFORM_ARGS=()
for argument in "$@"; do
    if [[ "${argument}" == --* || "${argument}" != *=* ]]; then
        IGNORED_PLATFORM_ARGS+=("${argument}")
    else
        TRAIN_CMD+=("${argument}")
    fi
done
if [[ "${#IGNORED_PLATFORM_ARGS[@]}" -gt 0 ]]; then
    echo "[WARNING] ignored ${#IGNORED_PLATFORM_ARGS[@]} platform launcher argument(s); only Hydra key=value overrides are forwarded"
fi

if [[ "${VOPD_DRY_RUN}" == "1" ]]; then
    printf '[dry-run] command:'
    printf ' %q' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

echo "[train] log=${LOG_FILE}"
"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

LATEST_CKPT=""
LATEST_STEP=-1
for checkpoint in "${VOPD_CHECKPOINT_DIR}"/global_step_*; do
    [[ -d "${checkpoint}" ]] || continue
    step="${checkpoint##*_}"
    if [[ "${step}" =~ ^[0-9]+$ && "${step}" -gt "${LATEST_STEP}" ]]; then
        LATEST_STEP="${step}"
        LATEST_CKPT="${checkpoint}"
    fi
done

if [[ "${VOPD_AUTO_MERGE}" == "1" && -n "${LATEST_CKPT}" ]]; then
    bash "${PROJECT_ROOT}/scripts/merge_checkpoint.sh" "${LATEST_CKPT}" \
        "${VOPD_MERGED_DIR}/global_step_${LATEST_STEP}"
fi

echo "[done] checkpoints=${VOPD_CHECKPOINT_DIR}"
echo "[done] rollouts=${VOPD_ROLLOUT_DIR}"
echo "[done] merged=${VOPD_MERGED_DIR}"
echo "[done] logs=${VOPD_LOG_DIR}"
