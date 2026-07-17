# Vision-OPD on Ascend 910B

This repository provides a conservative Ascend 910B / Atlas A2 path for
Vision-OPD training and vLLM inference. The CUDA environment and the Ascend
environment must not be mixed.

## Platform-provided software environment

The production entry does not install `requirements-ascend.txt`. It delegates
dependency preparation to the Huawei Qwen3.5 initializer configured by
`VOPD_DEPENDENCY_SETUP_DIR` and `VOPD_DEPENDENCY_SETUP_SCRIPT`, then applies
the sample's `accelerate==1.11.0` pin. PyTorch, torch-npu, Transformers, vLLM
and vLLM-Ascend versions are owned by that initializer and reported by the
runtime preflight.

The defaults from the provided platform example load CANN 8.5.1, NNAL/ATB
with `--cxx_abi=0`, and ASDSIP. These external paths remain overridable through
`CANN_ENV_SCRIPT`, `NNAL_ENV_SCRIPT`, and `ASDSIP_ENV_SCRIPT`.

Driver, firmware, CANN and NNAL are system components. They must be installed
by the machine administrator or supplied by the base container before running
the repository installer. `npu-smi info` must succeed.

## Unified lifecycle entry

Configure `vision_opd_ascend.env` and use one public entry for both an
exploration environment and a ModelArts task:

```bash
bash scripts/start_vision_opd_ascend.sh
```

The same entry performs, in order:

1. source the configuration and export `VOPD_*` variables;
2. load the existing driver/CANN/NNAL runtime;
3. call `scripts/install_ascend.sh` by project-relative path; that installer
   delegates to the provided Huawei dependency initializer;
4. reuse or prepare data according to `VOPD_PREPARE_DATA_IF_MISSING`;
5. run NPU and configuration preflight checks;
6. train and save FSDP checkpoints;
7. optionally merge the latest checkpoint according to `VOPD_AUTO_MERGE`.

No Vision-OPD training code or LLaMA-Factory training command is executed by
the dependency installer. Host firmware, driver, CANN and NNAL are supplied by
the platform environment.

If CANN is installed in a non-standard location:

```bash
CANN_ENV_SCRIPT=/opt/Ascend/ascend-toolkit/set_env.sh \
NNAL_ENV_SCRIPT=/opt/Ascend/nnal/atb/set_env.sh \
bash scripts/start_vision_opd_ascend.sh
```

## ModelArts task semantics

ModelArts may execute a rank-table boot file once per NPU. The production entry
uses `RANK_ID` and `ASCEND_DEVICE_ID` so only global rank 0 starts the verl/Ray
driver; all other invocations exit successfully. Do not configure
`MA_RUN_METHOD=torchrun` for this entry because the project creates its own Ray
workers instead of using torchrun ranks.

The entry maps platform variables as follows:

| ModelArts variable | Vision-OPD use |
| --- | --- |
| `MA_NUM_GPUS` | default `VOPD_GPUS_PER_NODE` |
| `MA_NUM_HOSTS` | default `VOPD_NNODES` |
| `MA_MOUNT_PATH` | default output/rollout parent |
| `MA_LOG_DIR` | default log directory |
| `RANK_ID`, `ASCEND_DEVICE_ID` | single-driver guard |
| `RANK_TABLE_FILE` | preserved for the Ascend/HCCL runtime |

Custom job variables must use the `VOPD_` prefix, not the platform-reserved
`MA_` prefix. Edit `vision_opd_ascend.env`, or inject variables with the same
names in the job configuration; injected variables take precedence.

The unified entry intentionally supports exactly one node. Its default is one
node exposing eight 910B NPUs; `MA_NUM_GPUS` may override the card count. A
multi-node job is rejected before installation or training because it needs a
separate Ray head/worker bootstrap.

## Data and training

Prepare the dataset in the same format as the original project:

```bash
python scripts/prepare_data.py --data-dir ./data
```

Start the platform-managed single-node training job:

```bash
bash scripts/start_vision_opd_ascend.sh
```

Before a long run, execute one complete reduced-size optimization step:

```bash
bash scripts/smoke_test_vision_opd_ascend.sh
```

This is a real hardware test: it loads the model and data, performs multimodal
rollout, teacher/student JSD, EMA, backward, optimizer update and weight sync,
then saves and merges after one step. It is not run during repository-only
validation. The wrapper still delegates to the same unified lifecycle entry.

For platform parameters, `VOPD_TOTAL_TRAINING_STEPS=1` bounds a validation job,
`VOPD_RESUME_MODE=auto` resumes the latest checkpoint, and
`VOPD_RESUME_MODE=disable` starts a clean run in the selected output directory.

The Ascend launcher selects the following safe initial execution path:

- NPU device and HCCL communication;
- PyTorch SDPA instead of the CUDA FlashAttention2 extension;
- eager vLLM-Ascend execution instead of CUDA graph capture;
- no `torch.compile`, Triton fused kernel or FlashInfer path;
- forced native PyTorch Qwen3.5 causal-conv/DeltaNet fallback instead of its
  optional CUDA `causal-conv1d`/FLA fast path;
- safetensors rollout loading;
- parameter and optimizer CPU offload;
- 50% initial vLLM memory utilization.

Configuration values are provided through `VOPD_*` environment variables or
Hydra arguments. Example for a smaller four-NPU exploration run:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
VOPD_GPUS_PER_NODE=4 \
VOPD_TRAIN_FILE=/mnt/data/train.parquet \
VOPD_TRAIN_BATCH_SIZE=32 \
VOPD_PPO_MINI_BATCH_SIZE=32 \
VOPD_ROLLOUT_N=4 \
VOPD_ROLLOUT_MEMORY_UTILIZATION=0.4 \
bash scripts/start_vision_opd_ascend.sh
```

The paper hyperparameters target eight devices. Reducing the card count may
require further reductions in sequence length, rollout count or micro-batch
size. It also changes experiment throughput and possibly optimization dynamics.

## Checkpoint and inference

The unified lifecycle prints and records all artifact locations. With the
checked-in defaults they are:

- FSDP actor shards: `$VOPD_OUTPUT_DIR/global_step_*/actor`;
- rollout records: `$VOPD_ROLLOUT_DIR`;
- merged HuggingFace model: `$VOPD_MERGED_MODEL_DIR`;
- TensorBoard events: `$TENSORBOARD_DIR`;
- complete console lifecycle log: `$VOPD_LIFECYCLE_LOG`;
- machine-readable location manifest: `$VOPD_LOG_DIR/artifacts.env`.

On ModelArts these roots default below `MA_MOUNT_PATH`/`MA_LOG_DIR`; override
the corresponding `VOPD_*` variables when the task defines a dedicated output
mount that must be persisted to OBS.

Merge FSDP shards on CPU using the existing command:

```bash
bash scripts/merge_checkpoint.sh ./checkpoints/<experiment>/<global_step>
```

Serve the merged model on eight NPUs:

```bash
bash scripts/serve_vision_opd_ascend.sh <merged_model_path>
```

Override tensor parallelism or vLLM memory utilization when needed:

```bash
TENSOR_PARALLEL_SIZE=4 GPU_MEMORY_UTILIZATION=0.75 \
bash scripts/serve_vision_opd_ascend.sh <merged_model_path>
```

The option name `--gpu-memory-utilization` is retained because it is the public
vLLM CLI name; vLLM-Ascend applies it to NPU memory.

## Preflight checks

The launchers automatically run:

```bash
python scripts/check_ascend_env.py --min-npus 8
```

It reports the actual package versions supplied by the Huawei initializer and
verifies required imports, CANN environment loading, visible NPU count,
torch-npu and vLLM-Ascend, loaded transformer patches, Ray's `NPU` resources,
`npu-smi`, and a small BF16 forward/backward operation.
Repository-only validation, which does not require Ascend hardware or
dependencies, is available as:

```bash
python scripts/check_ascend_env.py --static-only
```

## Hardware validation boundary

Static checks cannot prove that every Qwen3.5 operator supplied by a particular
CANN/torch-npu build supports the full multimodal backward pass. The first real
machine run must therefore be treated as an operator validation run. Start with
one full training step and confirm all of the following before a long job:

1. Qwen3.5 vision and Gated DeltaNet forward/backward complete without a CPU or
   CUDA-only operator fallback.
2. The full-image student and cropped-image teacher both produce finite logits.
3. Top-k JSD and EMA teacher updates remain finite.
4. FSDP-to-vLLM weight synchronization completes on all ranks.
5. HCCL collectives and vLLM sleep/wakeup complete without timeout or OOM.

If an installed Transformers build attempts to import CUDA `flash-attn`,
`causal-conv1d`, FlashInfer or Triton for Qwen3.5, do not add the CUDA package to
the Ascend environment. Keep the SDPA/eager configuration and use the native
PyTorch/torch-npu fallback from the matched Transformers build, or adapt that
specific operator for NPU.
