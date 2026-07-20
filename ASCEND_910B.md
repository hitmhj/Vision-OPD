# Vision-OPD on Ascend 910B

This repository provides a conservative Ascend 910B / Atlas A2 path for
Vision-OPD training and vLLM inference. The CUDA environment and the Ascend
environment must not be mixed.

## Platform-provided software environment

WebStudio can edit the repository and cross-prepare portable assets, but it
never creates the runtime venv or runs training. The production entry selects
Python and installs dependencies on the actual NPU worker. Python 3.10 and 3.11
are supported; the resolver prefers an interpreter that already has a complete
matching wheelhouse and creates `envs/runtime/.venv-ascend-cp310` or
`envs/runtime/.venv-ascend-cp311`.
`VOPD_INSTALL_MODE=auto` reuses the
environment only when all three Ascend lock files and the installer itself are
unchanged.

The binary stack is the official stable vLLM-Ascend 0.18.0 Atlas A2 matrix:
CANN 8.5.1, torch/torchaudio 2.9.0, the ABI-matched cp310/cp311 aarch64 special
`torch-npu==2.9.0.post1+git4c901a4` build,
`triton-ascend==3.2.0.dev20260322`, and vLLM/vLLM-Ascend 0.18.0. Do not replace
individual members of this compatibility unit. Common settings from the
LLaMAFactory sample are retained, but its TRL, Gradio and DeepSpeed dependencies
are not installed.

The authoritative compatibility table is the
[vLLM-Ascend versioning policy](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html),
and the special wheel filenames come from the
[v0.18.0 release notes](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0).

`CANN_ENV_SCRIPT`, `NNAL_ENV_SCRIPT`, and, when available,
`ASDSIP_ENV_SCRIPT` are sourced in that order. CANN and ATB are required;
ASDSIP is optional unless `VOPD_REQUIRE_ASDSIP=1`. Bash nounset is disabled and
`ZSH_VERSION` is explicitly defined before ATB, covering both forms of the
vendor-script failure.

CANN 8.5.1 remains the tested baseline, but version detection is advisory by
default (`VOPD_REQUIRE_CANN_VERSION_MATCH=0`). An unknown or different CANN
version emits a warning and proceeds to real torch-npu, NPU BF16 backward,
vLLM-Ascend and Ray checks. Strict equality can be restored with
`VOPD_REQUIRE_CANN_VERSION_MATCH=1`.

Driver, firmware, CANN and NNAL are system components. They must be installed
by the machine administrator or supplied by the base container before running
the repository installer. `npu-smi info` must succeed.

## Portable asset preparation

Run the asset preparation entry before the training entry. It derives every
default from the repository root and creates `envs/models/Qwen3.5-4B`,
`envs/wheels/cp311-aarch64` (the current preparation default), `envs/cache` and `envs/runtime`.

The production-safe default uses no network. The preparation entry supports
x86_64/Python 3.9 WebStudio by passing an explicit CPython 3.10/3.11 aarch64 target
to pip. It can gather compatible wheels from one or more platform directories
and copy a complete model snapshot:

```bash
VOPD_INTERNAL_WHEEL_DIRS=/opt/platform/whls:/mounted/extra/whls \
VOPD_MODEL_SOURCE_DIR=/mounted/models/Qwen3.5-4B \
bash scripts/prepare_ascend_assets.sh
```

Compiled wheels for the other Python ABI and all x86_64 wheels are rejected;
only the selected cp310/cp311 aarch64, compatible `abi3/aarch64`, and universal
wheels are collected. Select the target before preparation when necessary:

```bash
VOPD_PREPARE_TARGET_PYTHON=3.11 bash scripts/prepare_ascend_assets.sh --online
```

If
the WebStudio or another preparation host has approved network access, use:

```bash
bash scripts/prepare_ascend_assets.sh --online
```

`--online` may install a small isolated Hugging Face download helper under
`envs/runtime/asset-preparer`, download the complete Qwen snapshot, fetch the
two vLLM-Ascend special wheels from the official Huawei OBS location, and ask
pip to collect every direct and transitive wheel. Without `--online`, none of
those hosts is contacted. The NPU training entry never uses `--online`. A
successful preparation writes a lock fingerprint
and SHA-256 inventory into the wheelhouse and records the immutable Qwen model
revision. `--check-only` validates both manifests without contacting a host.

The generated `.venv-ascend-cp310`/`.venv-ascend-cp311` is deliberately not a portable asset: the start
entry creates it on the real worker because virtual environments contain
host-specific paths and binary ABIs.

## Unified training lifecycle entry

After assets are ready, configure `vision_opd_ascend.env` and keep using the
same public training entry for both an exploration environment and a ModelArts
task:

```bash
bash scripts/start_vision_opd_ascend.sh
```

The same entry performs, in order:

1. source the configuration and export `VOPD_*` variables;
2. fingerprint the immutable worker and resolve Python, ABI wheelhouse and venv
   as one profile;
3. call `scripts/install_ascend.sh` by project-relative path; that installer
   creates/reuses the isolated venv, installs the
   CANN-8.5.1 core, generic runtime, vLLM plugin wheels and editable Vision-OPD
   package in that order, then validates versions, imports and dependency
   metadata;
4. source the configured CANN, NNAL/ATB and ASDSIP scripts exactly once;
5. reuse or prepare data according to `VOPD_PREPARE_DATA_IF_MISSING`;
6. run NPU and configuration preflight checks;
7. train and save FSDP checkpoints;
8. optionally merge the latest checkpoint according to `VOPD_AUTO_MERGE`.

Production installation is offline by default. Mount a complete wheelhouse for
the worker Python ABI. With `VOPD_TARGET_PYTHON=auto`, the resolver selects an
available Python 3.10/3.11 and its matching cp310/cp311 wheelhouse. Set
`VOPD_LOCAL_WHEEL_DIR` only for a nonstandard mounted directory.
`VOPD_PIP_NO_INDEX=1` is already the default.
The earlier LLaMAFactory wheelhouse is not complete for Vision-OPD and cannot
be used as that directory.

The installer uses `--no-index --find-links`, ignores inherited pip
configuration, and clears inherited `PIP_EXTRA_INDEX_URL`/`PIP_FIND_LINKS`.
It validates the resolved wheelhouse manifest and performs `pip --dry-run` resolution before
installing the NPU stack. Neither PyTorch, Huawei, Hugging Face nor GitHub is
contacted by the production default. The wheelhouse must include the exact
ABI-matched cp310/cp311 aarch64 `torch_npu-2.9.0.post1+git4c901a4` and
`triton_ascend-3.2.0.dev20260322` wheels plus every direct and transitive package
needed by the three Ascend lock files.

`arctic-inference==0.1.1` is deliberately not part of the portable lock. Its
upstream release is source-only and is used for vLLM suffix speculative
decoding; Vision-OPD does not enable that optional decoding mode. The active
rollout remains ordinary vLLM sampling.

Model and dataset downloads are disabled in the production entry by default.
Upload or mount Qwen3.5-4B under the default `envs/models/Qwen3.5-4B`, or inject
`VOPD_MODEL_PATH=/mounted/model/path`. Relative injected paths are resolved
against the repository rather than the algorithm launch directory.
`VOPD_REQUIRE_LOCAL_MODEL=1` validates the pinned revision manifest, config,
processor, tokenizer, weight
index and all referenced shards before dependency
installation, and `VOPD_HF_OFFLINE=1` exports the Hugging Face/Transformers/
Datasets offline flags. The local compressed dataset layout documented below is
then prepared without contacting Hugging Face.

The installer requires prebuilt vLLM and vLLM-Ascend wheels and performs an
immediate Ascend-platform import check. Source fallback is disabled by default
because an offline ModelArts worker cannot fetch source or missing build
dependencies. It can be enabled explicitly only when a complete local source
build wheelhouse has been prepared.

`hydra-core==1.3.2` and `omegaconf==2.3.0` require
`antlr4-python3-runtime==4.9.*`. Version 4.9.3 is published upstream only as a
source archive, so the online WebStudio preparation step builds it once into
the portable `py3-none-any` wheel. The Atlas worker remains binary-only and
offline; do not replace it with ANTLR 4.11 or newer.

All repository commands derive `PROJECT_ROOT` from the startup script location.
The Vision-OPD folder can therefore be mounted anywhere. Host firmware, driver,
CANN and NNAL remain platform components; Python packages are installed inside
the project-controlled venv on the worker.

If CANN is installed in a non-standard location:

```bash
CANN_ENV_SCRIPT=/opt/Ascend/cann-8.5.1/set_env.sh \
NNAL_ENV_SCRIPT=/opt/Ascend/nnal/atb/set_env.sh \
ASDSIP_ENV_SCRIPT=/opt/Ascend/nnal/asdsip/set_env.sh \
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

### Fixed-entry staged validation

The platform command never changes:

```bash
bash scripts/start_vision_opd_ascend.sh
```

Set one job environment variable to choose the gate:

| `VOPD_RUN_MODE` | Last completed stage |
| --- | --- |
| `probe` | worker/Python/CANN/glibc/NPU fingerprint; no installation |
| `dependencies` | isolated offline dependency installation and import validation |
| `preflight` | model/data integrity plus NPU BF16 backward and Ray resource checks |
| `smoke` | one reduced Vision-OPD optimization step and checkpoint |
| `train` | full configured training lifecycle (default) |

Move through these gates on a new image. A failure remains at the smallest
relevant stage and the next run reuses the verified venv and assets.

## Data and training

Prepare the dataset in the same format as the original project:

```bash
python scripts/prepare_data.py --data-dir ./data
```

When the repository already contains `data/train.jsonl` together with
`data/images/images.tar.gz*` and
`data/teacher_images/teacher_images.tar.gz`, the production entry automatically
uses local-only preparation. It extracts those archives without contacting
Hugging Face, preserves the compressed sources, validates the referenced image
files and writes `data/train.parquet`.

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

It checks every exact version from all Ascend lock files, the Qwen3.5 API used
by Vision-OPD, vLLM's Ascend platform registration, required imports, CANN
environment loading, visible NPU count, torch-npu, loaded transformer patches,
Ray's `NPU` resources, `npu-smi`, and a small BF16 forward/backward operation.
`pip check` output is accepted only for the explicitly enumerated
CUDA/plugin metadata entries superseded by the official Ascend matrix; every other
missing or conflicting dependency fails installation.
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
