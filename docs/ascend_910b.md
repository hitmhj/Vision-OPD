# Vision-OPD on Atlas 910B

This adaptation targets:

`pytorch_2.6.0-cann_8.2.rc1-py_3.11-euler_2.10.11-aarch64-snt9b`

It does not install or replace `torch`, `torch-npu`, CANN, HCCL, or an
accelerator-specific `torchvision`. The startup script prints the versions it
actually imports from the target image. That output is authoritative; this
repository cannot truthfully claim the image's patch build without running the
image.

## Lifecycle audit

The original entrypoint starts one Python driver, `verl.trainer.main_ppo`. It
initializes Ray, creates NPU/GPU resource pools, and launches the
`ActorRolloutRefWorker` FSDP group. The actor generates trajectories, constructs
privileged teacher inputs from `bbox_images`, computes student and teacher
top-k distributions, applies generalized JSD (`alpha=0.5`), backpropagates the
Vision-OPD policy loss, updates the actor, and then updates the colocated teacher
with EMA (`0.05`). Each FSDP rank saves model/optimizer/RNG shards plus Hugging
Face config and processor files. `verl.model_merger` reconstructs a standard
Hugging Face checkpoint.

The old shell script selected online `Qwen/Qwen3.5-4B`, CUDA-oriented vLLM,
FlashAttention/remove-padding paths, and derived node count from `WORLD_SIZE`.
Its frozen requirements also contained CUDA torch 2.10, CUDA libraries, Triton,
xFormers, vLLM 0.18, and incompatible package versions.

The Ascend lifecycle keeps the same algorithm and checkpoint flow:

1. Resolve the project root and project-relative configuration.
2. Source the CANN/NNAL/ATB/ASDSIP files that actually exist in the image.
3. Reuse and report the image's `torch`/`torch-npu` stack.
4. Optionally install only non-core wheels from the offline cp311-aarch64
   wheelhouse.
5. Convert an already staged `train.jsonl` when parquet is absent; never
   download data or a model during the job.
6. Start one Ray driver. On multi-node ModelArts jobs rank 0 creates the Ray
   head, other platform ranks join it, and only rank 0 runs the driver.
7. Run FSDP over `npu`/HCCL and use the current FSDP actor itself for
   Transformers rollout.
8. Save sharded checkpoints and optionally merge the newest step into a
   different output directory.

## Compatibility matrix

| Component | Needed | Image/user value | Adopted value | Handling and basis |
| --- | --- | --- | --- | --- |
| Python | yes | 3.11 / cp311 / aarch64 | image 3.11 | Reuse; wheelhouse must be cp311-aarch64. |
| CANN | yes | 8.2 RC1 | image 8.2 RC1 | System component; discover `set_env.sh`, no string gate. |
| PyTorch | yes | image name says 2.6.0 | image actual 2.6.x | Never pip-install. Runtime import is logged. |
| torch-npu | yes | requested 2.5.1 | image-matched actual, expected 2.6.x | Ascend's matrix pairs torch 2.6 with torch-npu 2.6; 2.5.1 belongs to the torch 2.5 line. Keep the image unit. |
| transformers | yes | requested 4.52.4 | 5.5.0 | Unavoidable deviation: Qwen3.5 was added after 4.52 and its official model repo has no remote modeling module that 4.52 could load. |
| torchvision | no on this image path | unspecified | image copy only; 0.21 if the image vendor supplies the torch 2.6 pair | Vision-OPD image loading uses Pillow/qwen-vl-utils. Do not install a CUDA wheel. |
| vLLM/vLLM-Ascend | rollout only | none / old 0.18 history | not installed | The CANN 8.2-era vLLM-Ascend line predates Qwen3.5; Qwen3.5-capable lines require newer CANN. Replaced with HF rollout. |
| Ray | yes | unspecified | 2.53.0 | cp311/aarch64 wheel; Ray owns worker launch and uses the custom NPU resource already supported by verl. |
| FSDP collectives | yes | project used NCCL default on CUDA | HCCL | `get_nccl_backend()` selects HCCL after explicit torch-npu registration. |
| NumPy | yes | coexistence not specified | 1.26.4 | Conservative ABI for pandas 2.3.1 and pyarrow 21.0.0. |
| pandas / pyarrow | data | 2.3.1 / 21.0.0 | exact requested versions | Used by datasets/parquet. |
| Pillow / cachetools / psutil / PyYAML | yes | 10.3.0 / 5.3.3 / 7.0.0 / 6.0.2 | exact requested versions | Direct imports on the active path. |
| tensordict / torchdata | yes | unspecified | 0.10.0 / 0.11.0 | verl explicitly requires tensordict >=0.10; DataLoader utilities import torchdata. |
| pydantic | yes | implicit conflict with optional spaCy 3.5.3 | 2.12.5 | verl uses `model_validate`, `model_dump`, and v2 validators. spaCy is not in the training environment install set. |

Primary compatibility references: [Ascend PyTorch matrix](https://github.com/Ascend/pytorch),
[vLLM-Ascend release notes](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/release_notes.md),
[Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[official Qwen3.5-4B files](https://huggingface.co/Qwen/Qwen3.5-4B/tree/main), and
[PyTorch version pairs](https://docs.pytorch.org/get-started/previous-versions/).

The complete versions supplied by the user remain recorded in
`constraints-user-environment.txt`. They are not blindly installed because
most are unrelated Web/scientific packages and some have mutually incompatible
transitive requirements. The active training subset retains every supplied
version it directly uses. The two deviations above are required for a working
NPU ABI and for Qwen3.5 model recognition.

## Rollout and algorithm invariants

`HFRolloutServer` serializes requests and fans each request out to every FSDP
rank. All ranks switch from trainer state to rollout state together, summon the
same current actor parameters, receive the same multimodal image/video data,
and use identical sampling RNG state. The proxy rejects rank-divergent token
sequences. After rollout it restores trainer RNG state and parameter offload.
There is no stale model copy, so the trajectories remain on-policy.

The multimodal processor rebuilds pixel tensors and verifies that its token IDs
exactly equal the AgentLoop prompt IDs. It fails instead of silently omitting a
visual input. This backend is correctness-oriented and slower than a supported
serving engine. A later vLLM-Ascend switch is valid only after moving to a
Qwen3.5-capable CANN/torch-npu image and testing output/log-prob equivalence.

Unchanged research behavior includes regional `bbox_images` teacher inputs,
student/teacher forward passes, top-k generalized JSD, `alpha=0.5`, EMA teacher,
gradient backward, optimizer/scheduler update, GRPO grouping, rollout
importance correction, checkpoint contents, and merger semantics.

## Dependencies

`requirements-ascend.txt` contains only direct active-path packages. The
offline wheelhouse must contain their full transitive closure for Python 3.11
and aarch64. `VOPD_INSTALL_DEPS=1` enables a no-index install; it is off by
default so a prebuilt image or mounted environment is reused.

Necessary additions are accelerate (FSDP loading), Hydra/OmegaConf/ANTLR
(configuration), Ray/cloudpickle/cachetools (distributed orchestration),
datasets/pandas/pyarrow (parquet), tensordict/torchdata (verl protocols and data
loading), Transformers/tokenizers/safetensors (Qwen3.5 and checkpoints),
qwen-vl-utils/Pillow (multimodal processing), PEFT (unconditional worker import
and optional LoRA), pydantic (agent schemas), and small direct utilities such as
codetiming, einops, packaging, psutil, PyYAML, and tqdm.

Removed from the old freeze are torch/torchvision/torchaudio, every NVIDIA CUDA
wheel, cupy-cuda, Triton, xFormers, FlashAttention/FlashInfer, vLLM,
flash-linear-attention/fla kernels, and unrelated server/development packages.
Qwen3.5's optional fast DeltaNet kernels are deliberately not used; its native
PyTorch fallback plus SDPA is selected.

## Modified files

- Runtime core: `verl/utils/device.py`,
  `verl/utils/checkpoint/fsdp_checkpoint_manager.py`,
  `verl/workers/fsdp_workers.py`, `verl/workers/rollout/hf_rollout.py`, and
  `verl/workers/rollout/replica.py`.
- Agent/multimodal flow: `verl/experimental/agent_loop/agent_loop.py`,
  `single_turn_agent_loop.py`, `tool_agent_loop.py`,
  `verl/utils/dataset/rl_dataset.py`, and `vision_utils.py`.
- Entrypoints/data/checkpoints: `scripts/start_vision_opd_ascend.sh`,
  `run_vision_opd.sh`, `prepare_data.py`, `merge_checkpoint.sh`,
  `infer_vision_opd_ascend.py`, `run_inference_ascend.sh`, and
  `check_vision_opd_ascend.py`.
- Environment/delivery: `vision_opd_ascend.env`, `requirements.txt`,
  `requirements-ascend.txt`, `constraints-user-environment.txt`, `README.md`,
  and this guide.

## Layout and commands

Default layout:

```text
envs/models/Qwen3.5-4B/       local base model
envs/wheels/cp311-aarch64/    complete offline wheelhouse
envs/cache/                   HF/datasets caches
envs/runtime/                 Ray temporary files
data/train.jsonl              staged source metadata (optional)
data/train.parquet            prepared training data
output/checkpoints/           FSDP checkpoints
output/rollouts/              generated trajectory records
output/merged/                merged Hugging Face checkpoints
output/logs/                  driver logs
```

Training uses one fixed command:

```bash
bash scripts/start_vision_opd_ascend.sh
```

Useful overrides:

```bash
VOPD_INSTALL_DEPS=1 \
VOPD_NPUS_PER_NODE=8 \
VOPD_SAVE_FREQ=20 \
VOPD_AUTO_MERGE=1 \
bash scripts/start_vision_opd_ascend.sh
```

For an externally managed multi-node Ray cluster, set `VOPD_RAY_ADDRESS` and
run the entrypoint on platform rank 0. Otherwise ModelArts variables
`MA_NUM_HOSTS`, `MA_NUM_GPUS`, `VC_TASK_INDEX`, and `VC_WORKER_HOSTS` are used to
form the cluster. Do not wrap this entrypoint in `torchrun`.

Merge explicitly, keeping shards and merged output separate:

```bash
bash scripts/merge_checkpoint.sh \
  output/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_50 \
  output/merged/global_step_50
```

Offline inference:

```bash
VOPD_INFER_MODEL_DIR=output/merged/global_step_50 \
bash scripts/run_inference_ascend.sh \
  --image data/example.jpg --prompt "Describe the fine-grained detail."
```

For a non-mutating launch simulation:

```bash
python3 scripts/check_vision_opd_ascend.py
VOPD_DRY_RUN=1 bash scripts/start_vision_opd_ascend.sh
```

## Remaining risks and diagnostics

Static checks cannot validate a real 910B kernel, image contents, HCCL network,
or memory capacity. The first target-image run should retain the startup version
banner and training log. Failure collection:

```bash
python3 -c "import torch; print(torch.__version__); import torch_npu; print(torch_npu.__version__); print(torch.npu.is_available(), torch.npu.device_count())"
python3 -m pip show torch torch-npu transformers ray tensordict torchdata
npu-smi info
env | grep -E 'ASCEND|HCCL|RAY|MA_|VC_'
ray status
ls -la /usr/local/Ascend /usr/local/Ascend/ascend-toolkit
```

Also collect `${VOPD_LOG_DIR}/train-*.log`, the Ray session logs under
`${VOPD_RUNTIME_DIR}/ray`, and Ascend device logs configured by the platform.
Version strings only produce diagnostics; real import, model load, collective,
or operator failures return their original nonzero status.

In particular, the launcher never exits or raises solely because Python,
torch, torch-npu, Transformers, or Ray has a different version string. It emits
`[WARNING] ...; continuing` and lets the real import/model/operator/training
path determine success. An import or installation failure is not converted into
a version precheck and retains its original error.
