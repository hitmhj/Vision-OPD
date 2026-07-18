# Portable Ascend assets

Run `bash scripts/prepare_ascend_assets.sh` on the Python 3.10/aarch64
ModelArts worker before starting training. The preparation entry creates this
layout:

```text
envs/
  models/Qwen3.5-4B/          complete Hugging Face snapshot
  wheels/cp310-aarch64/       complete offline wheelhouse
  cache/                      Hugging Face, vLLM, Torch and pip caches
  runtime/.venv-ascend/       generated on the NPU worker
```

The preparation command records the pinned Qwen revision beside the model and
a requirements fingerprint plus wheel SHA-256 inventory beside the wheelhouse.
The training entry requires both manifests, so copying assets manually is not a
replacement for running the preparation phase.

Model weights, wheel archives, caches and virtual environments are ignored by
Git. They must be included when the complete repository folder is uploaded as
a ModelArts dataset, or provided through the `VOPD_*` path overrides.

The runtime venv is not portable and must never be prepared on WebStudio or
copied from another host. The training entry recreates it on the real Atlas
910B worker.
