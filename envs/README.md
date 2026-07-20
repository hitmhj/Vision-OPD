# Portable Ascend assets

Run `bash scripts/prepare_ascend_assets.sh` in WebStudio or another preparation
host before starting training. On x86_64/Python 3.9 it instructs pip to resolve
the configured CPython 3.10/3.11 aarch64 target explicitly. The preparation entry creates this layout:

```text
envs/
  models/Qwen3.5-4B/          complete Hugging Face snapshot
  wheels/cp311-aarch64/       complete offline wheelhouse (current preparation default)
  wheels/cp310-aarch64/       optional wheelhouse for Python 3.10 workers
  cache/                      Hugging Face, vLLM, Torch and pip caches
  runtime/.venv-ascend-cp311/ generated on the detected Python 3.11 NPU worker
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
