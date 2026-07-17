# Local base model

Place the complete `Qwen/Qwen3.5-4B` snapshot in `models/Qwen3.5-4B`, or set
`VOPD_MODEL_PATH` to another mounted directory. The launcher validates the
configuration, tokenizer, processor, safetensors index and every referenced
weight shard before installing dependencies.

Model weights are intentionally ignored by Git and must be uploaded or mounted
with the ModelArts dataset.
