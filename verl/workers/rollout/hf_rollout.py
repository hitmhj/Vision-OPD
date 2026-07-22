# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Transformers rollout backend for accelerator-native FSDP workers.

This backend intentionally has no vLLM/SGLang dependency.  In hybrid mode the
Ray proxy invokes every FSDP rank for each request, so parameter unsharding and
generation remain collective while only one identical result is returned.
It is slower than a dedicated serving engine, but preserves on-policy weights
and multimodal inputs on Ascend stacks for which no compatible serving engine
exists.
"""

import contextlib
from typing import Any, Generator, Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.device import get_device_id, get_device_name

from .base import BaseRollout

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    """Token-in/token-out Hugging Face rollout over the current actor module."""

    def __init__(self, module: nn.Module, config, model_config, device_mesh, processor=None, tokenizer=None):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        self.module = module
        self.processor = processor
        self.tokenizer = tokenizer

    async def resume(self, tags: list[str]):
        # The actor owns the weights and has already been moved to the NPU.
        return None

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        # No copy is necessary: rollout and training use the same FSDP module.
        return None

    async def release(self):
        # Parameter offload is handled by ActorRolloutRefWorker.trainer_mode.
        return None

    def _prepare_qwen35_position_ids(self, model_inputs: dict[str, torch.Tensor]) -> None:
        """Convert processor-only modality ids into Qwen3.5 position ids.

        Transformers 5.5's Qwen3.5 processor emits ``mm_token_type_ids`` for
        M-RoPE construction, while the model forward used by this verl branch
        does not accept that processor-only key.  Consume it here without
        discarding its semantics, matching the AgentLoop training path.
        """
        mm_token_type_ids = model_inputs.pop("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            return

        model_type = getattr(self.model_config.hf_config, "model_type", None)
        if model_type not in {"qwen3_5", "qwen3_5_moe"}:
            raise RuntimeError(f"HF rollout received mm_token_type_ids for unsupported model type {model_type!r}.")

        get_rope_index = getattr(self.processor, "get_rope_index", None)
        if not callable(get_rope_index):
            raise RuntimeError("Qwen3.5 processor does not expose get_rope_index required for multimodal rollout.")

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        rope_output = get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            attention_mask=attention_mask,
        )
        vision_position_ids = rope_output[0] if isinstance(rope_output, tuple) else rope_output
        if vision_position_ids.ndim != 3:
            raise RuntimeError(
                f"Qwen3.5 get_rope_index returned shape {tuple(vision_position_ids.shape)}, expected 3 dimensions."
            )
        if vision_position_ids.shape[0] == input_ids.shape[0] and vision_position_ids.shape[1] == 3:
            vision_position_ids = vision_position_ids.transpose(0, 1)
        elif not (
            vision_position_ids.shape[0] == 3 and vision_position_ids.shape[1] == input_ids.shape[0]
        ):
            raise RuntimeError(
                f"Qwen3.5 get_rope_index returned shape {tuple(vision_position_ids.shape)}, "
                "expected (3, batch, sequence) or (batch, 3, sequence)."
            )

        text_position_ids = attention_mask.to(dtype=vision_position_ids.dtype).cumsum(dim=-1) - 1
        text_position_ids.masked_fill_(attention_mask == 0, 1)
        model_inputs["position_ids"] = torch.cat(
            (text_position_ids.unsqueeze(0), vision_position_ids), dim=0
        )  # (4, batch, sequence), as required by Qwen3.5 GenerationMixin

    def _prepare_inputs(
        self,
        prompt_ids: list[int],
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        text_data: Optional[str],
    ) -> dict[str, torch.Tensor]:
        device = torch.device(get_device_name(), get_device_id())
        if image_data is None and video_data is None:
            return {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device),
                "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device),
            }

        if self.processor is None:
            raise RuntimeError("Multimodal HF rollout requires the model's local AutoProcessor.")

        # AgentLoop has already inserted the processor-specific visual tokens.
        # Decode with special tokens retained, then let the same processor build
        # pixel tensors and grids.  Refuse to continue if tokenization changes:
        # silently losing/reordering visual tokens would alter the algorithm.
        raw_prompt = text_data or self.tokenizer.decode(
            prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        videos = video_data
        video_metadatas = None
        if videos and isinstance(videos[0], tuple):
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)

        processor_inputs = self.processor(
            text=[raw_prompt],
            images=image_data,
            videos=videos,
            video_metadatas=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        rebuilt_ids = processor_inputs["input_ids"].squeeze(0).tolist()
        if rebuilt_ids != prompt_ids:
            raise RuntimeError(
                "HF rollout could not reproduce AgentLoop multimodal prompt ids. "
                "Check the local Qwen3.5 processor/chat-template files and disable prompt truncation."
            )
        # Build Qwen3.5's four-axis position ids on CPU.  Besides avoiding an
        # unnecessary NPU concat, this keeps their required layout explicit:
        # (text/temporal/height/width, batch, sequence).
        model_inputs = dict(processor_inputs)
        self._prepare_qwen35_position_ids(model_inputs)
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in model_inputs.items()}

    @torch.no_grad()
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        text_data: Optional[str] = None,
    ):
        del request_id
        sampling_params = dict(sampling_params)
        configured_max_tokens = int(self.config.response_length)
        max_tokens = int(
            sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", configured_max_tokens))
        )
        max_model_len = self.config.max_model_len or getattr(self.model_config.hf_config, "max_position_embeddings")
        max_tokens = max(0, min(max_tokens, configured_max_tokens, int(max_model_len) - len(prompt_ids)))
        if max_tokens == 0:
            from verl.workers.rollout.replica import TokenOutput

            return TokenOutput(token_ids=[], log_probs=[], stop_reason="completed")

        want_logprobs = bool(sampling_params.pop("logprobs", False))
        temperature = float(sampling_params.pop("temperature", self.config.temperature))
        top_p = float(sampling_params.pop("top_p", self.config.top_p))
        top_k = int(sampling_params.pop("top_k", self.config.top_k))
        repetition_penalty = float(sampling_params.pop("repetition_penalty", self.config.repetition_penalty))
        do_sample = bool(self.config.do_sample and temperature > 0)

        model_inputs = self._prepare_inputs(prompt_ids, image_data, video_data, text_data)
        self.module.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)

        generation_kwargs = {
            "do_sample": do_sample,
            "num_beams": 1,
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "output_scores": want_logprobs,
            "return_dict_in_generate": True,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p, top_k=max(0, top_k))

        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(**model_inputs, **generation_kwargs)

        generated = output.sequences[0, len(prompt_ids) :].tolist()
        log_probs = None
        if want_logprobs:
            log_probs = [
                torch.log_softmax(score[0].float(), dim=-1)[token_id].item()
                for score, token_id in zip(output.scores, generated, strict=True)
            ]

        from verl.workers.rollout.replica import TokenOutput

        return TokenOutput(token_ids=generated, log_probs=log_probs, stop_reason="completed")

    def generate_sequences(self, prompts):
        raise RuntimeError("HFRollout is used through the async AgentLoop token-in/token-out path.")
