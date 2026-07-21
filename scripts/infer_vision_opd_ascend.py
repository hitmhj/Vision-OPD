"""Offline single-image inference for a merged Vision-OPD checkpoint."""

import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - registers torch.npu
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local merged model directory")
    parser.add_argument("--image", required=True, help="Local image path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Merged model directory not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("npu", torch.npu.current_device())
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        local_files_only=True,
    ).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    with torch.no_grad(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    response_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    print(processor.batch_decode(response_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
