"""
Download and preprocess Vision-OPD-6K training data from HuggingFace.

Usage:
    python scripts/prepare_data.py --data-dir ./data

This script:
1. Optionally downloads train.jsonl and image archives from Vision-OPD-6K
2. Extracts local image archives when they are present
3. Validates image paths and converts train.jsonl to the expected parquet format
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import datasets
from huggingface_hub import snapshot_download


REMOVE_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Vision-OPD-6K training data.")
    parser.add_argument("--data-dir", default="./data", help="Output directory for processed data")
    parser.add_argument("--hf-repo", default="yuanqianhao/Vision-OPD-6K", help="HuggingFace dataset repo")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, only preprocess")
    return parser.parse_args()


def download_dataset(repo_id: str, data_dir: str) -> None:
    print(f"Downloading dataset from {repo_id} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=data_dir)


def archive_sort_key(filename: str) -> list[str | int]:
    """Sort split archive suffixes numerically (for example .2 before .10)."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", filename)]


def extract_local_archives(data_dir: str) -> None:
    """Extract already-mounted archives without requiring a network download."""

    images_dir = os.path.join(data_dir, "images")
    teacher_dir = os.path.join(data_dir, "teacher_images")

    tar_files = []
    if os.path.isdir(images_dir):
        tar_files = sorted(
            (f for f in os.listdir(images_dir) if f.startswith("images.tar.gz")),
            key=archive_sort_key,
        )
    if tar_files:
        print("Extracting student images ...")
        # The dataset stores the large archive in multiple numbered parts. Join
        # them without a shell pipeline so paths containing spaces remain safe.
        combined_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".tar.gz", dir=images_dir, delete=False
            ) as combined:
                combined_path = combined.name
                for filename in tar_files:
                    with open(os.path.join(images_dir, filename), "rb") as part:
                        shutil.copyfileobj(part, combined)
            subprocess.run(["tar", "-xf", combined_path, "-C", "."], cwd=images_dir, check=True)
        finally:
            if combined_path and os.path.exists(combined_path):
                os.remove(combined_path)

    teacher_tar = os.path.join(teacher_dir, "teacher_images.tar.gz")
    if os.path.exists(teacher_tar):
        print("Extracting teacher images ...")
        subprocess.run(["tar", "-xf", "teacher_images.tar.gz", "-C", "."], cwd=teacher_dir, check=True)

    if tar_files or os.path.exists(teacher_tar):
        print("Image extraction complete; source archives were preserved.")


def clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{REMOVE_HINT}", "")
    text = text.replace(REMOVE_HINT, "")
    return text.strip()


def build_record(item: dict[str, Any], data_dir: str) -> dict[str, Any]:
    image_rel = item["images"][0]
    teacher_rel = item["teacher_images"][0]
    image_path = os.path.join(data_dir, image_rel)
    teacher_path = os.path.join(data_dir, teacher_rel)
    question = clean_question(item.get("problem", ""))

    missing_paths = [path for path in (image_path, teacher_path) if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError("Dataset record references missing image(s): " + ", ".join(missing_paths))

    return {
        "data_source": "zwz_rl_vqa_bbox_teacher",
        "prompt": [{"role": "user", "content": item["problem"]}],
        "images": [{"path": image_path}],
        "bbox_images": [{"path": teacher_path}],
        "ability": "visual_question_answering",
        "reward_model": {
            "style": "none",
            "ground_truth": item.get("answer", ""),
        },
        "extra_info": {
            "answer": item.get("answer", ""),
            "question": question,
            "source_extra_info": item.get("extra_info", {}),
        },
    }


def convert_to_parquet(data_dir: str) -> None:
    jsonl_path = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Converting train.jsonl to train.parquet ...")
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            records.append(build_record(item, data_dir))

    dataset = datasets.Dataset.from_list(records)
    output_path = os.path.join(data_dir, "train.parquet")
    dataset.to_parquet(output_path)
    print(f"Saved {len(records)} records to {output_path}")


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if not args.skip_download:
        download_dataset(args.hf_repo, data_dir)

    # Extraction is intentionally independent of downloading: ModelArts jobs
    # commonly mount train.jsonl and the archives before the task starts.
    extract_local_archives(data_dir)
    convert_to_parquet(data_dir)
    print(f"\nData preparation complete. Training data at: {data_dir}/train.parquet")


if __name__ == "__main__":
    main()
