#!/usr/bin/env python3
"""Validate offline Vision-OPD assets without importing third-party packages."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


LOCK_FILES = (
    "requirements-ascend.txt",
    "requirements-ascend-core.txt",
    "requirements-ascend-plugins.txt",
)


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)


def check_model(model_dir: Path) -> bool:
    success = True
    if not model_dir.is_dir():
        fail(f"local model directory does not exist: {model_dir}")
        return False

    for filename in ("config.json", "tokenizer_config.json"):
        if not (model_dir / filename).is_file():
            fail(f"Qwen3.5 model file is missing: {model_dir / filename}")
            success = False

    processor_files = ("preprocessor_config.json", "processor_config.json")
    if not any((model_dir / filename).is_file() for filename in processor_files):
        fail(
            "Qwen3.5 processor configuration is missing; expected one of: "
            + ", ".join(processor_files)
        )
        success = False

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(index["weight_map"].values()))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            fail(f"cannot read model weight index {index_path}: {exc}")
            success = False
        else:
            missing = [name for name in shard_names if not (model_dir / name).is_file()]
            if missing:
                fail("Qwen3.5 weight shards referenced by the index are missing: " + ", ".join(missing))
                success = False
    elif not (model_dir / "model.safetensors").is_file():
        fail(f"Qwen3.5 safetensors weights or weight index are missing under {model_dir}")
        success = False

    incomplete = sorted(path.name for path in model_dir.rglob("*.incomplete"))
    if incomplete:
        fail("incomplete model downloads are present: " + ", ".join(incomplete[:10]))
        success = False

    if success:
        print(f"[ OK ] complete local Qwen3.5 model: {model_dir}")
    return success


def pinned_requirements(project_root: Path) -> list[tuple[str, str]]:
    pins: list[tuple[str, str]] = []
    for filename in LOCK_FILES:
        for raw_line in (project_root / filename).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s;]+)", line)
            if match is None:
                raise ValueError(f"non-exact requirement in {filename}: {line}")
            pins.append(match.groups())
    return pins


def check_wheelhouse(project_root: Path, wheel_dir: Path) -> bool:
    if not wheel_dir.is_dir():
        fail(f"offline wheelhouse does not exist: {wheel_dir}")
        return False

    files = [path.name for path in wheel_dir.iterdir() if path.is_file()]
    lowered = [name.lower() for name in files]
    success = True
    encoded_names = [name for name in files if "%2b" in name.lower()]
    if encoded_names:
        fail(
            "wheel filenames must contain '+' rather than the URL escape '%2B': "
            + ", ".join(encoded_names[:5])
        )
        success = False

    if not any(name.startswith("pip-") for name in lowered):
        fail("offline wheelhouse lacks pip>=23.3,<26 required to resolve the environment")
        success = False

    try:
        pins = pinned_requirements(project_root)
    except (OSError, ValueError) as exc:
        fail(str(exc))
        return False

    missing: list[str] = []
    for distribution, version in pins:
        normalized = re.sub(r"[-_.]+", "_", distribution).lower()
        prefix = f"{normalized}-{version.lower()}"
        if not any(name.startswith(prefix) for name in lowered):
            missing.append(f"{distribution}=={version}")
    if missing:
        fail(
            "offline wheelhouse lacks direct pinned distributions (transitive dependencies are "
            "checked later by pip --dry-run): " + ", ".join(missing)
        )
        success = False

    required_worker_wheels = {
        "torch-npu": re.compile(
            r"^torch_npu-2\.9\.0\.post1\+git4c901a4-cp310-cp310-.*aarch64\.whl$",
            re.IGNORECASE,
        ),
        "triton-ascend": re.compile(
            r"^triton_ascend-3\.2\.0\.dev20260322-cp310-cp310-.*aarch64\.whl$",
            re.IGNORECASE,
        ),
    }
    for distribution, pattern in required_worker_wheels.items():
        if not any(pattern.fullmatch(name) for name in files):
            fail(f"offline wheelhouse lacks the official cp310/aarch64 {distribution} wheel")
            success = False

    if success:
        print(f"[ OK ] offline wheelhouse direct assets: {wheel_dir}")
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--wheel-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    success = True
    if args.model_dir is not None:
        success = check_model(args.model_dir.resolve()) and success
    if args.wheel_dir is not None:
        success = check_wheelhouse(project_root, args.wheel_dir.resolve()) and success
    if args.model_dir is None and args.wheel_dir is None:
        fail("at least one of --model-dir or --wheel-dir is required")
        return 2
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
