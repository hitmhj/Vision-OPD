#!/usr/bin/env python3
"""Preflight checks for the Vision-OPD Ascend 910B execution path."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_VERSIONS = {
    "torch": "2.9.0",
    "torch-npu": "2.9.0.post2",
    "torchdata": "0.11.0",
    "triton-ascend": "3.2.1",
    "vllm": "0.18.0",
    "vllm-ascend": "0.18.0",
}

CUDA_ONLY_QWEN_FAST_PATHS = (
    "causal-conv1d",
    "flash-attn",
    "flash-linear-attention",
)


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def check_version(distribution: str, expected: str) -> bool:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        fail(f"missing Python distribution: {distribution}=={expected}")
        return False
    # PyTorch CPU/base wheels may add a local suffix such as "+cpu" while
    # remaining ABI-compatible with torch_npu's required public version.
    public_actual = actual.split("+", maxsplit=1)[0]
    if public_actual != expected:
        fail(f"{distribution} version is {actual}; expected exactly {expected}")
        return False
    ok(f"{distribution}=={actual}")
    return True


def pinned_requirements(project_root: Path) -> dict[str, str]:
    """Return every exact direct dependency from the Ascend lock file."""
    requirements_file = project_root / "requirements-ascend.txt"
    requirements: dict[str, str] = {}
    for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s;]+)", line)
        if match is None:
            fail(f"Ascend dependency must use an exact == pin: {line}")
            continue
        requirements[match.group(1)] = match.group(2)
    return requirements


def check_declared_dependencies(project_root: Path) -> bool:
    """Detect partial environments before the lifecycle decides to skip pip."""
    success = True
    for distribution, expected in pinned_requirements(project_root).items():
        success = check_version(distribution, expected) and success
    try:
        version = importlib.metadata.version("verl")
        ok(f"verl=={version} (editable project package)")
    except importlib.metadata.PackageNotFoundError:
        fail("the Vision-OPD project package is not installed; run pip install -e . --no-deps")
        success = False
    return success


def check_lifecycle_config(min_npus: int) -> bool:
    """Validate resolved one-node values before Hydra or Ray starts."""
    success = True

    def integer(name: str, default: int, minimum: int = 1) -> int:
        nonlocal success
        raw = os.environ.get(name, str(default))
        try:
            value = int(raw)
        except ValueError:
            fail(f"{name} must be an integer; got {raw!r}")
            success = False
            return default
        if value < minimum:
            fail(f"{name} must be >= {minimum}; got {value}")
            success = False
        return value

    nnodes = integer("VOPD_NNODES", 1)
    gpus_per_node = integer("VOPD_GPUS_PER_NODE", min_npus)
    train_batch = integer("VOPD_TRAIN_BATCH_SIZE", 96)
    mini_batch = integer("VOPD_PPO_MINI_BATCH_SIZE", 96)
    rollout_n = integer("VOPD_ROLLOUT_N", 8, minimum=2)
    tensor_parallel = integer("VOPD_ROLLOUT_TP_SIZE", 1)
    prompt_length = integer("VOPD_MAX_PROMPT_LENGTH", 8192)
    response_length = integer("VOPD_MAX_RESPONSE_LENGTH", 1024)
    reprompt_length = integer("VOPD_MAX_REPROMPT_LENGTH", 10240)
    token_budget = integer("VOPD_MAX_TOKENS_PER_NPU", 11264)

    if nnodes != 1:
        fail(f"the unified lifecycle supports exactly one node; got VOPD_NNODES={nnodes}")
        success = False
    if gpus_per_node != min_npus:
        fail(f"VOPD_GPUS_PER_NODE={gpus_per_node} differs from requested NPU count {min_npus}")
        success = False
    if mini_batch > train_batch or train_batch % mini_batch != 0:
        fail("VOPD_TRAIN_BATCH_SIZE must be divisible by VOPD_PPO_MINI_BATCH_SIZE")
        success = False
    if gpus_per_node % tensor_parallel != 0:
        fail("VOPD_ROLLOUT_TP_SIZE must divide VOPD_GPUS_PER_NODE")
        success = False
    required_token_budget = max(prompt_length, reprompt_length) + response_length
    if token_budget < required_token_budget:
        fail(
            f"VOPD_MAX_TOKENS_PER_NPU must be at least {required_token_budget} to cover "
            "the longest student/teacher prompt plus the response"
        )
        success = False

    steps = os.environ.get("VOPD_TOTAL_TRAINING_STEPS", "null")
    if steps != "null":
        try:
            if int(steps) < 1:
                raise ValueError
        except ValueError:
            fail("VOPD_TOTAL_TRAINING_STEPS must be 'null' or a positive integer")
            success = False

    resume_mode = os.environ.get("VOPD_RESUME_MODE", "auto")
    if resume_mode not in {"auto", "disable", "resume_path"}:
        fail("VOPD_RESUME_MODE must be auto, disable, or resume_path")
        success = False
    if resume_mode == "resume_path":
        resume_path = os.environ.get("VOPD_RESUME_FROM_PATH", "null")
        if resume_path == "null" or not Path(resume_path).is_dir():
            fail("VOPD_RESUME_FROM_PATH must be an existing checkpoint directory")
            success = False

    try:
        memory_utilization = float(os.environ.get("VOPD_ROLLOUT_MEMORY_UTILIZATION", "0.5"))
        if not 0 < memory_utilization < 1:
            raise ValueError
    except ValueError:
        fail("VOPD_ROLLOUT_MEMORY_UTILIZATION must be a number strictly between 0 and 1")
        success = False

    if success:
        ok(
            f"one-node lifecycle configuration ({gpus_per_node} NPUs, "
            f"batch={train_batch}, rollout_n={rollout_n})"
        )
    return success


def check_static(project_root: Path) -> bool:
    required_files = [
        "requirements-ascend.txt",
        "scripts/ascend_env.sh",
        "scripts/install_ascend.sh",
        "scripts/prepare_data.py",
        "scripts/bootstrap_vision_opd_ascend.sh",
        "scripts/start_vision_opd_ascend.sh",
        "scripts/run_vision_opd.sh",
        "scripts/run_vision_opd_ascend.sh",
        "scripts/smoke_test_vision_opd_ascend.sh",
        "scripts/serve_vision_opd_ascend.sh",
        "verl/utils/device.py",
        "verl/models/transformers/npu_patch.py",
        "vision_opd_ascend.env",
    ]
    success = True
    for relative_path in required_files:
        path = project_root / relative_path
        if not path.is_file():
            fail(f"required repository file is missing: {relative_path}")
            success = False
    if not success:
        return False

    requirements = (project_root / "requirements-ascend.txt").read_text(encoding="utf-8")
    for raw_line in requirements.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s;]+)", line) is None:
            fail(f"Ascend dependency must use an exact == pin: {line}")
            success = False
    for distribution, expected in EXPECTED_VERSIONS.items():
        pattern = rf"(?m)^{re.escape(distribution)}=={re.escape(expected)}$"
        if re.search(pattern, requirements) is None:
            fail(f"requirements-ascend.txt does not pin {distribution}=={expected}")
            success = False
    for distribution in CUDA_ONLY_QWEN_FAST_PATHS:
        if re.search(rf"(?m)^{re.escape(distribution)}(?:==|>=|<=|~=)", requirements):
            fail(f"requirements-ascend.txt includes CUDA-only package: {distribution}")
            success = False
    launcher = (project_root / "scripts/run_vision_opd.sh").read_text(encoding="utf-8")
    required_overrides = [
        "trainer.device=npu",
        "attn_implementation=sdpa",
        "use_torch_compile=False",
        "rollout.enforce_eager=True",
        "rollout.load_format=safetensors",
        "trainer.total_training_steps=",
        "trainer.resume_mode=",
    ]
    for override in required_overrides:
        if override not in launcher:
            fail(f"Ascend safety override is missing from training launcher: {override}")
            success = False
    job_entry = (project_root / "scripts/start_vision_opd_ascend.sh").read_text(encoding="utf-8")
    job_entry_stages = [
        "RANK_ID",
        "ASCEND_DEVICE_ID",
        "MA_NUM_GPUS",
        "MA_NUM_HOSTS",
        "VOPD_TRAIN_FILE",
        "VOPD_CACHE_DIR",
        "check_ascend_env.py",
        "run_vision_opd_ascend.sh",
    ]
    for stage in job_entry_stages:
        if stage not in job_entry:
            fail(f"ModelArts job entry is missing stage: {stage}")
            success = False
    lifecycle_stages = [
        "vision_opd_ascend.env",
        "install_ascend.sh",
        "prepare_data.py",
        "run_vision_opd_ascend.sh",
        "merge_checkpoint.sh",
    ]
    for stage in lifecycle_stages:
        if stage not in job_entry:
            fail(f"unified Ascend lifecycle is missing stage: {stage}")
            success = False
    env_template = (project_root / "vision_opd_ascend.env").read_text(encoding="utf-8")
    for variable in (
        "VOPD_MODEL_PATH",
        "VOPD_TRAIN_FILE",
        "VOPD_OUTPUT_DIR",
        "VOPD_LR",
        "VOPD_TOTAL_TRAINING_STEPS",
        "VOPD_MAX_TOKENS_PER_NPU",
        "VOPD_RESUME_MODE",
    ):
        if variable not in env_template:
            fail(f"Ascend lifecycle configuration is missing: {variable}")
            success = False
    npu_patch = (project_root / "verl/models/transformers/npu_patch.py").read_text(encoding="utf-8")
    if "_disable_qwen3_5_cuda_fast_path" not in npu_patch:
        fail("Qwen3.5 CUDA fast-path guard is missing from the NPU patch")
        success = False
    transformers_init = (project_root / "verl/models/transformers/__init__.py").read_text(encoding="utf-8")
    if "NPU_PATCH_LOADED" not in transformers_init or "npu_patch" not in transformers_init:
        fail("Ascend transformer patches are not connected to model package initialization")
        success = False
    checkpoint_manager = (project_root / "verl/utils/checkpoint/fsdp_checkpoint_manager.py").read_text(
        encoding="utf-8"
    )
    if "OFFLOAD_STATE_DICT_TO_CPU = is_cuda_available or is_npu_available" not in checkpoint_manager:
        fail("FSDP checkpoint CPU offload is not enabled for NPU")
        success = False
    if success:
        ok("repository Ascend launch configuration")
    return success


def check_runtime(project_root: Path, min_npus: int) -> bool:
    success = check_declared_dependencies(project_root)
    success = check_lifecycle_config(min_npus) and success
    if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
        fail(f"Python {platform.python_version()} is unsupported; use Python 3.10 or 3.11")
        success = False
    else:
        ok(f"Python {platform.python_version()}")

    for distribution, expected in EXPECTED_VERSIONS.items():
        success = check_version(distribution, expected) and success

    for distribution in CUDA_ONLY_QWEN_FAST_PATHS:
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        fail(
            f"CUDA-only optional package {distribution}=={installed_version} is installed; "
            "use a clean Ascend environment"
        )
        success = False

    if not os.environ.get("ASCEND_HOME_PATH"):
        fail("ASCEND_HOME_PATH is unset; source the CANN set_env.sh first")
        success = False
    else:
        ok(f"ASCEND_HOME_PATH={os.environ['ASCEND_HOME_PATH']}")

    try:
        torch = importlib.import_module("torch")
        importlib.import_module("torch_npu")
        importlib.import_module("vllm_ascend")
    except Exception as exc:
        fail(f"failed to import Ascend runtime packages: {exc}")
        return False

    try:
        device_utils = importlib.import_module("verl.utils.device")
        if device_utils.get_device_name() != "npu":
            fail(f"verl selected {device_utils.get_device_name()!r} instead of 'npu'")
            success = False
        elif device_utils.get_nccl_backend() != "hccl":
            fail(f"verl selected {device_utils.get_nccl_backend()!r} instead of 'hccl'")
            success = False
        else:
            ok("verl NPU device and HCCL backend selection")
        model_patches = importlib.import_module("verl.models.transformers")
        if not model_patches.NPU_PATCH_LOADED:
            fail("verl Ascend transformer patches were not loaded")
            success = False
        else:
            ok("verl Ascend transformer patches loaded")
    except Exception as exc:
        fail(f"failed to import verl with its Ascend model patches: {exc}")
        success = False

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        fail("torch.npu.is_available() is false")
        success = False
    else:
        count = torch.npu.device_count()
        if count < min_npus:
            fail(f"only {count} NPU(s) are visible; {min_npus} required")
            success = False
        else:
            ok(f"{count} visible Ascend NPU(s)")

        try:
            device = torch.device("npu:0")
            lhs = torch.randn((32, 32), device=device, dtype=torch.bfloat16, requires_grad=True)
            rhs = torch.randn((32, 32), device=device, dtype=torch.bfloat16)
            lhs.matmul(rhs).float().mean().backward()
            torch.npu.synchronize()
            ok("NPU BF16 forward/backward smoke test")
        except Exception as exc:
            fail(f"NPU BF16 forward/backward smoke test failed: {exc}")
            success = False

    if shutil.which("npu-smi"):
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=False
        )
        if result.returncode == 0:
            ok("npu-smi driver query")
        else:
            fail(f"npu-smi exited with code {result.returncode}: {result.stderr.strip()}")
            success = False
    else:
        fail("npu-smi was not found in PATH")
        success = False

    if shutil.which("vllm"):
        ok("vllm command is available")
    else:
        fail("vllm command was not found in PATH")
        success = False

    # verl schedules NPU workers through Ray's custom "NPU" resource. Merely
    # seeing devices in torch_npu is insufficient: a missing Ray resource would
    # leave the workers pending forever.
    ray_started_here = False
    try:
        ray = importlib.import_module("ray")
        if not ray.is_initialized():
            ray.init(include_dashboard=False, logging_level="ERROR")
            ray_started_here = True
        ray_npus = float(ray.cluster_resources().get("NPU", 0))
        if ray_npus < min_npus:
            fail(
                f"Ray advertises {ray_npus:g} NPU resource(s), but training requests {min_npus}; "
                "check ASCEND_RT_VISIBLE_DEVICES and Ray's NPU accelerator support"
            )
            success = False
        else:
            ok(f"Ray advertises {ray_npus:g} NPU resource(s)")
    except Exception as exc:
        fail(f"Ray local NPU resource check failed: {exc}")
        success = False
    finally:
        if ray_started_here:
            ray.shutdown()

    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="check repository files without importing NPU packages",
    )
    parser.add_argument(
        "--dependencies-only",
        action="store_true",
        help="check all exact Python pins and the editable project package",
    )
    parser.add_argument("--min-npus", type=int, default=1, help="minimum visible NPU count")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_npus < 1:
        fail("--min-npus must be at least 1")
        return 2
    project_root = Path(__file__).resolve().parents[1]
    print(f"Vision-OPD Ascend preflight on {platform.system()} {platform.machine()}")
    success = check_static(project_root)
    if args.dependencies_only:
        success = check_declared_dependencies(project_root) and success
    elif not args.static_only:
        success = check_runtime(project_root, args.min_npus) and success
    if success:
        ok("all requested Ascend checks passed")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
