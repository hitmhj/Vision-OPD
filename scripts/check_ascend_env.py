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
    "torchvision": "0.24.0",
    "torchaudio": "2.9.0",
    "torch-npu": "2.9.0.post1+git4c901a4",
    "torchdata": "0.11.0",
    "triton-ascend": "3.2.0.dev20260322",
    "vllm": "0.18.0",
    "vllm-ascend": "0.18.0",
    "transformers": "5.5.0",
}

LOCK_FILES = (
    "requirements-ascend.txt",
    "requirements-ascend-core.txt",
    "requirements-ascend-plugins.txt",
)

# vLLM 0.18 was published with CUDA/PyTorch-2.10 metadata. The official
# vLLM-Ascend 0.18 Atlas A2 matrix deliberately replaces these generic
# CUDA/plugin metadata dependencies.
# Only these owner/dependency pairs may be ignored; every other pip-check
# failure remains fatal.
ALLOWED_METADATA_DIVERGENCES = {
    "vllm": {
        "flashinfer-python",
        "nvidia-cudnn-frontend",
        "nvidia-cutlass-dsl",
        "opencv-python-headless",
        "quack-kernels",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
    },
    "vllm-ascend": {
        "torch-npu",
        "triton-ascend",
    },
}

CUDA_ONLY_QWEN_FAST_PATHS = (
    "causal-conv1d",
    "flash-attn",
    "flash-linear-attention",
)

RUNTIME_IMPORTS = {
    "accelerate": "accelerate",
    "codetiming": "codetiming",
    "datasets": "datasets",
    "hydra-core": "hydra",
    "omegaconf": "omegaconf",
    "peft": "peft",
    "Pillow": "PIL",
    "pyarrow": "pyarrow",
    "pyzmq": "zmq",
    "qwen-vl-utils": "qwen_vl_utils",
    "ray": "ray",
    "safetensors": "safetensors",
    "tensordict": "tensordict",
    "torch": "torch",
    "torch-npu": "torch_npu",
    "torchdata": "torchdata",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "triton-ascend": "triton",
    "vllm": "vllm",
    "vllm-ascend": "vllm_ascend",
}


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
    comparison_actual = actual if "+" in expected else actual.split("+", maxsplit=1)[0]
    if comparison_actual != expected:
        fail(f"{distribution} version is {actual}; expected exactly {expected}")
        return False
    ok(f"{distribution}=={actual}")
    return True


def pinned_requirements(project_root: Path) -> dict[str, str]:
    """Return every exact direct dependency from all three Ascend locks."""
    requirements: dict[str, str] = {}
    for relative_path in LOCK_FILES:
        requirements_file = project_root / relative_path
        for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s;]+)", line)
            if match is None:
                fail(f"Ascend dependency must use an exact == pin ({relative_path}): {line}")
                continue
            distribution, version = match.groups()
            canonical = distribution.lower().replace("_", "-")
            previous = requirements.get(canonical)
            if previous is not None and previous != version:
                fail(f"conflicting direct pins for {canonical}: {previous} and {version}")
                continue
            requirements[canonical] = version
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


def check_runtime_imports() -> bool:
    """Import the packages used by the active Vision-OPD training path."""
    success = True
    for distribution, module_name in RUNTIME_IMPORTS.items():
        try:
            importlib.import_module(module_name)
            version = importlib.metadata.version(distribution)
            ok(f"runtime import: {distribution}=={version}")
        except Exception as exc:
            fail(f"runtime cannot import {distribution}: {exc}")
            success = False
    return success


def check_qwen35_transformers_api() -> bool:
    """Verify the exact Transformers API patched by the Vision-OPD trainer."""
    try:
        module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        required_symbols = (
            "Qwen3_5CausalLMOutputWithPast",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5Model",
            "Qwen3_5TextModel",
            "Qwen3_5VisionModel",
        )
        missing = [name for name in required_symbols if not hasattr(module, name)]
        if missing:
            fail(f"Transformers Qwen3.5 API is missing: {', '.join(missing)}")
            return False
    except Exception as exc:
        fail(f"cannot load the Transformers Qwen3.5 implementation: {exc}")
        return False
    ok("Transformers Qwen3.5 training API")
    return True


def check_vllm_ascend_registration() -> bool:
    """Ensure vLLM selected the Ascend plugin instead of its CUDA platform."""
    try:
        importlib.import_module("vllm_ascend")
        platforms = importlib.import_module("vllm.platforms")
        current_platform = platforms.current_platform
        device_type = getattr(current_platform, "device_type", None)
        if device_type != "npu":
            fail(
                f"vLLM selected platform {type(current_platform).__name__} "
                f"(device_type={device_type!r}) instead of Ascend NPU"
            )
            return False
    except Exception as exc:
        fail(f"vLLM Ascend plugin registration failed: {exc}")
        return False
    ok(f"vLLM Ascend platform registration ({type(current_platform).__name__})")
    return True


def check_metadata_consistency() -> bool:
    """Run pip check while accepting only documented accelerator divergences."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output_lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    if result.returncode == 0:
        ok("Python package metadata consistency")
        return True

    success = True
    pattern = re.compile(
        r"^(?P<owner>[A-Za-z0-9_.-]+)\s+\S+\s+(?:has requirement|requires)\s+"
        r"(?P<dependency>[A-Za-z0-9_.-]+)",
        re.IGNORECASE,
    )
    for line in output_lines:
        match = pattern.match(line)
        if match is None:
            fail(f"unrecognized pip-check failure: {line}")
            success = False
            continue
        owner = match.group("owner").lower().replace("_", "-")
        dependency = match.group("dependency").lower().replace("_", "-")
        if dependency in ALLOWED_METADATA_DIVERGENCES.get(owner, set()):
            ok(f"documented Ascend metadata override: {owner} -> {dependency}")
        else:
            fail(f"unexpected dependency conflict: {line}")
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

    rollout_engine = os.environ.get("VOPD_ROLLOUT_ENGINE", "vllm")
    if rollout_engine != "vllm":
        fail(
            f"VOPD_ROLLOUT_ENGINE={rollout_engine!r} is unavailable in the Ascend lock; "
            "use 'vllm'"
        )
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
        "requirements-ascend-core.txt",
        "requirements-ascend-plugins.txt",
        "scripts/ascend_env.sh",
        "scripts/install_ascend.sh",
        "scripts/check_ascend_assets.py",
        "scripts/prepare_ascend_assets.sh",
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

    lock_contents = {
        relative_path: (project_root / relative_path).read_text(encoding="utf-8")
        for relative_path in LOCK_FILES
    }
    requirements = "\n".join(lock_contents.values())
    for relative_path, content in lock_contents.items():
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            if re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s;]+)", line) is None:
                fail(f"Ascend dependency must use an exact == pin ({relative_path}): {line}")
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
    forbidden_sources = (
        "download.pytorch.org",
        "mirrors.huaweicloud.com/ascend/repos/pypi",
        "/whl/cpu",
    )
    for forbidden_source in forbidden_sources:
        if forbidden_source in requirements:
            fail(f"Ascend locks contain unreachable or CUDA-oriented source: {forbidden_source}")
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
        "check_ascend_assets.py",
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
    if "prepare_ascend_assets.sh" in job_entry:
        fail("training entry must not download or prepare portable assets")
        success = False
    dependency_installer = (project_root / "scripts/install_ascend.sh").read_text(encoding="utf-8")
    for stage in ("VOPD_VENV_DIR", "VOPD_ASCEND_REQUIREMENTS", "VOPD_BOOTSTRAP_PYTHON"):
        if stage not in dependency_installer:
            fail(f"NPU-worker dependency installer is missing: {stage}")
            success = False
    for forbidden in ("VOPD_DEPENDENCY_SETUP_DIR", "VOPD_DEPENDENCY_SETUP_SCRIPT", "init_env_qwen"):
        if forbidden in dependency_installer:
            fail(f"dependency installer still depends on the LLaMAFactory sample: {forbidden}")
            success = False
    for required in (
        "requirements-ascend.txt",
        "requirements-ascend-core.txt",
        "requirements-ascend-plugins.txt",
        "-m venv",
        "-m pip install",
        "--constraint",
        "--dry-run",
        "check_ascend_assets.py",
        "--require-manifests",
        "--only-binary=:all:",
        "--no-deps",
        '--editable "$PROJECT_ROOT"',
        "--dependencies-only",
    ):
        if required not in dependency_installer:
            fail(f"dependency installer is missing lifecycle operation: {required}")
            success = False

    asset_preparer = (project_root / "scripts/prepare_ascend_assets.sh").read_text(
        encoding="utf-8"
    )
    for required in (
        "VOPD_PREPARE_ONLINE",
        "VOPD_INTERNAL_WHEEL_DIRS",
        "VOPD_MODEL_SOURCE_DIR",
        "--online",
        "--check-only",
        "snapshot_download",
        "pip download",
        "check_ascend_assets.py",
        "--write-wheel-manifest",
        "--require-manifests",
        "--expected-model-repo-id",
        "--platform manylinux_2_28_aarch64",
        "--python-version 3.10",
        "--implementation cp",
        "--abi cp310",
        'PREPARE_MODE="cross"',
    ):
        if required not in asset_preparer:
            fail(f"portable asset preparer is missing: {required}")
            success = False
    if "Wheel preparation must run with Python 3.10 on aarch64" in asset_preparer:
        fail("portable asset preparation still blocks the x86_64/Python 3.9 WebStudio host")
        success = False

    runtime_loader = (project_root / "scripts/ascend_env.sh").read_text(encoding="utf-8")
    vendor_sources = [
        'source "$CANN_ENV_SCRIPT"',
        'source "$NNAL_ENV_SCRIPT" --cxx_abi=0',
        'source "$ASDSIP_ENV_SCRIPT"',
    ]
    source_positions = [runtime_loader.find(source_line) for source_line in vendor_sources]
    if any(position < 0 for position in source_positions) or source_positions != sorted(source_positions):
        fail("Ascend vendor scripts must follow the sample CANN -> ATB -> ASDSIP order")
        success = False
    if re.search(r"(?m)^\s*(?:source|_vision_opd_source)[^#\n]*/usr/local/Ascend", runtime_loader):
        fail("Ascend runtime loader must not fall back to a different /usr/local stack")
        success = False
    if "set +u" not in runtime_loader or "VOPD_ASCEND_ENV_READY=1" not in runtime_loader:
        fail("Ascend runtime loader lacks nounset compatibility or its one-time load guard")
        success = False
    sample_exports = [
        "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "ASCEND_SLOG_PRINT_TO_STDOUT=0",
        "ASCEND_GLOBAL_LOG_LEVEL=3",
        "TASK_QUEUE_ENABLE=2",
        "TASK_QUEUE=0",
        "COMBINED_ENABLE=1",
        "CPU_AFFINITY_CONF=1",
        "HCCL_ASYNC_ERROR_HANDLING=0",
        "HCCL_IF_BASE_PORT=64000",
        "HCCL_CONNECT_TIMEOUT=7200",
        "HCCL_EXEC_TIMEOUT=18000",
        "HCCL_EXEC_TIMEOT=3600",
        "HCCL_CONNECT_TIMEOT=3600",
        "ASCEND_LAUNCH_BLOCKING=0",
        "ACLNN_CACHE_LIMIT=100000",
        'PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"',
    ]
    for sample_export in sample_exports:
        if sample_export not in runtime_loader:
            fail(f"Ascend runtime loader is missing sample setting: {sample_export}")
            success = False

    vendor_entry_files = [
        "scripts/start_vision_opd_ascend.sh",
        "scripts/install_ascend.sh",
        "scripts/run_vision_opd_ascend.sh",
        "scripts/serve_vision_opd_ascend.sh",
    ]
    for relative_path in vendor_entry_files:
        entry_text = (project_root / relative_path).read_text(encoding="utf-8")
        if re.search(r"(?m)^\s*set\s+-[^\n#]*u", entry_text):
            fail(f"Huawei environment entry must not enable Bash nounset: {relative_path}")
            success = False
    blocking_position = job_entry.find("export ASCEND_LAUNCH_BLOCKING=1")
    training_position = job_entry.find('run_vision_opd_ascend.sh" "$@"')
    if blocking_position < 0 or training_position < 0 or blocking_position > training_position:
        fail("ASCEND_LAUNCH_BLOCKING=1 must be set immediately before Vision-OPD training")
        success = False
    ascend_runner = (project_root / "scripts/run_vision_opd_ascend.sh").read_text(encoding="utf-8")
    if "VOPD_ASCEND_ENV_READY" not in ascend_runner or "PROJECT_ROOT" not in ascend_runner:
        fail("Ascend training wrapper lacks the one-time environment guard or relocatable project root")
        success = False
    env_template = (project_root / "vision_opd_ascend.env").read_text(encoding="utf-8")
    for variable in (
        "VOPD_ASSET_ROOT",
        "VOPD_MODEL_PATH",
        "VOPD_REQUIRE_LOCAL_MODEL",
        "VOPD_HF_OFFLINE",
        "VOPD_TRAIN_FILE",
        "VOPD_OUTPUT_DIR",
        "VOPD_LR",
        "VOPD_TOTAL_TRAINING_STEPS",
        "VOPD_MAX_TOKENS_PER_NPU",
        "VOPD_ROLLOUT_ENGINE",
        "VOPD_RESUME_MODE",
        "VOPD_INSTALL_MODE",
        "VOPD_VENV_DIR",
        "VOPD_ASCEND_REQUIREMENTS",
        "VOPD_ASCEND_CORE_REQUIREMENTS",
        "VOPD_ASCEND_PLUGIN_REQUIREMENTS",
        "VOPD_BOOTSTRAP_PYTHON",
        "VOPD_LOCAL_WHEEL_DIR",
        "VOPD_PREPARE_ONLINE",
        "VOPD_INTERNAL_WHEEL_DIRS",
        "VOPD_MODEL_SOURCE_DIR",
        "VOPD_MODEL_REPO_ID",
        "VOPD_MODEL_REVISION",
        "DO_NOT_TRACK",
        "HF_HUB_DISABLE_TELEMETRY",
        "VLLM_NO_USAGE_STATS",
        "RAY_USAGE_STATS_ENABLED",
        "VOPD_PREPARE_PIP_INDEX_URL",
        "VOPD_PIP_INDEX_URL",
        "VOPD_PIP_EXTRA_INDEX_URL",
        "VOPD_PIP_TRUSTED_HOST",
        "VOPD_PIP_CONFIG_FILE",
        "VOPD_PIP_NO_INDEX",
        "VOPD_EXPECTED_CANN_VERSION",
        "VOPD_REQUIRE_CANN_VERSION_MATCH",
        "CANN_ENV_SCRIPT",
        "NNAL_ENV_SCRIPT",
        "ASDSIP_ENV_SCRIPT",
        "VOPD_REQUIRE_ASDSIP",
    ):
        if variable not in env_template:
            fail(f"Ascend lifecycle configuration is missing: {variable}")
            success = False
    offline_defaults = (
        'VOPD_ASSET_ROOT="${VOPD_ASSET_ROOT:-envs}"',
        'VOPD_PREPARE_ONLINE="${VOPD_PREPARE_ONLINE:-0}"',
        'VOPD_PIP_NO_INDEX="${VOPD_PIP_NO_INDEX:-1}"',
        'VOPD_HF_OFFLINE="${VOPD_HF_OFFLINE:-1}"',
        'VOPD_REQUIRE_LOCAL_MODEL="${VOPD_REQUIRE_LOCAL_MODEL:-1}"',
        'VOPD_EXPECTED_CANN_VERSION="${VOPD_EXPECTED_CANN_VERSION:-8.5.1}"',
        'DO_NOT_TRACK="${DO_NOT_TRACK:-1}"',
        'HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"',
        'VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"',
        'RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"',
    )
    for default in offline_defaults:
        if default not in env_template:
            fail(f"Ascend production default is not locked: {default}")
            success = False
    revision_match = re.search(
        r'VOPD_MODEL_REVISION="\$\{VOPD_MODEL_REVISION:-([^}]+)\}"', env_template
    )
    if revision_match is None or revision_match.group(1) == "main" or not re.fullmatch(
        r"[0-9a-f]{40}", revision_match.group(1)
    ):
        fail("VOPD_MODEL_REVISION must default to an immutable 40-character commit hash")
        success = False

    rank_position = job_entry.find('if [[ "${VOPD_SINGLE_DRIVER_GUARD:-1}" == "1" ]]')
    worker_position = job_entry.find('command -v npu-smi')
    model_position = job_entry.find('check_ascend_assets.py')
    installer_position = job_entry.find('install_ascend.sh')
    if min(rank_position, worker_position, model_position, installer_position) < 0 or not (
        rank_position < worker_position < model_position < installer_position
    ):
        fail("job entry order must be rank guard -> NPU worker guard -> model check -> install")
        success = False

    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
    for legacy_asset_rule in (".venv-ascend/", "models/*", "whls/*"):
        if legacy_asset_rule not in gitignore:
            fail(f".gitignore lacks the legacy asset safety rule: {legacy_asset_rule}")
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
    # The lock file and isolated venv are owned by this repository. Check the
    # full lock before importing binary modules so mixed base-image packages
    # cannot silently leak into the training process.
    success = check_declared_dependencies(project_root)
    success = check_runtime_imports() and success
    success = check_qwen35_transformers_api() and success
    success = check_vllm_ascend_registration() and success
    success = check_metadata_consistency() and success
    success = check_lifecycle_config(min_npus) and success
    if sys.version_info[:2] != (3, 10):
        fail(f"Python {platform.python_version()} is unsupported; use the pinned Python 3.10 worker")
        success = False
    else:
        ok(f"Python {platform.python_version()}")

    if not os.environ.get("ASCEND_HOME_PATH"):
        fail("ASCEND_HOME_PATH is unset; source the CANN set_env.sh first")
        success = False
    else:
        ok(f"ASCEND_HOME_PATH={os.environ['ASCEND_HOME_PATH']}")

    expected_cann = os.environ.get("VOPD_EXPECTED_CANN_VERSION", "8.5.1")
    detected_cann = os.environ.get("VOPD_DETECTED_CANN_VERSION")
    if detected_cann != expected_cann:
        fail(f"CANN version is {detected_cann or 'unknown'}; expected exactly {expected_cann}")
        success = False
    else:
        ok(f"CANN {detected_cann}")

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
        success = check_runtime_imports() and success
        success = check_qwen35_transformers_api() and success
        success = check_vllm_ascend_registration() and success
        success = check_metadata_consistency() and success
    elif not args.static_only:
        success = check_runtime(project_root, args.min_npus) and success
    if success:
        ok("all requested Ascend checks passed")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
