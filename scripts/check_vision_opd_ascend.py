"""Pure-static validation for the Ascend adaptation (no imports or NPU work)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()

    required_files = [
        "vision_opd_ascend.env",
        "requirements-ascend.txt",
        "constraints-user-environment.txt",
        "scripts/start_vision_opd_ascend.sh",
        "scripts/run_inference_ascend.sh",
        "scripts/infer_vision_opd_ascend.py",
        "docs/ascend_910b.md",
    ]
    for relative in required_files:
        assert (root / relative).is_file(), f"missing {relative}"

    compiled = 0
    for path in [root / "verl", root / "scripts"]:
        for python_file in path.rglob("*.py"):
            compile(python_file.read_text(encoding="utf-8"), str(python_file), "exec")
            compiled += 1

    requirements = (root / "requirements-ascend.txt").read_text(encoding="utf-8")
    install_lines = [line.strip().lower() for line in requirements.splitlines() if line and not line.startswith("#")]
    banned_prefixes = ("torch==", "torch-npu", "vllm", "nvidia-", "cupy", "flash-attn", "xformers", "triton==")
    assert not [line for line in install_lines if line.startswith(banned_prefixes)]

    launcher = (root / "scripts/start_vision_opd_ascend.sh").read_text(encoding="utf-8")
    env_text = (root / "vision_opd_ascend.env").read_text(encoding="utf-8")

    probe_marker = '"${VOPD_PYTHON}" - <<\'PY\''
    assert probe_marker in launcher, "missing warning-only version probe"
    version_probe = launcher.split(probe_marker, maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0].lstrip("\n")
    compile(version_probe, "start_vision_opd_ascend.sh:version_probe", "exec")
    assert "importlib.metadata" in version_probe, "version probe must use distribution metadata"
    native_probe_imports = ("import torch", "import torch_npu", "import ray", "import transformers")
    assert not any(value in version_probe for value in native_probe_imports), (
        "warning-only version probe must not import native/runtime packages"
    )
    assert "PYTHONFAULTHANDLER=1" in launcher, "native training failures should emit Python fault diagnostics"
    assert '${NPU_ASD_CONFIG:=enable:false}' in env_text, (
        "torch-npu 2.6 must skip the disabled optional ASD native capability probe"
    )

    used = set(re.findall(r"\$\{(VOPD_[A-Z0-9_]+)", launcher))
    defined = set(re.findall(r"\$\{(VOPD_[A-Z0-9_]+):=", env_text))
    assert not sorted(used - defined - {"VOPD_MASTER_ADDR"}), "launcher contains undefined VOPD variables"

    invariants = [
        "actor_rollout_ref.rollout.name=${VOPD_ROLLOUT_BACKEND}",
        "trainer.device=npu",
        "teacher_model_source=legacy",
        "teacher_regularization=ema",
        "self_distillation.alpha=0.5",
        "trainer.save_freq=${VOPD_SAVE_FREQ}",
    ]
    for invariant in invariants:
        assert invariant in launcher, f"missing launch invariant: {invariant}"

    forbidden_paths = ("/home/ma-user/", "/opt/huawei/dataset/")
    for relative in ("scripts/start_vision_opd_ascend.sh", "vision_opd_ascend.env"):
        text = (root / relative).read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden_paths), f"hard-coded platform path in {relative}"

    print(f"Static Ascend validation passed: {compiled} Python files, {len(used)} launcher variables.")
    print("No package import, model load, NPU operation, training, inference, or download was executed.")


if __name__ == "__main__":
    main()
