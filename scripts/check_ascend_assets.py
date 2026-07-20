#!/usr/bin/env python3
"""Validate offline Vision-OPD assets without importing third-party packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


LOCK_FILES = (
    "requirements-ascend.txt",
    "requirements-ascend-core.txt",
    "requirements-ascend-plugins.txt",
)
MODEL_MANIFEST = ".vision_opd_model_manifest.json"
WHEELHOUSE_MANIFEST = ".vision_opd_wheelhouse_manifest.json"


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)


def check_model(
    model_dir: Path,
    *,
    expected_repo_id: str | None = None,
    expected_revision: str | None = None,
    require_manifest: bool = False,
) -> bool:
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

    manifest_path = model_dir / MODEL_MANIFEST
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != 1:
                raise ValueError("unsupported schema")
            repo_id = manifest["repo_id"]
            requested_revision = manifest["requested_revision"]
            resolved_revision = manifest["resolved_revision"]
            if not isinstance(repo_id, str) or not repo_id:
                raise ValueError("repo_id must be a non-empty string")
            if not isinstance(requested_revision, str) or not requested_revision:
                raise ValueError("requested_revision must be a non-empty string")
            if not isinstance(resolved_revision, str) or not resolved_revision:
                raise ValueError("resolved_revision must be a non-empty string")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            fail(f"invalid model asset manifest {manifest_path}: {exc}")
            success = False
        else:
            if expected_repo_id and repo_id != expected_repo_id:
                fail(f"model repository mismatch: manifest has {repo_id}, expected {expected_repo_id}")
                success = False
            if expected_revision and requested_revision != expected_revision:
                fail(
                    "requested model revision mismatch: "
                    f"manifest has {requested_revision}, expected {expected_revision}"
                )
                success = False
            if (
                expected_revision
                and re.fullmatch(r"[0-9a-fA-F]{40}", expected_revision)
                and resolved_revision.lower() != expected_revision.lower()
            ):
                fail(
                    "resolved model commit mismatch: "
                    f"manifest has {resolved_revision}, expected {expected_revision}"
                )
                success = False
    elif require_manifest:
        fail(
            f"model asset manifest is missing: {manifest_path}; "
            "run scripts/prepare_ascend_assets.sh first"
        )
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


def requirements_digest(project_root: Path) -> str:
    digest = hashlib.sha256()
    for filename in LOCK_FILES:
        path = project_root / filename
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def wheel_is_cp310_aarch64_compatible(filename: str) -> bool:
    """Check wheel tags without importing packaging on the preparation host."""
    if not filename.lower().endswith(".whl"):
        return False
    try:
        _, python_tag, abi_tag, platform_tag = filename[:-4].rsplit("-", 3)
    except ValueError:
        return False

    platforms = platform_tag.lower().split(".")
    if "any" not in platforms and not any(tag.endswith("aarch64") for tag in platforms):
        return False

    python_tags = python_tag.lower().split(".")
    abi_tags = abi_tag.lower().split(".")
    for tag in python_tags:
        if tag == "py3" and "none" in abi_tags:
            return True
        # pip's CPython 3.10 compatible tag set includes pure-Python wheels
        # tagged for an earlier Python 3 minor (for example py37-none-any).
        # These contain no native ABI, while py311+ remains incompatible.
        py_match = re.fullmatch(r"py(\d)(\d+)", tag)
        if py_match is not None and "none" in abi_tags:
            py_version = (int(py_match.group(1)), int(py_match.group(2)))
            if py_version[0] == 3 and py_version <= (3, 10):
                return True
        match = re.fullmatch(r"cp(\d)(\d+)", tag)
        if match is None:
            continue
        version = (int(match.group(1)), int(match.group(2)))
        if version == (3, 10) and any(abi in {"cp310", "abi3", "none"} for abi in abi_tags):
            return True
        if version <= (3, 10) and "abi3" in abi_tags:
            return True
    return False


def write_wheelhouse_manifest(project_root: Path, wheel_dir: Path) -> Path:
    wheel_paths = sorted(path for path in wheel_dir.iterdir() if path.suffix.lower() == ".whl")
    manifest = {
        "schema": 1,
        "resolution_complete": True,
        "python": "3.10",
        "platform": "aarch64",
        "requirements_sha256": requirements_digest(project_root),
        "wheels": [
            {
                "filename": path.name,
                "size": path.stat().st_size,
                "sha256": file_digest(path),
            }
            for path in wheel_paths
        ],
    }
    manifest_path = wheel_dir / WHEELHOUSE_MANIFEST
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest_path


def check_wheelhouse_manifest(project_root: Path, wheel_dir: Path, files: list[str]) -> bool:
    manifest_path = wheel_dir / WHEELHOUSE_MANIFEST
    if not manifest_path.is_file():
        fail(
            f"resolved wheelhouse manifest is missing: {manifest_path}; "
            "run scripts/prepare_ascend_assets.sh before --check-only or training"
        )
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != 1 or manifest.get("resolution_complete") is not True:
            raise ValueError("unsupported schema or incomplete pip resolution")
        if manifest.get("python") != "3.10" or manifest.get("platform") != "aarch64":
            raise ValueError("manifest target must be Python 3.10/aarch64")
        if manifest.get("requirements_sha256") != requirements_digest(project_root):
            raise ValueError("requirements locks changed after the wheelhouse was resolved")
        entries = manifest["wheels"]
        if not isinstance(entries, list):
            raise TypeError("wheels must be a list")
        recorded = {entry["filename"]: entry for entry in entries}
        if len(recorded) != len(entries):
            raise ValueError("duplicate wheel filenames are present")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"invalid resolved wheelhouse manifest {manifest_path}: {exc}")
        return False

    actual_names = set(files)
    recorded_names = set(recorded)
    if actual_names != recorded_names:
        missing = sorted(recorded_names - actual_names)
        extra = sorted(actual_names - recorded_names)
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:10]))
        if extra:
            details.append("unrecorded=" + ", ".join(extra[:10]))
        fail("wheelhouse inventory differs from the resolved manifest: " + "; ".join(details))
        return False

    success = True
    for filename in sorted(recorded):
        path = wheel_dir / filename
        entry = recorded[filename]
        if entry.get("size") != path.stat().st_size:
            fail(f"wheel size differs from manifest: {filename}")
            success = False
            continue
        if entry.get("sha256") != file_digest(path):
            fail(f"wheel SHA-256 differs from manifest: {filename}")
            success = False
    return success


def check_wheelhouse(
    project_root: Path,
    wheel_dir: Path,
    *,
    require_manifest: bool = False,
    check_existing_manifest: bool = True,
) -> bool:
    if not wheel_dir.is_dir():
        fail(f"offline wheelhouse does not exist: {wheel_dir}")
        return False

    wheel_paths = sorted(path for path in wheel_dir.iterdir() if path.suffix.lower() == ".whl")
    files = [path.name for path in wheel_paths]
    lowered = [name.lower() for name in files]
    success = True
    encoded_names = [name for name in files if "%2b" in name.lower()]
    if encoded_names:
        fail(
            "wheel filenames must contain '+' rather than the URL escape '%2B': "
            + ", ".join(encoded_names[:5])
        )
        success = False

    invalid_archives = [path.name for path in wheel_paths if not zipfile.is_zipfile(path)]
    if invalid_archives:
        fail(
            "invalid or truncated wheel archives are present: "
            + ", ".join(invalid_archives[:10])
        )
        success = False

    incompatible_wheels = [
        path.name for path in wheel_paths if not wheel_is_cp310_aarch64_compatible(path.name)
    ]
    if incompatible_wheels:
        fail(
            "wheelhouse contains wheels incompatible with CPython 3.10/aarch64: "
            + ", ".join(incompatible_wheels[:20])
        )
        success = False

    pip_versions = []
    for name in lowered:
        match = re.match(r"^pip-([0-9]+(?:\.[0-9]+)*)-", name)
        if match:
            pip_versions.append(match.group(1))
    if not any(version_tuple(version) >= (23, 3) and version_tuple(version) < (26,) for version in pip_versions):
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

    if require_manifest or (
        check_existing_manifest and (wheel_dir / WHEELHOUSE_MANIFEST).is_file()
    ):
        success = check_wheelhouse_manifest(project_root, wheel_dir, files) and success

    if success:
        print(f"[ OK ] offline wheelhouse direct assets: {wheel_dir}")
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--expected-model-repo-id")
    parser.add_argument("--expected-model-revision")
    parser.add_argument(
        "--require-manifests",
        action="store_true",
        help="require preparation manifests and validate their target, locks and SHA-256 inventory",
    )
    parser.add_argument(
        "--write-wheel-manifest",
        action="store_true",
        help="record a wheelhouse after pip download resolved successfully",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    success = True
    if args.model_dir is not None:
        success = check_model(
            args.model_dir.resolve(),
            expected_repo_id=args.expected_model_repo_id,
            expected_revision=args.expected_model_revision,
            require_manifest=args.require_manifests,
        ) and success
    if args.wheel_dir is not None:
        wheel_dir = args.wheel_dir.resolve()
        success = check_wheelhouse(
            project_root,
            wheel_dir,
            require_manifest=args.require_manifests and not args.write_wheel_manifest,
            check_existing_manifest=not args.write_wheel_manifest,
        ) and success
        if args.write_wheel_manifest and success:
            manifest_path = write_wheelhouse_manifest(project_root, wheel_dir)
            print(f"[ OK ] wrote resolved wheelhouse manifest: {manifest_path}")
    elif args.write_wheel_manifest:
        fail("--write-wheel-manifest requires --wheel-dir")
        return 2
    if args.model_dir is None and args.wheel_dir is None:
        fail("at least one of --model-dir or --wheel-dir is required")
        return 2
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
