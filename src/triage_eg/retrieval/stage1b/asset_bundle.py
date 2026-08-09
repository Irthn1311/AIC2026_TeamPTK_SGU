"""Build a portable official OpenAI CLIP asset bundle from existing local files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from triage_eg.retrieval.stage1b.assets import sha256_file
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl

SOURCE_ROOT_FILES = (
    "setup.py",
    "requirements.txt",
    "LICENSE",
    "LICENSE.txt",
    "README.md",
)
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "tests"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class AssetBundleConfig:
    source_root: Path
    checkpoint: Path
    output_root: Path
    source_commit: str | None = None
    overwrite: bool = False
    dry_run: bool = False
    create_zip: bool = False
    dependency_wheels: tuple[Path, ...] = ()


def _within(path: Path, root: Path) -> bool:
    resolved, parent = path.resolve(strict=False), root.resolve(strict=False)
    return resolved == parent or parent in resolved.parents


def _validate(config: AssetBundleConfig) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    source = config.source_root.expanduser().resolve(strict=True)
    checkpoint = config.checkpoint.expanduser().resolve(strict=True)
    output = config.output_root.expanduser().resolve(strict=False)
    if not (source / "clip/__init__.py").is_file():
        raise ValueError("OPENAI_CLIP_PACKAGE_INVALID")
    if not (source / "clip/bpe_simple_vocab_16e6.txt.gz").is_file():
        raise ValueError("OPENAI_CLIP_TOKENIZER_ASSET_MISSING")
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError("ENCODER_CHECKPOINT_INVALID")
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError("Asset bundle output root is too broad")
    if _within(output, source) or _within(output, checkpoint.parent):
        raise ValueError("Asset bundle output must be outside source/checkpoint roots")
    wheels = tuple(path.expanduser().resolve(strict=True) for path in config.dependency_wheels)
    if any(path.suffix.lower() != ".whl" or not path.is_file() for path in wheels):
        raise ValueError("OFFLINE_DEPENDENCY_WHEEL_INVALID")
    if len({path.name for path in wheels}) != len(wheels):
        raise ValueError("OFFLINE_DEPENDENCY_WHEEL_DUPLICATE")
    return source, checkpoint, output, wheels


def _source_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted((source / "clip").rglob("*")):
        relative = path.relative_to(source)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.is_symlink() and not _within(path.resolve(strict=True), source):
            raise ValueError(f"Source symlink escapes source root: {relative}")
        if path.is_file() and path.suffix.lower() not in EXCLUDED_SUFFIXES:
            files.append(path)
    for name in SOURCE_ROOT_FILES:
        path = source / name
        if path.is_file():
            files.append(path)
    if not any(path.name.startswith("LICENSE") for path in files):
        raise ValueError("Official source LICENSE is missing")
    return sorted(set(files), key=lambda path: path.relative_to(source).as_posix())


def _git_value(source: Path, *args: str) -> str | None:
    if not (source / ".git").exists():
        return None
    result = subprocess.run(
        ["git", *args],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _normalized_repository(value: str) -> str:
    normalized = value.strip().replace("\\", "/").lower()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized.removesuffix("/").removesuffix(".git")


def _source_provenance(
    source: Path,
    declared: str | None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_repository = "https://github.com/openai/CLIP.git"
    origin = _git_value(source, "remote", "get-url", "origin")
    if origin and _normalized_repository(origin) != _normalized_repository(canonical_repository):
        raise ValueError("OPENAI_CLIP_SOURCE_REPOSITORY_INVALID")
    actual_commit = _git_value(source, "rev-parse", "HEAD")
    if not actual_commit and existing:
        existing_repository = str(existing.get("source_repository", ""))
        existing_commit = str(existing.get("source_commit", "")).strip()
        if _normalized_repository(existing_repository) != _normalized_repository(
            canonical_repository
        ):
            raise ValueError("OPENAI_CLIP_SOURCE_REPOSITORY_INVALID")
        if declared and existing_commit and declared.strip() != existing_commit:
            raise ValueError("OPENAI_CLIP_SOURCE_COMMIT_MISMATCH")
        return {
            "source_repository": canonical_repository,
            "source_commit": declared.strip() if declared else existing_commit or "UNKNOWN",
            "source_branch": str(existing.get("source_branch", "UNKNOWN")),
            "source_commit_timestamp": str(
                existing.get("source_commit_timestamp", "UNKNOWN")
            ),
            "source_acquisition": str(existing.get("source_acquisition", "git_clone")),
            "source_destination": "source/openai_clip",
            "nested_git_directory_included": False,
        }
    source_commit = declared.strip() if declared else actual_commit or "UNKNOWN"
    if declared and actual_commit and source_commit != actual_commit:
        raise ValueError("OPENAI_CLIP_SOURCE_COMMIT_MISMATCH")
    return {
        "source_repository": canonical_repository,
        "source_commit": source_commit,
        "source_branch": _git_value(source, "branch", "--show-current") or "UNKNOWN",
        "source_commit_timestamp": (
            _git_value(source, "show", "-s", "--format=%cI", "HEAD") or "UNKNOWN"
        ),
        "source_acquisition": "git_clone" if actual_commit else "caller_supplied_local_source",
        "source_destination": "source/openai_clip",
        "nested_git_directory_included": False,
    }


def _publish(staging: Path, output: Path, overwrite: bool) -> None:
    if not output.exists():
        os.replace(staging, output)
        return
    if not overwrite:
        raise FileExistsError(f"Asset bundle exists: {output}")
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        raise FileExistsError(f"Stale asset bundle backup exists: {backup}")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except Exception:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def _publish_preserving_checkpoint(staging: Path, output: Path, overwrite: bool) -> None:
    """Publish beside an existing in-output checkpoint without replacing the weight file."""
    if not output.exists():
        os.replace(staging, output)
        return
    if not overwrite:
        raise FileExistsError(f"Asset bundle exists: {output}")
    allowed_entries = {"checkpoint", "source", "manifests"}
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed_entries)
    if unexpected:
        raise ValueError(f"Unexpected existing asset-root entries: {unexpected}")
    existing_checkpoint = output / "checkpoint/ViT-B-32.pt"
    staged_checkpoint = staging / "checkpoint/ViT-B-32.pt"
    if not existing_checkpoint.is_file() or sha256_file(existing_checkpoint) != sha256_file(
        staged_checkpoint
    ):
        raise ValueError("CHECKPOINT_HASH_MISMATCH")
    backups: dict[str, Path] = {}
    published: list[Path] = []
    try:
        for name in ("source", "manifests"):
            target = output / name
            backup = output / f".{name}.previous"
            if backup.exists():
                raise FileExistsError(f"Stale asset component backup exists: {backup}")
            if target.exists():
                os.replace(target, backup)
                backups[name] = backup
        for name in ("source", "manifests"):
            target = output / name
            os.replace(staging / name, target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            shutil.rmtree(target, ignore_errors=True)
        for name, backup in backups.items():
            os.replace(backup, output / name)
        raise
    for backup in backups.values():
        shutil.rmtree(backup)
    shutil.rmtree(staging)


def _create_zip(root: Path, overwrite: bool) -> Path:
    target = root.with_suffix(".zip")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Asset ZIP exists: {target}")
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.resolve() == target.resolve():
                continue
            archive.write(path, arcname=path.relative_to(root).as_posix())
    return target


def build_openai_clip_asset_bundle(config: AssetBundleConfig) -> dict[str, Any]:
    source, checkpoint, output, dependency_wheels = _validate(config)
    source_files = _source_files(source)
    checkpoint_hash = sha256_file(checkpoint)
    existing_provenance = None
    existing_provenance_path = output / "manifests/source_provenance.json"
    if existing_provenance_path.is_file():
        try:
            value = json.loads(existing_provenance_path.read_text(encoding="utf-8"))
            existing_provenance = value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            existing_provenance = None
    source_provenance = _source_provenance(
        source, config.source_commit, existing=existing_provenance
    )
    source_commit = str(source_provenance["source_commit"])
    plan = {
        "source_root": str(source),
        "checkpoint": str(checkpoint),
        "output_root": str(output),
        "source_files": [path.relative_to(source).as_posix() for path in source_files],
        "checkpoint_sha256": checkpoint_hash,
        "source_commit": source_commit,
        "dependency_wheels": [str(path) for path in dependency_wheels],
        "dry_run": config.dry_run,
    }
    if config.dry_run:
        return plan
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"Asset bundle exists: {output}")
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for source_path in source_files:
            relative = source_path.relative_to(source)
            destination = staging / "source/openai_clip" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path.resolve(strict=True), destination)
        checkpoint_target = staging / "checkpoint/ViT-B-32.pt"
        checkpoint_target.parent.mkdir(parents=True)
        shutil.copy2(checkpoint, checkpoint_target)
        dependency_records = []
        for wheel in dependency_wheels:
            target = staging / "source/dependencies" / wheel.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheel, target)
            dependency_records.append(
                {
                    "relative_path": target.relative_to(staging).as_posix(),
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
        manifests = staging / "manifests"
        manifests.mkdir()
        (manifests / "checkpoint.sha256").write_text(
            f"{checkpoint_hash}  checkpoint/ViT-B-32.pt\n", encoding="utf-8"
        )
        (manifests / "SOURCE_COMMIT.txt").write_text(source_commit + "\n", encoding="utf-8")
        write_json(manifests / "source_provenance.json", source_provenance)
        manifest = {
            "asset_bundle_version": "0.1.0",
            "implementation": "openai_clip",
            "architecture": "ViT-B/32",
            "pretrained": "openai",
            "source_repository": source_provenance["source_repository"],
            "source_commit": source_commit,
            "checkpoint_relative_path": "checkpoint/ViT-B-32.pt",
            "checkpoint_filename": "ViT-B-32.pt",
            "checkpoint_size_bytes": checkpoint_target.stat().st_size,
            "checkpoint_sha256": checkpoint_hash,
            "source_relative_path": "source/openai_clip",
            "tokenizer_asset_relative_path": (
                "source/openai_clip/clip/bpe_simple_vocab_16e6.txt.gz"
            ),
            "internet_required_at_runtime": False,
            "runtime_model_load_policy": "absolute_local_checkpoint_path_only",
            "offline_dependency_wheels": dependency_records,
            "created_at": datetime.now(UTC).isoformat(),
            "notes": [
                "This bundle is a Stage 1B encoder compatibility hypothesis.",
                "The asset manifest does not prove that BTC used this implementation.",
                "Compatibility must be established by the empirical Stage 1B probe.",
            ],
        }
        write_json(manifests / "asset_manifest.json", manifest)
        inventory = []
        inventory_paths = sorted(
            (item for item in staging.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(staging).as_posix(),
        )
        for path in inventory_paths:
            inventory.append(
                {
                    "relative_path": path.relative_to(staging).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        write_jsonl(manifests / "file_inventory.jsonl", inventory)
        if _within(checkpoint, output):
            _publish_preserving_checkpoint(staging, output, config.overwrite)
        else:
            _publish(staging, output, config.overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    plan["zip_path"] = str(_create_zip(output, config.overwrite)) if config.create_zip else None
    plan["files_written"] = sum(1 for path in output.rglob("*") if path.is_file())
    return plan
