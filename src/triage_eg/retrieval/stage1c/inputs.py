"""Bounded resolution and validation of saved Stage 1B evaluation artifacts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile

STAGE1B_REQUIRED = (
    "stage1b_summary.json",
    "run_manifest.json",
    "encoder/selected_encoder_contract.json",
    "encoder/runtime_adapter_manifest.json",
)


def _complete(root: Path) -> bool:
    return root.is_dir() and all((root / name).is_file() for name in STAGE1B_REQUIRED)


def _extract(archive_path: Path, output_root: Path) -> Path:
    output = output_root.resolve(strict=False)
    if _complete(output):
        return output
    if output.exists():
        raise FileExistsError(f"Incomplete Stage 1B materialization root exists: {output}")
    staging = output.with_name(f".{output.name}.extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Stage 1B ZIP contains duplicate members")
            missing = [name for name in STAGE1B_REQUIRED if name not in names]
            if missing:
                raise FileNotFoundError(f"Stage 1B ZIP is missing {missing}")
            for info in archive.infolist():
                if info.is_dir():
                    continue
                relative = Path(info.filename.replace("\\", "/"))
                target = (staging / relative).resolve(strict=False)
                if relative.is_absolute() or staging.resolve() not in target.parents:
                    raise ValueError(f"Unsafe Stage 1B ZIP member: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        if not _complete(staging):
            raise FileNotFoundError("Materialized Stage 1B artifacts are incomplete")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return output
    except (BadZipFile, OSError, ValueError):
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def resolve_stage1b_root(
    requested_root: str | Path,
    *,
    search_root: str | Path | None = None,
    materialize_root: str | Path | None = None,
    max_depth: int = 4,
    max_directories: int = 500,
) -> Path:
    requested = Path(requested_root).expanduser().resolve(strict=False)
    if _complete(requested):
        return requested
    materialized = (
        Path(materialize_root).expanduser().resolve(strict=False)
        if materialize_root is not None
        else None
    )
    if materialized is not None and _complete(materialized):
        return materialized
    archives: list[Path] = []
    if requested.is_file() and requested.suffix.lower() == ".zip":
        archives.append(requested)
    roots = [requested] if requested.is_dir() else []
    if search_root is not None:
        search = Path(search_root).expanduser().resolve(strict=True)
        if search not in roots:
            roots.append(search)
    frontier = [(root, 0) for root in roots]
    matches: list[Path] = []
    visited = 0
    seen: set[Path] = set()
    while frontier and visited < max_directories:
        current, depth = frontier.pop(0)
        current = current.resolve(strict=False)
        if current in seen:
            continue
        seen.add(current)
        visited += 1
        if _complete(current):
            matches.append(current)
            continue
        if depth >= max_depth:
            continue
        try:
            for child in sorted(current.iterdir(), key=lambda item: item.name):
                if child.is_dir():
                    frontier.append((child, depth + 1))
                elif (
                    child.suffix.lower() == ".zip"
                    and "stage1b" in child.name.lower()
                    and "compatibility" in child.name.lower()
                ):
                    archives.append(child.resolve())
        except OSError:
            continue
    matches = sorted(set(matches))
    archives = sorted(set(archives))
    if len(matches) == 1:
        return matches[0]
    if not matches and len(archives) == 1 and materialized is not None:
        return _extract(archives[0], materialized)
    raise FileNotFoundError(
        f"Expected exactly one complete Stage 1B root; found {len(matches)} roots and "
        f"{len(archives)} candidate ZIPs"
    )
