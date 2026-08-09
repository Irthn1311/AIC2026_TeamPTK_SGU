"""Bounded discovery of attached, already-built Stage 1A outputs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from triage_eg.retrieval.stage1.writers import INDEX_MEMBERS, STAGE1B_INPUT_MEMBERS

STAGE1_REQUIRED = ("stage1_summary.json", *INDEX_MEMBERS)


def _complete_stage1(root: Path) -> bool:
    return root.is_dir() and all((root / name).is_file() for name in STAGE1_REQUIRED)


def _materialize_stage1b_bundle(archive_path: Path, output_root: Path) -> Path:
    output = output_root.expanduser().resolve(strict=False)
    if _complete_stage1(output):
        return output
    if output.exists():
        raise FileExistsError(f"Incomplete Stage 1B materialization root already exists: {output}")
    staging = output.with_name(f".{output.name}.extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with ZipFile(archive_path) as archive:
            archive_names = archive.namelist()
            names = set(archive_names)
            missing = [name for name in STAGE1B_INPUT_MEMBERS if name not in names]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete Stage 1B input ZIP {archive_path}; missing: {missing}"
                )
            unexpected = sorted(names - set(STAGE1B_INPUT_MEMBERS))
            if unexpected or len(archive_names) != len(names):
                raise ValueError(
                    f"Stage 1B input ZIP contains unexpected or duplicate members: {unexpected}"
                )
            for name in STAGE1B_INPUT_MEMBERS:
                relative = Path(name)
                destination = (staging / relative).resolve(strict=False)
                if relative.is_absolute() or staging.resolve() not in destination.parents:
                    raise ValueError(f"Unsafe Stage 1B ZIP member: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        if not _complete_stage1(staging):
            raise FileNotFoundError("Materialized Stage 1B input is incomplete")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return output
    except (BadZipFile, OSError, ValueError):
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def resolve_stage1_root(
    requested_root: str | Path,
    *,
    search_root: str | Path | None = None,
    max_depth: int = 4,
    max_directories: int = 500,
    excluded_roots: tuple[str | Path, ...] = (),
    materialize_root: str | Path | None = None,
) -> Path:
    """Resolve exactly one complete saved Stage 1A root without corpus scanning."""

    requested = Path(requested_root).expanduser().resolve(strict=False)
    if _complete_stage1(requested):
        return requested
    materialized = (
        Path(materialize_root).expanduser().resolve(strict=False)
        if materialize_root is not None
        else None
    )
    if materialized is not None and _complete_stage1(materialized):
        return materialized
    if not 0 <= max_depth <= 4 or max_directories <= 0:
        raise ValueError("Invalid bounded Stage 1A discovery limits")
    requested_missing = None
    archive_matches: list[Path] = []
    if requested.exists():
        if requested.is_file() and requested.suffix.lower() == ".zip":
            archive_matches.append(requested)
        elif not requested.is_dir():
            raise FileNotFoundError(f"Stage 1A root is not a directory: {requested}")
        else:
            requested_missing = [
                name for name in STAGE1_REQUIRED if not (requested / name).is_file()
            ]
    elif search_root is None:
        raise FileNotFoundError(f"Stage 1A root does not exist: {requested}")
    excluded = tuple(Path(path).resolve(strict=False) for path in excluded_roots)
    search_bases = [requested] if requested.is_dir() else []
    if search_root is not None:
        mounted = Path(search_root).expanduser().resolve(strict=True)
        if mounted not in search_bases:
            search_bases.append(mounted)
    frontier = [(root, 0) for root in search_bases]
    matches: list[Path] = []
    seen: set[Path] = set()
    visited = 0
    while frontier and visited < max_directories:
        current, depth = frontier.pop(0)
        current = current.resolve(strict=False)
        if current in seen:
            continue
        seen.add(current)
        visited += 1
        if any(current == root or root in current.parents for root in excluded):
            continue
        if _complete_stage1(current):
            if current not in matches:
                matches.append(current)
            continue
        if depth >= max_depth:
            continue
        try:
            for child in sorted(current.iterdir(), key=lambda value: value.name):
                if child.is_dir():
                    frontier.append((child, depth + 1))
                elif (
                    child.is_file()
                    and child.suffix.lower() == ".zip"
                    and "stage1b" in child.name.lower()
                    and "input" in child.name.lower()
                ):
                    archive_matches.append(child.resolve())
        except OSError:
            continue
    if len(matches) == 1:
        return matches[0].resolve()
    archives = sorted(set(archive_matches))
    if not matches and len(archives) == 1 and materialized is not None:
        return _materialize_stage1b_bundle(archives[0], materialized)
    if not matches and len(archives) > 1:
        raise FileNotFoundError(
            f"Expected at most one Stage 1B input ZIP; found {len(archives)}"
        )
    if not matches and requested_missing is not None:
        raise FileNotFoundError(
            f"Incomplete Stage 1A mount {requested}; missing: {requested_missing}. "
            "No nested complete saved Stage 1A root was found. Attach the full saved "
            "Stage 1A output or an index bundle containing the required index arrays; "
            "the report-only bundle is insufficient for Stage 1B."
        )
    raise FileNotFoundError(
        f"Expected exactly one complete saved Stage 1A root; found {len(matches)}"
    )
