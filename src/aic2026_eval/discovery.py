"""Bounded, marker-based discovery for nested Kaggle dataset mounts."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def bounded_directories(
    root: str | Path,
    *,
    max_depth: int = 6,
    max_directories: int = 6000,
):
    queue = deque([(Path(root), 0)])
    visited = 0
    resolved_seen: set[Path] = set()
    while queue:
        current, depth = queue.popleft()
        if not current.is_dir():
            continue
        resolved = current.resolve()
        if resolved in resolved_seen:
            continue
        resolved_seen.add(resolved)
        visited += 1
        if visited > max_directories:
            raise RuntimeError("input discovery exceeded its directory bound")
        yield current
        if depth < max_depth:
            queue.extend(
                (child, depth + 1)
                for child in sorted(current.iterdir(), key=lambda path: path.name)
                if child.is_dir()
            )


def _unique(candidates: set[Path], description: str) -> Path:
    ordered = sorted(candidates)
    if len(ordered) != 1:
        raise RuntimeError(f"expected exactly one {description}; found: {ordered}")
    return ordered[0]


def _root_candidates(
    root: Path,
    predicate,
    *,
    max_depth: int = 6,
    max_directories: int = 6000,
) -> set[Path]:
    """Find roots without descending after a directory's marker matches."""
    queue = deque([(root, 0)])
    candidates: set[Path] = set()
    visited = 0
    resolved_seen: set[Path] = set()
    while queue:
        current, depth = queue.popleft()
        if not current.is_dir():
            continue
        resolved = current.resolve()
        if resolved in resolved_seen:
            continue
        resolved_seen.add(resolved)
        visited += 1
        if visited > max_directories:
            raise RuntimeError("input root discovery exceeded its directory bound")
        if predicate(current):
            candidates.add(current.resolve())
            continue
        if depth < max_depth:
            queue.extend(
                (child, depth + 1)
                for child in sorted(current.iterdir(), key=lambda path: path.name)
                if child.is_dir()
            )
    return candidates


def resolve_dataset_root(
    requested: str | Path,
    *,
    search_root: str | Path = "/kaggle/input",
) -> Path:
    requested_path = Path(requested)
    base = requested_path if requested_path.exists() else Path(search_root)
    candidates = _root_candidates(
        base,
        lambda directory: any(directory.glob("Videos_L*/video"))
        and (directory / "map-keyframes-aic25-b1/map-keyframes").is_dir(),
    )
    return _unique(candidates, "AIC raw dataset root")


def resolve_repository_root(
    requested: str | Path,
    *,
    search_root: str | Path = "/kaggle/input",
) -> Path:
    requested_path = Path(requested)
    base = requested_path if requested_path.exists() else Path(search_root)
    candidates = _root_candidates(
        base,
        lambda directory: (directory / "src/aic2026_eval/pipeline.py").is_file()
        and (directory / "pyproject.toml").is_file(),
    )
    return _unique(candidates, "TEAM-EVAL repository root")


def resolve_named_file(
    requested: str | Path | None,
    filename: str,
    *,
    search_root: str | Path = "/kaggle/input",
    optional: bool = False,
) -> Path | None:
    requested_path = Path(requested) if requested else None
    if requested_path is not None and requested_path.is_file():
        if requested_path.name != filename:
            raise RuntimeError(f"expected {filename}, got {requested_path.name}")
        return requested_path.resolve()
    if requested_path is not None and not requested_path.exists():
        if optional:
            return None
        raise FileNotFoundError(f"required input root does not exist: {requested_path}")
    base = requested_path if requested_path is not None else Path(search_root)
    candidates = {
        (directory / filename).resolve()
        for directory in bounded_directories(base)
        if (directory / filename).is_file()
    }
    if not candidates and optional:
        return None
    return _unique(candidates, filename)


def resolve_or_pack_archive(
    requested: str | Path,
    archive_name: str,
    required_files: set[str],
    staging_path: str | Path,
    *,
    optional: bool = False,
) -> Path | None:
    """Resolve an intact ZIP or repack a Kaggle-expanded dataset deterministically."""
    root = Path(requested)
    if not root.exists():
        if optional:
            return None
        raise FileNotFoundError(f"required input does not exist: {root}")
    if root.is_file():
        if root.name != archive_name:
            raise RuntimeError(f"expected {archive_name}, got {root.name}")
        return root.resolve()
    archives = {
        (directory / archive_name).resolve()
        for directory in bounded_directories(root)
        if (directory / archive_name).is_file()
    }
    if archives:
        return _unique(archives, archive_name)
    package_roots = {
        directory.resolve()
        for directory in bounded_directories(root)
        if all((directory / name).is_file() for name in required_files)
    }
    if not package_roots:
        if optional:
            return None
        raise RuntimeError(
            f"neither {archive_name} nor expanded files {sorted(required_files)} "
            f"were found under {root}"
        )
    package_root = _unique(package_roots, f"expanded package for {archive_name}")
    target = Path(staging_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for name in sorted(required_files):
            archive.write(package_root / name, name)
    return target.resolve()


__all__ = [
    "bounded_directories",
    "resolve_dataset_root",
    "resolve_named_file",
    "resolve_or_pack_archive",
    "resolve_repository_root",
]
