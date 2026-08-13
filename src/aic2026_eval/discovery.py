"""Bounded, marker-based discovery for nested Kaggle dataset mounts."""

from __future__ import annotations

from collections import deque
from pathlib import Path


def bounded_directories(
    root: str | Path,
    *,
    max_depth: int = 6,
    max_directories: int = 6000,
):
    queue = deque([(Path(root), 0)])
    visited = 0
    while queue:
        current, depth = queue.popleft()
        if not current.is_dir() or current.is_symlink():
            continue
        visited += 1
        if visited > max_directories:
            raise RuntimeError("input discovery exceeded its directory bound")
        yield current
        if depth < max_depth:
            queue.extend(
                (child, depth + 1)
                for child in sorted(current.iterdir(), key=lambda path: path.name)
                if child.is_dir() and not child.is_symlink()
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
    while queue:
        current, depth = queue.popleft()
        if not current.is_dir() or current.is_symlink():
            continue
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
                if child.is_dir() and not child.is_symlink()
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


__all__ = [
    "bounded_directories",
    "resolve_dataset_root",
    "resolve_named_file",
    "resolve_repository_root",
]
