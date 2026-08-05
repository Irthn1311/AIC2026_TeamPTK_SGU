"""Bounded discovery of attached, already-built Stage 1A outputs."""

from __future__ import annotations

from pathlib import Path

STAGE1_REQUIRED = (
    "stage1_summary.json",
    "index/index_manifest.json",
    "index/clip_vectors.f16.npy",
    "index/vector_norms.f32.npy",
    "index/frame_n.npy",
)


def _complete_stage1(root: Path) -> bool:
    return root.is_dir() and all((root / name).is_file() for name in STAGE1_REQUIRED)


def resolve_stage1_root(
    requested_root: str | Path,
    *,
    search_root: str | Path | None = None,
    max_depth: int = 4,
    max_directories: int = 500,
    excluded_roots: tuple[str | Path, ...] = (),
) -> Path:
    """Resolve exactly one complete saved Stage 1A root without corpus scanning."""

    requested = Path(requested_root).expanduser().resolve(strict=False)
    if _complete_stage1(requested):
        return requested
    if requested.exists():
        missing = [name for name in STAGE1_REQUIRED if not (requested / name).is_file()]
        raise FileNotFoundError(f"Incomplete Stage 1A root {requested}; missing: {missing}")
    if search_root is None:
        raise FileNotFoundError(f"Stage 1A root does not exist: {requested}")
    if not 0 <= max_depth <= 4 or max_directories <= 0:
        raise ValueError("Invalid bounded Stage 1A discovery limits")
    excluded = tuple(Path(path).resolve(strict=False) for path in excluded_roots)
    frontier = [(Path(search_root).resolve(strict=True), 0)]
    matches: list[Path] = []
    visited = 0
    while frontier and visited < max_directories:
        current, depth = frontier.pop(0)
        visited += 1
        if any(current == root or root in current.parents for root in excluded):
            continue
        if _complete_stage1(current):
            matches.append(current)
            continue
        if depth >= max_depth:
            continue
        try:
            frontier.extend(
                (child, depth + 1)
                for child in sorted(current.iterdir(), key=lambda value: value.name)
                if child.is_dir()
            )
        except OSError:
            continue
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one complete saved Stage 1A root; found {len(matches)}"
        )
    return matches[0].resolve()
