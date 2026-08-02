"""Project-relative path helpers."""

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return its resolved path."""

    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve an absolute path or a path relative to the repository root."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
    return (root / candidate).resolve()


def resolve_artifact_path(
    artifact_name: str,
    run_id: str,
    output_root: str | Path = "artifacts",
    project_root: str | Path | None = None,
) -> Path:
    """Create and return ``<output_root>/<artifact_name>/<run_id>``."""

    _validate_path_component(artifact_name, "artifact_name")
    _validate_path_component(run_id, "run_id")
    root = resolve_project_path(output_root, project_root)
    return ensure_directory(root / artifact_name / run_id)


def _validate_path_component(value: str, name: str) -> None:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be one safe path component")
