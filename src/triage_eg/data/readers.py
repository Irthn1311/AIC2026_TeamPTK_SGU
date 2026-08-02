"""Small file-reader helpers shared by command-line scripts."""

from pathlib import Path


def require_input_file(path: str | Path) -> Path:
    """Return an existing input file or raise a clear error."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {input_path}")
    return input_path

