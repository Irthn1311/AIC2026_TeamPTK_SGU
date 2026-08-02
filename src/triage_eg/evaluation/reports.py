"""JSON report writer for evaluation command-line tools."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_evaluation_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Write a human-readable UTF-8 evaluation report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
