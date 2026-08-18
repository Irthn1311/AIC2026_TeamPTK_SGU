"""BCF-1 deterministic LF-only JSONL serialization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_jsonl_lf(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write the frozen JSONL format identically on Windows and Linux."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    target.write_bytes(payload.encode("utf-8"))
    return target


__all__ = ["write_jsonl_lf"]
