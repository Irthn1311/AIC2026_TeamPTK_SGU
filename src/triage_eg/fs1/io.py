"""Deterministic FS1 artifact I/O and pre-GT state gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PreGTGate:
    def __init__(self, benchmarks: tuple[str, ...] = ("cross", "l21")) -> None:
        self.required = {(benchmark, arm) for benchmark in benchmarks for arm in ("B0", "M0", "M1")}
        self.hashes: dict[str, dict[str, str]] = {benchmark: {} for benchmark in benchmarks}
        self.gt_opened = False

    def finalize(self, benchmark: str, arm: str, path: Path) -> str:
        if self.gt_opened:
            raise RuntimeError("FS1_PREDICTION_AFTER_GT_FORBIDDEN")
        digest = sha256(path)
        self.hashes[benchmark][arm] = digest
        return digest

    def open_gt(self) -> None:
        actual = {(benchmark, arm) for benchmark, values in self.hashes.items() for arm in values}
        if actual != self.required:
            raise RuntimeError(f"FS1_PRE_GT_HASH_GATE_INCOMPLETE: {sorted(self.required - actual)}")
        self.gt_opened = True
