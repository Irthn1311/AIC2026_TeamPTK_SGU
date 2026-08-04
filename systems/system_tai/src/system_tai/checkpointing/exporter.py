"""Skeleton exporter adapter for the current proposed UTF-8 JSONL boundary."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from system_tai.common.schemas import RankedKISRecord


class CheckpointExporter:
    def export(self, records: Sequence[RankedKISRecord], destination: Path) -> None:
        raise NotImplementedError("CheckpointExporter.export is not implemented")
