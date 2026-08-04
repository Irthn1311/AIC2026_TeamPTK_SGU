"""Skeleton for local compatibility with the future shared validator."""

from __future__ import annotations

from pathlib import Path

from system_tai.common.schemas import ValidationResult
from system_tai.data.video_catalog import BenchmarkVideoCatalog


class CheckpointValidator:
    def validate(
        self,
        checkpoint_path: Path,
        catalog: BenchmarkVideoCatalog,
        *,
        query_set_path: Path,
    ) -> ValidationResult:
        raise NotImplementedError("CheckpointValidator.validate is not implemented")
