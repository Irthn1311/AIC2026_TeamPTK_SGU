"""Skeleton for fixture-level KIS evaluation, not official BTC scoring."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class KISFixtureEvaluator:
    def evaluate(
        self,
        checkpoint_path: Path,
        ground_truth_path: Path,
    ) -> Mapping[str, Any]:
        raise NotImplementedError("KISFixtureEvaluator.evaluate is not implemented")
