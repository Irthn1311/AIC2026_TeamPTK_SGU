"""Frozen FS1 protocol constants and small value contracts."""

from __future__ import annotations

from dataclasses import dataclass

ANCHOR_COMMIT = "56c2f37df6841af0e7fe858632ccf8554e8ac4e1"
B0_CROSS_SHA256 = "801e9e4a8e33916cb0430c9c391694410972a84b212d0db949d63671be39e2dc"
B0_L21_SHA256 = "3c4dbd2bf4766b286d1efceded120c801e59696d08ab3deb19dd38669074fd16"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


@dataclass(frozen=True)
class FS1Settings:
    rrf_k: int = 60
    protected_prefix: int = 5
    max_predictions: int = 100
    qwen_budget: int = 20
    graph_revision_limit: int = 1

    def __post_init__(self) -> None:
        if (self.rrf_k, self.protected_prefix, self.max_predictions, self.qwen_budget) != (
            60,
            5,
            100,
            20,
        ):
            raise ValueError("FS1 frozen constants may not be tuned")
        if self.graph_revision_limit != 1:
            raise ValueError("FS1 permits exactly one graph revision pass")


@dataclass(frozen=True)
class RouteDecision:
    task: str
    modalities: tuple[str, ...]
    reasons: tuple[str, ...]
    event_index: int | None = None
