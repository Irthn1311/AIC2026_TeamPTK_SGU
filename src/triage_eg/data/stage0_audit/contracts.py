"""Typed contracts for Stage 0 BTC data auditing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AUDIT_VERSION = "0.1.0"
DATASET_VERSION = "aic25-b1"


@dataclass(frozen=True)
class AuditConfig:
    dataset_root: Path
    output_root: Path
    mode: str = "sample"
    sample_size: int = 10
    video_ids: tuple[str, ...] = ()
    seed: int = 2026
    clip_validation: str = "full"
    object_validation: str = "full"
    max_object_json_bytes: int = 1_048_576
    ffprobe_timeout_seconds: int = 30
    expected_clip_dimension: int = 512
    resume: bool = False
    overwrite: bool = False
    strict_root: bool = False
    workers: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"sample", "full"}:
            raise ValueError("mode must be sample or full")
        if self.clip_validation not in {"shape", "full"}:
            raise ValueError("clip_validation must be shape or full")
        if self.object_validation not in {"filenames", "full"}:
            raise ValueError("object_validation must be filenames or full")
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite cannot both be enabled")
        if self.sample_size <= 0 or self.max_object_json_bytes <= 0:
            raise ValueError("sample_size and byte limit must be positive")
        if self.ffprobe_timeout_seconds <= 0 or self.workers != 1:
            raise ValueError("timeout must be positive and Stage 0 v0.1 workers must equal 1")

    def fingerprint_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("resume")
        payload.pop("overwrite")
        payload["dataset_root"] = str(self.dataset_root.resolve(strict=False))
        payload["output_root"] = str(self.output_root.resolve(strict=False))
        return payload


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    video_id: str | None = None
    ordinal_n: int | None = None
    asset_type: str = "DATASET"
    path: str | None = None
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    blocks_btc_baseline: bool = False
    blocks_raw_video_pipeline: bool = False
    audit_version: str = AUDIT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssetPaths:
    video_id: str
    video_partition: str | None
    keyframe_partition: str | None
    video: Path
    mapping: Path
    keyframe_directory: Path
    clip: Path
    object_directory: Path
    metadata: Path


def issue(
    code: str,
    severity: str,
    *,
    video_id: str | None = None,
    ordinal_n: int | None = None,
    asset_type: str = "DATASET",
    path: str | Path | None = None,
    message: str = "",
    evidence: dict[str, Any] | None = None,
    btc: bool = False,
    raw: bool = False,
) -> AuditIssue:
    return AuditIssue(
        code,
        severity,
        video_id,
        ordinal_n,
        asset_type,
        str(path) if path is not None else None,
        message,
        evidence or {},
        btc,
        raw,
    )
