"""Config-driven Stage 1B candidate registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from triage_eg.common.config import load_yaml_config
from triage_eg.retrieval.stage1b.contracts import (
    STAGE1B_VERSION,
    CandidateContract,
    CompatibilityGate,
)


def load_candidate_registry(
    path: str | Path, candidate_ids: tuple[str, ...] = ()
) -> tuple[list[CandidateContract], CompatibilityGate, dict[str, Any], str]:
    config = load_yaml_config(path)
    if config.get("stage1b_version") != STAGE1B_VERSION:
        raise ValueError("Unsupported Stage 1B candidate config version")
    candidates_value = config.get("candidates")
    if not isinstance(candidates_value, list):
        raise ValueError("Stage 1B candidates must be a list")
    candidates = [CandidateContract.from_dict(item) for item in candidates_value]
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("Duplicate Stage 1B candidate_id")
    requested = set(candidate_ids)
    unknown = requested - {item.candidate_id for item in candidates}
    if unknown:
        raise ValueError(f"Unknown requested candidate IDs: {', '.join(sorted(unknown))}")
    if requested:
        candidates = [item for item in candidates if item.candidate_id in requested]
    gate = CompatibilityGate.from_dict(config.get("compatibility_gate", {}))
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return candidates, gate, config, hashlib.sha256(encoded).hexdigest()
