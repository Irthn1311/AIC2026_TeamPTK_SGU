"""Config-driven Stage 1B candidate registry."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from triage_eg.retrieval.stage1b.contracts import (
    STAGE1B_VERSION,
    CandidateContract,
    CompatibilityGate,
)

_ENVIRONMENT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_environment(value: Any) -> Any:
    if isinstance(value, str):
        return _ENVIRONMENT.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)
    if isinstance(value, dict):
        return {key: _resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_environment(item) for item in value]
    return value


def _load_registry_yaml(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {candidate}")
    loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Stage 1B candidate config must contain a mapping")
    return _resolve_environment(loaded)


def load_candidate_registry(
    path: str | Path, candidate_ids: tuple[str, ...] = ()
) -> tuple[list[CandidateContract], CompatibilityGate, dict[str, Any], str]:
    config = _load_registry_yaml(path)
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


def write_official_runtime_config(
    template_path: str | Path,
    output_path: str | Path,
    *,
    asset_root: str | Path,
    source_root: str | Path,
    checkpoint_path: str | Path,
    device: str = "auto",
    batch_size: int = 16,
) -> Path:
    """Write a resolved runtime YAML without mutating the repository template."""

    template = _load_registry_yaml(template_path)
    candidates = template.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("Official runtime template must contain exactly one candidate")
    root = Path(asset_root).expanduser().resolve(strict=False)
    candidate = candidates[0]
    candidate["source_root"] = str(Path(source_root).expanduser().resolve(strict=False))
    candidate["checkpoint_path"] = str(Path(checkpoint_path).expanduser().resolve(strict=False))
    candidate["asset_manifest_path"] = str(root / "manifests/asset_manifest.json")
    candidate["device"] = device
    candidate["batch_size"] = batch_size
    template.setdefault("probe", {})["batch_size"] = batch_size
    target = Path(output_path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(template, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target
