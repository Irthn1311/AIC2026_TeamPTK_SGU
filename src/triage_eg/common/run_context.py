"""Artifact run context and reproducibility manifest creation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from triage_eg.common.paths import resolve_artifact_path
from triage_eg.common.schemas import RunManifest, dataclass_to_dict


def current_git_commit(project_root: str | Path | None = None) -> str:
    """Return the checked-out commit, or ``UNKNOWN`` outside a Git worktree."""

    command = ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "UNKNOWN"
    return result.stdout.strip() or "UNKNOWN"


@dataclass
class RunContext:
    """Mutable lifecycle wrapper around an immutable run manifest."""

    artifact_dir: Path
    manifest: RunManifest

    def write_manifest(self, status: str | None = None) -> RunManifest:
        """Write current metadata, optionally transitioning its status."""

        if status is not None:
            self.manifest = replace(self.manifest, status=status)
        target = self.artifact_dir / "run_manifest.json"
        target.write_text(
            json.dumps(dataclass_to_dict(self.manifest), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.manifest


def create_run_context(
    *,
    artifact_name: str,
    config_path: str | Path,
    config: dict[str, object],
    data_version: str,
    command: str,
    output_root: str | Path = "artifacts",
    project_root: str | Path | None = None,
) -> RunContext:
    """Create an artifact directory and persist its exact config and manifest."""

    now = datetime.now(UTC)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    artifact_dir = resolve_artifact_path(artifact_name, run_id, output_root, project_root)
    used_config = artifact_dir / "used_config.yaml"
    used_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = RunManifest(
        run_id=run_id,
        created_at_utc=now.isoformat(),
        git_commit=current_git_commit(project_root),
        config_path=str(config_path),
        data_version=data_version,
        artifact_name=artifact_name,
        command=command,
        status="RUNNING",
    )
    context = RunContext(artifact_dir=artifact_dir, manifest=manifest)
    context.write_manifest()
    return context
