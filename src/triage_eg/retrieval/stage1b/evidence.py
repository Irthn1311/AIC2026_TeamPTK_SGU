"""Bounded repository and dataset-metadata encoder evidence discovery."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KEYWORDS = re.compile(
    r"CLIP|OpenAI CLIP|OpenCLIP|open_clip|ViT-B/32|ViT-B-32|pretrained|tokenizer|"
    r"preprocess|normalize|checkpoint",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".py", ".ipynb"}
DATASET_METADATA_NAMES = {"readme", "metadata", "config", "model", "contract", "manifest"}


@dataclass(frozen=True)
class EvidenceLimits:
    repository_files: int = 600
    repository_directories: int = 800
    dataset_directories: int = 40
    dataset_files: int = 100
    max_depth: int = 2
    max_file_bytes: int = 1024 * 1024
    excerpt_chars: int = 240


def _supports(text: str) -> dict[str, Any]:
    lower = text.lower()
    implementation = "open_clip" if "open_clip" in lower or "openclip" in lower else None
    architecture = (
        "ViT-B/32" if "vit-b/32" in lower else "ViT-B-32" if "vit-b-32" in lower else None
    )
    return {
        "implementation": implementation,
        "architecture": architecture,
        "pretrained": "openai" if "openai" in lower and "clip" in lower else None,
        "tokenizer": "mentioned" if "tokenizer" in lower else None,
        "image_preprocessing": "mentioned" if "preprocess" in lower else None,
        "text_preprocessing": "mentioned" if "text preprocess" in lower else None,
        "normalization": "mentioned" if "normaliz" in lower else None,
    }


def _record(path: Path, root: Path, source_type: str, line: int, text: str, limit: int) -> dict:
    excerpt = " ".join(text.strip().split())[:limit]
    relative = path.relative_to(root).as_posix() if root in path.parents else str(path)
    identifier = hashlib.sha256(f"{source_type}:{relative}:{line}".encode()).hexdigest()[:12]
    return {
        "evidence_id": f"ev_{source_type.lower()}_{identifier}",
        "source_type": source_type,
        "path": relative,
        "line_or_key": str(line),
        "excerpt": excerpt,
        "supports": _supports(excerpt),
        "confidence": "MEDIUM" if source_type == "REPOSITORY_FILE" else "LOW",
        "authoritative": False,
    }


def _scan_files(files: list[Path], root: Path, source: str, limits: EvidenceLimits) -> list[dict]:
    records: list[dict] = []
    for path in files:
        try:
            if path.stat().st_size > limits.max_file_bytes:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if KEYWORDS.search(line):
                    records.append(
                        _record(path, root, source, line_number, line, limits.excerpt_chars)
                    )
        except OSError:
            continue
    return records


def _dataset_metadata_files(root: Path, limits: EvidenceLimits) -> list[Path]:
    files: list[Path] = []
    frontier = [(root, 0)]
    visited = 0
    while frontier and visited < limits.dataset_directories and len(files) < limits.dataset_files:
        current, depth = frontier.pop(0)
        visited += 1
        try:
            children = sorted(current.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES:
                stem = child.stem.lower()
                if any(token in stem for token in DATASET_METADATA_NAMES):
                    files.append(child)
            elif child.is_dir() and depth < limits.max_depth:
                name = child.name.lower()
                if not any(token in name for token in ("keyframe", "object", "video")):
                    frontier.append((child, depth + 1))
            if len(files) >= limits.dataset_files:
                break
    return files


def discover_encoder_evidence(
    repo_root: str | Path,
    dataset_root: str | Path,
    limits: EvidenceLimits | None = None,
) -> tuple[list[dict], dict]:
    limits = limits or EvidenceLimits()
    repo = Path(repo_root).resolve(strict=True)
    dataset = Path(dataset_root).resolve(strict=True)
    repository_files: list[Path] = []
    frontier = [(repo, 0)]
    visited = 0
    while (
        frontier
        and len(repository_files) < limits.repository_files
        and visited < limits.repository_directories
    ):
        current, depth = frontier.pop(0)
        visited += 1
        try:
            children = sorted(current.iterdir(), key=lambda value: value.name)
        except OSError:
            continue
        for path in children:
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                repository_files.append(path)
            elif path.is_dir() and depth < 8 and path.name not in {".git", "outputs", ".venv"}:
                frontier.append((path, depth + 1))
            if len(repository_files) >= limits.repository_files:
                break
    dataset_files = _dataset_metadata_files(dataset, limits)
    records = _scan_files(repository_files, repo, "REPOSITORY_FILE", limits)
    records.extend(_scan_files(dataset_files, dataset, "DATASET_METADATA", limits))
    summary = {
        "authoritative_metadata_found": any(item["authoritative"] for item in records),
        "records": len(records),
        "repository_files_examined": len(repository_files),
        "repository_directories_examined": visited,
        "dataset_metadata_files_examined": len(dataset_files),
        "bounded": True,
        "status": ("EVIDENCE_FOUND" if records else "AUTHORITATIVE_ENCODER_METADATA_NOT_FOUND"),
    }
    if not summary["authoritative_metadata_found"]:
        summary["status"] = "AUTHORITATIVE_ENCODER_METADATA_NOT_FOUND"
    return records, summary
