"""Fail-closed Stage 0 artifact loading without raw-layout discovery."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

REQUIRED_FILES = (
    "audit_summary.json",
    "run_manifest.json",
    "btc_frame_manifest.jsonl",
    "clip_manifest.jsonl",
    "contract_notes.json",
)
MAX_STAGE0_BUNDLE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class Stage0Bundle:
    root: Path
    summary: dict[str, Any]
    run_manifest: dict[str, Any]
    contract_notes: dict[str, Any]
    clip_records: tuple[dict[str, Any], ...]

    @property
    def frame_manifest_path(self) -> Path:
        return self.root / "btc_frame_manifest.jsonl"


def _has_required_files(root: Path) -> bool:
    return root.is_dir() and all((root / name).is_file() for name in REQUIRED_FILES)


def _is_excluded(path: Path, excluded_roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve(strict=False)
    return any(resolved == root or root in resolved.parents for root in excluded_roots)


def _bounded_stage0_directories(
    root: Path, max_depth: int, excluded_roots: tuple[Path, ...] = ()
) -> list[Path]:
    """Find artifact roots at shallow mount levels without entering corpus directories."""

    if not root.is_dir():
        return []
    found: list[Path] = []
    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if _is_excluded(current, excluded_roots):
            continue
        if _has_required_files(current):
            found.append(current.resolve())
            continue
        if depth >= max_depth:
            continue
        try:
            children = sorted(path for path in current.iterdir() if path.is_dir())
        except OSError:
            continue
        frontier.extend((child, depth + 1) for child in children)
    return found


def _bounded_stage0_zips(
    root: Path, max_depth: int, excluded_roots: tuple[Path, ...] = ()
) -> list[Path]:
    if not root.is_dir():
        return []
    found: list[Path] = []
    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if _is_excluded(current, excluded_roots):
            continue
        try:
            found.extend(
                path.resolve()
                for pattern in ("*stage0*audit*.zip", "*stage0*bundle*.zip")
                for path in current.glob(pattern)
                if path.is_file()
            )
            if depth < max_depth:
                frontier.extend(
                    (path, depth + 1) for path in sorted(current.iterdir()) if path.is_dir()
                )
        except OSError:
            continue
    return sorted(set(found))


def _materialize_stage0_zip(bundle: Path, target: Path) -> Path:
    if target.exists():
        raise FileExistsError(
            f"Cannot materialize Stage 0 ZIP over existing incomplete path: {target}"
        )
    staging = target.with_name(f".{target.name}.extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with ZipFile(bundle) as archive:
            selected: dict[str, Any] = {}
            for info in archive.infolist():
                normalized = info.filename.replace("\\", "/")
                basename = normalized.rstrip("/").rsplit("/", 1)[-1]
                if info.is_dir() or basename not in REQUIRED_FILES:
                    continue
                if basename in selected:
                    raise ValueError(f"Duplicate Stage 0 artifact in ZIP: {basename}")
                selected[basename] = info
            missing = [name for name in REQUIRED_FILES if name not in selected]
            if missing:
                raise ValueError(f"Stage 0 ZIP is missing: {', '.join(missing)}")
            total_bytes = sum(info.file_size for info in selected.values())
            if total_bytes > MAX_STAGE0_BUNDLE_BYTES:
                raise ValueError("Stage 0 required artifacts exceed safe extraction size")
            for name in REQUIRED_FILES:
                with archive.open(selected[name]) as source, (staging / name).open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
    except (BadZipFile, OSError) as error:
        raise ValueError(f"Cannot materialize Stage 0 ZIP {bundle}: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target.resolve()


def resolve_stage0_root(
    requested_root: str | Path,
    *,
    bundle_path: str | Path | None = None,
    search_root: str | Path | None = None,
    search_max_depth: int = 3,
    excluded_roots: tuple[str | Path, ...] = (),
) -> Path:
    """Resolve Stage 0 artifacts without rerunning Stage 0 or scanning the raw corpus."""

    requested = Path(requested_root).expanduser().resolve(strict=False)
    if _has_required_files(requested):
        return requested
    if requested.exists():
        missing = [name for name in REQUIRED_FILES if not (requested / name).is_file()]
        raise FileNotFoundError(
            f"Incomplete Stage 0 root {requested}; missing: {', '.join(missing)}"
        )
    if not 0 <= search_max_depth <= 4:
        raise ValueError("search_max_depth must be between 0 and 4")
    excluded = tuple(Path(path).expanduser().resolve(strict=False) for path in excluded_roots)

    explicit = Path(bundle_path).expanduser().resolve(strict=False) if bundle_path else None
    if explicit is not None:
        if _has_required_files(explicit):
            return explicit
        if explicit.is_dir():
            roots = sorted(set(_bounded_stage0_directories(explicit, 2, excluded)))
            if len(roots) == 1:
                return roots[0]
            if len(roots) > 1:
                raise ValueError("AIC_STAGE0_BUNDLE directory contains multiple artifact roots")
            bundles = _bounded_stage0_zips(explicit, 2, excluded)
            if len(bundles) == 1:
                return _materialize_stage0_zip(bundles[0], requested)
            if len(bundles) > 1:
                raise ValueError("AIC_STAGE0_BUNDLE directory contains multiple Stage 0 ZIPs")
        elif explicit.is_file() and explicit.suffix.lower() == ".zip":
            return _materialize_stage0_zip(explicit, requested)
        raise FileNotFoundError(
            f"AIC_STAGE0_BUNDLE is not a valid Stage 0 directory/ZIP: {explicit}"
        )

    if search_root is not None:
        mounted = Path(search_root).expanduser().resolve(strict=False)
        roots = sorted(set(_bounded_stage0_directories(mounted, search_max_depth, excluded)))
        if len(roots) == 1:
            return roots[0]
        if len(roots) > 1:
            raise ValueError(
                "Multiple Stage 0 roots found under Kaggle Input; set AIC_STAGE0_BUNDLE explicitly"
            )
        bundles = _bounded_stage0_zips(mounted, search_max_depth, excluded)
        if len(bundles) == 1:
            return _materialize_stage0_zip(bundles[0], requested)
        if len(bundles) > 1:
            raise ValueError(
                "Multiple Stage 0 ZIPs found under Kaggle Input; set AIC_STAGE0_BUNDLE explicitly"
            )

    raise FileNotFoundError(
        "Stage 0 artifacts are unavailable. In Stage 1, use Add Input to attach the "
        "saved output of the successful Stage 0 notebook, or upload "
        "triage_eg_stage0_audit_bundle.zip as a private Kaggle Dataset. You may also "
        "set AIC_STAGE0_BUNDLE to the mounted directory/ZIP path. Stage 1 will not "
        "rerun Stage 0 and generated artifacts do not belong in Git."
    )


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Stage 0 JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Stage 0 artifact must contain an object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def load_stage0_bundle(root: str | Path, *, require_full: bool = True) -> Stage0Bundle:
    resolved = Path(root).expanduser().resolve(strict=False)
    missing = [name for name in REQUIRED_FILES if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 0 artifacts: {', '.join(missing)}")
    summary = _json(resolved / "audit_summary.json")
    run_manifest = _json(resolved / "run_manifest.json")
    notes = _json(resolved / "contract_notes.json")
    if run_manifest.get("status") != "COMPLETE":
        raise ValueError("Stage 0 run status must be COMPLETE")
    if summary.get("audit_version") != "0.1.0":
        raise ValueError("Unsupported Stage 0 audit version")
    if require_full and (
        summary.get("mode") != "full"
        or summary.get("videos_completed") != summary.get("videos_discovered")
        or summary.get("videos_completed") != 873
    ):
        raise ValueError("Stage 0 must be a complete 873-video full audit")
    if summary.get("gates", {}).get("btc_baseline") == "FAIL":
        raise ValueError("Stage 0 BTC baseline gate is FAIL")
    if summary.get("mapping_rows") != summary.get("clip_rows"):
        raise ValueError("Stage 0 mapping_rows must equal clip_rows")
    if notes.get("original_frame_policy") != (
        "CSV frame_idx is authoritative; never reconstruct from pts_time*fps"
    ):
        raise ValueError("Stage 0 original-frame policy is incompatible")
    unknown = {str(item).lower() for item in summary.get("unknown_contracts", [])}
    if not any("clip" in item and "compat" in item for item in unknown):
        raise ValueError("Stage 0 must retain unknown CLIP model compatibility")
    clips = tuple(iter_jsonl(resolved / "clip_manifest.jsonl"))
    if len(clips) != summary.get("videos_completed"):
        raise ValueError("clip_manifest row count must equal completed videos")
    if sum(int(item.get("row_count", -1)) for item in clips) != summary.get("clip_rows"):
        raise ValueError("clip_manifest total rows do not match audit summary")
    return Stage0Bundle(resolved, summary, run_manifest, notes, clips)
