"""Fail-closed resolution of Stage 1C and OPUS-MT offline inputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from triage_eg.retrieval.stage1c.contracts import QueryRecord
from triage_eg.retrieval.stage1c.query_suite import load_query_suite

from .contracts import TRANSLATOR_MODEL_ID, TRANSLATOR_REVISION

STAGE1C_REQUIRED = (
    "stage1c_summary.json",
    "run_manifest.json",
    "query_suite/query_suite.jsonl",
    "query_suite/query_suite_manifest.json",
)
TRANSLATOR_REQUIRED = (
    "model/config.json",
    "model/generation_config.json",
    "model/pytorch_model.bin",
    "model/source.spm",
    "model/target.spm",
    "model/tokenizer_config.json",
    "model/vocab.json",
    "manifests/MODEL_REVISION.txt",
    "manifests/asset_manifest.json",
    "manifests/file_inventory.jsonl",
)


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def _complete(root: Path, required: tuple[str, ...]) -> bool:
    return root.is_dir() and all((root / name).is_file() for name in required)


def _safe_extract(archive_path: Path, output_root: Path) -> Path:
    output = output_root.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Materialization root already exists: {output}")
    staging = output.with_name(f".{output.name}.extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("ZIP contains duplicate members")
            for info in archive.infolist():
                if info.is_dir():
                    continue
                relative = Path(info.filename.replace("\\", "/"))
                target = (staging / relative).resolve(strict=False)
                if relative.is_absolute() or staging.resolve() not in target.parents:
                    raise ValueError(f"Unsafe ZIP member: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return output
    except (BadZipFile, OSError, ValueError):
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _bounded_roots(
    root: Path,
    required: tuple[str, ...],
    *,
    max_depth: int = 4,
    max_directories: int = 500,
) -> list[Path]:
    matches: list[Path] = []
    frontier = [(root, 0)]
    seen: set[Path] = set()
    while frontier and len(seen) < max_directories:
        current, depth = frontier.pop(0)
        current = current.resolve(strict=False)
        if current in seen:
            continue
        seen.add(current)
        if _complete(current, required):
            matches.append(current)
            continue
        if depth >= max_depth or not current.is_dir():
            continue
        try:
            frontier.extend(
                (child, depth + 1)
                for child in sorted(current.iterdir(), key=lambda item: item.name)
                if child.is_dir()
            )
        except OSError:
            continue
    return sorted(set(matches))


def resolve_input_root(
    requested_root: str | Path,
    *,
    required: tuple[str, ...],
    materialize_root: str | Path,
    search_root: str | Path | None = None,
    archive_keyword: str,
) -> tuple[Path, str]:
    requested = Path(requested_root).expanduser().resolve(strict=False)
    if _complete(requested, required):
        return requested, "DIRECT_DIRECTORY"
    roots = [requested] if requested.is_dir() else []
    if search_root is not None:
        search = Path(search_root).expanduser().resolve(strict=True)
        if search not in roots:
            roots.append(search)
    matches: list[Path] = []
    archives: list[Path] = []
    if requested.is_file() and requested.suffix.lower() == ".zip":
        archives.append(requested)
    for root in roots:
        matches.extend(_bounded_roots(root, required))
        try:
            archives.extend(
                path.resolve()
                for path in root.rglob("*.zip")
                if archive_keyword in path.name.lower()
                and len(path.relative_to(root).parts) <= 5
            )
        except (OSError, ValueError):
            continue
    matches = sorted(set(matches))
    archives = sorted(set(archives))
    if len(matches) == 1:
        return matches[0], "DIRECT_DIRECTORY"
    target = Path(materialize_root).expanduser().resolve(strict=False)
    if _complete(target, required):
        nested = _bounded_roots(target, required)
        if len(nested) == 1:
            return nested[0], "EXTRACTED_ZIP"
    if not matches and len(archives) == 1:
        extracted = _safe_extract(archives[0], target)
        nested = _bounded_roots(extracted, required)
        if len(nested) == 1:
            return nested[0], "EXTRACTED_ZIP"
    raise FileNotFoundError(
        f"Expected one complete {archive_keyword} input; found "
        f"{len(matches)} directories and {len(archives)} ZIPs"
    )


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_translator_asset(root: str | Path) -> dict[str, Any]:
    try:
        asset = Path(root).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"TRANSLATOR_ASSET_NOT_FOUND: {Path(root).expanduser()}"
        ) from error
    if not _complete(asset, TRANSLATOR_REQUIRED):
        raise FileNotFoundError("TRANSLATOR_ASSET_NOT_FOUND: required runtime files are missing")
    try:
        manifest = read_json(asset / "manifests/asset_manifest.json")
        revision = (asset / "manifests/MODEL_REVISION.txt").read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, ValueError) as error:
        raise ValueError(f"TRANSLATOR_ASSET_MANIFEST_INVALID: {error}") from error
    if manifest.get("model_id") != TRANSLATOR_MODEL_ID:
        raise ValueError("TRANSLATOR_ASSET_MANIFEST_INVALID: unexpected model_id")
    if manifest.get("exact_revision") != TRANSLATOR_REVISION or revision != TRANSLATOR_REVISION:
        raise ValueError("TRANSLATOR_REVISION_MISMATCH")
    if manifest.get("internet_required_at_runtime") is not False:
        raise ValueError("TRANSLATOR_ASSET_MANIFEST_INVALID: asset is not offline")
    runtime_relative = Path(str(manifest.get("runtime_model_path", "")))
    model_root = (asset / runtime_relative).resolve(strict=False)
    if (
        runtime_relative.is_absolute()
        or asset not in model_root.parents
        or model_root != (asset / "model").resolve()
    ):
        raise ValueError("TRANSLATOR_ASSET_MANIFEST_INVALID: unsafe runtime path")
    inventory_path = asset / "manifests/file_inventory.jsonl"
    inventory: list[dict[str, Any]] = []
    try:
        for line in inventory_path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("inventory row is not an object")
            inventory.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"TRANSLATOR_ASSET_MANIFEST_INVALID: {error}") from error
    by_path = {str(item.get("path")): item for item in inventory}
    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, dict):
        raise ValueError("TRANSLATOR_ASSET_MANIFEST_INVALID: file_hashes missing")
    checked = []
    for required in TRANSLATOR_REQUIRED:
        path = asset / required
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"TRANSLATOR_ASSET_NOT_FOUND: {required}")
        if not required.startswith("model/"):
            continue
        item = by_path.get(required)
        expected = item.get("sha256") if isinstance(item, dict) else None
        if not expected or file_hashes.get(required) != expected:
            raise ValueError(f"TRANSLATOR_ASSET_MANIFEST_INVALID: hash missing for {required}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"TRANSLATOR_FILE_HASH_MISMATCH: {required}")
        checked.append(
            {"path": required, "size_bytes": path.stat().st_size, "sha256": actual}
        )
    return {
        "status": "VALID",
        "asset_root": asset,
        "model_root": model_root,
        "model_id": TRANSLATOR_MODEL_ID,
        "exact_revision": TRANSLATOR_REVISION,
        "architecture": manifest.get("architecture"),
        "runtime_files": checked,
        "hash_verification": "PASS",
        "local_only": True,
    }


@dataclass(frozen=True)
class FrozenBaseline:
    root: Path
    summary: dict[str, Any]
    queries: list[QueryRecord]
    query_suite_fingerprint: str
    frames_by_query: dict[str, list[dict[str, Any]]]
    videos_by_query: dict[str, list[dict[str, Any]]]
    diagnostics_by_query: dict[str, dict[str, Any]]
    pairs: dict[str, dict[str, QueryRecord]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSONL artifact {path}: {error}") from error
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"Expected JSON objects in {path}")
    return values


def load_frozen_baseline(
    root: str | Path,
    *,
    stage1_fingerprint: str,
    stage1b_contract: dict[str, Any],
    stage1b_runtime: dict[str, Any],
    explicit_query_suite: str | Path | None = None,
    expected_query_count: int = 28,
    expected_pair_count: int = 14,
    pair_ids: tuple[str, ...] = (),
) -> FrozenBaseline:
    source = Path(root).expanduser().resolve(strict=True)
    try:
        summary = read_json(source / "stage1c_summary.json")
        queries, suite = load_query_suite(source / "query_suite/query_suite.jsonl")
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError(f"STAGE1C_FROZEN_BASELINE_INVALID: {error}") from error
    retrieval = summary.get("retrieval", {})
    query_summary = summary.get("query_suite", {})
    if (
        summary.get("evaluation_status") not in {"COMPLETE", "COMPLETE_WITH_WARNINGS"}
        or retrieval.get("queries_completed") != expected_query_count
        or retrieval.get("queries_failed") != 0
        or query_summary.get("query_count") != expected_query_count
        or query_summary.get("pair_count") != expected_pair_count
        or sorted(query_summary.get("languages", [])) != ["en", "vi"]
    ):
        raise ValueError("STAGE1C_FROZEN_BASELINE_INVALID: summary contract mismatch")
    saved_fingerprint = query_summary.get("fingerprint")
    if saved_fingerprint != suite["fingerprint"]:
        raise ValueError("STAGE1C_QUERY_SUITE_FINGERPRINT_MISMATCH")
    if explicit_query_suite is not None:
        _, explicit = load_query_suite(explicit_query_suite)
        if explicit["fingerprint"] != saved_fingerprint:
            raise ValueError("STAGE1C_QUERY_SUITE_FINGERPRINT_MISMATCH")
    if summary.get("stage1_index_fingerprint") != stage1_fingerprint:
        raise ValueError("STAGE1_INDEX_FINGERPRINT_MISMATCH")
    frozen_encoder = summary.get("stage1b_encoder", {})
    if (
        frozen_encoder.get("candidate_id") != stage1b_contract.get("selected_candidate_id")
        or frozen_encoder.get("compatibility_status") != "VERIFIED"
        or frozen_encoder.get("model_space_status")
        != stage1b_runtime.get("model_space_status")
        or frozen_encoder.get("checkpoint_sha256")
        != stage1b_contract.get("checkpoint_sha256")
    ):
        raise ValueError("STAGE1B_ENCODER_NOT_VERIFIED")
    pairs: dict[str, dict[str, QueryRecord]] = {}
    for query in queries:
        pairs.setdefault(query.pair_id, {})[query.language] = query
    if len(pairs) != expected_pair_count or any(
        set(value) != {"en", "vi"} for value in pairs.values()
    ):
        raise ValueError("STAGE1C_FROZEN_BASELINE_INVALID: incomplete query pairs")
    requested = set(pair_ids)
    missing = sorted(requested - set(pairs))
    if missing:
        raise ValueError(f"STAGE1C_FROZEN_BASELINE_INVALID: unknown pairs {missing}")
    selected_pairs = {
        pair_id: value
        for pair_id, value in sorted(pairs.items())
        if not requested or pair_id in requested
    }
    frames_by_query: dict[str, list[dict[str, Any]]] = {}
    videos_by_query: dict[str, list[dict[str, Any]]] = {}
    diagnostics_by_query: dict[str, dict[str, Any]] = {}
    for members in selected_pairs.values():
        for query in members.values():
            query_root = source / "queries" / query.query_id
            frame_path = query_root / "ranked_frames.jsonl"
            video_path = query_root / "ranked_videos.jsonl"
            diagnostics_path = query_root / "retrieval_diagnostics.json"
            if (
                not frame_path.is_file()
                or not video_path.is_file()
                or not diagnostics_path.is_file()
            ):
                raise ValueError(
                    f"STAGE1C_FROZEN_BASELINE_INVALID: missing ranking for {query.query_id}"
                )
            frames = _read_jsonl(frame_path)
            videos = _read_jsonl(video_path)
            if len(frames) < 50 or [item.get("rank") for item in frames[:50]] != list(
                range(1, 51)
            ):
                raise ValueError(
                    f"STAGE1C_FROZEN_BASELINE_INVALID: raw Top-50 invalid for {query.query_id}"
                )
            frames_by_query[query.query_id] = frames
            videos_by_query[query.query_id] = videos
            diagnostics_by_query[query.query_id] = read_json(diagnostics_path)
    selected_queries = [
        query
        for query in queries
        if query.pair_id in selected_pairs and query.language in {"en", "vi"}
    ]
    return FrozenBaseline(
        source,
        summary,
        selected_queries,
        str(saved_fingerprint),
        frames_by_query,
        videos_by_query,
        diagnostics_by_query,
        selected_pairs,
    )
