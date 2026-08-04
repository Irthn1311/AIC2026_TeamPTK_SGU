"""Stage 0 audit orchestration, gates, checkpoints, and final artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from triage_eg.common.run_context import current_git_commit
from triage_eg.data.dataset_survey import KAGGLE_INPUT_ROOT, _is_within
from triage_eg.data.stage0_audit.asset_resolver import (
    discover_layout,
    relative,
    resolve_assets,
    validate_video_id,
)
from triage_eg.data.stage0_audit.auditors import (
    audit_clip,
    audit_keyframes,
    audit_mapping,
    audit_metadata,
    audit_objects,
    probe_video,
)
from triage_eg.data.stage0_audit.contracts import (
    AUDIT_VERSION,
    DATASET_VERSION,
    AuditConfig,
    issue,
)
from triage_eg.data.stage0_audit.writers import (
    FINAL_ARTIFACTS,
    atomic_text,
    markdown_report,
    write_json,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditRunResult:
    output_root: Path
    summary: dict[str, Any]
    paths: dict[str, Path]
    elapsed_seconds: float


def _fingerprint(config: AuditConfig) -> str:
    payload = json.dumps(config.fingerprint_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_sizes(paths) -> dict[str, int | str | None]:
    values = {
        "video": paths.video,
        "mapping": paths.mapping,
        "clip": paths.clip,
        "metadata": paths.metadata,
    }

    def directory_fingerprint(directory: Path) -> str | None:
        if not directory.is_dir():
            return None
        digest = hashlib.sha256()
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_file() and not path.is_symlink():
                digest.update(f"{path.name}:{path.stat().st_size}\n".encode())
        return digest.hexdigest()

    return {
        name: path.stat().st_size if path.is_file() else None for name, path in values.items()
    } | {
        "keyframe_directory_entries": sum(1 for _ in paths.keyframe_directory.iterdir())
        if paths.keyframe_directory.is_dir()
        else None,
        "object_directory_entries": sum(1 for _ in paths.object_directory.iterdir())
        if paths.object_directory.is_dir()
        else None,
        "keyframe_directory_size_fingerprint": directory_fingerprint(paths.keyframe_directory),
        "object_directory_size_fingerprint": directory_fingerprint(paths.object_directory),
    }


def _select_ids(
    all_ids: list[str], video_partitions: dict[str, str], config: AuditConfig
) -> list[str]:
    if config.video_ids:
        for video_id in config.video_ids:
            validate_video_id(video_id)
            if video_id not in all_ids:
                raise ValueError(f"Requested video_id not discovered: {video_id}")
        return sorted(set(config.video_ids))
    if config.mode == "full":
        return all_ids
    grouped: dict[str, list[str]] = {}
    for video_id in all_ids:
        grouped.setdefault(video_partitions.get(video_id, video_id.split("_", 1)[0]), []).append(
            video_id
        )
    selected: list[str] = []
    randomizer = random.Random(config.seed)
    for partition in sorted(grouped):
        choices = sorted(grouped[partition])
        selected.append(choices[randomizer.randrange(len(choices))])
        if len(selected) == config.sample_size:
            break
    remaining = sorted(set(all_ids) - set(selected))
    randomizer.shuffle(remaining)
    selected.extend(remaining[: max(0, config.sample_size - len(selected))])
    return sorted(selected)


def _checkpoint_valid(
    payload: Any, *, video_id: str, fingerprint: str, root: Path, source_sizes: dict[str, Any]
) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("audit_version") == AUDIT_VERSION
        and payload.get("video_id") == video_id
        and payload.get("config_fingerprint") == fingerprint
        and payload.get("dataset_root") == str(root)
        and payload.get("source_file_sizes") == source_sizes
        and payload.get("completion_status") == "COMPLETE"
        and isinstance(payload.get("records"), dict)
    )


def _audit_video(
    root: Path,
    video_id: str,
    video_partitions: dict[str, str],
    keyframe_partitions: dict[str, str],
    config: AuditConfig,
) -> dict[str, Any]:
    paths = resolve_assets(root, video_id, video_partitions, keyframe_partitions)
    for source in (
        paths.video,
        paths.mapping,
        paths.keyframe_directory,
        paths.clip,
        paths.object_directory,
        paths.metadata,
    ):
        if source.exists() and not _is_within(source.resolve(), root):
            raise ValueError(f"Source asset escapes dataset root through a symlink: {source}")
    mapping_record, mapping_rows, mapping_issues = audit_mapping(paths.mapping, video_id)
    expected = {row["n"] for row in mapping_rows}
    keyframe_record, keyframe_details, keyframe_issues = audit_keyframes(root, paths, expected)
    clip_record, clip_issues = audit_clip(
        root,
        paths,
        len(mapping_rows),
        expected_dimension=config.expected_clip_dimension,
        mode=config.clip_validation,
    )
    object_record, object_issues = audit_objects(
        root, paths, expected, mode=config.object_validation, max_bytes=config.max_object_json_bytes
    )
    metadata_record, metadata_issues = audit_metadata(root, paths)
    video_record, video_issues = probe_video(root, paths, timeout=config.ffprobe_timeout_seconds)
    issues = (
        mapping_issues
        + keyframe_issues
        + clip_issues
        + object_issues
        + metadata_issues
        + video_issues
    )
    nb_frames = video_record.get("nb_frames")
    duplicate_sizes = {
        int(frame): len(values)
        for frame, values in mapping_record.get("duplicate_groups", {}).items()
    }
    frame_records: list[dict[str, Any]] = []
    for row in mapping_rows:
        ordinal = row["n"]
        if nb_frames is not None and not 0 <= row["frame_idx"] < nb_frames:
            issues.append(
                issue(
                    "FRAME_IDX_OUT_OF_VIDEO_RANGE",
                    "ERROR",
                    video_id=video_id,
                    ordinal_n=ordinal,
                    asset_type="MAPPING",
                    path=paths.mapping,
                    raw=True,
                    evidence={"frame_idx": row["frame_idx"], "nb_frames": nb_frames},
                )
            )
        key = keyframe_details.get(ordinal, {})
        object_path = paths.object_directory / f"{ordinal:03d}.json"
        frame_records.append(
            {
                "audit_version": AUDIT_VERSION,
                "dataset_version": DATASET_VERSION,
                "video_id": video_id,
                "n": ordinal,
                "clip_row_index": ordinal - 1,
                "pts_time": row["pts_time"],
                "mapping_fps": row["fps"],
                "original_frame_idx": row["frame_idx"],
                "keyframe_relative_path": key.get(
                    "path", relative(root, paths.keyframe_directory / f"{ordinal:03d}.jpg")
                ),
                "keyframe_exists": key.get("exists", False),
                "keyframe_size_bytes": key.get("size_bytes"),
                "keyframe_header_valid": key.get("header_valid", False),
                "object_relative_path": relative(root, object_path),
                "object_exists": object_path.is_file(),
                "duplicate_frame_idx_group_size": duplicate_sizes.get(row["frame_idx"], 1),
                "issues": [],
            }
        )
    btc_blocked = any(item.blocks_btc_baseline for item in issues)
    raw_blocked = btc_blocked or any(item.blocks_raw_video_pipeline for item in issues)
    warnings = any(item.severity in {"INFO", "WARNING"} for item in issues)
    btc_gate = "FAIL" if btc_blocked else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    raw_gate = "FAIL" if raw_blocked else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    cross = {
        "audit_version": AUDIT_VERSION,
        "video_id": video_id,
        "video_exists": paths.video.is_file(),
        "mapping_exists": paths.mapping.is_file(),
        "clip_exists": paths.clip.is_file(),
        "metadata_exists": paths.metadata.is_file(),
        "keyframe_directory_exists": paths.keyframe_directory.is_dir(),
        "object_directory_exists": paths.object_directory.is_dir(),
        "mapping_rows": len(mapping_rows),
        "clip_rows": clip_record["row_count"],
        "keyframe_files": keyframe_record["observed_count"],
        "object_files": object_record["observed_file_count"],
        "counts_all_equal": len(
            {
                len(mapping_rows),
                clip_record["row_count"],
                keyframe_record["observed_count"],
                object_record["observed_file_count"],
            }
        )
        == 1,
        "ordinal_sets_all_equal": keyframe_record["ordinal_set_matches_mapping"]
        and object_record["ordinal_set_matches_mapping"],
        "btc_baseline_gate": btc_gate,
        "raw_video_gate": raw_gate,
        "issues": list(Counter(item.code for item in issues)),
    }
    if not cross["counts_all_equal"]:
        issues.append(
            issue(
                "CROSS_ASSET_COUNT_MISMATCH",
                "ERROR",
                video_id=video_id,
                asset_type="CROSS_ASSET",
                btc=True,
                evidence={
                    name: cross[name]
                    for name in ("mapping_rows", "clip_rows", "keyframe_files", "object_files")
                },
            )
        )
        cross["btc_baseline_gate"] = cross["raw_video_gate"] = "FAIL"
    if not cross["ordinal_sets_all_equal"]:
        issues.append(
            issue(
                "CROSS_ASSET_ORDINAL_MISMATCH",
                "ERROR",
                video_id=video_id,
                asset_type="CROSS_ASSET",
                btc=True,
            )
        )
        cross["btc_baseline_gate"] = cross["raw_video_gate"] = "FAIL"
    return {
        "video": video_record,
        "frames": frame_records,
        "clip": clip_record,
        "object": object_record,
        "metadata": metadata_record,
        "cross": cross,
        "issues": [item.to_dict() for item in issues],
        "source_file_sizes": _source_sizes(paths),
    }


def _overall_gate(records: list[dict[str, Any]], key: str) -> str:
    values = [record[key] for record in records]
    if any(value == "FAIL" for value in values):
        return "FAIL"
    return (
        "PASS_WITH_WARNINGS" if any(value == "PASS_WITH_WARNINGS" for value in values) else "PASS"
    )


def _prepare_output(config: AuditConfig) -> Path:
    root = config.output_root.expanduser().resolve(strict=False)
    dataset = config.dataset_root.expanduser().resolve(strict=False)
    if _is_within(root, dataset) or _is_within(root, KAGGLE_INPUT_ROOT.resolve(strict=False)):
        raise ValueError("Audit output must not be under the read-only dataset or /kaggle/input")
    if root.exists() and config.overwrite:
        if root == root.anchor or len(root.parts) < 3:
            raise ValueError(f"Refusing broad overwrite target: {root}")
        shutil.rmtree(root)
    if root.exists() and any(root.iterdir()) and not config.resume:
        raise FileExistsError(f"Output is non-empty; use --overwrite or --resume: {root}")
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root


def run_audit(config: AuditConfig, *, project_root: str | Path | None = None) -> AuditRunResult:
    started_clock = monotonic()
    started_at = datetime.now(UTC).isoformat()
    root = config.dataset_root.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if config.strict_root and not _is_within(root, KAGGLE_INPUT_ROOT):
        raise ValueError(f"Strict dataset root must be under {KAGGLE_INPUT_ROOT}")
    output = _prepare_output(config)
    fingerprint = _fingerprint(config)
    git_commit = current_git_commit(project_root)
    video_partitions, keyframe_partitions = discover_layout(root)
    mapping_root = root / "map-keyframes-aic25-b1" / "map-keyframes"
    all_ids = sorted(path.stem for path in mapping_root.glob("*.csv") if path.is_file())
    selected = _select_ids(all_ids, video_partitions, config)
    bundles: list[dict[str, Any]] = []
    resumed = 0
    checkpoint_issues: list[dict[str, Any]] = []
    for index, video_id in enumerate(selected, start=1):
        LOGGER.info("Auditing %s (%d/%d)", video_id, index, len(selected))
        paths = resolve_assets(root, video_id, video_partitions, keyframe_partitions)
        sizes = _source_sizes(paths)
        checkpoint_path = output / "checkpoints" / f"{video_id}.json"
        if config.resume and checkpoint_path.is_file():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                checkpoint = None
            if _checkpoint_valid(
                checkpoint,
                video_id=video_id,
                fingerprint=fingerprint,
                root=root,
                source_sizes=sizes,
            ):
                bundles.append(checkpoint["records"])
                resumed += 1
                continue
            mismatch_code = (
                "CHECKPOINT_CONFIG_MISMATCH"
                if isinstance(checkpoint, dict)
                and checkpoint.get("config_fingerprint") != fingerprint
                else "CHECKPOINT_INVALID"
            )
            checkpoint_issues.append(
                issue(
                    mismatch_code,
                    "WARNING",
                    video_id=video_id,
                    asset_type="CHECKPOINT",
                    path=checkpoint_path,
                    message="Checkpoint did not match current config/assets",
                ).to_dict()
            )
        bundle = _audit_video(root, video_id, video_partitions, keyframe_partitions, config)
        bundles.append(bundle)
        write_json(
            checkpoint_path,
            {
                "audit_version": AUDIT_VERSION,
                "video_id": video_id,
                "config_fingerprint": fingerprint,
                "dataset_root": str(root),
                "git_commit": git_commit,
                "source_asset_paths": asdict(paths),
                "source_file_sizes": bundle["source_file_sizes"],
                "completion_status": "COMPLETE",
                "issue_counts": dict(Counter(item["code"] for item in bundle["issues"])),
                "output_record_references": [
                    f"video_manifest.jsonl:{video_id}",
                    f"cross_asset_manifest.jsonl:{video_id}",
                ],
                "records": bundle,
            },
        )
    video_records = [bundle["video"] for bundle in bundles]
    frame_records = [record for bundle in bundles for record in bundle["frames"]]
    clip_records = [bundle["clip"] for bundle in bundles]
    object_records = [bundle["object"] for bundle in bundles]
    metadata_records = [bundle["metadata"] for bundle in bundles]
    cross_records = [bundle["cross"] for bundle in bundles]
    issues = [item for bundle in bundles for item in bundle["issues"]] + checkpoint_issues
    severity = Counter(item["severity"] for item in issues)
    codes = Counter(item["code"] for item in issues)
    summary = {
        "audit_version": AUDIT_VERSION,
        "dataset_root": str(root),
        "dataset_version": DATASET_VERSION,
        "mode": config.mode,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "config_fingerprint": fingerprint,
        "videos_discovered": len(all_ids),
        "videos_selected": len(selected),
        "videos_completed": len(bundles),
        "videos_failed": sum(record["btc_baseline_gate"] == "FAIL" for record in cross_records),
        "videos_resumed": resumed,
        "mapping_rows": len(frame_records),
        "keyframe_files": sum(record["keyframe_files"] for record in cross_records),
        "clip_rows": sum(record["clip_rows"] for record in cross_records),
        "object_files": sum(record["object_files"] for record in cross_records),
        "detections_observed": sum(record["detections_observed"] for record in object_records),
        "metadata_files": sum(record["parse_status"] == "SUCCESS" for record in metadata_records),
        "issues": {
            "total": len(issues),
            "by_severity": dict(sorted(severity.items())),
            "by_code": dict(sorted(codes.items())),
        },
        "gates": {
            "btc_baseline": _overall_gate(cross_records, "btc_baseline_gate"),
            "raw_video": _overall_gate(cross_records, "raw_video_gate"),
        },
        "bounded_evidence_vs_corpus_results": {
            "prior_bounded_numeric_evidence": "LOCKED",
            "stage0_selected_videos": len(selected),
            "stage0_is_full_corpus": config.mode == "full",
        },
        "unknown_contracts": ["bbox_order", "CLIP model compatibility"],
    }
    contract_notes = {
        "audit_version": AUDIT_VERSION,
        "btc_ordinal_contract": "CSV n <-> keyframe/Object n:03d <-> CLIP row n-1",
        "original_frame_policy": (
            "CSV frame_idx is authoritative; never reconstruct from pts_time*fps"
        ),
        "duplicate_frame_idx_policy": "DUPLICATE_MAPPING_PRESERVED",
        "numeric_string_policy": (
            "Trim while preserving raw; reject invalid/null/bool/non-finite; no default/clamp/drop"
        ),
        "bbox_order": "UNKNOWN",
        "coordinate_range_semantics": "[0,1] is bounded evidence until full audit",
        "score_range_semantics": "Probability-like range is not calibrated probability",
        "clip_model_compatibility": "UNVERIFIED",
    }
    paths = {name: output / name for name in FINAL_ARTIFACTS}
    run_manifest = {
        "audit_version": AUDIT_VERSION,
        "status": "COMPLETE",
        "git_commit": git_commit,
        "config_fingerprint": fingerprint,
        "dataset_root": str(root),
        "output_root": str(output),
        "mode": config.mode,
        "selected_video_ids": selected,
        "command_scope": "STAGE_0_BTC_DATA_AUDIT",
    }
    write_json(paths["run_manifest.json"], run_manifest)
    write_json(paths["audit_summary.json"], summary)
    atomic_text(paths["audit_report.md"], markdown_report(summary))
    write_jsonl(
        paths["video_manifest.jsonl"], video_records, sort_key=lambda item: item["video_id"]
    )
    write_jsonl(
        paths["btc_frame_manifest.jsonl"],
        frame_records,
        sort_key=lambda item: (item["video_id"], item["n"]),
    )
    for name, records in (
        ("clip_manifest.jsonl", clip_records),
        ("object_manifest.jsonl", object_records),
        ("metadata_manifest.jsonl", metadata_records),
        ("cross_asset_manifest.jsonl", cross_records),
    ):
        write_jsonl(paths[name], records, sort_key=lambda item: item["video_id"])
    write_jsonl(
        paths["audit_issues.jsonl"],
        issues,
        sort_key=lambda item: (
            item.get("video_id") or "",
            item.get("ordinal_n") or -1,
            item["code"],
        ),
    )
    write_json(paths["contract_notes.json"], contract_notes)
    elapsed = monotonic() - started_clock
    atomic_text(
        output / "logs" / "run.log",
        f"completed videos={len(bundles)} elapsed_seconds={elapsed:.3f}\n",
    )
    return AuditRunResult(output, summary, paths, elapsed)
