"""Canonicalize the human-authored L21 draft into a TEAM-EVAL benchmark."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .census import build_corpus_inventory
from .contracts import SLICE_NAMES, validate_query
from .io import read_jsonl, sha256_file, write_json, write_jsonl
from .mapping import read_mapping
from .render import (
    RawDecoder,
    decode_raw_frames,
    evenly_spaced,
    render_actual_frame_sheet,
)

BENCHMARK_ID = "DEV_L21_150"
BENCHMARK_VERSION = "v1"
GT_SOURCE = "SOURCE_HUMAN_TIMESTAMP_INTERVAL_TECHNICALLY_CANONICALIZED"
REQUIRED_DRAFT_FILES = {
    "queries.jsonl",
    "gt_provisional.jsonl",
    "anchor_index_provisional.jsonl",
    "manifest.json",
}
BENCHMARK_FILES = {"queries.jsonl", "gt.jsonl", "manifest.json", "annotation_audit.jsonl"}
REGISTRY = {
    "registry_version": "TEAM_EVAL_v1",
    "benchmarks": [
        {
            "benchmark_id": "DEV_L21_150",
            "role": "PUBLIC_REGRESSION_DEBUG",
            "scores_reported_separately": True,
        },
        {
            "benchmark_id": "DEV_CROSS_60",
            "role": "PUBLIC_CROSS_LEVEL_DEVELOPMENT",
            "scores_reported_separately": True,
        },
        {
            "benchmark_id": "SEALED_FINAL_30",
            "role": "FINAL_HELDOUT_GATE",
            "content_in_development_bundle": False,
        },
    ],
    "combined_unweighted_dev_score_allowed": False,
}


def _extract_named(archive_path: Path, output: Path, allowed: set[str]) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            parts = Path(member.filename).parts
            if (
                member.is_dir()
                or name not in allowed
                or ".." in parts
                or any("sealed" in part.casefold() for part in parts)
            ):
                continue
            if name in found:
                raise ValueError(f"duplicate archive member basename: {name}")
            target = output / name
            target.write_bytes(archive.read(member))
            found[name] = target
    missing = sorted(allowed - set(found))
    if missing:
        raise ValueError(f"archive missing required files: {missing}")
    return found


def normalize_queries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical_tags = {name.casefold(): name for name in SLICE_NAMES}
    normalized = []
    for row in rows:
        value = dict(row)
        difficulty = str(value.get("difficulty", "")).strip().lower()
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"invalid difficulty for {value.get('query_id')}: {difficulty}")
        tags = value.get("tags")
        if not isinstance(tags, list):
            raise ValueError(f"tags must be a list for {value.get('query_id')}")
        normalized_tags = []
        for tag in tags:
            canonical = canonical_tags.get(str(tag).strip().casefold())
            if canonical is None:
                raise ValueError(f"unknown TEAM-EVAL tag for {value.get('query_id')}: {tag}")
            if canonical not in normalized_tags:
                normalized_tags.append(canonical)
        value["difficulty"] = difficulty
        value["tags"] = normalized_tags
        value["language"] = str(value.get("language", "")).strip().lower()
        if not value["language"]:
            raise ValueError(f"language is required for {value.get('query_id')}")
        if value.get("task") == "QA" and value.get("qa_prompt_split_status") == (
            "SOURCE_COMBINED_PROMPT_NOT_SPLIT"
        ):
            value["question"] = value["query"]
        if value.get("task") == "TRAKE":
            descriptions = value.get("event_descriptions")
            if not isinstance(descriptions, list) or len(descriptions) != value.get("event_count"):
                raise ValueError("TRAKE event_descriptions must match event_count")
        normalized.append(validate_query(value))
    counts = Counter(row["task"] for row in normalized)
    if len(normalized) != 150 or counts != {"KIS": 50, "QA": 50, "TRAKE": 50}:
        raise ValueError(f"L21 query contract requires 150 and 50/50/50; got {dict(counts)}")
    if len({row["query_id"] for row in normalized}) != 150:
        raise ValueError("L21 query_id values must be unique")
    return normalized


def aliases_from_source(values: Any) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("QA provisional_accepted_answers must be a non-empty list")
    aliases = []
    for raw in values:
        answer = str(raw).strip()
        if not answer:
            raise ValueError("QA source answer must not be empty")
        candidates = [answer]
        if "/" in answer:
            candidates.extend(part.strip() for part in answer.split("/") if part.strip())
        for candidate in candidates:
            if candidate not in aliases:
                aliases.append(candidate)
    return aliases


def validate_draft_integrity(
    queries: list[dict[str, Any]],
    provisional_gt: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> None:
    query_by_id = {row["query_id"]: row for row in queries}
    gt_by_id = {row.get("query_id"): row for row in provisional_gt}
    if len(provisional_gt) != 150 or len(gt_by_id) != 150 or set(gt_by_id) != set(query_by_id):
        raise ValueError("provisional GT must align one-to-one with all 150 queries")
    gt_counts = Counter(row.get("task") for row in provisional_gt)
    if gt_counts != {"KIS": 50, "QA": 50, "TRAKE": 50}:
        raise ValueError(f"provisional GT task counts are invalid: {dict(gt_counts)}")
    if any(row.get("human_reviewed") is not True for row in provisional_gt):
        raise ValueError("all source GT rows must retain human_reviewed=true provenance")
    if len(anchors) != 99 or len({row.get("anchor_id") for row in anchors}) != 99:
        raise ValueError("draft must contain exactly 99 unique canonicalization anchors")
    source_counts = Counter(
        source_id for anchor in anchors for source_id in anchor.get("source_query_ids", [])
    )
    expected_sources = {row["source_query_id"] for row in queries if row["task"] in {"KIS", "QA"}}
    if set(source_counts) != expected_sources or any(
        count != 1 for count in source_counts.values()
    ):
        raise ValueError("every KIS/QA source anchor must appear exactly once in the anchor index")
    interval_keys = {
        (row.get("video_id"), tuple(row.get("provisional_raw_interval", []))) for row in anchors
    }
    for row in provisional_gt:
        if row["task"] in {"KIS", "QA"}:
            intervals = [row.get("provisional_raw_interval")]
        else:
            intervals = row.get("provisional_event_intervals", [])
            if len(intervals) != row.get("event_count"):
                raise ValueError(f"TRAKE provisional event_count mismatch: {row['query_id']}")
        if any(
            (row.get("correct_video"), tuple(interval or [])) not in interval_keys
            for interval in intervals
        ):
            raise ValueError(
                f"provisional interval is not linked to an indexed anchor: {row['query_id']}"
            )


def _keyframes_by_n(directory: Path) -> dict[int, Path]:
    result = {}
    if directory.is_dir():
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    result[int(path.stem)] = path
                except ValueError:
                    continue
    return result


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def visual_similarity(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    from PIL import Image

    left_gray = np.asarray(
        Image.fromarray(left).convert("L").resize((64, 36), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    right_gray = np.asarray(
        Image.fromarray(right).convert("L").resize((64, 36), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    mae = float(np.mean(np.abs(left_gray - right_gray)) / 255.0)
    if float(left_gray.std()) < 1e-6 or float(right_gray.std()) < 1e-6:
        correlation = 1.0 if mae < 1e-6 else 0.0
    else:
        correlation = float(np.corrcoef(left_gray.ravel(), right_gray.ravel())[0, 1])
    return {
        "grayscale_mae": mae,
        "grayscale_correlation": correlation,
        "correspondence_pass": bool(correlation >= 0.90 or mae <= 0.12),
    }


def verify_frame_coordinate_contract(
    inventory: list[dict[str, Any]],
    *,
    decoder: RawDecoder = decode_raw_frames,
    samples_per_video: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks = []
    for video in sorted(inventory, key=lambda row: row["video_id"]):
        mapping = read_mapping(video["mapping_path"])
        keyframes = _keyframes_by_n(Path(video["keyframe_directory"]))
        sample_rows = evenly_spaced(mapping, samples_per_video)
        frame_ids = list(dict.fromkeys(row["frame_idx"] for row in sample_rows))
        if not Path(video["video_path"]).is_file():
            checks.append(
                {
                    "video_id": video["video_id"],
                    "status": "RAW_VIDEO_MISSING",
                    "samples": [],
                }
            )
            continue
        try:
            decoded_rows = list(decoder(Path(video["video_path"]), frame_ids))
        except Exception as error:
            checks.append(
                {
                    "video_id": video["video_id"],
                    "status": "RAW_DECODE_FAILED",
                    "error": str(error),
                    "samples": [],
                }
            )
            continue
        actual_ids = [frame_id for frame_id, _ in decoded_rows]
        decoded = dict(decoded_rows)
        raw_identity_exact = actual_ids == frame_ids
        samples = []
        for row in sample_rows:
            keyframe_path = keyframes.get(row["n"])
            sample = {
                "btc_n": row["n"],
                "mapping_frame_idx": row["frame_idx"],
                "actual_decoded_frame_idx": row["frame_idx"]
                if row["frame_idx"] in decoded
                else None,
                "raw_decode_identity_exact": row["frame_idx"] in decoded,
                "keyframe_path": str(keyframe_path) if keyframe_path else None,
            }
            try:
                if keyframe_path is None or row["frame_idx"] not in decoded:
                    raise RuntimeError("required decoded raw frame or BTC keyframe is missing")
                sample.update(
                    visual_similarity(decoded[row["frame_idx"]], _load_rgb(keyframe_path))
                )
            except Exception as error:
                sample.update({"correspondence_pass": False, "error": str(error)})
            samples.append(sample)
        checks.append(
            {
                "video_id": video["video_id"],
                "status": "PASS" if raw_identity_exact else "RAW_FRAME_IDENTITY_MISMATCH",
                "btc_jpeg_correspondence_status": (
                    "CLEAN" if samples and all(row["correspondence_pass"] for row in samples)
                    else "HAS_MISMATCH"
                ),
                "samples": samples,
                "duplicate_frame_idx_preserved": True,
            }
        )
    passed = bool(checks) and all(row["status"] == "PASS" for row in checks)
    samples = [sample for check in checks for sample in check.get("samples", [])]
    jpeg_passes = sum(sample.get("correspondence_pass") is True for sample in samples)
    return {
        "status": "PASS" if passed else "FAIL",
        "video_count": len(checks),
        "sample_count": len(samples),
        "btc_jpeg_correspondence_pass_count": jpeg_passes,
        "btc_jpeg_correspondence_fail_count": len(samples) - jpeg_passes,
        "jpeg_correspondence_used_as_hard_gate": False,
        "mapping_authority": "BTC CSV frame_idx",
        "raw_video_source_of_truth": True,
        "timestamp_fps_reconstruction_used": False,
    }, checks


def _relevant_mapping_rows(
    mapping: list[dict[str, Any]], start: int, end: int, fps: float
) -> list[dict[str, Any]]:
    margin = max(1, round(fps * 2))
    relevant = [row for row in mapping if start - margin <= row["frame_idx"] <= end + margin]
    if not relevant and mapping:
        relevant = sorted(
            mapping,
            key=lambda row: min(abs(row["frame_idx"] - start), abs(row["frame_idx"] - end)),
        )[:2]
        relevant.sort(key=lambda row: row["n"])
    return relevant


def audit_anchors(
    anchors: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    *,
    decoder: RawDecoder = decode_raw_frames,
) -> list[dict[str, Any]]:
    by_video = {row["video_id"]: row for row in inventory}
    audits = []
    for source in sorted(anchors, key=lambda row: row["anchor_id"]):
        anchor_id = source.get("anchor_id")
        video_id = source.get("video_id")
        interval = source.get("provisional_raw_interval")
        center = source.get("provisional_raw_reference_frame")
        audit = {
            "anchor_id": anchor_id,
            "video_id": video_id,
            "provisional_raw_interval": interval,
            "provisional_raw_reference_frame": center,
            "source_query_ids": source.get("source_query_ids", []),
            "gt_source": GT_SOURCE,
            "human_reviewed": True,
            "timestamp_fps_final_reconstruction_used": False,
            "duplicate_mapping_rows_preserved": True,
        }
        video = by_video.get(video_id)
        if video is None:
            audit.update({"status": "NEEDS_VISUAL_REVIEW", "reason": "RAW_VIDEO_MISSING"})
            audits.append(audit)
            continue
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in interval)
            or interval[0] < 0
            or interval[1] < interval[0]
            or not isinstance(center, int)
            or not interval[0] <= center <= interval[1]
        ):
            audit.update({"status": "NEEDS_VISUAL_REVIEW", "reason": "SOURCE_INTERVAL_INVALID"})
            audits.append(audit)
            continue
        start, end = interval
        if end >= video["total_frames"]:
            audit.update({"status": "NEEDS_VISUAL_REVIEW", "reason": "OUT_OF_BOUNDS"})
            audits.append(audit)
            continue
        mapping = read_mapping(video["mapping_path"])
        if any(
            row["frame_idx"] < 0 or row["frame_idx"] >= video["total_frames"]
            for row in mapping
        ):
            audit.update(
                {
                    "status": "NEEDS_VISUAL_REVIEW",
                    "reason": "MAPPING_FRAME_IDX_OUT_OF_BOUNDS",
                }
            )
            audits.append(audit)
            continue
        monotonic = all(
            left["frame_idx"] <= right["frame_idx"]
            for left, right in zip(mapping, mapping[1:], strict=False)
        )
        if not monotonic:
            audit.update(
                {
                    "status": "NEEDS_VISUAL_REVIEW",
                    "reason": "MAPPING_NON_MONOTONIC_COORDINATE_CONFLICT",
                }
            )
            audits.append(audit)
            continue
        relevant = _relevant_mapping_rows(mapping, start, end, float(video["fps"]))
        selected = evenly_spaced(relevant, min(5, len(relevant)))
        requested = list(
            dict.fromkeys([start, center, end, *(row["frame_idx"] for row in selected)])
        )
        audit["relevant_mapping_rows"] = relevant
        audit["suggested_raw_frame_ids"] = requested
        try:
            decoded = list(decoder(Path(video["video_path"]), requested))
            actual = [frame_id for frame_id, _ in decoded]
        except Exception as error:
            audit.update(
                {
                    "status": "NEEDS_VISUAL_REVIEW",
                    "reason": "RAW_DECODE_FAILED",
                    "decode_error": str(error),
                }
            )
            audits.append(audit)
            continue
        if actual != requested:
            audit.update(
                {
                    "status": "NEEDS_VISUAL_REVIEW",
                    "reason": "RAW_FRAME_IDENTITY_MISMATCH",
                    "decode_error": f"requested={requested}, actual={actual}",
                }
            )
        else:
            audit.update(
                {
                    "status": "RESOLVED",
                    "reason": "SOURCE_HUMAN_INTERVAL_TECHNICALLY_CANONICALIZED",
                    "canonical_interval": [start, end],
                    "decoded_actual_frame_ids": actual,
                    "btc_frame_inside_interval": any(
                        start <= row["frame_idx"] <= end for row in relevant
                    ),
                }
            )
        audits.append(audit)
    return audits


def materialize_ground_truth(
    queries: list[dict[str, Any]],
    provisional_gt: list[dict[str, Any]],
    anchor_audit: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provisional_by_id = {row["query_id"]: row for row in provisional_gt}
    source_anchor = {
        source_id: row for row in anchor_audit for source_id in row.get("source_query_ids", [])
    }
    interval_anchor = {
        (row["video_id"], tuple(row["provisional_raw_interval"])): row
        for row in anchor_audit
        if isinstance(row.get("provisional_raw_interval"), list)
    }
    output, issues = [], []
    for query in queries:
        source = provisional_by_id.get(query["query_id"])
        if source is None:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "PROVISIONAL_GT_MISSING",
                    "query_id": query["query_id"],
                }
            )
            continue
        base = {
            "query_id": query["query_id"],
            "task": query["task"],
            "correct_video": source["correct_video"],
            "difficulty": query["difficulty"],
            "tags": query["tags"],
            "gt_source": GT_SOURCE,
            "human_reviewed": True,
            "gt_claim": "INTERNAL_BENCHMARK_NOT_OFFICIAL_BTC_GT",
        }
        if query["task"] in {"KIS", "QA"}:
            anchor = source_anchor.get(query.get("source_query_id"))
            if anchor is None or anchor.get("status") != "RESOLVED":
                issues.append(
                    {
                        "severity": "ERROR",
                        "code": "QUERY_ANCHOR_UNRESOLVED",
                        "query_id": query["query_id"],
                        "anchor_id": anchor.get("anchor_id") if anchor else None,
                    }
                )
                continue
            base.update(
                {
                    "acceptable_intervals": [anchor["canonical_interval"]],
                    "anchor_id": anchor["anchor_id"],
                }
            )
            if query["task"] == "QA":
                base.update(
                    {
                        "accepted_answers": aliases_from_source(
                            source.get("provisional_accepted_answers")
                        ),
                        "semantic_review_required": False,
                    }
                )
        else:
            event_anchors = [
                interval_anchor.get((source["correct_video"], tuple(interval)))
                for interval in source.get("provisional_event_intervals", [])
            ]
            if len(event_anchors) != query["event_count"] or any(
                anchor is None or anchor.get("status") != "RESOLVED" for anchor in event_anchors
            ):
                issues.append(
                    {
                        "severity": "ERROR",
                        "code": "TRAKE_LINKED_ANCHOR_UNRESOLVED",
                        "query_id": query["query_id"],
                    }
                )
                continue
            event_intervals = [anchor["canonical_interval"] for anchor in event_anchors]
            if any(
                left[0] >= right[0]
                for left, right in zip(event_intervals, event_intervals[1:], strict=False)
            ):
                issues.append(
                    {
                        "severity": "ERROR",
                        "code": "TRAKE_EVENT_ORDER_INVALID",
                        "query_id": query["query_id"],
                    }
                )
                continue
            base.update(
                {
                    "event_intervals": event_intervals,
                    "event_anchor_ids": [anchor["anchor_id"] for anchor in event_anchors],
                }
            )
        output.append(base)
    return output, issues


def _zip_members(root: Path, target: Path, members: list[Path]) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(root).as_posix())
    return target


def create_l21_bundle(root: Path, target: Path) -> Path:
    missing = sorted(name for name in BENCHMARK_FILES if not (root / name).is_file())
    if missing:
        raise ValueError(f"scoring-ready L21 bundle is incomplete: {missing}")
    members = [root / name for name in sorted(BENCHMARK_FILES)]
    for optional in ("issues.jsonl", "review_requests.jsonl"):
        if (root / optional).is_file():
            members.append(root / optional)
    return _zip_members(root, target, members)


def create_review_bundle(root: Path, target: Path) -> Path:
    allowed = {
        "queries.jsonl",
        "manifest.json",
        "annotation_audit.jsonl",
        "issues.jsonl",
        "review_requests.jsonl",
        "frame_coordinate_contract.json",
        "frame_coordinate_checks.jsonl",
        "coordinate_anomaly_diagnostics.json",
        "failed_sample_neighborhood_diagnostics.jsonl",
        "neighbor_crosscheck.jsonl",
        "expanded_coordinate_checks.jsonl",
    }
    members = [root / name for name in sorted(allowed) if (root / name).is_file()]
    review_root = root / "review"
    if review_root.is_dir():
        members.extend(path for path in sorted(review_root.glob("*.jpg")) if path.is_file())
    return _zip_members(root, target, members)


def create_coordinate_diagnostics_bundle(root: Path, target: Path) -> Path:
    names = {
        "frame_coordinate_contract.json",
        "frame_coordinate_checks.jsonl",
        "coordinate_anomaly_diagnostics.json",
        "failed_sample_neighborhood_diagnostics.jsonl",
        "neighbor_crosscheck.jsonl",
        "expanded_coordinate_checks.jsonl",
    }
    missing = sorted(name for name in names if not (root / name).is_file())
    if missing:
        raise ValueError(f"coordinate diagnostic bundle missing files: {missing}")
    return _zip_members(root, target, [root / name for name in sorted(names)])


def create_development_bundle(
    l21_root: Path,
    dev_cross_zip: Path,
    output_root: Path,
    target: Path,
) -> Path:
    cross_temp = output_root / "_cross_extract"
    cross = _extract_named(dev_cross_zip, cross_temp, BENCHMARK_FILES)
    cross_manifest = json.loads(cross["manifest.json"].read_text(encoding="utf-8"))
    if cross_manifest.get("benchmark_id") != "DEV_CROSS_60" or not cross_manifest.get(
        "scoring_ready"
    ):
        raise ValueError("DEV_CROSS_60 input is not the unchanged scoring-ready benchmark")
    bundle = output_root / "team_dev_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    for prefix, files in (
        ("benchmarks/dev_l21_150", {name: l21_root / name for name in BENCHMARK_FILES}),
        ("benchmarks/dev_cross_60", cross),
    ):
        destination = bundle / prefix
        destination.mkdir(parents=True, exist_ok=True)
        for name, path in files.items():
            shutil.copy2(path, destination / name)
    write_json(bundle / "benchmark_registry.json", REGISTRY)
    (bundle / "README.md").write_text(
        "# AIC2026 TEAM-EVAL development benchmarks\n\n"
        "DEV_L21_150 and DEV_CROSS_60 must be scored and reported separately. "
        "No unweighted 210-query aggregate is defined. SEALED_FINAL_30 content is "
        "intentionally absent. Use the shared evaluator for KIS, QA, and TRAKE R@1/5/20/50/100.\n",
        encoding="utf-8",
    )
    members = [path for path in sorted(bundle.rglob("*")) if path.is_file()]
    if any("sealed" in path.as_posix().casefold() for path in members):
        raise RuntimeError("development bundle attempted to include sealed content")
    return _zip_members(bundle, target, members)


def run_l21_finalization(
    *,
    dataset_root: str | Path,
    draft_zip: str | Path,
    dev_cross_zip: str | Path | None,
    output_root: str | Path,
    l21_zip_path: str | Path,
    dev_zip_path: str | Path,
    git_commit: str,
    decoder: RawDecoder = decode_raw_frames,
) -> dict[str, Any]:
    dataset = Path(dataset_root).resolve(strict=True)
    draft = Path(draft_zip).resolve(strict=True)
    output = Path(output_root)
    if not output.name.startswith("aic2026_team_eval_l21"):
        raise ValueError("L21 output directory must start with aic2026_team_eval_l21")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    extracted = _extract_named(draft, output / "_draft", REQUIRED_DRAFT_FILES)
    queries = normalize_queries(read_jsonl(extracted["queries.jsonl"]))
    provisional_gt = read_jsonl(extracted["gt_provisional.jsonl"])
    anchors = read_jsonl(extracted["anchor_index_provisional.jsonl"])
    validate_draft_integrity(queries, provisional_gt, anchors)
    source_videos = {row["correct_video"] for row in provisional_gt}
    if len(source_videos) != 16:
        raise ValueError(f"draft integrity failed: source videos={len(source_videos)}")
    inventory, corpus_summary, inventory_issues = build_corpus_inventory(
        dataset, video_ids=source_videos
    )
    missing_source_videos = source_videos - {row["video_id"] for row in inventory}
    if missing_source_videos:
        raise RuntimeError(
            f"L21 raw inventory is incomplete; missing={sorted(missing_source_videos)}"
        )
    coordinate_summary, coordinate_checks = verify_frame_coordinate_contract(
        inventory, decoder=decoder
    )
    from .coordinate_anomaly import close_coordinate_policy

    (
        coordinate_decision,
        anomaly_diagnostics,
        neighbor_crosscheck,
        expanded_coordinate_checks,
    ) = close_coordinate_policy(inventory, coordinate_checks)
    coordinate_summary = {
        **coordinate_summary,
        **coordinate_decision,
    }
    write_json(output / "frame_coordinate_contract.json", coordinate_summary)
    write_jsonl(output / "frame_coordinate_checks.jsonl", coordinate_checks)
    write_json(
        output / "coordinate_anomaly_diagnostics.json",
        {
            "diagnostic_scope": "FINAL_POLICY_CLOSURE_FROM_COMPLETED_BOUNDED_KAGGLE_EVIDENCE",
            "status": coordinate_summary["status"],
            "failed_sample_count": len(anomaly_diagnostics),
            "classifications": dict(Counter(row["classification"] for row in anomaly_diagnostics)),
            "raw_frame_coordinate_contract": coordinate_summary["raw_frame_coordinate_contract"],
            "btc_jpeg_correspondence_cleanliness": coordinate_summary[
                "btc_jpeg_correspondence_cleanliness"
            ],
            "similarity_thresholds_changed": False,
            "semantic_gt_touched_before_decision": False,
            "neighborhood_search_rerun": False,
            "jpeg_correspondence_used_as_hard_gate": False,
        },
    )
    write_jsonl(output / "failed_sample_neighborhood_diagnostics.jsonl", anomaly_diagnostics)
    write_jsonl(output / "neighbor_crosscheck.jsonl", neighbor_crosscheck)
    write_jsonl(output / "expanded_coordinate_checks.jsonl", expanded_coordinate_checks)
    coordinate_diagnostics_archive = create_coordinate_diagnostics_bundle(
        output, output.parent / "aic2026_l21_coordinate_diagnostics_v1.zip"
    )
    issues = list(inventory_issues)
    coordinate_accepted = coordinate_summary["status"] in {
        "PASS",
        "PASS_WITH_DOCUMENTED_LOCAL_BTC_ASSET_OFFSETS",
    }
    if not coordinate_accepted:
        issues.append(
            {
                "severity": "ERROR",
                "code": "L21_FRAME_COORDINATE_CONTRACT_NOT_ACCEPTED",
                "status": coordinate_summary["status"],
            }
        )
        write_jsonl(output / "queries.jsonl", queries)
        write_json(
            output / "manifest.json",
            {
                "benchmark_id": BENCHMARK_ID,
                "benchmark_version": BENCHMARK_VERSION,
                "role": "PUBLIC_REGRESSION_DEBUG",
                "status": f"{coordinate_summary['status']}_FRAME_COORDINATE_CONTRACT",
                "scoring_ready": False,
                "query_count": 150,
                "canonical_anchor_count": len(anchors),
                "resolved_anchor_count": 0,
                "unresolved_anchor_count": len(anchors),
                "gt_claim": "INTERNAL_BENCHMARK_NOT_OFFICIAL_BTC_GT",
                "frame_coordinate_contract": coordinate_summary["status"],
                "btc_jpeg_correspondence_cleanliness": coordinate_summary[
                    "btc_jpeg_correspondence_cleanliness"
                ],
                "raw_video_source_of_truth": True,
                "mapping_authority": "BTC CSV frame_idx",
                "semantic_intervals_modified_due_to_btc_jpeg_anomalies": False,
            },
        )
        write_jsonl(output / "issues.jsonl", issues)
        review_archive = create_review_bundle(
            output, output.parent / "aic2026_dev_l21_150_review_v1.zip"
        )
        return {
            "TEAM_EVAL_L21_FINALIZATION": (
                "FAIL" if coordinate_summary["status"] == "FAIL" else "PARTIAL"
            ),
            "L21_QUERY_COUNT": len(queries),
            "L21_KIS_COUNT": 50,
            "L21_QA_COUNT": 50,
            "L21_TRAKE_COUNT": 50,
            "L21_FRAME_COORDINATE_CONTRACT": coordinate_summary["status"],
            "BTC_LOCAL_ASSET_OFFSETS_DOCUMENTED": coordinate_summary[
                "documented_btc_local_asset_offset_count"
            ],
            "L21_CANONICAL_ANCHORS": len(anchors),
            "L21_RESOLVED_ANCHORS": 0,
            "L21_UNRESOLVED_ANCHORS": len(anchors),
            "DEV_L21_150_SCORING_READY": "NO",
            "DEV_CROSS_60_STATUS": "UNCHANGED_READY",
            "SEALED_FINAL_30_STATUS": "UNCHANGED_SEALED",
            "TEAM_DEV_BUNDLE": "BLOCKED",
            "RETURN_TO_MAIN_PIPELINE": "YES",
            "corpus_summary": corpus_summary,
            "l21_zip_path": None,
            "dev_zip_path": None,
            "review_zip_path": str(review_archive),
            "coordinate_diagnostics_zip_path": str(coordinate_diagnostics_archive),
            "coordinate_summary": coordinate_summary,
            "coordinate_anomalies": anomaly_diagnostics,
            "EXPANDED_COORDINATE_EVIDENCE": (
                "FAIL" if coordinate_summary["status"] == "FAIL" else "UNRESOLVED"
            ),
        }
    audit = audit_anchors(anchors, inventory, decoder=decoder)
    unresolved = [row for row in audit if row["status"] != "RESOLVED"]
    gt, materialization_issues = materialize_ground_truth(queries, provisional_gt, audit)
    issues.extend(materialization_issues)
    review_requests = []
    inventory_by_id = {row["video_id"]: row for row in inventory}
    for row in unresolved:
        request = {
            "anchor_id": row["anchor_id"],
            "video_id": row["video_id"],
            "provisional_interval": row.get("provisional_raw_interval"),
            "reason": row.get("reason"),
            "relevant_mapping_rows": row.get("relevant_mapping_rows", []),
            "suggested_raw_frame_ids": row.get("suggested_raw_frame_ids", []),
        }
        video = inventory_by_id.get(row["video_id"])
        if video is not None and request["suggested_raw_frame_ids"]:
            try:
                decoded = decoder(Path(video["video_path"]), request["suggested_raw_frame_ids"])
                sheet = output / "review" / f"{row['anchor_id']}.jpg"
                render_actual_frame_sheet(
                    sheet,
                    title=f"{row['anchor_id']} | unresolved | {row.get('reason')}",
                    decoded_frames=decoded,
                )
                request["review_sheet"] = sheet.relative_to(output).as_posix()
            except Exception as error:
                request["review_render_error"] = str(error)
        review_requests.append(request)
    scoring_ready = not unresolved and not materialization_issues and len(gt) == 150
    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "role": "PUBLIC_REGRESSION_DEBUG",
        "query_count": 150,
        "KIS": 50,
        "QA": 50,
        "TRAKE": 50,
        "task_counts": {"KIS": 50, "QA": 50, "TRAKE": 50},
        "source_video_count": 16,
        "status": "READY" if scoring_ready else "PARTIAL_NEEDS_REVIEW",
        "scoring_ready": scoring_ready,
        "canonical_anchor_count": len(audit),
        "resolved_anchor_count": len(audit) - len(unresolved),
        "unresolved_anchor_count": len(unresolved),
        "frame_coordinate_semantics": "original_frame_idx / raw original video frame coordinate",
        "mapping_authority": "BTC CSV frame_idx",
        "raw_video_source_of_truth": True,
        "gt_claim": "INTERNAL_BENCHMARK_NOT_OFFICIAL_BTC_GT",
        "source_human_reviewed": True,
        "timestamp_fps_final_reconstruction_used": False,
        "strict_plus_minus_4_trake_windows_used": False,
        "frame_coordinate_contract_status": coordinate_summary["status"],
        "frame_coordinate_contract": coordinate_summary["status"],
        "btc_jpeg_correspondence_cleanliness": coordinate_summary[
            "btc_jpeg_correspondence_cleanliness"
        ],
        "semantic_intervals_modified_due_to_btc_jpeg_anomalies": False,
        "documented_coordinate_anomalies": coordinate_summary.get("documented_anomalies", []),
        "source_draft_sha256": sha256_file(draft),
        "git_commit": git_commit,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_jsonl(output / "queries.jsonl", queries)
    write_jsonl(output / "annotation_audit.jsonl", audit)
    write_json(output / "manifest.json", manifest)
    write_jsonl(output / "issues.jsonl", issues)
    if review_requests:
        write_jsonl(output / "review_requests.jsonl", review_requests)
    l21_archive = None
    dev_archive = None
    review_archive = None
    if scoring_ready:
        write_jsonl(output / "gt.jsonl", gt)
        l21_archive = create_l21_bundle(output, Path(l21_zip_path))
        if dev_cross_zip is not None:
            dev_archive = create_development_bundle(
                output,
                Path(dev_cross_zip),
                output,
                Path(dev_zip_path),
            )
    else:
        review_archive = create_review_bundle(
            output, output.parent / "aic2026_dev_l21_150_review_v1.zip"
        )
    shutil.rmtree(output / "_draft", ignore_errors=True)
    shutil.rmtree(output / "_cross_extract", ignore_errors=True)
    complete = scoring_ready and dev_archive is not None
    return {
        "TEAM_EVAL_L21_FINALIZATION": "COMPLETE" if complete else "PARTIAL",
        "L21_QUERY_COUNT": len(queries),
        "L21_KIS_COUNT": 50,
        "L21_QA_COUNT": 50,
        "L21_TRAKE_COUNT": 50,
        "L21_FRAME_COORDINATE_CONTRACT": coordinate_summary["status"],
        "BTC_LOCAL_ASSET_OFFSETS_DOCUMENTED": coordinate_summary[
            "documented_btc_local_asset_offset_count"
        ],
        "L21_CANONICAL_ANCHORS": len(audit),
        "L21_RESOLVED_ANCHORS": len(audit) - len(unresolved),
        "L21_UNRESOLVED_ANCHORS": len(unresolved),
        "DEV_L21_150_SCORING_READY": "YES" if scoring_ready else "NO",
        "DEV_CROSS_60_STATUS": "UNCHANGED_READY",
        "SEALED_FINAL_30_STATUS": "UNCHANGED_SEALED",
        "TEAM_DEV_BUNDLE": "READY" if dev_archive else "BLOCKED",
        "RETURN_TO_MAIN_PIPELINE": "YES",
        "corpus_summary": corpus_summary,
        "manifest": manifest,
        "issues": issues,
        "l21_zip_path": str(l21_archive) if l21_archive else None,
        "dev_zip_path": str(dev_archive) if dev_archive else None,
        "review_zip_path": str(review_archive) if review_archive else None,
        "coordinate_diagnostics_zip_path": str(coordinate_diagnostics_archive),
        "coordinate_summary": coordinate_summary,
        "coordinate_anomalies": anomaly_diagnostics,
        "EXPANDED_COORDINATE_EVIDENCE": "PASS",
    }


__all__ = [
    "REGISTRY",
    "aliases_from_source",
    "audit_anchors",
    "create_development_bundle",
    "create_coordinate_diagnostics_bundle",
    "create_l21_bundle",
    "create_review_bundle",
    "materialize_ground_truth",
    "normalize_queries",
    "run_l21_finalization",
    "validate_draft_integrity",
    "verify_frame_coordinate_contract",
    "visual_similarity",
]
