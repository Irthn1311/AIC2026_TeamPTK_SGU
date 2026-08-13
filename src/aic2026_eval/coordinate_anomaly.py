"""Targeted diagnosis for isolated L21 BTC JPEG/raw correspondence failures."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .l21_finalize import _keyframes_by_n, _load_rgb, visual_similarity
from .mapping import read_mapping
from .render import RawDecoder, decode_raw_frames, evenly_spaced

TARGET_FAILURES = {
    ("L21_V012", 113, 13323),
    ("L21_V016", 135, 15147),
}
BENIGN_CLASSIFICATIONS = {
    "LOCAL_KEYFRAME_EXTRACTION_OFFSET",
    "LOCAL_BTC_KEYFRAME_OR_MAPPING_ROW_ANOMALY",
    "SHOT_BOUNDARY_AMBIGUITY",
}


def _row_by_n(mapping: list[dict[str, Any]], btc_n: int) -> tuple[int, dict[str, Any]]:
    matches = [(index, row) for index, row in enumerate(mapping) if row["n"] == btc_n]
    if len(matches) != 1:
        raise ValueError(f"expected one mapping row for btc_n={btc_n}; found={len(matches)}")
    return matches[0]


def _search_jpeg_neighborhood(
    video: dict[str, Any],
    jpeg_path: Path,
    mapping_frame_idx: int,
    *,
    decoder: RawDecoder,
    radius_seconds: float = 2.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    radius = max(1, round(float(video["fps"]) * radius_seconds))
    start = max(0, mapping_frame_idx - radius)
    end = min(int(video["total_frames"]) - 1, mapping_frame_idx + radius)
    requested = list(range(start, end + 1))
    decoded = decoder(Path(video["video_path"]), requested)
    actual = [frame_id for frame_id, _ in decoded]
    if actual != requested:
        raise RuntimeError("neighborhood raw decode did not preserve exact frame identities")
    jpeg = _load_rgb(jpeg_path)
    candidates = []
    for frame_id, image in decoded:
        candidates.append({"raw_frame_idx": frame_id, **visual_similarity(image, jpeg)})
    highest_correlation = max(
        candidates,
        key=lambda row: (row["grayscale_correlation"], -row["grayscale_mae"]),
    )
    lowest_mae = min(
        candidates,
        key=lambda row: (row["grayscale_mae"], -row["grayscale_correlation"]),
    )
    mapping_similarity = next(
        row for row in candidates if row["raw_frame_idx"] == mapping_frame_idx
    )
    decoded_by_id = dict(decoded)
    transitions = []
    for left_id, right_id in (
        (mapping_frame_idx - 1, mapping_frame_idx),
        (mapping_frame_idx, mapping_frame_idx + 1),
    ):
        if left_id in decoded_by_id and right_id in decoded_by_id:
            transitions.append(
                {
                    "left_frame_idx": left_id,
                    "right_frame_idx": right_id,
                    **visual_similarity(decoded_by_id[left_id], decoded_by_id[right_id]),
                }
            )
    shot_boundary_evidence = any(
        row["grayscale_correlation"] < 0.50 and row["grayscale_mae"] > 0.15 for row in transitions
    )
    best = highest_correlation
    return {
        "search_start_frame": start,
        "search_end_frame": end,
        "search_radius_seconds": radius_seconds,
        "candidate_count": len(candidates),
        "raw_decode_identity_exact": True,
        "best_raw_frame_idx": best["raw_frame_idx"],
        "best_offset_frames": best["raw_frame_idx"] - mapping_frame_idx,
        "best_offset_seconds": (best["raw_frame_idx"] - mapping_frame_idx) / float(video["fps"]),
        "best_correlation": best["grayscale_correlation"],
        "best_mae": best["grayscale_mae"],
        "highest_correlation_raw_frame_idx": highest_correlation["raw_frame_idx"],
        "highest_correlation": highest_correlation["grayscale_correlation"],
        "lowest_mae_raw_frame_idx": lowest_mae["raw_frame_idx"],
        "lowest_mae": lowest_mae["grayscale_mae"],
        "mapping_frame_similarity": {
            key: value for key, value in mapping_similarity.items() if key != "raw_frame_idx"
        },
        "adjacent_raw_frame_transitions": transitions,
        "shot_boundary_evidence": shot_boundary_evidence,
    }, candidates


def _local_mapping_evidence(
    mapping: list[dict[str, Any]], index: int, fps: float
) -> dict[str, Any]:
    rows = mapping[max(0, index - 5) : index + 6]
    frame_deltas = [
        right["frame_idx"] - left["frame_idx"] for left, right in zip(rows, rows[1:], strict=False)
    ]
    pts_deltas = [
        right["pts_time"] - left["pts_time"] for left, right in zip(rows, rows[1:], strict=False)
    ]
    return {
        "rows": rows,
        "frame_idx_monotonic": all(value >= 0 for value in frame_deltas),
        "pts_time_monotonic": all(value >= 0 for value in pts_deltas),
        "local_frame_idx_spacing": frame_deltas,
        "local_pts_time_spacing": pts_deltas,
        "frame_over_nominal_fps_minus_pts_time": [
            row["frame_idx"] / fps - row["pts_time"] for row in rows
        ],
        "timestamp_fps_used_for_reconstruction": False,
    }


def _neighbor_checks(
    video: dict[str, Any],
    mapping: list[dict[str, Any]],
    index: int,
    failed_jpeg: Any,
    *,
    decoder: RawDecoder,
) -> list[dict[str, Any]]:
    selected = mapping[max(0, index - 2) : index + 3]
    frame_ids = list(dict.fromkeys(row["frame_idx"] for row in selected))
    decoded = dict(decoder(Path(video["video_path"]), frame_ids))
    if set(decoded) != set(frame_ids):
        raise RuntimeError("neighbor raw decode did not preserve exact frame identities")
    keyframes = _keyframes_by_n(Path(video["keyframe_directory"]))
    output = []
    target_n = mapping[index]["n"]
    for row in selected:
        keyframe = keyframes.get(row["n"])
        item = {
            "video_id": video["video_id"],
            "target_btc_n": target_n,
            "neighbor_btc_n": row["n"],
            "mapping_frame_idx": row["frame_idx"],
            "actual_decoded_frame_idx": row["frame_idx"] if row["frame_idx"] in decoded else None,
            "raw_decode_identity_exact": row["frame_idx"] in decoded,
        }
        if keyframe is None:
            item.update({"own_correspondence_pass": False, "error": "BTC_KEYFRAME_MISSING"})
        else:
            metrics = visual_similarity(decoded[row["frame_idx"]], _load_rgb(keyframe))
            item.update(
                {
                    "own_grayscale_mae": metrics["grayscale_mae"],
                    "own_grayscale_correlation": metrics["grayscale_correlation"],
                    "own_correspondence_pass": metrics["correspondence_pass"],
                }
            )
        failed_metrics = visual_similarity(decoded[row["frame_idx"]], failed_jpeg)
        item.update(
            {
                "failed_jpeg_grayscale_mae": failed_metrics["grayscale_mae"],
                "failed_jpeg_grayscale_correlation": failed_metrics["grayscale_correlation"],
                "failed_jpeg_correspondence_pass": failed_metrics["correspondence_pass"],
            }
        )
        output.append(item)
    return output


def _expanded_checks(
    video: dict[str, Any],
    mapping: list[dict[str, Any]],
    failed_index: int,
    *,
    decoder: RawDecoder,
) -> list[dict[str, Any]]:
    selected = evenly_spaced(mapping, 9)
    selected.extend(
        mapping[index]
        for index in range(max(0, failed_index - 2), min(len(mapping), failed_index + 3))
    )
    by_n = {row["n"]: row for row in selected}
    selected = [by_n[n] for n in sorted(by_n)]
    frame_ids = list(dict.fromkeys(row["frame_idx"] for row in selected))
    decoded = dict(decoder(Path(video["video_path"]), frame_ids))
    keyframes = _keyframes_by_n(Path(video["keyframe_directory"]))
    output = []
    for row in selected:
        item = {
            "video_id": video["video_id"],
            "btc_n": row["n"],
            "mapping_frame_idx": row["frame_idx"],
            "actual_decoded_frame_idx": row["frame_idx"] if row["frame_idx"] in decoded else None,
            "raw_decode_identity_exact": row["frame_idx"] in decoded,
            "is_original_failed_row": row["n"] == mapping[failed_index]["n"],
        }
        keyframe = keyframes.get(row["n"])
        if keyframe is None or row["frame_idx"] not in decoded:
            item.update({"correspondence_pass": False, "error": "ASSET_OR_RAW_FRAME_MISSING"})
        else:
            item.update(visual_similarity(decoded[row["frame_idx"]], _load_rgb(keyframe)))
            if not item["correspondence_pass"]:
                neighborhood, _ = _search_jpeg_neighborhood(
                    video,
                    keyframe,
                    row["frame_idx"],
                    decoder=decoder,
                )
                item["diagnostic_best_offset_frames"] = neighborhood["best_offset_frames"]
                item["diagnostic_best_correlation"] = neighborhood["best_correlation"]
                item["diagnostic_best_mae"] = neighborhood["best_mae"]
        output.append(item)
    return output


def classify_failure(
    neighborhood: dict[str, Any],
    neighbors: list[dict[str, Any]],
    local_mapping: dict[str, Any],
    expanded: list[dict[str, Any]],
) -> str:
    if not neighborhood.get("raw_decode_identity_exact"):
        return "UNRESOLVED"
    if not local_mapping.get("frame_idx_monotonic") or not local_mapping.get("pts_time_monotonic"):
        return "REAL_COORDINATE_DRIFT"
    adjacent = [row for row in neighbors if row["neighbor_btc_n"] != row["target_btc_n"]]
    adjacent_pass = sum(row.get("own_correspondence_pass") is True for row in adjacent)
    expanded_non_target = [row for row in expanded if not row["is_original_failed_row"]]
    expanded_pass = sum(row.get("correspondence_pass") is True for row in expanded_non_target)
    expanded_ratio = expanded_pass / len(expanded_non_target) if expanded_non_target else 0.0
    failed_expanded = [row for row in expanded_non_target if not row.get("correspondence_pass")]
    consistent_offsets = [
        row.get("diagnostic_best_offset_frames")
        for row in failed_expanded
        if row.get("diagnostic_best_offset_frames") not in (None, 0)
    ]
    if len(consistent_offsets) >= 2 and len(set(consistent_offsets)) == 1:
        return "REAL_COORDINATE_DRIFT"
    if adjacent_pass != len(adjacent) or expanded_ratio < 0.85:
        return "UNRESOLVED"
    best_matches = bool(
        neighborhood.get("best_correlation", -1.0) >= 0.90
        or neighborhood.get("best_mae", 1.0) <= 0.12
    )
    if best_matches and neighborhood.get("best_offset_frames") != 0:
        if (
            neighborhood.get("shot_boundary_evidence")
            and abs(neighborhood["best_offset_frames"]) <= 2
        ):
            return "SHOT_BOUNDARY_AMBIGUITY"
        return "LOCAL_KEYFRAME_EXTRACTION_OFFSET"
    return "LOCAL_BTC_KEYFRAME_OR_MAPPING_ROW_ANOMALY"


def decide_coordinate_contract(
    original_checks: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    expanded_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    samples = [sample for check in original_checks for sample in check.get("samples", [])]
    passed = sum(sample.get("correspondence_pass") is True for sample in samples)
    failures = len(samples) - passed
    failed_checks = [check for check in original_checks if check.get("status") != "PASS"]
    if any(check.get("status") == "RAW_DECODE_FAILED" for check in failed_checks):
        status = "FAIL"
    elif failures == 0 and failed_checks:
        status = "UNRESOLVED"
    elif failures == 0:
        status = "PASS"
    elif any(item["classification"] == "REAL_COORDINATE_DRIFT" for item in diagnostics):
        status = "FAIL"
    elif any(item["classification"] == "UNRESOLVED" for item in diagnostics):
        status = "UNRESOLVED"
    else:
        raw_identity = all(
            sample.get("actual_decoded_frame_idx") == sample.get("mapping_frame_idx")
            for sample in samples
        ) and all(item.get("raw_decode_identity_exact") for item in expanded_checks)
        local_monotonic = all(
            item["local_mapping_consistency"]["frame_idx_monotonic"]
            and item["local_mapping_consistency"]["pts_time_monotonic"]
            for item in diagnostics
        )
        by_video = Counter()
        pass_by_video = Counter()
        for row in expanded_checks:
            by_video[row["video_id"]] += 1
            pass_by_video[row["video_id"]] += row.get("correspondence_pass") is True
        overwhelming = bool(by_video) and all(
            pass_by_video[video_id] / count >= 0.85 for video_id, count in by_video.items()
        )
        exact_targets = {
            (item["video_id"], item["btc_n"], item["mapping_frame_idx"]) for item in diagnostics
        } == TARGET_FAILURES
        classifications_allowed = all(
            item["classification"] in BENIGN_CLASSIFICATIONS for item in diagnostics
        )
        status = (
            "PASS_WITH_DOCUMENTED_LOCAL_BTC_ANOMALIES"
            if raw_identity
            and local_monotonic
            and overwhelming
            and exact_targets
            and classifications_allowed
            else "UNRESOLVED"
        )
    return {
        "status": status,
        "raw_frame_coordinate_contract": status,
        "btc_jpeg_correspondence_cleanliness": (
            "CLEAN" if failures == 0 else "DOCUMENTED_LOCAL_ANOMALIES"
        ),
        "original_sample_count": len(samples),
        "original_pass_count": passed,
        "original_fail_count": failures,
        "original_check_failure_count": len(failed_checks),
        "documented_anomalies": diagnostics,
        "majority_vote_used_as_contract_basis": False,
        "similarity_thresholds_changed": False,
        "timestamp_fps_reconstruction_used": False,
    }


def diagnose_failed_samples(
    inventory: list[dict[str, Any]],
    original_checks: list[dict[str, Any]],
    *,
    decoder: RawDecoder = decode_raw_frames,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    failed = [
        (check["video_id"], sample)
        for check in original_checks
        for sample in check.get("samples", [])
        if not sample.get("correspondence_pass")
    ]
    if not failed:
        decision = decide_coordinate_contract(original_checks, [], [])
        return decision, [], [], []
    observed = {
        (video_id, sample.get("btc_n"), sample.get("mapping_frame_idx"))
        for video_id, sample in failed
    }
    if observed != TARGET_FAILURES:
        diagnostics = [
            {
                "video_id": video_id,
                "btc_n": sample.get("btc_n"),
                "mapping_frame_idx": sample.get("mapping_frame_idx"),
                "classification": "UNRESOLVED",
                "reason": "UNEXPECTED_FAILURE_SET_OUTSIDE_BOUNDED_DIAGNOSTIC",
            }
            for video_id, sample in failed
        ]
        decision = decide_coordinate_contract(original_checks, diagnostics, [])
        return decision, diagnostics, [], []
    inventory_by_id = {row["video_id"]: row for row in inventory}
    diagnostics, neighbor_rows, expanded_rows = [], [], []
    for video_id, sample in sorted(failed, key=lambda item: (item[0], item[1]["btc_n"])):
        video = inventory_by_id[video_id]
        mapping = read_mapping(video["mapping_path"])
        index, row = _row_by_n(mapping, sample["btc_n"])
        if row["frame_idx"] != sample["mapping_frame_idx"]:
            raise RuntimeError("failed sample no longer matches authoritative mapping row")
        keyframe_path = _keyframes_by_n(Path(video["keyframe_directory"])).get(row["n"])
        if keyframe_path is None:
            raise RuntimeError(f"failed BTC keyframe is missing: {video_id} n={row['n']}")
        neighborhood, candidate_metrics = _search_jpeg_neighborhood(
            video, keyframe_path, row["frame_idx"], decoder=decoder
        )
        neighborhood["candidate_metrics"] = candidate_metrics
        failed_jpeg = _load_rgb(keyframe_path)
        neighbors = _neighbor_checks(video, mapping, index, failed_jpeg, decoder=decoder)
        expanded = _expanded_checks(video, mapping, index, decoder=decoder)
        local_mapping = _local_mapping_evidence(mapping, index, float(video["fps"]))
        classification = classify_failure(neighborhood, neighbors, local_mapping, expanded)
        diagnostic = {
            "video_id": video_id,
            "btc_n": row["n"],
            "mapping_frame_idx": row["frame_idx"],
            **neighborhood,
            "local_mapping_consistency": local_mapping,
            "classification": classification,
            "authoritative_mapping_changed": False,
        }
        diagnostics.append(diagnostic)
        neighbor_rows.extend(neighbors)
        expanded_rows.extend(expanded)
    decision = decide_coordinate_contract(original_checks, diagnostics, expanded_rows)
    return decision, diagnostics, neighbor_rows, expanded_rows


__all__ = [
    "BENIGN_CLASSIFICATIONS",
    "TARGET_FAILURES",
    "classify_failure",
    "decide_coordinate_contract",
    "diagnose_failed_samples",
]
