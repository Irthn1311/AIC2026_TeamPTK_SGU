from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from src.preprocessing.yoloe_postprocess import _bbox_containment, _bbox_iou


DEFAULT_HYBRID_CONFIG: dict[str, Any] = {
    "prompt_free_min_confidence": 0.35,
    "prompt_free_min_label_frequency": 2,
    "min_bbox_area_ratio": 0.0001,
    "text_overlap_iou_threshold": 0.70,
    "text_overlap_containment_threshold": 0.90,
    "visualization_limit": 20,
    "prompt_free_reject_labels": [],
}


THRESHOLD_ANALYSIS_VALUES = [0.25, 0.30, 0.35, 0.40, 0.50]


def get_hybrid_config(config: dict[str, Any]) -> dict[str, Any]:
    hybrid = deepcopy(DEFAULT_HYBRID_CONFIG)
    hybrid.update(config.get("hybrid") or {})
    return hybrid


def build_hybrid_outputs(
    text_records: list[dict[str, Any]],
    prompt_free_records: list[dict[str, Any]],
    hybrid_config: dict[str, Any],
    cleanup_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if len(text_records) != len(prompt_free_records):
        raise ValueError(
            f"Text and prompt-free record count mismatch: {len(text_records)} != {len(prompt_free_records)}"
        )

    prompt_free_label_counts = count_prompt_free_candidate_labels(prompt_free_records, hybrid_config)
    hybrid_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []

    for text_record, prompt_free_record in zip(text_records, prompt_free_records):
        _validate_record_pair(text_record, prompt_free_record)
        image_area = _infer_image_area(text_record, prompt_free_record)
        hybrid_record, frame_audit = merge_hybrid_record(
            text_record,
            prompt_free_record,
            prompt_free_label_counts,
            hybrid_config,
            cleanup_config,
            image_area=image_area,
        )
        hybrid_records.append(hybrid_record)
        audit_records.extend(frame_audit)

    stats = summarize_hybrid_records(hybrid_records, audit_records, prompt_free_label_counts)
    return hybrid_records, stats, audit_records


def merge_hybrid_record(
    text_record: dict[str, Any],
    prompt_free_record: dict[str, Any],
    prompt_free_label_counts: Counter[str],
    hybrid_config: dict[str, Any],
    cleanup_config: dict[str, Any],
    image_area: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text_clean = [_with_source(item, "text") for item in text_record.get("detections", [])]
    prompt_free_clean = [_copy_detection(item) for item in prompt_free_record.get("detections", [])]
    hybrid_detections = [_copy_detection(item) for item in text_clean]
    audit_records: list[dict[str, Any]] = []

    for detection in prompt_free_clean:
        decision = _evaluate_prompt_free_detection(
            detection,
            text_clean,
            prompt_free_label_counts,
            hybrid_config,
            cleanup_config,
            image_area,
        )
        audit_item = {
            "video_id": text_record.get("video_id"),
            "frame_id": text_record.get("frame_id"),
            "timestamp": text_record.get("timestamp"),
            "keyframe_name": text_record.get("keyframe_name"),
            "detection": _copy_detection(detection),
            "accepted": decision["accepted"],
            "reason": decision["reason"],
        }
        if decision.get("matched_text_detection"):
            audit_item["matched_text_detection"] = decision["matched_text_detection"]
        audit_records.append(audit_item)

        if decision["accepted"]:
            hybrid_detections.append(_with_source(detection, "prompt_free"))

    hybrid_detections.sort(
        key=lambda item: (
            0 if item.get("source") == "text" else 1,
            -float(item.get("confidence", 0.0)),
            str(item.get("label", "")),
        )
    )

    hybrid_record = {
        "video_id": text_record.get("video_id"),
        "frame_id": text_record.get("frame_id"),
        "timestamp": text_record.get("timestamp"),
        "keyframe_name": text_record.get("keyframe_name"),
        "keyframe_path": text_record.get("keyframe_path"),
        "raw_text_detections": text_record.get("raw_detections", []),
        "raw_prompt_free_detections": prompt_free_record.get("raw_detections", []),
        "text_detections": text_clean,
        "prompt_free_detections": prompt_free_clean,
        "hybrid_detections": hybrid_detections,
    }
    return hybrid_record, audit_records


def count_prompt_free_candidate_labels(
    prompt_free_records: list[dict[str, Any]],
    hybrid_config: dict[str, Any],
) -> Counter[str]:
    min_confidence = float(hybrid_config.get("prompt_free_min_confidence", 0.35))
    min_area_ratio = float(hybrid_config.get("min_bbox_area_ratio", 0.0001))
    counts: Counter[str] = Counter()

    for record in prompt_free_records:
        image_area = _infer_image_area(record)
        for detection in record.get("detections", []):
            if float(detection.get("confidence", 0.0)) < min_confidence:
                continue
            if _bbox_area_ratio(detection.get("bbox", []), image_area) < min_area_ratio:
                continue
            label = str(detection.get("label") or "").strip()
            if label.lower() in _reject_label_set(hybrid_config):
                continue
            if label:
                counts[label] += 1
    return counts


def summarize_hybrid_records(
    hybrid_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
    prompt_free_label_counts: Counter[str],
) -> dict[str, Any]:
    text_counts: Counter[str] = Counter()
    prompt_free_counts: Counter[str] = Counter()
    hybrid_counts: Counter[str] = Counter()
    prompt_free_added_counts: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    accepted_reasons: Counter[str] = Counter()

    total_text = 0
    total_prompt_free = 0
    total_hybrid = 0
    text_only_contributed = 0
    prompt_free_only_contributed = 0
    matched_text_prompt_free = 0

    for record in hybrid_records:
        text_items = record.get("text_detections", [])
        prompt_free_items = record.get("prompt_free_detections", [])
        hybrid_items = record.get("hybrid_detections", [])
        total_text += len(text_items)
        total_prompt_free += len(prompt_free_items)
        total_hybrid += len(hybrid_items)
        text_only_contributed += sum(1 for item in hybrid_items if item.get("source") == "text")
        prompt_free_only_contributed += sum(1 for item in hybrid_items if item.get("source") == "prompt_free")
        text_counts.update(_labels(text_items))
        prompt_free_counts.update(_labels(prompt_free_items))
        hybrid_counts.update(_labels(hybrid_items))
        prompt_free_added_counts.update(_labels(item for item in hybrid_items if item.get("source") == "prompt_free"))

    for audit_item in audit_records:
        reason = str(audit_item.get("reason") or "unknown")
        if audit_item.get("accepted"):
            accepted_reasons[reason] += 1
        else:
            rejected_reasons[reason] += 1
            if reason == "duplicate_with_text":
                matched_text_prompt_free += 1

    total_frames = len(hybrid_records)
    min_frequency = 0
    if prompt_free_label_counts:
        min_frequency = min(prompt_free_label_counts.values())

    return {
        "total_frames": total_frames,
        "frames_with_hybrid_detection": sum(bool(record.get("hybrid_detections")) for record in hybrid_records),
        "text_clean_detections": total_text,
        "prompt_free_clean_detections": total_prompt_free,
        "prompt_free_detections_after_filter": prompt_free_only_contributed,
        "final_hybrid_detections": total_hybrid,
        "text_only_contributed": text_only_contributed,
        "prompt_free_only_contributed": prompt_free_only_contributed,
        "matched_text_prompt_free": matched_text_prompt_free,
        "prompt_free_removed_by_reason": dict(sorted(rejected_reasons.items())),
        "prompt_free_accepted_by_reason": dict(sorted(accepted_reasons.items())),
        "unique_labels_text": sorted(text_counts),
        "unique_labels_prompt_free": sorted(prompt_free_counts),
        "unique_labels_hybrid": sorted(hybrid_counts),
        "unique_label_count_text": len(text_counts),
        "unique_label_count_prompt_free": len(prompt_free_counts),
        "unique_label_count_hybrid": len(hybrid_counts),
        "average_detections_per_frame_text": round(total_text / total_frames, 4) if total_frames else 0.0,
        "average_detections_per_frame_prompt_free": round(total_prompt_free / total_frames, 4) if total_frames else 0.0,
        "average_detections_per_frame_hybrid": round(total_hybrid / total_frames, 4) if total_frames else 0.0,
        "top_prompt_free_added_labels": prompt_free_added_counts.most_common(30),
        "rare_prompt_free_labels": sorted(
            label for label, count in prompt_free_label_counts.items() if count <= max(1, min_frequency)
        )[:100],
        "prompt_free_candidate_label_frequency": dict(sorted(prompt_free_label_counts.items())),
    }


def threshold_analysis(
    records_by_video: dict[str, list[dict[str, Any]]],
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or THRESHOLD_ANALYSIS_VALUES
    output: dict[str, Any] = {}

    for video_id, records in records_by_video.items():
        total_frames = len(records)
        video_rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            detections = [
                detection
                for record in records
                for detection in record.get("detections", [])
                if float(detection.get("confidence", 0.0)) >= threshold
            ]
            labels = Counter(str(item.get("label", "")) for item in detections if item.get("label"))
            frames_with_detection = sum(
                any(float(item.get("confidence", 0.0)) >= threshold for item in record.get("detections", []))
                for record in records
            )
            video_rows.append(
                {
                    "confidence_threshold": threshold,
                    "total_frames": total_frames,
                    "frames_with_detection": frames_with_detection,
                    "detections_kept": len(detections),
                    "unique_label_count": len(labels),
                    "average_detections_per_frame": round(len(detections) / total_frames, 4) if total_frames else 0.0,
                    "top_labels": labels.most_common(30),
                }
            )
        output[video_id] = video_rows

    return {
        "mode": "prompt_free_clean_detections_only",
        "note": "Counts are computed from existing prompt-free clean JSON outputs without re-running YOLOE.",
        "videos": output,
    }


def _evaluate_prompt_free_detection(
    detection: dict[str, Any],
    text_detections: list[dict[str, Any]],
    prompt_free_label_counts: Counter[str],
    hybrid_config: dict[str, Any],
    cleanup_config: dict[str, Any],
    image_area: int | None,
) -> dict[str, Any]:
    confidence = float(detection.get("confidence", 0.0))
    if confidence < float(hybrid_config.get("prompt_free_min_confidence", 0.35)):
        return {"accepted": False, "reason": "low_confidence"}

    if _bbox_area_ratio(detection.get("bbox", []), image_area) < float(hybrid_config.get("min_bbox_area_ratio", 0.0001)):
        return {"accepted": False, "reason": "small_bbox"}

    label = str(detection.get("label") or "").strip()
    if label.lower() in _reject_label_set(hybrid_config):
        return {"accepted": False, "reason": "blocked_label"}

    if prompt_free_label_counts.get(label, 0) < int(hybrid_config.get("prompt_free_min_label_frequency", 2)):
        return {"accepted": False, "reason": "rare_label"}

    matched_text = _find_text_duplicate(detection, text_detections, hybrid_config, cleanup_config)
    if matched_text:
        return {
            "accepted": False,
            "reason": "duplicate_with_text",
            "matched_text_detection": matched_text,
        }

    return {"accepted": True, "reason": "accepted_prompt_free"}


def _find_text_duplicate(
    prompt_free_detection: dict[str, Any],
    text_detections: list[dict[str, Any]],
    hybrid_config: dict[str, Any],
    cleanup_config: dict[str, Any],
) -> dict[str, Any] | None:
    iou_threshold = float(hybrid_config.get("text_overlap_iou_threshold", 0.70))
    containment_threshold = float(hybrid_config.get("text_overlap_containment_threshold", 0.90))

    for text_detection in text_detections:
        if not _same_semantic_group(prompt_free_detection, text_detection, cleanup_config):
            continue
        iou = _bbox_iou(prompt_free_detection["bbox"], text_detection["bbox"])
        containment = _bbox_containment(prompt_free_detection["bbox"], text_detection["bbox"])
        if iou >= iou_threshold or containment >= containment_threshold:
            matched = _copy_detection(text_detection)
            matched["overlap_iou"] = round(iou, 4)
            matched["overlap_containment"] = round(containment, 4)
            return matched
    return None


def _same_semantic_group(a: dict[str, Any], b: dict[str, Any], cleanup_config: dict[str, Any]) -> bool:
    labels_a = _detection_label_keys(a)
    labels_b = _detection_label_keys(b)
    if labels_a & labels_b:
        return True

    parent_a = str(a.get("parent_label") or "").lower()
    parent_b = str(b.get("parent_label") or "").lower()
    if parent_a and parent_a == parent_b:
        return True

    groups = cleanup_config.get("normalization", {}).get("dedup_groups", {}) or {}
    for labels in groups.values():
        group = {str(label).lower() for label in labels}
        if labels_a & group and labels_b & group:
            return True
    return False


def _detection_label_keys(detection: dict[str, Any]) -> set[str]:
    return {
        str(detection.get("raw_label") or "").lower(),
        str(detection.get("label") or "").lower(),
    } - {""}


def _reject_label_set(hybrid_config: dict[str, Any]) -> set[str]:
    return {str(label).strip().lower() for label in hybrid_config.get("prompt_free_reject_labels", []) if str(label).strip()}


def _infer_image_area(*records: dict[str, Any]) -> int | None:
    for record in records:
        for key in ("image_width", "width"):
            width = record.get(key)
            height = record.get("image_height") or record.get("height")
            if width and height:
                return int(width) * int(height)
    return None


def _bbox_area_ratio(bbox: list[int], image_area: int | None) -> float:
    if image_area is None or image_area <= 0 or len(bbox) < 4:
        return 1.0
    x1, y1, x2, y2 = bbox[:4]
    area = max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
    return area / image_area


def _validate_record_pair(text_record: dict[str, Any], prompt_free_record: dict[str, Any]) -> None:
    for key in ("video_id", "frame_id", "timestamp", "keyframe_name"):
        if text_record.get(key) != prompt_free_record.get(key):
            raise ValueError(
                "Text/prompt-free record mismatch at "
                f"{key}: {text_record.get(key)!r} != {prompt_free_record.get(key)!r}"
            )


def _with_source(detection: dict[str, Any], source: str) -> dict[str, Any]:
    item = _copy_detection(detection)
    item["source"] = source
    return item


def _copy_detection(detection: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(detection)


def _labels(items) -> list[str]:
    return [str(item.get("label")) for item in items if item.get("label")]
