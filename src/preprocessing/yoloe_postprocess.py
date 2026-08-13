from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any


DEFAULT_CLEANUP_CONFIG: dict[str, Any] = {
    "deduplication": {
        "enabled": True,
        "same_label_iou_threshold": 0.78,
        "synonym_iou_threshold": 0.82,
        "containment_threshold": 0.92,
        "compare_parent_labels": True,
    },
    "normalization": {
        "label_aliases": {},
        "parent_labels": {},
        "dedup_groups": {},
    },
}


def flatten_prompt_classes(classes_config: Any) -> list[str]:
    if isinstance(classes_config, dict):
        values: list[str] = []
        for group_items in classes_config.values():
            values.extend(flatten_prompt_classes(group_items))
        return _dedupe_keep_order(values)
    if isinstance(classes_config, (list, tuple)):
        return _dedupe_keep_order(str(item).strip() for item in classes_config if str(item).strip())
    return []


def get_cleanup_config(config: dict[str, Any]) -> dict[str, Any]:
    cleanup = deepcopy(DEFAULT_CLEANUP_CONFIG)
    user_cleanup = config.get("cleanup") or {}
    _deep_update(cleanup, user_cleanup)
    return cleanup


def clean_detections(
    raw_detections: list[dict[str, Any]],
    cleanup_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched = [_normalize_detection(det, cleanup_config) for det in raw_detections]
    if not cleanup_config.get("deduplication", {}).get("enabled", True):
        return enriched, _cleanup_stats(raw_detections, enriched, 0)

    sorted_detections = sorted(enriched, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    removed = 0

    for candidate in sorted_detections:
        if any(_is_duplicate(candidate, existing, cleanup_config) for existing in kept):
            removed += 1
            continue
        kept.append(candidate)

    kept.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return kept, _cleanup_stats(raw_detections, kept, removed)


def summarize_video_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    raw_counts: Counter[str] = Counter()
    clean_counts: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()
    total_raw = 0
    total_clean = 0

    for record in records:
        raw_items = record.get("raw_detections", [])
        clean_items = record.get("detections", [])
        total_raw += len(raw_items)
        total_clean += len(clean_items)
        raw_counts.update(str(item.get("label", "")) for item in raw_items if item.get("label"))
        clean_counts.update(str(item.get("label", "")) for item in clean_items if item.get("label"))
        parent_counts.update(str(item.get("parent_label", "")) for item in clean_items if item.get("parent_label"))

    total_frames = len(records)
    return {
        "total_frames": total_frames,
        "frames_with_detection": sum(bool(record.get("detections")) for record in records),
        "total_detections_raw": total_raw,
        "total_detections_clean": total_clean,
        "detections_removed_as_duplicate": total_raw - total_clean,
        "raw_label_counts": dict(sorted(raw_counts.items())),
        "normalized_label_counts": dict(sorted(clean_counts.items())),
        "parent_label_counts": dict(sorted(parent_counts.items())),
        "average_detections_per_frame_raw": round(total_raw / total_frames, 4) if total_frames else 0.0,
        "average_detections_per_frame_clean": round(total_clean / total_frames, 4) if total_frames else 0.0,
    }


def _normalize_detection(det: dict[str, Any], cleanup_config: dict[str, Any]) -> dict[str, Any]:
    raw_label = str(det.get("raw_label") or det.get("label") or "").strip()
    raw_key = raw_label.lower()
    norm_cfg = cleanup_config.get("normalization", {})
    aliases = _lower_key_map(norm_cfg.get("label_aliases", {}))
    parent_labels = _lower_key_map(norm_cfg.get("parent_labels", {}))
    label = aliases.get(raw_key, raw_label)
    parent_label = parent_labels.get(label.lower()) or parent_labels.get(raw_key)

    clean = {
        "raw_label": raw_label,
        "label": label,
        "parent_label": parent_label,
        "confidence": round(float(det.get("confidence", 0.0)), 4),
        "bbox": [int(round(float(value))) for value in det.get("bbox", [])[:4]],
    }
    return clean


def _is_duplicate(candidate: dict[str, Any], existing: dict[str, Any], cleanup_config: dict[str, Any]) -> bool:
    dedup_cfg = cleanup_config.get("deduplication", {})
    iou = _bbox_iou(candidate["bbox"], existing["bbox"])
    containment = _bbox_containment(candidate["bbox"], existing["bbox"])

    if candidate["label"].lower() == existing["label"].lower():
        return iou >= float(dedup_cfg.get("same_label_iou_threshold", 0.78)) or containment >= float(
            dedup_cfg.get("containment_threshold", 0.92)
        )

    if _same_dedup_group(candidate, existing, cleanup_config):
        return iou >= float(dedup_cfg.get("synonym_iou_threshold", 0.82)) or containment >= float(
            dedup_cfg.get("containment_threshold", 0.92)
        )

    if dedup_cfg.get("compare_parent_labels", True) and candidate.get("parent_label") and candidate.get("parent_label") == existing.get("parent_label"):
        return iou >= float(dedup_cfg.get("synonym_iou_threshold", 0.82))

    return False


def _same_dedup_group(candidate: dict[str, Any], existing: dict[str, Any], cleanup_config: dict[str, Any]) -> bool:
    groups = cleanup_config.get("normalization", {}).get("dedup_groups", {}) or {}
    candidate_labels = {str(candidate.get("raw_label", "")).lower(), str(candidate.get("label", "")).lower()}
    existing_labels = {str(existing.get("raw_label", "")).lower(), str(existing.get("label", "")).lower()}
    for labels in groups.values():
        group = {str(label).lower() for label in labels}
        if candidate_labels & group and existing_labels & group:
            return True
    return False


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def _bbox_containment(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    smaller = min(area_a, area_b)
    return inter_area / smaller if smaller else 0.0


def _cleanup_stats(raw_detections: list[dict[str, Any]], clean_detections_list: list[dict[str, Any]], removed: int) -> dict[str, Any]:
    return {
        "total_detections_raw": len(raw_detections),
        "total_detections_clean": len(clean_detections_list),
        "detections_removed_as_duplicate": removed,
    }


def _dedupe_keep_order(values) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _lower_key_map(values: dict[str, Any]) -> dict[str, Any]:
    return {str(key).lower(): value for key, value in values.items()}
