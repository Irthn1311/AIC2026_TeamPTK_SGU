"""
OCR Temporal Merger & Text Normalizer for AI Challenge 2026 (OCR Branch V3)
=============================================================================
Merges consecutive keyframe OCR detections across time using:
  - Spatiotemporal proximity (Max gap seconds, BBox IoU)
  - Text similarity (Normalized Levenshtein + Token Jaccard)
  - Consensus text selection (Frequency + Confidence + Pairwise Similarity)
  - Text normalization (Raw, Search, Accent-removed Search)
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def remove_vietnamese_accents(text: str) -> str:
    """Removes Vietnamese accents and converts 'đ' -> 'd'."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return text.lower().strip()


def normalize_text_search(text: str) -> str:
    """Normalizes text for search indexing (lowercase, NFKC, clean whitespace)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s\d]", " ", text)
    text = " ".join(text.split())
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def text_similarity(s1: str, s2: str) -> float:
    s1_clean = normalize_text_search(s1)
    s2_clean = normalize_text_search(s2)
    if not s1_clean or not s2_clean:
        return 1.0 if s1_clean == s2_clean else 0.0

    max_len = max(len(s1_clean), len(s2_clean))
    lev_dist = levenshtein_distance(s1_clean, s2_clean)
    lev_sim = 1.0 - (lev_dist / max_len)

    t1 = set(s1_clean.split())
    t2 = set(s2_clean.split())
    jaccard = len(t1 & t2) / max(1, len(t1 | t2))

    return 0.6 * lev_sim + 0.4 * jaccard


def bbox_iou(b1: list[int], b2: list[int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))

    return inter_area / float(area1 + area2 - inter_area)


def compute_consensus_text(candidates: list[tuple[str, float]]) -> str:
    """Selects the best consensus text from a list of candidate (text, confidence) pairs."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0][0]

    unique_texts: dict[str, list[float]] = {}
    for txt, conf in candidates:
        txt_str = txt.strip()
        if not txt_str:
            continue
        unique_texts.setdefault(txt_str, []).append(conf)

    if not unique_texts:
        return candidates[0][0]

    best_text = ""
    best_score = -1.0

    # Sort texts by frequency descending, take top candidates if list is large
    texts_list = sorted(unique_texts.keys(), key=lambda t: (len(unique_texts[t]), np.mean(unique_texts[t])), reverse=True)
    top_candidates = texts_list[:8]

    for t in top_candidates:
        confs = unique_texts[t]
        freq = len(confs)
        mean_conf = sum(confs) / freq

        sim_sum = 0.0
        for t_other in top_candidates:
            if t != t_other:
                sim_sum += text_similarity(t, t_other)

        # Consensus score formula
        score = (freq * 2.0) + (mean_conf * 1.5) + (sim_sum * 0.5) + (len(t) * 0.05)
        if score > best_score:
            best_score = score
            best_text = t

    return best_text


@dataclass
class OCRSegment:
    ocr_segment_id: str
    video_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    region_type: str
    bbox_mean: list[int]
    text_raw_candidates: list[str]
    text_consensus: str
    text_search: str
    text_search_no_accent: str
    mean_confidence: float
    source_frames: list[int]
    use_for_semantic_search: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_horizontal_words_in_frame(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merges words lying on the same horizontal line into continuous text phrases."""
    if not detections or len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda d: (d["bbox"][1], d["bbox"][0]))
    merged = []
    visited = [False] * len(sorted_dets)

    for i in range(len(sorted_dets)):
        if visited[i]:
            continue
        curr = sorted_dets[i]
        curr_box = list(curr["bbox"])
        curr_text = curr["text"]
        curr_conf = [curr["confidence"]]
        visited[i] = True

        for j in range(i + 1, len(sorted_dets)):
            if visited[j]:
                continue
            nxt = sorted_dets[j]
            nxt_box = nxt["bbox"]

            h_curr = max(1, curr_box[3] - curr_box[1])
            h_nxt = max(1, nxt_box[3] - nxt_box[1])
            y_overlap = max(0, min(curr_box[3], nxt_box[3]) - max(curr_box[1], nxt_box[1]))
            x_gap = nxt_box[0] - curr_box[2]

            if y_overlap / min(h_curr, h_nxt) >= 0.35 and -30 <= x_gap <= max(h_curr, h_nxt) * 2.5:
                curr_box[0] = min(curr_box[0], nxt_box[0])
                curr_box[1] = min(curr_box[1], nxt_box[1])
                curr_box[2] = max(curr_box[2], nxt_box[2])
                curr_box[3] = max(curr_box[3], nxt_box[3])
                curr_text = f"{curr_text} {nxt['text']}".strip()
                curr_conf.append(nxt["confidence"])
                visited[j] = True

        merged.append({
            "frame_idx": curr["frame_idx"],
            "timestamp_seconds": curr["timestamp_seconds"],
            "bbox": curr_box,
            "region_type": curr["region_type"],
            "text": curr_text,
            "confidence": sum(curr_conf) / len(curr_conf),
        })

    return merged


def merge_video_ocr_records(
    video_records: list[dict[str, Any]],
    max_gap_seconds: float = 3.0,
    min_bbox_iou: float = 0.30,
    min_text_similarity: float = 0.70,
    include_frame_aggregates: bool = False,
) -> list[OCRSegment]:
    """
    Merges per-frame OCR detection records into temporally continuous OCR Segments
    with horizontal line reconstruction.
    """
    if not video_records:
        return []

    video_id = video_records[0].get("video_id", "")

    # Flatten and group words in each frame into coherent lines
    flat_detections = []
    frame_aggregates = []

    for rec in video_records:
        f_idx = int(rec.get("frame_idx", 0))
        t_sec = float(rec.get("timestamp_seconds", 0.0))
        frame_dets = []
        for det in rec.get("detections", []):
            txt = str(det.get("text", "")).strip()
            if not txt:
                continue
            bbox = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                bbox_xyxy = [int(v) for v in bbox]
            elif isinstance(bbox, list) and len(bbox) >= 4 and isinstance(bbox[0], (list, tuple)):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                bbox_xyxy = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            else:
                continue

            region_type = str(det.get("region_type", "scene_text"))
            conf = float(det.get("confidence", 0.85))

            frame_dets.append({
                "frame_idx": f_idx,
                "timestamp_seconds": t_sec,
                "bbox": bbox_xyxy,
                "region_type": region_type,
                "text": txt,
                "confidence": conf,
            })

        # 1. Merge words on the same line horizontally
        merged_line_dets = _merge_horizontal_words_in_frame(frame_dets)
        flat_detections.extend(merged_line_dets)

        # 2. Add full-frame combined text aggregate if requested
        if include_frame_aggregates:
            combined_txt = str(rec.get("combined_text", "")).strip()
            if len(combined_txt.split()) >= 2:
                frame_aggregates.append({
                    "frame_idx": f_idx,
                    "timestamp_seconds": t_sec,
                    "bbox": [0, 0, 1920, 1080],
                    "region_type": "scene_text",
                    "text": combined_txt,
                    "confidence": float(rec.get("mean_confidence", 0.88)),
                })

    # Sort chronologically
    flat_detections.sort(key=lambda d: (d["frame_idx"], d["bbox"][1], d["bbox"][0]))

    # Group into temporal clusters per region_type with fast sliding active window
    all_clusters: list[list[dict[str, Any]]] = []
    active_by_region: dict[str, list[list[dict[str, Any]]]] = {}

    for det in flat_detections:
        r_type = det["region_type"]
        cur_t = float(det["timestamp_seconds"])
        active_list = active_by_region.setdefault(r_type, [])

        # Prune expired clusters from active window
        still_active = []
        for cluster in active_list:
            if cur_t - cluster[-1]["timestamp_seconds"] <= max_gap_seconds:
                still_active.append(cluster)
        active_by_region[r_type] = still_active

        matched_cluster = None
        for cluster in reversed(still_active):
            last_det = cluster[-1]
            iou = bbox_iou(det["bbox"], last_det["bbox"])
            sim = text_similarity(det["text"], last_det["text"])

            if iou >= min_bbox_iou:
                det_t = det["text"].strip().lower()
                last_t = last_det["text"].strip().lower()
                is_sub = (len(det_t) >= 4 and det_t in last_t) or (len(last_t) >= 4 and last_t in det_t)
                if sim >= 0.35 or is_sub or r_type in ("clock_time", "logo_channel"):
                    matched_cluster = cluster
                    break
            elif sim >= min_text_similarity and iou >= 0.10:
                matched_cluster = cluster
                break

        if matched_cluster is not None:
            matched_cluster.append(det)
        else:
            new_cluster = [det]
            all_clusters.append(new_cluster)
            active_by_region[r_type].append(new_cluster)

    clusters = all_clusters

    # Convert clusters into OCRSegments
    segments: list[OCRSegment] = []
    for idx, cl in enumerate(clusters, start=1):
        start_frame = cl[0]["frame_idx"]
        end_frame = cl[-1]["frame_idx"]
        start_time = cl[0]["timestamp_seconds"]
        end_time = cl[-1]["timestamp_seconds"]
        region_type = cl[0]["region_type"]

        # Mean BBox calculation
        b0 = [int(sum(d["bbox"][i] for d in cl) / len(cl)) for i in range(4)]

        candidates = [(d["text"], d["confidence"]) for d in cl]
        text_raw_candidates = list(dict.fromkeys([d["text"] for d in cl]))
        consensus = compute_consensus_text(candidates)

        text_search = normalize_text_search(consensus)
        text_search_no_accent = remove_vietnamese_accents(consensus)
        mean_conf = float(sum(d["confidence"] for d in cl) / len(cl))
        source_frames = sorted(list(set(d["frame_idx"] for d in cl)))

        # Rule 2: logo_channel and clock_time are not used for semantic search
        use_for_semantic_search = region_type not in ("logo_channel", "clock_time")

        seg_id = f"{video_id}_ocr_{idx:06d}"
        segment = OCRSegment(
            ocr_segment_id=seg_id,
            video_id=video_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=round(start_time, 2),
            end_time=round(end_time, 2),
            region_type=region_type,
            bbox_mean=b0,
            text_raw_candidates=text_raw_candidates,
            text_consensus=consensus,
            text_search=text_search,
            text_search_no_accent=text_search_no_accent,
            mean_confidence=round(mean_conf, 3),
            source_frames=source_frames,
            use_for_semantic_search=use_for_semantic_search,
        )
        segments.append(segment)

    # Append frame-level aggregate segments if requested
    if include_frame_aggregates:
        for f_agg in frame_aggregates:
            idx += 1
            seg_id = f"{video_id}_ocr_frame_{idx:06d}"
            txt = f_agg["text"]
            segment = OCRSegment(
                ocr_segment_id=seg_id,
                video_id=video_id,
                start_frame=f_agg["frame_idx"],
                end_frame=f_agg["frame_idx"],
                start_time=round(f_agg["timestamp_seconds"], 2),
                end_time=round(f_agg["timestamp_seconds"] + 2.0, 2),
                region_type="scene_text",
                bbox_mean=f_agg["bbox"],
                text_raw_candidates=[txt],
                text_consensus=txt,
                text_search=normalize_text_search(txt),
                text_search_no_accent=remove_vietnamese_accents(txt),
                mean_confidence=round(f_agg["confidence"], 3),
                source_frames=[f_agg["frame_idx"]],
                use_for_semantic_search=True,
            )
            segments.append(segment)

    return segments
