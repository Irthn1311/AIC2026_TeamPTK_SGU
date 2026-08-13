from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.preprocessing.ocr_qwen_validator_v3 import normalize_light
from src.preprocessing.ocr_temporal_merger import bbox_iou, remove_vietnamese_accents, text_similarity


COMMON_CONSENSUS_FIXES: list[tuple[str, str]] = [
    (r"\bNGHỈ\s+LÊ\s+QUỐC\s+KHÁNH\b", "NGHỈ LỄ QUỐC KHÁNH"),
    (r"\bQUẢ\s+BƯỜI\b", "QUẢ BƯỞI"),
    (r"\bGIÁM\s+LẠI\s+SUẤT\b", "GIẢM LÃI SUẤT"),
    (r"\bLẦN\s+ĐẦU\s+GIÁM\s+LẠI\s+SUẤT\b", "LẦN ĐẦU GIẢM LÃI SUẤT"),
    (r"\bLẨN\s+ĐẨU\b", "LẦN ĐẦU"),
    (r"\bSÚI\s+CẢO\b", "SỦI CẢO"),
    (r"\bSÂN\s+PHẨM\b", "sản phẩm"),
]


@dataclass
class OCRObservation:
    observation_id: str
    video_id: str
    keyframe_name: str
    keyframe_path: str
    keyframe_v2_idx: int
    global_id: int
    frame_idx: int
    timestamp_seconds: float
    shot_id: int
    bbox: list[int]
    region_type: str
    text: str
    confidence: float
    easyocr_text: str = ""
    easyocr_confidence: float | None = None
    vietocr_text: str = ""
    vietocr_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRTrack:
    track_id: str
    video_id: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    region_type: str
    detections: list[dict[str, Any]]
    raw_variants: list[str]
    raw_searchable_text: str
    consensus_text: str
    corrected_text: str
    semantic_search_text: str
    track_confidence: float
    uncertainty: str
    consensus_support: float
    variant_disagreement: float
    mean_ocr_confidence: float
    temporal_support: float
    string_agreement: float
    solution_type: str = "needs_review"
    local_rule_applied: bool = False
    local_rule_reasons: list[str] = field(default_factory=list)
    qwen_used: bool = False
    qwen_action: str = "not_requested"
    qwen_reason: str = ""
    qwen_confidence: float | None = None
    qwen_raw_response: str = ""
    qwen_validation_status: str = ""
    needs_review: bool = False
    representative_frame_id: int = 0
    representative_image_path: str = ""
    representative_keyframe_name: str = ""
    representative_global_id: int = 0
    representative_time: float = 0.0
    shot_ids: list[int] = field(default_factory=list)
    source_frames: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = remove_vietnamese_accents(text.lower())
    text = re.sub(r"[^\w\s%:/.-]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w%:/.-]+", normalize_for_match(text), flags=re.UNICODE)


def is_noise_text(text: str) -> bool:
    text = normalize_light(text)
    if not text:
        return True
    if text.upper() in {"HD", "H", "HTV", "HTV9", "VTV", "VTV1", "VTV3"}:
        return True
    if text.lower() in {"giây", "giay", "gia"}:
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", text):
        return True
    nums = re.findall(r"\d+", text)
    words = re.findall(r"[A-Za-zÀ-ỹĐđ]+", text)
    if len(nums) >= 8 and len(words) <= 2:
        return True
    if len(nums) >= 16 and len(nums) >= len(words) * 1.8:
        return True
    return False


def apply_consensus_fixes(text: str, enabled: bool = True) -> tuple[str, list[str]]:
    corrected = normalize_light(text)
    if not enabled:
        return corrected, []
    reasons = []
    for pattern, replacement in COMMON_CONSENSUS_FIXES:
        new_text = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        if new_text != corrected:
            reasons.append(pattern)
            corrected = new_text
    return normalize_light(corrected), reasons


def parse_bbox(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            nums = re.findall(r"-?\d+(?:\.\d+)?", value)
            return [int(round(float(v))) for v in nums[:4]] if len(nums) >= 4 else [0, 0, 0, 0]
    if isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return [int(round(float(v))) for v in value]
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        xs = [float(p[0]) for p in value if len(p) >= 2]
        ys = [float(p[1]) for p in value if len(p) >= 2]
        if xs and ys:
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    return [0, 0, 0, 0]


def bbox_center_distance(a: list[int], b: list[int], width: int = 1280, height: int = 720) -> float:
    ax = ((a[0] + a[2]) / 2.0) / max(1, width)
    ay = ((a[1] + a[3]) / 2.0) / max(1, height)
    bx = ((b[0] + b[2]) / 2.0) / max(1, width)
    by = ((b[1] + b[3]) / 2.0) / max(1, height)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def bbox_quality(bbox: list[int], width: int = 1280, height: int = 720) -> float:
    w = max(0, bbox[2] - bbox[0]) / max(1, width)
    h = max(0, bbox[3] - bbox[1]) / max(1, height)
    if w <= 0 or h <= 0:
        return 0.0
    area = w * h
    size_score = min(1.0, area / 0.08)
    aspect_score = 1.0 if w >= h else max(0.25, w / max(h, 1e-6))
    return round((size_score * 0.65) + (aspect_score * 0.35), 4)


def region_tracking_cfg(cfg: dict[str, Any], region_type: str) -> dict[str, Any]:
    base = {k: v for k, v in cfg.items() if k != "region_overrides"}
    overrides = cfg.get("region_overrides", {}) or {}
    region_cfg = overrides.get(region_type, {}) or {}
    merged = dict(base)
    merged.update(region_cfg)
    return merged


def flatten_v2_records(records: list[dict[str, Any]], cfg: dict[str, Any]) -> list[OCRObservation]:
    roi_cfg = cfg.get("roi", {})
    keep_regions = set(roi_cfg.get("keep_region_types", ["headline", "ticker", "scene_text"]))
    drop_regions = set(roi_cfg.get("drop_region_types", ["logo_channel", "clock_time"]))
    min_conf = float(roi_cfg.get("min_confidence", 0.35))
    min_len = int(roi_cfg.get("min_text_length", 2))
    observations: list[OCRObservation] = []

    for rec in records:
        video_id = str(rec.get("video_id", ""))
        for det_idx, det in enumerate(rec.get("detections", []) or []):
            region = str(det.get("region_type", "scene_text"))
            if region in drop_regions or region not in keep_regions:
                continue
            text = normalize_light(str(det.get("text", "")))
            conf = float(det.get("confidence", 0.0) or 0.0)
            if not text or len(text) < min_len or conf < min_conf:
                continue
            if is_noise_text(text) and region != "headline":
                continue
            bbox = parse_bbox(det.get("bbox", det.get("box", [0, 0, 0, 0])))
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            observations.append(
                OCRObservation(
                    observation_id=f"{video_id}_{rec.get('keyframe_v2_idx', 0):06d}_{det_idx:03d}",
                    video_id=video_id,
                    keyframe_name=str(rec.get("keyframe_name", "")),
                    keyframe_path=str(rec.get("keyframe_path", "")),
                    keyframe_v2_idx=int(rec.get("keyframe_v2_idx", 0) or 0),
                    global_id=int(rec.get("global_id", 0) or 0),
                    frame_idx=int(rec.get("frame_idx", 0) or 0),
                    timestamp_seconds=float(rec.get("timestamp_seconds", 0.0) or 0.0),
                    shot_id=int(rec.get("shot_id", -1) or -1),
                    bbox=bbox,
                    region_type=region,
                    text=text,
                    confidence=round(conf, 4),
                    easyocr_text=normalize_light(str(det.get("easyocr_text", ""))),
                    easyocr_confidence=det.get("easyocr_confidence"),
                    vietocr_text=normalize_light(str(det.get("vietocr_text", ""))),
                    vietocr_confidence=det.get("vietocr_confidence"),
                )
            )
    return sorted(observations, key=lambda o: (o.timestamp_seconds, o.bbox[1], o.bbox[0]))


def evaluate_match(obs: OCRObservation, track: list[OCRObservation], cfg: dict[str, Any]) -> dict[str, Any]:
    last = track[-1]
    rcfg = region_tracking_cfg(cfg, obs.region_type)
    details: dict[str, Any] = {
        "matched": False,
        "reject_reason": "",
        "text_similarity": 0.0,
        "bbox_similarity": 0.0,
        "bbox_iou": 0.0,
        "center_distance": 0.0,
        "time_score": 0.0,
        "region_score": 0.0,
        "final_match_score": 0.0,
        "same_shot": False,
        "cross_shot": False,
    }
    if obs.region_type != last.region_type:
        details["reject_reason"] = "region_type_mismatch"
        return details

    gap = obs.timestamp_seconds - last.timestamp_seconds
    max_gap = float(rcfg.get("max_gap_seconds", 8.0))
    keyframe_gap = obs.keyframe_v2_idx - last.keyframe_v2_idx
    if gap < 0:
        details["reject_reason"] = "negative_time_gap"
        return details
    if gap > max_gap:
        details["reject_reason"] = "time_gap"
        details["time_score"] = 0.0
        return details
    if keyframe_gap > int(rcfg.get("max_gap_keyframes", 4)):
        details["reject_reason"] = "keyframe_gap"
        details["time_score"] = round(max(0.0, 1.0 - gap / max(max_gap, 1e-6)), 4)
        return details

    recent = track[-int(rcfg.get("recent_observations_for_match", 4)) :]
    sim = max(text_similarity(obs.text, prev.text) for prev in recent)
    iou = max(bbox_iou(obs.bbox, prev.bbox) for prev in recent)
    dist = min(bbox_center_distance(obs.bbox, prev.bbox) for prev in recent)
    same_shot = obs.shot_id == last.shot_id and obs.shot_id >= 0
    cross_shot = obs.shot_id != last.shot_id

    max_dist = float(rcfg.get("center_distance_threshold", 0.18))
    dist_score = max(0.0, 1.0 - min(1.0, dist / max_dist))
    bbox_similarity = max(iou, dist_score)
    time_score = max(0.0, 1.0 - gap / max(max_gap, 1e-6))
    region_score = 1.0
    score = (
        sim * float(rcfg.get("text_weight", 0.54))
        + bbox_similarity * float(rcfg.get("bbox_weight", 0.26))
        + time_score * float(rcfg.get("time_weight", 0.12))
        + region_score * float(rcfg.get("region_weight", 0.08))
    )
    if same_shot:
        score += float(rcfg.get("same_shot_bonus", 0.12))

    details.update(
        {
            "text_similarity": round(sim, 4),
            "bbox_similarity": round(bbox_similarity, 4),
            "bbox_iou": round(iou, 4),
            "center_distance": round(dist, 4),
            "time_score": round(time_score, 4),
            "region_score": round(region_score, 4),
            "final_match_score": round(score, 4),
            "same_shot": same_shot,
            "cross_shot": cross_shot,
        }
    )

    min_text = float(rcfg.get("min_text_similarity", 0.55))
    min_iou = float(rcfg.get("min_bbox_iou", 0.12))
    min_score = float(rcfg.get("match_score_threshold", 0.58))

    if cross_shot:
        cross_max_gap = float(rcfg.get("cross_shot_max_gap_seconds", min(3.0, max_gap)))
        cross_max_dist = float(rcfg.get("cross_shot_center_distance_threshold", max_dist * 0.8))
        min_cross_chars = int(rcfg.get("cross_shot_min_text_chars", 0))
        min_cross_tokens = int(rcfg.get("cross_shot_min_tokens", 0))
        norm_text = normalize_for_match(obs.text)
        if min_cross_chars or min_cross_tokens:
            enough_chars = len(norm_text.replace(" ", "")) >= min_cross_chars
            enough_tokens = len(tokenize(obs.text)) >= min_cross_tokens
            if not (enough_chars or enough_tokens):
                details["reject_reason"] = "cross_shot_short_text"
                return details
        if gap > cross_max_gap:
            details["reject_reason"] = "cross_shot_time_gap"
            return details
        if sim < float(rcfg.get("cross_shot_min_similarity", 0.88)):
            details["reject_reason"] = "cross_shot_text_similarity"
            return details
        if iou < min_iou and dist > cross_max_dist:
            details["reject_reason"] = "cross_shot_bbox"
            return details
        if score < float(rcfg.get("cross_shot_min_score", 0.78)):
            details["reject_reason"] = "cross_shot_score"
            return details

    ok = (
        score >= min_score
        and sim >= min_text
        and (iou >= min_iou or dist <= max_dist)
    )
    if ok:
        details["matched"] = True
        details["reject_reason"] = ""
    elif sim < min_text:
        details["reject_reason"] = "text_similarity"
    elif iou < min_iou and dist > max_dist:
        details["reject_reason"] = "bbox"
    else:
        details["reject_reason"] = "match_score"
    return details


def match_score(obs: OCRObservation, track: list[OCRObservation], cfg: dict[str, Any]) -> tuple[bool, float]:
    result = evaluate_match(obs, track, cfg)
    return bool(result["matched"]), float(result["final_match_score"])


def _tracking_debug_row(
    obs: OCRObservation,
    candidate_track: str,
    result: dict[str, Any] | None,
    matched: bool,
    reject_reason: str,
) -> dict[str, Any]:
    result = result or {}
    return {
        "frame": obs.frame_idx,
        "time": round(obs.timestamp_seconds, 3),
        "keyframe_v2_idx": obs.keyframe_v2_idx,
        "shot_id": obs.shot_id,
        "region_type": obs.region_type,
        "text": obs.text,
        "bbox": json.dumps(obs.bbox, ensure_ascii=False),
        "candidate_track": candidate_track,
        "text_similarity": result.get("text_similarity", 0.0),
        "bbox_similarity": result.get("bbox_similarity", 0.0),
        "time_score": result.get("time_score", 0.0),
        "region_score": result.get("region_score", 0.0),
        "final_match_score": result.get("final_match_score", 0.0),
        "matched": matched,
        "reject_reason": reject_reason,
    }


def build_temporal_tracks(
    observations: list[OCRObservation],
    cfg: dict[str, Any],
    return_debug: bool = False,
) -> list[list[OCRObservation]] | tuple[list[list[OCRObservation]], list[dict[str, Any]]]:
    active: dict[str, list[dict[str, Any]]] = defaultdict(list)
    finished: list[list[OCRObservation]] = []
    debug_rows: list[dict[str, Any]] = []
    next_track_num = 1

    for obs in observations:
        rcfg = region_tracking_cfg(cfg, obs.region_type)
        max_gap = float(rcfg.get("max_gap_seconds", cfg.get("max_gap_seconds", 8.0)))
        region_active: list[dict[str, Any]] = []
        for tr in active[obs.region_type]:
            observations_for_track = tr["observations"]
            if obs.timestamp_seconds - observations_for_track[-1].timestamp_seconds <= max_gap:
                region_active.append(tr)
            else:
                finished.append(observations_for_track)
        active[obs.region_type] = region_active

        best_idx = -1
        best_score = 0.0
        candidate_results: list[tuple[int, dict[str, Any]]] = []
        for idx, tr in enumerate(active[obs.region_type]):
            result = evaluate_match(obs, tr["observations"], cfg)
            candidate_results.append((idx, result))
            if result["matched"] and float(result["final_match_score"]) > best_score:
                best_idx = idx
                best_score = float(result["final_match_score"])

        if return_debug:
            if not candidate_results:
                debug_rows.append(_tracking_debug_row(obs, "", None, False, "no_active_track"))
            max_debug = int(cfg.get("debug_max_candidates_per_observation", 8))
            ranked = sorted(candidate_results, key=lambda item: float(item[1].get("final_match_score", 0.0)), reverse=True)
            for idx, result in ranked[:max_debug]:
                candidate_id = str(active[obs.region_type][idx]["temp_track_id"])
                debug_rows.append(
                    _tracking_debug_row(
                        obs,
                        candidate_id,
                        result,
                        matched=idx == best_idx,
                        reject_reason="" if idx == best_idx else str(result.get("reject_reason", "rejected")),
                    )
                )
        if best_idx >= 0:
            active[obs.region_type][best_idx]["observations"].append(obs)
        else:
            temp_track_id = f"tmp_{next_track_num:06d}"
            next_track_num += 1
            active[obs.region_type].append({"temp_track_id": temp_track_id, "observations": [obs]})

    for tracks in active.values():
        finished.extend(tr["observations"] for tr in tracks)
    finished = sorted(finished, key=lambda tr: (tr[0].timestamp_seconds, tr[0].region_type, tr[0].bbox[1], tr[0].bbox[0]))
    if return_debug:
        return finished, debug_rows
    return finished


def pairwise_agreement(texts: list[str]) -> float:
    unique = list(dict.fromkeys([t for t in texts if t]))
    if len(unique) <= 1:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            total += text_similarity(unique[i], unique[j])
            pairs += 1
    return total / max(1, pairs)


def choose_consensus(observations: list[OCRObservation], cfg: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for obs in observations:
        key = normalize_for_match(obs.text)
        if not key:
            continue
        row = grouped.setdefault(key, {"texts": [], "confidence": [], "frames": []})
        row["texts"].append(obs.text)
        row["confidence"].append(obs.confidence)
        row["frames"].append(obs.frame_idx)
    if not grouped:
        return {"text": "", "support": 0.0, "raw_variants": [], "agreement": 0.0, "fix_reasons": []}

    raw_variants = list(dict.fromkeys(obs.text for obs in observations if obs.text))
    best_key = ""
    best_score = -1.0
    for key, row in grouped.items():
        freq = len(row["texts"])
        mean_conf = sum(row["confidence"]) / max(1, len(row["confidence"]))
        sim_to_others = sum(text_similarity(row["texts"][0], other) for other in raw_variants) / max(1, len(raw_variants))
        score = (freq * 2.6) + (mean_conf * 1.4) + (sim_to_others * 0.6) + (len(key) * 0.015)
        if score > best_score:
            best_score = score
            best_key = key

    best_text_counts = Counter(grouped[best_key]["texts"])
    best_text = best_text_counts.most_common(1)[0][0]
    fixed_text, fix_reasons = apply_consensus_fixes(best_text, bool(cfg.get("local_fixes", {}).get("enabled", True)))
    support = len(grouped[best_key]["texts"]) / max(1, len(observations))
    agreement = pairwise_agreement(raw_variants)
    return {
        "text": fixed_text,
        "pre_fix_text": best_text,
        "support": round(support, 4),
        "raw_variants": raw_variants,
        "agreement": round(agreement, 4),
        "fix_reasons": fix_reasons,
    }


def representative_observation(observations: list[OCRObservation], consensus_text: str) -> OCRObservation:
    best = observations[0]
    best_score = -1.0
    for obs in observations:
        score = (
            obs.confidence * 0.45
            + text_similarity(obs.text, consensus_text) * 0.40
            + bbox_quality(obs.bbox) * 0.15
        )
        if score > best_score:
            best = obs
            best_score = score
    return best


def track_reliability(
    observations: list[OCRObservation],
    consensus: dict[str, Any],
    rep: OCRObservation,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    weights = cfg.get("reliability_weights", {})
    mean_conf = sum(o.confidence for o in observations) / max(1, len(observations))
    duration = max(0.0, observations[-1].timestamp_seconds - observations[0].timestamp_seconds)
    temporal_support = min(1.0, (len(observations) / 4.0) * 0.65 + min(duration / 8.0, 1.0) * 0.35)
    consensus_support = float(consensus.get("support", 0.0))
    string_agreement = float(consensus.get("agreement", 0.0))
    rep_quality = bbox_quality(rep.bbox)
    score = (
        float(weights.get("mean_confidence", 0.25)) * mean_conf
        + float(weights.get("consensus_support", 0.30)) * consensus_support
        + float(weights.get("temporal_support", 0.20)) * temporal_support
        + float(weights.get("string_agreement", 0.15)) * string_agreement
        + float(weights.get("representative_quality", 0.10)) * rep_quality
    )
    variant_disagreement = round(max(0.0, 1.0 - string_agreement), 4)
    min_conf = float(cfg.get("min_confident_track_confidence", 0.72))
    min_support = float(cfg.get("min_consensus_support", 0.55))
    max_disagreement = float(cfg.get("max_variant_disagreement", 0.42))
    uncertain = score < min_conf or consensus_support < min_support or variant_disagreement > max_disagreement
    return {
        "track_confidence": round(max(0.0, min(1.0, score)), 4),
        "uncertainty": "uncertain" if uncertain else "confident",
        "mean_ocr_confidence": round(mean_conf, 4),
        "temporal_support": round(temporal_support, 4),
        "variant_disagreement": variant_disagreement,
        "string_agreement": round(string_agreement, 4),
    }


def build_track_records(tracks: list[list[OCRObservation]], cfg: dict[str, Any], video_id: str) -> list[OCRTrack]:
    records: list[OCRTrack] = []
    consensus_cfg = cfg.get("consensus", {})
    for idx, observations in enumerate(tracks, start=1):
        observations = sorted(observations, key=lambda o: (o.timestamp_seconds, o.bbox[1], o.bbox[0]))
        consensus = choose_consensus(observations, consensus_cfg)
        rep = representative_observation(observations, consensus["text"])
        rel = track_reliability(observations, consensus, rep, consensus_cfg)
        raw_searchable = normalize_light(" | ".join(consensus["raw_variants"]))
        corrected = consensus["text"]
        semantic = corrected
        local_reasons = list(consensus.get("fix_reasons", []) or [])
        if local_reasons and rel["uncertainty"] == "confident":
            solution_type = "local_rule"
        elif len(observations) == 1 and rel["uncertainty"] == "confident":
            solution_type = "singleton_confidence"
        elif len(observations) >= 2 and rel["uncertainty"] == "confident":
            solution_type = "temporal_consensus"
        else:
            solution_type = "needs_review"
        records.append(
            OCRTrack(
                track_id=f"{video_id}_ocrtempv3_{idx:06d}",
                video_id=video_id,
                start_time=round(observations[0].timestamp_seconds, 3),
                end_time=round(observations[-1].timestamp_seconds, 3),
                start_frame=observations[0].frame_idx,
                end_frame=observations[-1].frame_idx,
                region_type=observations[0].region_type,
                detections=[o.to_dict() for o in observations],
                raw_variants=consensus["raw_variants"],
                raw_searchable_text=raw_searchable,
                consensus_text=consensus["text"],
                corrected_text=corrected,
                semantic_search_text=semantic,
                track_confidence=rel["track_confidence"],
                uncertainty=rel["uncertainty"],
                consensus_support=round(float(consensus["support"]), 4),
                variant_disagreement=rel["variant_disagreement"],
                mean_ocr_confidence=rel["mean_ocr_confidence"],
                temporal_support=rel["temporal_support"],
                string_agreement=rel["string_agreement"],
                solution_type=solution_type,
                local_rule_applied=bool(local_reasons),
                local_rule_reasons=local_reasons,
                needs_review=rel["uncertainty"] == "uncertain",
                representative_frame_id=rep.frame_idx,
                representative_image_path=rep.keyframe_path,
                representative_keyframe_name=rep.keyframe_name,
                representative_global_id=rep.global_id,
                representative_time=round(rep.timestamp_seconds, 3),
                shot_ids=sorted({o.shot_id for o in observations}),
                source_frames=sorted({o.frame_idx for o in observations}),
            )
        )
    return records


def should_send_to_qwen(track: OCRTrack, cfg: dict[str, Any]) -> bool:
    qcfg = cfg.get("qwen", {})
    if track.region_type not in set(qcfg.get("only_roles", ["headline", "ticker", "scene_text"])):
        return False
    if track.uncertainty != "uncertain":
        return False
    text = track.consensus_text
    if not text or is_noise_text(text):
        return False
    if len(text.split()) <= 1 and track.mean_ocr_confidence >= 0.90:
        return False
    return True


def build_documents(tracks: list[OCRTrack], cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    for track in tracks:
        bm25_text = normalize_light(f"{track.raw_searchable_text} {track.corrected_text}")
        rows.append(
            {
                "document_id": track.track_id,
                "track_id": track.track_id,
                "video_id": track.video_id,
                "start_time": track.start_time,
                "end_time": track.end_time,
                "frame_id": track.representative_frame_id,
                "keyframe_name": track.representative_keyframe_name,
                "image_path": track.representative_image_path,
                "region_type": track.region_type,
                "bm25_text": bm25_text,
                "bm25_text_no_accent": normalize_for_match(bm25_text),
                "semantic_search_text": normalize_light(track.semantic_search_text),
                "raw_searchable_text": track.raw_searchable_text,
                "consensus_text": track.consensus_text,
                "corrected_text": track.corrected_text,
                "track_confidence": track.track_confidence,
                "uncertainty": track.uncertainty,
                "needs_review": track.needs_review,
            }
        )
    return rows
