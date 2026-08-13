from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.correct_ocr_v2_with_qwen_context import load_qwen, parse_qwen_json
from src.preprocessing.ocr_qwen_validator_v3 import validate_qwen_correction
from src.preprocessing.ocr_temporal_tracker_v3 import (
    OCRTrack,
    build_documents,
    build_temporal_tracks,
    build_track_records,
    flatten_v2_records,
    normalize_light,
    should_send_to_qwen,
)


QWEN_TRACK_SYSTEM_PROMPT = """Bạn là bộ sửa lỗi OCR track tiếng Việt.

Bạn chỉ được sửa dựa trên temporal evidence của cùng một OCR track.

Luật:
- Chỉ được trả action: keep, correct, needs_review.
- Không được xóa raw OCR.
- Không thêm thông tin mới không có trong raw variants/detections.
- Giữ nguyên số, ngày, giờ, phần trăm, tỉ số, số điện thoại, mã sản phẩm nếu không có bằng chứng rất rõ.
- Nếu không chắc, action = needs_review và corrected_text = consensus_text hoặc gần consensus_text.
- Trả về JSON hợp lệ duy nhất với đúng keys: action, corrected_text, confidence, reason.
- Không lặp lại dữ liệu đầu vào."""


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_safe(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def tracks_debug_rows(tracks: list[OCRTrack]) -> list[dict[str, Any]]:
    rows = []
    for t in tracks:
        rows.append(
            {
                "track_id": t.track_id,
                "video_id": t.video_id,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "num_observations": len(t.detections),
                "region_type": t.region_type,
                "raw_variants": json_safe(t.raw_variants),
                "consensus_text": t.consensus_text,
                "track_confidence": t.track_confidence,
                "uncertainty": t.uncertainty,
                "qwen_used": t.qwen_used,
                "qwen_action": t.qwen_action,
                "qwen_reason": t.qwen_reason,
                "qwen_confidence": t.qwen_confidence,
                "qwen_raw_response": t.qwen_raw_response,
                "qwen_validation_status": t.qwen_validation_status,
                "corrected_text": t.corrected_text,
                "semantic_search_text": t.semantic_search_text,
                "representative_frame_id": t.representative_frame_id,
                "representative_image_path": t.representative_image_path,
                "representative_keyframe_name": t.representative_keyframe_name,
                "solution_type": t.solution_type,
                "local_rule_applied": t.local_rule_applied,
                "local_rule_reasons": json_safe(t.local_rule_reasons),
                "needs_review": t.needs_review,
                "consensus_support": t.consensus_support,
                "variant_disagreement": t.variant_disagreement,
                "mean_ocr_confidence": t.mean_ocr_confidence,
                "temporal_support": t.temporal_support,
                "string_agreement": t.string_agreement,
            }
        )
    return rows


def count_track_merges(tracks: list[OCRTrack]) -> tuple[int, int]:
    same_shot = 0
    cross_shot = 0
    for track in tracks:
        detections = sorted(track.detections, key=lambda d: (d.get("timestamp_seconds", 0.0), d.get("frame_idx", 0)))
        for prev, cur in zip(detections, detections[1:]):
            if prev.get("shot_id") == cur.get("shot_id") and int(prev.get("shot_id", -1) or -1) >= 0:
                same_shot += 1
            else:
                cross_shot += 1
    return same_shot, cross_shot


def compute_track_metrics(tracks: list[OCRTrack], num_observations: int) -> dict[str, Any]:
    lengths = [len(t.detections) for t in tracks]
    durations = [max(0.0, float(t.end_time) - float(t.start_time)) for t in tracks]
    same_shot_merges, cross_shot_merges = count_track_merges(tracks)
    return {
        "singleton_tracks": sum(1 for n in lengths if n == 1),
        "tracks_len_2": sum(1 for n in lengths if n == 2),
        "tracks_len_3_plus": sum(1 for n in lengths if n >= 3),
        "tracks_len_5_plus": sum(1 for n in lengths if n >= 5),
        "mean_detections_per_track": round(statistics.fmean(lengths), 4) if lengths else 0.0,
        "median_detections_per_track": round(float(statistics.median(lengths)), 4) if lengths else 0.0,
        "max_detections_per_track": max(lengths) if lengths else 0,
        "mean_track_duration": round(statistics.fmean(durations), 4) if durations else 0.0,
        "median_track_duration": round(float(statistics.median(durations)), 4) if durations else 0.0,
        "same_shot_merges": same_shot_merges,
        "cross_shot_merges": cross_shot_merges,
        "solved_by_singleton_confidence": sum(1 for t in tracks if t.solution_type == "singleton_confidence"),
        "solved_by_temporal_consensus": sum(1 for t in tracks if t.solution_type == "temporal_consensus" and len(t.detections) >= 2),
        "solved_by_local_rule": sum(1 for t in tracks if t.solution_type == "local_rule"),
        "needs_review": sum(1 for t in tracks if t.needs_review),
        "singleton_percent": round(100.0 * sum(1 for n in lengths if n == 1) / max(1, len(lengths)), 2),
        "tracks_len_2_plus": sum(1 for n in lengths if n >= 2),
        "tracks_len_3_plus_percent": round(100.0 * sum(1 for n in lengths if n >= 3) / max(1, len(lengths)), 2),
        "observations_accounted_for": num_observations,
    }


def find_nearby_similar_track(track: OCRTrack, tracks: list[OCRTrack]) -> OCRTrack | None:
    from src.preprocessing.ocr_temporal_tracker_v3 import bbox_center_distance, text_similarity

    best: tuple[float, OCRTrack] | None = None
    for other in tracks:
        if other.track_id == track.track_id or other.region_type != track.region_type:
            continue
        gap = min(abs(track.start_time - other.end_time), abs(other.start_time - track.end_time))
        if gap > 10.0:
            continue
        sim = text_similarity(track.consensus_text, other.consensus_text)
        bbox_gap = bbox_center_distance(
            track.detections[-1].get("bbox", [0, 0, 0, 0]),
            other.detections[0].get("bbox", [0, 0, 0, 0]),
        )
        score = sim - min(0.5, bbox_gap)
        if sim >= 0.62 and bbox_gap <= 0.28 and (best is None or score > best[0]):
            best = (score, other)
    return best[1] if best else None


def classify_track_for_manual_sample(track: OCRTrack, tracks: list[OCRTrack]) -> tuple[str, str]:
    cross_shot = len(track.shot_ids) > 1
    nearby = find_nearby_similar_track(track, tracks) if len(track.detections) == 1 else None
    if nearby is not None:
        return "false split", f"nearby_similar_track={nearby.track_id}"
    if cross_shot and track.variant_disagreement > 0.65 and track.consensus_support < 0.50:
        return "false merge", "cross_shot_high_disagreement_low_support"
    if track.variant_disagreement > 0.50 and track.consensus_support < 0.75:
        return "bad consensus", "high_variant_disagreement_low_support"
    return "correct track", "heuristic_high_agreement_or_singleton"


def manual_tracking_samples(tracks: list[OCRTrack], video_id: str) -> list[dict[str, Any]]:
    selected: list[OCRTrack] = []
    seen: set[str] = set()

    def add_bucket(bucket: list[OCRTrack], limit: int) -> None:
        added = 0
        for track in bucket:
            if track.track_id in seen:
                continue
            selected.append(track)
            seen.add(track.track_id)
            added += 1
            if added >= limit:
                break

    ordered = sorted(tracks, key=lambda t: (t.start_time, t.region_type, t.track_id))
    add_bucket([t for t in ordered if len(t.detections) == 1], 10)
    add_bucket([t for t in ordered if len(t.detections) == 2], 10)
    add_bucket([t for t in ordered if len(t.detections) >= 3], 10)
    suspicious = sorted(
        [t for t in ordered if len(t.shot_ids) > 1 or t.variant_disagreement >= 0.35 or t.needs_review],
        key=lambda t: (len(t.shot_ids) <= 1, -t.variant_disagreement, t.start_time),
    )
    add_bucket(suspicious, 10)

    rows = []
    for track in selected[:40]:
        classification, basis = classify_track_for_manual_sample(track, tracks)
        rows.append(
            {
                "video_id": video_id,
                "track_id": track.track_id,
                "start_time": track.start_time,
                "end_time": track.end_time,
                "duration": round(float(track.end_time) - float(track.start_time), 3),
                "num_observations": len(track.detections),
                "shot_ids": json_safe(track.shot_ids),
                "region_type": track.region_type,
                "raw_variants": json_safe(track.raw_variants),
                "consensus_text": track.consensus_text,
                "track_confidence": track.track_confidence,
                "uncertainty": track.uncertainty,
                "inspection_classification": classification,
                "classification_basis": basis,
                "representative_frame_id": track.representative_frame_id,
                "representative_image_path": track.representative_image_path,
            }
        )
    return rows


def dataframe_for_export(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].map(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].map(lambda x: json_safe(x) if isinstance(x, (list, dict)) else x)
    return df


def build_qwen_payload(track: OCRTrack) -> dict[str, Any]:
    detections = []
    for det in track.detections[:18]:
        detections.append(
            {
                "timestamp_seconds": det.get("timestamp_seconds"),
                "frame_idx": det.get("frame_idx"),
                "shot_id": det.get("shot_id"),
                "text": det.get("text", ""),
                "confidence": det.get("confidence", 0.0),
                "easyocr_text": det.get("easyocr_text", ""),
                "vietocr_text": det.get("vietocr_text", ""),
            }
        )
    return {
        "track_id": track.track_id,
        "region_type": track.region_type,
        "consensus_text": track.consensus_text,
        "raw_variants": track.raw_variants,
        "track_confidence": track.track_confidence,
        "consensus_support": track.consensus_support,
        "variant_disagreement": track.variant_disagreement,
        "detections": detections,
    }


def call_qwen_for_track(tokenizer: Any, model: Any, track: OCRTrack, max_new_tokens: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": QWEN_TRACK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Dữ liệu OCR track:\n"
                + json.dumps(build_qwen_payload(track), ensure_ascii=False)
                + '\n\nChỉ trả JSON dạng {"action":"keep|correct|needs_review","corrected_text":"...","confidence":0.0,"reason":"..."}'
            ),
        },
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.03,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
    parsed = parse_qwen_json(raw)
    parsed["_raw_response"] = raw
    return parsed


def apply_qwen_to_tracks(tracks: list[OCRTrack], cfg: dict[str, Any]) -> tuple[list[OCRTrack], dict[str, Any]]:
    qcfg = cfg.get("qwen", {})
    candidates = [t for t in tracks if should_send_to_qwen(t, cfg)]
    if not bool(qcfg.get("enabled", False)):
        for t in candidates:
            t.qwen_action = "not_enabled"
            t.needs_review = True
        return tracks, {
            "enabled": False,
            "candidate_tracks": len(candidates),
            "actual_calls": 0,
            "changed": 0,
            "rejected": 0,
            "failed": 0,
        }

    region_priority = {name: idx for idx, name in enumerate(qcfg.get("candidate_region_priority", []))}
    default_priority = len(region_priority)
    candidates = sorted(
        candidates,
        key=lambda t: (
            region_priority.get(t.region_type, default_priority),
            -len(t.detections),
            t.track_confidence,
            t.start_time,
        ),
    )
    max_tracks = int(qcfg.get("max_tracks_per_video", 12))
    selected = candidates[:max_tracks] if max_tracks >= 0 else candidates
    tokenizer, model, model_meta = load_qwen(
        model_name=str(qcfg.get("model_name", "Qwen/Qwen3-4B")),
        cache_dir=resolve_path(cfg.get("paths", {}).get("qwen_model_cache_dir", ".model_cache/qwen3_4b")),
        device=str(qcfg.get("device", "cuda")),
        use_4bit=bool(qcfg.get("use_4bit", True)),
    )

    changed = rejected = failed = 0
    for track in tqdm(selected, desc="Qwen uncertain OCR tracks"):
        try:
            result = call_qwen_for_track(tokenizer, model, track, int(qcfg.get("max_new_tokens", 96)))
            action = str(result.get("action", "needs_review")).strip()
            if action not in {"keep", "correct", "needs_review"}:
                action = "needs_review"
            proposed = normalize_light(str(result.get("corrected_text", track.consensus_text)))
            evidence = [track.consensus_text, *track.raw_variants]
            validation = validate_qwen_correction(track.consensus_text, proposed, evidence, cfg.get("validation", {}))
            track.qwen_used = True
            track.qwen_action = action
            track.qwen_reason = normalize_light(str(result.get("reason", "")))
            try:
                track.qwen_confidence = round(float(result.get("confidence")), 4)
            except (TypeError, ValueError):
                track.qwen_confidence = None
            track.qwen_raw_response = normalize_light(str(result.get("_raw_response", "")))
            track.qwen_validation_status = validation.reason
            if validation.accepted and action in {"correct", "keep"}:
                track.corrected_text = validation.proposed_text
                track.semantic_search_text = validation.proposed_text
                track.needs_review = action == "needs_review"
                changed += int(track.corrected_text != track.consensus_text)
            elif validation.accepted and action == "needs_review":
                track.corrected_text = validation.proposed_text or track.consensus_text
                track.semantic_search_text = track.corrected_text
                track.needs_review = True
            else:
                rejected += 1
                track.qwen_action = "rejected"
                track.corrected_text = track.consensus_text
                track.semantic_search_text = track.consensus_text
                track.needs_review = True
        except Exception as exc:
            failed += 1
            track.qwen_used = True
            track.qwen_action = "failed"
            track.qwen_reason = repr(exc)
            track.qwen_validation_status = "failed"
            track.needs_review = True

    return tracks, {
        "enabled": True,
        "model": model_meta,
        "candidate_tracks": len(candidates),
        "actual_calls": len(selected),
        "changed": changed,
        "rejected": rejected,
        "failed": failed,
        "limited_by_max_tracks": max(0, len(candidates) - len(selected)),
    }


def process_video(video_id: str, input_dir: Path, selected_root: Path, output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    per_video_path = input_dir / "per_video" / f"{video_id}.jsonl"
    if not per_video_path.exists():
        raise FileNotFoundError(f"Missing OCR V2 per-video file: {per_video_path}")
    records = read_jsonl(per_video_path)
    observations = flatten_v2_records(records, cfg)
    raw_detection_count = sum(len(r.get("detections", []) or []) for r in records)
    tracks_raw, tracking_debug = build_temporal_tracks(observations, cfg.get("tracking", {}), return_debug=True)
    tracks = build_track_records(tracks_raw, cfg, video_id)
    tracks, qwen_meta = apply_qwen_to_tracks(tracks, cfg)

    per_video_dir = output_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)
    track_rows = [t.to_dict() for t in tracks]
    write_jsonl(per_video_dir / f"{video_id}_tracks.jsonl", track_rows)
    pd.DataFrame(tracks_debug_rows(tracks)).to_csv(per_video_dir / f"{video_id}_tracks_debug.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tracking_debug).to_csv(per_video_dir / f"{video_id}_tracking_debug.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manual_tracking_samples(tracks, video_id)).to_csv(
        per_video_dir / f"{video_id}_manual_tracking_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = compute_track_metrics(tracks, len(observations))

    summary = {
        "video_id": video_id,
        "keyframe_records": len(records),
        "ocr_detections_raw": raw_detection_count,
        "ocr_detections_after_roi": len(observations),
        "ocr_tracks": len(tracks),
        "average_detections_per_track": round(len(observations) / max(1, len(tracks)), 4),
        "tracks_solved_by_consensus": metrics["solved_by_temporal_consensus"],
        "solved_by_singleton_confidence": metrics["solved_by_singleton_confidence"],
        "solved_by_temporal_consensus": metrics["solved_by_temporal_consensus"],
        "solved_by_local_rule": metrics["solved_by_local_rule"],
        "tracks_sent_to_qwen": int(qwen_meta.get("actual_calls", 0)),
        "qwen_candidate_tracks": int(qwen_meta.get("candidate_tracks", 0)),
        "qwen_calls_avoided": max(0, len(tracks) - int(qwen_meta.get("actual_calls", 0))),
        "needs_review": metrics["needs_review"],
        "tracking_metrics": metrics,
        "qwen": qwen_meta,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OCR Temporal V3 tracks from OCR V2 selected-keyframe outputs.")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--selected-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ocr_temporal_v3.yaml"))
    args = parser.parse_args()

    started = time.time()
    cfg = load_config(args.config)
    input_dir = resolve_path(args.input_dir or cfg.get("paths", {}).get("input_dir", "outputs/ocr_v2_selected_keyframes"))
    selected_root = resolve_path(args.selected_root or cfg.get("paths", {}).get("selected_root", "outputs/keyframe_v2_full"))
    output_dir = resolve_path(args.output_dir or cfg.get("paths", {}).get("output_dir", "outputs/ocr_temporal_v3"))
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [v.strip() for v in args.video_id if v.strip()]
    if requested:
        video_ids = requested
    else:
        video_ids = sorted(p.stem for p in (input_dir / "per_video").glob("L21_V*.jsonl"))
    if not video_ids:
        raise FileNotFoundError(f"No per-video OCR V2 files found in {input_dir / 'per_video'}")

    summaries = []
    all_tracks = []
    for video_id in video_ids:
        summary = process_video(video_id, input_dir, selected_root, output_dir, cfg)
        summaries.append(summary)
        all_tracks.extend(read_jsonl(output_dir / "per_video" / f"{video_id}_tracks.jsonl"))

    track_df = dataframe_for_export(all_tracks)
    track_df.to_csv(output_dir / "l21_ocr_tracks.csv", index=False, encoding="utf-8-sig")
    track_df.to_parquet(output_dir / "l21_ocr_tracks.parquet", index=False)

    documents = build_documents([OCRTrack(**row) for row in all_tracks], cfg.get("documents", {}))
    doc_df = pd.DataFrame(documents)
    doc_df.to_parquet(output_dir / "l21_ocr_documents.parquet", index=False)

    totals = {
        "videos": len(summaries),
        "ocr_detections": int(sum(s["ocr_detections_after_roi"] for s in summaries)),
        "ocr_tracks": int(sum(s["ocr_tracks"] for s in summaries)),
        "average_detections_per_track": round(
            sum(s["ocr_detections_after_roi"] for s in summaries) / max(1, sum(s["ocr_tracks"] for s in summaries)),
            4,
        ),
        "tracks_solved_by_consensus": int(sum(s["tracks_solved_by_consensus"] for s in summaries)),
        "solved_by_singleton_confidence": int(sum(s["solved_by_singleton_confidence"] for s in summaries)),
        "solved_by_temporal_consensus": int(sum(s["solved_by_temporal_consensus"] for s in summaries)),
        "solved_by_local_rule": int(sum(s["solved_by_local_rule"] for s in summaries)),
        "tracks_sent_to_qwen": int(sum(s["tracks_sent_to_qwen"] for s in summaries)),
        "qwen_candidate_tracks": int(sum(s["qwen_candidate_tracks"] for s in summaries)),
        "qwen_calls_avoided": int(sum(s["qwen_calls_avoided"] for s in summaries)),
        "needs_review": int(sum(s["needs_review"] for s in summaries)),
        "singleton_tracks": int(sum(s["tracking_metrics"]["singleton_tracks"] for s in summaries)),
        "tracks_len_2": int(sum(s["tracking_metrics"]["tracks_len_2"] for s in summaries)),
        "tracks_len_3_plus": int(sum(s["tracking_metrics"]["tracks_len_3_plus"] for s in summaries)),
        "tracks_len_5_plus": int(sum(s["tracking_metrics"]["tracks_len_5_plus"] for s in summaries)),
        "same_shot_merges": int(sum(s["tracking_metrics"]["same_shot_merges"] for s in summaries)),
        "cross_shot_merges": int(sum(s["tracking_metrics"]["cross_shot_merges"] for s in summaries)),
    }
    metadata = {
        "config": str(resolve_path(args.config)),
        "input_dir": str(input_dir),
        "selected_root": str(selected_root),
        "output_dir": str(output_dir),
        "runtime_sec": round(time.time() - started, 2),
        "summaries": summaries,
        "totals": totals,
        "outputs": {
            "tracks_parquet": str(output_dir / "l21_ocr_tracks.parquet"),
            "tracks_csv": str(output_dir / "l21_ocr_tracks.csv"),
            "documents_parquet": str(output_dir / "l21_ocr_documents.parquet"),
            "per_video": str(output_dir / "per_video"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"totals": totals, "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
