from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ocr_video_v4 import (
    load_keyframes,
    read_frame,
    read_jsonl,
    resolve_path,
    run_paddle_stage,
    setup_e_drive_cache,
    video_metadata,
)
from src.preprocessing.keyframe_ocr import preprocess_crop_image
from src.preprocessing.ocr_video_v5 import (
    ROLE_CLOCK,
    ROLE_HEADLINE,
    ROLE_LOGO,
    ROLE_SCENE,
    ROLE_SCOREBOARD,
    ROLE_TICKER,
    build_ocr_corpus,
    build_raw_observations,
    build_temporal_tracks,
    detect_headline_boundaries,
    merge_horizontal_observations,
    normalize_for_match,
    normalize_light,
    searchable_fields,
    text_similarity,
    tracks_to_records,
    write_jsonl,
)
from src.retrieval.ocr_v5_index import OCRV5LexicalSearcher


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dataframe_to_csv_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_vietocr_primary{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def audit_v4() -> dict[str, Any]:
    return {
        "paddle_currently_performs": "detection_and_recognition_via_rec_polys_rec_texts_rec_scores",
        "observation_fields": [
            "bbox",
            "timestamp/frame_idx",
            "region_type",
            "det_conf/rec_conf/confidence",
            "paddle_text/text",
            "crop_path_when_save_crops",
        ],
        "vietocr_call": "V4 applies VietOCR after temporal merge on representative segment crops, then may replace text_consensus.",
        "vietocr_confidence": "V4 requests return_prob=True and stores a probability when available; V5 keeps unavailable values as null and does not fabricate confidence.",
        "temporal_merge": "V4 groups same-line detections, then clusters by region_type, max_gap_seconds, bbox IoU, text similarity, and special logo/clock handling.",
        "quality_filter": "V4 removes empty/too-short/noise/numeric-garbage/latin-hallucination/single-char/no-semantic logo-clock from semantic search.",
        "e5_text_field": "V4 embeds mapped_df['text'], derived from text_consensus after optional VietOCR replacement.",
        "keyframe_mapping": "V4 maps OCR segments to Keyframe V2 interval candidates, falling back to nearest keyframe.",
    }


def map_tracks_to_keyframes(tracks: list[dict[str, Any]], keyframes: pd.DataFrame) -> list[dict[str, Any]]:
    if keyframes.empty:
        return tracks
    out = []
    for track in tracks:
        start = float(track["start_time"])
        end = float(track["end_time"])
        rep = (start + end) / 2.0
        in_interval = keyframes[(keyframes["keyframe_time"].astype(float) >= start) & (keyframes["keyframe_time"].astype(float) <= end)]
        candidates = in_interval if not in_interval.empty else keyframes
        distances = (candidates["keyframe_time"].astype(float) - rep).abs()
        mapped = candidates.loc[distances.idxmin()]
        row = dict(track)
        row["representative_time"] = round(rep, 3)
        row["mapped_global_id"] = int(mapped["mapped_global_id"])
        row["mapped_frame_id"] = int(mapped["mapped_frame_id"])
        row["mapped_frame_idx"] = int(mapped["mapped_frame_idx"])
        row["mapped_keyframe_name"] = str(mapped["mapped_keyframe_name"])
        row["mapped_keyframe_path"] = str(mapped["mapped_keyframe_path"])
        row["mapped_keyframe_timestamp"] = float(mapped["keyframe_time"])
        row["distance_to_keyframe"] = round(abs(float(mapped["keyframe_time"]) - rep), 4)
        row["mapping_case"] = "interval" if not in_interval.empty else "nearest"
        out.append(row)
    return out


def init_vietocr(cfg: dict[str, Any], device: str) -> tuple[Any | None, str, str]:
    if not bool(cfg.get("ocr", {}).get("use_vietocr_if_available", True)):
        return None, "disabled", "disabled_by_config"
    try:
        from scripts.run_ocr_video_v4 import init_vietocr as init_v4_vietocr

        recognizer, recognizer_device = init_v4_vietocr(cfg, device)
        return recognizer, recognizer_device, "ready"
    except Exception as exc:
        return None, "unavailable", repr(exc)


def init_easyocr_detector(cfg: dict[str, Any], device: str) -> tuple[Any | None, str]:
    v2_cfg = cfg.get("ocr", {}).get("v2_primary", {})
    if not bool(v2_cfg.get("use_easyocr_detector", True)):
        return None, "disabled_by_config"
    try:
        import easyocr

        use_gpu = device != "cpu" and torch.cuda.is_available()
        languages = list(v2_cfg.get("easyocr_languages", ["vi", "en"]))
        return easyocr.Reader(languages, gpu=use_gpu), "ready"
    except Exception as exc:
        return None, repr(exc)


def track_should_use_v2_primary(track: list[dict[str, Any]], cfg: dict[str, Any]) -> bool:
    v2_cfg = cfg.get("ocr", {}).get("v2_primary", {})
    if not bool(v2_cfg.get("enabled", False)):
        return False
    if not track:
        return False
    first = track[len(track) // 2]
    bbox = [int(v) for v in first.get("bbox", [0, 0, 0, 0])]
    width = max(1, int(first.get("source_width") or 1))
    height = max(1, int(first.get("source_height") or 1))
    x1_n = bbox[0] / width
    y1_n = bbox[1] / height
    w_n = max(0.0, (bbox[2] - bbox[0]) / width)
    h_n = max(0.0, (bbox[3] - bbox[1]) / height)
    hint = str(first.get("region_hint", ""))

    if hint in set(v2_cfg.get("skip_region_hints", ["logo_channel", "clock_time"])):
        return False
    if y1_n < float(v2_cfg.get("skip_top_y_max", 0.28)) and w_n < float(v2_cfg.get("skip_top_width_max", 0.35)):
        return False
    if w_n < float(v2_cfg.get("min_width_norm", 0.08)) or h_n < float(v2_cfg.get("min_height_norm", 0.015)):
        return False
    preferred_hints = set(v2_cfg.get("region_hints", ["headline", "ticker", "scene_text"]))
    return hint in preferred_hints or w_n >= float(v2_cfg.get("fallback_min_width_norm", 0.20))


def easyocr_boxes_on_crop(detector: Any, crop: Image.Image, min_confidence: float) -> list[dict[str, Any]]:
    if detector is None:
        return []
    import cv2

    arr = np.array(crop.convert("RGB"))
    detections = []
    for bbox, text, conf in detector.readtext(arr):
        if float(conf) < min_confidence:
            continue
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        if w <= 4 or h <= 4:
            continue
        detections.append({
            "bbox": [max(0, x), max(0, y), min(crop.width, x + w), min(crop.height, y + h)],
            "text": normalize_light(str(text)),
            "confidence": float(conf),
        })
    return sorted(detections, key=lambda d: (d["bbox"][1], d["bbox"][0]))


def predict_vietocr_text(recognizer: Any, image: Image.Image) -> tuple[str, float | None]:
    try:
        text, prob = recognizer.predict(image, return_prob=True)
    except TypeError:
        text = recognizer.predict(image)
        prob = None
    return normalize_light(str(text)), float(prob) if prob is not None else None


def recognize_v2_primary_crop(
    recognizer: Any,
    detector: Any,
    crop: Image.Image,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    v2_cfg = cfg.get("ocr", {}).get("v2_primary", {})
    min_easy_conf = float(v2_cfg.get("min_easyocr_confidence", 0.25))
    scale = float(v2_cfg.get("scale_factor", 2.5))
    pad = int(v2_cfg.get("pad_px", 6))
    max_parts = int(v2_cfg.get("max_detected_parts", 6))

    boxes = easyocr_boxes_on_crop(detector, crop, min_easy_conf)[:max_parts]
    part_rows = []
    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            part = crop.crop((x1, y1, x2, y2))
            processed = preprocess_crop_image(part, scale_factor=scale, pad_px=pad)
            text, prob = predict_vietocr_text(recognizer, processed)
            if text:
                part_rows.append({
                    "text": text,
                    "confidence": prob,
                    "easyocr_text": box["text"],
                    "easyocr_confidence": round(float(box["confidence"]), 4),
                    "bbox": box["bbox"],
                })
    if not part_rows:
        processed = preprocess_crop_image(crop, scale_factor=scale, pad_px=pad)
        text, prob = predict_vietocr_text(recognizer, processed)
        if text:
            part_rows.append({
                "text": text,
                "confidence": prob,
                "easyocr_text": "",
                "easyocr_confidence": None,
                "bbox": [0, 0, crop.width, crop.height],
            })

    return {
        "text": normalize_light(" ".join(row["text"] for row in part_rows)),
        "confidence": max((row["confidence"] for row in part_rows if row["confidence"] is not None), default=None),
        "parts": part_rows,
        "detected_parts": len(boxes),
    }


def apply_vietocr_representative(
    tracks: list[list[dict[str, Any]]],
    video_path: Path,
    cfg: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    if str(cfg.get("ocr", {}).get("vietocr_scope", "representative_track")) != "representative_track":
        return {"enabled": False, "reason": "scope_not_representative_track", "crops": 0}
    recognizer, recognizer_device, status = init_vietocr(cfg, device)
    if recognizer is None:
        return {"enabled": False, "device": recognizer_device, "status": status, "crops": 0}
    v2_mode = str(cfg.get("ocr", {}).get("recognition_priority", "")) == "v2_vietocr_primary"
    easyocr_detector = None
    easyocr_status = "not_requested"
    if v2_mode:
        easyocr_detector, easyocr_status = init_easyocr_detector(cfg, device)

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"enabled": False, "device": recognizer_device, "status": "video_open_failed", "crops": 0}

    crops: list[tuple[dict[str, Any], Image.Image]] = []
    v2_applied = 0
    v2_fallbacks = 0
    for track in tracks:
        if not track:
            continue
        obs = track[len(track) // 2]
        frame_id = int(obs["frame_id"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in obs["bbox"]]
        h, w = frame.shape[:2]
        if v2_mode and track_should_use_v2_primary(track, cfg):
            expand = int(cfg.get("ocr", {}).get("v2_primary", {}).get("roi_expand_px", 16))
        else:
            expand = 0
        x1 -= expand
        y1 -= expand
        x2 += expand
        y2 += expand
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 - x1 <= 4 or y2 - y1 <= 4:
            continue
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_img = Image.fromarray(crop_rgb)
        if v2_mode and track_should_use_v2_primary(track, cfg):
            result = recognize_v2_primary_crop(recognizer, easyocr_detector, crop_img, cfg)
            if result["text"]:
                obs["vietocr_text"] = result["text"]
                obs["vietocr_conf"] = result["confidence"]
                obs["vietocr_source"] = "v2_easyocr_vietocr_primary" if result["detected_parts"] else "v2_whole_crop_vietocr_fallback"
                obs["v2_easyocr_parts"] = result["parts"]
                v2_applied += 1
                if not result["detected_parts"]:
                    v2_fallbacks += 1
            continue
        crops.append((obs, preprocess_crop_image(crop_img, scale_factor=2.0, pad_px=6)))
    cap.release()

    batch_size = int(cfg.get("ocr", {}).get("vietocr_batch_size", 64))
    failures = 0
    for start in tqdm(range(0, len(crops), batch_size), desc="V5 VietOCR representative"):
        batch = crops[start : start + batch_size]
        images = [item[1] for item in batch]
        try:
            texts, probs = recognizer.predict_batch(images, return_prob=True)
        except Exception:
            texts, probs = [], []
            for img in images:
                try:
                    text, prob = recognizer.predict(img, return_prob=True)
                except Exception:
                    text, prob = "", None
                    failures += 1
                texts.append(text)
                probs.append(prob)
        for (obs, _), text, prob in zip(batch, texts, probs):
            clean = normalize_light(str(text))
            if clean:
                obs["vietocr_text"] = clean
                obs["vietocr_conf"] = float(prob) if prob is not None else None

    return {
        "enabled": True,
        "device": recognizer_device,
        "status": "ok",
        "mode": "v2_vietocr_primary" if v2_mode else "vietocr_representative",
        "easyocr_status": easyocr_status,
        "crops": int(len(crops)),
        "v2_primary_crops": int(v2_applied),
        "v2_whole_crop_fallbacks": int(v2_fallbacks),
        "failures": int(failures),
    }


def needs_qwen(track: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if not bool(cfg.get("qwen", {}).get("enabled", False)):
        return False
    threshold = float(cfg.get("qwen", {}).get("reliability_threshold", 0.58))
    return (
        float(track.get("reliability_score", 0.0)) < threshold
        or bool(track.get("numeric_conflict", False))
        or track.get("quality_status") == "LOW_QUALITY"
    )


def apply_qwen_corrections(tracks: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    qcfg = cfg.get("qwen", {})
    if not bool(qcfg.get("enabled", False)):
        return tracks, {"enabled": False, "invoked": 0, "changed": 0, "failed": 0}, []

    corrections = []
    invoked = changed = failed = 0
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = str(qcfg.get("model", "Qwen/Qwen3-4B"))
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, torch_dtype="auto", device_map="auto")
        model.eval()
    except Exception as exc:
        for track in tracks:
            if needs_qwen(track, cfg):
                track["qwen_status"] = "unavailable"
        return tracks, {"enabled": True, "invoked": 0, "changed": 0, "failed": 0, "load_error": repr(exc)}, corrections

    for track in tracks:
        if not needs_qwen(track, cfg):
            continue
        invoked += 1
        prompt = {
            "task": "OCR consensus correction",
            "rules": [
                "Never invent text unsupported by observations.",
                "Prefer characters repeatedly observed across frames.",
                "Preserve numbers, dates, times, scores, URLs and abbreviations.",
                "Do not paraphrase or expand abbreviations.",
                "If evidence is insufficient, keep the strongest OCR candidate unchanged.",
                "Return valid JSON only.",
            ],
            "canonical_candidate": track.get("canonical_text", ""),
            "paddle_candidates": track.get("paddle_candidates", []),
            "vietocr_candidates": track.get("vietocr_candidates", []),
            "role": track.get("role", ""),
        }
        try:
            messages = [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=int(qcfg.get("max_new_tokens", 128)),
                    temperature=float(qcfg.get("temperature", 0.0)),
                    do_sample=False,
                )
            decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            match = re.search(r"\{.*\}", decoded, flags=re.S)
            payload = json.loads(match.group(0) if match else decoded)
            proposed = normalize_light(str(payload.get("canonical_text", "")))
            old = str(track.get("canonical_text", ""))
            if proposed:
                track["canonical_text"] = proposed
                track["qwen_changed"] = proposed != old
                changed += int(proposed != old)
            track["qwen_used"] = True
            track["qwen_status"] = "ok"
            track["qwen_confidence"] = payload.get("confidence")
            track["qwen_reason"] = payload.get("reason", "")
            corrections.append({"track_id": track["track_id"], "old_text": old, "new_text": track["canonical_text"], "raw_response": payload})
        except Exception as exc:
            failed += 1
            track["qwen_used"] = True
            track["qwen_status"] = "failed"
            track["qwen_error"] = repr(exc)
            corrections.append({"track_id": track["track_id"], "status": "failed", "error": repr(exc)})
    return tracks, {"enabled": True, "invoked": invoked, "changed": changed, "failed": failed}, corrections


def build_semantic_index(corpus_df: pd.DataFrame, output_dir: Path, cfg: dict[str, Any], device_arg: str) -> dict[str, Any]:
    scfg = cfg.get("retrieval", {}).get("semantic", {})
    if not bool(scfg.get("enabled", True)):
        return {"built": False, "reason": "disabled"}
    keep = corpus_df[
        (corpus_df["quality_status"].astype(str) == "KEEP")
        & (~corpus_df["role"].astype(str).isin([ROLE_LOGO, ROLE_CLOCK]))
        & (corpus_df["text"].astype(str).str.strip() != "")
    ].copy()
    if keep.empty:
        return {"built": False, "reason": "empty_corpus"}
    try:
        index_dir = output_dir / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        model_name = str(scfg.get("model_name", "intfloat/multilingual-e5-small"))
        batch_size = int(scfg.get("batch_size", 32))
        device = "cuda:0" if device_arg != "cpu" and torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        model.eval()
        texts = [f"passage: {t}" for t in keep["text"].astype(str).tolist()]
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Build OCR V5 E5 index"):
                inputs = tokenizer(texts[i : i + batch_size], return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy().astype("float32")
                embeddings.append(emb)
        arr = np.vstack(embeddings).astype("float32")
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        faiss_path = index_dir / "ocr_video_v5_flat_ip.faiss"
        map_path = index_dir / "ocr_video_v5_index_map.parquet"
        faiss.write_index(index, str(faiss_path))
        keep[["track_id", "video_id", "role", "mapped_global_id", "mapped_frame_id", "mapped_frame_idx"]].to_parquet(map_path, index=False)
        return {
            "built": True,
            "faiss_path": str(faiss_path),
            "index_map_path": str(map_path),
            "vectors": int(index.ntotal),
            "embedding_dim": int(index.d),
            "model_name": model_name,
            "device": device,
        }
    except Exception as exc:
        return {"built": False, "error": repr(exc)}


def make_debug_images(video_path: Path, tracks: list[dict[str, Any]], output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    import cv2

    debug_dir = output_dir / "debug" / "role_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        ROLE_HEADLINE: "red",
        ROLE_TICKER: "orange",
        ROLE_SCENE: "lime",
        ROLE_LOGO: "cyan",
        ROLE_CLOCK: "yellow",
        ROLE_SCOREBOARD: "magenta",
    }
    max_tracks = int(cfg.get("debug", {}).get("max_tracks", 120))
    made = []
    for track in tracks[:max_tracks]:
        frame = read_frame(video_path, int(track.get("start_frame", 0)))
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        bbox = [int(v) for v in track.get("bbox_mean", [0, 0, 0, 0])]
        role = str(track.get("role", "OTHER"))
        color = colors.get(role, "white")
        draw.rectangle(bbox, outline=color, width=3)
        label = f"{role} {track.get('reliability_score', 0):.2f}"
        draw.rectangle([bbox[0], max(0, bbox[1] - 20), bbox[0] + max(120, len(label) * 8), bbox[1]], fill=color)
        draw.text((bbox[0] + 4, max(0, bbox[1] - 18)), label, fill="black")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(track["track_id"]))
        path = debug_dir / f"{safe}.jpg"
        img.save(path, quality=88)
        made.append(path)
    sheet_path = output_dir / "debug" / "ocr_v5_roles_contact_sheet.jpg"
    if made:
        thumbs = []
        tw = int(cfg.get("debug", {}).get("thumb_width", 480))
        th = int(cfg.get("debug", {}).get("thumb_height", 270))
        for path in made:
            thumbs.append(Image.open(path).convert("RGB").resize((tw, th)))
        cols = max(1, int(cfg.get("debug", {}).get("contact_sheet_cols", 2)))
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * th), "white")
        for idx, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((idx % cols) * tw, (idx // cols) * th))
        sheet.save(sheet_path, quality=90)
    return {"images": len(made), "contact_sheet": str(sheet_path) if made else ""}


def summarize_manual_cases(tracks: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    cases = []
    desired_roles = [ROLE_HEADLINE, ROLE_TICKER, ROLE_SCENE, ROLE_LOGO, ROLE_CLOCK, ROLE_SCOREBOARD]
    for role in desired_roles:
        for track in tracks:
            if track.get("role") == role:
                cases.append(track)
                break
    for track in sorted(tracks, key=lambda t: (t.get("numeric_conflict") is not True, -float(t.get("reliability_score", 0.0)))):
        if track not in cases:
            cases.append(track)
        if len(cases) >= limit:
            break
    out = []
    for track in cases[:limit]:
        out.append({
            "timestamp": track.get("start_time"),
            "crop_path": "",
            "paddle": track.get("paddle_candidates", [])[:5],
            "vietocr": track.get("vietocr_candidates", [])[:5],
            "temporal_consensus": track.get("temporal_consensus_text", ""),
            "qwen": track.get("qwen_status", "not_requested"),
            "final": track.get("canonical_text", ""),
            "role": track.get("role"),
            "role_confidence": track.get("role_confidence"),
            "reliability": track.get("reliability_score"),
            "quality_status": track.get("quality_status"),
        })
    return out


def role_counts(tracks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for track in tracks:
        role = str(track.get("role", "OTHER"))
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


def quality_counts(tracks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for track in tracks:
        status = str(track.get("quality_status", ""))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def retrieval_demo(corpus_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    searcher = OCRV5LexicalSearcher(corpus_df)
    return {
        "Genki Plaza": searcher.search_ocr("Genki Plaza", top_k=5, roles=[ROLE_SCENE]),
        "mặt bằng nhà phố ế ẩm": searcher.search_ocr("mặt bằng nhà phố ế ẩm", top_k=5, roles=[ROLE_HEADLINE]),
        "mat bang nha pho e am": searcher.search_ocr("mat bang nha pho e am", top_k=5, roles=[ROLE_HEADLINE]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured OCR Video V5 on exactly one test video.")
    parser.add_argument("--video-id", default="L21_V002")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ocr_video_v5.yaml"))
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-paddle", action="store_true")
    parser.add_argument("--skip-semantic-index", action="store_true")
    args = parser.parse_args()

    if args.video_id != "L21_V002":
        print(f"[WARN] OCR V5 task is scoped to L21_V002; requested {args.video_id}. Continuing with explicit single-video request.")

    started = time.time()
    setup_e_drive_cache()
    cfg = load_config(args.config)
    if args.skip_semantic_index:
        cfg.setdefault("retrieval", {}).setdefault("semantic", {})["enabled"] = False
    video_id = args.video_id.strip()
    sparse_fps = float(cfg["sampling"].get("sparse_fps", 2.0))
    dense_fps = float(cfg["sampling"].get("dense_fps", 0.0))
    dense_window = float(cfg["sampling"].get("dense_window_sec", 0.0))
    output_base = resolve_path(args.output_dir or cfg["paths"].get("output_dir", "outputs/ocr_video_v5_test"))
    output_dir = output_base / video_id
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"Summary already exists: {summary_path}. Use --force to overwrite V5 experimental output.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "debug").mkdir(parents=True, exist_ok=True)

    video_path = resolve_path(cfg["paths"].get("video_root", "datasets_L21/Videos_L21_a/video")) / f"{video_id}.mp4"
    meta = video_metadata(video_path)
    keyframes = load_keyframes(video_id, cfg)
    audit = audit_v4()

    paddle_summary = run_paddle_stage(
        video_id=video_id,
        cfg=cfg,
        config_path=args.config,
        output_base=output_base,
        sparse_fps=sparse_fps,
        dense_fps=dense_fps,
        dense_window=dense_window,
        force=args.force_paddle,
    )
    frame_records = read_jsonl(output_dir / "paddle_frame_records.jsonl")
    if not frame_records:
        raise RuntimeError(f"No PaddleOCR frame records found for {video_id}")

    raw_observations = build_raw_observations(frame_records, meta)
    write_jsonl(output_dir / "raw_observations.jsonl", raw_observations)
    pd.DataFrame(raw_observations).to_parquet(output_dir / "raw_observations.parquet", index=False)

    line_observations = merge_horizontal_observations(raw_observations)
    tracks = build_temporal_tracks(line_observations, cfg.get("consensus", {}).get("temporal", {}))
    vietocr_meta = apply_vietocr_representative(tracks, video_path, cfg, args.device)
    track_records = tracks_to_records(tracks, cfg, video_id)
    track_records = map_tracks_to_keyframes(track_records, keyframes)
    track_records, qwen_meta, qwen_rows = apply_qwen_corrections(track_records, cfg)

    for track in track_records:
        track.update(searchable_fields(track))
    headline_tracks = [t for t in track_records if t.get("role") == ROLE_HEADLINE]
    boundary_rows = detect_headline_boundaries(track_records, cfg.get("topic_boundary", {}))
    corpus_rows = build_ocr_corpus(track_records, cfg.get("retrieval", {}).get("role_weights", {}))
    corpus_df = pd.DataFrame(corpus_rows)

    write_jsonl(output_dir / "ocr_tracks.jsonl", track_records)
    write_jsonl(output_dir / "ocr_consensus.jsonl", track_records)
    write_jsonl(output_dir / "ocr_cleaned.jsonl", [t for t in track_records if t.get("quality_status") != "REJECT"])
    write_jsonl(output_dir / "ocr_roles.jsonl", track_records)
    write_jsonl(output_dir / "headline_tracks.jsonl", headline_tracks)
    write_jsonl(output_dir / "topic_boundary_candidates.jsonl", boundary_rows)
    write_jsonl(output_dir / "qwen_corrections.jsonl", qwen_rows)

    tracks_csv_path = dataframe_to_csv_with_fallback(
        pd.DataFrame(track_records).drop(columns=["raw_observations"], errors="ignore"),
        output_dir / "ocr_tracks.csv",
    )
    corpus_df.to_parquet(output_dir / "ocr_corpus.parquet", index=False)
    corpus_df.to_csv(output_dir / "ocr_corpus.csv", index=False, encoding="utf-8-sig")
    corpus_df[["track_id", "video_id", "role", "mapped_global_id", "mapped_frame_id", "mapped_frame_idx", "mapped_keyframe_name"]].to_parquet(
        output_dir / "ocr_index_map.parquet",
        index=False,
    )

    quality_report = {
        "total_tracks": int(len(track_records)),
        "role_counts": role_counts(track_records),
        "quality_counts": quality_counts(track_records),
        "numeric_conflicts": int(sum(bool(t.get("numeric_conflict")) for t in track_records)),
        "paddle_vietocr_disagreement_rate": round(
            sum(
                1
                for t in track_records
                if t.get("vietocr_candidates")
                and text_similarity(" ".join(t.get("paddle_candidates", [])), " ".join(t.get("vietocr_candidates", []))) < 0.82
            )
            / max(1, sum(1 for t in track_records if t.get("vietocr_candidates"))),
            4,
        ),
    }
    (output_dir / "quality_report.json").write_text(json.dumps(quality_report, indent=2, ensure_ascii=False), encoding="utf-8")

    semantic_meta = build_semantic_index(corpus_df, output_dir, cfg, args.device)
    debug_meta = make_debug_images(video_path, track_records, output_dir, cfg)
    demo = retrieval_demo(corpus_df)
    (output_dir / "retrieval_demo.json").write_text(json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")

    boundary_debug = pd.DataFrame(boundary_rows)
    if not boundary_debug.empty:
        boundary_debug.to_csv(output_dir / "topic_boundary_debug.csv", index=False, encoding="utf-8-sig")
    else:
        (output_dir / "topic_boundary_debug.csv").write_text("", encoding="utf-8")

    manual_cases = summarize_manual_cases(track_records, limit=30)
    write_jsonl(output_dir / "manual_inspection_samples.jsonl", manual_cases)

    summary = {
        "video_id": video_id,
        "video_path": str(video_path),
        "duration_sec": round(float(meta["duration_sec"]), 3),
        "fps": float(meta["fps"]),
        "total_video_frames": int(meta["total_frames"]),
        "runtime_sec": round(time.time() - started, 2),
        "audit_findings": audit,
        "paddle_stage": paddle_summary,
        "vietocr": vietocr_meta,
        "qwen": qwen_meta,
        "semantic_index": semantic_meta,
        "debug": debug_meta,
        "metrics": {
            "total_ocr_observations": int(len(raw_observations)),
            "total_line_observations": int(len(line_observations)),
            "total_ocr_tracks": int(len(track_records)),
            "temporal_consensus_corrections": int(
                sum(
                    bool(t.get("paddle_candidates")) and str(t.get("temporal_consensus_text", "")) != str(t.get("paddle_candidates", [""])[0])
                    for t in track_records
                )
            ),
            "qwen_invoked": int(qwen_meta.get("invoked", 0)),
            "qwen_changed": int(qwen_meta.get("changed", 0)),
            "qwen_failed": int(qwen_meta.get("failed", 0)),
            "role_counts": quality_report["role_counts"],
            "quality_counts": quality_report["quality_counts"],
            "headline_track_count": int(len(headline_tracks)),
            "topic_boundary_candidate_count": int(len(boundary_rows)),
        },
        "outputs": {
            "raw_observations": str(output_dir / "raw_observations.jsonl"),
            "ocr_tracks": str(output_dir / "ocr_tracks.jsonl"),
            "ocr_tracks_csv": str(tracks_csv_path),
            "ocr_consensus": str(output_dir / "ocr_consensus.jsonl"),
            "ocr_cleaned": str(output_dir / "ocr_cleaned.jsonl"),
            "ocr_roles": str(output_dir / "ocr_roles.jsonl"),
            "headline_tracks": str(output_dir / "headline_tracks.jsonl"),
            "topic_boundary_candidates": str(output_dir / "topic_boundary_candidates.jsonl"),
            "ocr_corpus": str(output_dir / "ocr_corpus.parquet"),
            "ocr_index_map": str(output_dir / "ocr_index_map.parquet"),
            "quality_report": str(output_dir / "quality_report.json"),
            "qwen_corrections": str(output_dir / "qwen_corrections.jsonl"),
            "retrieval_demo": str(output_dir / "retrieval_demo.json"),
            "manual_inspection_samples": str(output_dir / "manual_inspection_samples.jsonl"),
            "topic_boundary_debug": str(output_dir / "topic_boundary_debug.csv"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
