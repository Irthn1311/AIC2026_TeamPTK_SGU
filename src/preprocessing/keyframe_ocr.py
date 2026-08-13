from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.retrieval.logging_utils import setup_logger, stage_summary, timestamp_token
from src.retrieval.mapping_loader import load_keyframe_mapping
from src.preprocessing.model_assets import ensure_vietocr_weights


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = " ".join(text.split())
    return text


def _sort_detections(items):
    def key(it):
        bbox = it.get("bbox") or it.get("box") or []
        if not bbox:
            return (0, 0)
        if isinstance(bbox, list) and len(bbox) >= 4 and all(isinstance(v, (int, float)) for v in bbox[:4]):
            x1, y1, x2, y2 = bbox[:4]
            return (round(min(y1, y2), 1), round(min(x1, x2), 1))
        ys = [p[1] for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
        xs = [p[0] for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not xs or not ys:
            return (0, 0)
        return (round(min(ys), 1), round(min(xs), 1))

    return sorted(items, key=key)


def _parse_paddle_result(result) -> list[dict[str, Any]]:
    detections = []
    if not result:
        return detections
    item = result[0] if isinstance(result, list) and len(result) == 1 else result
    if isinstance(item, dict) and "rec_texts" in item:
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])
        boxes = item.get("rec_boxes", None)
        if boxes is None or len(boxes) == 0:
            boxes = item.get("dt_polys", [])
        for i, text in enumerate(texts):
            conf = float(scores[i]) if i < len(scores) else 0.0
            bbox = boxes[i].tolist() if i < len(boxes) and hasattr(boxes[i], "tolist") else boxes[i] if i < len(boxes) else []
            detections.append(
                {
                    "text": _clean_text(str(text)),
                    "confidence": conf,
                    "bbox": bbox,
                    "reading_order": 0,
                }
            )
    else:
        lines = item if isinstance(item, list) else result
        for line in lines or []:
            if not line:
                continue
            if isinstance(line, tuple) and len(line) >= 2:
                bbox, (text, conf) = line[0], line[1]
            elif isinstance(line, list) and len(line) >= 2:
                bbox, info = line[0], line[1]
                if isinstance(info, (list, tuple)) and len(info) >= 2:
                    text, conf = info[0], info[1]
                else:
                    text, conf = str(info), 0.0
            elif isinstance(line, dict):
                text = line.get("text", "")
                conf = line.get("confidence", 0.0)
                bbox = line.get("bbox", [])
            else:
                continue
            detections.append(
                {
                    "text": _clean_text(str(text)),
                    "confidence": float(conf),
                    "bbox": bbox,
                    "reading_order": 0,
                }
            )
    detections = [d for d in detections if d["text"]]
    detections = _sort_detections(detections)
    for i, det in enumerate(detections, start=1):
        det["reading_order"] = i
    return detections


def classify_roi(bbox_xyxy: list[int], img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    x1_n = x1 / max(1, img_w)
    y1_n = y1 / max(1, img_h)
    x2_n = x2 / max(1, img_w)
    w_n = (x2 - x1) / max(1, img_w)
    h_n = (y2 - y1) / max(1, img_h)

    if y1_n < 0.25 and (x1_n < 0.30 or x2_n > 0.70) and h_n < 0.15 and w_n < 0.35:
        return "logo_channel"
    if y1_n < 0.30 and h_n < 0.10 and w_n < 0.25:
        return "clock_time"
    if y1_n >= 0.82 and w_n >= 0.30:
        return "ticker"
    if 0.45 <= y1_n <= 0.85 and w_n >= 0.20:
        return "headline"
    return "scene_text"



def preprocess_crop_image(crop_pil: Image.Image, scale_factor: float = 2.0, pad_px: int = 4) -> Image.Image:
    w, h = crop_pil.size
    if w <= 0 or h <= 0:
        return crop_pil
    if h < 32:
        scale_factor = max(scale_factor, 36.0 / max(1, h))
    padded = Image.new("RGB", (w + 2 * pad_px, h + 2 * pad_px), color=(255, 255, 255))
    padded.paste(crop_pil, (pad_px, pad_px))
    new_w = max(1, int(padded.width * scale_factor))
    new_h = max(1, int(padded.height * scale_factor))
    if new_w > 1200:
        ratio = 1200.0 / new_w
        new_w = 1200
        new_h = max(16, int(new_h * ratio))
    upscaled = padded.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    return upscaled


def _combine_text(detections: list[dict[str, Any]], min_confidence: float) -> tuple[str, float, int]:
    # Prioritize semantic text (headline, ticker, scene_text) over noisy logo_channel / clock_time
    kept = []
    for d in detections:
        conf = float(d.get("confidence", 0.0))
        txt = str(d.get("text", "")).strip()
        region = d.get("region_type", "scene_text")

        if not txt or conf < min_confidence:
            continue
        # Skip logo channel / clock time in combined text to prevent retrieval noise
        if region in ("logo_channel", "clock_time") and len(txt) <= 8:
            continue
        kept.append(txt)

    if not kept:
        kept = [str(d.get("text", "")).strip() for d in detections if float(d.get("confidence", 0.0)) >= min_confidence and str(d.get("text", "")).strip()]

    text = " ".join(kept)
    mean_conf = float(sum(float(d.get("confidence", 0.0)) for d in detections) / len(detections)) if detections else 0.0
    return text, mean_conf, len(kept)


def extract_keyframe_ocr(
    dataset_root: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    resume: bool = True,
    min_confidence: float = 0.35,
    max_images: int | None = None,
    video_ids: list[str] | None = None,
    logger=None,
):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    per_video_dir = output_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or setup_logger("keyframe_ocr")
    started = time.time()
    project_root = Path(__file__).resolve().parents[2]

    keyframe_root = dataset_root / "Keyframes_L21" / "keyframes"
    mapping_root = dataset_root / "map-keyframes-aic25-b1" / "map-keyframes"

    local_cache = project_root / ".ocr_cache"
    local_cache.mkdir(parents=True, exist_ok=True)
    temp_dir = local_cache / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    local_home = local_cache / "home"
    local_home.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(local_cache / "paddlex")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["MODELSCOPE_CACHE"] = str(local_cache / "modelscope")
    os.environ["MODELSCOPE_CREDENTIALS_PATH"] = str(local_home / ".modelscope" / "credentials")
    os.environ["HF_HOME"] = str(local_cache / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(local_cache / "huggingface" / "transformers")
    os.environ["HOME"] = str(local_home)
    os.environ["USERPROFILE"] = str(local_home)
    os.environ["HOMEDRIVE"] = str(local_home.drive)
    os.environ["HOMEPATH"] = str(local_home)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)

    use_vietocr = False
    vietocr_recognizer = None
    try:
        import torch
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        config = Cfg.load_config_from_name('vgg_transformer')
        weights = Path(str(config.get("weights", "")))
        if not weights.is_absolute():
            weights = project_root / weights
        if not weights.is_file():
            weights = ensure_vietocr_weights(weights)
        config["weights"] = str(weights)
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        vietocr_recognizer = Predictor(config)
        use_vietocr = True
        logger.info("VietOCR (vgg_transformer) initialized on device: %s", config['device'])
    except Exception as exc:
        logger.warning("Could not initialize VietOCR: %s", exc)

    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        ocr_engine = easyocr.Reader(['vi', 'en'], gpu=use_gpu)
        is_easyocr = True
        logger.info("Detector (EasyOCR CRAFT) initialized on GPU: %s", use_gpu)
    except Exception as exc:
        logger.warning("Falling back to PaddleOCR: %s", exc)
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(device="cpu", cpu_threads=8)
        is_easyocr = False

    requested_video_ids = {str(v).strip() for v in (video_ids or []) if str(v).strip()}
    video_dirs = sorted([p for p in keyframe_root.iterdir() if p.is_dir() and p.name.startswith("L21_V")])
    if requested_video_ids:
        video_dirs = [p for p in video_dirs if p.name in requested_video_ids]
        missing_video_ids = sorted(requested_video_ids - {p.name for p in video_dirs})
        if missing_video_ids:
            raise FileNotFoundError(f"Missing keyframe directories for video_ids: {missing_video_ids}")
    all_rows = []
    processed = skipped = errors = 0
    image_counter = 0
    for video_idx, video_dir in enumerate(video_dirs, start=1):
        video_id = video_dir.name
        mapping_path = mapping_root / f"{video_id}.csv"
        if not mapping_path.exists():
            logger.warning("Missing mapping for %s", video_id)
            continue
        per_video_path = per_video_dir / f"{video_id}.jsonl"
        cached_by_name: dict[str, dict[str, Any]] = {}
        if resume and per_video_path.exists():
            try:
                for line in per_video_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("ocr_status") == "ok":
                        cached_by_name[rec.get("keyframe_name")] = rec
            except Exception as exc:
                logger.warning("Could not read cache %s: %s", per_video_path, exc)
        mapping = load_keyframe_mapping(mapping_path, keyframe_root)
        rows = []
        uncached_items = []
        for item_idx, (_, row) in enumerate(mapping.iterrows(), start=1):
            cached = cached_by_name.get(row["keyframe_name"])
            if cached is not None:
                rows.append(cached)
                skipped += 1
                continue
            if max_images is not None and image_counter >= max_images:
                break
            image_counter += 1
            uncached_items.append((item_idx, row))

        for item_idx, row in uncached_items:
            keyframe_path = Path(row["keyframe_path"])
            record = {
                "video_id": video_id,
                "keyframe_name": row["keyframe_name"],
                "keyframe_path": str(keyframe_path),
                "frame_idx": int(row["frame_idx"]),
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "detections": [],
                "combined_text": "",
                "mean_confidence": 0.0,
                "num_text_boxes": 0,
                "ocr_status": "ok",
                "error": "",
            }
            try:
                if is_easyocr:
                    raw_res = ocr_engine.readtext(str(keyframe_path))
                    detections = []
                    img_pil = None
                    if raw_res:
                        import cv2
                        import numpy as np
                        img_pil = Image.open(keyframe_path).convert('RGB')
                        img_w, img_h = img_pil.size
                    
                    for bbox, text, conf in raw_res:
                        box_pts = [[int(pt[0]), int(pt[1])] for pt in bbox]
                        pts = np.array(box_pts, dtype=np.int32)
                        x, y, w, h = cv2.boundingRect(pts)
                        bbox_xyxy = [max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)]

                        region_type = classify_roi(bbox_xyxy, img_w, img_h)
                        final_text = str(text)
                        final_conf = float(conf)

                        if use_vietocr and img_pil is not None and w > 4 and h > 4:
                            try:
                                raw_crop = img_pil.crop((bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]))
                                enhanced_crop = preprocess_crop_image(raw_crop, scale_factor=2.5, pad_px=6)
                                vt_text = vietocr_recognizer.predict(enhanced_crop)
                                if vt_text and vt_text.strip():
                                    final_text = vt_text.strip()
                                    final_conf = max(final_conf, 0.88)
                            except Exception:
                                pass

                        detections.append({
                            "box": box_pts,
                            "bbox": bbox_xyxy,
                            "region_type": region_type,
                            "text": final_text,
                            "text_raw": final_text,
                            "confidence": final_conf,
                        })
                else:
                    result = ocr_engine.ocr(str(keyframe_path), cls=False)
                    detections = _parse_paddle_result(result)

                text, mean_conf, num_boxes = _combine_text(detections, min_confidence=min_confidence)
                record["detections"] = detections
                record["combined_text"] = text
                record["mean_confidence"] = mean_conf
                record["num_text_boxes"] = num_boxes
            except Exception as exc:
                record["ocr_status"] = "error"
                record["error"] = str(exc)
                errors += 1
            rows.append(record)
            processed += 1
            print(f" 🚀 [PyTorch GPU OCR] [{video_id}] Frame {row['frame_idx']} ({item_idx}/{len(mapping)}) -> Text: '{record['combined_text'][:40]}'", end="\r")

            if processed % 32 == 0 and rows:
                print(f" 🚀 [GPU MAX SPEED] [{video_idx}/{len(video_dirs)}] {video_id} | Processed: {processed} keyframes | Text: '{rows[-1]['combined_text'][:40]}'")
                logger.info("Video %s/%s | Processed %s | Skipped cached %s | Errors %s", video_idx, len(video_dirs), processed, skipped, errors)

        per_video_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        all_rows.extend(rows)
        print(f" ✅ [COMPLETED] Video {video_id} -> Processed {len(rows)} keyframes")
        if max_images is not None and image_counter >= max_images:
            break

    df = pd.DataFrame(all_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(output_dir / "l21_keyframe_ocr.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_csv(output_dir / "l21_keyframe_ocr.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(output_dir / "l21_keyframe_ocr.parquet", index=False)
    except Exception:
        pass
    meta = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "device": device,
        "min_confidence": min_confidence,
        "resume": resume,
        "video_ids": sorted(requested_video_ids),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "timestamp": timestamp_token(),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "ocr_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(stage_summary("keyframe_ocr", "ok" if errors == 0 else "partial", input_path=str(keyframe_root), processed=processed, skipped=skipped, errors=errors, output=str(output_dir), elapsed=time.time() - started))
    return df, meta
