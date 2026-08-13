from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
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

from src.preprocessing.keyframe_ocr import classify_roi, preprocess_crop_image
from src.preprocessing.model_assets import ensure_vietocr_weights
from src.preprocessing.ocr_temporal_merger import (
    merge_video_ocr_records,
    normalize_text_search,
    remove_vietnamese_accents,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_e_drive_cache() -> None:
    cache_root = PROJECT_ROOT / ".ocr_cache_video_v4"
    temp_dir = cache_root / "temp"
    home_dir = cache_root / "home"
    for p in (cache_root, temp_dir, home_dir):
        p.mkdir(parents=True, exist_ok=True)

    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_root / "paddlex")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["MODELSCOPE_CACHE"] = str(cache_root / "modelscope")
    os.environ["MODELSCOPE_CREDENTIALS_PATH"] = str(home_dir / ".modelscope" / "credentials")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "huggingface" / "transformers")
    os.environ["HOME"] = str(home_dir)
    os.environ["USERPROFILE"] = str(home_dir)
    os.environ["HOMEDRIVE"] = str(home_dir.drive)
    os.environ["HOMEPATH"] = str(home_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["MPLCONFIGDIR"] = str(temp_dir / "matplotlib")
    os.environ["EASYOCR_MODULE_PATH"] = str(PROJECT_ROOT / ".ocr_cache" / "home" / ".EasyOCR")
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"


def normalize_light(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = " ".join(text.strip().split())
    return text


def bbox_to_xyxy(bbox: Any, width: int, height: int) -> list[int]:
    if isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()
    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        x1, y1, x2, y2 = bbox
    elif isinstance(bbox, list) and bbox and isinstance(bbox[0], (list, tuple)):
        xs = [float(p[0]) for p in bbox if len(p) >= 2]
        ys = [float(p[1]) for p in bbox if len(p) >= 2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        x1, y1, x2, y2 = 0, 0, 0, 0
    return [
        int(max(0, min(width - 1, round(x1)))),
        int(max(0, min(height - 1, round(y1)))),
        int(max(0, min(width, round(x2)))),
        int(max(0, min(height, round(y2)))),
    ]


def video_metadata(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps,
        "width": width,
        "height": height,
    }


def load_shots(video_id: str, cfg: dict[str, Any]) -> pd.DataFrame:
    v2_root = resolve_path(cfg["paths"].get("keyframe_v2_root", "outputs/keyframe_v2_full"))
    shot_path = v2_root / video_id / "shots.csv"
    if shot_path.exists():
        return pd.read_csv(shot_path)
    return pd.DataFrame(columns=["video_id", "shot_id", "start_frame", "end_frame"])


def shot_id_for_frame(shots: pd.DataFrame, frame_idx: int) -> int:
    if shots.empty:
        return -1
    mask = (shots["start_frame"].astype(int) <= frame_idx) & (shots["end_frame"].astype(int) >= frame_idx)
    if mask.any():
        return int(shots[mask].iloc[0]["shot_id"])
    return -1


def build_sparse_samples(video_id: str, meta: dict[str, Any], shots: pd.DataFrame, sparse_fps: float) -> pd.DataFrame:
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])
    step = max(1, int(round(fps / max(0.001, sparse_fps))))
    frames = set(range(0, total_frames, step))

    # Include shot starts/midpoints/ends so very short title cards are still represented.
    for _, row in shots.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        mid = int((start + end) // 2)
        for fid in (start, mid, end):
            if 0 <= fid < total_frames:
                frames.add(fid)

    rows = []
    for fid in sorted(frames):
        rows.append({
            "video_id": video_id,
            "frame_idx": int(fid),
            "timestamp_sec": float(fid / fps),
            "shot_id": shot_id_for_frame(shots, int(fid)),
            "sample_stage": "sparse",
        })
    return pd.DataFrame(rows)


def dense_samples_from_hits(
    sparse_hits: pd.DataFrame,
    meta: dict[str, Any],
    shots: pd.DataFrame,
    dense_fps: float,
    dense_window_sec: float,
) -> pd.DataFrame:
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])
    step_sec = 1.0 / max(0.001, dense_fps)
    frames: set[int] = set()
    for _, row in sparse_hits.iterrows():
        center = float(row["timestamp_sec"])
        t = center - dense_window_sec
        while t <= center + dense_window_sec + 1e-9:
            fid = int(round(t * fps))
            if 0 <= fid < total_frames:
                frames.add(fid)
            t += step_sec

    rows = []
    for fid in sorted(frames):
        rows.append({
            "video_id": str(sparse_hits.iloc[0]["video_id"]) if not sparse_hits.empty else "",
            "frame_idx": int(fid),
            "timestamp_sec": float(fid / fps),
            "shot_id": shot_id_for_frame(shots, int(fid)),
            "sample_stage": "dense",
        })
    return pd.DataFrame(rows)


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def init_ocr(cfg: dict[str, Any], device: str) -> tuple[Any, Any | None, bool]:
    from paddleocr import PaddleOCR

    det_dir = resolve_path(cfg.get("ocr", {}).get("paddle_detection_model_dir", ".ocr_cache/paddlex/official_models/PP-OCRv6_medium_det"))
    rec_dir = resolve_path(cfg.get("ocr", {}).get("paddle_recognition_model_dir", ".ocr_cache/paddlex/official_models/PP-OCRv6_medium_rec"))
    paddle_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": "cpu",
        "enable_mkldnn": False,
    }
    if det_dir.is_dir() and rec_dir.is_dir():
        paddle_kwargs["text_detection_model_dir"] = str(det_dir)
        paddle_kwargs["text_recognition_model_dir"] = str(rec_dir)
    reader = PaddleOCR(**paddle_kwargs)

    recognizer = None
    use_gpu = device != "cpu" and torch.cuda.is_available()
    use_vietocr = bool(cfg.get("ocr", {}).get("use_vietocr_if_available", False))
    if use_vietocr:
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor

            vcfg_path = resolve_path(cfg.get("ocr", {}).get("vietocr_config", "configs/vietocr_vgg_transformer_local.yaml"))
            if vcfg_path.exists():
                vcfg = Cfg.load_config_from_file(str(vcfg_path))
                weights = Path(str(vcfg.get("weights", "")))
                if not weights.is_absolute():
                    weights = resolve_path(weights)
                if not weights.is_file():
                    weights = ensure_vietocr_weights(weights)
                vcfg["weights"] = str(weights)
            else:
                vcfg = Cfg.load_config_from_name("vgg_transformer")
                vcfg["weights"] = str(ensure_vietocr_weights())
            vcfg["device"] = "cuda:0" if use_gpu else "cpu"
            recognizer = Predictor(vcfg)
        except Exception as exc:
            print(f"[WARN] VietOCR unavailable, using PaddleOCR recognition fallback: {exc}")
            recognizer = None
    return reader, recognizer, use_gpu


def ocr_frame(
    reader: Any,
    recognizer: Any | None,
    frame_bgr: np.ndarray,
    min_confidence: float,
) -> list[dict[str, Any]]:
    height, width = frame_bgr.shape[:2]
    raw = reader.predict(frame_bgr)
    item = raw[0] if raw else {}
    boxes = item.get("rec_polys", item.get("dt_polys", []))
    paddle_texts = item.get("rec_texts", [])
    paddle_scores = item.get("rec_scores", [])
    detections = []
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    for idx, bbox in enumerate(boxes):
        bbox_xyxy = bbox_to_xyxy(bbox, width, height)
        text_raw = normalize_light(str(paddle_texts[idx] if idx < len(paddle_texts) else ""))
        paddle_conf = float(paddle_scores[idx]) if idx < len(paddle_scores) else 0.0
        rec_conf = paddle_conf
        final_text = text_raw

        if recognizer is not None:
            try:
                x1, y1, x2, y2 = bbox_xyxy
                if x2 - x1 > 4 and y2 - y1 > 4:
                    crop = pil_img.crop((x1, y1, x2, y2))
                    crop = preprocess_crop_image(crop, scale_factor=2.5, pad_px=6)
                    vt_result = recognizer.predict(crop, return_prob=True)
                    if isinstance(vt_result, tuple):
                        vt_text, vt_prob = vt_result
                    else:
                        vt_text, vt_prob = vt_result, None
                    vt_text = normalize_light(vt_text)
                    if vt_text:
                        final_text = vt_text
                        if vt_prob is not None:
                            try:
                                rec_conf = max(rec_conf, float(vt_prob))
                            except Exception:
                                pass
            except Exception:
                pass

        final_text = normalize_light(final_text)
        if not final_text:
            continue
        detections.append({
            "bbox": bbox_xyxy,
            "box": [[int(x), int(y)] for x, y in bbox],
            "region_type": classify_roi(bbox_xyxy, width, height),
            "text": final_text,
            "text_raw": text_raw,
            "text_normalized": normalize_light(final_text),
            "det_conf": paddle_conf,
            "rec_conf": rec_conf,
            "confidence": rec_conf,
            "paddle_text": text_raw,
            "ocr_backend": "paddleocr+vietocr" if recognizer is not None else "paddleocr",
        })
    return [d for d in detections if float(d.get("confidence", 0.0)) >= min_confidence]


def combine_text(detections: list[dict[str, Any]], min_confidence: float) -> tuple[str, float, int]:
    kept = [
        str(d["text_normalized"]).strip()
        for d in detections
        if str(d.get("text_normalized", "")).strip() and float(d.get("confidence", 0.0)) >= min_confidence
    ]
    mean_conf = float(np.mean([float(d.get("confidence", 0.0)) for d in detections])) if detections else 0.0
    return " ".join(kept), mean_conf, len(kept)


def process_samples(
    video_path: Path,
    samples: pd.DataFrame,
    reader: Any,
    recognizer: Any | None,
    min_confidence: float,
    stage_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_records = []
    observation_rows = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    for _, sample in tqdm(samples.sort_values("frame_idx").iterrows(), total=len(samples), desc=f"OCR {stage_label}"):
        frame_idx = int(sample["frame_idx"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        detections = ocr_frame(reader, recognizer, frame, min_confidence=min_confidence)
        text, mean_conf, num_boxes = combine_text(detections, min_confidence)
        record = {
            "video_id": str(sample["video_id"]),
            "frame_idx": frame_idx,
            "timestamp_seconds": float(sample["timestamp_sec"]),
            "timestamp_sec": float(sample["timestamp_sec"]),
            "shot_id": int(sample["shot_id"]),
            "sample_stage": str(sample["sample_stage"]),
            "detections": detections,
            "combined_text": text,
            "mean_confidence": mean_conf,
            "num_text_boxes": num_boxes,
            "ocr_status": "ok",
        }
        frame_records.append(record)
        for det_idx, det in enumerate(detections):
            observation_rows.append({
                "video_id": record["video_id"],
                "frame_idx": frame_idx,
                "timestamp": record["timestamp_seconds"],
                "timestamp_sec": record["timestamp_seconds"],
                "shot_id": record["shot_id"],
                "sample_stage": record["sample_stage"],
                "det_idx": det_idx,
                "text_raw": det["text_raw"],
                "text_normalized": det["text_normalized"],
                "text": det["text"],
                "det_conf": float(det["det_conf"]),
                "rec_conf": float(det["rec_conf"]),
                "confidence": float(det["confidence"]),
                "region_type": det["region_type"],
                "bbox": det["bbox"],
            })
    cap.release()
    return frame_records, observation_rows


def load_keyframes(video_id: str, cfg: dict[str, Any]) -> pd.DataFrame:
    v2_path = resolve_path(cfg["paths"].get("keyframe_v2_global_map", "outputs/keyframe_v2_full/indexes/keyframe_v2_global_map.parquet"))
    if v2_path.exists():
        df = pd.read_parquet(v2_path) if v2_path.suffix.lower() == ".parquet" else pd.read_csv(v2_path)
        df = df[df["video_id"].astype(str) == video_id].copy()
        df["keyframe_time"] = df["timestamp_sec"].astype(float)
        df["mapped_frame_id"] = df["keyframe_v2_idx"].astype(int)
        df["mapped_frame_idx"] = df["actual_frame_id"].astype(int)
        df["mapped_global_id"] = df["global_v2_id"].astype(int)
        df["mapped_keyframe_name"] = df["image_path"].astype(str).map(lambda p: Path(p).name)
        df["mapped_keyframe_path"] = df["image_path"].astype(str)
        return df.sort_values("keyframe_time").reset_index(drop=True)

    mapping_root = resolve_path(cfg["paths"]["btc_mapping_root"])
    btc_path = mapping_root / f"{video_id}.csv"
    df = pd.read_csv(btc_path)
    df["keyframe_time"] = df["pts_time"].astype(float)
    df["mapped_frame_id"] = df["n"].astype(int)
    df["mapped_frame_idx"] = df["frame_idx"].astype(int)
    df["mapped_global_id"] = range(len(df))
    df["mapped_keyframe_name"] = df["n"].astype(int).map(lambda n: f"{n:03d}.jpg")
    df["mapped_keyframe_path"] = ""
    return df.sort_values("keyframe_time").reset_index(drop=True)


def map_segments_to_keyframes(segments: list[dict[str, Any]], keyframes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seg in segments:
        start = float(seg["start_time"])
        end = float(seg["end_time"])
        rep = float((start + end) / 2.0)
        in_interval = keyframes[(keyframes["keyframe_time"] >= start) & (keyframes["keyframe_time"] <= end)]
        candidates = in_interval if not in_interval.empty else keyframes
        distances = (candidates["keyframe_time"].astype(float) - rep).abs()
        mapped = candidates.loc[distances.idxmin()]
        out = dict(seg)
        out["text"] = seg.get("text_consensus", "")
        out["text_normalized"] = normalize_light(seg.get("text_consensus", ""))
        out["representative_time"] = rep
        out["confidence"] = float(seg.get("mean_confidence", 0.0))
        out["mapped_global_id"] = int(mapped["mapped_global_id"])
        out["mapped_frame_id"] = int(mapped["mapped_frame_id"])
        out["mapped_frame_idx"] = int(mapped["mapped_frame_idx"])
        out["mapped_keyframe_name"] = str(mapped["mapped_keyframe_name"])
        out["mapped_keyframe_path"] = str(mapped["mapped_keyframe_path"])
        out["mapped_keyframe_timestamp"] = float(mapped["keyframe_time"])
        out["distance_to_keyframe"] = float(abs(float(mapped["keyframe_time"]) - rep))
        out["mapping_case"] = "interval" if not in_interval.empty else "nearest"
        rows.append(out)
    return pd.DataFrame(rows)


def build_test_index(mapped_df: pd.DataFrame, output_dir: Path, cfg: dict[str, Any], device_arg: str) -> dict[str, Any]:
    index_cfg = cfg.get("index", {})
    if not bool(index_cfg.get("build", True)) or mapped_df.empty:
        return {"built": False}

    index_dir = output_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    corpus = mapped_df[mapped_df.get("use_for_semantic_search", True).astype(bool)].copy()
    corpus = corpus[corpus["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    corpus_path = index_dir / "ocr_video_v4_corpus.parquet"
    corpus.to_parquet(corpus_path, index=False)

    model_name = str(index_cfg.get("model_name", "intfloat/multilingual-e5-small"))
    batch_size = int(index_cfg.get("batch_size", 32))
    device = "cuda:0" if device_arg != "cpu" and torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    texts = [f"passage: {t}" for t in corpus["text"].astype(str).tolist()]
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Build OCR V4 E5 index"):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy().astype("float32")
            embeddings.append(emb)

    arr = np.vstack(embeddings).astype("float32") if embeddings else np.zeros((0, 384), dtype="float32")
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss_path = index_dir / "ocr_video_v4_flat_ip.faiss"
    faiss.write_index(index, str(faiss_path))
    index_map_path = index_dir / "ocr_video_v4_index_map.parquet"
    corpus[["ocr_segment_id", "video_id", "mapped_global_id", "mapped_frame_id", "mapped_frame_idx"]].to_parquet(index_map_path, index=False)
    return {
        "built": True,
        "corpus_path": str(corpus_path),
        "faiss_path": str(faiss_path),
        "index_map_path": str(index_map_path),
        "vectors": int(index.ntotal),
        "embedding_dim": int(index.d),
        "model_name": model_name,
        "device": device,
    }


def draw_pair_image(video_path: Path, seg: dict[str, Any], output_path: Path) -> Path | None:
    source_frame_idx = int(seg.get("source_frames", [seg.get("start_frame", 0)])[0])
    source = read_frame(video_path, source_frame_idx)
    if source is None:
        return None
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    left = Image.fromarray(source_rgb).resize((480, 270))
    right_path = Path(str(seg.get("mapped_keyframe_path", "")))
    if right_path.exists():
        right = Image.open(right_path).convert("RGB").resize((480, 270))
    else:
        right = Image.new("RGB", (480, 270), "white")

    draw_l = ImageDraw.Draw(left)
    bbox = seg.get("bbox_mean", [0, 0, 0, 0])
    scale_x = 480 / max(1, source_rgb.shape[1])
    scale_y = 270 / max(1, source_rgb.shape[0])
    box = [int(bbox[0] * scale_x), int(bbox[1] * scale_y), int(bbox[2] * scale_x), int(bbox[3] * scale_y)]
    draw_l.rectangle(box, outline="red", width=3)

    canvas = Image.new("RGB", (960, 350), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (480, 0))
    draw = ImageDraw.Draw(canvas)
    text = str(seg.get("text", seg.get("text_consensus", "")))[:120]
    label = (
        f"{seg.get('ocr_segment_id')} | {text}\n"
        f"OCR t={float(seg.get('representative_time', 0.0)):.2f}s frame={source_frame_idx} "
        f"-> keyframe={seg.get('mapped_frame_idx')} t={float(seg.get('mapped_keyframe_timestamp', 0.0)):.2f}s "
        f"dt={float(seg.get('distance_to_keyframe', 0.0)):.2f}s conf={float(seg.get('confidence', 0.0)):.2f}"
    )
    draw.text((8, 278), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)
    return output_path


def make_contact_sheet(image_paths: list[Path], output_path: Path, cols: int = 2) -> None:
    if not image_paths:
        return
    imgs = [Image.open(p).convert("RGB") for p in image_paths if p.exists()]
    if not imgs:
        return
    w, h = imgs[0].size
    rows = int(np.ceil(len(imgs) / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, img in enumerate(imgs):
        sheet.paste(img, ((i % cols) * w, (i // cols) * h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_paddle_stage(
    video_id: str,
    cfg: dict[str, Any],
    config_path: str | Path,
    output_base: Path,
    sparse_fps: float,
    dense_fps: float,
    dense_window: float,
    force: bool,
) -> dict[str, Any]:
    summary_path = output_base / video_id / "paddle_stage_summary.json"
    records_path = output_base / video_id / "paddle_frame_records.jsonl"
    if records_path.exists() and summary_path.exists() and not force:
        print(f"[OCR_V4] Reusing existing PaddleOCR records: {records_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    stage_python_cfg = cfg.get("ocr", {}).get("paddle_stage_python") or sys.executable
    stage_python = resolve_path(stage_python_cfg)
    stage_script = PROJECT_ROOT / "scripts" / "run_ocr_video_v4_paddle_stage.py"
    if not stage_python.exists():
        print(f"[OCR_V4] PaddleOCR stage Python not found, falling back to current interpreter: {stage_python}")
        stage_python = Path(sys.executable)
    if not stage_script.exists():
        raise FileNotFoundError(f"PaddleOCR stage script not found: {stage_script}")

    cmd = [
        str(stage_python),
        str(stage_script),
        "--video-id",
        video_id,
        "--config",
        str(resolve_path(config_path)),
        "--output-dir",
        str(output_base),
        "--sparse-fps",
        str(sparse_fps),
        "--dense-fps",
        str(dense_fps),
        "--dense-window",
        str(dense_window),
        "--device",
        str(cfg.get("ocr", {}).get("paddle_device", "cpu")),
        "--batch-size",
        str(cfg.get("ocr", {}).get("batch_size", 16)),
    ]
    if force:
        cmd.append("--force")

    requested_device = str(cfg.get("ocr", {}).get("paddle_device", "cpu"))
    if requested_device.startswith("gpu") or requested_device.startswith("cuda"):
        try:
            import paddle

            if not paddle.device.is_compiled_with_cuda():
                requested_device = "cpu"
        except Exception:
            requested_device = "cpu"
    cmd[cmd.index("--device") + 1] = requested_device

    print("[OCR_V4] Starting PaddleOCR subprocess:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(64)
        if not chunk:
            break
        sys.stdout.write(chunk)
        sys.stdout.flush()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"PaddleOCR stage failed with exit code {rc}")

    if not summary_path.exists():
        raise FileNotFoundError(f"PaddleOCR stage did not create summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def init_vietocr(cfg: dict[str, Any], device: str) -> tuple[Any, str]:
    from src.preprocessing.model_assets import ensure_vietocr_weights
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    use_gpu = device != "cpu" and torch.cuda.is_available()
    vcfg_path = resolve_path(cfg.get("ocr", {}).get("vietocr_config", "configs/vietocr_vgg_transformer_local.yaml"))
    if vcfg_path.exists():
        vcfg = Cfg.load_config_from_file(str(vcfg_path))
        weights = Path(str(vcfg.get("weights", "")))
        if not weights.is_absolute():
            weights = resolve_path(weights)
        if not weights.is_file():
            weights = ensure_vietocr_weights(weights)
        vcfg["weights"] = str(weights)
    else:
        vcfg = Cfg.load_config_from_name("vgg_transformer")
        vcfg["weights"] = str(ensure_vietocr_weights())
    vcfg["device"] = "cuda:0" if use_gpu else "cpu"
    return Predictor(vcfg), str(vcfg["device"])


VIETNAMESE_ACCENT_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵ"
    "đ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴ"
    "Đ"
)

OCR_NOISE_BLACKLIST = {
    "hd",
    "htm",
    "htv",
    "htv9",
    "htv hd",
    "giay",
    "giây",
    "sec",
    "s",
}


def has_vietnamese_accents(text: str) -> bool:
    return any(ch in VIETNAMESE_ACCENT_CHARS for ch in text or "")


def normalized_no_accent(text: str) -> str:
    return remove_vietnamese_accents(normalize_light(text)).lower()


def looks_numeric_garbage(text: str) -> bool:
    text = normalize_light(text)
    if re.fullmatch(r"0[0-9]{6,}", text):
        return True
    digits = sum(ch.isdigit() for ch in text)
    alnum = sum(ch.isalnum() for ch in text)
    return alnum >= 6 and digits / max(1, alnum) >= 0.75


def looks_latin_hallucination(text: str) -> bool:
    text = normalize_light(text)
    compact = re.sub(r"[^A-Za-z]", "", text)
    if len(compact) < 6 or has_vietnamese_accents(text):
        return False
    if text.upper() == text and re.fullmatch(r"[A-Z\s-]{6,}", text):
        return True
    suspicious_suffixes = (
        "TION",
        "TIONS",
        "ALITY",
        "ALITIES",
        "ISM",
        "ISMS",
        "IZED",
        "IZING",
        "ATIONAL",
        "PRESSION",
        "CONTRACTION",
    )
    return compact.upper().endswith(suspicious_suffixes) or "CONTRACTION" in compact.upper()


def too_many_single_char_tokens(text: str) -> bool:
    tokens = [t for t in re.split(r"\s+", normalize_light(text)) if t]
    if len(tokens) < 4:
        return False
    singles = sum(len(re.sub(r"\W+", "", t)) <= 1 for t in tokens)
    return singles / max(1, len(tokens)) >= 0.45


def semantic_noise_flags(text: str, region_type: str) -> list[str]:
    text = normalize_light(text)
    flags: list[str] = []
    if not text:
        flags.append("empty_text")
        return flags
    no_accent = normalized_no_accent(text)
    if len(text) <= 2:
        flags.append("semantic_too_short")
    if no_accent in OCR_NOISE_BLACKLIST:
        flags.append("semantic_noise_blacklist")
    if looks_numeric_garbage(text):
        flags.append("numeric_garbage")
    if looks_latin_hallucination(text):
        flags.append("latin_hallucination_like")
    if too_many_single_char_tokens(text):
        flags.append("too_many_single_char_tokens")
    if region_type in {"logo_channel", "clock_time"}:
        flags.append("non_semantic_region")
    return flags


def should_use_vietocr_candidate(viet_text: str, paddle_text: str, prob: float, region_type: str) -> tuple[bool, str]:
    viet_text = normalize_light(viet_text)
    paddle_text = normalize_light(paddle_text)
    if not viet_text:
        return False, "vietocr_empty"
    if looks_numeric_garbage(viet_text):
        return False, "vietocr_numeric_garbage"
    if looks_latin_hallucination(viet_text):
        return False, "vietocr_latin_hallucination"
    if normalized_no_accent(viet_text) in OCR_NOISE_BLACKLIST and normalized_no_accent(paddle_text) not in OCR_NOISE_BLACKLIST:
        return False, "vietocr_noise_blacklist"
    if len(viet_text) <= 2 and len(paddle_text) > 2:
        return False, "vietocr_too_short"
    if too_many_single_char_tokens(viet_text):
        return False, "vietocr_single_char_noise"
    if region_type in {"logo_channel", "clock_time"} and paddle_text:
        return False, "keep_paddle_for_non_semantic_region"

    # VietOCR is used as a corrector, not an unconditional replacement.
    if prob >= 0.78 and (has_vietnamese_accents(viet_text) or len(viet_text.split()) >= 2 or not paddle_text):
        return True, "vietocr_clean_high_conf"
    if prob >= 0.90 and len(viet_text) >= 4 and not looks_latin_hallucination(viet_text):
        return True, "vietocr_clean_very_high_conf"
    return False, "vietocr_not_better_than_paddle"


def apply_vietocr_to_segments(
    segments: list[dict[str, Any]],
    video_path: Path,
    cfg: dict[str, Any],
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not bool(cfg.get("ocr", {}).get("use_vietocr_if_available", True)) or not segments:
        return segments, {"enabled": False, "device": "disabled", "crops": 0}

    recognizer, vietocr_device = init_vietocr(cfg, device)
    batch_size = max(1, int(cfg.get("ocr", {}).get("vietocr_batch_size", cfg.get("ocr", {}).get("batch_size", 64))))

    # Group segments by their representative frame
    frame_to_crops: dict[int, list[tuple[int, list[int]]]] = {}
    for seg_idx, seg in enumerate(segments):
        source_frames = seg.get("source_frames", [])
        if not source_frames:
            source_frames = [seg.get("start_frame", 0)]
        rep_frame = int(source_frames[len(source_frames) // 2])
        bbox = seg.get("bbox_mean", [0, 0, 0, 0])
        frame_to_crops.setdefault(rep_frame, []).append((seg_idx, bbox))

    # Fast sequential frame decode
    sorted_fids = sorted(frame_to_crops.keys())
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    seg_crops: list[tuple[int, Image.Image]] = []
    cur_pos = 0
    for fid in tqdm(sorted_fids, desc="Extract representative segment crops"):
        gap = fid - cur_pos
        if gap == 0:
            ok, frame = cap.read()
            cur_pos += 1
        elif 0 < gap <= 30:
            for _ in range(gap - 1):
                cap.grab()
            ok, frame = cap.read()
            cur_pos = fid + 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            cur_pos = fid + 1

        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for seg_idx, bbox in frame_to_crops[fid]:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w, int(x2)))
            y2 = max(0, min(h, int(y2)))
            if (x2 - x1) > 4 and (y2 - y1) > 4:
                crop = frame_rgb[y1:y2, x1:x2]
                pil_crop = Image.fromarray(crop)
                pil_proc = preprocess_crop_image(pil_crop, scale_factor=2.0, pad_px=6)
                seg_crops.append((seg_idx, pil_proc))

    cap.release()

    # Batch GPU recognition
    for i in tqdm(range(0, len(seg_crops), batch_size), desc="VietOCR segment recognition"):
        batch = seg_crops[i : i + batch_size]
        imgs = [item[1] for item in batch]
        indices = [item[0] for item in batch]
        try:
            texts, probs = recognizer.predict_batch(imgs, return_prob=True)
        except Exception:
            texts, probs = [], []
            for img in imgs:
                try:
                    t, p = recognizer.predict(img, return_prob=True)
                    texts.append(t)
                    probs.append(p)
                except Exception:
                    texts.append("")
                    probs.append(0.0)

        for seg_idx, text, prob in zip(indices, texts, probs):
            text_clean = normalize_light(str(text))
            p_val = float(prob) if prob is not None else 0.0
            seg = segments[seg_idx]
            paddle_text = seg.get("text_consensus", "")
            seg["vietocr_text"] = text_clean
            seg["paddle_text"] = paddle_text

            use_vietocr, decision = should_use_vietocr_candidate(
                text_clean,
                str(paddle_text),
                p_val,
                str(seg.get("region_type", "scene_text")),
            )

            if use_vietocr:
                final_text = text_clean
                final_conf = round(max(float(seg.get("mean_confidence", 0.0)), p_val), 3)
                seg["ocr_backend"] = "paddle_gpu+vietocr_segment"
            else:
                final_text = paddle_text
                final_conf = float(seg.get("mean_confidence", 0.85))
                seg["ocr_backend"] = "paddle_gpu"
            seg["vietocr_decision"] = decision

            seg["text_consensus"] = final_text
            seg["text"] = final_text
            seg["text_normalized"] = final_text
            seg["text_search"] = normalize_text_search(final_text)
            seg["text_search_no_accent"] = remove_vietnamese_accents(final_text)
            seg["confidence"] = final_conf
            seg["mean_confidence"] = final_conf

    return segments, {
        "enabled": True,
        "device": vietocr_device,
        "crops": int(len(seg_crops)),
        "batch_size": batch_size,
    }


def apply_ocr_quality_filters(mapped_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if mapped_df.empty:
        return mapped_df, {"enabled": True, "flagged_segments": 0, "semantic_before": 0, "semantic_after": 0}

    df = mapped_df.copy()
    original_semantic = df["use_for_semantic_search"].astype(bool) if "use_for_semantic_search" in df else pd.Series([True] * len(df))
    all_flags: list[str] = []
    keep_semantic: list[bool] = []
    critical_flags = {
        "empty_text",
        "semantic_too_short",
        "semantic_noise_blacklist",
        "numeric_garbage",
        "latin_hallucination_like",
        "too_many_single_char_tokens",
        "non_semantic_region",
    }

    for _, row in df.iterrows():
        flags = semantic_noise_flags(str(row.get("text", row.get("text_consensus", ""))), str(row.get("region_type", "scene_text")))
        try:
            if float(row.get("distance_to_keyframe", 0.0)) > 3.0:
                flags.append("far_keyframe_mapping")
        except Exception:
            pass
        all_flags.append(";".join(flags))
        is_bad = any(flag in critical_flags for flag in flags)
        keep_semantic.append(bool(row.get("use_for_semantic_search", True)) and not is_bad)

    df["ocr_quality_flags"] = all_flags
    df["ocr_quality_pass"] = [not any(flag in critical_flags for flag in flags.split(";") if flag) for flags in all_flags]
    df["use_for_semantic_search"] = keep_semantic

    flag_counts: dict[str, int] = {}
    for flags in all_flags:
        for flag in [f for f in flags.split(";") if f]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    meta = {
        "enabled": True,
        "flagged_segments": int(sum(bool(f) for f in all_flags)),
        "semantic_before": int(original_semantic.sum()),
        "semantic_after": int(sum(keep_semantic)),
        "removed_from_semantic": int(original_semantic.sum() - sum(keep_semantic)),
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda kv: kv[1], reverse=True)),
    }
    return df, meta


def records_to_observation_rows(frame_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in frame_records:
        for det_idx, det in enumerate(rec.get("detections", [])):
            rows.append({
                "video_id": rec["video_id"],
                "frame_idx": int(rec["frame_idx"]),
                "timestamp": float(rec["timestamp_seconds"]),
                "timestamp_sec": float(rec["timestamp_seconds"]),
                "shot_id": int(rec.get("shot_id", -1)),
                "sample_stage": str(rec.get("sample_stage", "")),
                "det_idx": int(det.get("det_idx", det_idx)),
                "text_raw": det.get("text_raw", det.get("text", "")),
                "text_normalized": det.get("text_normalized", det.get("text", "")),
                "text": det.get("text", ""),
                "det_conf": float(det.get("det_conf", 0.0)),
                "rec_conf": float(det.get("rec_conf", 0.0)),
                "confidence": float(det.get("confidence", 0.0)),
                "region_type": det.get("region_type", "scene_text"),
                "bbox": det.get("bbox", [0, 0, 0, 0]),
                "crop_path": det.get("crop_path", ""),
                "paddle_text": det.get("paddle_text", ""),
                "vietocr_text": det.get("vietocr_text", ""),
                "ocr_backend": det.get("ocr_backend", ""),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run video-based OCR V4 on one video.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ocr_video_v4.yaml"))
    parser.add_argument("--sparse-fps", type=float, default=None)
    parser.add_argument("--dense-fps", type=float, default=None)
    parser.add_argument("--dense-window", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-paddle", action="store_true")
    args = parser.parse_args()

    setup_e_drive_cache()
    cfg = load_config(args.config)
    video_id = args.video_id.strip()
    sparse_fps = float(args.sparse_fps if args.sparse_fps is not None else cfg["sampling"].get("sparse_fps", 2.0))
    dense_fps = float(args.dense_fps if args.dense_fps is not None else cfg["sampling"].get("dense_fps", 5.0))
    dense_window = float(args.dense_window if args.dense_window is not None else cfg["sampling"].get("dense_window_sec", 1.0))
    detect_threshold = float(cfg["sampling"].get("detection_confidence_threshold", 0.45))
    min_confidence = float(cfg["ocr"].get("min_confidence", 0.35))

    output_base = resolve_path(args.output_dir or cfg["paths"].get("output_dir", "outputs/ocr_video_v4_test"))
    output_dir = output_base / video_id
    summary_file = output_dir / "ocr_video_summary.json"
    if summary_file.exists() and not args.force:
        raise FileExistsError(f"Summary already exists: {summary_file}. Use --force to overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    video_path = resolve_path(cfg["paths"].get("video_root", "datasets_L21/Videos_L21_a/video")) / f"{video_id}.mp4"
    started = time.time()
    meta = video_metadata(video_path)
    keyframes = load_keyframes(video_id, cfg)

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
    
    observation_rows = records_to_observation_rows(frame_records)

    segments = [s.to_dict() for s in merge_video_ocr_records(
        frame_records,
        max_gap_seconds=float(cfg["temporal_merge"].get("max_gap_seconds", 3.0)),
        min_bbox_iou=float(cfg["temporal_merge"].get("min_bbox_iou", 0.30)),
        min_text_similarity=float(cfg["temporal_merge"].get("min_text_similarity", 0.70)),
    )]

    segments, vietocr_meta = apply_vietocr_to_segments(segments, video_path, cfg, args.device)
    mapped_df = map_segments_to_keyframes(segments, keyframes) if segments else pd.DataFrame()
    mapped_df, quality_meta = apply_ocr_quality_filters(mapped_df)

    sampled_path = output_dir / "sampled_frames.csv"
    sampled_df = pd.read_csv(sampled_path) if sampled_path.exists() else pd.DataFrame()
    raw_df = pd.DataFrame(observation_rows)

    pd.DataFrame(frame_records).to_json(output_dir / "ocr_frame_records.jsonl", orient="records", lines=True, force_ascii=False)
    raw_df.to_parquet(output_dir / "raw_ocr_observations.parquet", index=False)
    raw_df.to_csv(output_dir / "raw_ocr_observations.csv", index=False, encoding="utf-8-sig")
    mapped_df.to_parquet(output_dir / "ocr_temporal_segments.parquet", index=False)
    mapped_df.to_csv(output_dir / "ocr_temporal_segments.csv", index=False, encoding="utf-8-sig")
    mapped_df.to_parquet(output_dir / "ocr_keyframe_mapping.parquet", index=False)
    mapped_df.to_csv(output_dir / "ocr_keyframe_mapping.csv", index=False, encoding="utf-8-sig")
    write_jsonl(output_dir / "ocr_temporal_segments.jsonl", mapped_df.to_dict(orient="records"))
    if not mapped_df.empty:
        audit_df = mapped_df[mapped_df["ocr_quality_flags"].fillna("").astype(str) != ""].copy()
        audit_df.to_csv(debug_dir / "ocr_quality_audit_after_filter.csv", index=False, encoding="utf-8-sig")

    index_meta = build_test_index(mapped_df, output_dir, cfg, args.device)

    debug_limit = int(cfg.get("debug", {}).get("max_segments", 80))
    debug_images = []
    if not mapped_df.empty:
        debug_source = mapped_df.sort_values(["start_time", "ocr_segment_id"]).head(debug_limit)
        for _, row in tqdm(debug_source.iterrows(), total=len(debug_source), desc="Debug segment images"):
            seg = row.to_dict()
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(seg["ocr_segment_id"]))
            out_img = debug_dir / "segments" / f"{safe_id}.jpg"
            made = draw_pair_image(video_path, seg, out_img)
            if made:
                debug_images.append(made)
        make_contact_sheet(debug_images, debug_dir / "ocr_segments_contact_sheet.jpg", cols=int(cfg.get("debug", {}).get("contact_sheet_cols", 2)))

    runtime = round(time.time() - started, 2)
    paddle_device = str(paddle_summary.get("paddle_device", cfg.get("ocr", {}).get("paddle_device", "cpu")))
    vietocr_device = str(vietocr_meta.get("device", "unavailable"))
    paddle_label = "GPU" if paddle_device.startswith("gpu") else "CPU"
    vietocr_label = "GPU" if vietocr_device.startswith("cuda") else "CPU"

    summary = {
        "video_id": video_id,
        "video_path": str(video_path),
        "duration_sec": round(float(meta["duration_sec"]), 3),
        "fps": float(meta["fps"]),
        "total_video_frames": int(meta["total_frames"]),
        "shot_metadata": str(resolve_path(cfg["paths"].get("keyframe_v2_root", "outputs/keyframe_v2_full")) / video_id / "shots.csv"),
        "keyframe_mapping_source": "keyframe_v2" if "global_v2_id" in keyframes.columns else "btc",
        "keyframes_available": int(len(keyframes)),
        "sparse_fps": sparse_fps,
        "dense_fps": dense_fps,
        "dense_window_sec": dense_window,
        "sparse_frames": int(paddle_summary.get("sparse_frames", 0)),
        "dense_trigger_frames": int(paddle_summary.get("dense_trigger_frames", 0)),
        "dense_extra_frames": int(paddle_summary.get("dense_extra_frames", 0)),
        "total_ocr_frames": int(len(sampled_df)),
        "deduplicated_ocr_frames": int(len(sampled_df)),
        "raw_ocr_observations": int(len(raw_df)),
        "temporal_segments": int(len(mapped_df)),
        "mapped_segments": int(len(mapped_df[mapped_df["mapped_frame_idx"].notna()])) if not mapped_df.empty else 0,
        "semantic_segments": int(mapped_df["use_for_semantic_search"].astype(bool).sum()) if not mapped_df.empty and "use_for_semantic_search" in mapped_df else 0,
        "ocr_processing_time_sec": runtime,
        "runtime_sec": runtime,
        "device_requested": args.device,
        "paddle_device": paddle_device,
        "paddle_stage_runtime_sec": float(paddle_summary.get("runtime_sec", 0.0)),
        "vietocr_device": vietocr_device,
        "vietocr_crops": int(vietocr_meta.get("crops", 0)),
        "ocr_backend": f"PaddleOCR({paddle_label} subprocess)+VietOCR({vietocr_label})",
        "ocr_quality_filter": quality_meta,
        "index": index_meta,
        "outputs": {
            "raw_ocr_observations": str(output_dir / "raw_ocr_observations.parquet"),
            "ocr_temporal_segments": str(output_dir / "ocr_temporal_segments.parquet"),
            "ocr_keyframe_mapping": str(output_dir / "ocr_keyframe_mapping.parquet"),
            "summary": str(output_dir / "ocr_video_summary.json"),
            "contact_sheet": str(debug_dir / "ocr_segments_contact_sheet.jpg"),
        },
    }
    (output_dir / "ocr_video_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
