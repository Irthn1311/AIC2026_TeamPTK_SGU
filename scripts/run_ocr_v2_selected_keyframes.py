from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.keyframe_ocr import classify_roi, preprocess_crop_image
from src.preprocessing.model_assets import ensure_vietocr_weights


def setup_e_drive_cache() -> None:
    if Path("/kaggle").exists():
        return
    cache_root = PROJECT_ROOT / ".ocr_cache"
    home = cache_root / "home"
    temp = cache_root / "temp"
    for path in [cache_root, home, temp]:
        path.mkdir(parents=True, exist_ok=True)
    os.environ["EASYOCR_MODULE_PATH"] = str(home / ".EasyOCR")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "huggingface" / "transformers")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    os.environ["TMP"] = str(temp)
    os.environ["TEMP"] = str(temp)
    os.environ["TMPDIR"] = str(temp)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_selected_keyframes(selected_root: Path, video_ids: list[str] | None, skip_video_ids: set[str] | None = None) -> list[dict[str, Any]]:
    skip_set = skip_video_ids or set()
    requested = {v.strip() for v in video_ids or [] if v.strip()}
    video_dirs = sorted(p for p in selected_root.iterdir() if p.is_dir() and "_V" in p.name and p.name.startswith("L"))
    if requested:
        video_dirs = [p for p in video_dirs if p.name in requested]
        missing = sorted(requested - {p.name for p in video_dirs})
        if missing:
            raise FileNotFoundError(f"Missing selected keyframe dirs: {missing}")

    if skip_set:
        video_dirs = [p for p in video_dirs if p.name not in skip_set]

    rows: list[dict[str, Any]] = []
    for video_dir in video_dirs:
        map_path = video_dir / "keyframe_btc_map.csv"
        if not map_path.exists():
            map_path = video_dir / "keyframe_v2_map.csv"
        if map_path.exists():
            df = pd.read_csv(map_path)
            records = df.to_dict("records")
            for idx, row in enumerate(records):
                img_path = str(row.get("image_path", ""))
                image_path = Path(img_path) if Path(img_path).is_absolute() else PROJECT_ROOT / img_path
                rows.append({
                    "video_id": str(row.get("video_id", video_dir.name)),
                    "keyframe_name": image_path.name,
                    "keyframe_path": str(image_path),
                    "keyframe_v2_idx": int(row.get("keyframe_v2_idx", idx)),
                    "global_id": int(row.get("global_id", idx)),
                    "frame_idx": int(row.get("actual_frame_id", row.get("frame_idx", 0))),
                    "timestamp_seconds": float(row.get("timestamp_ms", 0.0)) / 1000.0,
                    "shot_id": int(row.get("shot_id", -1)),
                })
        else:
            keyframe_dir = video_dir / "keyframes"
            for idx, image_path in enumerate(sorted(keyframe_dir.glob("*.jpg"))):
                rows.append({
                    "video_id": video_dir.name,
                    "keyframe_name": image_path.name,
                    "keyframe_path": str(image_path),
                    "keyframe_v2_idx": idx,
                    "global_id": idx,
                    "frame_idx": 0,
                    "timestamp_seconds": 0.0,
                    "shot_id": -1,
                })
    return rows


def init_models(device: str, vietocr_config: Path) -> tuple[Any, Any | None, str]:
    import easyocr
    import torch

    use_gpu = device != "cpu" and torch.cuda.is_available()
    print("  [1/2] 🧠 Loading EasyOCR Detector (gpu={}) ...".format(use_gpu))
    detector = easyocr.Reader(["vi", "en"], gpu=use_gpu)
    print("  [1/2] ✅ EasyOCR Detector Ready!")

    print("  [2/2] 🧠 Loading VietOCR Recognizer ...")
    recognizer = None
    recognizer_status = "unavailable"
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        if vietocr_config.exists():
            config = Cfg.load_config_from_file(str(vietocr_config))
            weights = Path(str(config.get("weights", "")))
            if not weights.is_absolute():
                weights = resolve_path(weights)
            if not weights.is_file():
                weights = ensure_vietocr_weights(weights)
            config["weights"] = str(weights)
        else:
            config = Cfg.load_config_from_name("vgg_transformer")
            config["weights"] = str(ensure_vietocr_weights())
        config["device"] = "cuda:0" if use_gpu else "cpu"
        recognizer = Predictor(config)
        recognizer_status = f"ready:{config['device']}"
        print("  [2/2] ✅ VietOCR Recognizer Ready!")
    except Exception as exc:
        recognizer_status = repr(exc)
        print(f"  [2/2] ⚠️ VietOCR fallback to EasyOCR only: {exc}")
    return detector, recognizer, recognizer_status


def predict_vietocr(recognizer: Any, crop: Image.Image) -> tuple[str, float | None]:
    import torch
    with torch.inference_mode():
        try:
            text, prob = recognizer.predict(crop, return_prob=True)
        except TypeError:
            text = recognizer.predict(crop)
            prob = None
    return " ".join(str(text).strip().split()), float(prob) if prob is not None else None


def ocr_image(
    image_path: Path,
    detector: Any,
    recognizer: Any | None,
    min_confidence: float,
    scale_factor: float,
    pad_px: int,
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], str, float]:
    import torch
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    with torch.inference_mode():
        raw_res = detector.readtext(str(image_path), batch_size=batch_size)
    detections: list[dict[str, Any]] = []
    for bbox, easy_text, easy_conf in raw_res:
        pts = np.array([[int(pt[0]), int(pt[1])] for pt in bbox], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        if w <= 4 or h <= 4:
            continue
        bbox_xyxy = [max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)]
        region_type = classify_roi(bbox_xyxy, img_w, img_h)
        final_text = " ".join(str(easy_text).strip().split())
        final_conf = float(easy_conf)
        vietocr_text = ""
        vietocr_conf = None
        if recognizer is not None:
            crop = img.crop((bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]))
            crop = preprocess_crop_image(crop, scale_factor=scale_factor, pad_px=pad_px)
            try:
                vietocr_text, vietocr_conf = predict_vietocr(recognizer, crop)
            except Exception:
                vietocr_text, vietocr_conf = "", None
            if vietocr_text:
                final_text = vietocr_text
                final_conf = max(final_conf, float(vietocr_conf or 0.0))

        detections.append({
            "bbox": bbox_xyxy,
            "box": [[int(pt[0]), int(pt[1])] for pt in bbox],
            "region_type": region_type,
            "text": final_text,
            "text_raw": final_text,
            "easyocr_text": str(easy_text).strip(),
            "easyocr_confidence": round(float(easy_conf), 4),
            "vietocr_text": vietocr_text,
            "vietocr_confidence": vietocr_conf,
            "confidence": round(final_conf, 4),
        })

    kept = [
        d["text"]
        for d in detections
        if float(d.get("confidence", 0.0)) >= min_confidence
        and str(d.get("text", "")).strip()
        and not (d.get("region_type") in {"logo_channel", "clock_time"} and len(str(d.get("text", ""))) <= 8)
    ]
    combined = " ".join(kept)
    mean_conf = sum(float(d.get("confidence", 0.0)) for d in detections) / max(1, len(detections))
    return detections, combined, mean_conf


def write_aggregate_outputs(rows: list[dict[str, Any]], output_dir: Path, started: float, summary_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    df.to_json(output_dir / "l21_keyframe_ocr.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_csv(output_dir / "l21_keyframe_ocr.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(output_dir / "l21_keyframe_ocr.parquet", index=False)
    except Exception:
        pass

    summary = {
        "output_dir": str(output_dir),
        "video_ids": sorted(set(str(r["video_id"]) for r in rows)),
        "images": len(rows),
        "errors": int(sum(1 for r in rows if r.get("ocr_status") == "error")),
        "elapsed_seconds": round(time.time() - started, 2),
        "outputs": {
            "csv": str(output_dir / "l21_keyframe_ocr.csv"),
            "jsonl": str(output_dir / "l21_keyframe_ocr.jsonl"),
            "per_video": str(output_dir / "per_video"),
        },
    }
    if summary_extra:
        summary.update(summary_extra)
    (output_dir / "ocr_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def aggregate_per_video_outputs(output_dir: Path, video_ids: list[str] | None = None) -> dict[str, Any]:
    per_video_dir = output_dir / "per_video"
    requested = {str(v).strip() for v in video_ids or [] if str(v).strip()}
    paths = sorted(per_video_dir.glob("L*_V*.jsonl"))
    if requested:
        paths = [path for path in paths if path.stem in requested]
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No per-video OCR V2 rows found in {per_video_dir}")
    return write_aggregate_outputs(rows, output_dir, time.time(), {"mode": "aggregate_only", "detector": "cached", "recognizer": "cached"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR V2 on selected keyframes from keyframe_v2 output.")
    parser.add_argument("--selected-root", default="outputs/keyframe_v2_full")
    parser.add_argument("--output-dir", default="outputs/ocr_v2_selected_keyframes")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--scale-factor", type=float, default=2.5)
    parser.add_argument("--pad-px", type=int, default=6)
    parser.add_argument("--vietocr-config", default="configs/vietocr_vgg_transformer_local.yaml")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16, help="EasyOCR batch size.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of GPU shards.")
    parser.add_argument("--shard-idx", type=int, default=0, help="Zero-indexed shard ID for this process.")
    parser.add_argument("--no-aggregate", action="store_true", help="Write per-video JSONL only. Useful for GPU sharded runs.")
    parser.add_argument("--aggregate-only", action="store_true", help="Merge existing per-video JSONL files without loading OCR models.")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip videos that already have non-empty per-video JSONL.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Force recomputing existing per-video JSONL.")
    args = parser.parse_args()

    started = time.time()
    setup_e_drive_cache()
    selected_root = resolve_path(args.selected_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_video_dir = output_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        print(json.dumps(aggregate_per_video_outputs(output_dir, args.video_id), ensure_ascii=False, indent=2))
        return

    existing_video_ids: set[str] = set()
    if args.resume:
        existing_video_ids = {p.stem for p in per_video_dir.glob("*.jsonl") if p.stat().st_size > 0}
        if existing_video_ids:
            print(f"[OCR V2 Resume] Found {len(existing_video_ids)} completed videos on disk, skipping them...")

    print("📂 Fast-loading keyframe metadata CSVs...")
    t_load = time.time()
    items = load_selected_keyframes(selected_root, args.video_id, skip_video_ids=existing_video_ids)
    print(f"✅ Loaded {len(items)} remaining keyframe items in {time.time() - t_load:.2f}s!")

    if not items:
        print("[OCR V2 Resume] No remaining items to process. All videos complete!")
        if not args.no_aggregate:
            print(json.dumps(aggregate_per_video_outputs(output_dir, args.video_id), ensure_ascii=False, indent=2))
        return

    if args.max_images and args.max_images > 0:
        items = items[: args.max_images]

    video_groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        video_groups.setdefault(str(item["video_id"]), []).append(item)

    if args.num_shards > 1:
        all_vids = sorted(video_groups.keys())
        shard_vids = set(v for i, v in enumerate(all_vids) if i % args.num_shards == args.shard_idx)
        video_groups = {k: v for k, v in video_groups.items() if k in shard_vids}
        print(f"[GPU Shard {args.shard_idx + 1}/{args.num_shards}] Processing {len(video_groups)} assigned videos.")

    detector, recognizer, recognizer_status = init_models(args.device, resolve_path(args.vietocr_config))

    rows: list[dict[str, Any]] = []
    errors = 0
    for video_id, v_items in tqdm(video_groups.items(), desc=f"OCR V2 [GPU {args.shard_idx}]"):
        v_records = []
        for item in v_items:
            image_path = Path(item["keyframe_path"])
            record = {
                **item,
                "detections": [],
                "combined_text": "",
                "mean_confidence": 0.0,
                "num_text_boxes": 0,
                "ocr_status": "ok",
                "error": "",
            }
            try:
                detections, combined, mean_conf = ocr_image(
                    image_path,
                    detector,
                    recognizer,
                    min_confidence=args.min_confidence,
                    scale_factor=args.scale_factor,
                    pad_px=args.pad_px,
                    batch_size=args.batch_size,
                )
                record["detections"] = detections
                record["combined_text"] = combined
                record["mean_confidence"] = round(mean_conf, 4)
                record["num_text_boxes"] = len(detections)
            except Exception as exc:
                errors += 1
                record["ocr_status"] = "error"
                record["error"] = repr(exc)
            v_records.append(record)
            rows.append(record)

        v_path = per_video_dir / f"{video_id}.jsonl"
        v_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in v_records), encoding="utf-8")

    summary_extra = {
        "selected_root": str(selected_root),
        "detector": "EasyOCR(['vi','en'])",
        "recognizer": f"VietOCR vgg_transformer {recognizer_status}",
        "scale_factor": args.scale_factor,
        "pad_px": args.pad_px,
        "min_confidence": args.min_confidence,
        "mode": "per_video_only" if args.no_aggregate else "full",
    }
    if args.no_aggregate:
        summary = {
            **summary_extra,
            "output_dir": str(output_dir),
            "video_ids": sorted(set(str(r["video_id"]) for r in rows)),
            "images": len(rows),
            "errors": errors,
            "elapsed_seconds": round(time.time() - started, 2),
            "outputs": {"per_video": str(per_video_dir)},
        }
    else:
        summary = write_aggregate_outputs(rows, output_dir, started, summary_extra)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
