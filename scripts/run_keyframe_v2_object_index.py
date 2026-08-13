from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
import yaml

from _bootstrap import PROJECT_ROOT
from src.preprocessing.yoloe_detector import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_FREE_MODEL_NAME,
    YOLOEDetector,
    save_detection_visualization,
)
from src.preprocessing.yoloe_hybrid_filter import build_hybrid_outputs, get_hybrid_config
from src.preprocessing.yoloe_postprocess import clean_detections, flatten_prompt_classes, get_cleanup_config
from src.retrieval.object_index import build_object_index


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_detector(args: argparse.Namespace, model_cfg: dict[str, Any]) -> YOLOEDetector:
    return YOLOEDetector(
        model_name=args.model or model_cfg.get("name", DEFAULT_MODEL_NAME),
        prompt_free_model_name=args.prompt_free_model or model_cfg.get("prompt_free_name", DEFAULT_PROMPT_FREE_MODEL_NAME),
        cache_dir=args.cache_dir,
        conf=args.conf if args.conf is not None else model_cfg.get("conf", 0.25),
        iou=args.iou if args.iou is not None else model_cfg.get("iou", 0.70),
        imgsz=args.imgsz if args.imgsz is not None else model_cfg.get("imgsz", 640),
        device=args.device,
    )


def run_detection_mode(
    detector: YOLOEDetector,
    rows: pd.DataFrame,
    mode: str,
    classes: list[str],
    cleanup_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        image_path = Path(row["image_path"])
        try:
            if mode == "text":
                raw = detector.detect(image_path, mode="text", classes=classes)
            else:
                raw = detector.detect(image_path, mode="prompt_free")
            detections, cleanup_stats = clean_detections(raw, cleanup_cfg)
        except Exception as exc:
            raw = []
            detections = []
            cleanup_stats = {"total_detections_raw": 0, "total_detections_clean": 0}
            errors.append(
                {
                    "video_id": str(row["video_id"]),
                    "global_v2_id": int(row["global_v2_id"]),
                    "keyframe_v2_idx": int(row["keyframe_v2_idx"]),
                    "actual_frame_id": int(row["actual_frame_id"]),
                    "mode": mode,
                    "exception": repr(exc),
                }
            )
        records.append(
            {
                "video_id": str(row["video_id"]),
                "global_v2_id": int(row["global_v2_id"]),
                "keyframe_v2_idx": int(row["keyframe_v2_idx"]),
                "frame_id": int(row["actual_frame_id"]),
                "timestamp": float(row["timestamp_sec"]),
                "keyframe_name": Path(str(row["image_path"])).name,
                "keyframe_path": str(image_path),
                "mode": mode,
                "raw_detections": raw,
                "detections": detections,
                "cleanup_stats": cleanup_stats,
            }
        )
    return records, errors


def flatten_hybrid_records(hybrid_records: list[dict[str, Any]], row_lookup: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detection_rows: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    for record in hybrid_records:
        key = (str(record["video_id"]), str(record["keyframe_name"]))
        source_row = row_lookup[key]
        labels = []
        object_scores: dict[str, float] = {}
        for det_idx, det in enumerate(record.get("hybrid_detections", [])):
            label = str(det.get("label") or "")
            confidence = float(det.get("confidence", 0.0))
            labels.append(label)
            object_scores[label] = max(object_scores.get(label, 0.0), confidence)
            raw_source = str(det.get("source") or "hybrid")
            source = "text_prompt" if raw_source == "text" else ("prompt_free" if raw_source == "prompt_free" else raw_source)
            detection_rows.append(
                {
                    "video_id": source_row["video_id"],
                    "keyframe_v2_idx": int(source_row["keyframe_v2_idx"]),
                    "actual_frame_id": int(source_row["actual_frame_id"]),
                    "timestamp_sec": float(source_row["timestamp_sec"]),
                    "global_v2_id": int(source_row["global_v2_id"]),
                    "object_label": label,
                    "confidence": confidence,
                    "bbox": json.dumps(det.get("bbox", []), ensure_ascii=False),
                    "source": source,
                    "pipeline_source": "hybrid",
                    "detection_index": det_idx,
                    "image_path": source_row["image_path"],
                }
            )
        corpus_rows.append(
            {
                "global_v2_id": int(source_row["global_v2_id"]),
                "video_id": source_row["video_id"],
                "keyframe_v2_idx": int(source_row["keyframe_v2_idx"]),
                "actual_frame_id": int(source_row["actual_frame_id"]),
                "frame_idx": int(source_row["actual_frame_id"]),
                "timestamp_sec": float(source_row["timestamp_sec"]),
                "timestamp_seconds": float(source_row["timestamp_sec"]),
                "keyframe_name": Path(str(source_row["image_path"])).name,
                "keyframe_path": source_row["image_path"],
                "image_path": source_row["image_path"],
                "object_labels": json.dumps(sorted(set(labels)), ensure_ascii=False),
                "object_scores": json.dumps(object_scores, ensure_ascii=False),
                "object_count": len(labels),
                "search_text": " ".join(sorted(set(labels))),
            }
        )
    return detection_rows, corpus_rows


def write_detection_visualizations(
    detection_rows: list[dict[str, Any]],
    vis_dir: Path,
    limit: int,
) -> int:
    if limit == 0 or not detection_rows:
        return 0
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in detection_rows:
        grouped.setdefault(str(row["image_path"]), []).append(row)

    saved = 0
    for image_path, rows in sorted(grouped.items()):
        if limit > 0 and saved >= limit:
            break
        detections = []
        for row in rows:
            try:
                bbox = json.loads(str(row.get("bbox", "[]")))
            except json.JSONDecodeError:
                bbox = []
            if len(bbox) < 4:
                continue
            detections.append(
                {
                    "label": str(row.get("object_label", "")),
                    "confidence": float(row.get("confidence", 0.0)),
                    "bbox": bbox,
                }
            )
        if not detections:
            continue
        video_id = str(rows[0].get("video_id", "unknown"))
        keyframe_name = Path(image_path).name
        output_path = vis_dir / video_id / keyframe_name
        try:
            save_detection_visualization(image_path, detections, output_path)
            saved += 1
        except Exception as exc:
            print(f"[WARN] Failed to save object visualization for {image_path}: {exc}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOE Hybrid Object V2 over Keyframe V2 final keyframes.")
    parser.add_argument("--global-map", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full" / "indexes" / "keyframe_v2_global_map.parquet"))
    parser.add_argument("--classes-file", default=str(PROJECT_ROOT / "configs" / "yoloe_objects.yaml"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full" / "object_v2"))
    parser.add_argument("--index-output", default=str(PROJECT_ROOT / "outputs" / "keyframe_v2_full" / "indexes" / "object"))
    parser.add_argument("--model")
    parser.add_argument("--prompt-free-model")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--visualization-limit", type=int, default=None, help="Max bbox images to write. Use -1 for all.")
    args = parser.parse_args()

    started = time.time()
    config = load_yaml(args.classes_file)
    model_cfg = config.get("model") or {}
    cleanup_cfg = get_cleanup_config(config)
    hybrid_cfg = get_hybrid_config(config)
    classes = flatten_prompt_classes(config.get("classes") or [])
    if not classes:
        raise RuntimeError("No YOLOE text-prompt classes configured.")

    output_root = Path(args.output_root)
    detections_dir = output_root / "detections"
    stats_dir = output_root / "stats"
    vis_dir = output_root / "visualization"
    detections_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    index_output = Path(args.index_output)
    index_output.mkdir(parents=True, exist_ok=True)

    global_map = pd.read_parquet(args.global_map).sort_values("global_v2_id").reset_index(drop=True)
    if args.limit_frames:
        global_map = global_map.head(args.limit_frames)

    detector = build_detector(args, model_cfg)
    all_detection_rows: list[dict[str, Any]] = []
    all_corpus_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    video_stats: list[dict[str, Any]] = []

    for video_id, rows in global_map.groupby("video_id", sort=True):
        out_video = detections_dir / f"{video_id}.parquet"
        corpus_video = detections_dir / f"{video_id}_records.parquet"
        if out_video.exists() and corpus_video.exists() and not args.force:
            print(f"SKIP Object V2 complete: {video_id}")
            det_df = pd.read_parquet(out_video)
            corpus_df = pd.read_parquet(corpus_video)
            all_detection_rows.extend(det_df.to_dict("records"))
            all_corpus_rows.extend(corpus_df.to_dict("records"))
            continue
        print(f"RUN Object V2: {video_id} ({len(rows)} keyframes)")
        text_records, text_errors = run_detection_mode(detector, rows, "text", classes, cleanup_cfg)
        pf_records, pf_errors = run_detection_mode(detector, rows, "prompt_free", classes, cleanup_cfg)
        errors.extend(text_errors + pf_errors)
        hybrid_records, stats, audit = build_hybrid_outputs(text_records, pf_records, hybrid_cfg, cleanup_cfg)
        row_lookup = {
            (str(row["video_id"]), Path(str(row["image_path"])).name): row.to_dict()
            for _, row in rows.iterrows()
        }
        detection_rows, corpus_rows = flatten_hybrid_records(hybrid_records, row_lookup)
        pd.DataFrame(detection_rows).to_parquet(out_video, index=False)
        pd.DataFrame(corpus_rows).to_parquet(corpus_video, index=False)
        vis_limit = args.visualization_limit
        if vis_limit is None:
            vis_limit = int(hybrid_cfg.get("visualization_limit", 20))
        visualizations_saved = write_detection_visualizations(detection_rows, vis_dir, vis_limit)
        (stats_dir / f"{video_id}.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        (stats_dir / f"{video_id}_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        all_detection_rows.extend(detection_rows)
        all_corpus_rows.extend(corpus_rows)
        video_stats.append({"video_id": video_id, "visualizations_saved": visualizations_saved, **stats})

    det_all = pd.DataFrame(all_detection_rows)
    corpus_all = pd.DataFrame(all_corpus_rows)
    det_all.to_parquet(output_root / "l21_objects_v2_detections.parquet", index=False)
    corpus_all.to_parquet(output_root / "l21_objects_v2.parquet", index=False)
    corpus_all.to_parquet(index_output / "l21_objects_v2.parquet", index=False)

    _, idx_meta = build_object_index(index_output / "l21_objects_v2.parquet", index_output)
    label_counts = Counter(det_all["object_label"].astype(str)) if not det_all.empty else Counter()
    stats_json = {
        "total_keyframes": int(len(corpus_all)),
        "total_detections": int(len(det_all)),
        "unique_labels": int(len(label_counts)),
        "top_labels": label_counts.most_common(50),
        "video_stats": video_stats,
        "object_index": idx_meta,
        "visualization_dir": str(vis_dir.resolve()),
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_root / "object_v2_stats.json").write_text(json.dumps(stats_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "object_v2_metadata.json").write_text(
        json.dumps(
            {
                "global_map": str(Path(args.global_map).resolve()),
                "classes_file": str(Path(args.classes_file).resolve()),
                "model": args.model or model_cfg.get("name", DEFAULT_MODEL_NAME),
                "prompt_free_model": args.prompt_free_model or model_cfg.get("prompt_free_name", DEFAULT_PROMPT_FREE_MODEL_NAME),
                "cache_dir": args.cache_dir,
                "output_root": str(output_root.resolve()),
                "index_output": str(index_output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(errors).to_csv(output_root / "object_v2_errors.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(stats_json, indent=2, ensure_ascii=False))

    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
