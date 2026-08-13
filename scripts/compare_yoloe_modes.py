from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT
from src.preprocessing.yoloe_detector import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_FREE_MODEL_NAME,
    YOLOEDetector,
    save_detection_visualization,
)
from src.preprocessing.yoloe_postprocess import clean_detections, flatten_prompt_classes, get_cleanup_config
from src.retrieval.mapping_loader import load_keyframe_mapping


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_classes(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if args.classes:
        return args.classes
    classes = flatten_prompt_classes(config.get("classes") or [])
    if not classes:
        raise ValueError("No classes configured. Use --classes or --classes-file.")
    return classes


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


def run_mode(
    args: argparse.Namespace,
    mode: str,
    mapping_rows,
    classes: list[str],
    cleanup_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detector = build_detector(args, model_cfg)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    inference_times: list[float] = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    for _, row in tqdm(mapping_rows.iterrows(), total=len(mapping_rows), desc=f"YOLOE {args.video_id} {mode}"):
        image_path = Path(row["keyframe_path"])
        frame_id = int(row["frame_idx"])
        timestamp = float(row["timestamp_seconds"])
        try:
            frame_started = time.perf_counter()
            if mode == "text":
                raw_detections = detector.detect(image_path=image_path, mode="text", classes=classes)
            else:
                raw_detections = detector.detect(image_path=image_path, mode="prompt_free")
            inference_time = time.perf_counter() - frame_started
            detections, cleanup_stats = clean_detections(raw_detections, cleanup_cfg)
        except Exception as exc:
            inference_time = 0.0
            raw_detections = []
            detections = []
            cleanup_stats = {
                "total_detections_raw": 0,
                "total_detections_clean": 0,
                "detections_removed_as_duplicate": 0,
            }
            errors.append(
                {
                    "video_id": args.video_id,
                    "frame_id": frame_id,
                    "keyframe_name": str(row["keyframe_name"]),
                    "error": str(exc),
                }
            )

        inference_times.append(inference_time)
        records.append(
            {
                "video_id": args.video_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "keyframe_name": str(row["keyframe_name"]),
                "keyframe_path": str(image_path),
                "mode": mode,
                "raw_detections": raw_detections,
                "detections": detections,
                "cleanup_stats": cleanup_stats,
                "inference_time_seconds": round(inference_time, 6),
            }
        )

    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
    metrics = summarize_mode_records(records, inference_times, elapsed, peak_allocated, peak_reserved, errors)

    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records, metrics


def summarize_mode_records(
    records: list[dict[str, Any]],
    inference_times: list[float],
    elapsed: float,
    peak_allocated: int | None,
    peak_reserved: int | None,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_counts: Counter[str] = Counter()
    clean_counts: Counter[str] = Counter()
    clean_confidences: list[float] = []
    clean_per_frame: list[int] = []
    total_raw = 0
    total_clean = 0

    for record in records:
        raw = record["raw_detections"]
        clean = record["detections"]
        total_raw += len(raw)
        total_clean += len(clean)
        clean_per_frame.append(len(clean))
        raw_counts.update(str(item.get("label", "")) for item in raw if item.get("label"))
        clean_counts.update(str(item.get("label", "")) for item in clean if item.get("label"))
        clean_confidences.extend(float(item.get("confidence", 0.0)) for item in clean)

    total_frames = len(records)
    metrics = {
        "total_frames": total_frames,
        "frames_with_detection": sum(bool(record["detections"]) for record in records),
        "total_raw_detections": total_raw,
        "total_clean_detections": total_clean,
        "duplicate_removed": total_raw - total_clean,
        "unique_raw_labels": sorted(raw_counts),
        "unique_raw_label_count": len(raw_counts),
        "unique_clean_labels": sorted(clean_counts),
        "unique_clean_label_count": len(clean_counts),
        "raw_label_counts": dict(sorted(raw_counts.items())),
        "clean_label_counts": dict(sorted(clean_counts.items())),
        "average_detections_per_frame": round(total_clean / total_frames, 4) if total_frames else 0.0,
        "median_detections_per_frame": _median(clean_per_frame),
        "average_confidence": round(sum(clean_confidences) / len(clean_confidences), 4) if clean_confidences else 0.0,
        "median_confidence": _median(clean_confidences),
        "inference_time_total": round(sum(inference_times), 4),
        "wall_time_total": round(elapsed, 4),
        "average_inference_time_per_frame": round(sum(inference_times) / total_frames, 6) if total_frames else 0.0,
        "peak_gpu_memory_allocated_mb": round(peak_allocated / 1024 / 1024, 2) if peak_allocated is not None else None,
        "peak_gpu_memory_reserved_mb": round(peak_reserved / 1024 / 1024, 2) if peak_reserved is not None else None,
        "errors": errors,
        "error_count": len(errors),
    }
    return metrics


def compare_labels(text_metrics: dict[str, Any], prompt_free_metrics: dict[str, Any]) -> dict[str, Any]:
    text_raw = set(text_metrics["unique_raw_labels"])
    pf_raw = set(prompt_free_metrics["unique_raw_labels"])
    text_clean = set(text_metrics["unique_clean_labels"])
    pf_clean = set(prompt_free_metrics["unique_clean_labels"])
    return {
        "raw_labels": {
            "common": sorted(text_raw & pf_raw),
            "text_only": sorted(text_raw - pf_raw),
            "prompt_free_only": sorted(pf_raw - text_raw),
        },
        "clean_labels": {
            "common": sorted(text_clean & pf_clean),
            "text_only": sorted(text_clean - pf_clean),
            "prompt_free_only": sorted(pf_clean - text_clean),
        },
    }


def choose_visualization_frames(
    text_records: list[dict[str, Any]],
    prompt_free_records: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for text_record, pf_record in zip(text_records, prompt_free_records):
        text_labels = {item["label"] for item in text_record["detections"]}
        pf_labels = {item["label"] for item in pf_record["detections"]}
        count_delta = abs(len(text_record["detections"]) - len(pf_record["detections"]))
        label_delta = len(text_labels ^ pf_labels)
        prompt_free_extra = len(pf_labels - text_labels)
        text_extra = len(text_labels - pf_labels)
        score = count_delta * 3 + label_delta * 2 + prompt_free_extra + text_extra
        if score <= 0:
            continue
        candidates.append(
            {
                "frame_id": text_record["frame_id"],
                "timestamp": text_record["timestamp"],
                "keyframe_name": text_record["keyframe_name"],
                "keyframe_path": text_record["keyframe_path"],
                "score": score,
                "text_clean_count": len(text_record["detections"]),
                "prompt_free_clean_count": len(pf_record["detections"]),
                "text_labels": sorted(text_labels),
                "prompt_free_labels": sorted(pf_labels),
            }
        )
    candidates.sort(key=lambda item: (item["score"], item["prompt_free_clean_count"], item["text_clean_count"]), reverse=True)
    return candidates[:limit]


def save_comparison_visualizations(
    frames: list[dict[str, Any]],
    text_records: list[dict[str, Any]],
    prompt_free_records: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image in output_dir.glob("*_text.jpg"):
        old_image.unlink()
    for old_image in output_dir.glob("*_prompt_free.jpg"):
        old_image.unlink()
    by_key_text = {record["keyframe_name"]: record for record in text_records}
    by_key_pf = {record["keyframe_name"]: record for record in prompt_free_records}
    saved: list[dict[str, Any]] = []
    for frame in frames:
        key = frame["keyframe_name"]
        image_path = Path(frame["keyframe_path"])
        stem = Path(key).stem
        text_path = output_dir / f"{stem}_text.jpg"
        pf_path = output_dir / f"{stem}_prompt_free.jpg"
        save_detection_visualization(image_path, by_key_text[key]["detections"], text_path)
        save_detection_visualization(image_path, by_key_pf[key]["detections"], pf_path)
        saved.append({**frame, "text_visualization": str(text_path), "prompt_free_visualization": str(pf_path)})
    return saved


def _median(values: list[float] | list[int]) -> float:
    return round(float(statistics.median(values)), 4) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YOLOE text-prompt mode and official prompt-free mode.")
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "datasets_L21"))
    parser.add_argument("--classes", nargs="+", help="Optional text-prompt class override for text mode.")
    parser.add_argument("--classes-file", default=str(PROJECT_ROOT / "configs" / "yoloe_objects.yaml"))
    parser.add_argument("--model", help="YOLOE text/visual prompt model name or local .pt path.")
    parser.add_argument("--prompt-free-model", help="YOLOE prompt-free model name or local .pt path.")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", help="cuda:0 or cpu. Defaults to CUDA when available.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "yoloe" / "experiments"))
    parser.add_argument("--max-frames", type=int, help="Optional debug limit. Omit for full video.")
    parser.add_argument("--visualization-limit", type=int, default=20)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    mapping_path = dataset_root / "map-keyframes-aic25-b1" / "map-keyframes" / f"{args.video_id}.csv"
    keyframe_root = dataset_root / "Keyframes_L21" / "keyframes"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing keyframe mapping: {mapping_path}")
    if not (keyframe_root / args.video_id).is_dir():
        raise FileNotFoundError(f"Missing keyframe directory: {keyframe_root / args.video_id}")

    config = load_config(args.classes_file)
    model_cfg = config.get("model") or {}
    cleanup_cfg = get_cleanup_config(config)
    classes = resolve_classes(args, config)

    mapping_df = load_keyframe_mapping(mapping_path, keyframe_root).sort_values("feature_index").reset_index(drop=True)
    if args.max_frames:
        mapping_df = mapping_df.head(args.max_frames)

    experiment_dir = Path(args.output_dir) / args.video_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    text_records, text_metrics = run_mode(args, "text", mapping_df, classes, cleanup_cfg, model_cfg)
    prompt_free_records, prompt_free_metrics = run_mode(args, "prompt_free", mapping_df, classes, cleanup_cfg, model_cfg)

    label_comparison = compare_labels(text_metrics, prompt_free_metrics)
    selected_frames = choose_visualization_frames(text_records, prompt_free_records, args.visualization_limit)
    saved_visualizations = save_comparison_visualizations(
        selected_frames,
        text_records,
        prompt_free_records,
        experiment_dir / "visualization",
    )

    comparison = {
        "video_id": args.video_id,
        "text_prompt_model": args.model or model_cfg.get("name", DEFAULT_MODEL_NAME),
        "prompt_free_model": args.prompt_free_model or model_cfg.get("prompt_free_name", DEFAULT_PROMPT_FREE_MODEL_NAME),
        "classes_file": args.classes_file,
        "text_prompt_class_count": len(classes),
        "metrics": {
            "text": text_metrics,
            "prompt_free": prompt_free_metrics,
        },
        "label_comparison": label_comparison,
        "selected_visualization_frames": saved_visualizations,
    }

    (experiment_dir / "text_prompt.json").write_text(json.dumps(text_records, indent=2, ensure_ascii=False), encoding="utf-8")
    (experiment_dir / "prompt_free.json").write_text(json.dumps(prompt_free_records, indent=2, ensure_ascii=False), encoding="utf-8")
    (experiment_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    (experiment_dir / "label_comparison.json").write_text(
        json.dumps(label_comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Video: {args.video_id}")
    print(f"Text raw/clean: {text_metrics['total_raw_detections']} / {text_metrics['total_clean_detections']}")
    print(
        "Prompt-free raw/clean: "
        f"{prompt_free_metrics['total_raw_detections']} / {prompt_free_metrics['total_clean_detections']}"
    )
    print(f"Text unique clean labels: {text_metrics['unique_clean_label_count']}")
    print(f"Prompt-free unique clean labels: {prompt_free_metrics['unique_clean_label_count']}")
    print(f"Visualization pairs saved: {len(saved_visualizations)}")
    print(f"Output: {experiment_dir}")


if __name__ == "__main__":
    main()
