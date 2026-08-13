from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT
from src.preprocessing.model_assets import ensure_yoloe_weights
from src.preprocessing.yoloe_detector import DEFAULT_CACHE_ROOT, YOLOEDetector, save_detection_visualization
from src.preprocessing.yoloe_postprocess import clean_detections, flatten_prompt_classes, get_cleanup_config, summarize_video_records
from src.retrieval.mapping_loader import load_keyframe_mapping


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_classes(args: argparse.Namespace, config: dict) -> list[str]:
    if args.classes:
        return args.classes
    classes = flatten_prompt_classes(config.get("classes") or [])
    if not classes:
        raise ValueError("No classes configured. Use --classes or --classes-file.")
    return classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOE text-prompt detection on existing L21 keyframes.")
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "datasets_L21"))
    parser.add_argument("--classes", nargs="+", help="Text-prompt classes to detect.")
    parser.add_argument("--classes-file", default=str(PROJECT_ROOT / "configs" / "yoloe_objects.yaml"))
    parser.add_argument("--model", help="YOLOE model name or local .pt path.")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", help="cuda:0 or cpu. Defaults to CUDA when available.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "yoloe"))
    parser.add_argument("--max-frames", type=int, help="Optional debug limit. Omit for full video.")
    parser.add_argument("--visualization-limit", type=int, default=20, help="Max visualization images. Use -1 for no limit.")
    parser.add_argument("--visualize-raw", action="store_true", help="Draw raw YOLOE detections instead of clean detections.")
    parser.add_argument("--visualize-empty", action="store_true", help="Also save visualization images for frames with no detections.")
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

    detector = YOLOEDetector(
        model_name=str(ensure_yoloe_weights(args.model or model_cfg.get("name", "yoloe-26s-seg.pt"), args.cache_dir)),
        cache_dir=args.cache_dir,
        conf=args.conf if args.conf is not None else model_cfg.get("conf", 0.25),
        iou=args.iou if args.iou is not None else model_cfg.get("iou", 0.70),
        imgsz=args.imgsz if args.imgsz is not None else model_cfg.get("imgsz", 640),
        device=args.device,
    )

    mapping_df = load_keyframe_mapping(mapping_path, keyframe_root)
    mapping_df = mapping_df.sort_values("feature_index").reset_index(drop=True)
    if args.max_frames:
        mapping_df = mapping_df.head(args.max_frames)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.video_id}.json"
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"{args.video_id}.json"
    vis_dir = output_dir / "visualization" / args.video_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    errors: list[dict] = []
    visualizations_saved = 0

    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc=f"YOLOE {args.video_id}"):
        image_path = Path(row["keyframe_path"])
        frame_id = int(row["frame_idx"])
        timestamp = float(row["timestamp_seconds"])
        try:
            raw_detections = detector.detect(image_path=image_path, classes=classes)
            detections, _ = clean_detections(raw_detections, cleanup_cfg)
        except Exception as exc:
            raw_detections = []
            detections = []
            errors.append(
                {
                    "video_id": args.video_id,
                    "frame_id": frame_id,
                    "keyframe_name": str(row["keyframe_name"]),
                    "error": str(exc),
                }
            )

        records.append(
            {
                "video_id": args.video_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "raw_detections": raw_detections,
                "detections": detections,
            }
        )

        should_visualize_frame = bool(detections) or args.visualize_empty
        under_visualization_limit = args.visualization_limit < 0 or visualizations_saved < args.visualization_limit
        if should_visualize_frame and under_visualization_limit:
            vis_path = vis_dir / f"{Path(row['keyframe_name']).stem}_yoloe.jpg"
            try:
                vis_detections = raw_detections if args.visualize_raw else detections
                save_detection_visualization(image_path, vis_detections, vis_path)
                visualizations_saved += 1
            except Exception as exc:
                errors.append(
                    {
                        "video_id": args.video_id,
                        "frame_id": frame_id,
                        "keyframe_name": str(row["keyframe_name"]),
                        "error": f"visualization: {exc}",
                    }
                )

    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    stats = summarize_video_records(records)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    errors_path = output_dir / f"{args.video_id}_errors.json"
    if errors:
        errors_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    elif errors_path.exists():
        errors_path.unlink()

    print(f"Device: {detector.runtime_device}")
    print(f"Frames processed: {len(records)}")
    print(f"Frames with detections: {sum(1 for item in records if item['detections'])}")
    print(f"Raw detections: {stats['total_detections_raw']}")
    print(f"Clean detections: {stats['total_detections_clean']}")
    print(f"Duplicates removed: {stats['detections_removed_as_duplicate']}")
    print(f"Visualization images saved: {visualizations_saved}")
    print(f"JSON: {json_path}")
    print(f"Stats: {stats_path}")
    if errors:
        print(f"Errors: {errors_path}")


if __name__ == "__main__":
    main()
