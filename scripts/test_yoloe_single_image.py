from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from _bootstrap import PROJECT_ROOT
from src.preprocessing.yoloe_detector import DEFAULT_CACHE_ROOT, YOLOEDetector, save_detection_visualization
from src.preprocessing.yoloe_postprocess import clean_detections, flatten_prompt_classes, get_cleanup_config


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
    parser = argparse.ArgumentParser(description="Run YOLOE text-prompt detection on one keyframe.")
    parser.add_argument("--image", required=True, help="Path to a keyframe image.")
    parser.add_argument("--classes", nargs="+", help="Text-prompt classes to detect.")
    parser.add_argument("--classes-file", default=str(PROJECT_ROOT / "configs" / "yoloe_objects.yaml"))
    parser.add_argument("--model", help="YOLOE model name or local .pt path.")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", help="cuda:0 or cpu. Defaults to CUDA when available.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "yoloe" / "test"))
    parser.add_argument("--visualize-raw", action="store_true", help="Draw raw YOLOE detections instead of clean detections.")
    args = parser.parse_args()

    config = load_config(args.classes_file)
    model_cfg = config.get("model") or {}
    cleanup_cfg = get_cleanup_config(config)
    classes = resolve_classes(args, config)

    detector = YOLOEDetector(
        model_name=args.model or model_cfg.get("name", "yoloe-26s-seg.pt"),
        cache_dir=args.cache_dir,
        conf=args.conf if args.conf is not None else model_cfg.get("conf", 0.25),
        iou=args.iou if args.iou is not None else model_cfg.get("iou", 0.70),
        imgsz=args.imgsz if args.imgsz is not None else model_cfg.get("imgsz", 640),
        device=args.device,
    )

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_detections = detector.detect(image_path=image_path, classes=classes)
    detections, cleanup_stats = clean_detections(raw_detections, cleanup_cfg)
    json_path = output_dir / f"{image_path.stem}_detections.json"
    vis_path = output_dir / f"{image_path.stem}_yoloe.jpg"
    vis_detections = raw_detections if args.visualize_raw else detections

    output = {
        "image_path": str(image_path),
        "classes": classes,
        "raw_detections": raw_detections,
        "detections": detections,
        "cleanup_stats": cleanup_stats,
    }
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    save_detection_visualization(image_path, vis_detections, vis_path)

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Device: {detector.runtime_device}")
    print(f"JSON: {json_path}")
    print(f"Visualization: {vis_path}")


if __name__ == "__main__":
    main()
