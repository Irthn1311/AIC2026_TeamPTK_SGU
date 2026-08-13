from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

from _bootstrap import PROJECT_ROOT
from src.preprocessing.yoloe_hybrid_filter import (
    THRESHOLD_ANALYSIS_VALUES,
    build_hybrid_outputs,
    get_hybrid_config,
    threshold_analysis,
)
from src.preprocessing.yoloe_postprocess import get_cleanup_config


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_experiment_records(experiment_root: Path, video_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    video_dir = experiment_root / video_id
    text_path = video_dir / "text_prompt.json"
    prompt_free_path = video_dir / "prompt_free.json"
    if not text_path.exists() or not prompt_free_path.exists():
        raise FileNotFoundError(
            "Missing YOLOE V3 experiment JSON. Run first:\n"
            f"  E:\\conda_envs\\aic2026\\python.exe scripts\\compare_yoloe_modes.py --video-id {video_id}\n"
            f"Expected:\n  {text_path}\n  {prompt_free_path}"
        )

    text_records = load_json(text_path)
    prompt_free_records = load_json(prompt_free_path)
    return text_records, prompt_free_records


def attach_image_sizes(records: list[dict[str, Any]]) -> None:
    size_cache: dict[str, tuple[int, int]] = {}
    for record in records:
        image_path = str(record.get("keyframe_path") or "")
        if not image_path:
            continue
        if image_path not in size_cache:
            image = cv2.imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            size_cache[image_path] = (width, height)
        width, height = size_cache[image_path]
        record["image_width"] = width
        record["image_height"] = height


def choose_visualization_frames(hybrid_records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in hybrid_records:
        prompt_free_added = [item for item in record.get("hybrid_detections", []) if item.get("source") == "prompt_free"]
        if not prompt_free_added:
            continue
        candidates.append(
            {
                "video_id": record.get("video_id"),
                "frame_id": record.get("frame_id"),
                "timestamp": record.get("timestamp"),
                "keyframe_name": record.get("keyframe_name"),
                "keyframe_path": record.get("keyframe_path"),
                "prompt_free_added_count": len(prompt_free_added),
                "hybrid_detection_count": len(record.get("hybrid_detections", [])),
                "prompt_free_added_labels": sorted(Counter(item["label"] for item in prompt_free_added).items()),
            }
        )
    candidates.sort(
        key=lambda item: (item["prompt_free_added_count"], item["hybrid_detection_count"], item["frame_id"]),
        reverse=True,
    )
    return candidates[:limit]


def save_hybrid_visualizations(
    selected_frames: list[dict[str, Any]],
    hybrid_records: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image in output_dir.glob("*.jpg"):
        old_image.unlink()

    records_by_name = {record["keyframe_name"]: record for record in hybrid_records}
    saved: list[dict[str, Any]] = []
    for frame in selected_frames:
        record = records_by_name[frame["keyframe_name"]]
        output_path = output_dir / f"{Path(frame['keyframe_name']).stem}_hybrid.jpg"
        save_source_visualization(record["keyframe_path"], record["hybrid_detections"], output_path)
        saved.append({**frame, "visualization": str(output_path)})
    return saved


def save_source_visualization(image_path: str | Path, detections: list[dict[str, Any]], output_path: str | Path) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image for visualization: {image_path}")

    height, width = image.shape[:2]
    thickness = max(2, int(round(min(width, height) / 360)))
    font_scale = max(0.45, min(width, height) / 900)

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        source = str(detection.get("source") or "text")
        color = (40, 180, 80) if source == "text" else (30, 130, 240)
        caption = f"{detection['label']} {float(detection['confidence']):.2f} {source}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        y_text = max(text_h + baseline + 4, y1)
        cv2.rectangle(
            image,
            (x1, y_text - text_h - baseline - 6),
            (min(width - 1, x1 + text_w + 8), y_text + baseline),
            color,
            -1,
        )
        cv2.putText(
            image,
            caption,
            (x1 + 4, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise IOError(f"Failed to write visualization: {output_path}")
    return output_path


def update_threshold_analysis(experiment_root: Path, output_path: Path, video_ids: list[str]) -> dict[str, Any]:
    records_by_video: dict[str, list[dict[str, Any]]] = {}
    for video_id in video_ids:
        prompt_free_path = experiment_root / video_id / "prompt_free.json"
        if prompt_free_path.exists():
            records_by_video[video_id] = load_json(prompt_free_path)
    analysis = threshold_analysis(records_by_video, THRESHOLD_ANALYSIS_VALUES)
    write_json(output_path, analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build YOLOE hybrid detections from existing text-prompt and prompt-free experiment JSON."
    )
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument(
        "--classes-file",
        default=str(PROJECT_ROOT / "configs" / "yoloe_objects.yaml"),
        help="YAML with cleanup and hybrid thresholds.",
    )
    parser.add_argument(
        "--experiment-root",
        default=str(PROJECT_ROOT / "outputs" / "yoloe" / "experiments"),
        help="Directory containing V3 text_prompt.json and prompt_free.json per video.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "yoloe" / "hybrid"),
    )
    parser.add_argument(
        "--visualization-limit",
        type=int,
        help="Number of hybrid debug images. Defaults to config hybrid.visualization_limit.",
    )
    parser.add_argument(
        "--threshold-analysis-videos",
        nargs="+",
        default=["L21_V001", "L21_V002"],
        help="Existing V3 prompt-free JSONs to include in threshold_analysis.json.",
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.classes_file))
    cleanup_config = get_cleanup_config(config)
    hybrid_config = get_hybrid_config(config)
    visualization_limit = (
        args.visualization_limit
        if args.visualization_limit is not None
        else int(hybrid_config.get("visualization_limit", 20))
    )

    experiment_root = Path(args.experiment_root)
    output_root = Path(args.output_root)
    text_records, prompt_free_records = load_experiment_records(experiment_root, args.video_id)
    attach_image_sizes(text_records)
    attach_image_sizes(prompt_free_records)

    hybrid_records, stats, audit_records = build_hybrid_outputs(
        text_records,
        prompt_free_records,
        hybrid_config,
        cleanup_config,
    )
    selected_frames = choose_visualization_frames(hybrid_records, visualization_limit)
    saved_visualizations = save_hybrid_visualizations(
        selected_frames,
        hybrid_records,
        output_root / "visualization" / args.video_id,
    )
    stats["selected_visualization_frames"] = saved_visualizations
    stats["hybrid_config"] = hybrid_config
    stats["classes_file"] = args.classes_file
    stats["source_experiment_root"] = str(experiment_root / args.video_id)

    write_json(output_root / f"{args.video_id}.json", hybrid_records)
    write_json(output_root / "stats" / f"{args.video_id}.json", stats)
    write_json(output_root / "audit" / f"{args.video_id}.json", audit_records)
    update_threshold_analysis(
        experiment_root,
        output_root / "threshold_analysis.json",
        args.threshold_analysis_videos,
    )

    print(f"Video: {args.video_id}")
    print(f"Frames: {stats['total_frames']}")
    print(f"Text clean detections: {stats['text_clean_detections']}")
    print(f"Prompt-free clean detections: {stats['prompt_free_clean_detections']}")
    print(f"Prompt-free added after filter: {stats['prompt_free_detections_after_filter']}")
    print(f"Final hybrid detections: {stats['final_hybrid_detections']}")
    print(f"Prompt-free removed by reason: {stats['prompt_free_removed_by_reason']}")
    print(f"Visualization saved: {len(saved_visualizations)}")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
