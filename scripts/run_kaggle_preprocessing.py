from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_step(name: str, cmd: list[str], *, cwd: Path = PROJECT_ROOT, required: bool = True) -> dict[str, object]:
    print("\n" + "=" * 88)
    print(f"[KAGGLE] {name}")
    print(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))
    print("=" * 88)
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd))
    elapsed = round(time.time() - started, 2)
    record = {
        "name": name,
        "returncode": int(proc.returncode),
        "elapsed_sec": elapsed,
        "command": cmd,
        "required": required,
    }
    if proc.returncode != 0 and required:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if proc.returncode != 0:
        print(f"[KAGGLE][WARN] Optional step failed: {name} rc={proc.returncode}")
    return record


def discover_video_ids(video_root: Path, limit: int | None) -> list[str]:
    videos = sorted(path.stem for path in video_root.glob("*.mp4") if not path.name.startswith("."))
    if limit is not None:
        videos = videos[: max(0, limit)]
    return videos


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def make_keyframe_config(data_root: Path, output_root: Path) -> Path:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "keyframe_v2.yaml")
    paths = cfg.setdefault("paths", {})
    paths["dataset_root"] = str(data_root)
    paths["video_root"] = str(data_root / "Videos_L21_a" / "video")
    paths["btc_keyframe_root"] = str(data_root / "Keyframes_L21" / "keyframes")
    paths["btc_mapping_root"] = str(data_root / "map-keyframes-aic25-b1" / "map-keyframes")
    paths["clip_feature_root"] = str(data_root / "clip-features-32-aic25-b1" / "clip-features-32")
    paths["model_cache"] = str(PROJECT_ROOT / ".model_cache")
    paths["hf_cache"] = str(PROJECT_ROOT / ".model_cache" / "huggingface")
    cfg.setdefault("shot_detection", {})["require_transnetv2"] = False
    clip_cfg = cfg.setdefault("clip", {})
    clip_cfg["pretrained"] = ""
    clip_cfg["download_root"] = ".model_cache"
    clip_cfg["open_clip_weights"] = (
        ".model_cache/models--timm--vit_base_patch32_clip_224.openai/"
        "snapshots/*/open_clip_model.safetensors"
    )
    return write_yaml(cfg, output_root / "kaggle_configs" / "keyframe_v2.yaml")


def make_ocr_temporal_config(ocr_v2_root: Path, keyframe_root: Path, ocr_temporal_root: Path, output_root: Path) -> Path:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "ocr_temporal_v3.yaml")
    paths = cfg.setdefault("paths", {})
    paths["input_dir"] = str(ocr_v2_root)
    paths["selected_root"] = str(keyframe_root)
    paths["output_dir"] = str(ocr_temporal_root)
    return write_yaml(cfg, output_root / "kaggle_configs" / "ocr_temporal_v3.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kaggle L21 preprocessing stages with smoke/full modes.")
    parser.add_argument("--data-root", default=os.environ.get("AIC_DATA_ROOT", "/kaggle/input/datasets/nadkli/dataset-aic"))
    parser.add_argument("--output-root", default=os.environ.get("AIC_OUTPUT_ROOT", "/kaggle/working/artifacts"))
    parser.add_argument("--smoke-video-count", type=int, default=int(os.environ.get("AIC_SMOKE_VIDEO_COUNT", "1")))
    parser.add_argument("--full", action="store_true", help="Process every discovered video instead of the smoke limit.")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--skip-keyframes", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-objects", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=os.environ.get("AIC_DEVICE", "cuda"))
    parser.add_argument("--ocr-device", default=os.environ.get("AIC_OCR_DEVICE", "auto"), choices=["auto", "cpu", "cuda"])
    parser.add_argument("--asr-compute-type", default=os.environ.get("AIC_ASR_COMPUTE_TYPE", "float16"))
    parser.add_argument("--object-visualization-limit", type=int, default=int(os.environ.get("AIC_OBJECT_VISUALIZATION_LIMIT", "-1")))
    parser.add_argument("--allow-missing-package", action="store_true")
    parser.add_argument("--skip-visual-report", action="store_true")
    parser.add_argument("--visual-report-limit", type=int, default=int(os.environ.get("AIC_VISUAL_REPORT_LIMIT", "12")))
    parser.add_argument("--visual-report-video-clips", type=int, default=int(os.environ.get("AIC_VISUAL_REPORT_VIDEO_CLIPS", "1")))
    parser.add_argument("--manifest-limit", type=int, default=20)
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root)
    video_root = data_root / "Videos_L21_a" / "video"
    keyframe_root = output_root / "keyframe_v2_full"
    ocr_v2_root = output_root / "ocr_v2_selected_keyframes"
    ocr_temporal_root = output_root / "ocr_temporal_v3_full_tracking"
    ocr_index_root = output_root / "indexes" / "ocr_temporal_v3_full_tracking"
    asr_root = output_root / "asr"
    audio_root = output_root / "audio"
    object_root = keyframe_root / "object_v2"
    object_index_root = keyframe_root / "indexes" / "object"
    index_root = output_root / "indexes"

    os.environ["AIC_DATA_ROOT"] = str(data_root)
    os.environ["AIC_OUTPUT_ROOT"] = str(output_root)
    os.environ["AIC_KEYFRAME_OUTPUT_ROOT"] = str(keyframe_root)
    os.environ["AIC_OCR_V2_OUTPUT_ROOT"] = str(ocr_v2_root)
    os.environ["AIC_OCR_TEMPORAL_OUTPUT_ROOT"] = str(ocr_temporal_root)
    os.environ["AIC_OCR_INDEX_OUTPUT_ROOT"] = str(ocr_index_root)
    os.environ["AIC_OCR_OUTPUT_ROOT"] = str(ocr_temporal_root)
    os.environ["AIC_ASR_OUTPUT_ROOT"] = str(asr_root)
    os.environ["AIC_OBJECT_OUTPUT_ROOT"] = str(object_root)
    os.environ["AIC_INDEX_OUTPUT_ROOT"] = str(index_root)
    os.environ["AIC_ALLOW_HISTDIFF_FALLBACK"] = "1"

    for path in (output_root, keyframe_root, ocr_v2_root, ocr_temporal_root, ocr_index_root, asr_root, audio_root, index_root):
        path.mkdir(parents=True, exist_ok=True)

    if not video_root.is_dir():
        raise FileNotFoundError(f"Missing Kaggle video root: {video_root}")

    limit = None if args.full else args.smoke_video_count
    video_ids = discover_video_ids(video_root, limit)
    if not video_ids:
        raise FileNotFoundError(f"No .mp4 videos found in {video_root}")

    print(json.dumps({
        "mode": "full" if args.full else "smoke",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "video_count": len(video_ids),
        "video_ids": video_ids[:20],
    }, indent=2, ensure_ascii=False))

    records: list[dict[str, object]] = []
    py = sys.executable
    keyframe_config = make_keyframe_config(data_root, output_root)
    generated_keyframe_cfg = load_yaml(keyframe_config)
    print(json.dumps({
        "generated_keyframe_config": str(keyframe_config),
        "require_transnetv2": generated_keyframe_cfg.get("shot_detection", {}).get("require_transnetv2"),
        "clip_pretrained": generated_keyframe_cfg.get("clip", {}).get("pretrained"),
        "clip_open_clip_weights": generated_keyframe_cfg.get("clip", {}).get("open_clip_weights"),
    }, indent=2, ensure_ascii=False))
    ocr_temporal_config = make_ocr_temporal_config(ocr_v2_root, keyframe_root, ocr_temporal_root, output_root)

    if not args.skip_assets:
        records.append(run_step("prepare assets/checkpoints", [py, "scripts/prepare_kaggle_assets.py"]))

    if not args.skip_keyframes:
        cmd = [
            py,
            "scripts/run_keyframe_v2_full.py",
            "--video-root",
            str(video_root),
            "--config",
            str(keyframe_config),
            "--output",
            str(keyframe_root),
            "--limit",
            str(len(video_ids)),
        ]
        if args.force:
            cmd.append("--force")
        records.append(run_step("keyframe v2", cmd))
        global_map_path = keyframe_root / "indexes" / "keyframe_v2_global_map.parquet"
        if not global_map_path.is_file():
            raise FileNotFoundError(f"Keyframe V2 did not create global map: {global_map_path}")
        try:
            import pandas as pd

            global_rows = len(pd.read_parquet(global_map_path))
        except Exception as exc:
            raise RuntimeError(f"Cannot read Keyframe V2 global map: {global_map_path}: {exc}") from exc
        if global_rows <= 0:
            raise RuntimeError(
                f"Keyframe V2 produced 0 keyframes at {global_map_path}; "
                "stopping before Visual/Object/OCR/ASR packaging."
            )
        records.append(run_step(
            "visual clip faiss v2",
            [
                py,
                "scripts/build_keyframe_v2_visual_index.py",
                "--global-map",
                str(keyframe_root / "indexes" / "keyframe_v2_global_map.parquet"),
                "--config",
                str(keyframe_config),
                "--output-dir",
                str(keyframe_root / "indexes" / "visual"),
            ],
        ))

    if not args.skip_objects:
        cmd = [
            py,
            "scripts/run_keyframe_v2_object_index.py",
            "--global-map",
            str(keyframe_root / "indexes" / "keyframe_v2_global_map.parquet"),
            "--output-root",
            str(object_root),
            "--index-output",
            str(object_index_root),
            "--cache-dir",
            str(PROJECT_ROOT / ".model_cache"),
            "--device",
            args.device,
            "--visualization-limit",
            str(args.object_visualization_limit),
        ]
        if not args.full:
            cmd.extend(["--limit-frames", os.environ.get("AIC_OBJECT_SMOKE_FRAME_LIMIT", "100")])
        if args.force:
            cmd.append("--force")
        records.append(run_step("object v2 yoloe", cmd))

    if not args.skip_ocr:
        cmd = [
            py,
            "scripts/run_ocr_v2_selected_keyframes.py",
            "--selected-root",
            str(keyframe_root),
            "--output-dir",
            str(ocr_v2_root),
            "--device",
            args.ocr_device,
        ]
        for video_id in video_ids:
            cmd.extend(["--video-id", video_id])
        records.append(run_step("ocr v2 selected keyframes", cmd))

        cmd = [
            py,
            "scripts/build_ocr_temporal_v3.py",
            "--input-dir",
            str(ocr_v2_root),
            "--selected-root",
            str(keyframe_root),
            "--output-dir",
            str(ocr_temporal_root),
            "--config",
            str(ocr_temporal_config),
        ]
        for video_id in video_ids:
            cmd.extend(["--video-id", video_id])
        records.append(run_step("ocr temporal v3 tracking", cmd))

        records.append(run_step(
            "ocr temporal v3 faiss",
            [
                py,
                "scripts/build_ocr_temporal_v3_index.py",
                "--documents",
                str(ocr_temporal_root / "l21_ocr_documents.parquet"),
                "--output-dir",
                str(ocr_index_root),
                "--device",
                args.ocr_device,
            ],
        ))

    if not args.skip_asr:
        if shutil.which("ffmpeg") is None:
            print("[KAGGLE][WARN] ffmpeg not found on PATH; ASR audio extraction will likely fail.")
        cmd = [
            py,
            "scripts/run_batch_asr_whisper.py",
            "--video-dir",
            str(video_root),
            "--output-dir",
            str(asr_root),
            "--audio-dir",
            str(audio_root),
            "--device",
            args.device,
            "--compute-type",
            args.asr_compute_type,
            "--index-output-dir",
            str(index_root / "asr_v3"),
        ]
        if args.force:
            cmd.append("--overwrite")
        if not args.full:
            cmd.extend(["--limit", str(len(video_ids))])
        records.append(run_step("asr faster-whisper v3", cmd))

    if not args.skip_package:
        cmd = [
            py,
            "scripts/validate_package_kaggle_outputs.py",
            "--output-root",
            str(output_root),
        ]
        if args.allow_missing_package:
            cmd.append("--allow-missing")
        records.append(run_step("validate and package outputs", cmd))

    if not args.skip_visual_report:
        records.append(run_step(
            "visual html report",
            [
                py,
                "scripts/build_kaggle_visual_report.py",
                "--output-root",
                str(output_root),
                "--data-root",
                str(data_root),
                "--limit",
                str(args.visual_report_limit),
                "--video-preview-count",
                str(args.visual_report_video_clips),
            ],
            required=False,
        ))

    records.append(run_step(
        "manifest 20 outputs/group",
        [py, "scripts/kaggle_output_manifest.py", "--limit", str(args.manifest_limit), "--output-root", str(output_root)],
    ))

    report_path = output_root / "kaggle_preprocessing_run_report.json"
    report_path.write_text(json.dumps({
        "mode": "full" if args.full else "smoke",
        "video_ids": video_ids,
        "steps": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[KAGGLE] Wrote run report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
