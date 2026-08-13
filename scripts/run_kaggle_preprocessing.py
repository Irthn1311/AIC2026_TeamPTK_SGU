from __future__ import annotations

import argparse
import glob
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


def run_parallel_steps(name: str, jobs: list[dict[str, object]], *, cwd: Path = PROJECT_ROOT, required: bool = True) -> dict[str, object]:
    print("\n" + "=" * 88)
    print(f"[KAGGLE] {name} ({len(jobs)} parallel jobs)")
    print("=" * 88)
    started = time.time()
    running = []
    for idx, job in enumerate(jobs, start=1):
        cmd = [str(part) for part in job["cmd"]]
        env_updates = {str(k): str(v) for k, v in dict(job.get("env_updates") or {}).items()}
        env = os.environ.copy()
        env.update(env_updates)
        print(f"[{idx}/{len(jobs)}] " + " ".join(f'"{part}"' if " " in part else part for part in cmd))
        if env_updates:
            print(f"[{idx}/{len(jobs)}] env " + " ".join(f"{key}={value}" for key, value in sorted(env_updates.items())))
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)
        running.append((job, proc))
    job_records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for job, proc in running:
        returncode = int(proc.wait())
        job_record = {
            "name": str(job.get("name", "job")),
            "returncode": returncode,
            "command": [str(part) for part in job["cmd"]],
            "env_updates": dict(job.get("env_updates") or {}),
        }
        job_records.append(job_record)
        if returncode != 0:
            failures.append(job_record)
    elapsed = round(time.time() - started, 2)
    record = {
        "name": name,
        "returncode": 0 if not failures else int(failures[0]["returncode"]),
        "elapsed_sec": elapsed,
        "required": required,
        "jobs": job_records,
    }
    if failures and required:
        raise subprocess.CalledProcessError(int(failures[0]["returncode"]), failures[0]["command"])
    if failures:
        print(f"[KAGGLE][WARN] Optional parallel step failed: {name} failures={len(failures)}")
    return record


def split_evenly(items: list[str], parts: int) -> list[list[str]]:
    parts = max(1, min(parts, len(items)))
    return [items[idx::parts] for idx in range(parts) if items[idx::parts]]


def resolve_gpu_devices(requested: str, device: str) -> list[str]:
    if str(device).lower() == "cpu":
        return []
    if requested.strip().lower() != "auto":
        return [item.strip() for item in requested.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    return [str(idx) for idx in range(count)]


def natural_video_key(path: Path | str) -> tuple[str, int]:
    stem = Path(path).stem
    try:
        prefix, number = stem.rsplit("_V", 1)
        return prefix, int(number)
    except Exception:
        return stem, 0


def discover_video_roots(data_root: Path, video_root_glob: str) -> list[Path]:
    pattern = video_root_glob.strip()
    if not pattern:
        pattern = "Videos_L21_a/video"
    candidate = Path(pattern).expanduser()
    if candidate.is_absolute():
        if any(char in pattern for char in "*?["):
            roots = [Path(path) for path in sorted(glob.glob(pattern))]
        else:
            roots = [candidate]
    else:
        roots = sorted(data_root.glob(pattern))
    return [root for root in roots if root.is_dir()]


def discover_video_paths(video_roots: list[Path], limit: int | None) -> list[Path]:
    videos: dict[str, Path] = {}
    for video_root in video_roots:
        for path in sorted(video_root.glob("*.mp4"), key=natural_video_key):
            if path.name.startswith("."):
                continue
            videos.setdefault(path.stem, path)
    discovered = sorted(videos.values(), key=natural_video_key)
    if limit is not None:
        discovered = discovered[: max(0, limit)]
    return discovered


def append_repeated_arg(cmd: list[str], flag: str, values: list[Path | str]) -> None:
    for value in values:
        cmd.extend([flag, str(value)])


def resolve_btc_keyframe_root(data_root: Path) -> Path:
    for candidate in [
        data_root / "keyframes",
        data_root / "Keyframes_L21" / "keyframes",
    ]:
        if candidate.is_dir():
            return candidate
    return data_root / "keyframes"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def make_keyframe_config(data_root: Path, output_root: Path, video_roots: list[Path], visual_batch_size: int) -> Path:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "keyframe_v2.yaml")
    paths = cfg.setdefault("paths", {})
    paths["dataset_root"] = str(data_root)
    paths["video_root"] = str(video_roots[0] if video_roots else data_root / "Videos_L21_a" / "video")
    paths["btc_keyframe_root"] = str(resolve_btc_keyframe_root(data_root))
    paths["btc_mapping_root"] = str(data_root / "map-keyframes-aic25-b1" / "map-keyframes")
    paths["clip_feature_root"] = str(data_root / "clip-features-32-aic25-b1" / "clip-features-32")
    paths["model_cache"] = str(PROJECT_ROOT / ".model_cache")
    paths["hf_cache"] = str(PROJECT_ROOT / ".model_cache" / "huggingface")
    shot_cfg = cfg.setdefault("shot_detection", {})
    shot_cfg["require_transnetv2"] = False
    shot_cfg["use_histdiff_only"] = True
    shot_cfg["backend"] = "histdiff"
    shot_cfg["fallback_sampling_mode"] = "sequential"
    candidate_cfg = cfg.setdefault("candidates", {})
    candidate_cfg["save_candidate_frames"] = False
    clip_cfg = cfg.setdefault("clip", {})
    clip_cfg["pretrained"] = ""
    clip_cfg["download_root"] = ".model_cache"
    clip_cfg["open_clip_weights"] = (
        ".model_cache/models--timm--vit_base_patch32_clip_224.openai/"
        "snapshots/*/open_clip_model.safetensors"
    )
    clip_cfg["batch_size"] = int(visual_batch_size)
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
    parser.add_argument(
        "--video-root-glob",
        default=os.environ.get("AIC_VIDEO_ROOT_GLOB", "Videos_L21_a/video"),
        help='Video root glob under --data-root, e.g. "Videos_L*_*/video" for all Kaggle folders.',
    )
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
    parser.add_argument("--gpu-devices", default=os.environ.get("AIC_GPU_DEVICES", "auto"), help="Comma-separated GPU ids for sharded full runs, or auto.")
    parser.add_argument("--parallel-gpu-workers", type=int, default=int(os.environ.get("AIC_PARALLEL_GPU_WORKERS", "0")), help="0 means use every visible GPU in --full mode.")
    parser.add_argument("--visual-batch-size", type=int, default=int(os.environ.get("AIC_VISUAL_BATCH_SIZE", "128")))
    parser.add_argument("--ocr-index-batch-size", type=int, default=int(os.environ.get("AIC_OCR_INDEX_BATCH_SIZE", "256")))
    parser.add_argument("--asr-index-batch-size", type=int, default=int(os.environ.get("AIC_ASR_INDEX_BATCH_SIZE", "256")))
    parser.add_argument("--object-visualization-limit", type=int, default=int(os.environ.get("AIC_OBJECT_VISUALIZATION_LIMIT", "-1")))
    parser.add_argument("--allow-missing-package", action="store_true")
    parser.add_argument("--skip-visual-report", action="store_true")
    parser.add_argument("--visual-report-limit", type=int, default=int(os.environ.get("AIC_VISUAL_REPORT_LIMIT", "12")))
    parser.add_argument("--visual-report-video-clips", type=int, default=int(os.environ.get("AIC_VISUAL_REPORT_VIDEO_CLIPS", "1")))
    parser.add_argument("--manifest-limit", type=int, default=20)
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root)
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
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

    for path in (output_root, keyframe_root, ocr_v2_root, ocr_temporal_root, ocr_index_root, asr_root, audio_root, index_root):
        path.mkdir(parents=True, exist_ok=True)

    limit = None if args.full else args.smoke_video_count
    video_roots = discover_video_roots(data_root, args.video_root_glob)
    if not video_roots:
        raise FileNotFoundError(f"No Kaggle video roots matched {args.video_root_glob!r} under {data_root}")
    video_paths = discover_video_paths(video_roots, limit)
    video_ids = [path.stem for path in video_paths]
    if not video_ids:
        raise FileNotFoundError(f"No .mp4 videos found in roots: {[str(root) for root in video_roots]}")
    gpu_devices = resolve_gpu_devices(args.gpu_devices, args.device)
    if args.parallel_gpu_workers > 0:
        gpu_devices = gpu_devices[: args.parallel_gpu_workers]
    sharded_gpu_devices = gpu_devices if len(gpu_devices) > 1 and len(video_ids) > 1 else []

    print(json.dumps({
        "mode": "full" if args.full else "smoke",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "video_root_glob": args.video_root_glob,
        "video_roots": [str(root) for root in video_roots],
        "video_count": len(video_ids),
        "video_ids": video_ids[:20],
        "gpu_devices": gpu_devices,
        "sharded_gpu_devices": sharded_gpu_devices,
        "batch_sizes": {
            "visual": args.visual_batch_size,
            "ocr_index": args.ocr_index_batch_size,
            "asr_index": args.asr_index_batch_size,
        },
    }, indent=2, ensure_ascii=False))

    records: list[dict[str, object]] = []
    py = sys.executable
    keyframe_config = make_keyframe_config(data_root, output_root, video_roots, args.visual_batch_size)
    generated_keyframe_cfg = load_yaml(keyframe_config)
    print(json.dumps({
        "generated_keyframe_config": str(keyframe_config),
        "require_transnetv2": generated_keyframe_cfg.get("shot_detection", {}).get("require_transnetv2"),
        "clip_pretrained": generated_keyframe_cfg.get("clip", {}).get("pretrained"),
        "clip_open_clip_weights": generated_keyframe_cfg.get("clip", {}).get("open_clip_weights"),
        "clip_batch_size": generated_keyframe_cfg.get("clip", {}).get("batch_size"),
    }, indent=2, ensure_ascii=False))
    ocr_temporal_config = make_ocr_temporal_config(ocr_v2_root, keyframe_root, ocr_temporal_root, output_root)

    if not args.skip_assets:
        records.append(run_step("prepare assets/checkpoints", [py, "scripts/prepare_kaggle_assets.py"]))

    if not args.skip_keyframes:
        keyframe_cmd = [
            py,
            "scripts/run_keyframe_v2_full.py",
            "--config",
            str(keyframe_config),
            "--output",
            str(keyframe_root),
            "--limit",
            str(len(video_ids)),
        ]
        append_repeated_arg(keyframe_cmd, "--video-root", video_roots)
        if args.force:
            keyframe_cmd.append("--force")
        if sharded_gpu_devices:
            jobs = []
            for gpu_id, shard_video_ids in zip(sharded_gpu_devices, split_evenly(video_ids, len(sharded_gpu_devices))):
                shard_cmd = [
                    py,
                    "scripts/run_keyframe_v2_full.py",
                    "--config",
                    str(keyframe_config),
                    "--output",
                    str(keyframe_root),
                    "--no-aggregate",
                ]
                append_repeated_arg(shard_cmd, "--video-root", video_roots)
                for video_id in shard_video_ids:
                    shard_cmd.extend(["--video-id", video_id])
                if args.force:
                    shard_cmd.append("--force")
                jobs.append({
                    "name": f"keyframe_gpu{gpu_id}",
                    "cmd": shard_cmd,
                    "env_updates": {
                        "CUDA_VISIBLE_DEVICES": str(gpu_id),
                        "AIC_FAST_SHOT_DETECTION": "1",
                        "AIC_ALLOW_HISTDIFF_FALLBACK": "1",
                    },
                })
            records.append(run_parallel_steps("keyframe v2 sharded", jobs))
            records.append(run_step(
                "keyframe v2 aggregate map",
                [
                    py,
                    "scripts/run_keyframe_v2_full.py",
                    "--config",
                    str(keyframe_config),
                    "--output",
                    str(keyframe_root),
                    "--aggregate-only",
                ],
            ))
        else:
            records.append(run_step("keyframe v2", keyframe_cmd))
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
                "--batch-size",
                str(args.visual_batch_size),
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
        if sharded_gpu_devices:
            jobs = []
            for gpu_id, shard_video_ids in zip(sharded_gpu_devices, split_evenly(video_ids, len(sharded_gpu_devices))):
                shard_map = output_root / "tmp" / "shards" / f"object_gpu{gpu_id}_global_map.parquet"
                shard_map.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import pandas as pd

                    global_df = pd.read_parquet(keyframe_root / "indexes" / "keyframe_v2_global_map.parquet")
                    global_df[global_df["video_id"].astype(str).isin(shard_video_ids)].to_parquet(shard_map, index=False)
                except Exception as exc:
                    raise RuntimeError(f"Cannot create object shard map for GPU {gpu_id}: {exc}") from exc
                shard_cmd = cmd.copy()
                shard_cmd[shard_cmd.index("--global-map") + 1] = str(shard_map)
                shard_cmd.append("--no-aggregate")
                jobs.append({"name": f"object_gpu{gpu_id}", "cmd": shard_cmd, "env_updates": {"CUDA_VISIBLE_DEVICES": str(gpu_id)}})
            records.append(run_parallel_steps("object v2 yoloe sharded", jobs))
            records.append(run_step(
                "object v2 aggregate index",
                [
                    py,
                    "scripts/run_keyframe_v2_object_index.py",
                    "--global-map",
                    str(keyframe_root / "indexes" / "keyframe_v2_global_map.parquet"),
                    "--output-root",
                    str(object_root),
                    "--index-output",
                    str(object_index_root),
                    "--aggregate-only",
                ],
            ))
        else:
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
        if sharded_gpu_devices:
            jobs = []
            for gpu_id, shard_video_ids in zip(sharded_gpu_devices, split_evenly(video_ids, len(sharded_gpu_devices))):
                shard_cmd = [
                    py,
                    "scripts/run_ocr_v2_selected_keyframes.py",
                    "--selected-root",
                    str(keyframe_root),
                    "--output-dir",
                    str(ocr_v2_root),
                    "--device",
                    args.ocr_device,
                    "--no-aggregate",
                ]
                for video_id in shard_video_ids:
                    shard_cmd.extend(["--video-id", video_id])
                jobs.append({"name": f"ocr_v2_gpu{gpu_id}", "cmd": shard_cmd, "env_updates": {"CUDA_VISIBLE_DEVICES": str(gpu_id)}})
            records.append(run_parallel_steps("ocr v2 selected keyframes sharded", jobs))
            aggregate_cmd = [
                py,
                "scripts/run_ocr_v2_selected_keyframes.py",
                "--selected-root",
                str(keyframe_root),
                "--output-dir",
                str(ocr_v2_root),
                "--aggregate-only",
            ]
            for video_id in video_ids:
                aggregate_cmd.extend(["--video-id", video_id])
            records.append(run_step("ocr v2 aggregate outputs", aggregate_cmd))
        else:
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
                "--batch-size",
                str(args.ocr_index_batch_size),
            ],
        ))

    if not args.skip_asr:
        if shutil.which("ffmpeg") is None:
            print("[KAGGLE][WARN] ffmpeg not found on PATH; ASR audio extraction will likely fail.")
        base_asr_cmd = [
            py,
            "scripts/run_batch_asr_whisper.py",
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
            "--index-batch-size",
            str(args.asr_index_batch_size),
        ]
        append_repeated_arg(base_asr_cmd, "--video-dir", video_roots)
        if args.force:
            base_asr_cmd.append("--overwrite")
        if sharded_gpu_devices:
            jobs = []
            for gpu_id, shard_video_ids in zip(sharded_gpu_devices, split_evenly(video_ids, len(sharded_gpu_devices))):
                shard_cmd = base_asr_cmd.copy()
                shard_cmd.append("--skip-index")
                for video_id in shard_video_ids:
                    shard_cmd.extend(["--video-id", video_id])
                jobs.append({"name": f"asr_gpu{gpu_id}", "cmd": shard_cmd, "env_updates": {"CUDA_VISIBLE_DEVICES": str(gpu_id)}})
            records.append(run_parallel_steps("asr faster-whisper v3 sharded", jobs))
            records.append(run_step(
                "asr faster-whisper v3 aggregate index",
                [
                    py,
                    "scripts/build_asr_v3_index.py",
                    "--asr-dir",
                    str(asr_root),
                    "--output-dir",
                    str(index_root / "asr_v3"),
                    "--batch-size",
                    str(args.asr_index_batch_size),
                ],
            ))
        else:
            cmd = base_asr_cmd
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
        "video_root_glob": args.video_root_glob,
        "video_roots": [str(root) for root in video_roots],
        "video_ids": video_ids,
        "steps": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[KAGGLE] Wrote run report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
