"""Dual-T4 ASR inventory, scheduling, benchmark, and merge contracts."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .asr import decode_mp4_waveform, normalize_text
from .contracts import WHISPER_ID, WHISPER_REVISION


def atomic_write_json(path: Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, target)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def probe_audio(video_id: str, input_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,duration:format=duration",
        "-of",
        "json",
        str(input_path),
    ]
    started = time.monotonic()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    row: dict[str, Any] = {
        "video_id": str(video_id),
        "input_path": str(Path(input_path)),
        "has_audio": False,
        "duration_seconds": 0.0,
        "codec": None,
        "sample_rate": None,
        "probe_status": "FAILED",
        "probe_wall_seconds": time.monotonic() - started,
    }
    if process.returncode != 0:
        row["probe_error"] = process.stderr.strip()[-2000:]
        return row
    try:
        payload = json.loads(process.stdout)
        streams = payload.get("streams", [])
        if not streams:
            row["probe_status"] = "NO_AUDIO"
            return row
        stream = streams[0]
        duration = stream.get("duration") or payload.get("format", {}).get("duration")
        duration_seconds = float(duration)
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("non-positive audio duration")
        row.update(
            {
                "has_audio": True,
                "duration_seconds": duration_seconds,
                "codec": stream.get("codec_name"),
                "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                "probe_status": "PASS",
            }
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        row["probe_error"] = f"{type(error).__name__}: {error}"
    return row


def lpt_partition(rows: list[dict[str, Any]], shard_count: int) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    eligible = [row for row in rows if row.get("has_audio") and row.get("probe_status") == "PASS"]
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    totals = [0.0] * shard_count
    for row in sorted(
        eligible,
        key=lambda item: (-float(item["duration_seconds"]), str(item["video_id"])),
    ):
        index = min(range(shard_count), key=lambda item: (totals[item], item))
        shards[index].append(row)
        totals[index] += float(row["duration_seconds"])
    maximum, minimum = max(totals, default=0.0), min(totals, default=0.0)
    imbalance = 0.0 if maximum == 0 else 100.0 * (maximum - minimum) / maximum
    return [
        {
            "shard_id": index,
            "video_count": len(shards[index]),
            "total_audio_seconds": totals[index],
            "imbalance_percent": imbalance,
            "videos": shards[index],
        }
        for index in range(shard_count)
    ]


def representative_sample(rows: list[dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    eligible = sorted(
        (row for row in rows if row.get("has_audio") and row.get("probe_status") == "PASS"),
        key=lambda item: (float(item["duration_seconds"]), str(item["video_id"])),
    )
    if len(eligible) < count:
        raise RuntimeError(f"ASR_BENCHMARK_REQUIRES_{count}_AUDIO_VIDEOS")
    indices = [round(index * (len(eligible) - 1) / (count - 1)) for index in range(count)]
    return [eligible[index] for index in indices]


def decode_audio_for_worker(
    input_path: Path,
    duration_seconds: float,
    *,
    memory_threshold_bytes: int = 512 * 1024 * 1024,
) -> tuple[np.ndarray, list[str], str]:
    estimated_bytes = int(math.ceil(duration_seconds * 16000 * 4))
    if estimated_bytes <= memory_threshold_bytes:
        waveform, command = decode_mp4_waveform(input_path)
        return waveform, command, "F32LE_PIPE"
    descriptor, temporary_name = tempfile.mkstemp(prefix="triage_asr_", suffix=".f32le")
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-y",
        str(temporary),
    ]
    try:
        process = subprocess.run(command, capture_output=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"FFMPEG_AUDIO_FILE_DECODE_FAILED: {process.stderr.decode(errors='replace')}"
            )
        waveform = np.fromfile(temporary, dtype="<f4")
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise RuntimeError("FFMPEG_AUDIO_FILE_EMPTY_OR_NONFINITE")
        return waveform, command, "TEMP_F32LE_DELETED_AFTER_READ"
    finally:
        temporary.unlink(missing_ok=True)


def normalize_segments(
    chunks: list[dict[str, Any]], audio_duration_seconds: float
) -> tuple[list[dict[str, Any]], int]:
    segments: list[dict[str, Any]] = []
    recovered = 0
    for index, chunk in enumerate(chunks):
        timestamp = chunk.get("timestamp", (None, None))
        start, end = timestamp if timestamp is not None else (None, None)
        if start is None:
            continue
        start = float(start)
        if end is None:
            next_start = None
            if index + 1 < len(chunks):
                following = chunks[index + 1].get("timestamp", (None, None))
                next_start = following[0] if following else None
            end = float(next_start) if next_start is not None else float(audio_duration_seconds)
            recovered += 1
        end = min(float(end), float(audio_duration_seconds))
        previous_end = segments[-1]["end_seconds"] if segments else 0.0
        if start < previous_end or end < start or not math.isfinite(start + end):
            continue
        raw_text = str(chunk.get("text", ""))
        segments.append(
            {
                "start_seconds": start,
                "end_seconds": end,
                "raw_text": raw_text,
                "normalized_text": normalize_text(raw_text),
            }
        )
    return segments, recovered


def timestamps_monotonic(segments: list[dict[str, Any]]) -> bool:
    previous = 0.0
    for segment in segments:
        start, end = float(segment["start_seconds"]), float(segment["end_seconds"])
        if start < previous or end < start:
            return False
        previous = end
    return True


def transcript_tokens(row: dict[str, Any]) -> set[str]:
    tokens = {
        token
        for segment in row.get("segments", [])
        for token in re.findall(r"\w+", segment.get("normalized_text", "").casefold())
    }
    if not tokens:
        tokens.update(re.findall(r"\w+", str(row.get("normalized_text", "")).casefold()))
    return tokens


def material_consistency(left: dict[str, Any], right: dict[str, Any]) -> float:
    first, second = transcript_tokens(left), transcript_tokens(right)
    if not first and not second:
        return 1.0
    return len(first & second) / max(len(first | second), 1)


def lexical_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "PASS":
            continue
        for segment in row.get("segments", []):
            for token in sorted(set(re.findall(r"\w+", segment["normalized_text"].casefold()))):
                output.setdefault(token, []).append(
                    {
                        "video_id": row["video_id"],
                        "start_seconds": segment["start_seconds"],
                        "end_seconds": segment["end_seconds"],
                        "text": segment["normalized_text"],
                    }
                )
    return output


def worker_run(
    manifest_path: Path,
    checkpoint_path: Path,
    progress_path: Path,
    asset_root: Path,
    batch_size: int,
    *,
    benchmark: bool = False,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    existing = {row["video_id"]: row for row in read_jsonl(checkpoint_path)}
    completed = {video_id for video_id, row in existing.items() if row.get("status") == "PASS"}
    torch.cuda.reset_peak_memory_stats(0)
    worker_started = time.monotonic()
    model_started = time.monotonic()
    processor = AutoProcessor.from_pretrained(asset_root, local_files_only=True)
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            asset_root,
            local_files_only=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .to("cuda:0")
        .eval()
    )
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.float16,
        device=0,
        chunk_length_s=30,
    )
    model_load_seconds = time.monotonic() - model_started
    decode_seconds = 0.0
    whisper_seconds = 0.0
    processed_audio_seconds = sum(
        float(existing[video_id].get("duration_seconds", 0.0)) for video_id in completed
    )
    failed: list[str] = []
    for item in manifest["videos"]:
        video_id = str(item["video_id"])
        if video_id in completed:
            continue
        try:
            decode_started = time.monotonic()
            waveform, ffmpeg_command, decode_mode = decode_audio_for_worker(
                Path(item["input_path"]), float(item["duration_seconds"])
            )
            decode_seconds += time.monotonic() - decode_started
            duration = len(waveform) / 16000.0
            asr_started = time.monotonic()
            result = transcriber(
                {"array": waveform, "sampling_rate": 16000},
                batch_size=batch_size,
                return_timestamps=True,
                generate_kwargs={"task": "transcribe"},
            )
            whisper_seconds += time.monotonic() - asr_started
            segments, recovered = normalize_segments(result.get("chunks", []), duration)
            if not timestamps_monotonic(segments):
                raise RuntimeError("ASR_TIMESTAMP_MONOTONICITY_FAILED")
            existing[video_id] = {
                "video_id": video_id,
                "status": "PASS",
                "language": result.get("language"),
                "segments": segments,
                "raw_text": str(result.get("text", "")),
                "normalized_text": normalize_text(str(result.get("text", ""))),
                "duration_seconds": duration,
                "model_id": WHISPER_ID,
                "model_revision": WHISPER_REVISION,
                "ffmpeg_command": ffmpeg_command,
                "decode_mode": decode_mode,
                "recovered_timestamp_boundaries": recovered,
                "worker_batch_size": batch_size,
            }
            completed.add(video_id)
            processed_audio_seconds += duration
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            existing[video_id] = {
                "video_id": video_id,
                "status": "OOM",
                "duration_seconds": float(item["duration_seconds"]),
                "error": f"{type(error).__name__}: {error}",
            }
            failed.append(video_id)
            atomic_write_jsonl(checkpoint_path, (existing[key] for key in sorted(existing)))
            break
        except Exception as error:  # noqa: BLE001 - persisted concrete per-file failure
            existing[video_id] = {
                "video_id": video_id,
                "status": "ASR_FAILED",
                "duration_seconds": float(item["duration_seconds"]),
                "error": f"{type(error).__name__}: {error}",
            }
            failed.append(video_id)
        atomic_write_jsonl(checkpoint_path, (existing[key] for key in sorted(existing)))
        wall = time.monotonic() - worker_started
        atomic_write_json(
            progress_path,
            {
                "processed_video_ids": sorted(completed),
                "failed_video_ids": sorted(failed),
                "accumulated_audio_seconds": processed_audio_seconds,
                "wall_seconds": wall,
                "current_rtf": wall / max(processed_audio_seconds, 1e-9),
            },
        )
        if len(existing) % 5 == 0 or len(existing) == len(manifest["videos"]):
            print(
                {
                    "shard": manifest["shard_id"],
                    "processed": len(completed),
                    "total": len(manifest["videos"]),
                    "failed": len(failed),
                    "audio_hours": processed_audio_seconds / 3600.0,
                    "rtf": wall / max(processed_audio_seconds, 1e-9),
                },
                flush=True,
            )
    wall_seconds = time.monotonic() - worker_started
    rows = [existing[key] for key in sorted(existing)]
    success = [row for row in rows if row.get("status") == "PASS"]
    nonempty = [row for row in success if transcript_tokens(row)]
    report = {
        "shard_id": manifest["shard_id"],
        "benchmark": benchmark,
        "batch_size": batch_size,
        "video_count": len(manifest["videos"]),
        "completed_count": len(success),
        "failed_count": len(rows) - len(success),
        "nonempty_count": len(nonempty),
        "timestamp_valid_count": sum(
            timestamps_monotonic(row.get("segments", [])) for row in success
        ),
        "total_audio_seconds": sum(float(row.get("duration_seconds", 0.0)) for row in rows),
        "wall_seconds": wall_seconds,
        "model_load_seconds": model_load_seconds,
        "ffmpeg_wall_seconds": decode_seconds,
        "whisper_wall_seconds": whisper_seconds,
        "rtf": wall_seconds / max(processed_audio_seconds, 1e-9),
        "audio_hours_per_wall_hour": processed_audio_seconds / max(wall_seconds, 1e-9),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(0),
        "device_name": torch.cuda.get_device_name(0),
        "model_id": WHISPER_ID,
        "model_revision": WHISPER_REVISION,
    }
    atomic_write_json(
        Path(progress_path).with_name(Path(progress_path).stem + "_report.json"), report
    )
    return report


def merge_shards(
    inventory: list[dict[str, Any]], shard_paths: list[Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {
        str(row["video_id"])
        for row in inventory
        if row.get("has_audio") and row.get("probe_status") == "PASS"
    }
    merged: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for path in shard_paths:
        for row in read_jsonl(path):
            video_id = str(row["video_id"])
            if video_id in merged:
                duplicates.append(video_id)
            merged[video_id] = row
    actual = set(merged)
    diagnostics = {
        "expected_audio_video_count": len(expected),
        "merged_video_count": len(actual),
        "duplicate_video_ids": sorted(set(duplicates)),
        "missing_video_ids": sorted(expected - actual),
        "unexpected_video_ids": sorted(actual - expected),
    }
    if any(
        diagnostics[key]
        for key in ("duplicate_video_ids", "missing_video_ids", "unexpected_video_ids")
    ):
        raise RuntimeError(f"ASR_SHARD_MERGE_GATE_FAILED: {diagnostics}")
    return [merged[key] for key in sorted(merged)], diagnostics


def aggregate_worker_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    wall = max(float(report["wall_seconds"]) for report in reports)
    audio = sum(float(report["total_audio_seconds"]) for report in reports)
    return {
        "gpu_count": len(reports),
        "batch_size": reports[0]["batch_size"],
        "video_count": sum(int(report["video_count"]) for report in reports),
        "total_audio_seconds": audio,
        "wall_seconds": wall,
        "ffmpeg_wall_seconds": sum(float(report["ffmpeg_wall_seconds"]) for report in reports),
        "whisper_wall_seconds": sum(float(report["whisper_wall_seconds"]) for report in reports),
        "rtf": wall / max(audio, 1e-9),
        "audio_hours_per_wall_hour": audio / max(wall, 1e-9),
        "peak_allocated_vram_bytes_per_gpu": [
            report["peak_allocated_vram_bytes"] for report in reports
        ],
        "peak_reserved_vram_bytes_per_gpu": [
            report["peak_reserved_vram_bytes"] for report in reports
        ],
        "transcript_success_count": sum(int(report["completed_count"]) for report in reports),
        "empty_transcript_count": sum(
            int(report["completed_count"]) - int(report["nonempty_count"]) for report in reports
        ),
        "timestamp_valid_count": sum(int(report["timestamp_valid_count"]) for report in reports),
        "workers": reports,
    }


def launch_workers(
    repo_root: Path,
    manifests: list[Path],
    checkpoint_paths: list[Path],
    progress_paths: list[Path],
    asset_root: Path,
    batch_size: int,
    gpu_indices: list[int],
    *,
    benchmark: bool = False,
) -> dict[str, Any]:
    if not (len(manifests) == len(checkpoint_paths) == len(progress_paths) == len(gpu_indices)):
        raise ValueError("worker launch lists must have equal length")
    processes: list[tuple[subprocess.Popen[bytes], Path, int]] = []
    for manifest, checkpoint, progress, gpu_index in zip(
        manifests, checkpoint_paths, progress_paths, gpu_indices, strict=True
    ):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        source_root = str(Path(repo_root) / "src")
        environment["PYTHONPATH"] = source_root + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        command = [
            os.sys.executable,
            "-m",
            "triage_eg.fs1_v11.asr_v12_worker",
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
            "--progress",
            str(progress),
            "--asset-root",
            str(asset_root),
            "--batch-size",
            str(batch_size),
        ]
        if benchmark:
            command.append("--benchmark")
        processes.append(
            (
                subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=environment,
                ),
                progress,
                gpu_index,
            )
        )
    reports = []
    for process, progress, gpu_index in processes:
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                f"ASR_WORKER_FAILED gpu={gpu_index} returncode={process.returncode} "
                f"progress={progress}"
            )
        report_path = progress.with_name(progress.stem + "_report.json")
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    return aggregate_worker_reports(reports)


def valid_benchmark(report: dict[str, Any]) -> bool:
    return (
        report.get("transcript_success_count") == report.get("video_count")
        and report.get("timestamp_valid_count") == report.get("video_count")
        and all(worker.get("failed_count") == 0 for worker in report.get("workers", []))
        and bool(report.get("workers"))
    )


def select_lowest_rtf(reports: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((report for report in reports if int(report["batch_size"]) == 4), None)
    if baseline is None or not valid_benchmark(baseline):
        raise RuntimeError("ASR_BATCH4_REFERENCE_INVALID")
    stable = [
        report
        for report in reports
        if valid_benchmark(report)
        and int(report["empty_transcript_count"]) <= int(baseline["empty_transcript_count"])
    ]
    if not stable:
        raise RuntimeError("NO_STABLE_ASR_BATCH_CONFIGURATION")
    return min(stable, key=lambda report: (float(report["rtf"]), int(report["batch_size"])))
