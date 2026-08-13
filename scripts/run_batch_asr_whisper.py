"""
Batch Faster-Whisper ASR Pipeline (Phase 1 for all L21 Videos)
=============================================================
Runs Faster-Whisper large-v3-turbo across all videos in L21,
creates 15-30s retrieval chunks, and builds the unified ASR V3 FAISS + Parquet Index.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio_asr_parser import (
    extract_audio_wav,
    build_asr_retrieval_chunks,
)
from scripts.build_asr_v3_index import build_asr_v3_index


def save_json(data: any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_batch_whisper(
    video_dir: Path | str | list[Path | str] = None,
    output_dir: Path | str = None,
    audio_dir: Path | str = None,
    model_size: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
    overwrite: bool = False,
    limit: int | None = None,
    index_output_dir: Path | str | None = None,
    video_ids: list[str] | None = None,
    skip_index: bool = False,
    index_batch_size: int = 32,
):
    if video_dir is None:
        video_dir = PROJECT_ROOT / "datasets_L21" / "Videos_L21_a" / "video"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "asr"
    if audio_dir is None:
        audio_dir = PROJECT_ROOT / "outputs" / "audio"

    if isinstance(video_dir, (list, tuple)):
        video_dirs = [Path(path) for path in video_dir]
    else:
        video_dirs = [Path(video_dir)]
    output_dir = Path(output_dir)
    audio_dir = Path(audio_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    video_by_id: dict[str, Path] = {}
    for root in video_dirs:
        for path in sorted(root.glob("*.mp4")):
            if path.name.startswith("."):
                continue
            video_by_id.setdefault(path.stem, path)
    video_files = sorted(video_by_id.values())
    if not video_files:
        # Check subdirectories
        video_files = sorted(list(PROJECT_ROOT.glob("datasets_L21/**/L21_*.mp4")))
    if video_ids:
        requested = {str(video_id).strip() for video_id in video_ids if str(video_id).strip()}
        video_files = [path for path in video_files if path.stem in requested]
    if limit is not None:
        video_files = video_files[: max(0, limit)]

    print("=" * 80)
    print(f" 🚀 BATCH ASR PIPELINE (FASTER-WHISPER {model_size.upper()})")
    print(f" Total Videos Found: {len(video_files)}")
    print(f" Output Directory  : {output_dir}")
    print(f" Audio Directory   : {audio_dir}")
    print(f" Device / Compute  : {device} / {compute_type}")
    print("=" * 80)

    # Initialize Whisper Model once to reuse across all videos in memory
    from faster_whisper import WhisperModel
    from src.preprocessing.audio_asr_parser import get_default_hf_cache
    
    download_root = get_default_hf_cache()
    print(f"Loading Faster-Whisper Model '{model_size}' into {device}...")
    t_model_0 = time.time()
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )
    print(f"Model ready in {time.time() - t_model_0:.2f}s.\n")

    summary_records = []
    total_start_time = time.time()

    for idx, v_path in enumerate(video_files, 1):
        vid_id = v_path.stem
        out_asr_json = output_dir / f"{vid_id}_asr.json"
        out_chunk_json = output_dir / f"{vid_id}_asr_chunks.json"
        out_wav = audio_dir / f"{vid_id}.wav"

        print(f"[{idx:02d}/{len(video_files):02d}] Processing Video: {vid_id} ({v_path.name})...")
        t_v0 = time.time()

        # Check existing
        if not overwrite and out_asr_json.exists() and out_chunk_json.exists():
            print(f"  ⏭️ Already processed. Skipping {vid_id}.")
            try:
                with open(out_chunk_json, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                with open(out_asr_json, "r", encoding="utf-8") as f:
                    asr_data = json.load(f)
                summary_records.append({
                    "video_id": vid_id,
                    "segments": len(asr_data),
                    "chunks": len(chunks_data),
                    "speech_duration_s": round(float(asr_data[-1]["end"] if asr_data else 0.0), 2),
                    "status": "cached",
                    "elapsed_s": 0.0,
                })
            except Exception:
                pass
            continue

        # 1. Extract 16kHz mono Audio
        ok_audio = extract_audio_wav(v_path, out_wav)
        if not ok_audio or not out_wav.exists():
            print(f"  ❌ Failed to extract audio for {vid_id}!")
            continue

        # 2. Transcribe Audio with Whisper
        t_trans = time.time()
        segments_gen, info = model.transcribe(
            str(out_wav),
            language="vi",
            vad_filter=True,
            word_timestamps=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments = []
        for s_idx, seg in enumerate(segments_gen):
            text_raw = str(seg.text).strip()
            if not text_raw:
                continue
            words_list = []
            if seg.words:
                for w in seg.words:
                    words_list.append({
                        "word": str(w.word).strip(),
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                        "probability": round(float(w.probability), 4),
                    })
            segments.append({
                "video_id": vid_id,
                "asr_id": f"{vid_id}_asr_{s_idx:04d}",
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text_raw": text_raw,
                "words": words_list,
                "word_probability": round(float(np.mean([w['probability'] for w in words_list]) if words_list else 1.0), 4),
            })

        # Save Raw ASR JSON
        save_json(segments, out_asr_json)

        # 3. Build Retrieval Chunks (15-30s)
        chunks = build_asr_retrieval_chunks(segments, vid_id)
        save_json(chunks, out_chunk_json)

        elapsed = time.time() - t_v0
        speech_dur = segments[-1]["end"] if segments else 0.0
        print(f"  ✅ Done: {len(segments)} segments, {len(chunks)} chunks ({speech_dur:.1f}s speech) in {elapsed:.1f}s.")

        summary_records.append({
            "video_id": vid_id,
            "segments": len(segments),
            "chunks": len(chunks),
            "speech_duration_s": round(float(speech_dur), 2),
            "status": "completed",
            "elapsed_s": round(elapsed, 2),
        })

    # Summary table
    total_elapsed = time.time() - total_start_time
    total_segs = sum(r["segments"] for r in summary_records)
    total_chunks = sum(r["chunks"] for r in summary_records)
    total_speech_s = sum(r["speech_duration_s"] for r in summary_records)

    print("\n" + "=" * 80)
    print(f" 🎉 BATCH ASR PIPELINE COMPLETED")
    print(f" Total Videos Processed: {len(summary_records)}/{len(video_files)}")
    print(f" Total Speech Duration : {total_speech_s / 60:.2f} minutes")
    print(f" Total ASR Segments    : {total_segs}")
    print(f" Total Retrieval Chunks: {total_chunks}")
    print(f" Total Pipeline Time   : {total_elapsed / 60:.2f} minutes")
    print("=" * 80)

    # 4. Automatically Build Full ASR V3 FAISS Index
    if not skip_index:
        print("\nBuilding Full ASR V3 Hybrid Index from all generated chunks...")
        build_asr_v3_index(asr_dir=output_dir, output_dir=index_output_dir, batch_size=index_batch_size)
        print("✅ All indices built and search-ready!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Faster-Whisper ASR Pipeline")
    parser.add_argument("--video-dir", action="append", default=None, help="Directory containing mp4 videos. Can be repeated.")
    parser.add_argument("--output-dir", default=None, help="Output directory for ASR JSON chunks")
    parser.add_argument("--audio-dir", default=None, help="Output directory for extracted WAVs")
    parser.add_argument("--model-size", default="large-v3-turbo", help="Faster-Whisper model size")
    parser.add_argument("--device", default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--compute-type", default="float16", help="Compute type")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ASR outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional development limit. Omit for full run.")
    parser.add_argument("--index-output-dir", default=None, help="Output directory for the ASR FAISS index")
    parser.add_argument("--video-id", action="append", default=[], help="Only process this video id. Can be repeated.")
    parser.add_argument("--skip-index", action="store_true", help="Only write ASR JSON/chunks; build FAISS index separately.")
    parser.add_argument("--index-batch-size", type=int, default=32)
    args = parser.parse_args()

    run_batch_whisper(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        audio_dir=args.audio_dir,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        overwrite=args.overwrite,
        limit=args.limit,
        index_output_dir=args.index_output_dir,
        video_ids=args.video_id,
        skip_index=args.skip_index,
        index_batch_size=args.index_batch_size,
    )
