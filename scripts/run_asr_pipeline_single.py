"""
Run Faster-Whisper ASR pipeline on a single video (L21_V001).
Extracts 16kHz mono audio, transcribes with large-v3-turbo (vi, vad_filter, word_timestamps),
generates 15-30s retrieval chunks (~5s overlap), and saves JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set cache directories to E: drive to ensure ample disk space
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")

from src.preprocessing.audio_asr_parser import (
    build_asr_retrieval_chunks,
    extract_audio_wav,
    run_faster_whisper_asr,
)
from src.retrieval.logging_utils import setup_logger


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS.ms string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def main():
    parser = argparse.ArgumentParser(description="Run Faster-Whisper ASR for single video.")
    parser.add_argument("--video-id", default="L21_V001", help="Video identifier")
    parser.add_argument(
        "--video-path",
        default=str(PROJECT_ROOT / "datasets_L21" / "Videos_L21_a" / "video" / "L21_V001.mp4"),
        help="Path to source MP4 video",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(PROJECT_ROOT / "outputs" / "audio"),
        help="Directory to save extracted 16kHz mono WAV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "asr"),
        help="Directory to save ASR JSON and chunk outputs",
    )
    parser.add_argument("--model-size", default="large-v3-turbo", help="Faster-whisper model")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--compute-type", default="float16", help="Compute type")
    parser.add_argument("--language", default="vi", help="Audio spoken language")
    args = parser.parse_args()

    logger = setup_logger("run-asr-single")

    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error("Video file not found at %s", video_path)
        sys.exit(1)

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{args.video_id}.wav"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  FASTER-WHISPER ASR PIPELINE - SINGLE VIDEO TEST")
    print(f"  Target Video   : {args.video_id} ({video_path.name})")
    print(f"  Video Path     : {video_path}")
    print(f"  Model          : {args.model_size} ({args.device}, {args.compute_type})")
    print(f"  Language       : {args.language} (VAD=True, word_timestamps=True)")
    print("=" * 80)

    # 1. Extract audio (16kHz mono WAV)
    print(f"\n[Step 1/3] Extracting 16kHz mono WAV audio from video...")
    t_start = time.time()
    ok = extract_audio_wav(video_path, wav_path, sample_rate=16000, channels=1)
    if not ok or not wav_path.exists():
        logger.error("Failed to extract audio from %s", video_path)
        sys.exit(1)
    wav_size_mb = wav_path.stat().st_size / (1024 * 1024)
    print(f"  --> Extracted audio: {wav_path} ({wav_size_mb:.2f} MB)")

    # 2. Run Faster-Whisper ASR
    print(f"\n[Step 2/3] Running Faster-Whisper ASR ({args.model_size})...")
    segments = run_faster_whisper_asr(
        audio_path=wav_path,
        video_id=args.video_id,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        vad_filter=True,
        word_timestamps=True,
    )

    asr_json_path = output_dir / f"{args.video_id}_asr.json"
    with open(asr_json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"  --> Saved ASR segments: {asr_json_path} ({len(segments)} segments)")

    # 3. Build retrieval chunks (15-30s, ~5s overlap)
    print(f"\n[Step 3/3] Generating ASR retrieval chunks (15-30s, ~5s overlap)...")
    chunks = build_asr_retrieval_chunks(
        segments=segments,
        video_id=args.video_id,
        min_chunk_duration=15.0,
        target_chunk_duration=25.0,
        max_chunk_duration=30.0,
        overlap_duration=5.0,
    )

    chunks_json_path = output_dir / f"{args.video_id}_asr_chunks.json"
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  --> Saved ASR chunks: {chunks_json_path} ({len(chunks)} chunks)")

    # 4. Statistics and Inspection
    total_speech_duration = sum((s["end"] - s["start"]) for s in segments)
    avg_segment_duration = total_speech_duration / len(segments) if segments else 0.0
    avg_chunk_duration = sum(c["duration"] for c in chunks) / len(chunks) if chunks else 0.0

    print("\n" + "=" * 80)
    print("  THỐNG KÊ KẾT QUẢ ASR (STATISTICS)")
    print("=" * 80)
    print(f"  * Video ID                   : {args.video_id}")
    print(f"  * Số segment (Segments count): {len(segments)}")
    print(f"  * Số chunk (Chunks count)    : {len(chunks)}")
    print(f"  * Tổng thời lượng speech     : {total_speech_duration:.2f} giây ({format_duration(total_speech_duration)})")
    print(f"  * Thời lượng TB mỗi segment  : {avg_segment_duration:.2f} giây")
    print(f"  * Thời lượng TB mỗi chunk    : {avg_chunk_duration:.2f} giây")
    print(f"  * File kết quả ASR segments  : {asr_json_path}")
    print(f"  * File kết quả ASR chunks    : {chunks_json_path}")
    print(f"  * Tổng thời gian thực thi    : {time.time() - t_start:.2f} giây")

    print("\n" + "=" * 80)
    print("  20 SEGMENT MẪU KÈM TIMESTAMP (ĐỂ KIỂM TRA THỦ CÔNG)")
    print("=" * 80)
    
    # Pick 20 sample segments distributed across the video
    if len(segments) <= 20:
        sample_segments = segments
    else:
        # Pick 20 evenly distributed segments
        indices = [int(i * (len(segments) - 1) / 19) for i in range(20)]
        sample_segments = [segments[idx] for idx in sorted(set(indices))]

    header = f" {'#':<3} | {'ASR ID':<18} | {'Start':<8} | {'End':<8} | {'Dur(s)':<6} | {'Conf':<6} | {'Text Raw'}"
    print(header)
    print("-" * 120)
    for i, seg in enumerate(sample_segments, 1):
        dur = seg["end"] - seg["start"]
        conf = seg.get("word_probability", 1.0)
        text = seg["text_raw"]
        if len(text) > 75:
            text = text[:72] + "..."
        print(f" {i:<3} | {seg['asr_id']:<18} | {seg['start']:<8.2f} | {seg['end']:<8.2f} | {dur:<6.2f} | {conf:<6.2f} | {text}")

    print("=" * 120)

    # Also display 5 sample retrieval chunks
    print("\n" + "=" * 80)
    print("  5 CHUNK RETRIEVAL MẪU (KIỂM TRA CHUNKING 15-30s, OVERLAP ~5s)")
    print("=" * 80)
    for i, chk in enumerate(chunks[:5], 1):
        print(f" [Chunk #{i}] ID: {chk['chunk_id']} | Range: {chk['start']:.2f}s -> {chk['end']:.2f}s | Duration: {chk['duration']:.2f}s | Segments: {len(chk['segment_ids'])}")
        print(f"   Text: \"{chk['text']}\"")
        print()
    print("=" * 80)


if __name__ == "__main__":
    main()
