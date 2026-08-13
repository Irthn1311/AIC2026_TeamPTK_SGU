"""
Script 12: Test Branch 2 (Audio -> Whisper ASR -> ASR Index)
============================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import PROJECT_ROOT
from src.preprocessing.audio_asr_parser import extract_audio_wav, map_asr_to_keyframes, run_whisper_asr
from src.retrieval.asr_index import build_asr_index
from src.retrieval.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="L21_V005")
    parser.add_argument("--dataset-root", default=r"e:\AI Challenge TP.HCM 2026\CodeBase\datasets_L21")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "indexes" / "asr"))
    args = parser.parse_args()

    logger = setup_logger("asr-test")

    video_path = Path(args.dataset_root) / "Videos_L21_a" / "video" / f"{args.video_id}.mp4"
    wav_path = PROJECT_ROOT / "outputs" / "audio" / f"{args.video_id}.wav"
    mapping_dir = Path(args.dataset_root) / "map-keyframes-aic25-b1" / "map-keyframes"
    keyframe_dir = Path(args.dataset_root) / "Keyframes_L21" / "keyframes"

    print("=" * 60)
    print(f"  TESTING BRANCH 2 (RAW VIDEO -> WHISPER ASR INDEX)")
    print(f"  Target Video: {args.video_id}")
    print("=" * 60)

    # Step 1: Extract Audio
    logger.info("Extracting WAV audio for %s...", args.video_id)
    ok = extract_audio_wav(video_path, wav_path)
    if not ok:
        print(f"Error: Failed to extract audio from {video_path}")
        return

    # Step 2: Run Whisper ASR
    logger.info("Running Whisper ASR on %s...", wav_path)
    asr_segments = run_whisper_asr(wav_path, model_size="tiny")
    print(f"Whisper ASR extracted {len(asr_segments)} speech segments.")

    # Step 3: Map ASR to Keyframes
    logger.info("Mapping ASR segments to keyframes...")
    asr_records = map_asr_to_keyframes(args.video_id, asr_segments, mapping_dir, keyframe_dir)
    df_asr = pd.DataFrame(asr_records)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_asr.to_parquet(out_dir / "l21_asr_corpus.parquet", index=False)

    # Step 4: Build ASR FAISS Index
    logger.info("Building ASR FAISS Index...")
    index, df_filtered, meta = build_asr_index(out_dir / "l21_asr_corpus.parquet", out_dir, logger_inst=logger)

    print(f"\n[BRANCH 2 RESULT]")
    print(f"  Speech segments transcribed: {len(asr_segments)}")
    print(f"  Keyframes with speech: {len(df_filtered)}")
    if not df_filtered.empty:
        print("  Sample ASR keyframe transcriptions:")
        for _, row in df_filtered.head(3).iterrows():
            print(f"    - Frame {row['frame_idx']} ({row['timestamp_seconds']:.2f}s): \"{row['asr_text'][:70]}\"")

    print("\n" + "=" * 60)
    print("  BRANCH 2 STATUS: READY & FUNCTIONAL")
    print("=" * 60)


if __name__ == "__main__":
    main()
