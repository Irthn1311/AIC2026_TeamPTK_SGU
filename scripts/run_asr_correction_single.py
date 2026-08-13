"""
Run ASR Text Correction using Qwen3-4B Local for a single video (L21_V001).
Loads raw ASR segments and OCR detections, performs contextual phonetic correction,
saves corrected segments and chunks, and prints 20 sample changes.
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

# Prioritize E: drive model cache
MODEL_CACHE_DIR = PROJECT_ROOT / ".model_cache" / "qwen3_4b"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR)

from src.preprocessing.asr_text_correction import (
    Qwen3ASRCorrector,
    build_corrected_retrieval_chunks,
    load_ocr_keyframes,
)
from src.retrieval.logging_utils import setup_logger


def format_ts(seconds: float) -> str:
    """Format seconds into MM:SS.ms string."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"


def main():
    parser = argparse.ArgumentParser(description="ASR Text Correction with Qwen3-4B Local.")
    parser.add_argument("--video-id", default="L21_V001", help="Video identifier (e.g. L21_V001)")
    parser.add_argument(
        "--asr-input",
        default=str(PROJECT_ROOT / "outputs" / "asr" / "L21_V001_asr.json"),
        help="Path to raw ASR JSON file",
    )
    parser.add_argument(
        "--chunks-input",
        default=str(PROJECT_ROOT / "outputs" / "asr" / "L21_V001_asr_chunks.json"),
        help="Path to raw ASR chunks JSON file",
    )
    parser.add_argument(
        "--ocr-input",
        default=str(PROJECT_ROOT / "outputs" / "ocr_full" / "per_video" / "L21_V001.jsonl"),
        help="Path to OCR keyframes JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "asr"),
        help="Directory to save corrected JSON files",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B", help="Hugging Face model ID")
    parser.add_argument("--cache-dir", default=str(MODEL_CACHE_DIR), help="Model storage directory on E drive")
    parser.add_argument("--device", default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--use-4bit", action="store_true", default=True, help="Use 4-bit quantization")
    args = parser.parse_args()

    logger = setup_logger("run-asr-correction")

    # 1. Check Input Files
    asr_path = Path(args.asr_input)
    if not asr_path.exists():
        # Fallback to output-dir pattern
        asr_path = Path(args.output_dir) / f"{args.video_id}_asr.json"

    if not asr_path.exists():
        logger.error("Raw ASR file not found at %s. Please run Whisper first.", asr_path)
        sys.exit(1)

    chunks_path = Path(args.chunks_input)
    if not chunks_path.exists():
        chunks_path = Path(args.output_dir) / f"{args.video_id}_asr_chunks.json"

    ocr_path = Path(args.ocr_input)
    if not ocr_path.exists():
        ocr_path = PROJECT_ROOT / "outputs" / "smoke_test_ocr" / "per_video" / f"{args.video_id}.jsonl"

    print("=" * 80)
    print("  ASR TEXT CORRECTION PIPELINE (Qwen3-4B Local + OCR Context)")
    print(f"  Target Video       : {args.video_id}")
    print(f"  Raw ASR File       : {asr_path}")
    print(f"  OCR Context File   : {ocr_path if ocr_path.exists() else 'None (Audio-only context)'}")
    print(f"  Model              : {args.model_name}")
    print(f"  Model Cache (E:)   : {args.cache_dir}")
    print("=" * 80)

    # 2. Load Inputs
    with open(asr_path, "r", encoding="utf-8") as f:
        raw_segments = json.load(f)

    raw_chunks = []
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

    ocr_records = load_ocr_keyframes(ocr_path) if ocr_path.exists() else []

    print(f"\n[Step 1/3] Loaded {len(raw_segments)} raw ASR segments and {len(ocr_records)} OCR keyframe records.")

    # 3. Initialize Model Corrector
    print(f"\n[Step 2/3] Loading Qwen3-4B model (4-bit quantized on GPU)...")
    corrector = Qwen3ASRCorrector(
        model_name_or_path=args.model_name,
        cache_dir=args.cache_dir,
        use_4bit=args.use_4bit,
        device=args.device,
    )
    print(f"  --> Active Device       : {corrector.device_used.upper()}")
    print(f"  --> Quantization        : {corrector.quantization_used}")
    print(f"  --> Model Local Path    : {corrector.cache_dir}")

    # 4. Perform Correction
    t0 = time.time()
    corrected_segments = corrector.correct_all_segments(
        segments=raw_segments,
        ocr_records=ocr_records,
        video_id=args.video_id,
    )
    elapsed = time.time() - t0

    # 5. Save Outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corrected_asr_path = output_dir / f"{args.video_id}_asr_corrected.json"
    with open(corrected_asr_path, "w", encoding="utf-8") as f:
        json.dump(corrected_segments, f, ensure_ascii=False, indent=2)

    corrected_chunks_path = output_dir / f"{args.video_id}_asr_chunks_corrected.json"
    corrected_chunks = build_corrected_retrieval_chunks(corrected_segments, raw_chunks, args.video_id)
    with open(corrected_chunks_path, "w", encoding="utf-8") as f:
        json.dump(corrected_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[Step 3/3] Saved corrected results:")
    print(f"  --> Corrected ASR file   : {corrected_asr_path}")
    print(f"  --> Corrected Chunks file: {corrected_chunks_path}")

    # 6. Find and Print 20 Changed Segments for Manual Review
    changed_segments = []
    for seg in corrected_segments:
        r = seg["text_raw"].strip()
        n = seg["text_normalized"].strip()
        if r != n:
            changed_segments.append(seg)

    print("\n" + "=" * 80)
    print("  BÁO CÁO THỰC THI & CẤU HÌNH HỆ THỐNG")
    print("=" * 80)
    print(f"  1. Thư mục model lưu trên ổ E : {corrector.cache_dir}")
    print(f"  2. File corrected ASR         : {corrected_asr_path}")
    print(f"  3. File corrected chunks      : {corrected_chunks_path}")
    print(f"  4. GPU/CPU & Quantization     : Device={corrector.device_used.upper()} | {corrector.quantization_used}")
    print(f"  5. Tổng số segment được sửa   : {len(changed_segments)} / {len(corrected_segments)} segments")
    print(f"  6. Tổng thời gian inference   : {elapsed:.2f}s (~{elapsed / len(corrected_segments):.3f}s / segment)")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("  20 CẶP SEGMENT ĐÃ SỬA ĐỔI (RAW → NORMALIZED) ĐỂ KIỂM TRA THỦ CÔNG")
    print("=" * 80)

    sample_changes = changed_segments[:20] if len(changed_segments) >= 20 else changed_segments

    for idx, seg in enumerate(sample_changes, 1):
        ts_str = f"{format_ts(seg['start'])}–{format_ts(seg['end'])}"
        print(f"\n[{idx:02d}] {ts_str} (ID: {seg['asr_id']})")
        print(f"    RAW        : {seg['text_raw']}")
        print(f"    NORMALIZED : {seg['text_normalized']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
