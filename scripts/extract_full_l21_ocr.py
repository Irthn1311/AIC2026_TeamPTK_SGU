"""
Full OCR Extraction & Index Building Script for All Keyframes
=============================================================
Runs OCR extraction across ALL local keyframe images and builds
the OCR corpus parquet and FAISS index.

Usage:
    python scripts/extract_full_l21_ocr.py [--max-images N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.preprocessing.keyframe_ocr import extract_keyframe_ocr
from src.retrieval.logging_utils import setup_logger
from build_ocr_v3_index import build_ocr_v3_full_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=r"e:\AI Challenge TP.HCM 2026\CodeBase\datasets_L21")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "indexes" / "ocr_v3"))
    parser.add_argument("--device", default="auto", help="Device to run on ('cuda', 'gpu', 'cpu', or 'auto')")
    parser.add_argument("--max-images", type=int, default=None, help="Set maximum images or omit for ALL")
    parser.add_argument("--force", action="store_true", help="Force re-extract all keyframes without loading cache")
    args = parser.parse_args()

    logger = setup_logger("ocr-full-extract")

    import torch
    use_device = args.device
    if use_device == "auto":
        use_device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 75)
    print(f" 🚀 RUNNING FULL KEYFRAME OCR EXTRACTION & FAISS INDEXING (Device: {use_device.upper()}, Force: {args.force})")
    print("=" * 75)

    ocr_out = PROJECT_ROOT / "outputs" / "ocr_full"
    ocr_out.mkdir(parents=True, exist_ok=True)

    # STEP 1: Extract OCR across keyframes
    df, meta = extract_keyframe_ocr(
        dataset_root=args.dataset_root,
        output_dir=ocr_out,
        device=use_device,
        resume=not args.force,
        max_images=args.max_images,
        logger=logger,
    )

    print(f"\n✅ Đã trích xuất xong OCR cho {len(df)} Keyframes!")

    # STEP 2: Build FAISS Index for OCR V3
    print("\n[STEP 2] Building FAISS Index for OCR Corpus V3...")
    build_ocr_v3_full_index()
    print("\n🎉 HOÀN THÀNH BỘ CHỈ MỤC OCR FULL V3 FAISS INDEX!")


if __name__ == "__main__":
    main()
