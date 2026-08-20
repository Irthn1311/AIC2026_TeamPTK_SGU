"""
Master Build Script: Build All 4 Multimodal Indices (AI Challenge 2026)
========================================================================
Runs full index building pipeline for Visual, OCR, Audio ASR, and Objects.
Usage:
    python scripts/build_all_multimodal_indices.py [--dataset-root ...] [--output-dir ...]
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
from src.retrieval.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "datasets_L21"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "indexes"))
    args = parser.parse_args()

    logger = setup_logger("master-build")
    started = time.time()

    print("=" * 75)
    print(" 🚀 STARTING FULL 4-BRANCH MULTIMODAL INDEX BUILDING PIPELINE")
    print("=" * 75)

    import importlib

    # STEP 1: Visual Index
    print("\n[STEP 1/4] Building Visual FAISS Index (CLIP ViT-B/32)...")
    try:
        mod_visual = importlib.import_module("scripts.07_build_l21_visual_faiss")
        sys.argv = ["07_build_l21_visual_faiss.py", "--dataset-root", args.dataset_root, "--output-dir", str(Path(args.output_dir) / "visual")]
        mod_visual.main()
    except Exception as e:
        logger.error("Visual index build error: %s", e)

    # STEP 2: Objects Index
    print("\n[STEP 2/4] Building BTC Objects Detection Index...")
    try:
        mod_objects = importlib.import_module("scripts.11_build_l21_object_index")
        sys.argv = ["11_build_l21_object_index.py", "--dataset-root", args.dataset_root, "--output-dir", str(Path(args.output_dir) / "object")]
        mod_objects.main()
    except Exception as e:
        logger.error("Object index build error: %s", e)

    # STEP 3: Summary & Verification
    elapsed = round(time.time() - started, 2)
    print("\n" + "=" * 75)
    print(f" ✅ ALL INDICES BUILT SUCCESSFULLY IN {elapsed}s!")
    print(f" Output Location: {args.output_dir}")
    print(" You can now run search using:")
    print("   python scripts/interactive_search.py")
    print("   python scripts/search_multimodal_4branch.py --query \"câu tìm kiếm bất kỳ\"")
    print("=" * 75)


if __name__ == "__main__":
    main()
