from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.preprocessing.keyframe_ocr import extract_keyframe_ocr
from src.retrieval.logging_utils import setup_logger, timestamp_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "interim" / "l21_ocr"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--video-id", action="append", default=None, help="Run OCR only for this video id. Can be repeated.")
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_file = log_dir / f"multimodal_ocr_{timestamp_token()}.log"
    logger = setup_logger("ocr", log_file)
    extract_keyframe_ocr(
        args.dataset_root,
        args.output_dir,
        device=args.device,
        resume=args.resume,
        min_confidence=args.min_confidence,
        max_images=args.max_images,
        video_ids=args.video_id,
        logger=logger,
    )


if __name__ == "__main__":
    main()
