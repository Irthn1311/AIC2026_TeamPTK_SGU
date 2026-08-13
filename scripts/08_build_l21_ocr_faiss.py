from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.retrieval.logging_utils import setup_logger, timestamp_token
from src.retrieval.ocr_index import build_ocr_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-data", required=True)
    parser.add_argument("--global-id-map", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "indexes" / "ocr"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()
    log_file = PROJECT_ROOT / "outputs" / "logs" / f"multimodal_ocr_index_{timestamp_token()}.log"
    logger = setup_logger("ocr-build", log_file)
    build_ocr_index(args.ocr_data, args.global_id_map, args.output_dir, device=args.device, batch_size=args.batch_size, model_name=args.model_name, logger=logger)


if __name__ == "__main__":
    main()

