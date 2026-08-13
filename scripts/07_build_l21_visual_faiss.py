from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.retrieval.logging_utils import setup_logger, timestamp_token
from src.retrieval.visual_index import build_visual_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "indexes" / "visual"))
    parser.add_argument("--id-map-output", default=str(PROJECT_ROOT / "outputs" / "indexes" / "l21_global_id_map.parquet"))
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    log_file = PROJECT_ROOT / "outputs" / "logs" / f"multimodal_visual_index_{timestamp_token()}.log"
    logger = setup_logger("visual-build", log_file)
    build_visual_index(args.dataset_root, args.output_dir, args.id_map_output, batch_size=args.batch_size, logger=logger)


if __name__ == "__main__":
    main()

