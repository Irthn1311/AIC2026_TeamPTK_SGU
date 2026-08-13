"""
Script 11: Parse BTC Objects & Build Object Index (Branch 3)
============================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.preprocessing.btc_object_parser import build_btc_objects_corpus
from src.retrieval.logging_utils import setup_logger, timestamp_token
from src.retrieval.object_index import build_object_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "datasets_L21"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "indexes" / "object"))
    parser.add_argument("--min-score", type=float, default=0.25)
    args = parser.parse_args()

    log_file = PROJECT_ROOT / "outputs" / "logs" / f"multimodal_object_index_{timestamp_token()}.log"
    logger = setup_logger("object-build", log_file)

    output_dir = Path(args.output_dir)
    logger.info("=== STEP 1: Parsing BTC Object JSONs ===")
    df, meta_parse = build_btc_objects_corpus(args.dataset_root, output_dir, min_score=args.min_score, logger_inst=logger)

    logger.info("=== STEP 2: Building Object Inverted Index ===")
    index_inst, meta_idx = build_object_index(output_dir / "l21_objects.parquet", output_dir, logger_inst=logger)

    logger.info("SUCCESS: Branch 3 (BTC Objects Index) completed! Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
