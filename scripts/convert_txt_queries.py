"""
Converter script — converts a directory of individual BTC query .txt files
into a single standardized queries.json file.

Usage:
    python scripts/convert_txt_queries.py \
        --input-dir D:/Dataset_AIC/queries_txt \
        --output datasets/queries/converted_queries.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.query_loader import load_queries
from src.utils.logger import get_logger

logger = get_logger("convert_txt_queries")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of individual query .txt files to a queries.json file"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Path to folder containing .txt query files (e.g. query-p1-5-kis.txt)")
    parser.add_argument("--output", default="datasets/queries/queries_from_txt.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    queries = load_queries(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully converted {len(queries)} query text files → {output_path}")


if __name__ == "__main__":
    main()
