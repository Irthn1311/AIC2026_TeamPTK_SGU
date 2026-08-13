from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.retrieval.dataset_inspector import inspect_l21_dataset
from src.retrieval.logging_utils import setup_logger, timestamp_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "reports"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_file = log_dir / f"multimodal_inspect_{timestamp_token()}.log"
    logger = setup_logger("inspect", log_file)

    df, meta = inspect_l21_dataset(args.dataset_root, PROJECT_ROOT, output_dir)
    df.to_csv(output_dir / "l21_multimodal_inventory.csv", index=False, encoding="utf-8-sig")
    (output_dir / "l21_multimodal_summary.json").write_text(json.dumps(meta["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "l21_folder_tree.txt").write_text(meta["folder_tree"], encoding="utf-8")
    logger.info("Wrote inventory to %s", output_dir)


if __name__ == "__main__":
    main()

