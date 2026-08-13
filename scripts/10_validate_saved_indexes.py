from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import faiss
import pandas as pd

from _bootstrap import PROJECT_ROOT
from src.retrieval.logging_utils import setup_logger, timestamp_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-index", required=True)
    parser.add_argument("--ocr-index", required=True)
    parser.add_argument("--global-id-map", required=True)
    parser.add_argument("--ocr-index-map", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "reports"))
    args = parser.parse_args()
    log_file = PROJECT_ROOT / "outputs" / "logs" / f"multimodal_validate_{timestamp_token()}.log"
    logger = setup_logger("validate", log_file)

    visual = faiss.read_index(args.visual_index)
    ocr = faiss.read_index(args.ocr_index)
    gmap = pd.read_parquet(args.global_id_map) if args.global_id_map.endswith(".parquet") else pd.read_csv(args.global_id_map)
    omap = pd.read_parquet(args.ocr_index_map) if args.ocr_index_map.endswith(".parquet") else pd.read_csv(args.ocr_index_map)
    checks = {
        "visual_ntotal": int(visual.ntotal),
        "ocr_ntotal": int(ocr.ntotal),
        "global_map_rows": int(len(gmap)),
        "ocr_map_rows": int(len(omap)),
        "visual_dim": int(visual.d),
        "ocr_dim": int(ocr.d),
        "global_id_unique": bool(gmap["global_id"].is_unique),
        "ocr_index_id_unique": bool(omap["ocr_index_id"].is_unique),
        "ocr_global_id_subset": bool(set(omap["global_id"]).issubset(set(gmap["global_id"]))),
    }
    samples = gmap.sample(min(20, len(gmap)), random_state=42) if not gmap.empty else pd.DataFrame()
    sample_checks = []
    for _, row in samples.iterrows():
        sample_checks.append({"global_id": int(row["global_id"]), "keyframe_exists": Path(row["keyframe_path"]).exists()})
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "l21_saved_index_validation.json").write_text(json.dumps({"checks": checks, "samples": sample_checks}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([checks]).to_csv(out_dir / "l21_saved_index_validation.csv", index=False, encoding="utf-8-sig")
    logger.info("Validation written to %s", out_dir)


if __name__ == "__main__":
    main()

