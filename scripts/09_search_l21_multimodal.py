from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from _bootstrap import PROJECT_ROOT
from src.retrieval.logging_utils import setup_logger, timestamp_token
from src.retrieval.search_engine import MultimodalSearchEngine
from src.retrieval.visualization import make_contact_sheet
from PIL import Image


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _print_results(df: pd.DataFrame):
    cols = [c for c in ["rank", "fused_score", "visual_score", "ocr_score", "lexical_score", "video_id", "timestamp_text", "frame_idx", "ocr_text", "keyframe_path"] if c in df.columns]
    print(df[cols].to_string(index=False))


def _build_sheet(df: pd.DataFrame, output_path: Path):
    items = []
    for _, row in df.head(20).iterrows():
        img = None
        try:
            img = Image.open(row["keyframe_path"]).convert("RGB")
        except Exception:
            pass
        label = f"#{int(row['rank'])} {row['fused_score']:.3f}\nV:{row.get('visual_score',0):.3f} O:{row.get('ocr_score',0):.3f}\n{row.get('video_id','')} {row.get('timestamp_text','')}\n{str(row.get('ocr_text',''))[:80]}"
        items.append({"image": img, "label": label})
    make_contact_sheet(items, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=None)
    parser.add_argument("--visual-index", required=True)
    parser.add_argument("--ocr-index", default=None)
    parser.add_argument("--global-id-map", required=True)
    parser.add_argument("--ocr-index-map", default=None)
    parser.add_argument("--ocr-corpus", default=None)
    parser.add_argument("--visual-weight", type=float, default=0.7)
    parser.add_argument("--ocr-weight", type=float, default=0.25)
    parser.add_argument("--lexical-weight", type=float, default=0.05)
    parser.add_argument("--fusion-mode", default="weighted_sum")
    parser.add_argument("--dedup-window", type=float, default=5.0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "retrieval"))
    parser.add_argument("--video-filter", default=None)
    args = parser.parse_args()

    run_id = timestamp_token()
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "search.log"
    logger = setup_logger("search", log_file)
    ocr_corpus = args.ocr_corpus
    if ocr_corpus is None and args.ocr_index:
        inferred = Path(args.ocr_index).with_name("l21_ocr_corpus.parquet")
        if inferred.exists():
            ocr_corpus = str(inferred)
    engine = MultimodalSearchEngine(
        args.visual_index,
        args.global_id_map,
        ocr_index_path=args.ocr_index,
        ocr_index_map_path=args.ocr_index_map,
        ocr_corpus_path=ocr_corpus,
    )
    result = engine.search(
        args.query,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        visual_weight=args.visual_weight,
        ocr_weight=args.ocr_weight,
        lexical_weight=args.lexical_weight,
        fusion_mode=args.fusion_mode,
        dedup_window_seconds=args.dedup_window,
        dedup=True,
        video_filter=args.video_filter,
    )
    raw = result["fused_raw"]
    dedup = result["fused_dedup"]
    raw.to_csv(run_dir / "results_raw.csv", index=False, encoding="utf-8-sig")
    dedup.to_csv(run_dir / "results_deduplicated.csv", index=False, encoding="utf-8-sig")
    payload = {
        "query": args.query,
        "top_k": args.top_k,
        "candidate_pool": result["candidate_pool"],
        "fusion_mode": args.fusion_mode,
        "visual_weight": args.visual_weight,
        "ocr_weight": args.ocr_weight,
        "lexical_weight": args.lexical_weight,
        "dedup_window_seconds": args.dedup_window,
        "visual_model": "ViT-B-32/openai",
        "ocr_embedding_model": "intfloat/multilingual-e5-small",
        "search_time_ms": result["search_time_ms"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "query.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps({"raw": raw.to_dict(orient="records"), "dedup": dedup.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _build_sheet(raw, run_dir / "contact_sheet_raw.jpg")
    _build_sheet(dedup, run_dir / "contact_sheet_deduplicated.jpg")
    _print_results(raw)
    logger.info("Wrote results to %s", run_dir)


if __name__ == "__main__":
    main()
