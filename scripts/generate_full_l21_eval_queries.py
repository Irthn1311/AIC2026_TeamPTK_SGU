"""
Generate Comprehensive Evaluation Queries across ALL 29 L21 Videos
===================================================================
Automatically samples valid headlines & tickers from all 29 video OCR corpora,
generating Exact, Typo/No-Accent, and Keyword queries for full-dataset evaluation.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.preprocessing.ocr_temporal_merger import remove_vietnamese_accents, normalize_text_search


def generate_full_queries():
    corpus_path = PROJECT_ROOT / "outputs" / "indexes" / "ocr_v3" / "l21_ocr_v3_corpus.parquet"
    if not corpus_path.exists():
        print(f"Error: {corpus_path} does not exist.")
        return

    df = pd.read_parquet(corpus_path)
    
    # Filter for high-quality headline & ticker segments (> 15 chars)
    df_filtered = df[
        (df["region_type"].isin(["headline", "ticker"])) &
        (df["text_consensus"].str.len() >= 15)
    ].copy()

    random.seed(42)
    queries = []
    q_id = 1

    # Sample 2-3 queries per video
    for vid, group in df_filtered.groupby("video_id"):
        sample_rows = group.sample(n=min(3, len(group)), random_state=42)
        for _, row in sample_rows.iterrows():
            raw_text = str(row["text_consensus"]).strip()
            no_acc = remove_vietnamese_accents(raw_text).lower()
            
            # Select target keyword (longest word / phrase)
            words = [w for w in raw_text.split() if len(w) >= 3]
            if not words:
                continue
            target_kw = max(words, key=len)

            # 1. Exact Query
            queries.append({
                "query_id": f"Q{q_id:03d}_exact",
                "type": "exact",
                "query": raw_text,
                "target_video": vid,
                "target_keyword": target_kw,
            })

            # 2. Typo / No Accent Query
            queries.append({
                "query_id": f"Q{q_id:03d}_typo",
                "type": "typo",
                "query": no_acc,
                "target_video": vid,
                "target_keyword": remove_vietnamese_accents(target_kw).lower(),
            })

            # 3. Keyword/Phrase Query (first 3-4 words)
            phrase = " ".join(words[:4])
            queries.append({
                "query_id": f"Q{q_id:03d}_keyword",
                "type": "semantic",
                "query": phrase,
                "target_video": vid,
                "target_keyword": target_kw,
            })

            q_id += 1

    out_file = PROJECT_ROOT / "outputs" / "evaluation" / "ocr_v3" / "full_l21_eval_queries.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Generated {len(queries)} evaluation queries spanning {df_filtered['video_id'].nunique()} videos!")
    print(f"📄 Saved to: {out_file}")
    return queries


if __name__ == "__main__":
    generate_full_queries()
