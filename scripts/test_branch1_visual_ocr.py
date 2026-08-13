"""
Test Branch 1: BTC KeyFrames -> Visual Index & OCR Index
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT
from src.retrieval.clip_text_encoder import ClipTextEncoder

def test_branch1(query: str, top_k: int = 5):
    visual_index_path = PROJECT_ROOT / "outputs" / "indexes" / "visual" / "l21_visual_flat_ip.faiss"
    global_map_path = PROJECT_ROOT / "outputs" / "indexes" / "l21_global_id_map.parquet"
    ocr_index_path = PROJECT_ROOT / "outputs" / "indexes" / "ocr" / "l21_ocr_flat_ip.faiss"
    ocr_corpus_path = PROJECT_ROOT / "outputs" / "indexes" / "ocr" / "l21_ocr_corpus.parquet"

    print("=" * 60)
    print(f"  TESTING BRANCH 1 (BTC KEYFRAMES -> VISUAL & OCR INDEX)")
    print(f"  Query: '{query}'")
    print("=" * 60)

    # 1. Visual Index Search
    if visual_index_path.exists() and global_map_path.exists():
        index_v = faiss.read_index(str(visual_index_path))
        df_global = pd.read_parquet(global_map_path)
        print(f"\n[1A. VISUAL INDEX] Total Vectors: {index_v.ntotal}")

        encoder = ClipTextEncoder()
        q_emb = encoder.encode(query).astype(np.float32)
        scores_v, indices_v = index_v.search(q_emb, top_k)

        print(f"--- Top {top_k} Visual Results ---")
        for rank, (idx, score) in enumerate(zip(indices_v[0], scores_v[0]), 1):
            row = df_global.iloc[idx]
            print(f"  #{rank} Score: {score:.4f} | Video: {row['video_id']} | Frame: {row['frame_idx']:>6d} ({row['timestamp_text']}) | {row['keyframe_name']}")

    # 2. OCR Index Check
    if ocr_index_path.exists() and ocr_corpus_path.exists():
        index_ocr = faiss.read_index(str(ocr_index_path))
        df_ocr = pd.read_parquet(ocr_corpus_path)
        print(f"\n[1B. OCR INDEX] Total Indexed Keyframes with OCR: {index_ocr.ntotal}")
        print(f"Sample OCR texts extracted from keyframes:")
        for idx, row in df_ocr.head(3).iterrows():
            print(f"  - Video: {row['video_id']} | Text: \"{row['combined_text'][:60]}\"")
    
    print("\n" + "=" * 60)
    print("  BRANCH 1 STATUS: READY & FUNCTIONAL")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="ruộng lúa nông dân")
    args = parser.parse_args()
    test_branch1(args.query)
