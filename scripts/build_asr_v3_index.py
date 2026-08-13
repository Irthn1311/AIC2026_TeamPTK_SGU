"""
Build ASR V3 Hybrid Retrieval Index (FAISS + Parquet Corpus)
============================================================
Aggregates all corrected ASR retrieval chunks from outputs/asr/
and encodes dense embeddings using multilingual-e5-small.
"""

from __future__ import annotations

import glob
import json
import time
import argparse
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from _bootstrap import PROJECT_ROOT
from src.preprocessing.ocr_temporal_merger import normalize_text_search, remove_vietnamese_accents


def build_asr_v3_index(
    asr_dir: Path | str = None,
    output_dir: Path | str = None,
    model_name: str = "intfloat/multilingual-e5-small",
    batch_size: int = 32,
):
    if asr_dir is None:
        asr_dir = PROJECT_ROOT / "outputs" / "asr"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "indexes" / "asr_v3"

    asr_dir = Path(asr_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ASR chunks from: {asr_dir}...")
    
    # Collect all video IDs
    all_raw_chunk_files = sorted(list(asr_dir.glob("*_asr_chunks.json")))
    all_corrected_chunk_files = {p.stem.replace("_asr_chunks_corrected", ""): p for p in asr_dir.glob("*_asr_chunks_corrected.json")}

    all_chunks = []
    loaded_files = 0
    for raw_f in all_raw_chunk_files:
        vid_id = raw_f.stem.replace("_asr_chunks", "")
        # Prefer corrected chunks if available, else raw chunks
        chosen_file = all_corrected_chunk_files.get(vid_id, raw_f)
        try:
            with open(chosen_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_chunks.extend(data)
                    loaded_files += 1
        except Exception as e:
            print(f"Error reading {chosen_file}: {e}")

    if not all_chunks:
        print("No ASR chunks found!")
        return

    df = pd.DataFrame(all_chunks)
    print(f"Loaded {len(df)} ASR chunks from {loaded_files} video file(s).")

    # Robust text extraction helper
    def extract_clean_text(r) -> str:
        for col in ["text_normalized", "text_raw", "text"]:
            if col in r:
                val = r[col]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    s = str(val).strip()
                    if s and s.lower() not in ("none", "nan", "null"):
                        return s
        return ""

    # Populate standardized text columns
    clean_texts = [extract_clean_text(row) for _, row in df.iterrows()]
    df["text_clean"] = clean_texts
    
    # Ensure text_raw and text_normalized are populated
    if "text_raw" not in df.columns:
        df["text_raw"] = df["text_clean"]
    else:
        df["text_raw"] = df["text_raw"].fillna(df["text_clean"])
        
    if "text_normalized" not in df.columns:
        df["text_normalized"] = df["text_clean"]
    else:
        df["text_normalized"] = df["text_normalized"].fillna(df["text_clean"])

    if "text" not in df.columns:
        df["text"] = df["text_clean"]
    else:
        df["text"] = df["text"].fillna(df["text_clean"])

    # Add search normalization
    df["text_search"] = df["text_clean"].map(normalize_text_search)
    df["text_search_no_accent"] = df["text_search"].map(remove_vietnamese_accents)

    # Encode with Multilingual-E5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(df)} chunks using '{model_name}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    passages = [f"passage: {t}" for t in df["text_search"]]
    embeddings_list = []

    t0 = time.time()
    for i in range(0, len(passages), batch_size):
        batch_texts = passages[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean / CLS pooling + normalization
            emb = outputs.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()
            embeddings_list.append(emb)

    embeddings = np.vstack(embeddings_list).astype(np.float32)
    dim = embeddings.shape[1]

    # Build FAISS Flat IP Index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save to disk
    corpus_path = output_dir / "l21_asr_v3_corpus.parquet"
    faiss_path = output_dir / "l21_asr_v3_flat_ip.faiss"
    meta_path = output_dir / "l21_asr_v3_metadata.json"

    df.to_parquet(corpus_path, index=False)
    faiss.write_index(index, str(faiss_path))

    metadata = {
        "model_name": model_name,
        "total_chunks": len(df),
        "embedding_dim": dim,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Successfully built ASR V3 Index:")
    print(f"   - Corpus:  {corpus_path} ({len(df)} records)")
    print(f"   - FAISS:   {faiss_path} ({index.ntotal} vectors)")
    print(f"   - Metadata:{meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ASR V3 FAISS + parquet index from ASR chunk JSON files.")
    parser.add_argument("--asr-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    build_asr_v3_index(
        asr_dir=args.asr_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
