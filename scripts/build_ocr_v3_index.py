"""
Build OCR V3 Segment Retrieval Index (5 Benchmark Videos)
=========================================================
Runs keyframe OCR processing + temporal merging across 5 videos (L21_V001 -> L21_V005),
encodes text using multilingual-e5-small into FAISS IndexFlatIP, and saves:
  1. outputs/indexes/ocr_v3/l21_ocr_v3_corpus.parquet
  2. outputs/indexes/ocr_v3/l21_ocr_v3_flat_ip.faiss
  3. outputs/indexes/ocr_v3/l21_ocr_v3_segments.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.preprocessing.keyframe_ocr import extract_keyframe_ocr
from src.preprocessing.ocr_temporal_merger import merge_video_ocr_records


def build_ocr_v3_full_index():
    print("=" * 80)
    print(" 🚀 BUILDING FULL OCR V3 SEGMENT FAISS INDEX (All Available Videos)")
    print("=" * 80)

    output_ocr_raw = PROJECT_ROOT / "outputs" / "ocr_full"
    per_video_dir = output_ocr_raw / "per_video"
    jsonl_files = sorted(list(per_video_dir.glob("*.jsonl")))
    all_segments = []

    print(f"Found {len(jsonl_files)} per-video OCR JSONL files in {per_video_dir}...")
    for jf in jsonl_files:
        records = [json.loads(line) for line in jf.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        segments = merge_video_ocr_records(records, max_gap_seconds=3.0, min_bbox_iou=0.30, min_text_similarity=0.70)
        all_segments.extend([seg.to_dict() for seg in segments])

    print(f"\n✅ Total Merged OCR Segments across {len(jsonl_files)} videos: {len(all_segments)}")

    # Save OCR Segments JSONL & Parquet
    out_idx_dir = PROJECT_ROOT / "outputs" / "indexes" / "ocr_v3"
    out_idx_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_segments)
    parquet_path = out_idx_dir / "l21_ocr_v3_corpus.parquet"
    jsonl_path = out_idx_dir / "l21_ocr_v3_segments.jsonl"

    df.to_parquet(parquet_path, index=False)
    jsonl_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_segments), encoding="utf-8")
    print(f"📄 Corpus saved to: {parquet_path}")

    # Step 3. Build FAISS E5 Embeddings
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "intfloat/multilingual-e5-small"
    print(f"\nEncoding {len(all_segments)} OCR segments using {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    passage_texts = [f"passage: {seg['text_consensus']}" for seg in all_segments]
    batch_size = 32
    embeddings_list = []

    with torch.no_grad():
        for i in range(0, len(passage_texts), batch_size):
            batch = passage_texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            embeds = outputs.last_hidden_state[:, 0, :]
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1).cpu().numpy()
            embeddings_list.append(embeds)

    embeddings = np.vstack(embeddings_list).astype("float32")
    dimension = embeddings.shape[1]

    # Create FAISS IndexFlatIP
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss_path = out_idx_dir / "l21_ocr_v3_flat_ip.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"⚡ FAISS index created ({index.ntotal} vectors, {dimension}d) at: {faiss_path}")
    return parquet_path, faiss_path


if __name__ == "__main__":
    build_ocr_v3_full_index()

