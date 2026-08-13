from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ocr_temporal_tracker_v3 import normalize_for_match


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def build_corpus(documents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in documents.reset_index(drop=True).iterrows():
        text_search = str(row.get("bm25_text", "") or row.get("corrected_text", "") or row.get("consensus_text", ""))
        semantic_text = str(row.get("semantic_search_text", "") or row.get("corrected_text", "") or row.get("consensus_text", ""))
        start_time = float(row.get("start_time", 0.0) or 0.0)
        end_time = float(row.get("end_time", start_time) or start_time)
        frame_id = int(row.get("frame_id", 0) or 0)
        rows.append(
            {
                "ocr_doc_id": idx,
                "document_id": row.get("document_id", row.get("track_id", f"ocrtempv3_{idx:08d}")),
                "track_id": row.get("track_id", ""),
                "video_id": row.get("video_id", ""),
                "start_time": start_time,
                "end_time": end_time,
                "frame_idx": frame_id,
                "global_id": int(row.get("global_id", row.get("frame_id", 0)) or 0),
                "keyframe_name": row.get("keyframe_name", ""),
                "image_path": row.get("image_path", ""),
                "region_type": row.get("region_type", ""),
                "combined_text": semantic_text,
                "text_consensus": row.get("consensus_text", semantic_text),
                "text_search": text_search,
                "text_search_no_accent": normalize_for_match(text_search),
                "semantic_search_text": semantic_text,
                "raw_searchable_text": row.get("raw_searchable_text", ""),
                "corrected_text": row.get("corrected_text", semantic_text),
                "mean_confidence": float(row.get("track_confidence", 0.0) or 0.0),
                "num_text_boxes": 1,
                "ocr_status": "ok",
                "detections": "[]",
                "needs_review": bool(row.get("needs_review", False)),
            }
        )
    return pd.DataFrame(rows)


def encode_passages(texts: list[str], model_name: str, batch_size: int, device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    vectors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encode OCR Temporal V3 passages"):
            batch = [f"passage: {text}" for text in texts[i : i + batch_size]]
            inputs = tokenizer(batch, return_tensors="pt", max_length=192, truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            embeds = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            vectors.append(embeds.cpu().numpy())
    return np.vstack(vectors).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS OCR index from OCR Temporal V3 retrieval documents.")
    parser.add_argument("--documents", default="outputs/ocr_temporal_v3_full_tracking/l21_ocr_documents.parquet")
    parser.add_argument("--output-dir", default="outputs/indexes/ocr_temporal_v3_full_tracking")
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    started = time.time()
    documents_path = resolve_path(args.documents)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not documents_path.exists():
        raise FileNotFoundError(f"Missing OCR Temporal V3 documents: {documents_path}")

    documents = pd.read_parquet(documents_path)
    corpus = build_corpus(documents)
    if corpus.empty:
        raise ValueError(f"No OCR documents found in {documents_path}")

    device = "cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    vectors = encode_passages(corpus["semantic_search_text"].astype(str).tolist(), args.model_name, args.batch_size, device)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    corpus_path = output_dir / "l21_ocr_temporal_v3_corpus.parquet"
    index_path = output_dir / "l21_ocr_temporal_v3_flat_ip.faiss"
    map_path = output_dir / "l21_ocr_temporal_v3_index_map.parquet"
    meta_path = output_dir / "l21_ocr_temporal_v3_metadata.json"

    corpus.to_parquet(corpus_path, index=False)
    faiss.write_index(index, str(index_path))
    corpus[["ocr_doc_id", "global_id", "video_id", "frame_idx", "track_id", "start_time", "end_time"]].to_parquet(map_path, index=False)
    meta = {
        "documents": str(documents_path),
        "output_dir": str(output_dir),
        "rows": int(len(corpus)),
        "faiss_vectors": int(index.ntotal),
        "dimension": int(index.d),
        "model_name": args.model_name,
        "device": device,
        "runtime_sec": round(time.time() - started, 2),
        "outputs": {
            "corpus": str(corpus_path),
            "faiss": str(index_path),
            "index_map": str(map_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
