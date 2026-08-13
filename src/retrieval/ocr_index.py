from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .logging_utils import setup_logger, stage_summary


def _prefix_texts(texts, model_name: str, kind: str):
    if "e5" in model_name.lower():
        prefix = "query: " if kind == "query" else "passage: "
        return [prefix + str(t) for t in texts]
    return [str(t) for t in texts]


def build_ocr_index(ocr_data_path: str | Path, global_id_map: str | Path, output_dir: str | Path, device: str = "auto", batch_size: int = 32, model_name: str = "intfloat/multilingual-e5-small", logger=None):
    ocr_data_path = Path(ocr_data_path)
    global_id_map = Path(global_id_map)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or setup_logger("ocr_index")

    started = time.time()
    df = pd.read_parquet(ocr_data_path) if ocr_data_path.suffix.lower() == ".parquet" else pd.read_csv(ocr_data_path)
    df = df.copy()
    df["combined_text"] = df.get("combined_text", "").fillna("").astype(str)
    df["ocr_status"] = df.get("ocr_status", "ok").fillna("ok").astype(str)
    df = df[df["combined_text"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No OCR records with non-empty text to index")

    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    model = SentenceTransformer(model_name, device=device)
    passages = _prefix_texts(df["combined_text"].tolist(), model_name, "passage")
    embeddings = model.encode(passages, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    global_map = pd.read_parquet(global_id_map) if Path(global_id_map).suffix.lower() == ".parquet" else pd.read_csv(global_id_map)
    if "global_id" not in df.columns and "global_id" in global_map.columns:
        merge_keys = ["video_id", "keyframe_name"] if "keyframe_name" in df.columns and "keyframe_name" in global_map.columns else ["video_id", "keyframe_path"]
        df = df.merge(global_map[["global_id", *merge_keys]], on=merge_keys, how="left")
    if "global_id" not in df.columns:
        raise RuntimeError("OCR data must include global_id or mergeable video/keyframe fields")
    missing_global_ids = int(df["global_id"].isna().sum())
    if missing_global_ids:
        raise RuntimeError(f"{missing_global_ids} OCR records could not be matched to the global id map")
    df["global_id"] = df["global_id"].astype(int)

    ocr_index_map = df[["global_id"]].copy()
    ocr_index_map.insert(0, "ocr_index_id", range(len(ocr_index_map)))

    index_path = output_dir / "l21_ocr_flat_ip.faiss"
    map_csv = output_dir / "l21_ocr_index_map.csv"
    map_parquet = output_dir / "l21_ocr_index_map.parquet"
    meta_path = output_dir / "l21_ocr_metadata.json"
    corpus_path = output_dir / "l21_ocr_corpus.parquet"
    excluded_path = output_dir / "excluded_ocr_records.json"

    faiss.write_index(index, str(index_path))
    ocr_index_map.to_csv(map_csv, index=False, encoding="utf-8-sig")
    ocr_index_map.to_parquet(map_parquet, index=False)
    df.to_parquet(corpus_path, index=False)

    meta = {
        "text_embedding_model": model_name,
        "embedding_dimension": int(index.d),
        "index_type": "IndexFlatIP",
        "metric": "IP",
        "normalized": True,
        "num_ocr_records": int(len(df)),
        "num_empty_ocr_records": int((df["combined_text"].str.strip() == "").sum()),
        "min_ocr_confidence": float(df.get("mean_confidence", pd.Series([0.0])).fillna(0.0).min()),
        "batch_size": int(batch_size),
        "device": device,
        "build_started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "build_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    excluded_path.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(stage_summary("ocr_index", "ok", input_path=str(ocr_data_path), processed=index.ntotal, output=str(index_path), elapsed=time.time() - started))
    return index, ocr_index_map, meta, df
