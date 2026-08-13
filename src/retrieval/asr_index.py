"""
ASR Search Index Module (Branch 2)
==================================
Builds FAISS Index + Text Search for ASR Transcriptions.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval.logging_utils import setup_logger

logger = logging.getLogger(__name__)

def build_asr_index(
    asr_corpus_path: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
    model_name: str = "intfloat/multilingual-e5-small",
    logger_inst=None,
) -> tuple[faiss.IndexFlatIP, pd.DataFrame, dict[str, Any]]:
    asr_corpus_path = Path(asr_corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_inst = logger_inst or setup_logger("asr_index")
    started = time.time()

    df = pd.read_parquet(asr_corpus_path) if asr_corpus_path.suffix.lower() == ".parquet" else pd.read_csv(asr_corpus_path)
    df["asr_text"] = df.get("asr_text", "").fillna("").astype(str)
    
    # Filter non-empty ASR keyframes
    df_filtered = df[df["asr_text"].str.strip() != ""].reset_index(drop=True)
    if df_filtered.empty:
        log_inst.warning("No non-empty ASR texts found in corpus")
        index = faiss.IndexFlatIP(384)
        return index, df, {}

    model = SentenceTransformer(model_name, device=device)
    passages = [f"passage: {t}" for t in df_filtered["asr_text"]]
    
    embeddings = model.encode(passages, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save outputs
    faiss_path = output_dir / "l21_asr_flat_ip.faiss"
    corpus_path = output_dir / "l21_asr_corpus.parquet"
    meta_path = output_dir / "l21_asr_metadata.json"

    faiss.write_index(index, str(faiss_path))
    df_filtered.to_parquet(corpus_path, index=False)

    meta = {
        "text_model": model_name,
        "embedding_dim": int(index.d),
        "total_asr_vectors": int(index.ntotal),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log_inst.info("Built ASR FAISS index with %d vectors in %s", index.ntotal, faiss_path)

    return index, df_filtered, meta
