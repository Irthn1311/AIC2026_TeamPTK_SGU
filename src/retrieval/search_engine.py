from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from .clip_text_encoder import ClipTextEncoder
from .multimodal_fusion import fuse_candidates


class MultimodalSearchEngine:
    def __init__(self, visual_index_path: str | Path, global_id_map_path: str | Path, ocr_index_path: str | Path | None = None, ocr_index_map_path: str | Path | None = None, ocr_corpus_path: str | Path | None = None, clip_model: str = "ViT-B-32", clip_pretrained: str = "openai", ocr_model_name: str = "intfloat/multilingual-e5-small", device: str = "auto"):
        self.visual_index_path = Path(visual_index_path)
        self.global_id_map_path = Path(global_id_map_path)
        self.ocr_index_path = Path(ocr_index_path) if ocr_index_path else None
        self.ocr_index_map_path = Path(ocr_index_map_path) if ocr_index_map_path else None
        self.ocr_corpus_path = Path(ocr_corpus_path) if ocr_corpus_path else None
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.ocr_model_name = ocr_model_name
        self.device = device
        self._visual_index = faiss.read_index(str(self.visual_index_path))
        self._global_map = pd.read_parquet(self.global_id_map_path) if self.global_id_map_path.suffix.lower() == ".parquet" else pd.read_csv(self.global_id_map_path)
        self._ocr_index = faiss.read_index(str(self.ocr_index_path)) if self.ocr_index_path and self.ocr_index_path.exists() else None
        self._ocr_index_map = pd.read_parquet(self.ocr_index_map_path) if self.ocr_index_map_path and self.ocr_index_map_path.exists() and self.ocr_index_map_path.suffix.lower() == ".parquet" else (pd.read_csv(self.ocr_index_map_path) if self.ocr_index_map_path and self.ocr_index_map_path.exists() else None)
        self._ocr_corpus = pd.read_parquet(self.ocr_corpus_path) if self.ocr_corpus_path and self.ocr_corpus_path.exists() else None
        self._clip = ClipTextEncoder(model_name=clip_model, pretrained=clip_pretrained, device="cpu")

    @property
    def visual_dim(self) -> int:
        return int(self._visual_index.d)

    @property
    def ocr_dim(self) -> int | None:
        return int(self._ocr_index.d) if self._ocr_index is not None else None

    def _visual_search(self, query: str, top_k: int) -> pd.DataFrame:
        q = self._clip.encode([query])[0]
        if q.shape[0] != self.visual_dim:
            raise ValueError(f"Query dim {q.shape[0]} does not match visual index dim {self.visual_dim}")
        scores, ids = self._visual_index.search(q.reshape(1, -1).astype(np.float32), top_k)
        ids = ids[0]
        scores = scores[0]
        rows = []
        for rank, (gid, score) in enumerate(zip(ids, scores), start=1):
            if gid < 0:
                continue
            row = self._global_map.iloc[int(gid)].to_dict()
            row.update({"visual_raw_score": float(score), "visual_rank": rank})
            rows.append(row)
        return pd.DataFrame(rows)

    def _ocr_search(self, query: str, top_k: int) -> pd.DataFrame:
        if self._ocr_index is None or self._ocr_index_map is None or self._ocr_corpus is None:
            return pd.DataFrame()
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.ocr_model_name, device="cpu")
        q_text = query
        if "e5" in self.ocr_model_name.lower():
            q_text = "query: " + q_text
        q = model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        if q.shape[0] != self._ocr_index.d:
            raise ValueError(f"Query dim {q.shape[0]} does not match OCR index dim {self._ocr_index.d}")
        scores, ids = self._ocr_index.search(q.reshape(1, -1), top_k)
        rows = []
        for rank, (oid, score) in enumerate(zip(ids[0], scores[0]), start=1):
            if oid < 0:
                continue
            map_row = self._ocr_index_map.iloc[int(oid)].to_dict()
            global_id = int(map_row["global_id"])
            global_row = self._global_map.iloc[global_id].to_dict()
            ocr_row = self._ocr_corpus.iloc[int(oid)].to_dict()
            row = dict(global_row)
            row["ocr_text"] = str(ocr_row.get("combined_text", row.get("ocr_text", "")))
            row["ocr_mean_confidence"] = float(ocr_row.get("mean_confidence", row.get("ocr_mean_confidence", 0.0)) or 0.0)
            row["ocr_num_boxes"] = int(ocr_row.get("num_text_boxes", row.get("ocr_num_boxes", 0)) or 0)
            row["ocr_status"] = ocr_row.get("ocr_status", "")
            row["ocr_detections"] = ocr_row.get("detections", [])
            row.update({"global_id": global_id, "ocr_raw_score": float(score), "ocr_rank": rank})
            rows.append(row)
        return pd.DataFrame(rows)

    def search(self, query: str, top_k: int = 20, candidate_pool: int | None = None, visual_weight: float = 0.7, ocr_weight: float = 0.25, lexical_weight: float = 0.05, fusion_mode: str = "weighted_sum", dedup_window_seconds: float = 5.0, dedup: bool = True, video_filter: str | None = None) -> dict:
        started = time.time()
        candidate_pool = candidate_pool or max(200, top_k * 10)
        visual_df = self._visual_search(query, candidate_pool)
        ocr_df = self._ocr_search(query, candidate_pool)
        fused = fuse_candidates(visual_df, ocr_df, query, visual_weight=visual_weight, ocr_weight=ocr_weight, lexical_weight=lexical_weight, mode=fusion_mode)
        if video_filter:
            fused = fused[fused["video_id"].astype(str).str.contains(str(video_filter), na=False)].reset_index(drop=True)
        raw = fused.head(top_k).copy()
        raw["rank"] = range(1, len(raw) + 1)
        from .result_deduplication import temporal_deduplicate

        dedup_df = temporal_deduplicate(raw, window_seconds=dedup_window_seconds) if dedup else raw.copy()
        if not dedup_df.empty and "dedup_rank" not in dedup_df.columns:
            dedup_df["dedup_rank"] = range(1, len(dedup_df) + 1)
        return {
            "query": query,
            "search_time_ms": round((time.time() - started) * 1000, 2),
            "visual": visual_df,
            "ocr": ocr_df,
            "fused_raw": raw,
            "fused_dedup": dedup_df,
            "candidate_pool": candidate_pool,
            "visual_dim": self.visual_dim,
            "ocr_dim": self.ocr_dim,
        }
