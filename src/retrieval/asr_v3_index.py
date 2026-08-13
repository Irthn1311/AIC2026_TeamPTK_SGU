"""
ASR V3 Retrieval Index (Hybrid Lexical BM25 + Semantic E5 Embedding FAISS)
===========================================================================
Indexes ASR Retrieval Chunks (15-30s normalized speech segments) produced by:
  - Whisper Speech-to-Text
  - Qwen3-4B Contextual Normalization & Correction

Provides:
  - BM25 Lexical matcher (handles exact spoken keywords, names, numbers, events)
  - Multilingual-E5 Semantic embedding (handles paraphrase, semantic speech queries)
  - Keyframe Timestamp Alignment (maps spoken speech chunk intervals to video keyframes)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.preprocessing.ocr_temporal_merger import normalize_text_search, remove_vietnamese_accents, text_similarity


class SimpleBM25:
    """Lightweight BM25 Lexical Matcher for ASR Text Chunks with Inverted Indexing."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_tokens = [c.split() for c in corpus]
        self.doc_lens = [len(dt) for dt in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.N = len(corpus)

        self.df: Counter[str] = Counter()
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}

        for idx, dt in enumerate(self.doc_tokens):
            tf_counts = Counter(dt)
            for w, tf in tf_counts.items():
                self.df[w] += 1
                if w not in self.inverted_index:
                    self.inverted_index[w] = []
                self.inverted_index[w].append((idx, tf))

        self.idf: dict[str, float] = {}
        for w, freq in self.df.items():
            self.idf[w] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        q_tokens = normalize_text_search(query).split()
        if not q_tokens:
            return []

        expanded_terms: dict[str, float] = {}
        all_vocab = list(self.inverted_index.keys())

        for qt in q_tokens:
            if qt in self.inverted_index:
                expanded_terms[qt] = 1.0
            else:
                for doc_word in all_vocab:
                    if abs(len(doc_word) - len(qt)) <= 2 and text_similarity(qt, doc_word) >= 0.80:
                        expanded_terms[doc_word] = 0.7
                        break

        if not expanded_terms:
            return []

        scores: Counter[int] = Counter()
        for term, weight in expanded_terms.items():
            idf = self.idf.get(term, 0.0) * weight
            if idf <= 0:
                continue
            for idx, tf in self.inverted_index.get(term, []):
                dlen = self.doc_lens[idx]
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (dlen / self.avgdl))
                scores[idx] += idf * (num / denom)

        ranked = scores.most_common(top_k)
        return [(idx, sc) for idx, sc in ranked if sc > 0.0]


class ASRV3HybridRetriever:
    """Hybrid BM25 + FAISS E5 Semantic Retriever for ASR Retrieval Chunks."""

    def __init__(
        self,
        chunks_df: pd.DataFrame,
        faiss_index_path: str | Path | None = None,
        model_name: str = "intfloat/multilingual-e5-small",
    ):
        self.df = chunks_df.copy().reset_index(drop=True)
        self.model_name = model_name

        # Prepare text search fields
        def _get_text(r):
            for k in ["text_clean", "text_normalized", "text_raw", "text", "text_search"]:
                if k in r and r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k])):
                    s = str(r[k]).strip()
                    if s and s.lower() not in ("none", "nan", "null"):
                        return s
            return ""

        if "text_search" not in self.df.columns or self.df["text_search"].astype(str).str.lower().isin(["nan", "none", ""]).any():
            self.df["text_search"] = self.df.apply(lambda r: normalize_text_search(_get_text(r)), axis=1)
        if "text_search_no_accent" not in self.df.columns or self.df["text_search_no_accent"].astype(str).str.lower().isin(["nan", "none", ""]).any():
            self.df["text_search_no_accent"] = self.df["text_search"].map(remove_vietnamese_accents)

        # Prepare corpus for BM25
        bm25_corpus = []
        for _, row in self.df.iterrows():
            t_search = str(row.get("text_search", ""))
            t_no_acc = str(row.get("text_search_no_accent", ""))
            bm25_corpus.append(f"{t_search} {t_no_acc}")

        self.bm25 = SimpleBM25(bm25_corpus)

        # Load FAISS index if available
        self.faiss_index = None
        if faiss_index_path and Path(faiss_index_path).exists():
            self.faiss_index = faiss.read_index(str(faiss_index_path))

        self.tokenizer = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._query_cache: dict[str, Any] = {}

    def _init_e5_model(self):
        if self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> list[dict[str, Any]]:
        query_clean = normalize_text_search(query)
        query_no_acc = remove_vietnamese_accents(query)

        # 1. Lexical BM25 Search
        bm25_results = self.bm25.search(query, top_k=top_k * 2) if alpha < 1.0 else []
        bm25_dict = {idx: sc for idx, sc in bm25_results}
        max_bm25 = max(bm25_dict.values(), default=1.0) or 1.0

        # 2. Semantic Embedding Search
        semantic_dict = {}
        if self.faiss_index is not None and alpha > 0.0:
            if query in self._query_cache:
                embeddings = self._query_cache[query]
            else:
                self._init_e5_model()
                q_text = f"query: {query}"
                with torch.no_grad():
                    inputs = self.tokenizer(q_text, return_tensors="pt", max_length=128, truncation=True, padding=True).to(self.device)
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()
                self._query_cache[query] = embeddings

            D, I = self.faiss_index.search(embeddings, min(len(self.df), top_k * 2))
            for score, idx in zip(D[0], I[0]):
                if idx >= 0:
                    semantic_dict[int(idx)] = float(score)

        # Combine Scores
        all_indices = set(bm25_dict.keys()) | set(semantic_dict.keys())
        combined = []

        for idx in all_indices:
            row = self.df.iloc[idx]

            bm25_norm = bm25_dict.get(idx, 0.0) / max_bm25
            sem_norm = semantic_dict.get(idx, 0.0)

            # Substring / Exact match boost
            t_search = str(row.get("text_search", ""))
            t_no_acc = str(row.get("text_search_no_accent", ""))
            exact_boost = 0.35 if (query_clean in t_search or query_no_acc in t_no_acc) else 0.0

            fused_score = alpha * sem_norm + (1.0 - alpha) * bm25_norm + exact_boost

            rec = row.to_dict()
            rec["score"] = round(float(fused_score), 4)
            rec["bm25_score"] = round(float(bm25_norm), 4)
            rec["semantic_score"] = round(float(sem_norm), 4)
            combined.append(rec)

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]
