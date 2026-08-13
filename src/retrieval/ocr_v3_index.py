"""
OCR V3 Retrieval Index (Hybrid Lexical BM25 + Semantic E5 Embedding FAISS)
===========================================================================
Indexes OCR Segments produced by temporal merging:
  - BM25 Lexical matcher (handles exact text, typos, proper nouns, abbreviations)
  - Multilingual-E5 Semantic embedding (handles paraphrase, semantic queries)
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.preprocessing.ocr_temporal_merger import normalize_text_search, remove_vietnamese_accents, text_similarity


_VIETNAMESE_STOPWORDS = {
    "va",
    "voi",
    "cua",
    "cho",
    "cac",
    "nhung",
    "mot",
    "trong",
    "tren",
    "duoi",
    "tai",
    "tu",
    "den",
    "la",
    "ve",
}


def _search_tokens(text: str, keep_short: bool = False) -> list[str]:
    normalized = normalize_text_search(remove_vietnamese_accents(str(text or "")))
    tokens = re.findall(r"\w+", normalized)
    min_len = 2 if keep_short else 3
    return [tok for tok in tokens if len(tok) >= min_len and tok not in _VIETNAMESE_STOPWORDS]


def _region_weight(region_type: str) -> float:
    region = str(region_type or "").strip().lower()
    if region == "headline":
        return 1.35
    if region == "ticker":
        return 1.18
    if region == "scene_text":
        return 0.82
    if region in {"logo_channel", "clock_time"}:
        return 0.55
    return 1.0


def _low_information_penalty(text: str, region_type: str) -> float:
    tokens_keep_short = _search_tokens(text, keep_short=True)
    tokens = _search_tokens(text)
    if not tokens_keep_short:
        return 0.25

    unique_short = set(tokens_keep_short)
    region = str(region_type or "").strip().lower()

    # OCR noise often appears as repeated fragments: "khoa khoa", "Hc Hc", "ngh ngh".
    if len(unique_short) == 1 and len(tokens_keep_short) >= 2:
        return 0.22 if region == "scene_text" else 0.45
    if len(tokens_keep_short) % 2 == 0:
        half = len(tokens_keep_short) // 2
        if tokens_keep_short[:half] == tokens_keep_short[half:] and len(unique_short) <= 2:
            return 0.28 if region == "scene_text" else 0.55
    if len(tokens) <= 1 and len(tokens_keep_short) <= 3:
        return 0.35 if region == "scene_text" else 0.65
    if region == "scene_text" and len(tokens) <= 2 and max(len(tok) for tok in tokens_keep_short) <= 4:
        return 0.45

    return 1.0


def _query_token_coverage(query: str, text: str) -> float:
    q_tokens = set(_search_tokens(query))
    if not q_tokens:
        return 0.0
    t_tokens = set(_search_tokens(text))
    if not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / max(1, len(q_tokens))


def _coverage_gate(query: str, coverage: float) -> float:
    q_len = len(set(_search_tokens(query)))
    if q_len < 4:
        return 1.0
    if coverage >= 0.65:
        return 1.18
    if coverage >= 0.50:
        return 1.05
    if coverage >= 0.34:
        return 0.72
    if coverage > 0.0:
        return 0.48
    return 0.35


class SimpleBM25:
    """Lightweight BM25 Lexical Matcher for OCR Text Segments with Inverted Indexing & Precomputed TF."""

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


class OCRV3HybridRetriever:
    """Hybrid BM25 + FAISS E5 Semantic Retriever for OCR V3 Segments."""

    def __init__(self, segments_df: pd.DataFrame, faiss_index_path: str | Path | None = None, model_name: str = "intfloat/multilingual-e5-small"):
        self.df = segments_df.copy().reset_index(drop=True)
        self.model_name = model_name

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
        self._query_cache: dict[str, Any] = {}

    def _init_e5_model(self):
        if self.model is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

            D, I = self.faiss_index.search(embeddings, top_k * 2)
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
            text_search = str(row.get("text_search", ""))
            text_no_acc = str(row.get("text_search_no_accent", ""))
            semantic_text = str(row.get("semantic_search_text", ""))
            text_blob = f"{text_search} {text_no_acc} {semantic_text}"
            region_type = str(row.get("region_type", ""))

            # Extra exact match boost
            exact_boost = 0.3 if query_clean in text_search or query_no_acc in text_no_acc else 0.0
            coverage = _query_token_coverage(query, text_blob)
            coverage_boost = 0.25 * coverage
            confidence = row.get("mean_confidence", row.get("track_confidence", 0.0))
            try:
                confidence_boost = 0.05 * max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence_boost = 0.0
            region_boost = _region_weight(region_type)
            quality_penalty = _low_information_penalty(text_blob, region_type)
            coverage_gate = _coverage_gate(query, coverage)

            base_score = alpha * sem_norm + (1 - alpha) * bm25_norm + exact_boost + coverage_boost + confidence_boost
            fused_score = base_score * region_boost * quality_penalty * coverage_gate

            rec = row.to_dict()
            rec["score"] = round(float(fused_score), 4)
            rec["bm25_score"] = round(float(bm25_norm), 4)
            rec["semantic_score"] = round(float(sem_norm), 4)
            rec["query_token_coverage"] = round(float(coverage), 4)
            rec["region_weight"] = round(float(region_boost), 4)
            rec["quality_penalty"] = round(float(quality_penalty), 4)
            rec["coverage_gate"] = round(float(coverage_gate), 4)
            combined.append(rec)

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]
