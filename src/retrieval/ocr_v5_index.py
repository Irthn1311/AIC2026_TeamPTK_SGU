from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.ocr_video_v5 import normalize_for_match, text_similarity, tokenize


class SimpleBM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in documents]
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        self.df: Counter[str] = Counter()
        self.index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, tokens in enumerate(self.docs):
            counts = Counter(tokens)
            for token, tf in counts.items():
                self.df[token] += 1
                self.index[token].append((idx, tf))

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        q_tokens = tokenize(query)
        n_docs = max(1, len(self.docs))
        for token in q_tokens:
            postings = self.index.get(token, [])
            if not postings:
                continue
            idf = math.log(1 + (n_docs - self.df[token] + 0.5) / (self.df[token] + 0.5))
            for doc_idx, tf in postings:
                dl = len(self.docs[doc_idx])
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(0.001, self.avgdl))
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / denom
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class OCRV5LexicalSearcher:
    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus.reset_index(drop=True).copy()
        docs = self.corpus["text"].fillna("").astype(str).tolist()
        self.bm25 = SimpleBM25(docs)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "OCRV5LexicalSearcher":
        return cls(pd.read_parquet(path))

    def search_ocr(self, query: str, top_k: int = 10, roles: list[str] | None = None) -> list[dict[str, Any]]:
        df = self.corpus
        allowed = set(roles or [])
        bm25_hits = dict(self.bm25.search(query, top_k=max(top_k * 8, 40)))
        query_norm = normalize_for_match(query, no_accent=True)
        candidates = set(bm25_hits)
        for idx, row in df.iterrows():
            if allowed and str(row.get("role", "")) not in allowed:
                continue
            text_norm = str(row.get("text_no_accent", ""))
            if query_norm and (query_norm in text_norm or text_norm in query_norm):
                candidates.add(idx)
            elif text_similarity(query_norm, text_norm) >= 0.72:
                candidates.add(idx)

        scored = []
        for idx in candidates:
            row = df.iloc[int(idx)]
            role = str(row.get("role", ""))
            if allowed and role not in allowed:
                continue
            text = str(row.get("text", ""))
            text_norm = str(row.get("text_no_accent", ""))
            lexical = float(bm25_hits.get(idx, 0.0))
            exact = 1.0 if query_norm and query_norm in text_norm else 0.0
            fuzzy = text_similarity(query_norm, text_norm)
            role_weight = float(row.get("role_weight", 0.5))
            reliability = float(row.get("reliability_score", 0.5))
            combined = (lexical * 0.55) + (exact * 1.0) + (fuzzy * 0.75) + (role_weight * 0.15) + (reliability * 0.10)
            scored.append({
                "video_id": row.get("video_id"),
                "timestamp": float(row.get("timestamp", 0.0)),
                "frame_id": int(row.get("frame_id", 0)),
                "text": text,
                "role": role,
                "lexical_score": round(lexical, 4),
                "semantic_score": None,
                "fuzzy_score": round(fuzzy, 4),
                "combined_score": round(combined, 4),
                "track_id": row.get("track_id"),
                "mapped_keyframe_name": row.get("mapped_keyframe_name", ""),
                "mapped_keyframe_path": row.get("mapped_keyframe_path", ""),
            })
        scored.sort(key=lambda r: r["combined_score"], reverse=True)
        return scored[:top_k]
