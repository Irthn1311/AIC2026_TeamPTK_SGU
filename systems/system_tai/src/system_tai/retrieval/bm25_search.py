"""BM25 Text Search and Document Retrieval Engine for Multimodal Fusion."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult


def tokenize_text(text: str) -> list[str]:
    """Tokenize and normalize Vietnamese and English text."""
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in normalized)
    return [t for t in cleaned.split() if t]


@dataclass(frozen=True, slots=True)
class BM25Document:
    doc_id: str
    video_id: str
    frame_id: int
    text: str
    clip_row: int = 0
    keyframe_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """Okapi BM25 Index implementation."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.documents: list[BM25Document] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_term_freqs: list[Counter[str]] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.idf_cache: dict[str, float] = {}

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    def add_documents(self, docs: Sequence[BM25Document]) -> None:
        """Add and index a sequence of BM25Document objects."""
        for doc in docs:
            tokens = tokenize_text(doc.text)
            term_freq = Counter(tokens)
            self.documents.append(doc)
            self.doc_lengths.append(len(tokens))
            self.doc_term_freqs.append(term_freq)
            for term in term_freq:
                self.doc_freqs[term] += 1

        total_len = sum(self.doc_lengths)
        self.avg_doc_length = total_len / len(self.documents) if self.documents else 0.0
        self._compute_idfs()

    def _compute_idfs(self) -> None:
        n_docs = len(self.documents)
        self.idf_cache.clear()
        for term, df in self.doc_freqs.items():
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self.idf_cache[term] = max(0.0, idf)

    def search(self, query_text: str, top_k: int = 100) -> list[tuple[BM25Document, float]]:
        """Search top-K most relevant documents for the query text."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.documents:
            return []

        query_tokens = tokenize_text(query_text)
        if not query_tokens:
            return []

        scores: list[float] = [0.0] * len(self.documents)
        avgdl = max(1.0, self.avg_doc_length)

        for token in query_tokens:
            idf = self.idf_cache.get(token, 0.0)
            if idf <= 0.0:
                continue

            for idx, term_freq in enumerate(self.doc_term_freqs):
                tf = term_freq.get(token, 0)
                if tf <= 0:
                    continue
                doc_len = self.doc_lengths[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                scores[idx] += idf * (numerator / denominator)

        # Pair scores with documents
        scored_pairs = [
            (self.documents[i], scores[i])
            for i in range(len(self.documents))
            if scores[i] > 0.0
        ]
        # Sort descending by score, then ascending by video_id, frame_id
        scored_pairs.sort(
            key=lambda item: (-item[1], item[0].video_id, item[0].frame_id, item[0].clip_row)
        )
        return scored_pairs[:top_k]


class BM25TextRetriever:
    """Retriever wrapper for BM25 text index."""

    def __init__(self, index: BM25Index | None = None) -> None:
        self.index = index or BM25Index()

    def add_documents(self, docs: Sequence[BM25Document]) -> None:
        self.index.add_documents(docs)

    def retrieve(self, query: KISQuery) -> KISResult:
        return self.search_text(
            query_id=query.query_id,
            query_text=query.text,
            top_k=query.top_k,
        )

    def search_text(
        self,
        *,
        query_id: str,
        query_text: str,
        top_k: int,
    ) -> KISResult:
        if not query_id.strip():
            raise ValueError("query_id must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scored_pairs = self.index.search(query_text, top_k=top_k * 2)

        # Deduplicate per (video_id, frame_id)
        best_by_identity: dict[tuple[str, int], tuple[BM25Document, float]] = {}
        for doc, score in scored_pairs:
            key = (doc.video_id, doc.frame_id)
            if key not in best_by_identity or score > best_by_identity[key][1]:
                best_by_identity[key] = (doc, score)

        sorted_candidates = sorted(
            best_by_identity.values(),
            key=lambda item: (-item[1], item[0].video_id, item[0].frame_id, item[0].clip_row),
        )[:top_k]

        candidates = tuple(
            CandidateFrame(
                video_id=doc.video_id,
                frame_id=doc.frame_id,
                clip_row=doc.clip_row,
                keyframe_order=doc.keyframe_order,
                score=float(score),
                rank=rank,
                source="bm25_text",
                diagnostic_metadata={
                    "doc_id": doc.doc_id,
                    "k1": self.index.k1,
                    "b": self.index.b,
                },
            )
            for rank, (doc, score) in enumerate(sorted_candidates, start=1)
        )

        return KISResult(
            query_id=query_id,
            ranked_candidates=candidates,
        )
