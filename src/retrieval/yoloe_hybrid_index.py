"""
YOLOE Hybrid Object Search Helper
=================================
Loads YOLOE hybrid JSON outputs and exposes a lightweight inverted-index search.
This augments the multimodal Object branch without modifying the BTC/Faster R-CNN
object corpus.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval.object_index import SYNONYMS_MAP

logger = logging.getLogger(__name__)

YOLOE_EXTRA_SYNONYMS: dict[str, list[str]] = {
    "mũ": ["hat", "helmet", "cap"],
    "nón": ["hat", "helmet", "cap"],
    "kính": ["glasses", "sunglasses"],
    "kính mắt": ["glasses", "sunglasses"],
    "cà vạt": ["tie", "necktie"],
    "đồng hồ": ["watch"],
    "vòng tay": ["bracelet"],
    "vòng cổ": ["necklace"],
    "túi xách": ["handbag", "bag"],
    "ô": ["umbrella"],
    "dù": ["umbrella"],
    "sách": ["book", "bible"],
    "điện thoại": ["phone", "smartphone", "mobile phone"],
    "tòa tháp": ["tower", "sky tower"],
    "cao ốc": ["skyscraper", "building"],
    "trâu": ["yak", "cow", "cattle", "bull", "ox", "buffalo", "bison", "livestock"],
    "đàn trâu": ["yak", "cow", "cattle", "bull", "ox", "buffalo", "bison", "livestock", "herd"],
    "con trâu": ["yak", "cow", "cattle", "bull", "ox", "buffalo", "bison", "livestock"],
    "bò": ["cow", "cattle", "bull", "calf", "ox", "yak", "buffalo", "livestock"],
    "đàn bò": ["cow", "cattle", "bull", "calf", "ox", "yak", "buffalo", "livestock", "herd"],
    "con bò": ["cow", "cattle", "bull", "calf", "ox", "yak", "buffalo", "livestock"],
    "gia súc": ["livestock", "cattle", "cow", "yak", "buffalo", "bull", "pig"],
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


class YOLOEHybridIndex:
    """Small inverted index over YOLOE hybrid detections."""

    def __init__(self, hybrid_dir: str | Path):
        self.hybrid_dir = Path(hybrid_dir)
        self.df = self._load_records()
        self._inverted_index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._build_index()

    def _load_records(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        if not self.hybrid_dir.exists():
            logger.warning("YOLOE hybrid directory not found: %s", self.hybrid_dir)
            return pd.DataFrame()

        for json_path in sorted(self.hybrid_dir.glob("L*_V*.json")):
            try:
                records = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not load YOLOE hybrid JSON %s: %s", json_path, exc)
                continue

            for record in records:
                detections = record.get("hybrid_detections") or []
                if not detections:
                    continue
                labels = [str(det.get("label", "")).strip() for det in detections if str(det.get("label", "")).strip()]
                raw_labels = [
                    str(det.get("raw_label", "")).strip()
                    for det in detections
                    if str(det.get("raw_label", "")).strip()
                ]
                parent_labels = [
                    str(det.get("parent_label", "")).strip()
                    for det in detections
                    if str(det.get("parent_label", "")).strip()
                ]
                label_counts = Counter(labels)
                rows.append(
                    {
                        "video_id": str(record.get("video_id", "")),
                        "frame_idx": int(record.get("frame_id", 0)),
                        "timestamp": float(record.get("timestamp", 0.0)),
                        "keyframe_name": str(record.get("keyframe_name", "")),
                        "detections": detections,
                        "unique_object_classes": sorted(label_counts),
                        "object_scores": {
                            label: max(
                                float(det.get("confidence", 0.0))
                                for det in detections
                                if str(det.get("label", "")) == label
                            )
                            for label in label_counts
                        },
                        "search_text": " ".join(labels + raw_labels + parent_labels),
                    }
                )

        return pd.DataFrame(rows)

    def _build_index(self) -> None:
        if self.df.empty:
            return
        logger.info("Building YOLOE Hybrid Object Index from %d keyframe records...", len(self.df))
        for row_idx, row in self.df.iterrows():
            object_scores = row.get("object_scores", {}) or {}
            max_score = max((float(value) for value in object_scores.values()), default=0.5)
            for token in set(_tokenize(str(row.get("search_text", "")))):
                self._inverted_index[token].append((row_idx, max_score))

    def search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        query_tokens = _tokenize(query)
        if not query_tokens or self.df.empty:
            return pd.DataFrame()

        stopwords = {
            "tìm",
            "cảnh",
            "các",
            "và",
            "ở",
            "trong",
            "theo",
            "cho",
            "với",
            "find",
            "scene",
            "scenes",
            "of",
            "and",
            "in",
            "on",
            "a",
            "the",
            "with",
            "for",
        }
        filtered_tokens = [token for token in query_tokens if token not in stopwords and len(token) > 1] or query_tokens
        expanded_tokens: set[str] = set(filtered_tokens)
        query_lower = query.lower()

        synonym_map = {**SYNONYMS_MAP, **YOLOE_EXTRA_SYNONYMS}

        for synonym_key, synonym_targets in synonym_map.items():
            if synonym_key in query_lower:
                for target in synonym_targets:
                    expanded_tokens.update(_tokenize(target))

        for token in list(filtered_tokens):
            if token in synonym_map:
                for target in synonym_map[token]:
                    expanded_tokens.update(_tokenize(target))

        scores: dict[int, float] = defaultdict(float)
        match_counts: dict[int, int] = defaultdict(int)
        for token in expanded_tokens:
            if token not in self._inverted_index:
                continue
            weight = 1.0 if token in filtered_tokens else 0.75
            for row_idx, object_score in self._inverted_index[token]:
                scores[row_idx] += (float(object_score) + 0.5) * weight
                match_counts[row_idx] += 1

        if not scores:
            return pd.DataFrame()

        ranked_indices = sorted(scores, key=lambda idx: (scores[idx], match_counts[idx]), reverse=True)[:top_k]
        rows = []
        for rank, row_idx in enumerate(ranked_indices, start=1):
            row = self.df.iloc[row_idx].to_dict()
            row["rank"] = rank
            row["yoloe_object_score"] = round(scores[row_idx], 4)
            row["matched_terms_count"] = match_counts[row_idx]
            rows.append(row)
        return pd.DataFrame(rows)

    def get_frame_objects(self, video_id: str, frame_idx: int, limit: int = 30) -> list[dict[str, Any]]:
        if self.df.empty:
            return []
        mask = (self.df["video_id"] == video_id) & (self.df["frame_idx"] == int(frame_idx))
        if not mask.any():
            return []
        detections = list(self.df[mask].iloc[0].get("detections", []) or [])
        detections.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return [
            {
                "label": str(det.get("label", "")),
                "raw_label": str(det.get("raw_label", det.get("label", ""))),
                "parent_label": det.get("parent_label"),
                "confidence": round(float(det.get("confidence", 0.0)), 4),
                "score": round(float(det.get("confidence", 0.0)), 4),
                "box": det.get("bbox"),
                "bbox": det.get("bbox"),
                "source": f"yoloe_{det.get('source', 'hybrid')}",
            }
            for det in detections[:limit]
        ]
