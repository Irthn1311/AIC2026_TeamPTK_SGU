"""
Evaluator — Offline evaluation metrics for AIC retrieval results.

Metrics supported:
  - Recall@K    (primary AIC metric for KIS)
  - MRR         (Mean Reciprocal Rank)
  - Precision@K
  - Per-query latency stats

Usage:
    evaluator = Evaluator()
    evaluator.add_result(query_id="q001", predicted_video="L21_V001",
                         gt_video="L21_V001", rank=1, latency=1.2)
    report = evaluator.report()
    print(report)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    query_id: str
    predicted_video: str
    predicted_frame_idx: int
    gt_video: str           # Ground truth video_id
    gt_frame_idx: int       # Ground truth frame_idx
    rank: int               # Rank of the correct answer (1-based; 0 = not found)
    latency: float          # Seconds


class Evaluator:
    """
    Tracks retrieval results and computes offline evaluation metrics.

    For AIC competition:
      - KIS is judged by whether you submit ANY frame from the correct video.
        Exact frame_idx is not required to be pixel-perfect.
      - BTC Final Score is calculated as:
            Final Score = (1/5) * sum_{k in {1, 5, 20, 50, 100}} R@k
        where R@k is the highest R-Score achieved in the top-k results.
    """

    BTC_K_THRESHOLDS = [1, 5, 20, 50, 100]

    def __init__(self, k_values: Optional[List[int]] = None):
        self.k_values = k_values or self.BTC_K_THRESHOLDS
        self._results: List[QueryResult] = []

    def add_result(
        self,
        query_id: str,
        predicted_video: str,
        predicted_frame_idx: int,
        gt_video: str,
        gt_frame_idx: int,
        rank: int = 1,
        latency: float = 0.0,
    ) -> None:
        self._results.append(QueryResult(
            query_id=query_id,
            predicted_video=predicted_video,
            predicted_frame_idx=predicted_frame_idx,
            gt_video=gt_video,
            gt_frame_idx=gt_frame_idx,
            rank=rank if predicted_video == gt_video else 0,
            latency=latency,
        ))

    def recall_at_k(self, k: int) -> float:
        """Fraction of queries where the correct video appears in top-K results."""
        if not self._results:
            return 0.0
        hits = sum(1 for r in self._results if 0 < r.rank <= k)
        return hits / len(self._results)

    def btc_final_score(self) -> float:
        """
        Computes the official BTC Final Score metric:
        Final Score = (1/5) * sum_{k in {1, 5, 20, 50, 100}} R@k
        """
        if not self._results:
            return 0.0
        r_scores = [self.recall_at_k(k) for k in self.BTC_K_THRESHOLDS]
        return sum(r_scores) / len(self.BTC_K_THRESHOLDS)

    def mrr(self) -> float:
        """Mean Reciprocal Rank."""
        if not self._results:
            return 0.0
        total = sum(1 / r.rank for r in self._results if r.rank > 0)
        return total / len(self._results)

    def avg_latency(self) -> float:
        if not self._results:
            return 0.0
        return sum(r.latency for r in self._results) / len(self._results)

    def report(self) -> Dict[str, Any]:
        """Generate a full evaluation report dict."""
        n = len(self._results)
        n_correct = sum(1 for r in self._results if r.rank > 0)

        report = {
            "total_queries": n,
            "correct": n_correct,
            "accuracy": round(n_correct / max(n, 1), 4),
            "BTC_Final_Score": round(self.btc_final_score(), 4),
            "MRR": round(self.mrr(), 4),
            "avg_latency_s": round(self.avg_latency(), 3),
        }
        for k in self.k_values:
            report[f"Recall@{k}"] = round(self.recall_at_k(k), 4)

        return report

    def print_report(self) -> None:
        r = self.report()
        print("\n" + "=" * 50)
        print(f"  AIC Evaluation Report ({r['total_queries']} queries)")
        print("=" * 50)
        print(f"  BTC Final Score: {r['BTC_Final_Score']:.4f} ⭐")
        print(f"  Accuracy:        {r['accuracy']:.1%}")
        print(f"  MRR:             {r['MRR']:.4f}")
        for k in self.k_values:
            print(f"  Recall@{k:<5}   {r[f'Recall@{k}']:.1%}")
        print(f"  Avg latency:     {r['avg_latency_s']:.2f}s")
        print("=" * 50 + "\n")

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "report": self.report(),
            "per_query": [
                {
                    "query_id": r.query_id,
                    "correct": r.rank > 0,
                    "rank": r.rank,
                    "predicted": r.predicted_video,
                    "gt": r.gt_video,
                    "latency_s": r.latency,
                }
                for r in self._results
            ],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation saved → {out}")

