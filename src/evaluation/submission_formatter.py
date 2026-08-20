"""
Submission Formatter — converts pipeline results to BTC-standard CSV files.

BTC submission formats:
  Dạng 1 (KIS):   video_id, frame_idx
  Dạng 2 (Q&A):   video_id, frame_idx, answer
  Dạng 3 (TRAKE): video_id, event_1_frame_idx, event_2_frame_idx, ...
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.types import (
    EvidenceResult, KISSubmission, QASubmission, TRAKESubmission, TRAKEEventResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SubmissionFormatter:
    """
    Converts EvidenceResult objects into BTC-standard CSV submission files.

    Usage:
        formatter = SubmissionFormatter(output_dir="outputs/submission")

        # KIS
        formatter.add_kis(query_id="q001", evidence=evidence)
        formatter.save_kis("submission_kis.csv")

        # Q&A
        formatter.add_qa(query_id="q002", evidence=evidence, answer="5")
        formatter.save_qa("submission_qa.csv")
    """

    def __init__(self, output_dir: str = "outputs/submission"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._kis_rows: List[Dict[str, Any]] = []
        self._qa_rows: List[Dict[str, Any]] = []
        self._trake_rows: List[Dict[str, Any]] = []
        self._top20_data: Dict[str, List[Dict[str, Any]]] = {}

    # ----------------------------------------------------------
    # Add results
    # ----------------------------------------------------------

    def add_top20(self, query_id: str, evidence: EvidenceResult) -> None:
        """Record top 20 candidate answers for a query."""
        if not evidence or not evidence.top_results:
            return

        top_candidates = []
        for rank, r in enumerate(evidence.top_results[:20], 1):
            top_candidates.append({
                "rank": rank,
                "video_id": r.video_id,
                "frame_idx": r.frame_idx,
                "n": r.n,
                "pts_time": round(float(r.pts_time), 2),
                "score": round(float(r.score), 4),
                "source": getattr(r, "retriever_source", "fusion"),
            })
        self._top20_data[query_id] = top_candidates

    def add_kis(self, query_id: str, evidence: EvidenceResult) -> None:
        """Add one KIS result."""
        self._kis_rows.append({
            "query_id":  query_id,
            "video_id":  evidence.video_id,
            "frame_idx": evidence.frame_idx,
        })
        self.add_top20(query_id, evidence)

    def add_qa(
        self,
        query_id: str,
        evidence: EvidenceResult,
        answer: str,
    ) -> None:
        """Add one Q&A result with its answer."""
        self._qa_rows.append({
            "query_id":  query_id,
            "video_id":  evidence.video_id,
            "frame_idx": evidence.frame_idx,
            "answer":    answer,
        })
        self.add_top20(query_id, evidence)

    def add_trake(
        self,
        query_id: str,
        video_id: str,
        event_frame_idxs: Dict[int, int],  # {event_id: frame_idx}
        evidence: Optional[EvidenceResult] = None,
    ) -> None:
        """
        Add one TRAKE result.

        Args:
            event_frame_idxs: {1: 120, 2: 145, 3: 178, 4: 210}
        """
        row: Dict[str, Any] = {"query_id": query_id, "video_id": video_id}
        for event_id, frame_idx in sorted(event_frame_idxs.items()):
            row[f"event_{event_id}_frame_idx"] = frame_idx
        self._trake_rows.append(row)
        if evidence:
            self.add_top20(query_id, evidence)

    # ----------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------

    def save_kis(self, filename: str = "submission_kis.csv") -> Path:
        """Write KIS results to CSV."""
        return self._write_csv(
            self._kis_rows,
            filename,
            fieldnames=["query_id", "video_id", "frame_idx"],
        )

    def save_qa(self, filename: str = "submission_qa.csv") -> Path:
        """Write Q&A results to CSV."""
        return self._write_csv(
            self._qa_rows,
            filename,
            fieldnames=["query_id", "video_id", "frame_idx", "answer"],
        )

    def save_trake(
        self,
        filename: str = "submission_trake.csv",
        n_events: Optional[int] = None,
    ) -> Path:
        """
        Write TRAKE results to CSV.

        Args:
            n_events: Number of event columns. If None, auto-detected from
                      the actual data (supports any number of events).
        """
        if n_events is None:
            # Auto-detect from data: find max event_id across all rows
            max_ev = 1
            for row in self._trake_rows:
                for k in row.keys():
                    if k.startswith("event_") and k.endswith("_frame_idx"):
                        try:
                            ev_num = int(k.split("_")[1])
                            max_ev = max(max_ev, ev_num)
                        except ValueError:
                            pass
            n_events = max_ev if self._trake_rows else 4

        fieldnames = ["query_id", "video_id"] + [
            f"event_{i}_frame_idx" for i in range(1, n_events + 1)
        ]
        return self._write_csv(self._trake_rows, filename, fieldnames)

    def _write_csv(
        self,
        rows: List[Dict[str, Any]],
        filename: str,
        fieldnames: List[str],
    ) -> Path:
        out_path = self.output_dir / filename
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved {len(rows)} rows → {out_path}")
        return out_path

    # ----------------------------------------------------------
    # Save / Load JSON (intermediate checkpoint)
    # ----------------------------------------------------------

    def save_json(self, filename: str = "results.json") -> Path:
        """Save all collected results as JSON for inspection."""
        out_path = self.output_dir / filename
        data = {
            "kis":   self._kis_rows,
            "qa":    self._qa_rows,
            "trake": self._trake_rows,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        total = len(self._kis_rows) + len(self._qa_rows) + len(self._trake_rows)
        logger.info(f"Saved {total} total results → {out_path}")
        return out_path

    def save_top20_json(self, filename: str = "query_top20_results.json") -> Path:
        """Save top 20 candidates for all queries to a JSON file."""
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._top20_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved top 20 candidates for {len(self._top20_data)} queries → {out_path}")
        return out_path

    def save_all(self) -> Dict[str, Path]:
        """
        Convenience method: save all 3 submission CSVs + JSON + Top 20 JSON in one call.

        Automatically detects the maximum number of events across TRAKE rows
        so the CSV columns are always correct regardless of event count.

        Returns:
            Dict with keys 'kis', 'qa', 'trake', 'json', 'top20' mapping to Path.
        """
        paths = {
            "kis":   self.save_kis(),
            "qa":    self.save_qa(),
            "trake": self.save_trake(),    # n_events auto-detected
            "json":  self.save_json(),
            "top20": self.save_top20_json(),
        }
        logger.info(
            f"[SubmissionFormatter] Saved all: "
            f"KIS={len(self._kis_rows)}, QA={len(self._qa_rows)}, "
            f"TRAKE={len(self._trake_rows)}, Top20_Queries={len(self._top20_data)}"
        )
        return paths

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "kis":   len(self._kis_rows),
            "qa":    len(self._qa_rows),
            "trake": len(self._trake_rows),
        }
