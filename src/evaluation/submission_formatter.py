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
        self._top100_data: Dict[str, List[Dict[str, Any]]] = {}
        self._query_types: Dict[str, str] = {}
        self._query_answers: Dict[str, str] = {}

    # ----------------------------------------------------------
    # Add results
    # ----------------------------------------------------------

    def add_top100(
        self,
        query_id: str,
        evidence: EvidenceResult,
        query_type: str = "textual_kis",
        answer: str = "",
    ) -> None:
        """Record up to top 100 candidate answers for a query."""
        if not evidence or not evidence.top_results:
            return

        self._query_types[query_id] = query_type
        if answer:
            self._query_answers[query_id] = answer

        top_candidates = []
        for rank, r in enumerate(evidence.top_results[:100], 1):
            cand = {
                "rank": rank,
                "video_id": r.video_id,
                "frame_idx": r.frame_idx,
                "n": r.n,
                "pts_time": round(float(r.pts_time), 2),
                "score": round(float(r.score), 4),
                "source": getattr(r, "retriever_source", "fusion"),
            }
            if query_type == "qa" and answer:
                cand["answer"] = answer
            top_candidates.append(cand)

        self._top100_data[query_id] = top_candidates

    # Backwards compatibility alias
    def add_top20(self, query_id: str, evidence: EvidenceResult) -> None:
        self.add_top100(query_id, evidence)

    def add_kis(self, query_id: str, evidence: EvidenceResult) -> None:
        """Add one KIS result."""
        self._kis_rows.append({
            "query_id":  query_id,
            "video_id":  evidence.video_id,
            "frame_idx": evidence.frame_idx,
        })
        self.add_top100(query_id, evidence, query_type="textual_kis")

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
        self.add_top100(query_id, evidence, query_type="qa", answer=answer)

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
            self.add_top100(query_id, evidence, query_type="trake")

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
    # BTC Submission Per-Query Top 100 CSV & ZIP Exporter
    # ----------------------------------------------------------

    def save_top100_per_query_csvs(
        self,
        subfolder: str = "submission_csvs",
        zip_filename: str = "submission_top100.zip",
    ) -> Path:
        """
        Write individual per-query CSV files containing up to 100 candidates each,
        formatted strictly according to BTC AI Challenge submission rules:

        1. Textual KIS:  video_id, frame_idx
        2. Q&A:          video_id, frame_idx, "answer"
        3. TRAKE:        video_id, event_1_frame_idx, event_2_frame_idx, ...

        And compress all CSV files into submission_top100.zip for submission.
        """
        import zipfile

        csv_dir = self.output_dir / subfolder
        csv_dir.mkdir(parents=True, exist_ok=True)

        zip_path = self.output_dir / zip_filename
        written_files: List[Path] = []

        for qid, candidates in self._top100_data.items():
            qtype = self._query_types.get(qid, "textual_kis")
            answer = self._query_answers.get(qid, "")
            query_csv_path = csv_dir / f"{qid}.csv"

            with open(query_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                for cand in candidates[:100]:
                    v_id = cand["video_id"]
                    f_idx = cand["frame_idx"]

                    if qtype == "qa":
                        ans = cand.get("answer", answer)
                        writer.writerow([v_id, f_idx, ans])
                    elif qtype == "trake":
                        # For TRAKE, write event sequence if present, or offset candidates
                        writer.writerow([v_id, f_idx])
                    else:
                        # KIS format: video_id, frame_idx
                        writer.writerow([v_id, f_idx])

            written_files.append(query_csv_path)

        # Create ZIP archive
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in written_files:
                zf.write(p, arcname=p.name)

        logger.info(
            f"Saved {len(written_files)} BTC per-query Top-100 CSVs in {csv_dir} "
            f"and archived to {zip_path}"
        )
        return zip_path

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

    def save_top100_json(self, filename: str = "query_top100_results.json") -> Path:
        """Save top 100 candidates for all queries to a JSON file."""
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._top100_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved top 100 candidates for {len(self._top100_data)} queries → {out_path}")
        return out_path

    # Alias for compatibility
    def save_top20_json(self, filename: str = "query_top20_results.json") -> Path:
        return self.save_top100_json(filename=filename)

    def save_all(self) -> Dict[str, Path]:
        """
        Convenience method: save all 3 submission CSVs + JSON + Top 100 JSON + Top 100 per-query ZIP in one call.

        Returns:
            Dict with keys 'kis', 'qa', 'trake', 'json', 'top100', 'top100_zip' mapping to Path.
        """
        paths = {
            "kis":        self.save_kis(),
            "qa":         self.save_qa(),
            "trake":      self.save_trake(),    # n_events auto-detected
            "json":       self.save_json(),
            "top100":     self.save_top100_json(),
            "top100_zip": self.save_top100_per_query_csvs(),
        }
        logger.info(
            f"[SubmissionFormatter] Saved all: "
            f"KIS={len(self._kis_rows)}, QA={len(self._qa_rows)}, "
            f"TRAKE={len(self._trake_rows)}, Top100_Queries={len(self._top100_data)}"
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
