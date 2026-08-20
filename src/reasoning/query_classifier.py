"""
Query Classifier for AIC Video Retrieval System.

Determines which of the 3 official AIC-HCMC query types a given
input belongs to, routing it to the correct pipeline.

Classification strategy (rule-based, no LLM needed for Sprint 2):
  - Input dict with explicit "type" field → use directly
  - Input dict with "question" key → Q&A (Dạng 2)
  - Input dict with "event_sequence" key → TRAKE (Dạng 3)
  - Otherwise → KIS (Dạng 1, default)
"""

from __future__ import annotations

from typing import Any, Dict, Union

from src.common.enums import QueryType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryClassifier:
    """
    Routes incoming query dicts to the correct QueryType.

    Expected input formats:

    KIS (Dạng 1):
        {"type": "textual_kis", "text": "..."}
        {"text": "..."}                           # type inferred

    Q&A (Dạng 2):
        {"type": "qa", "description": "...", "question": "..."}
        {"description": "...", "question": "..."}  # type inferred

    TRAKE (Dạng 3):
        {
          "type": "trake",
          "activity": "Nhảy cao",
          "events": [
            {"id": 1, "name": "Approach", "description": "...", "hint": "..."},
            ...
          ]
        }
    """

    def classify(self, query_dict: Dict[str, Any]) -> QueryType:
        """
        Classify a query dict and return its QueryType.

        Args:
            query_dict: Dict parsed from the input JSON file or API request

        Returns:
            QueryType enum value
        """
        # --- Explicit type field takes priority ---
        explicit_type = query_dict.get("type", "").lower()
        if explicit_type in ("textual_kis", "kis"):
            return QueryType.TEXTUAL_KIS
        if explicit_type in ("qa", "q&a", "question"):
            return QueryType.QA
        if explicit_type in ("trake", "temporal"):
            return QueryType.TRAKE

        # --- Infer from query_id stem if available ---
        qid = str(query_dict.get("query_id", "")).lower()
        if "-qa" in qid or "_qa" in qid:
            return QueryType.QA
        if "-trake" in qid or "_trake" in qid:
            return QueryType.TRAKE
        if "-kis" in qid or "_kis" in qid:
            return QueryType.TEXTUAL_KIS

        # --- Infer from structure ---
        if "event_sequence" in query_dict or "events" in query_dict:
            logger.debug("Inferred QueryType: TRAKE (has event_sequence/events key)")
            return QueryType.TRAKE

        if "question" in query_dict:
            logger.debug("Inferred QueryType: QA (has question key)")
            return QueryType.QA

        # --- Infer from text content patterns ---
        text_content = str(query_dict.get("text", "") or query_dict.get("description", ""))
        import re
        if re.search(r'\bE[1234][:\s]', text_content, re.IGNORECASE):
            logger.debug("Inferred QueryType: TRAKE (matches E1/E2 pattern in text)")
            return QueryType.TRAKE

        if re.search(r'(\bHỏi\b|\bCho biết\b|\?)', text_content, re.IGNORECASE):
            logger.debug("Inferred QueryType: QA (matches question keyword in text)")
            return QueryType.QA

        # Default to KIS
        logger.debug("Inferred QueryType: KIS (default)")
        return QueryType.TEXTUAL_KIS

    def classify_batch(
        self, query_dicts: list[Dict[str, Any]]
    ) -> list[QueryType]:
        """Classify a list of query dicts."""
        return [self.classify(q) for q in query_dicts]
