"""
Query Loader Module — supports loading queries from JSON files, TXT files,
or a directory of individual BTC query .txt files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_single_txt_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a single .txt query file from BTC competition format.

    Filename conventions:
      - query-p1-5-kis.txt   -> qtype = "textual_kis"
      - query-p1-15-qa.txt   -> qtype = "qa"
      - query-p1-16-trake.txt-> qtype = "trake"
    """
    stem = file_path.stem  # e.g. "query-p1-5-kis"
    qid = stem

    # Infer query type from filename
    stem_lower = stem.lower()
    if "qa" in stem_lower:
        qtype = "qa"
    elif "trake" in stem_lower:
        qtype = "trake"
    else:
        qtype = "textual_kis"

    content = file_path.read_text(encoding="utf-8", errors="ignore").strip()

    # Try parsing as JSON first if the txt file contains raw JSON
    if content.startswith("{") and content.endswith("}"):
        try:
            data = json.loads(content)
            data["query_id"] = data.get("query_id", qid)
            if "type" not in data:
                data["type"] = qtype
            return data
        except Exception:
            pass

    # Parse raw text content
    if qtype == "qa":
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        question = lines[0] if lines else content
        description = "\n".join(lines[1:]) if len(lines) > 1 else ""
        return {
            "query_id": qid,
            "type": "qa",
            "question": question,
            "description": description,
            "text": content,
        }

    elif qtype == "trake":
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        activity = lines[0] if lines else content
        event_lines = lines[1:] if len(lines) > 1 else lines
        events = [
            {"event_id": idx + 1, "description": ev_text}
            for idx, ev_text in enumerate(event_lines)
        ]
        return {
            "query_id": qid,
            "type": "trake",
            "activity": activity,
            "text": content,
            "events": events,
        }

    else:
        # KIS default
        return {
            "query_id": qid,
            "type": "textual_kis",
            "text": content,
        }


def load_queries(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load queries from a JSON file, a single TXT file, or a directory containing .txt/.json files.

    Args:
        path: Path to .json file, .txt file, or directory of query files.

    Returns:
        List of structured query dictionaries.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Query path does not exist: {target}")

    # Case 1: Directory of query files
    if target.is_dir():
        logger.info(f"Loading query directory: {target}")
        txt_files = sorted(
            list(target.glob("*.txt")) + list(target.glob("*.json")),
            key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.stem)]
        )
        if not txt_files:
            raise ValueError(f"No .txt or .json query files found in directory: {target}")

        queries = []
        for file in txt_files:
            if file.suffix.lower() == ".json":
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        queries.extend(data)
                    else:
                        queries.append(data)
            else:
                queries.append(parse_single_txt_file(file))

        logger.info(f"Loaded {len(queries)} queries from directory {target}")
        return queries

    # Case 2: Single JSON file
    if target.suffix.lower() == ".json":
        with open(target, encoding="utf-8") as f:
            queries = json.load(f)
            if isinstance(queries, dict):
                queries = [queries]
        logger.info(f"Loaded {len(queries)} queries from JSON file: {target}")
        return queries

    # Case 3: Single TXT file
    if target.suffix.lower() == ".txt":
        query = parse_single_txt_file(target)
        logger.info(f"Loaded 1 query from TXT file: {target}")
        return [query]

    raise ValueError(f"Unsupported query format: {target}")
