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


def split_qa_text(content: str) -> tuple[str, str]:
    """
    Split a raw QA text block into (description, question).
    
    Handles both multi-line format and single-paragraph format.
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) >= 2:
        if re.search(r'^(\bHỏi\b|\bCho biết\b|\bHãy cho biết\b)', lines[-1], re.IGNORECASE) or lines[-1].endswith('?'):
            return "\n".join(lines[:-1]), lines[-1]

    # Single line or merged lines: search for question transition in text
    m = re.search(r'(\bHỏi\b|\bCho biết\b|\bHãy cho biết\b)', content, re.IGNORECASE)
    if m:
        idx = m.start(1)
        desc = content[:idx].strip()
        quest = content[idx:].strip()
        if desc and quest:
            return desc, quest

    # Fallback: search for last question mark clause
    if '?' in content:
        m_q = re.search(r'([\.\!\;]\s*)([^\.\!\;]+\?.*)$', content)
        if m_q:
            desc = content[:m_q.start(2)].strip()
            quest = m_q.group(2).strip()
            if desc and quest:
                return desc, quest

    return content, content


def parse_trake_text(content: str) -> Dict[str, Any]:
    """
    Parse TRAKE query content into activity and structured event list.
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return {"activity": "", "events": []}

    # Check if Line 1 starts with an event marker (e.g. E1:, e1:, Event 1:, 1.)
    is_line1_event = bool(re.match(r'^(E\d+|e\d+|Event\s*\d+|\d+[\.:])', lines[0], re.IGNORECASE))

    if is_line1_event:
        activity = ""
        event_lines = lines
    else:
        activity = lines[0]
        event_lines = lines[1:]

    events = []
    for idx, line in enumerate(event_lines):
        ev_id_match = re.match(r'^(?:E|e|Event\s*)(\d+)[\.:\s]*', line, re.IGNORECASE)
        if ev_id_match:
            ev_id = int(ev_id_match.group(1))
            clean_text = line[ev_id_match.end():].strip()
        else:
            ev_id = idx + 1
            clean_text = re.sub(r'^\d+[\.:\s]*', '', line).strip()

        events.append({
            "event_id": ev_id,
            "id": ev_id,
            "name": clean_text,
            "event_name": clean_text,
            "description": clean_text,
        })

    if not activity and events:
        activity = " ".join(e["description"] for e in events[:2])

    return {"activity": activity, "events": events}


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
        description, question = split_qa_text(content)
        return {
            "query_id": qid,
            "type": "qa",
            "question": question,
            "description": description,
            "text": content,
        }

    elif qtype == "trake":
        parsed = parse_trake_text(content)
        return {
            "query_id": qid,
            "type": "trake",
            "activity": parsed["activity"],
            "text": content,
            "events": parsed["events"],
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
