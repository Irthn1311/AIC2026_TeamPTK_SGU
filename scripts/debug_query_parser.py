"""
Debug Utility for Query Understanding Parser
============================================
Usage:
    python scripts/debug_query_parser.py "thuyền máy chạy trên sông"
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_understanding.parser import RuleBasedQueryParser


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "thuyền máy chạy trên sông"

    parser = RuleBasedQueryParser()
    result = parser.parse(query)

    print("=" * 60)
    print(" 🧠 QUERY UNDERSTANDING BREAKDOWN")
    print("=" * 60)
    print(f"Original       : {result.original_query}")
    print(f"Intent         : {result.intent.value if hasattr(result.intent, 'value') else result.intent}")
    print(f"Confidence     : {result.confidence:.2f}")
    print("-" * 60)
    print(f"Objects ({len(result.object_terms)})    : {', '.join(result.object_terms) if result.object_terms else 'None'}")
    print(f"Actions ({len(result.actions)})    : {', '.join(result.actions) if result.actions else 'None'}")
    print(f"Scenes ({len(result.scene_terms)})     : {', '.join(result.scene_terms) if result.scene_terms else 'None'}")
    print(f"Temporal ({len(result.temporal_terms)})   : {', '.join(result.temporal_terms) if result.temporal_terms else 'None'}")
    print("-" * 60)
    print(f"OCR Query      : {result.ocr_query or 'None'}")
    print(f"ASR Query      : {result.asr_query or 'None'}")
    print(f"Visual Query   : {result.visual_query or 'None'}")
    print("-" * 60)
    print("Likelihoods    :")
    print(f"  Visual       : {result.visual_likelihood:.2f}")
    print(f"  OCR          : {result.ocr_likelihood:.2f}")
    print(f"  ASR          : {result.asr_likelihood:.2f}")
    print(f"  Object       : {result.object_likelihood:.2f}")
    print("-" * 60)
    print(f"Matched Rules ({len(result.matched_rules)}) :")
    for r in result.matched_rules:
        print(f"  - {r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
