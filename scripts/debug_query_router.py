"""
Debug Utility for Query Router & Dynamic Fusion Decision
========================================================
Usage:
    python scripts/debug_query_router.py "thuyền máy chạy trên sông"
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_understanding.parser import RuleBasedQueryParser
from src.query_understanding.router import QueryRouter


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "thuyền máy chạy trên sông"

    parser = RuleBasedQueryParser()
    router = QueryRouter()

    analysis = parser.parse(query)
    decision = router.route(analysis)

    print("=" * 60)
    print(" 🧭 QUERY ROUTING BREAKDOWN")
    print("=" * 60)
    print(f"Query: {analysis.original_query}")
    print(f"Intent: {analysis.intent.value if hasattr(analysis.intent, 'value') else analysis.intent}")
    print(f"Parser confidence: {analysis.confidence:.2f}")
    print("\nLikelihoods:")
    print(f"  Visual  : {analysis.visual_likelihood:.2f}")
    print(f"  OCR     : {analysis.ocr_likelihood:.2f}")
    print(f"  ASR     : {analysis.asr_likelihood:.2f}")
    print(f"  Object  : {analysis.object_likelihood:.2f}")
    print("\nBASELINE")
    print(f"  Visual  : {decision.baseline_weights.visual:.2f}")
    print(f"  OCR     : {decision.baseline_weights.ocr:.2f}")
    print(f"  ASR     : {decision.baseline_weights.asr:.2f}")
    print(f"  Object  : {decision.baseline_weights.object:.2f}")
    print("\nDYNAMIC POLICY")
    print(f"  Visual  : {decision.dynamic_weights.visual:.2f}")
    print(f"  OCR     : {decision.dynamic_weights.ocr:.2f}")
    print(f"  ASR     : {decision.dynamic_weights.asr:.2f}")
    print(f"  Object  : {decision.dynamic_weights.object:.2f}")
    print(f"\nBlend alpha: {decision.blend_factor:.2f}")
    print("\nFINAL (SUM = 1.0)")
    print(f"  Visual  : {decision.final_weights.visual:.2f}")
    print(f"  OCR     : {decision.final_weights.ocr:.2f}")
    print(f"  ASR     : {decision.final_weights.asr:.2f}")
    print(f"  Object  : {decision.final_weights.object:.2f}")
    print(f"\nDominant : {decision.dominant_branch.upper()}")
    print(f"Secondary: {decision.secondary_branch.upper()}")
    print("\nReason:")
    for r in decision.routing_reason:
        print(f"  - {r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
