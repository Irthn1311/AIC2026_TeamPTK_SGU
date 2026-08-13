"""
Test Branch 3: BTC Objects Index Search
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.retrieval.object_index import ObjectIndex

def test_branch3(query: str, top_k: int = 5):
    corpus_path = PROJECT_ROOT / "outputs" / "indexes" / "object" / "l21_objects.parquet"

    print("=" * 60)
    print("  TESTING BRANCH 3 (BTC OBJECTS INDEX)")
    print(f"  Query: '{query}'")
    print("=" * 60)

    if not corpus_path.exists():
        print(f"Error: Object index not found at {corpus_path}")
        return

    idx = ObjectIndex(corpus_path)
    res = idx.search(query, top_k=top_k)

    if res.empty:
        print("No matching objects found for query.")
        return

    print(f"\nTop {len(res)} Object Matching Results:")
    for _, row in res.iterrows():
        unique_objs = row.get("unique_object_classes", [])
        if isinstance(unique_objs, str):
            try:
                import ast
                unique_objs = ast.literal_eval(unique_objs)
            except Exception:
                pass
        objs_str = ", ".join(unique_objs[:5]) if isinstance(unique_objs, (list, tuple)) else str(unique_objs)
        print(f"  #{row['rank']} Score: {row['object_match_score']:.4f} | Video: {row['video_id']} | Frame: {row['frame_idx']:>6d} ({row['timestamp_seconds']:.2f}s) | Objects: [{objs_str}]")

    print("\n" + "=" * 60)
    print("  BRANCH 3 STATUS: READY & FUNCTIONAL")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="thuyền thuyền máy boat")
    args = parser.parse_args()
    test_branch3(args.query)
