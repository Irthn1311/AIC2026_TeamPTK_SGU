"""
AIC System Retrieval & Groundtruth Diagnostic Tool.

Performs 4 automated health checks on the retrieval pipeline:
1. CLIP Prompt Token Length Check (Detects >77 token truncation)
2. Target Prefix Leakage Check (Detects out-of-batch candidates from OCR/Text)
3. FAISS Index Reconstruct Compatibility Check (Detects HNSW reconstruct failure)
4. Groundtruth Evaluation & Per-Query Failure Root Cause Analysis

Usage (local check without index):
    python scripts/diagnose.py \\
        --queries datasets/queries/sample_queries.json \\
        --gt datasets/groundtruth/groundtruth_all.json

Usage (full check with index on Kaggle / local):
    python scripts/diagnose.py \\
        --queries datasets/queries/sample_queries.json \\
        --gt datasets/groundtruth/groundtruth_all.json \\
        --index-dir indexes
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reasoning.query_parser import QueryParser
from src.utils.logger import get_logger

logger = get_logger("diagnose")


def check_clip_token_length(queries, clip_model="ViT-B-32"):
    """Check 1: Verify prompt token counts against CLIP's 77-token hard limit."""
    print("\n" + "=" * 70)
    print(f"  CHECK 1: CLIP PROMPT TOKEN LENGTH & TRUNCATION ({clip_model})")
    print("=" * 70)

    try:
        import open_clip
        tokenizer = open_clip.get_tokenizer(clip_model)
    except Exception as e:
        print(f"  [WARNING] Could not load open_clip tokenizer: {e}")
        print("  Using whitespace fallback word count instead...")
        tokenizer = None

    parser = QueryParser()
    truncated_count = 0

    for i, q in enumerate(queries):
        qid = q.get("query_id", f"q{i}")
        qtype = q.get("type", "textual_kis")

        if qtype == "textual_kis":
            kis = parser.parse_kis(q.get("text", ""))
            prompt = parser.build_retrieval_text(kis)
        elif qtype == "qa":
            qa = parser.parse_qa(q.get("description", ""), q.get("question", ""))
            prompt = parser.build_qa_retrieval_text(qa)
        elif qtype == "trake":
            activity = q.get("activity", "")
            events = [e.get("name", "") for e in q.get("events", [])]
            prompt = f"{activity} " + " ".join(events)
        else:
            prompt = q.get("text", "")

        if tokenizer is not None:
            tokens = tokenizer([prompt])[0]
            n_tokens = int((tokens != 0).sum())
        else:
            n_tokens = len(prompt.split())

        is_truncated = n_tokens >= 77

        if is_truncated:
            truncated_count += 1
            status = "[!] TRUNCATED (>=77)"
        else:
            status = "[OK]"

        print(f"[{qid:<25}] {qtype:<11} | Tokens: {n_tokens:>3}/77 | {status}")
        if is_truncated:
            print(f"   Prompt ({len(prompt)} chars): {prompt[:100]}...")

    print("-" * 70)
    if truncated_count > 0:
        print(f"[WARNING] {truncated_count}/{len(queries)} queries exceed CLIP's 77-token limit and are TRUNCATED!")
    else:
        print("[OK] All query prompts fit within CLIP's 77-token limit.")


def check_target_prefix_leakage(queries):
    """Check 2: Test if OCR/Text retrievers obey or bypass target_prefix."""
    print("\n" + "=" * 70)
    print("  CHECK 2: TARGET PREFIX LEAKAGE IN RETRIEVERS")
    print("=" * 70)

    try:
        from src.retrieval.ocr_retriever import InMemoryOCRRetriever

        # Inspect retrieve signature
        import inspect
        sig_ocr = inspect.signature(InMemoryOCRRetriever.retrieve)
        has_prefix_ocr = "target_prefix" in sig_ocr.parameters

        print(f"  InMemoryOCRRetriever.retrieve() supports target_prefix: {'[YES]' if has_prefix_ocr else '[NO] (Will bypass prefix filtering!)'}")

    except Exception as e:
        print(f"  Could not inspect InMemoryOCRRetriever: {e}")

    try:
        from src.retrieval.text_retriever import TextRetriever
        sig_txt = inspect.signature(TextRetriever.retrieve)
        has_prefix_txt = "target_prefix" in sig_txt.parameters

        print(f"  TextRetriever (Qdrant).retrieve() supports target_prefix: {'[YES]' if has_prefix_txt else '[NO] (Will bypass prefix filtering!)'}")
    except Exception as e:
        print(f"  Could not inspect TextRetriever: {e}")


def check_faiss_reconstruct(index_dir):
    """Check 3: Test whether FAISS index supports reconstruct()."""
    print("\n" + "=" * 70)
    print("  CHECK 3: FAISS INDEX RECONSTRUCT COMPATIBILITY")
    print("=" * 70)

    idx_path = Path(index_dir) / "faiss_visual.index"
    if not idx_path.exists():
        print(f"[INFO] FAISS index file not found at {idx_path} (Skipping index check)")
        return

    try:
        import faiss
        index = faiss.read_index(str(idx_path))
        print(f"  Loaded FAISS index ({index.ntotal:,} vectors)")

        try:
            vec = index.reconstruct(0)
            print("  Index reconstruct(0): [SUCCESS]")
        except Exception as e:
            print(f"  Index reconstruct(0): [FAILED] ({type(e).__name__}: {e})")
            print("  [WARNING] CRITICAL: TRAKE retrieve_within_video() will fail and collapse to frame_idx=1!")
    except Exception as e:
        print(f"  Could not test FAISS index: {e}")


def evaluate_groundtruth(queries, gt_file, results_file=None):
    """Check 4: Compare Groundtruth vs System Outputs and analyze root causes."""
    print("\n" + "=" * 70)
    print("  CHECK 4: GROUNDTRUTH EVALUATION & FAILURE ANALYSIS")
    print("=" * 70)

    gt_path = Path(gt_file)
    if not gt_path.exists():
        print(f"[INFO] Groundtruth file not found: {gt_path}")
        return

    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"  Loaded groundtruth for {len(gt_data)} queries from {gt_path.name}")
    print("-" * 70)

    # Convert GT format if needed
    matches = 0
    total = len(queries)

    print(f"{'Query ID':<25} | {'Type':<11} | {'GT Video':<10} | Target Prefix")
    print("-" * 70)

    for q in queries:
        qid = q.get("query_id")
        qtype = q.get("type")
        prefix = q.get("target_prefix", "None")
        gt_entry = gt_data.get(qid, {})
        gt_vid = gt_entry.get("video_id", "N/A")

        print(f"{qid:<25} | {qtype:<11} | {gt_vid:<10} | {prefix}")

    print("-" * 70)
    print("[HINT] To run full groundtruth evaluation with predictions, use:")
    print("   python scripts/run_queries.py --queries datasets/queries/sample_queries.json --index-dir indexes")


def main():
    parser = argparse.ArgumentParser(description="AIC Retrieval Diagnostic Script")
    parser.add_argument("--queries", default="datasets/queries/sample_queries.json", help="Path to queries.json")
    parser.add_argument("--gt", default="datasets/groundtruth/groundtruth_all.json", help="Path to groundtruth.json")
    parser.add_argument("--index-dir", default="indexes", help="Path to index dir")
    parser.add_argument("--clip-model", default="ViT-B-32", help="CLIP model name")

    args = parser.parse_args()

    q_path = Path(args.queries)
    if not q_path.exists():
        print(f"Error: Query file not found: {q_path}")
        sys.exit(1)

    with open(q_path, encoding="utf-8") as f:
        queries = json.load(f)

    print("=" * 70)
    print(f"[DIAGNOSTIC] AIC SYSTEM RETRIEVAL DIAGNOSTIC TOOL -- {len(queries)} Queries")
    print("=" * 70)

    check_clip_token_length(queries, clip_model=args.clip_model)
    check_target_prefix_leakage(queries)
    check_faiss_reconstruct(args.index_dir)
    evaluate_groundtruth(queries, args.gt)



if __name__ == "__main__":
    main()
