"""
Query Runner CLI — batch-process AIC queries from a JSON file.

Reads a query JSON file, runs the RetrievalPipeline for each query,
and writes BTC-standard submission CSV files for all 3 task types.

Usage (local):
    python scripts/run_queries.py \\
        --queries datasets/queries/sample_queries.json \\
        --index-dir indexes \\
        --output-dir outputs/submission

Usage (Kaggle Notebook):
    !python AIC_System/scripts/run_queries.py \\
        --queries /kaggle/input/aic-queries/queries.json \\
        --index-dir /kaggle/input/aic-indexes \\
        --keyframe-root /kaggle/input/aic-data/keyframes \\
        --enable-vlm \\
        --output-dir /kaggle/working/submission

Input JSON format (array of query objects — all 3 types supported):
    [
        {"query_id": "q001", "type": "textual_kis", "text": "..."},
        {"query_id": "q002", "type": "qa", "description": "...", "question": "..."},
        {"query_id": "q003", "type": "trake", "activity": "...", "events": [...]}
    ]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.evaluation.submission_formatter import SubmissionFormatter
from src.common.enums import QueryType
from src.common.query_loader import load_queries
from src.reasoning.query_classifier import QueryClassifier
from src.utils.logger import get_logger

logger = get_logger("run_queries")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AIC queries (KIS / Q&A / TRAKE) and produce submission CSVs"
    )
    # ── Input: batch file hoặc single query ─────────────────────────────────
    parser.add_argument("--queries",       default="",
                        help="Path to queries JSON/TXT file or directory (dùng cho batch)")
    parser.add_argument("--query-text",   default="",
                        help="🔍 Chạy thử NGAY 1 câu query (thay thế --queries). Ví dụ: \"người mặc áo đỏ đứng trên sân khấu\"")
    parser.add_argument("--query-type",   default="textual_kis",
                        choices=["textual_kis", "qa", "trake"],
                        help="Loại query khi dùng --query-text (default: textual_kis)")
    # ── Index paths ──────────────────────────────────────────────────────────
    parser.add_argument("--index-dir",     default="indexes",
                        help="Dir containing faiss_visual.index + keyframe_master.parquet")
    parser.add_argument("--output-dir",    default="outputs/submission",
                        help="Output directory for submission CSV files")
    parser.add_argument("--keyframe-root", default="",
                        help="Root dir of keyframe images (required for Q&A and TRAKE VLM)")
    parser.add_argument("--ocr-dir",       default="",
                        help="Dir containing extracted per-video OCR JSON/JSONL files")
    parser.add_argument("--asr-dir",       default="",
                        help="Dir containing ASR corpus parquet (datasets/artifacts/indexes/asr_v3)")
    # ── Model options ────────────────────────────────────────────────────────
    parser.add_argument("--enable-vlm",    action="store_true",
                        help="Load Qwen2.5-VL for Q&A answer extraction and TRAKE verification")
    parser.add_argument("--vlm-model",     default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="VLM model name (default: Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--qdrant-url",    default="",
                        help="Qdrant URL for text retrieval (e.g. http://localhost:6333)")
    parser.add_argument("--top-k",         type=int, default=100,
                        help="Number of retrieval candidates (default: 100)")
    parser.add_argument("--clip-model",    default="ViT-B-32",
                        help="CLIP model name (default: ViT-B-32)")
    parser.add_argument("--device",        default=None,
                        help="Compute device: cuda / cpu (auto-detect if not set)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------------
    # Mode A: Chạy thử 1 câu query trực tiếp (--query-text)
    # --------------------------------------------------------
    if args.query_text:
        queries = [{
            "query_id": "test_q001",
            "type": args.query_type,
            "text": args.query_text,
        }]
        logger.info(f"[SINGLE QUERY MODE] type={args.query_type} | text='{args.query_text}'")
    elif args.queries:
        # --------------------------------------------------------
        # Mode B: Batch — đọc từ file JSON/TXT
        # --------------------------------------------------------
        queries_path = Path(args.queries)
        if not queries_path.exists():
            logger.error(f"Query path not found: {queries_path}")
            sys.exit(1)
        queries = load_queries(queries_path)
        logger.info(f"Successfully loaded {len(queries)} queries to execute.")
    else:
        logger.error("Cần truyền --query-text (single) hoặc --queries (batch)")
        sys.exit(1)

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------
    logger.info("Loading RetrievalPipeline...")
    t0 = time.time()
    pipeline = RetrievalPipeline.from_index_dir(
        index_dir=args.index_dir,
        clip_model=args.clip_model,
        device=args.device,
        keyframe_image_root=args.keyframe_root,
        enable_vlm=args.enable_vlm,
        vlm_model=args.vlm_model,
        vlm_load_in_4bit=True,
        qdrant_url=args.qdrant_url or None,
        top_k_retrieval=args.top_k,
        top_k_fusion=args.top_k,
        ocr_dir=args.ocr_dir or None,
        asr_dir=args.asr_dir or None,
    )
    logger.info(f"Pipeline ready in {time.time() - t0:.1f}s")

    # --------------------------------------------------------
    # Run queries
    # --------------------------------------------------------
    formatter  = SubmissionFormatter(output_dir=args.output_dir)
    classifier = QueryClassifier()

    t_total = time.time()
    errors = 0

    for i, query_dict in enumerate(queries):
        qid   = str(query_dict.get("query_id", i))
        qtype = classifier.classify(query_dict)
        t_q   = time.time()

        try:
            evidence = pipeline.run(query_dict, query_id=qid)
        except Exception as e:
            logger.error(f"Query {qid} failed: {e}")
            errors += 1
            continue

        if evidence is None:
            logger.warning(f"No result for query_id={qid}")
            continue

        # ── Route to appropriate submission bucket ────────────
        if qtype == QueryType.TEXTUAL_KIS:
            formatter.add_kis(qid, evidence)

        elif qtype == QueryType.QA:
            answer = evidence.metadata.get("answer", "")
            formatter.add_qa(qid, evidence, answer=answer)

        elif qtype == QueryType.TRAKE:
            trake_sub = evidence.metadata.get("trake_submission")
            if trake_sub is not None:
                # Full TRAKE: one frame_idx per event step
                event_frame_idxs = {ev.event_id: ev.frame_idx for ev in trake_sub.events}
                formatter.add_trake(qid, trake_sub.video_id, event_frame_idxs, evidence=evidence)
            else:
                # Fallback: map single frame across all events in query
                fidx = evidence.frame_idx if evidence.frame_idx > 0 else 1
                raw_events = query_dict.get("events", [])
                if isinstance(raw_events, list) and len(raw_events) > 0:
                    n_evs = len(raw_events)
                else:
                    # Count E1, E2, E3, E4... occurrences in query text
                    qtext = str(query_dict.get("text", ""))
                    e_matches = re.findall(r'(?:E|Event\s*)\d+', qtext, re.IGNORECASE)
                    n_evs = len(e_matches) if len(e_matches) > 0 else 4

                event_frame_idxs = {ev_id: fidx + (ev_id - 1) * 5 for ev_id in range(1, n_evs + 1)}
                formatter.add_trake(qid, evidence.video_id, event_frame_idxs, evidence=evidence)

        elapsed = time.time() - t_q
        logger.info(
            f"[{i+1}/{len(queries)}] {qid} ({qtype.value}): "
            f"{evidence.video_id} frame_idx={evidence.frame_idx} "
            f"pts={evidence.pts_time:.2f}s  ({elapsed:.2f}s)"
        )

    # --------------------------------------------------------
    # Save all submission files at once
    # --------------------------------------------------------
    paths = formatter.save_all()   # auto-detects TRAKE event count

    stats = formatter.stats()
    total_time = time.time() - t_total
    avg_time = total_time / max(len(queries), 1)

    logger.info("=" * 60)
    logger.info(f"Done in {total_time:.1f}s  (avg {avg_time:.2f}s/query)")
    logger.info(f"KIS:   {stats['kis']} results  → {paths['kis'].name}")
    logger.info(f"Q&A:   {stats['qa']} results  → {paths['qa'].name}")
    logger.info(f"TRAKE: {stats['trake']} results  → {paths['trake'].name}")
    logger.info(f"Top-100 ZIP: {paths['top100_zip']}")
    logger.info(f"  ↳ Mỗi query sẽ có 1 file CSV có tới 100 đáp án điểm cao nhất")
    logger.info(f"  ↳ Q&A: mỗi row gồm video_id, frame_idx, answer (answer khác nhau theo từng query)")
    logger.info(f"Errors: {errors}")
    logger.info(f"Output: {Path(args.output_dir).resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
