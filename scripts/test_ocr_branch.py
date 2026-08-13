"""
Interactive OCR V3 Retrieval & Query Testing Tool
=================================================
Allows testing arbitrary query search on the 65,286 OCR segment index
using the Hybrid BM25 (Inverted Index) + FAISS (E5 Semantic) engine.

Usage:
    python scripts/test_ocr_branch.py --query "sụt lún ở đồng bằng sông Cửu Long"
    python scripts/test_ocr_branch.py --query "cảnh sát giao thông" --top_k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.retrieval.ocr_v3_index import OCRV3HybridRetriever


def run_ocr_search(query: str, top_k: int = 10, alpha: float = 0.5):
    parquet_path = PROJECT_ROOT / "outputs" / "indexes" / "ocr_v3" / "l21_ocr_v3_corpus.parquet"
    faiss_path = PROJECT_ROOT / "outputs" / "indexes" / "ocr_v3" / "l21_ocr_v3_flat_ip.faiss"

    if not parquet_path.exists():
        print(f"❌ Không tìm thấy corpus OCR tại {parquet_path}.")
        return

    print("=" * 80)
    print(" 🔍 CÔNG CỤ TRUY XUẤT OCR V3 (HYBRID BM25 + SEMANTIC E5)")
    print(f" 📝 Truy vấn (Query): '{query}'")
    print(f" ⚙️ Cấu hình: top_k={top_k}, alpha={alpha} (BM25: {1-alpha:.1f} / Semantic: {alpha:.1f})")
    print("=" * 80)

    df_corpus = pd.read_parquet(parquet_path)
    retriever = OCRV3HybridRetriever(segments_df=df_corpus, faiss_index_path=faiss_path)

    results = retriever.search(query, top_k=top_k, alpha=alpha)

    if not results:
        print(f"⚠️ Không tìm thấy kết quả nào phù hợp với truy vấn '{query}'.")
        return

    print(f"\n✅ TÌM THẤY {len(results)} KẾT QUẢ PHÙ HỢP NHẤT:\n")
    print(f"{'Hạng':<5} | {'Điểm':<7} | {'Video ID':<10} | {'Thời gian':<16} | {'Khu vực':<9} | {'Văn bản OCR nhận dạng được'}")
    print("-" * 105)

    for rank, r in enumerate(results, start=1):
        vid = r.get("video_id", "N/A")
        t_start = r.get("time_start_sec", 0.0)
        t_end = r.get("time_end_sec", 0.0)
        time_str = f"{t_start:.1f}s - {t_end:.1f}s"
        score = r.get("score", 0.0)
        reg_type = r.get("region_type", "other")
        text = str(r.get("text_consensus", "")).strip()
        if len(text) > 55:
            text = text[:52] + "..."

        print(f"{rank:<5} | {score:<7.4f} | {vid:<10} | {time_str:<16} | {reg_type:<9} | {text}")

    print("=" * 105)
    print("💡 Mẹo: Thử nghiệm với các câu hỏi tiếng Việt có dấu, không dấu hoặc câu văn mô tả ngữ nghĩa!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCR V3 Hybrid Retrieval")
    parser.add_argument("--query", "-q", type=str, default="TÌNH TRẠNG SỤT LÚN Ở ĐBSCL", help="Text query to search")
    parser.add_argument("--top_k", "-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--alpha", "-a", type=float, default=0.5, help="Hybrid weight: 0.0 (BM25) to 1.0 (Semantic)")
    args = parser.parse_args()

    run_ocr_search(args.query, top_k=args.top_k, alpha=args.alpha)
