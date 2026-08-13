"""
OCR V3 Retrieval Final Audit (Failure Analysis, Normalization Check, Ablation & Hybrid Weight Grid)
==================================================================================================
1. Audits 25 retrieval test queries.
2. Generates detailed failure analysis for failed queries (outputs/evaluation/ocr_v3/retrieval_failure_analysis.csv).
3. Performs Retrieval Ablation (BM25 only vs Semantic only vs Hybrid).
4. Grid searches Hybrid Weight (alpha = 0.2, 0.3, 0.4, 0.5, 0.6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.retrieval.ocr_v3_index import OCRV3HybridRetriever

BENCHMARK_QUERIES = [
    # Category 1: Exact Text Queries
    {"query_id": "Q01", "type": "exact", "query": "TÌNH TRẠNG SỤT LÚN Ở ĐBSCL", "target_video": "L21_V001", "target_keyword": "sụt lún"},
    {"query_id": "Q02", "type": "exact", "query": "TRÁI TIM ĐƯỢC VẬN CHUYỂN CẤP TỐC VỀ HUẾ", "target_video": "L21_V001", "target_keyword": "trái tim"},
    {"query_id": "Q03", "type": "exact", "query": "CẢNH BÁO SẠT LỞ NGUY HIỂM", "target_video": "L21_V001", "target_keyword": "sạt lở"},
    {"query_id": "Q04", "type": "exact", "query": "TẠM DỪNG LƯU THÔNG", "target_video": "L21_V001", "target_keyword": "tạm dừng"},
    {"query_id": "Q05", "type": "exact", "query": "ĐỘI CSGT RẠCH CHIẾC", "target_video": "L21_V002", "target_keyword": "rạch chiếc"},
    {"query_id": "Q06", "type": "exact", "query": "Nghệ An Bình gas bất ngờ", "target_video": "L21_V002", "target_keyword": "bình gas"},
    {"query_id": "Q07", "type": "exact", "query": "Ngân hàng Trung ương Anh", "target_video": "L21_V002", "target_keyword": "ngân hàng"},
    {"query_id": "Q08", "type": "exact", "query": "giảm lãi suất", "target_video": "L21_V002", "target_keyword": "lãi suất"},
    {"query_id": "Q09", "type": "exact", "query": "Lào Cai", "target_video": "L21_V003", "target_keyword": "lào cai"},
    {"query_id": "Q10", "type": "exact", "query": "Hà Nội", "target_video": "L21_V005", "target_keyword": "hà nội"},

    # Category 2: Typo & No-Accent Queries
    {"query_id": "Q11", "type": "typo", "query": "tinh trang sut lun dbscl", "target_video": "L21_V001", "target_keyword": "sụt lún"},
    {"query_id": "Q12", "type": "typo", "query": "trai tim van chuyen cap toc hue", "target_video": "L21_V001", "target_keyword": "trái tim"},
    {"query_id": "Q13", "type": "typo", "query": "canh bao sat lo nguy hiem", "target_video": "L21_V001", "target_keyword": "sạt lở"},
    {"query_id": "Q14", "type": "typo", "query": "doi csgt rach chiec", "target_video": "L21_V002", "target_keyword": "rạch chiếc"},
    {"query_id": "Q15", "type": "typo", "query": "nghe an binh gas boc chay", "target_video": "L21_V002", "target_keyword": "bình gas"},
    {"query_id": "Q16", "type": "typo", "query": "ngan hang trung uong anh", "target_video": "L21_V002", "target_keyword": "ngân hàng"},
    {"query_id": "Q17", "type": "typo", "query": "lao cai thieu tien", "target_video": "L21_V003", "target_keyword": "lào cai"},
    {"query_id": "Q18", "type": "typo", "query": "ha noi tin tuc", "target_video": "L21_V005", "target_keyword": "hà nội"},

    # Category 3: Semantic & Paraphrase Queries
    {"query_id": "Q19", "type": "semantic", "query": "tin về tình trạng sụt lở đất tại đồng bằng sông Cửu Long", "target_video": "L21_V001", "target_keyword": "sụt lún"},
    {"query_id": "Q20", "type": "semantic", "query": "vận chuyển nội tạng tim ghép bệnh nhân", "target_video": "L21_V001", "target_keyword": "trái tim"},
    {"query_id": "Q21", "type": "semantic", "query": "cảnh sát giao thông giúp đỡ người dân", "target_video": "L21_V002", "target_keyword": "rạch chiếc"},
    {"query_id": "Q22", "type": "semantic", "query": "sự cố nổ bình gas cháy nhà ở Nghệ An", "target_video": "L21_V002", "target_keyword": "bình gas"},
    {"query_id": "Q23", "type": "semantic", "query": "chính sách tiền tệ cắt giảm lãi suất ngân hàng", "target_video": "L21_V002", "target_keyword": "lãi suất"},
    {"query_id": "Q24", "type": "semantic", "query": "thông tin giao thông tạm dừng xe qua lại", "target_video": "L21_V001", "target_keyword": "tạm dừng"},
    {"query_id": "Q25", "type": "semantic", "query": "báo cáo tài chính thu nhập kinh tế", "target_video": "L21_V002", "target_keyword": "lãi suất"},
]


def run_full_retrieval_audit():
    print("=" * 80)
    print(" 🎯 RUNNING FULL RETRIEVAL AUDIT & ABLATION STUDY (All 29 L21 Videos)")
    print("=" * 80)

    out_idx_dir = PROJECT_ROOT / "outputs" / "indexes" / "ocr_v3"
    parquet_path = out_idx_dir / "l21_ocr_v3_corpus.parquet"
    faiss_path = out_idx_dir / "l21_ocr_v3_flat_ip.faiss"

    queries = BENCHMARK_QUERIES
    full_queries_json = PROJECT_ROOT / "outputs" / "evaluation" / "ocr_v3" / "full_l21_eval_queries.json"
    if full_queries_json.exists():
        import json
        queries = json.loads(full_queries_json.read_text(encoding="utf-8"))
        print(f"📊 Loaded {len(queries)} comprehensive evaluation queries from: {full_queries_json}")
    else:
        print(f"📊 Using fallback {len(queries)} benchmark queries.")

    df_corpus = pd.read_parquet(parquet_path)
    retriever = OCRV3HybridRetriever(segments_df=df_corpus, faiss_index_path=faiss_path)

    # 1. Ablation Study: BM25 Only vs Semantic Only vs Hybrid (alpha=0.5)
    ablation_results = {}
    for mode_name, alpha_val in [("BM25 Only", 0.0), ("Semantic Only", 1.0), ("Hybrid (0.5/0.5)", 0.5)]:
        r1 = r5 = r10 = 0
        for q in queries:
            res = retriever.search(q["query"], top_k=10, alpha=alpha_val)
            for rank, item in enumerate(res, start=1):
                if item.get("video_id") == q["target_video"] and q["target_keyword"].lower() in str(item.get("text_consensus", "")).lower():
                    if rank == 1: r1 += 1
                    if rank <= 5: r5 += 1
                    if rank <= 10: r10 += 1
                    break
        total = len(queries)
        ablation_results[mode_name] = {
            "R@1": round((r1 / total) * 100, 1),
            "R@5": round((r5 / total) * 100, 1),
            "R@10": round((r10 / total) * 100, 1),
        }

    print("\n" + "=" * 60)
    print(" 🏆 RETRIEVAL ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"{'Branch / Engine':<25} | {'R@1 (%)':<8} | {'R@5 (%)':<8} | {'R@10 (%)':<8}")
    print("-" * 60)
    for mode, metrics in ablation_results.items():
        print(f"{mode:<25} | {metrics['R@1']:<8.1f} | {metrics['R@5']:<8.1f} | {metrics['R@10']:<8.1f}")
    print("=" * 60)

    # 2. Grid Search Hybrid Alpha Weights
    grid_results = {}
    for alpha_val in [0.2, 0.3, 0.4, 0.5, 0.6]:
        bm25_w = round(1 - alpha_val, 1)
        key_str = f"BM25 {bm25_w} / Sem {alpha_val}"
        r1 = r5 = r10 = 0
        for q in queries:
            res = retriever.search(q["query"], top_k=10, alpha=alpha_val)
            for rank, item in enumerate(res, start=1):
                if item.get("video_id") == q["target_video"] and q["target_keyword"].lower() in str(item.get("text_consensus", "")).lower():
                    if rank == 1: r1 += 1
                    if rank <= 5: r5 += 1
                    if rank <= 10: r10 += 1
                    break
        total = len(queries)
        grid_results[key_str] = {
            "R@1": round((r1 / total) * 100, 1),
            "R@5": round((r5 / total) * 100, 1),
            "R@10": round((r10 / total) * 100, 1),
        }

    print("\n" + "=" * 60)
    print(" 📊 HYBRID WEIGHT GRID SEARCH (Development Benchmark)")
    print("=" * 60)
    print(f"{'Weight Ratio':<25} | {'R@1 (%)':<8} | {'R@5 (%)':<8} | {'R@10 (%)':<8}")
    print("-" * 60)
    for mode, metrics in grid_results.items():
        print(f"{mode:<25} | {metrics['R@1']:<8.1f} | {metrics['R@5']:<8.1f} | {metrics['R@10']:<8.1f}")
    print("=" * 60)

    # 3. Failure Analysis for Failed Queries
    failure_records = []
    category_counts = {}

    for q in queries:
        q_text = q["query"]
        target_vid = q["target_video"]
        target_kw = q["target_keyword"].lower()

        res_hybrid = retriever.search(q_text, top_k=50, alpha=0.5)
        res_bm25 = retriever.search(q_text, top_k=50, alpha=0.0)
        res_sem = retriever.search(q_text, top_k=50, alpha=1.0)

        def get_rank(results_list):
            for rank, item in enumerate(results_list, start=1):
                if item.get("video_id") == target_vid and target_kw in str(item.get("text_consensus", "")).lower():
                    return rank, item
            return 999, None

        h_rank, h_item = get_rank(res_hybrid)
        b_rank, _ = get_rank(res_bm25)
        s_rank, _ = get_rank(res_sem)

        if h_rank > 10:
            # Failure categorization
            if b_rank <= 10 and s_rank > 20:
                cat = "G. HYBRID_FUSION_FAIL"
                notes = "Semantic branch pulled down high BM25 rank"
            elif b_rank > 20 and s_rank <= 10:
                cat = "E. BM25_RANKING_FAIL"
                notes = "Lexical BM25 failed to match tokens"
            elif b_rank > 50 and s_rank > 50:
                # Check corpus existence
                matching_seg = df_corpus[(df_corpus["video_id"] == target_vid) & (df_corpus["text_consensus"].str.lower().str.contains(target_kw))]
                if len(matching_seg) == 0:
                    cat = "A. OCR_FAIL"
                    notes = "Target OCR text missing from corpus segment"
                else:
                    cat = "D. QUERY_NORMALIZATION_FAIL"
                    notes = "Query phrasing or diacritics mismatch"
            else:
                cat = "F. SEMANTIC_RANKING_FAIL"
                notes = "Semantic distance too large"

            category_counts[cat] = category_counts.get(cat, 0) + 1

            failure_records.append({
                "query_id": q["query_id"],
                "query": q_text,
                "query_type": q["type"],
                "expected_video_id": target_vid,
                "expected_keyword": target_kw,
                "expected_segment_exists": 1 if h_item else 0,
                "bm25_rank": b_rank,
                "semantic_rank": s_rank,
                "hybrid_rank": h_rank,
                "failure_category": cat,
                "notes": notes,
            })

    # Save failure analysis CSV
    out_dir = PROJECT_ROOT / "outputs" / "evaluation" / "ocr_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_fail = pd.DataFrame(failure_records)
    fail_csv = out_dir / "retrieval_failure_analysis.csv"
    df_fail.to_csv(fail_csv, index=False, encoding="utf-8-sig")

    print(f"\n📄 Saved Failure Analysis CSV ({len(df_fail)} failed queries) to: {fail_csv}")
    print("\n--- FAILURE CATEGORY BREAKDOWN ---")
    for cat, cnt in category_counts.items():
        print(f"  └─ {cat:<30}: {cnt} queries")

    return ablation_results, grid_results, category_counts


if __name__ == "__main__":
    run_full_retrieval_audit()
