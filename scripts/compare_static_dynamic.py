"""
Static vs Dynamic A/B Comparison and Benchmark Suite (Phase 4)
=============================================================
Evaluates retrieval behavior across 25 diverse queries (15 Phase 3 + 10 L21 domain queries).
Measures:
- Effective weights (Static baseline vs Dynamic routing)
- Top-K candidate overlap (Overlap@5, Overlap@10, Overlap@20, Overlap@50, Overlap@100)
- Rank shifts and branch score contributions
- Latency and dynamic routing overhead
- Exports results to CSV and HTML reports in outputs/evaluation/
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval_service import RetrievalService
from backend.schemas import FusionWeights


BENCHMARK_QUERIES = [
    # 15 Phase 3 Test Queries
    "thuyền máy chạy trên sông",
    "người đàn ông mặc áo đỏ đang bước xuống xe",
    "trên màn hình xuất hiện dòng chữ Bộ Y tế",
    "phóng viên nói rằng mưa sẽ tiếp tục kéo dài",
    "người phụ nữ đang phát biểu trước tòa nhà",
    "logo HTV9 xuất hiện ở góc màn hình",
    "máy bay đang cất cánh",
    "người đàn ông nói chuyện",
    "một người ở ngoài",
    "trước khi bước lên xe, người đàn ông đứng nói chuyện với phóng viên",
    "HTV9",
    "Bộ Y tế",
    "người nói chuyện trước màn hình có dòng chữ COVID-19",
    "mưa lớn tại thành phố",
    "người đàn ông",
    # 10 Realistic L21 Domain Queries (Visual, OCR, ASR, Object, Mixed)
    "chương trình tin tức thời sự truyền hình HTV",
    "cảnh sát giao thông đang điều tiết luồng xe",
    "cây cầu bắc qua sông lớn vào buổi chiều",
    "chữ cảnh báo nguy hiểm sạt lở đất",
    "bác sĩ giải thích triệu chứng bệnh nhân",
    "đoàn xe cứu thương đang di chuyển khẩn cấp",
    "thành phố Hồ Chí Minh nhìn từ trên cao",
    "người dân xếp hàng mua lương thực thực phẩm",
    "thông tin dự báo thời tiết nhiệt độ các tỉnh",
    "tàu thuyền neo đậu tại bến cảng",
]


def compute_overlap(list_a: List[str], list_b: List[str], k: int) -> float:
    """Compute set intersection overlap ratio at rank K."""
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    if not set_a or not set_b:
        return 1.0 if not set_a and not set_b else 0.0
    denom = min(len(set_a), len(set_b))
    if denom == 0:
        return 0.0
    return round(len(set_a.intersection(set_b)) / denom, 4)


def run_ab_comparison():
    print("=" * 90)
    print(" 🧭 AIC 2026 MULTIMODAL RETRIEVAL: STATIC VS DYNAMIC A/B COMPARISON")
    print("=" * 90)

    service = RetrievalService.get_instance()
    service.initialize()

    # Create outputs directory
    eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / "static_vs_dynamic_report.csv"
    html_path = eval_dir / "static_vs_dynamic_report.html"

    report_records = []
    detailed_comparisons = []

    print(f"\nEvaluating {len(BENCHMARK_QUERIES)} diverse queries at Top-100 retrieval...\n")

    for q_idx, query in enumerate(BENCHMARK_QUERIES, start=1):
        # 1. Execute Static Search (Baseline weights: 0.40, 0.25, 0.25, 0.10)
        res_static = service.search(
            query=query,
            top_k=100,
            fusion_mode="static",
            temporal_dedup=True,
        )

        # 2. Execute Dynamic Search (Query understanding + Router)
        res_dynamic = service.search(
            query=query,
            top_k=100,
            fusion_mode="dynamic",
            temporal_dedup=True,
        )

        # Extract Static properties
        static_kfs = [r["keyframe_name"] for r in res_static["results"]]
        static_vids = [r["video_id"] for r in res_static["results"]]
        static_scores = [r["score"] for r in res_static["results"]]
        static_ew = res_static.get("effective_weights", {"visual": 0.40, "ocr": 0.25, "asr": 0.25, "object": 0.10})
        static_timing = res_static.get("timing", {})
        static_lat = static_timing.get("total_ms", res_static.get("elapsed_ms", 0.0))

        # Extract Dynamic properties
        dynamic_kfs = [r["keyframe_name"] for r in res_dynamic["results"]]
        dynamic_vids = [r["video_id"] for r in res_dynamic["results"]]
        dynamic_scores = [r["score"] for r in res_dynamic["results"]]
        dynamic_ew = res_dynamic.get("effective_weights", {})
        dynamic_qa = res_dynamic.get("query_analysis") or {}
        dynamic_routing = res_dynamic.get("routing") or {}
        dynamic_timing = res_dynamic.get("timing", {})
        dynamic_lat = dynamic_timing.get("total_ms", res_dynamic.get("elapsed_ms", 0.0))
        qu_lat = dynamic_timing.get("query_understanding_ms", 0.0)
        route_lat = dynamic_timing.get("routing_ms", 0.0)

        intent = dynamic_qa.get("intent", "unknown")
        conf = dynamic_qa.get("confidence", 0.0)

        # Compute Overlap metrics at various cutoffs
        overlap_5 = compute_overlap(static_kfs, dynamic_kfs, 5)
        overlap_10 = compute_overlap(static_kfs, dynamic_kfs, 10)
        overlap_20 = compute_overlap(static_kfs, dynamic_kfs, 20)
        overlap_50 = compute_overlap(static_kfs, dynamic_kfs, 50)
        overlap_100 = compute_overlap(static_kfs, dynamic_kfs, 100)

        # Candidate rank changes in Top 10
        rank_shifts = 0
        for rank, kf in enumerate(dynamic_kfs[:10], start=1):
            if kf in static_kfs:
                st_rank = static_kfs.index(kf) + 1
                if st_rank != rank:
                    rank_shifts += 1
            else:
                rank_shifts += 1

        top1_static_vid = static_vids[0] if static_vids else ""
        top1_static_kf = static_kfs[0] if static_kfs else ""
        top1_dyn_vid = dynamic_vids[0] if dynamic_vids else ""
        top1_dyn_kf = dynamic_kfs[0] if dynamic_kfs else ""
        top1_same = (top1_static_vid == top1_dyn_vid and top1_static_kf == top1_dyn_kf)

        # Console summary per query
        print("-" * 90)
        print(f"[{q_idx:02d}/{len(BENCHMARK_QUERIES):02d}] Query: \"{query}\"")
        print(f"     Intent: {intent} (conf: {conf:.2f}) | Dominant: {dynamic_routing.get('dominant_branch', 'visual')}")
        print(f"     Static Weights : V={static_ew.get('visual', 0):.2f}, O={static_ew.get('ocr', 0):.2f}, A={static_ew.get('asr', 0):.2f}, Obj={static_ew.get('object', 0):.2f} (Lat: {static_lat:.1f}ms)")
        print(f"     Dynamic Weights: V={dynamic_ew.get('visual', 0):.2f}, O={dynamic_ew.get('ocr', 0):.2f}, A={dynamic_ew.get('asr', 0):.2f}, Obj={dynamic_ew.get('object', 0):.2f} (Lat: {dynamic_lat:.1f}ms, QU: {qu_lat:.2f}ms, Route: {route_lat:.3f}ms)")
        print(f"     Overlap@5: {overlap_5:.0%} | Overlap@10: {overlap_10:.0%} | Overlap@100: {overlap_100:.0%} | Top-1 Identical: {top1_same}")

        # Top 3 Candidates comparison snippet
        print("     Static Top 3 : " + ", ".join([f"{static_vids[i]}/{static_kfs[i]} ({static_scores[i]:.3f})" for i in range(min(3, len(static_kfs)))]))
        print("     Dynamic Top 3: " + ", ".join([f"{dynamic_vids[i]}/{dynamic_kfs[i]} ({dynamic_scores[i]:.3f})" for i in range(min(3, len(dynamic_kfs)))]))

        # Record for CSV report
        record = {
            "query_id": q_idx,
            "query": query,
            "intent": intent,
            "confidence": conf,
            "static_visual": static_ew.get("visual", 0.40),
            "static_ocr": static_ew.get("ocr", 0.25),
            "static_asr": static_ew.get("asr", 0.25),
            "static_object": static_ew.get("object", 0.10),
            "dynamic_visual": dynamic_ew.get("visual", 0.40),
            "dynamic_ocr": dynamic_ew.get("ocr", 0.25),
            "dynamic_asr": dynamic_ew.get("asr", 0.25),
            "dynamic_object": dynamic_ew.get("object", 0.10),
            "static_top1_video": top1_static_vid,
            "static_top1_frame": top1_static_kf,
            "dynamic_top1_video": top1_dyn_vid,
            "dynamic_top1_frame": top1_dyn_kf,
            "top1_match": top1_same,
            "overlap_at_5": overlap_5,
            "overlap_at_10": overlap_10,
            "overlap_at_20": overlap_20,
            "overlap_at_50": overlap_50,
            "overlap_at_100": overlap_100,
            "rank_shifts_top10": rank_shifts,
            "static_latency_ms": static_lat,
            "dynamic_latency_ms": dynamic_lat,
            "query_understanding_ms": qu_lat,
            "router_latency_ms": route_lat,
        }
        report_records.append(record)

        # Store detailed top 5 items for HTML report
        top5_static = res_static["results"][:5]
        top5_dynamic = res_dynamic["results"][:5]
        detailed_comparisons.append({
            "query_id": q_idx,
            "query": query,
            "intent": intent,
            "confidence": conf,
            "static_ew": static_ew,
            "dynamic_ew": dynamic_ew,
            "overlap_10": overlap_10,
            "overlap_100": overlap_100,
            "top5_static": top5_static,
            "top5_dynamic": top5_dynamic,
            "static_lat": static_lat,
            "dynamic_lat": dynamic_lat,
        })

    # -------------------------------------------------------------
    # Save CSV Report
    # -------------------------------------------------------------
    df_report = pd.DataFrame(report_records)
    df_report.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV Evaluation Report saved to: {csv_path}")

    # -------------------------------------------------------------
    # Generate Standalone HTML Report
    # -------------------------------------------------------------
    generate_html_report(df_report, detailed_comparisons, html_path)
    print(f"✅ Interactive HTML Report saved to: {html_path}")

    # -------------------------------------------------------------
    # Repeated Performance & Throughput Benchmark (50 Iterations)
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" ⚡ RUNNING REPEATED THROUGHPUT & LATENCY BENCHMARK (50 ITERATIONS)...")
    print("=" * 90)

    test_subset = BENCHMARK_QUERIES[:10]
    static_times = []
    dynamic_times = []
    qu_times = []
    route_times = []

    # Warmup
    for q in test_subset[:2]:
        _ = service.search(query=q, top_k=20, fusion_mode="static")
        _ = service.search(query=q, top_k=20, fusion_mode="dynamic")

    for i in range(50):
        q = test_subset[i % len(test_subset)]
        
        # Static Timing
        t0 = time.perf_counter()
        res_s = service.search(query=q, top_k=20, fusion_mode="static")
        t1 = time.perf_counter()
        static_times.append((t1 - t0) * 1000)

        # Dynamic Timing
        t2 = time.perf_counter()
        res_d = service.search(query=q, top_k=20, fusion_mode="dynamic")
        t3 = time.perf_counter()
        dynamic_times.append((t3 - t2) * 1000)

        t_info = res_d.get("timing", {})
        qu_times.append(t_info.get("query_understanding_ms", 0.0))
        route_times.append(t_info.get("routing_ms", 0.0))

    static_med = float(np.median(static_times))
    dynamic_med = float(np.median(dynamic_times))
    qu_med = float(np.median(qu_times))
    route_med = float(np.median(route_times))
    overhead_med = dynamic_med - static_med

    print(f"  • Static Search Median Latency    : {static_med:.2f} ms")
    print(f"  • Dynamic Search Median Latency   : {dynamic_med:.2f} ms")
    print(f"  • Parser Latency (Query Understanding): {qu_med:.3f} ms")
    print(f"  • Router Latency (Policy Blending)  : {route_med:.4f} ms")
    print(f"  • Net Dynamic Routing Overhead    : {overhead_med:+.2f} ms (dominated by retrieval runtime)")
    print("=" * 90)


def generate_html_report(df: pd.DataFrame, details: List[Dict[str, Any]], output_path: Path):
    """Generate a clean and responsive standalone HTML report for visual inspection."""
    html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIC 2026 - Static vs Dynamic Fusion A/B Report</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --primary: #38bdf8;
    --accent: #818cf8;
    --success: #34d399;
    --warning: #fbbf24;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
    line-height: 1.5;
  }
  .container { max-width: 1400px; margin: 0 auto; }
  h1 { font-size: 24px; color: var(--primary); margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 14px; margin-bottom: 24px; }
  
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 16px;
    border-radius: 8px;
  }
  .stat-val { font-size: 24px; font-weight: bold; color: var(--primary); }
  .stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; margin-top: 4px; }

  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 32px;
    font-size: 13px;
  }
  th, td {
    padding: 12px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  th { background: #0b1120; color: var(--muted); font-weight: 600; }
  tr:hover { background: rgba(255,255,255,0.03); }
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
  .tag-intent { background: #1e1b4b; color: #a5b4fc; border: 1px solid #4338ca; }
  .tag-ocr { background: #3b0764; color: #d8b4fe; }
  .tag-asr { background: #064e3b; color: #6ee7b7; }
  .tag-visual { background: #1e3a8a; color: #93c5fd; }
  
  .card-detail {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }
  .query-title { font-size: 16px; font-weight: 600; color: var(--primary); margin-bottom: 8px; }
  .candidate-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 12px;
  }
  .branch-box {
    background: #0f172a;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .branch-header { font-size: 13px; font-weight: 600; color: var(--accent); margin-bottom: 8px; }
  .cand-item {
    font-size: 12px;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    justify-content: space-between;
  }
</style>
</head>
<body>
<div class="container">
  <h1>🏛️ AI Challenge HCMC 2026: Static vs Dynamic Fusion A/B Report</h1>
  <div class="subtitle">Audited comparison across 25 queries evaluating Top-100 rank shifts, weight distributions, and retrieval latency.</div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-val">""" + str(len(df)) + """</div>
      <div class="stat-label">Total Test Queries</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">""" + f"{df['overlap_at_10'].mean():.1%}" + """</div>
      <div class="stat-label">Mean Overlap@10</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">""" + f"{df['overlap_at_100'].mean():.1%}" + """</div>
      <div class="stat-label">Mean Overlap@100</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">""" + f"{df['query_understanding_ms'].mean():.2f} ms" + """</div>
      <div class="stat-label">Avg Parser Latency</div>
    </div>
  </div>

  <h2>📊 Summary Table (25 Queries)</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Query</th>
        <th>Intent</th>
        <th>Conf</th>
        <th>Static (V/O/A/Obj)</th>
        <th>Dynamic (V/O/A/Obj)</th>
        <th>Overlap@10</th>
        <th>Overlap@100</th>
        <th>Static Lat</th>
        <th>Dynamic Lat</th>
      </tr>
    </thead>
    <tbody>
"""

    for _, row in df.iterrows():
        intent = str(row["intent"])
        html_content += f"""
      <tr>
        <td>{row['query_id']}</td>
        <td><strong>{row['query']}</strong></td>
        <td><span class="tag tag-intent">{intent}</span></td>
        <td>{row['confidence']:.2f}</td>
        <td>{row['static_visual']:.2f} / {row['static_ocr']:.2f} / {row['static_asr']:.2f} / {row['static_object']:.2f}</td>
        <td><strong>{row['dynamic_visual']:.2f} / {row['dynamic_ocr']:.2f} / {row['dynamic_asr']:.2f} / {row['dynamic_object']:.2f}</strong></td>
        <td>{row['overlap_at_5']:.0%}</td>
        <td>{row['overlap_at_100']:.0%}</td>
        <td>{row['static_latency_ms']:.1f}ms</td>
        <td>{row['dynamic_latency_ms']:.1f}ms</td>
      </tr>
"""

    html_content += """
    </tbody>
  </table>

  <h2>🔍 Detailed Candidate Ranking Comparison (Top 5)</h2>
"""

    for item in details:
        q_id = item["query_id"]
        q_text = item["query"]
        intent = item["intent"]
        conf = item["confidence"]
        s_ew = item["static_ew"]
        d_ew = item["dynamic_ew"]
        t5_s = item["top5_static"]
        t5_d = item["top5_dynamic"]

        html_content += f"""
  <div class="card-detail">
    <div class="query-title">#{q_id}: {q_text}</div>
    <div style="font-size: 13px; color: var(--muted); margin-bottom: 8px;">
      Intent: <span class="tag tag-intent">{intent}</span> (conf: {conf:.2f}) | 
      Static Weights: [V:{s_ew.get('visual',0):.2f}, O:{s_ew.get('ocr',0):.2f}, A:{s_ew.get('asr',0):.2f}, Obj:{s_ew.get('object',0):.2f}] &rarr; 
      Dynamic Weights: [V:{d_ew.get('visual',0):.2f}, O:{d_ew.get('ocr',0):.2f}, A:{d_ew.get('asr',0):.2f}, Obj:{d_ew.get('object',0):.2f}] |
      Overlap@10: {item['overlap_10']:.0%} | Overlap@100: {item['overlap_100']:.0%}
    </div>
    <div class="candidate-grid">
      <div class="branch-box">
        <div class="branch-header">STATIC BASELINE TOP-5 (Lat: {item['static_lat']:.1f}ms)</div>
"""
        for r in t5_s:
            html_content += f"""
        <div class="cand-item">
          <span>#{r['rank']} <strong>{r['video_id']}</strong> ({r['keyframe_name']} @ {r['timestamp_text']})</span>
          <span>Score: <strong>{r['score']:.4f}</strong> [V:{r['scores']['visual']:.2f}, O:{r['scores']['ocr']:.2f}, A:{r['scores']['asr']:.2f}, Obj:{r['scores']['object']:.2f}]</span>
        </div>
"""
        html_content += f"""
      </div>
      <div class="branch-box">
        <div class="branch-header">DYNAMIC ROUTED TOP-5 (Lat: {item['dynamic_lat']:.1f}ms)</div>
"""
        for r in t5_d:
            html_content += f"""
        <div class="cand-item">
          <span>#{r['rank']} <strong>{r['video_id']}</strong> ({r['keyframe_name']} @ {r['timestamp_text']})</span>
          <span>Score: <strong>{r['score']:.4f}</strong> [V:{r['scores']['visual']:.2f}, O:{r['scores']['ocr']:.2f}, A:{r['scores']['asr']:.2f}, Obj:{r['scores']['object']:.2f}]</span>
        </div>
"""
        html_content += """
      </div>
    </div>
  </div>
"""

    html_content += """
</div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    run_ab_comparison()
