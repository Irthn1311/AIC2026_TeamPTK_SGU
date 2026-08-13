"""
Visual HTML Search Preview for OCR V3
======================================
Performs Hybrid OCR Search and renders a modern, interactive HTML web visualizer
with keyframe thumbnails, timestamps, ROI badges, and highlighted text matches.

Usage:
    python scripts/visualize_ocr_search.py --query "sụt lún ở ĐBSCL"
    python scripts/visualize_ocr_search.py --query "cảnh sát giao thông" --top_k 12
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.retrieval.ocr_v3_index import OCRV3HybridRetriever


def generate_html_search_report(query: str = "TÌNH TRẠNG SỤT LÚN Ở ĐBSCL", top_k: int = 12, alpha: float = 0.5):
    out_dir = PROJECT_ROOT / "outputs" / "evaluation" / "ocr_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_file = out_dir / "search_preview.html"

    parquet_path = PROJECT_ROOT / "outputs" / "indexes" / "ocr_v3" / "l21_ocr_v3_corpus.parquet"
    faiss_path = PROJECT_ROOT / "outputs" / "indexes" / "ocr_v3" / "l21_ocr_v3_flat_ip.faiss"

    if not parquet_path.exists():
        print(f"❌ Không tìm thấy corpus OCR tại {parquet_path}")
        return

    print("=" * 70)
    print(f" 🌐 ĐANG TẠO BÁO CÁO HTML CHO TRUY VẤN: '{query}'")
    print("=" * 70)

    df_corpus = pd.read_parquet(parquet_path)
    retriever = OCRV3HybridRetriever(segments_df=df_corpus, faiss_index_path=faiss_path)

    results = retriever.search(query, top_k=top_k, alpha=alpha)

    # Build HTML Cards
    cards_html = []
    keyframes_base = PROJECT_ROOT / "datasets_L21" / "Keyframes_L21" / "keyframes"

    for rank, item in enumerate(results, start=1):
        vid = item.get("video_id", "N/A")
        text = str(item.get("text_consensus", "")).strip()
        reg_type = item.get("region_type", "other")
        score = item.get("score", 0.0)
        bm25_sc = item.get("bm25_score", 0.0)
        sem_sc = item.get("semantic_score", 0.0)
        t_start = item.get("start_time", 0.0)
        t_end = item.get("end_time", 0.0)

        # Region Badge Colors
        badge_colors = {
            "headline": "linear-gradient(135deg, #ff416c, #ff4b2b)",
            "ticker": "linear-gradient(135deg, #11998e, #38ef7d)",
            "scene_text": "linear-gradient(135deg, #2193b0, #6dd5ed)",
            "logo_channel": "linear-gradient(135deg, #8e2de2, #4a00e0)",
        }
        badge_bg = badge_colors.get(reg_type, "linear-gradient(135deg, #757f9a, #d7dde8)")

        vid_dir = keyframes_base / vid
        img_src = ""
        jsonl_path = PROJECT_ROOT / "outputs" / "ocr_full" / "per_video" / f"{vid}.jsonl"
        best_kf = ""
        if jsonl_path.exists():
            import json
            best_diff = 999999.0
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    t_diff = abs(float(rec.get("timestamp_seconds", 0.0)) - t_start)
                    if t_diff < best_diff:
                        best_diff = t_diff
                        best_kf = rec.get("keyframe_name", "")
                    if best_diff <= 0.5:
                        break

        if best_kf and (vid_dir / best_kf).exists():
            img_src = f"../../../datasets_L21/Keyframes_L21/keyframes/{vid}/{best_kf}"
        elif vid_dir.exists():
            img_files = sorted(list(vid_dir.glob("*.jpg")))
            if img_files:
                img_src = f"../../../datasets_L21/Keyframes_L21/keyframes/{vid}/{img_files[0].name}"

        card = f"""
        <div class="result-card">
            <div class="card-header">
                <span class="rank-badge">#{rank}</span>
                <span class="video-id">{vid}</span>
                <span class="type-badge" style="background: {badge_bg};">{reg_type.upper()}</span>
                <span class="score-badge">Điểm: {score:.4f}</span>
            </div>
            <div class="card-body">
                <div class="img-container">
                    <img src="{img_src}" alt="Keyframe {vid}" onerror="this.src='https://via.placeholder.com/320x180?text=Keyframe+Image';">
                </div>
                <div class="card-details">
                    <div class="ocr-text">{text}</div>
                    <div class="meta-row">
                        <span>⏱️ Thời gian: <b>{t_start:.1f}s - {t_end:.1f}s</b></span>
                        <span>📊 BM25: <b>{bm25_sc:.3f}</b> | Semantic: <b>{sem_sc:.3f}</b></span>
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    cards_str = "\n".join(cards_html) if cards_html else "<div class='no-results'>Không tìm thấy kết quả phù hợp!</div>"

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kết Quả Tìm Kiếm OCR V3 - {query}</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #151c2e;
            --surface-hover: #1e2842;
            --primary: #3b82f6;
            --accent: #8b5cf6;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(255, 255, 255, 0.08);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Plus Jakarta Sans', sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 40px 20px; min-height: 100vh; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        
        .header {{ text-align: center; margin-bottom: 40px; }}
        .title {{ font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 12px; }}
        .query-box {{ display: inline-flex; align-items: center; gap: 10px; background: var(--surface); padding: 12px 24px; border-radius: 99px; border: 1px solid var(--border); font-size: 1.1rem; }}
        .query-text {{ color: #38bdf8; font-weight: 700; }}

        .grid {{ display: flex; flex-direction: column; gap: 20px; }}
        .result-card {{ background: var(--surface); border-radius: 16px; border: 1px solid var(--border); overflow: hidden; transition: transform 0.2s, border-color 0.2s; }}
        .result-card:hover {{ transform: translateY(-3px); border-color: rgba(59, 130, 246, 0.4); }}
        
        .card-header {{ display: flex; align-items: center; gap: 12px; padding: 14px 20px; background: rgba(0,0,0,0.2); border-bottom: 1px solid var(--border); }}
        .rank-badge {{ background: #2563eb; color: #fff; font-weight: 800; font-size: 0.85rem; padding: 4px 10px; border-radius: 8px; }}
        .video-id {{ font-weight: 700; font-size: 1.05rem; color: #e2e8f0; }}
        .type-badge {{ font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 99px; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.5); }}
        .score-badge {{ margin-left: auto; font-size: 0.9rem; font-weight: 700; color: #4ade80; background: rgba(74, 222, 128, 0.1); padding: 4px 12px; border-radius: 8px; }}

        .card-body {{ display: flex; gap: 20px; padding: 20px; }}
        .img-container {{ width: 280px; height: 160px; border-radius: 12px; overflow: hidden; background: #000; flex-shrink: 0; }}
        .img-container img {{ width: 100%; height: 100%; object-fit: cover; }}
        
        .card-details {{ display: flex; flex-direction: column; justify-content: space-between; flex-grow: 1; }}
        .ocr-text {{ font-size: 1.25rem; font-weight: 700; line-height: 1.5; color: #ffffff; background: rgba(255,255,255,0.03); padding: 14px 18px; border-radius: 12px; border-left: 4px solid #3b82f6; }}
        .meta-row {{ display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--text-muted); margin-top: 15px; border-top: 1px solid var(--border); padding-top: 12px; }}
        .meta-row b {{ color: #cbd5e1; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">AI Challenge 2026 - OCR V3 Visual Search</h1>
            <div class="query-box">
                <span>🔍 Từ khóa tìm kiếm:</span>
                <span class="query-text">"{query}"</span>
            </div>
        </div>
        <div class="grid">
            {cards_str}
        </div>
    </div>
</body>
</html>
"""
    html_file.write_text(html_content, encoding="utf-8")
    print(f"✅ Đã tạo giao diện HTML tại:\n{html_file.resolve()}")
    try:
        webbrowser.open(str(html_file.resolve()))
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visual HTML OCR Search Report")
    parser.add_argument("--query", "-q", type=str, default="TÌNH TRẠNG SỤT LÚN Ở ĐBSCL", help="Query to search and visualize")
    parser.add_argument("--top_k", "-k", type=int, default=10, help="Number of items to display")
    parser.add_argument("--alpha", "-a", type=float, default=0.5, help="Hybrid weight")
    args = parser.parse_args()

    generate_html_search_report(query=args.query, top_k=args.top_k, alpha=args.alpha)
