"""
Script 13: Complete 4-Branch Multimodal Search Engine
=====================================================
Fuses results from Visual Index (1A), OCR Index (1B), ASR Index (2), and Object Index (3).
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.retrieval.clip_text_encoder import ClipTextEncoder
from src.retrieval.logging_utils import setup_logger
from src.retrieval.object_index import ObjectIndex
import faiss
import numpy as np


import yaml

def load_fusion_config(config_path: str | Path = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "multimodal_fusion.yaml"
    config_path = Path(config_path)
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as exc:
            print(f"Warning: Failed to load YAML config {config_path}: {exc}")
    return {}


def multimodal_4branch_search(
    query: str,
    top_k: int = None,
    w_visual: float = None,
    w_ocr: float = None,
    w_asr: float = None,
    w_object: float = None,
    config_path: str | Path = None,
):
    cfg = load_fusion_config(config_path)
    weights_cfg = cfg.get("fusion_weights", {})
    retrieval_cfg = cfg.get("retrieval_params", {})
    index_cfg = cfg.get("index_paths", {})

    w_visual = w_visual if w_visual is not None else float(weights_cfg.get("visual", 0.50))
    w_ocr = w_ocr if w_ocr is not None else float(weights_cfg.get("ocr", 0.20))
    w_asr = w_asr if w_asr is not None else float(weights_cfg.get("asr", 0.15))
    w_object = w_object if w_object is not None else float(weights_cfg.get("object", 0.15))
    top_k = top_k if top_k is not None else int(retrieval_cfg.get("default_top_k", 10))

    visual_index_path = PROJECT_ROOT / index_cfg.get("visual_index", "outputs/indexes/visual/l21_visual_flat_ip.faiss")
    global_map_path = PROJECT_ROOT / index_cfg.get("global_id_map", "outputs/indexes/l21_global_id_map.parquet")
    ocr_index_path = PROJECT_ROOT / index_cfg.get("ocr_index", "outputs/indexes/ocr/l21_ocr_flat_ip.faiss")
    ocr_corpus_path = PROJECT_ROOT / index_cfg.get("ocr_corpus", "outputs/indexes/ocr/l21_ocr_corpus.parquet")
    asr_index_path = PROJECT_ROOT / index_cfg.get("asr_index", "outputs/indexes/asr/l21_asr_flat_ip.faiss")
    asr_corpus_path = PROJECT_ROOT / index_cfg.get("asr_corpus", "outputs/indexes/asr/l21_asr_corpus.parquet")
    obj_corpus_path = PROJECT_ROOT / index_cfg.get("object_corpus", "outputs/indexes/object/l21_objects.parquet")

    df_global = pd.read_parquet(global_map_path)
    df_global = df_global.copy()
    df_global["fused_score"] = 0.0
    df_global["visual_score"] = 0.0
    df_global["ocr_score"] = 0.0
    df_global["asr_score"] = 0.0
    df_global["object_score"] = 0.0

    print("=" * 70)
    print(f" 🚀 FULL 4-BRANCH MULTIMODAL SEARCH")
    print(f" Query: \"{query}\"")
    print(f" Weights: Visual={w_visual} | OCR={w_ocr} | ASR={w_asr} | Object={w_object}")
    print("=" * 70)

    # 1. Visual Search (Auto-translate query to English for CLIP model)
    query_en = query
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="en").translate(query)
        if translated:
            query_en = translated
            print(f" 🌐 Translated Query for CLIP Visual: '{query_en}'")
    except Exception:
        pass

    if visual_index_path.exists():
        index_v = faiss.read_index(str(visual_index_path))
        encoder = ClipTextEncoder()
        q_emb = encoder.encode(query_en).astype(np.float32)
        scores_v, indices_v = index_v.search(q_emb, 200)

        for idx, score in zip(indices_v[0], scores_v[0]):
            df_global.at[idx, "visual_score"] = float(score)

    # 2. OCR V3 Hybrid Search (BM25 + Semantic E5)
    matched_ocr_snippets: dict[int, str] = {}
    if ocr_corpus_path.exists() and ocr_index_path.exists():
        try:
            from src.retrieval.ocr_v3_index import OCRV3HybridRetriever
            df_ocr_corpus = pd.read_parquet(ocr_corpus_path)
            ocr_retriever = OCRV3HybridRetriever(segments_df=df_ocr_corpus, faiss_index_path=ocr_index_path)
            ocr_hits = ocr_retriever.search(query, top_k=200, alpha=0.5)
            
            for hit in ocr_hits:
                v_id = hit.get("video_id", "")
                t_sec = float(hit.get("start_time", 0.0))
                score = float(hit.get("score", 0.0))
                ocr_text = str(hit.get("text_search", ""))
                
                # Match to global frame map based on video_id and timestamp window (+- 3.0s)
                time_mask = (df_global["video_id"] == v_id) & (abs(df_global["timestamp_seconds"] - t_sec) <= 3.0)
                if time_mask.any():
                    matched_indices = df_global[time_mask].index
                    for m_idx in matched_indices:
                        if score > df_global.at[m_idx, "ocr_score"]:
                            df_global.at[m_idx, "ocr_score"] = score
                            matched_ocr_snippets[m_idx] = ocr_text
        except Exception as ocr_err:
            print(f"⚠️ OCR V3 search warning: {ocr_err}")

    # 3. ASR V3 Hybrid Search (Speech Chunks normalized by Qwen3-4B)
    matched_asr_snippets: dict[int, str] = {}
    if asr_corpus_path.exists() and asr_index_path.exists():
        try:
            from src.retrieval.asr_v3_index import ASRV3HybridRetriever
            df_asr_corpus = pd.read_parquet(asr_corpus_path)
            asr_retriever = ASRV3HybridRetriever(chunks_df=df_asr_corpus, faiss_index_path=asr_index_path)
            asr_hits = asr_retriever.search(query, top_k=100, alpha=0.5)

            for hit in asr_hits:
                v_id = str(hit.get("video_id", ""))
                start_sec = float(hit.get("start", 0.0))
                end_sec = float(hit.get("end", 0.0))
                score = float(hit.get("score", 0.0))
                speech_text = str(hit.get("text_normalized", hit.get("text_raw", "")))

                # Match to keyframes belonging to this video within [start - 3.5s, end + 3.5s]
                time_mask = (df_global["video_id"] == v_id) & (
                    (df_global["timestamp_seconds"] >= start_sec - 3.5) & 
                    (df_global["timestamp_seconds"] <= end_sec + 3.5)
                )
                if time_mask.any():
                    matched_indices = df_global[time_mask].index
                    for m_idx in matched_indices:
                        if score > df_global.at[m_idx, "asr_score"]:
                            df_global.at[m_idx, "asr_score"] = score
                            matched_asr_snippets[m_idx] = f"[{start_sec:.1f}s - {end_sec:.1f}s]: {speech_text}"
        except Exception as asr_err:
            print(f"⚠️ ASR V3 search warning: {asr_err}")

    # 4. Object Search
    if obj_corpus_path.exists():
        try:
            obj_idx = ObjectIndex(obj_corpus_path)
            res_obj = obj_idx.search(query, top_k=200)
            if not res_obj.empty:
                max_s = res_obj["object_match_score"].max() or 1.0
                for _, row in res_obj.iterrows():
                    v_id, f_idx = row["video_id"], row["frame_idx"]
                    match_mask = (df_global["video_id"] == v_id) & (df_global["frame_idx"] == f_idx)
                    if match_mask.any():
                        idx = df_global[match_mask].index[0]
                        norm_score = float(row["object_match_score"]) / max_s
                        df_global.at[idx, "object_score"] = norm_score
        except Exception as obj_err:
            print(f"⚠️ Object search warning: {obj_err}")

    # 5. Fuse Scores (Late Fusion)
    df_global["fused_score"] = (
        w_visual * df_global["visual_score"]
        + w_ocr * df_global["ocr_score"]
        + w_asr * df_global["asr_score"]
        + w_object * df_global["object_score"]
    )

    df_sorted = df_global.sort_values(by="fused_score", ascending=False).head(top_k).reset_index(drop=True)

    print(f"\nTop {top_k} Multimodal Fusion Results:")
    print("-" * 80)
    
    html_items = []
    for rank, row in df_sorted.iterrows():
        vid_id = row['video_id']
        orig_idx = row.name
        kf_name = str(row['keyframe_name'])
        kf_path = PROJECT_ROOT / "datasets_L21" / "Keyframes_L21" / "keyframes" / vid_id / kf_name
        
        ocr_snippet = matched_ocr_snippets.get(orig_idx, "")
        asr_snippet = matched_asr_snippets.get(orig_idx, "")

        print(f" #{rank+1:2d} Score: {row['fused_score']:.4f} [V:{row['visual_score']:.3f} | OCR:{row['ocr_score']:.3f} | ASR:{row['asr_score']:.3f} | Obj:{row['object_score']:.3f}]")
        print(f"     Video: {vid_id} | Frame: {row['frame_idx']:>6d} ({row['timestamp_text']}) | File: {kf_name}")
        if ocr_snippet:
            print(f"     📝 OCR Text: {ocr_snippet[:90]}")
        if asr_snippet:
            print(f"     🎙️ ASR Speech: {asr_snippet[:90]}...")
        print(f"     📸 Ảnh Keyframe: {kf_path}")
        
        if kf_path.exists():
            html_items.append(f"""
            <div style="border: 1px solid #ddd; background: white; padding: 15px; margin-bottom: 20px; border-radius: 8px; font-family: sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h3 style="margin-top: 0; color: #1a73e8;">#{rank+1} - Tổng Điểm: {row['fused_score']:.4f}</h3>
                <div style="margin-bottom: 10px; color: #555;">
                    <b>Chi tiết điểm:</b> Visual: <code>{row['visual_score']:.3f}</code> | OCR: <code>{row['ocr_score']:.3f}</code> | ASR: <code>{row['asr_score']:.3f}</code> | Object: <code>{row['object_score']:.3f}</code>
                </div>
                <p><b>Video:</b> <code>{vid_id}</code> | <b>Frame:</b> {row['frame_idx']} | <b>Timestamp:</b> <code>{row['timestamp_text']}</code></p>
                {f'<p style="background: #e8f0fe; padding: 8px; border-radius: 4px;"><b>📝 OCR Khớp:</b> {ocr_snippet}</p>' if ocr_snippet else ''}
                {f'<p style="background: #fef7e0; padding: 8px; border-radius: 4px;"><b>🎙️ Lời thoại ASR:</b> {asr_snippet}</p>' if asr_snippet else ''}
                <img src="{kf_path.as_uri()}" style="max-width: 650px; border-radius: 6px; border: 1px solid #eee;" />
                <p><small style="color: #888;">File: <code>{kf_path}</code></small></p>
            </div>
            """)

    print("-" * 80)
    
    # Save HTML Preview
    if html_items:
        preview_html_path = PROJECT_ROOT / "outputs" / "latest_search_results.html"
        html_content = f"""<html>
        <head>
            <title>Search Results: {query}</title>
            <meta charset="utf-8"/>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; padding: 30px; color: #333; }}
                h2 {{ color: #202124; }}
            </style>
        </head>
        <body>
            <h2>🔍 Kết quả tìm kiếm Đa Phương Thức cho câu: "{query}"</h2>
            <p><b>Cấu hình trọng số:</b> Visual: {w_visual} | OCR: {w_ocr} | ASR: {w_asr} | Object: {w_object}</p>
            {"".join(html_items)}
        </body>
        </html>"""
        preview_html_path.parent.mkdir(parents=True, exist_ok=True)
        preview_html_path.write_text(html_content, encoding="utf-8")
        print(f"\n🌐 Đã xuất trang xem ảnh kết quả HTML tại:\n    {preview_html_path}")

    return df_sorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="thuyền thuyền máy sông nước")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    multimodal_4branch_search(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
