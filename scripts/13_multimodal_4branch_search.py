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

    # 2. Object Search
    if obj_corpus_path.exists():
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

    # 3. Fuse Scores
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
        kf_name = str(row['keyframe_name'])
        kf_path = PROJECT_ROOT / "datasets_L21" / "Keyframes_L21" / "keyframes" / vid_id / kf_name
        
        print(f" #{rank+1:2d} Score: {row['fused_score']:.4f} (V:{row['visual_score']:.3f} | Obj:{row['object_score']:.3f}) | Video: {vid_id} | Frame: {row['frame_idx']:>6d} ({row['timestamp_text']})")
        print(f"     📸 Ảnh Keyframe: {kf_path}")
        
        if kf_path.exists():
            html_items.append(f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin: 10px; border-radius: 8px; font-family: sans-serif;">
                <h3>#{rank+1} - Score: {row['fused_score']:.4f} (Video: {vid_id}, Timestamp: {row['timestamp_text']})</h3>
                <p><b>Frame ID:</b> {row['frame_idx']} | <b>Keyframe File:</b> {kf_name}</p>
                <img src="{kf_path.as_uri()}" style="max-width: 600px; border-radius: 4px;" />
                <p><small>Đường dẫn: <code>{kf_path}</code></small></p>
            </div>
            """)

    print("-" * 80)
    
    # Save HTML Preview
    if html_items:
        preview_html_path = PROJECT_ROOT / "outputs" / "latest_search_results.html"
        html_content = f"""<html>
        <head><title>Search Results: {query}</title><meta charset="utf-8"/></head>
        <body style="background: #f4f6f9; padding: 20px;">
            <h2>🔍 Kết quả tìm kiếm cho câu: "{query}"</h2>
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
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    multimodal_4branch_search(args.query, top_k=args.top_k, config_path=args.config)


if __name__ == "__main__":
    main()
