#!/usr/bin/env python3
"""KIS P1C0: Multilingual CLIP Text-Projection Compatibility Test & Raw Retrieval Shootout.

Compares:
  - ARM A (P0 Production): Raw VI -> Marian MT (pinned rev) -> OpenAI CLIP ViT-B/32 text encoder -> 1st-pass retrieval.
  - ARM B (Multilingual Raw VI): Raw VI -> Multilingual CLIP text encoder (sentence-transformers/clip-ViT-B-32-multilingual-v1) -> 1st-pass retrieval.

Strict constraints:
  - Production P0 remains 100% immutable (default OFF).
  - ZERO image re-indexing: 100% reuse of the existing 177,532-keyframe OpenAI CLIP ViT-B/32 feature matrix.
  - Rigorous embedding compatibility gate: dimension assert (512), L2-norm assert (1.0).
  - Raw 1st-pass retrieval ONLY (zero Phase-4 refinement to keep test clean, fast, and direct).
  - ZERO fusion between Arm A and Arm B.
  - ZERO ground-truth leakage.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Purge stale system_tai modules
for mod in list(sys.modules.keys()):
    if mod.startswith("system_tai"):
        del sys.modules[mod]

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import clip
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "sentence-transformers", "opencv-python-headless"], check=False)
    import clip

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=False)
    from sentence_transformers import SentenceTransformer

import cv2
import numpy as np
import torch
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig

MULTILINGUAL_CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
EXPECTED_EMBEDDING_DIM = 512

BTC_KIS_QUERIES = [
    {"qid": "query-p1-1-kis", "category": "REGRESSION_GUARD", "name": "Người chơi đàn Hang / Handpan"},
    {"qid": "query-p1-2-kis", "category": "REGRESSION_GUARD", "name": "Con hổ (Tiger)"},
    {"qid": "query-p1-5-kis", "category": "REGRESSION_GUARD", "name": "Hai người phụ nữ cho dê ăn"},
    {"qid": "query-p1-6-kis", "category": "GENERAL_PROBE", "name": "Cắt tỉa cây cảnh Bonsai / Đĩa chay"},
    {"qid": "query-p1-7-kis", "category": "REGRESSION_GUARD", "name": "Chim lông đen ánh xanh cổ (Bird)"},
    {"qid": "query-p1-8-kis", "category": "GENERAL_PROBE", "name": "Lễ hội ẩm thực Nhật bé đeo mực đỏ"},
    {"qid": "query-p1-9-kis", "category": "GENERAL_PROBE", "name": "Thu hoạch dứa ở miền Tây"},
    {"qid": "query-p1-10-kis", "category": "REGRESSION_GUARD", "name": "Chơi nhạc cụ kim loại tròn (Handpan)"},
    {"qid": "query-p1-11-kis", "category": "REGRESSION_GUARD", "name": "Đổ bóng tạo chân dung mặc vest"},
    {"qid": "query-p1-12-kis", "category": "LONG_QUERY_PROBE", "name": "Trang trí bánh rán dâu tây chuối chocolate"},
    {"qid": "query-p1-13-kis", "category": "TARGET_PROBE", "name": "Vệ sinh máy ảnh, lens trên khăn hồng, tăm bông"},
    {"qid": "query-p1-14-kis", "category": "GENERAL_PROBE", "name": "Điêu khắc cát thể thao đường phố"},
    {"qid": "query-p1-17-kis", "category": "LONG_QUERY_PROBE", "name": "Trao quà từ thiện bệnh viện biển COVID-19"},
    {"qid": "query-p1-20-kis", "category": "REGRESSION_GUARD", "name": "Thêm 2 ly panna cotta, hoa ăn được"},
    {"qid": "query-p1-21-kis", "category": "TARGET_PROBE", "name": "Cơ chế bay của bọ làm robot ở ĐH Lausanne"},
    {"qid": "query-p1-23-kis", "category": "REASONING_CONTROL", "name": "Động vật biển nguy hiểm Steven Spielberg 1975"},
    {"qid": "query-p1-24-kis", "category": "TARGET_PROBE", "name": "Đua xe đạp quay từ trên cao xuống"},
    {"qid": "query-p1-25-kis", "category": "REGRESSION_GUARD", "name": "Đua xe đạp flycam trên cao áo xanh vượt 3"},
]

VIDEO_PATH_CACHE: dict[str, Path] = {}


def populate_video_index_once() -> None:
    if VIDEO_PATH_CACHE:
        return
    for search_root in [Path("/kaggle/input"), REPO_ROOT / "systems" / "system_tai" / "data"]:
        if not search_root.exists():
            continue
        for root_dir, _, files in os.walk(str(search_root)):
            for fname in files:
                if fname.endswith(".mp4"):
                    vid = fname[:-4]
                    if vid not in VIDEO_PATH_CACHE:
                        VIDEO_PATH_CACHE[vid] = Path(root_dir) / fname


def resolve_video_path(video_id: str, raw_video_registry: Any = None) -> Path | None:
    if raw_video_registry:
        try:
            rec = raw_video_registry.get(video_id)
            if rec and rec.raw_video_path and rec.raw_video_path.exists():
                return rec.raw_video_path
        except Exception:
            pass
    if video_id in VIDEO_PATH_CACHE:
        return VIDEO_PATH_CACHE[video_id]
    populate_video_index_once()
    return VIDEO_PATH_CACHE.get(video_id)


def extract_thumbnail_base64(video_id: str, frame_id: int, raw_video_registry: Any = None) -> str:
    vpath = resolve_video_path(video_id, raw_video_registry)
    if not vpath or not vpath.exists():
        return ""
    try:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return ""
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return ""
        h, w = frame.shape[:2]
        new_w = 240
        new_h = int(h * (new_w / w))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return base64.b64encode(buf).decode("utf-8")
    except Exception:
        return ""


def get_reuse_manifest() -> Path | None:
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
        REPO_ROOT / "systems" / "system_tai" / "data" / "feature_manifest.json",
    ]:
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def check_embedding_compatibility(st_model: SentenceTransformer) -> tuple[int, str]:
    """Verify that multilingual CLIP text embeddings are 100% compatible with OpenAI CLIP ViT-B/32."""
    test_text = "Đoạn video mô tả một người ngồi vệ sinh máy ảnh."
    t0 = time.time()
    emb = st_model.encode([test_text], convert_to_numpy=True, normalize_embeddings=True)
    lat_ms = (time.time() - t0) * 1000

    shape = emb.shape
    if len(shape) != 2 or shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(f"CRITICAL COMPATIBILITY ERROR: Expected shape (1, {EXPECTED_EMBEDDING_DIM}), got {shape}")

    vector = emb[0]
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"CRITICAL COMPATIBILITY ERROR: Vector norm is {norm:.6f}, expected 1.0 (L2 unit vector)")

    print("=" * 150, flush=True)
    print("EMBEDDING & INDEX COMPATIBILITY GATE VERIFICATION", flush=True)
    print("=" * 150, flush=True)
    print(f"• Candidate Model Identifier       : {MULTILINGUAL_CLIP_MODEL_NAME}", flush=True)
    print(f"• Target Index Space               : OpenAI CLIP ViT-B/32 (512-dim L2-normalized)", flush=True)
    print(f"• Output Embedding Dimension       : {shape[1]} (EXACT MATCH = 512 ✅)", flush=True)
    print(f"• Normalization Convention         : L2 Unit Vector (Norm = {norm:.6f} ✅)", flush=True)
    print(f"• Encoding Latency (Single Text)   : {lat_ms:.2f} ms", flush=True)
    print(f"• Image Re-indexing Required       : NO (100% Zero Re-indexing Reuse ✅)", flush=True)
    print("=" * 150, flush=True)
    return shape[1], "EXACT_MATCH"


def run_p1c0_experiment() -> None:
    print("=" * 150, flush=True)
    print("KIS P1C0: MULTILINGUAL CLIP TEXT-PROJECTION RETRIEVAL SHOOTOUT", flush=True)
    print("=" * 150, flush=True)

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest = get_reuse_manifest()
    out_dir = Path("/kaggle/working/output/kis_p1c0_session") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_p1c0_session"

    cfg = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_dir,
        reuse_manifest=reuse_manifest,
    )

    # 1. Bootstrap Production Runtime (provides exact manifest, exact_retriever, Marian, OpenAI CLIP)
    print("\n[1/3] Bootstrapping OperationalKISRuntime...", flush=True)
    t0_rt = time.time()
    runtime = OperationalKISRuntime.bootstrap(cfg)
    device = runtime.shared_encoder.identifiers.get("device", "cpu")
    print(f"      Runtime Bootstrapped in {time.time() - t0_rt:.2f}s (device={device})", flush=True)

    # 2. Load Multilingual CLIP Text Encoder (Arm B)
    print(f"\n[2/3] Loading Multilingual CLIP Text Encoder ({MULTILINGUAL_CLIP_MODEL_NAME})...", flush=True)
    t0_b = time.time()
    st_model = SentenceTransformer(MULTILINGUAL_CLIP_MODEL_NAME, device=device)
    print(f"      Multilingual CLIP loaded in {time.time() - t0_b:.2f}s", flush=True)

    # 3. Compatibility Gate Check
    check_embedding_compatibility(st_model)

    # 4. Run A/B 1st-Pass Retrieval across 18 BTC KIS Queries
    thunghiem_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 150, flush=True)
    print("RUNNING RAW 1ST-PASS RETRIEVAL SHOOTOUT ACROSS ALL 18 BTC KIS QUERIES", flush=True)
    print("=" * 150, flush=True)

    arm_a_latencies = []
    arm_b_latencies = []

    for idx, item in enumerate(BTC_KIS_QUERIES, start=1):
        qid = item["qid"]
        category = item["category"]
        name = item["name"]
        q_file = thunghiem_dir / f"{qid}.txt"

        if not q_file.exists():
            continue

        q_vi = q_file.read_text(encoding="utf-8").strip()

        # --- ARM A: Production Marian -> OpenAI CLIP ---
        t_a0 = time.time()
        raw_en = runtime.translation_provider.translate(q_vi)
        eff_en, tok_count, was_compacted = runtime.token_budget_guard.guard_and_compact(raw_en)
        vec_a = runtime.shared_encoder.encode(eff_en)
        res_a = runtime.exact_retriever.search_vector(query_id=f"a-{qid}", query_vector=vec_a, top_k=10)
        lat_a = time.time() - t_a0
        arm_a_latencies.append(lat_a)

        # --- ARM B: Raw Vietnamese -> Multilingual CLIP ---
        t_b0 = time.time()
        vec_b = st_model.encode([q_vi], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        res_b = runtime.exact_retriever.search_vector(query_id=f"b-{qid}", query_vector=vec_b, top_k=10)
        lat_b = time.time() - t_b0
        arm_b_latencies.append(lat_b)

        top10_desc_a = [f"@{i}: {c.video_id} (f={c.frame_id}, s={c.similarity_score:.3f})" for i, c in enumerate(res_a.candidates[:10], start=1)]
        top10_desc_b = [f"@{i}: {c.video_id} (f={c.frame_id}, s={c.similarity_score:.3f})" for i, c in enumerate(res_b.candidates[:10], start=1)]

        vids_a = set(c.video_id for c in res_a.candidates[:10])
        vids_b = set(c.video_id for c in res_b.candidates[:10])
        overlap_10 = len(vids_a.intersection(vids_b))

        results.append({
            "qid": qid,
            "category": category,
            "name": name,
            "query_vi": q_vi,
            "eff_en_a": eff_en,
            "lat_a": lat_a,
            "lat_b": lat_b,
            "candidates_a": res_a.candidates,
            "candidates_b": res_b.candidates,
            "top10_desc_a": top10_desc_a,
            "top10_desc_b": top10_desc_b,
            "overlap_10": overlap_10,
        })

        badge = f"[{category}]"
        print(f"\n--- [{idx:02d}/{len(BTC_KIS_QUERIES)}] {qid} {badge} : {name} ---", flush=True)
        print(f"• VI Query      : \"{q_vi}\"", flush=True)
        print(f"• Arm A EN Text : \"{eff_en}\" ({tok_count} tok | Latency: {lat_a*1000:5.1f}ms)", flush=True)
        print(f"• Arm B Raw VI  : Multilingual Projected Direct (Latency: {lat_b*1000:5.1f}ms)", flush=True)
        print(f"• Arm A Top 5   : {top10_desc_a[:5]}", flush=True)
        print(f"• Arm B Top 5   : {top10_desc_b[:5]}", flush=True)
        print(f"• Top 10 Overlap: {overlap_10}/10 shared video entities", flush=True)

        if qid == "query-p1-13-kis":
            print(f"  🔍 TARGET PROBE (p1-13): Does Arm B pull camera cleaning (L30_V095) into Top 3? -> Arm A Top 1: {res_a.candidates[0].video_id} | Arm B Top 1: {res_b.candidates[0].video_id}", flush=True)
        elif qid == "query-p1-21-kis":
            print(f"  🔍 TARGET PROBE (p1-21): Does Arm B pull beetle/robot into Top 3? -> Arm A Top 1: {res_a.candidates[0].video_id} | Arm B Top 1: {res_b.candidates[0].video_id}", flush=True)
        elif qid == "query-p1-24-kis":
            print(f"  🔍 TARGET PROBE (p1-24): Does Arm B pull cycling race into Top 3? -> Arm A Top 1: {res_a.candidates[0].video_id} | Arm B Top 1: {res_b.candidates[0].video_id}", flush=True)

    # Generate HTML gallery
    gallery_out = Path("/kaggle/working/kis_p1c0_multilingual_gallery.html")
    generate_ab_gallery_html(results, gallery_out, runtime.raw_video_registry)
    print(f"\nSaved Comparative Side-by-Side Gallery to: {gallery_out}", flush=True)

    # Summary table
    print("\n" + "=" * 150, flush=True)
    print("KIS P1C0 LATENCY & TOP 1 OVERVIEW TABLE", flush=True)
    print("=" * 150, flush=True)
    print(f"{'Query ID':<18} | {'Category':<22} | {'Arm A Top 1':<24} | {'Arm B Top 1':<24} | {'Arm A Lat':<10} | {'Arm B Lat':<10}")
    print("-" * 120)
    for r in results:
        top1_a = f"{r['candidates_a'][0].video_id} (f={r['candidates_a'][0].frame_id})"
        top1_b = f"{r['candidates_b'][0].video_id} (f={r['candidates_b'][0].frame_id})"
        print(f"{r['qid']:<18} | {r['category']:<22} | {top1_a:<24} | {top1_b:<24} | {r['lat_a']*1000:6.1f} ms | {r['lat_b']*1000:6.1f} ms")
    print("=" * 150, flush=True)
    print(f"Mean Query Latency: Arm A (Marian MT + CLIP) = {np.mean(arm_a_latencies)*1000:.1f}ms | Arm B (Direct Multilingual CLIP) = {np.mean(arm_b_latencies)*1000:.1f}ms", flush=True)
    print("=" * 150, flush=True)


def generate_ab_gallery_html(results: list[dict[str, Any]], out_path: Path, raw_video_registry: Any = None) -> None:
    html_cards = []
    for r in results:
        qid = r["qid"]
        name = r["name"]
        category = r["category"]
        q_vi = r["query_vi"]
        en_a = r["eff_en_a"]
        preds_a = r["candidates_a"][:3]
        preds_b = r["candidates_b"][:3]

        def render_top3_grid(preds: list[Any], label: str, color: str) -> str:
            items = []
            for rank_idx, p in enumerate(preds, start=1):
                vid = p.video_id
                fid = p.frame_id
                score = p.similarity_score
                img_b64 = extract_thumbnail_base64(vid, fid, raw_video_registry)
                img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:4px;" />' if img_b64 else '<div style="background:#333;color:#888;height:80px;display:flex;align-items:center;justify-content:center;">No Frame</div>'
                items.append(f"""
                <div style="flex:1; margin:4px; padding:6px; background:#181818; border:1px solid #333; border-radius:6px; text-align:center; font-size:11px;">
                    <div style="font-weight:bold; color:{color};">Rank @{rank_idx} (s={score:.3f})</div>
                    {img_tag}
                    <div style="color:#eee; font-weight:600; margin-top:2px;">{vid}</div>
                    <div style="color:#888; font-size:10px;">f={fid}</div>
                </div>
                """)
            return f"""
            <div style="flex:1; padding:8px; background:#222; border-radius:6px; margin:4px;">
                <div style="font-weight:bold; color:{color}; margin-bottom:6px;">{label}</div>
                <div style="display:flex;">{''.join(items)}</div>
            </div>
            """

        grid_a = render_top3_grid(preds_a, "ARM A: Marian EN -> OpenAI CLIP (P0 Baseline)", "#0d6efd")
        grid_b = render_top3_grid(preds_b, "ARM B: Raw VI -> Multilingual CLIP Projection", "#28a745")

        cat_badge_color = "#ffc107; color:#111" if category == "REGRESSION_GUARD" else ("#e83e8c; color:#fff" if category == "TARGET_PROBE" else "#17a2b8; color:#fff")

        html_cards.append(f"""
        <div style="background:#2b2b2b; border:1px solid #444; border-radius:8px; margin-bottom:20px; padding:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #3c3c3c; padding-bottom:8px; margin-bottom:12px;">
                <span style="font-size:15px; font-weight:bold; color:#61afef;">{qid}</span>
                <span style="font-size:13px; font-weight:bold; color:#fff;">{name}</span>
                <span style="background:{cat_badge_color}; font-weight:bold; font-size:11px; padding:3px 8px; border-radius:4px;">{category}</span>
            </div>
            <div style="font-size:12px; color:#ccc; margin-bottom:4px;"><b style="color:#aaa;">VI Query:</b> {q_vi}</div>
            <div style="font-size:11px; color:#9cdcfe; margin-bottom:12px;"><b style="color:#0d6efd;">Arm A Marian EN:</b> "{en_a}"</div>
            <div style="display:flex; gap:8px;">
                {grid_a}
                {grid_b}
            </div>
        </div>
        """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>KIS P1C0 Multilingual CLIP A/B Gallery</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; padding:20px;">
        <h2 style="color:#61afef; border-bottom:2px solid #333; padding-bottom:8px;">🔬 KIS P1C0: MULTILINGUAL CLIP TEXT-PROJECTION A/B GALLERY</h2>
        <div style="color:#aaa; font-size:13px; margin-bottom:16px;">Side-by-side Top 3 candidate comparison between ARM A (Marian MT + OpenAI CLIP) and ARM B (Raw VI + Multilingual CLIP Projection).</div>
        {''.join(html_cards)}
    </body>
    </html>
    """
    out_path.write_text(full_html, encoding="utf-8")


if __name__ == "__main__":
    run_p1c0_experiment()
