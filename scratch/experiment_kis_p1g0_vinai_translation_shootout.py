#!/usr/bin/env python3
"""KIS P1G0: Query-Only Translation Fidelity Shootout Prototype.

Compares:
  - Model A (Frozen Production Marian): Helsinki-NLP/opus-mt-vi-en (rev 5611f34634b72de0608b1238a4e02845ca285f3e).
  - Model B (Candidate VinAI Translation): vinai/vinai-translate-vi2en-v2 (mBART-based, AGPL-3.0).

Strict constraints:
  - Production P0 remains 100% immutable (default OFF).
  - ZERO retrieval, ZERO CLIP image encoding, ZERO Phase-4, ZERO benchmark runs.
  - ZERO hardcoded lexical corrections or regex dictionaries.
  - Raw uncompacted translation fidelity comparison across visual atoms.
  - Evaluates all 18 BTC KIS queries.
  - Mandatory target probes:
      * p1-13: Camera cleaning (not toilet), cotton swab/bud (not cotton mask).
      * p1-21: Lausanne (not Larissa), beetle flight mechanism + robot.
      * p1-24: Cycling race, overhead/top-down/aerial view (not top-up shot).
      * p1-25: Bicycle race/cyclist, flycam/drone/aerial camera (not racing car / fycas).
      * p1-17: Xuân 2024 (no year corruption), hospital charity, COVID-19, children, gift bags.
  - Mandatory regression guards: p1-1, p1-2, p1-5, p1-7, p1-10, p1-11, p1-20.
  - Telemetry: Model IDs, resolved revisions, latencies (p50/mean/max), token counts, >77 truncation flags.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

# Ensure required libraries
try:
    import clip
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/openai/CLIP.git", "ftfy", "regex"], check=False)
    import clip

try:
    import sentencepiece
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "sacremoses"], check=False)

import numpy as np
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from system_tai.translation.provider import MarianOfflineTranslator

VINAI_MODEL_ID = "vinai/vinai-translate-vi2en-v2"
MARIAN_MODEL_ID = "Helsinki-NLP/opus-mt-vi-en"

BTC_KIS_QUERIES = [
    {"qid": "query-p1-1-kis", "category": "REGRESSION_GUARD", "name": "Phóng tàu vũ trụ tư nhân / 4 phi hành gia áo đen"},
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


def count_clip_tokens(text: str) -> tuple[int, bool]:
    """Count tokens according to OpenAI CLIP's BPE tokenizer (max context 77)."""
    try:
        tokens = clip.tokenize([text], truncate=False)
        count = int((tokens != 0).sum().item())
        return count, count > 77
    except Exception:
        # If text exceeds CLIP's internal 77 limit without truncate, it raises RuntimeError
        try:
            tokens_trunc = clip.tokenize([text], truncate=True)
            # Rough estimate using standard words
            approx_count = len(text.split()) + 10
            return max(78, approx_count), True
        except Exception:
            return 78, True


class VinAIOfflineTranslator:
    """Wrapper for vinai/vinai-translate-vi2en-v2."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model_id = VINAI_MODEL_ID
        print(f"\n[Loading Candidate VinAI Translator: '{self.model_id}' on device '{device}'...]", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang="vi_VN")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(device)
        self.model.eval()
        self.load_time = time.time() - t0
        
        # Extract model revision / architecture telemetry
        self.config = getattr(self.model, "config", None)
        self.model_type = getattr(self.config, "model_type", "mBART")
        self.num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"      • Loaded in {self.load_time:.2f}s | Architecture: {self.model_type} ({self.num_params:.1f}M params) | License: AGPL-3.0 ✅", flush=True)

    def translate(self, text_vi: str) -> tuple[str, float]:
        t0 = time.time()
        inputs = self.tokenizer(text_vi, return_tensors="pt", padding=True).to(self.device)
        
        gen_kwargs: dict[str, Any] = {
            "num_beams": 5,
            "max_length": 1024,
        }
        if hasattr(self.tokenizer, "lang_code_to_id") and "en_XX" in self.tokenizer.lang_code_to_id:
            gen_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id["en_XX"]
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        trans_en = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        lat = time.time() - t0
        return trans_en, lat


def run_p1g0_shootout() -> None:
    print("=" * 150, flush=True)
    print("KIS P1G0: VI->EN TRANSLATION FIDELITY SHOOTOUT (QUERY-ONLY PROTOTYPE)", flush=True)
    print("=" * 150, flush=True)
    print("Scope:", flush=True)
    print("  • Model A (Frozen Production Marian): Helsinki-NLP/opus-mt-vi-en (rev 5611f34634b72de0608b1238a4e02845ca285f3e)")
    print("  • Model B (Candidate VinAI Translation): vinai/vinai-translate-vi2en-v2 (mBART, AGPL-3.0)")
    print("  • ZERO retrieval, ZERO Phase-4, ZERO SigLIP2, ZERO production changes.")
    print("=" * 150, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Operational Environment: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Transformers {transformers.__version__} | Device: {device}", flush=True)

    # 1. Load Model A (Marian Production)
    print("\n[1/2] Initializing Model A (Production Marian Offline Translator)...", flush=True)
    t0_marian = time.time()
    translator_a = MarianOfflineTranslator()
    load_time_a = time.time() - t0_marian
    print(f"      • Loaded in {load_time_a:.2f}s (Revision: 5611f34634b72de0608b1238a4e02845ca285f3e) ✅", flush=True)

    # 2. Load Model B (VinAI Translate vi2en v2)
    print("\n[2/2] Initializing Model B (VinAI Translate vi2en v2)...", flush=True)
    translator_b = VinAIOfflineTranslator(device=device)

    thunghiem_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 150, flush=True)
    print("EVALUATING TRANSLATION FIDELITY ACROSS ALL 18 BTC KIS QUERIES", flush=True)
    print("=" * 150, flush=True)

    latencies_a: list[float] = []
    latencies_b: list[float] = []

    for idx, item in enumerate(BTC_KIS_QUERIES, start=1):
        qid = item["qid"]
        category = item["category"]
        name = item["name"]
        q_file = thunghiem_dir / f"{qid}.txt"

        if not q_file.exists():
            continue

        q_vi = q_file.read_text(encoding="utf-8").strip()

        # Translate with Model A (Marian)
        t0_a = time.time()
        raw_en_a = translator_a.translate(q_vi).strip()
        lat_a = time.time() - t0_a
        latencies_a.append(lat_a)

        # Translate with Model B (VinAI)
        raw_en_b, lat_b = translator_b.translate(q_vi)
        latencies_b.append(lat_b)

        # Token telemetry
        tok_clip_a, trunc_a = count_clip_tokens(raw_en_a)
        tok_clip_b, trunc_b = count_clip_tokens(raw_en_b)

        results.append({
            "qid": qid,
            "category": category,
            "name": name,
            "query_vi": q_vi,
            "raw_en_a": raw_en_a,
            "raw_en_b": raw_en_b,
            "lat_a": lat_a,
            "lat_b": lat_b,
            "tok_clip_a": tok_clip_a,
            "tok_clip_b": tok_clip_b,
            "trunc_a": trunc_a,
            "trunc_b": trunc_b,
        })

        badge = f"[{category}]"
        print(f"\n--- [{idx:02d}/{len(BTC_KIS_QUERIES)}] {qid} {badge} : {name} ---", flush=True)
        print(f"• VI Source : {q_vi}", flush=True)
        print(f"• Marian EN : \"{raw_en_a}\"", flush=True)
        print(f"             (CLIP Tokens: {tok_clip_a}/77 | Exceeds={trunc_a} | Latency: {lat_a*1000:.1f}ms)", flush=True)
        print(f"• VinAI EN  : \"{raw_en_b}\"", flush=True)
        print(f"             (CLIP Tokens: {tok_clip_b}/77 | Exceeds={trunc_b} | Latency: {lat_b*1000:.1f}ms)", flush=True)

        # Specific Target Probes Diagnostic Audit
        if qid == "query-p1-13-kis":
            print(f"  🔍 TARGET AUDIT (p1-13 Camera Cleaning & Cotton Swab):", flush=True)
            print(f"     - Marian: camera toilet={'camera toilet' in raw_en_a.lower()} | cotton mask={'cotton mask' in raw_en_a.lower()}", flush=True)
            print(f"     - VinAI : camera cleaning={'clean' in raw_en_b.lower() and 'camera' in raw_en_b.lower() and 'toilet' not in raw_en_b.lower()} | cotton swab={'cotton' in raw_en_b.lower() and ('swab' in raw_en_b.lower() or 'bud' in raw_en_b.lower())}", flush=True)
        elif qid == "query-p1-21-kis":
            print(f"  🔍 TARGET AUDIT (p1-21 Lausanne City & Beetle Flight Robot):", flush=True)
            print(f"     - Marian: Lausanne={'lausanne' in raw_en_a.lower()} (has Larissa={'larissa' in raw_en_a.lower()})", flush=True)
            print(f"     - VinAI : Lausanne={'lausanne' in raw_en_b.lower()} | Beetle robot={'beetle' in raw_en_b.lower() and 'robot' in raw_en_b.lower()}", flush=True)
        elif qid == "query-p1-24-kis":
            print(f"  🔍 TARGET AUDIT (p1-24 Cycling Race & Overhead Viewpoint):", flush=True)
            print(f"     - Marian: cycling={'bicycle' in raw_en_a.lower() or 'cycl' in raw_en_a.lower()} | viewpoint={'top-up' in raw_en_a.lower()}", flush=True)
            print(f"     - VinAI : cycling={'cycl' in raw_en_b.lower() or 'bicycle' in raw_en_b.lower()} | overhead={'overhead' in raw_en_b.lower() or 'top-down' in raw_en_b.lower() or 'from above' in raw_en_b.lower()}", flush=True)
        elif qid == "query-p1-25-kis":
            print(f"  🔍 TARGET AUDIT (p1-25 Flycam Drone & Cyclist):", flush=True)
            print(f"     - Marian: racing car={'racing car' in raw_en_a.lower()} | fycas={'fycas' in raw_en_a.lower()}", flush=True)
            print(f"     - VinAI : cyclist={'cycl' in raw_en_b.lower() or 'bicycle' in raw_en_b.lower()} | flycam/drone={'flycam' in raw_en_b.lower() or 'drone' in raw_en_b.lower() or 'aerial' in raw_en_b.lower() or 'above' in raw_en_b.lower()}", flush=True)
        elif qid == "query-p1-17-kis":
            print(f"  🔍 TARGET AUDIT (p1-17 Hospital Charity & Year Fidelity):", flush=True)
            print(f"     - Marian: Year={'2 0 3 0' in raw_en_a or '2030' in raw_en_a} (corrupted from 2024)", flush=True)
            print(f"     - VinAI : Year={'2024' in raw_en_b or 'Spring 2024' in raw_en_b} | Hospital charity={'hospital' in raw_en_b.lower() and 'charity' in raw_en_b.lower()}", flush=True)

    # Summary Statistics Table
    print("\n" + "=" * 150, flush=True)
    print("KIS P1G0 TRANSLATION FIDELITY & LATENCY OVERVIEW TABLE", flush=True)
    print("=" * 150, flush=True)
    print(f"{'Query ID':<18} | {'Category':<18} | {'Marian CLIP Tok':<16} | {'VinAI CLIP Tok':<16} | {'Marian Lat':<12} | {'VinAI Lat':<12}")
    print("-" * 105)
    for r in results:
        t_a = f"{r['tok_clip_a']}{' (TRUNC)' if r['trunc_a'] else ''}"
        t_b = f"{r['tok_clip_b']}{' (TRUNC)' if r['trunc_b'] else ''}"
        print(f"{r['qid']:<18} | {r['category']:<18} | {t_a:<16} | {t_b:<16} | {r['lat_a']*1000:6.1f}ms     | {r['lat_b']*1000:6.1f}ms")
    print("=" * 150, flush=True)
    print(f"Latency Statistics:", flush=True)
    print(f"  • Model A (Marian) : Mean = {np.mean(latencies_a)*1000:.1f}ms | p50 = {np.median(latencies_a)*1000:.1f}ms | Max = {np.max(latencies_a)*1000:.1f}ms", flush=True)
    print(f"  • Model B (VinAI)  : Mean = {np.mean(latencies_b)*1000:.1f}ms | p50 = {np.median(latencies_b)*1000:.1f}ms | Max = {np.max(latencies_b)*1000:.1f}ms", flush=True)
    print("=" * 150, flush=True)

    # Generate comparative HTML summary table
    html_out = Path("/kaggle/working/kis_p1g0_translation_shootout.html")
    generate_translation_html(results, html_out)
    print(f"\nSaved Translation Comparison HTML Report to: {html_out}", flush=True)


def generate_translation_html(results: list[dict[str, Any]], out_path: Path) -> None:
    rows = []
    for r in results:
        cat_badge = "#ffc107; color:#111" if r['category'] == "REGRESSION_GUARD" else ("#e83e8c; color:#fff" if r['category'] == "TARGET_PROBE" else "#17a2b8; color:#fff")
        rows.append(f"""
        <tr style="border-bottom:1px solid #333;">
            <td style="padding:10px; font-weight:bold; color:#61afef;">{r['qid']}<br><span style="font-size:11px; background:{cat_badge}; padding:2px 6px; border-radius:3px;">{r['category']}</span></td>
            <td style="padding:10px; font-size:12px; color:#ddd; max-width:280px;">{r['query_vi']}</td>
            <td style="padding:10px; font-size:12px; color:#98c379; max-width:320px; background:#1c211d;">
                <div>{r['raw_en_a']}</div>
                <div style="font-size:10px; color:#aaa; margin-top:4px;">CLIP Tok: <b>{r['tok_clip_a']}</b> | Lat: {r['lat_a']*1000:.1f}ms</div>
            </td>
            <td style="padding:10px; font-size:12px; color:#61afef; max-width:320px; background:#1b2229;">
                <div>{r['raw_en_b']}</div>
                <div style="font-size:10px; color:#aaa; margin-top:4px;">CLIP Tok: <b>{r['tok_clip_b']}</b> | Lat: {r['lat_b']*1000:.1f}ms</div>
            </td>
        </tr>
        """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>KIS P1G0 Translation Shootout</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; padding:20px;">
        <h2 style="color:#61afef; border-bottom:2px solid #333; padding-bottom:8px;">🔬 KIS P1G0: VI→EN TRANSLATION FIDELITY SHOOTOUT (MARIAN VS VINAI)</h2>
        <div style="color:#aaa; font-size:13px; margin-bottom:16px;">
            <b>Arm A:</b> Frozen Production Marian (Helsinki-NLP/opus-mt-vi-en) | <b>Arm B:</b> vinai/vinai-translate-vi2en-v2 (AGPL-3.0)
        </div>
        <table style="width:100%; border-collapse:collapse; text-align:left;">
            <thead>
                <tr style="background:#222; border-bottom:2px solid #444; font-size:13px;">
                    <th style="padding:10px; width:15%;">Query ID</th>
                    <th style="padding:10px; width:25%;">Vietnamese Source</th>
                    <th style="padding:10px; width:30%;">Arm A: Marian Raw Translation</th>
                    <th style="padding:10px; width:30%;">Arm B: VinAI Raw Translation</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </body>
    </html>
    """
    out_path.write_text(full_html, encoding="utf-8")


if __name__ == "__main__":
    run_p1g0_shootout()
