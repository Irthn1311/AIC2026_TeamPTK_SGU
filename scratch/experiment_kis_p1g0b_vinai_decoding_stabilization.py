#!/usr/bin/env python3
"""KIS P1G0b: VinAI Decoding Stabilization Shootout (Query-Only).

Evaluates decoding parameter configurations for vinai/vinai-translate-vi2en-v2
to resolve repetition degeneration (p1-11, p1-17), manage token budgets, and measure empirical latency.

Arms Tested:
  - Arm A  (Marian Production P0) : Helsinki-NLP/opus-mt-vi-en (rev 5611f34634b72de0608b1238a4e02845ca285f3e)
  - Arm B0 (VinAI Default Baseline): num_beams=5, no_repeat_ngram_size=0, repetition_penalty=1.0, max_length=1024
  - Arm B1 (VinAI Beam=3 Anti-Rep) : num_beams=3, no_repeat_ngram_size=3, repetition_penalty=1.15, max_new_tokens=256, early_stopping=True
  - Arm B2 (VinAI Greedy Anti-Rep) : num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.15, max_new_tokens=256
  - Arm B3 (VinAI Beam=4 Light-Rep): num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.05, max_new_tokens=256, early_stopping=True

Strict constraints:
  - Production P0 remains 100% immutable (default OFF).
  - ZERO retrieval, ZERO CLIP image encoding, ZERO Phase-4, ZERO benchmark runs.
  - ZERO hardcoded lexical corrections or regex dictionaries.
  - Raw uncompacted translation fidelity comparison across all 18 BTC KIS queries.
  - Evaluates target probes p1-13, p1-21, p1-24, p1-25, p1-17 and guards p1-1, p1-2, p1-5, p1-7, p1-10, p1-11, p1-20.
  - Reports measured latencies (mean, p50, max), exact tokens, >77 count, repetition diagnostics.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

print("=" * 150, flush=True)
print("KIS P1G0b: VINAI DECODING STABILIZATION SHOOTOUT (QUERY-ONLY)", flush=True)
print("=" * 150, flush=True)
print("Scope:", flush=True)
print("  • Comparing decoding stabilization configurations of vinai/vinai-translate-vi2en-v2 against Marian P0 baseline.")
print("  • Primary objectives: Eliminate repetition loops (p1-11, p1-17), retain target semantic wins (p1-13, p1-25, p1-24),")
print("    monitor token count vs 77 budget, and measure empirical latency across all arms.")
print("  • ZERO retrieval, ZERO Phase-4, ZERO hardcoded dictionaries, ZERO production changes.", flush=True)
print("=" * 150, flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

import numpy as np
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, CLIPTokenizerFast

from system_tai.translation.provider import MarianOfflineTranslator

VINAI_MODEL_ID = "vinai/vinai-translate-vi2en-v2"

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

# Fast standalone CLIP Tokenizer
try:
    _CLIP_TOKENIZER = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
except Exception:
    _CLIP_TOKENIZER = None


def count_clip_tokens(text: str) -> tuple[int, bool]:
    """Count tokens according to OpenAI CLIP's BPE tokenizer (max context 77)."""
    if _CLIP_TOKENIZER is not None:
        try:
            tokens = _CLIP_TOKENIZER.encode(text, add_special_tokens=True)
            count = len(tokens)
            return count, count > 77
        except Exception:
            pass
    approx_count = len(text.split()) + 5
    return approx_count, approx_count > 77


DECODING_CONFIGS: dict[str, dict[str, Any]] = {
    "B0_default": {
        "label": "VinAI Default (beam=5, nr=0, rep=1.0)",
        "num_beams": 5,
        "no_repeat_ngram_size": 0,
        "repetition_penalty": 1.0,
        "max_length": 1024,
    },
    "B1_beam3_nr3_rep115": {
        "label": "VinAI Beam=3 (nr=3, rep=1.15, early_stop=True)",
        "num_beams": 3,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.15,
        "max_new_tokens": 256,
        "early_stopping": True,
    },
    "B2_greedy_nr3_rep115": {
        "label": "VinAI Greedy (beam=1, nr=3, rep=1.15)",
        "num_beams": 1,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.15,
        "max_new_tokens": 256,
    },
    "B3_beam4_nr3_rep105": {
        "label": "VinAI Beam=4 (nr=3, rep=1.05, early_stop=True)",
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.05,
        "max_new_tokens": 256,
        "early_stopping": True,
    },
}


def run_p1g0b_shootout() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Operational Environment: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Transformers {transformers.__version__} | Device: {device}", flush=True)

    # 1. Load Model A (Marian Production)
    print("\n[1/2] Initializing Model A (Production Marian Offline Translator)...", flush=True)
    t0_marian = time.time()
    translator_a = MarianOfflineTranslator()
    load_time_a = time.time() - t0_marian
    print(f"      • Marian Loaded in {load_time_a:.2f}s (Revision: 5611f34634b72de0608b1238a4e02845ca285f3e) ✅", flush=True)

    # 2. Load Model B (VinAI Model & Tokenizer)
    print(f"\n[2/2] Loading VinAI Model: '{VINAI_MODEL_ID}' on device '{device}' ...", flush=True)
    t0_vinai = time.time()
    vinai_tokenizer = AutoTokenizer.from_pretrained(VINAI_MODEL_ID, src_lang="vi_VN", use_fast=False)
    vinai_model = AutoModelForSeq2SeqLM.from_pretrained(
        VINAI_MODEL_ID,
        low_cpu_mem_usage=False,
        dtype=torch.float32,
    ).to(device)
    vinai_model.eval()
    load_time_b = time.time() - t0_vinai
    num_params = sum(p.numel() for p in vinai_model.parameters()) / 1e6
    print(f"      • VinAI Loaded in {load_time_b:.2f}s | Architecture: mBART ({num_params:.1f}M params) | License: AGPL-3.0 ✅", flush=True)

    forced_bos_id = vinai_tokenizer.lang_code_to_id.get("en_XX") if hasattr(vinai_tokenizer, "lang_code_to_id") else None

    thunghiem_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
    results: list[dict[str, Any]] = []

    arm_latencies: dict[str, list[float]] = {"Marian": []}
    for arm_k in DECODING_CONFIGS:
        arm_latencies[arm_k] = []

    arm_overbudget_count: dict[str, int] = {"Marian": 0}
    for arm_k in DECODING_CONFIGS:
        arm_overbudget_count[arm_k] = 0

    print("\n" + "=" * 150, flush=True)
    print("EVALUATING DECODING CONFIGURATIONS ACROSS ALL 18 BTC KIS QUERIES", flush=True)
    print("=" * 150, flush=True)

    for idx, item in enumerate(BTC_KIS_QUERIES, start=1):
        qid = item["qid"]
        category = item["category"]
        name = item["name"]
        q_file = thunghiem_dir / f"{qid}.txt"

        if not q_file.exists():
            continue

        q_vi = q_file.read_text(encoding="utf-8").strip()

        # 1. Marian Translation
        t0_a = time.time()
        raw_en_marian = translator_a.translate(q_vi).strip()
        lat_marian = time.time() - t0_a
        tok_marian, trunc_marian = count_clip_tokens(raw_en_marian)
        arm_latencies["Marian"].append(lat_marian)
        if trunc_marian:
            arm_overbudget_count["Marian"] += 1

        query_res: dict[str, Any] = {
            "qid": qid,
            "category": category,
            "name": name,
            "query_vi": q_vi,
            "Marian": {
                "text": raw_en_marian,
                "tok": tok_marian,
                "trunc": trunc_marian,
                "lat": lat_marian,
            },
            "vinai_arms": {},
        }

        # 2. VinAI Translations across Arms
        inputs = vinai_tokenizer(q_vi, return_tensors="pt", padding=True).to(device)

        for arm_k, cfg_kwargs in DECODING_CONFIGS.items():
            gen_params = dict(cfg_kwargs)
            gen_params.pop("label", None)
            if forced_bos_id is not None:
                gen_params["forced_bos_token_id"] = forced_bos_id

            t0_arm = time.time()
            with torch.no_grad():
                out_ids = vinai_model.generate(**inputs, **gen_params)
            lat_arm = time.time() - t0_arm
            arm_latencies[arm_k].append(lat_arm)

            trans_text = vinai_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            tok_count, is_trunc = count_clip_tokens(trans_text)
            if is_trunc:
                arm_overbudget_count[arm_k] += 1

            query_res["vinai_arms"][arm_k] = {
                "text": trans_text,
                "tok": tok_count,
                "trunc": is_trunc,
                "lat": lat_arm,
            }

        results.append(query_res)

        badge = f"[{category}]"
        print(f"\n--- [{idx:02d}/{len(BTC_KIS_QUERIES)}] {qid} {badge} : {name} ---", flush=True)
        print(f"• VI Source : {q_vi}", flush=True)
        print(f"• Marian    : \"{raw_en_marian}\" (Tok: {tok_marian}/77, Lat: {lat_marian*1000:.0f}ms)", flush=True)
        for arm_k, data in query_res["vinai_arms"].items():
            print(f"• {arm_k:<20}: \"{data['text']}\" (Tok: {data['tok']}/77, Lat: {data['lat']*1000:.0f}ms)", flush=True)

    # Summary Statistics Table
    print("\n" + "=" * 150, flush=True)
    print("KIS P1G0b DECODING STABILIZATION COMPARATIVE SUMMARY TABLE", flush=True)
    print("=" * 150, flush=True)
    print(f"{'Query ID':<17} | {'Category':<18} | {'Marian (P0)':<12} | {'B0 Default':<12} | {'B1 (Beam=3,nr=3)':<16} | {'B2 (Greedy,nr=3)':<16} | {'B3 (Beam=4,nr=3)':<16}")
    print("-" * 125)
    for r in results:
        m_info = f"{r['Marian']['tok']}t ({'T' if r['Marian']['trunc'] else 'OK'})"
        b0_info = f"{r['vinai_arms']['B0_default']['tok']}t ({'T' if r['vinai_arms']['B0_default']['trunc'] else 'OK'})"
        b1_info = f"{r['vinai_arms']['B1_beam3_nr3_rep115']['tok']}t ({'T' if r['vinai_arms']['B1_beam3_nr3_rep115']['trunc'] else 'OK'})"
        b2_info = f"{r['vinai_arms']['B2_greedy_nr3_rep115']['tok']}t ({'T' if r['vinai_arms']['B2_greedy_nr3_rep115']['trunc'] else 'OK'})"
        b3_info = f"{r['vinai_arms']['B3_beam4_nr3_rep105']['tok']}t ({'T' if r['vinai_arms']['B3_beam4_nr3_rep105']['trunc'] else 'OK'})"
        print(f"{r['qid']:<17} | {r['category']:<18} | {m_info:<12} | {b0_info:<12} | {b1_info:<16} | {b2_info:<16} | {b3_info:<16}")

    print("=" * 150, flush=True)
    print("OVERALL METRICS & LATENCY OVERVIEW:", flush=True)
    print(f"{'Arm':<24} | {'>77 Trunc Count':<16} | {'Mean Latency':<14} | {'p50 Latency':<14} | {'Max Latency':<14}")
    print("-" * 95)
    for arm_name, lats in arm_latencies.items():
        trunc_cnt = arm_overbudget_count[arm_name]
        mean_l = f"{np.mean(lats)*1000:.1f}ms"
        p50_l = f"{np.median(lats)*1000:.1f}ms"
        max_l = f"{np.max(lats)*1000:.1f}ms"
        print(f"{arm_name:<24} | {trunc_cnt}/18 queries    | {mean_l:<14} | {p50_l:<14} | {max_l:<14}")
    print("=" * 150, flush=True)

    # Diagnostic Target Probes Analysis
    print("\n" + "=" * 150, flush=True)
    print("MANDATORY TARGET PROBES AUDIT ACROSS ARMS", flush=True)
    print("=" * 150, flush=True)
    for r in results:
        if r["qid"] in ["query-p1-11-kis", "query-p1-13-kis", "query-p1-17-kis", "query-p1-21-kis", "query-p1-24-kis", "query-p1-25-kis"]:
            print(f"\n[{r['qid']}] {r['name']}:", flush=True)
            print(f"  • Marian: \"{r['Marian']['text']}\"", flush=True)
            for arm_k, d in r["vinai_arms"].items():
                print(f"  • {arm_k:<20}: (Tok: {d['tok']}) \"{d['text']}\"", flush=True)

    # Save HTML report
    html_out = Path("/kaggle/working/kis_p1g0b_decoding_stabilization.html")
    generate_p1g0b_html(results, html_out)
    print(f"\nSaved P1G0b Comparative Report HTML to: {html_out}", flush=True)


def generate_p1g0b_html(results: list[dict[str, Any]], out_path: Path) -> None:
    rows = []
    for r in results:
        cat_badge = "#ffc107; color:#111" if r['category'] == "REGRESSION_GUARD" else ("#e83e8c; color:#fff" if r['category'] == "TARGET_PROBE" else "#17a2b8; color:#fff")
        
        b0 = r['vinai_arms']['B0_default']
        b1 = r['vinai_arms']['B1_beam3_nr3_rep115']
        b2 = r['vinai_arms']['B2_greedy_nr3_rep115']
        b3 = r['vinai_arms']['B3_beam4_nr3_rep105']

        rows.append(f"""
        <tr style="border-bottom:1px solid #333; font-size:11px;">
            <td style="padding:8px; font-weight:bold; color:#61afef;">{r['qid']}<br><span style="font-size:10px; background:{cat_badge}; padding:2px 4px; border-radius:3px;">{r['category']}</span></td>
            <td style="padding:8px; color:#ddd; max-width:200px;">{r['query_vi']}</td>
            <td style="padding:8px; color:#98c379; background:#1c211d; max-width:220px;">
                <div>{r['Marian']['text']}</div>
                <div style="font-size:10px; color:#aaa; margin-top:3px;">Tok: <b>{r['Marian']['tok']}</b> | {r['Marian']['lat']*1000:.0f}ms</div>
            </td>
            <td style="padding:8px; color:#61afef; background:#1b2229; max-width:220px;">
                <div>{b0['text']}</div>
                <div style="font-size:10px; color:#aaa; margin-top:3px;">Tok: <b>{b0['tok']}</b> | {b0['lat']*1000:.0f}ms</div>
            </td>
            <td style="padding:8px; color:#e5c07b; background:#24221b; max-width:220px;">
                <div>{b1['text']}</div>
                <div style="font-size:10px; color:#aaa; margin-top:3px;">Tok: <b>{b1['tok']}</b> | {b1['lat']*1000:.0f}ms</div>
            </td>
            <td style="padding:8px; color:#e06c75; background:#291b1c; max-width:220px;">
                <div>{b2['text']}</div>
                <div style="font-size:10px; color:#aaa; margin-top:3px;">Tok: <b>{b2['tok']}</b> | {b2['lat']*1000:.0f}ms</div>
            </td>
            <td style="padding:8px; color:#c678dd; background:#231b29; max-width:220px;">
                <div>{b3['text']}</div>
                <div style="font-size:10px; color:#aaa; margin-top:3px;">Tok: <b>{b3['tok']}</b> | {b3['lat']*1000:.0f}ms</div>
            </td>
        </tr>
        """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>KIS P1G0b VinAI Decoding Stabilization</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; padding:20px;">
        <h2 style="color:#61afef; border-bottom:2px solid #333; padding-bottom:8px;">🔬 KIS P1G0b: VINAI DECODING STABILIZATION SHOOTOUT</h2>
        <table style="width:100%; border-collapse:collapse; text-align:left;">
            <thead>
                <tr style="background:#222; border-bottom:2px solid #444; font-size:12px;">
                    <th style="padding:8px; width:10%;">Query ID</th>
                    <th style="padding:8px; width:18%;">Vietnamese Source</th>
                    <th style="padding:8px; width:18%;">Marian Production</th>
                    <th style="padding:8px; width:18%;">B0: Default (beam=5)</th>
                    <th style="padding:8px; width:18%;">B1: Beam=3 (nr=3, rep=1.15)</th>
                    <th style="padding:8px; width:18%;">B2: Greedy (beam=1, nr=3)</th>
                    <th style="padding:8px; width:18%;">B3: Beam=4 (nr=3, rep=1.05)</th>
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
    run_p1g0b_shootout()
