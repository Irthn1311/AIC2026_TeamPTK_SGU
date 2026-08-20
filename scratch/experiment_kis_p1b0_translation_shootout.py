#!/usr/bin/env python3
"""KIS P1B0: Translation Fidelity Shootout Prototype (Query-Only Experiment).

Compares:
  - Model A: Frozen Production Marian (Helsinki-NLP/opus-mt-vi-en, rev 5611f34634b72de0608b1238a4e02845ca285f3e).
  - Model B: Multilingual NLLB-200 distilled 600M (facebook/nllb-200-distilled-600M).

Scope:
  - All 18 BTC KIS queries evaluated side-by-side.
  - Zero retrieval, zero CLIP image encoding, zero Phase-4, zero benchmark runs.
  - Zero hard-coded dictionary or lexical correction rules.
  - Evaluates fidelity across visual atoms:
    * Subjects / Objects / Entities
    * Actions / Verbs
    * Counts / Numbers
    * Colors / Clothing
    * Viewpoints / Camera angles
    * Rare visual discriminators
"""

from __future__ import annotations

import os
import re
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex", "transformers", "sentencepiece"], check=False)
    import clip

from system_tai.kis.session_schema import SessionConfig
from system_tai.translation.provider import MarianOfflineTranslator, NLLBOfflineTranslator, TokenBudgetGuard

BTC_KIS_QUERIES = [
    {"qid": "query-p1-1-kis", "category": "REGRESSION_GUARD", "name": "Người chơi đàn Hang / Handpan"},
    {"qid": "query-p1-2-kis", "category": "REGRESSION_GUARD", "name": "Con hổ (Tiger)"},
    {"qid": "query-p1-5-kis", "category": "REGRESSION_GUARD", "name": "Hai người phụ nữ cho dê ăn"},
    {"qid": "query-p1-6-kis", "category": "GENERAL_PROBE", "name": "Cắt tỉa cây cảnh Bonsai"},
    {"qid": "query-p1-7-kis", "category": "REGRESSION_GUARD", "name": "Phi hành gia mặc đồ phi hành (Astronaut)"},
    {"qid": "query-p1-8-kis", "category": "GENERAL_PROBE", "name": "Làm gốm thủ công (Pottery)"},
    {"qid": "query-p1-9-kis", "category": "GENERAL_PROBE", "name": "Trang trí bánh kem nghệ thuật"},
    {"qid": "query-p1-10-kis", "category": "REGRESSION_GUARD", "name": "Người trượt ván trên đường phố"},
    {"qid": "query-p1-11-kis", "category": "REGRESSION_GUARD", "name": "Đổ bóng tạo chân dung mặc vest"},
    {"qid": "query-p1-12-kis", "category": "LONG_QUERY_PROBE", "name": "Trang trí bánh rán dâu tây chuối chocolate"},
    {"qid": "query-p1-13-kis", "category": "LEXICAL_PROBE_TARGET", "name": "Vệ sinh máy ảnh, lens trên khăn hồng, tăm bông"},
    {"qid": "query-p1-14-kis", "category": "GENERAL_PROBE", "name": "Pha trà đạo truyền thống"},
    {"qid": "query-p1-17-kis", "category": "LONG_QUERY_PROBE", "name": "Trao quà từ thiện bệnh viện biển COVID-19"},
    {"qid": "query-p1-20-kis", "category": "REGRESSION_GUARD", "name": "Thêm 2 ly panna cotta, hoa ăn được"},
    {"qid": "query-p1-21-kis", "category": "ENTITY_PROBE_TARGET", "name": "Cơ chế bay của bọ làm robot ở ĐH Lausanne"},
    {"qid": "query-p1-23-kis", "category": "REASONING_CONTROL", "name": "Động vật biển nguy hiểm Steven Spielberg 1975"},
    {"qid": "query-p1-24-kis", "category": "VIEWPOINT_PROBE_TARGET", "name": "Đua xe đạp quay từ trên cao xuống"},
    {"qid": "query-p1-25-kis", "category": "REGRESSION_GUARD", "name": "Robot phục vụ trong nhà hàng"},
]


def run_translation_shootout() -> None:
    print("=" * 150, flush=True)
    print("KIS P1B0: TRANSLATION FIDELITY SHOOTOUT (MARIAN VS NLLB-200-DISTILLED-600M)", flush=True)
    print("=" * 150, flush=True)

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    cfg = SessionConfig.from_yaml(yaml_path)

    # 1. Initialize Marian
    print("\n[1/2] Loading Model A: Frozen Production Marian (opus-mt-vi-en)...", flush=True)
    t0_m = time.time()
    marian = MarianOfflineTranslator(
        revision=cfg.translation_revision,
        local_files_only=True,
    )
    print(f"      Marian loaded in {time.time() - t0_m:.2f}s", flush=True)

    # 2. Initialize NLLB
    print("\n[2/2] Loading Model B: Candidate NLLB-200 distilled 600M...", flush=True)
    print("      • Config: src_lang = vie_Latn", flush=True)
    print("      • Config: target_lang / forced_bos = eng_Latn", flush=True)
    print("      • Config: model = facebook/nllb-200-distilled-600M", flush=True)
    t0_n = time.time()
    nllb = NLLBOfflineTranslator(
        model_name_or_path="facebook/nllb-200-distilled-600M",
        local_files_only=False,
    )
    load_time_n = time.time() - t0_n
    print(f"      • Config: device = {nllb._device}", flush=True)
    print(f"      • Config: load_time = {load_time_n:.2f}s", flush=True)

    guard = TokenBudgetGuard()
    thunghiem_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"

    shootout_results: list[dict[str, Any]] = []

    print("\n" + "=" * 150, flush=True)
    print(f"RUNNING TRANSLATION SHOOTOUT ON ALL {len(BTC_KIS_QUERIES)} BTC KIS QUERIES", flush=True)
    print("=" * 150, flush=True)

    for idx, item in enumerate(BTC_KIS_QUERIES, start=1):
        qid = item["qid"]
        category = item["category"]
        name = item["name"]
        q_file = thunghiem_dir / f"{qid}.txt"

        if not q_file.exists():
            print(f"⚠️ Missing query file: {q_file}", flush=True)
            continue

        q_vi = q_file.read_text(encoding="utf-8").strip()

        # Measure Marian
        t_m0 = time.time()
        marian_en = marian.translate(q_vi)
        marian_lat = time.time() - t_m0
        marian_toks = guard.count_tokens(marian_en)

        # Measure NLLB
        t_n0 = time.time()
        nllb_en = nllb.translate(q_vi)
        nllb_lat = time.time() - t_n0
        nllb_toks = guard.count_tokens(nllb_en)

        shootout_results.append({
            "idx": idx,
            "qid": qid,
            "category": category,
            "name": name,
            "vi": q_vi,
            "marian_en": marian_en,
            "marian_lat": marian_lat,
            "marian_toks": marian_toks,
            "nllb_en": nllb_en,
            "nllb_lat": nllb_lat,
            "nllb_toks": nllb_toks,
        })

        badge = f"[{category}]"
        print(f"\n--- [{idx:02d}/{len(BTC_KIS_QUERIES)}] {qid} {badge} : {name} ---", flush=True)
        print(f"  • VI Source     : \"{q_vi}\"", flush=True)
        print(f"  • Model A Marian: \"{marian_en}\" ({marian_toks:3d} tokens | {marian_lat*1000:5.1f}ms)", flush=True)
        print(f"  • Model B NLLB  : \"{nllb_en}\" ({nllb_toks:3d} tokens | {nllb_lat*1000:5.1f}ms)", flush=True)

        # Specific diagnostics on key probes
        if qid == "query-p1-13-kis":
            print(f"    🔍 TARGET CHECK (p1-13): 'vệ sinh máy ảnh' -> Marian: {'camera toilet ❌' if 'camera toilet' in marian_en.lower() else 'OK'} | NLLB: {nllb_en[:60]}", flush=True)
        elif qid == "query-p1-21-kis":
            print(f"    🔍 TARGET CHECK (p1-21): 'Lausanne' -> Marian: {'Larissa ❌' if 'larissa' in marian_en.lower() else 'OK'} | NLLB: {'Lausanne ✅' if 'lausanne' in nllb_en.lower() else nllb_en[:60]}", flush=True)
        elif qid == "query-p1-24-kis":
            print(f"    🔍 TARGET CHECK (p1-24): 'từ trên cao xuống' -> Marian: {'top-up rotation ❌' if 'top-up' in marian_en.lower() else 'OK'} | NLLB: {nllb_en[:60]}", flush=True)
        elif qid == "query-p1-23-kis":
            print(f"    🔍 CONTROL CHECK (p1-23): Reasoning control -> Marian has shark: {'YES ⚠️' if 'shark' in marian_en.lower() else 'NO (Normal)'} | NLLB has shark: {'YES ⚠️' if 'shark' in nllb_en.lower() else 'NO (Normal)'}", flush=True)

    # Detailed table
    print("\n" + "=" * 150, flush=True)
    print("KIS P1B0 TRANSLATION SHOOTOUT DETAILED SUMMARY TABLE", flush=True)
    print("=" * 150, flush=True)
    print(f"{'Query ID':<18} | {'Category':<22} | {'Marian Tokens':<14} | {'NLLB Tokens':<12} | {'Marian Lat':<11} | {'NLLB Lat':<10}")
    print("-" * 100)
    for r in shootout_results:
        print(f"{r['qid']:<18} | {r['category']:<22} | {r['marian_toks']:<14d} | {r['nllb_toks']:<12d} | {r['marian_lat']*1000:6.1f} ms  | {r['nllb_lat']*1000:6.1f} ms")
    print("=" * 150, flush=True)

    # Aggregate metrics
    marian_tok_list = [r["marian_toks"] for r in shootout_results]
    nllb_tok_list = [r["nllb_toks"] for r in shootout_results]
    marian_lat_list = [r["marian_lat"] for r in shootout_results]
    nllb_lat_list = [r["nllb_lat"] for r in shootout_results]

    marian_over_77 = sum(1 for t in marian_tok_list if t > 77)
    nllb_over_77 = sum(1 for t in nllb_tok_list if t > 77)

    marian_mean_tok = sum(marian_tok_list) / len(marian_tok_list) if marian_tok_list else 0
    nllb_mean_tok = sum(nllb_tok_list) / len(nllb_tok_list) if nllb_tok_list else 0

    marian_max_tok = max(marian_tok_list) if marian_tok_list else 0
    nllb_max_tok = max(nllb_tok_list) if nllb_tok_list else 0

    marian_mean_lat = sum(marian_lat_list) / len(marian_lat_list) if marian_lat_list else 0
    nllb_mean_lat = sum(nllb_lat_list) / len(nllb_lat_list) if nllb_lat_list else 0

    print("\n" + "=" * 150, flush=True)
    print("AGGREGATE TRANSLATOR METRICS COMPARISON", flush=True)
    print("=" * 150, flush=True)
    print(f"{'Metric':<35} | {'Model A (Marian Baseline)':<30} | {'Model B (NLLB Candidate)':<30}")
    print("-" * 100)
    print(f"{'Queries > 77 Tokens':<35} | {f'{marian_over_77} / {len(shootout_results)} ({marian_over_77/len(shootout_results)*100:.1f}%)':<30} | {f'{nllb_over_77} / {len(shootout_results)} ({nllb_over_77/len(shootout_results)*100:.1f}%)':<30}")
    print(f"{'Mean CLIP Tokens':<35} | {f'{marian_mean_tok:.1f} tokens':<30} | {f'{nllb_mean_tok:.1f} tokens':<30}")
    print(f"{'Max CLIP Tokens':<35} | {f'{marian_max_tok} tokens':<30} | {f'{nllb_max_tok} tokens':<30}")
    print(f"{'Mean Translation Latency':<35} | {f'{marian_mean_lat*1000:.1f} ms':<30} | {f'{nllb_mean_lat*1000:.1f} ms':<30}")
    print("=" * 150, flush=True)


if __name__ == "__main__":
    run_translation_shootout()
