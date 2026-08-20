#!/usr/bin/env python3
"""KIS P1A1: Clause-Wise Visual Query Distillation Prototype (Query-Only Experiment).

Compares:
  - ARM P0: Full-query translation + prefix BPE truncation (current production baseline).
  - ARM P1A1: Clause-wise independent translation + generic boilerplate stripping + visual clause distillation (<=77 CLIP tokens).

Audits all 6 forensic probes:
  - query-p1-12-kis (donuts)
  - query-p1-13-kis (camera cleaning)
  - query-p1-17-kis (charity hospital)
  - query-p1-21-kis (beetle robot)
  - query-p1-23-kis (Spielberg 1975)
  - query-p1-24-kis (cycling overhead)

Strict constraints:
  - Zero retrieval, zero CLIP search, zero Phase-4, zero benchmark run.
  - Zero hardcoding of queries, concepts, or dictionary corrections.
  - Pinned Marian revision: 5611f34634b72de0608b1238a4e02845ca285f3e.
"""

from __future__ import annotations

import os
import re
import sys
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex"], check=False)
    import clip

from system_tai.kis.session_schema import SessionConfig
from system_tai.translation.provider import MarianOfflineTranslator, TokenBudgetGuard

FORENSIC_QUERIES = [
    {
        "qid": "query-p1-12-kis",
        "topic": "Trang trí bánh rán, rưới chocolate dâu tây chuối (Long Query Control)",
        "file": "query-p1-12-kis.txt",
    },
    {
        "qid": "query-p1-13-kis",
        "topic": "Vệ sinh máy ảnh, tháo ống kính đặt trên khăn hồng tím, tăm bông",
        "file": "query-p1-13-kis.txt",
    },
    {
        "qid": "query-p1-17-kis",
        "topic": "Trao quà từ thiện tại bệnh viện cho 4 em nhỏ nhận biển COVID-19",
        "file": "query-p1-17-kis.txt",
    },
    {
        "qid": "query-p1-21-kis",
        "topic": "Nghiên cứu cơ chế bay của bọ để chế tạo robot ở ĐH Lausanne",
        "file": "query-p1-21-kis.txt",
    },
    {
        "qid": "query-p1-23-kis",
        "topic": "Động vật biển nguy hiểm phim Steven Spielberg 1975 (Cá mập Jaws)",
        "file": "query-p1-23-kis.txt",
    },
    {
        "qid": "query-p1-24-kis",
        "topic": "Đua xe đạp quay từ trên cao, 3 tay đua thẳng hàng đồng phục",
        "file": "query-p1-24-kis.txt",
    },
]

# Generic boilerplate patterns to strip (purely structural / non-visual filler)
BOILERPLATE_PATTERNS = [
    re.compile(r"^(the\s+)?video\s+(depicts|shows|describes|of)\s+", re.IGNORECASE),
    re.compile(r"^(the\s+)?clip\s+(depicts|shows|describes|of)\s+", re.IGNORECASE),
    re.compile(r"^(the\s+)?scene\s+(begins|starts)\s+(as|with)\s+", re.IGNORECASE),
    re.compile(r"^the\s+next\s+scene\s+shows\s+(that\s+)?", re.IGNORECASE),
    re.compile(r"^find\s+(exactly\s+)?(the\s+)?(short\s+)?clip\s+(of\s+|with\s+)?", re.IGNORECASE),
    re.compile(r"^looking\s+for\s+(a\s+)?(scene|view|clip)\s+(of\s+|with\s+)?", re.IGNORECASE),
    re.compile(r"^(in\s+)?the\s+frame\s+(consists\s+of|includes|has)\s+", re.IGNORECASE),
]


def split_into_clauses_generic(text_vi: str) -> list[str]:
    """Split Vietnamese text into natural sentences and major clauses generically."""
    # Split on sentence terminals first
    raw_sentences = re.split(r"(?<=[.!?])\s+|[\r\n]+", text_vi.strip())
    clauses: list[str] = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        # Remove trailing period for cleaner independent clause translation
        s_clean = s.rstrip(".").strip()
        if s_clean:
            clauses.append(s_clean)
    return clauses


def clean_boilerplate_generic(en_text: str) -> str:
    """Strip generic video-description boilerplate without modifying domain concepts."""
    cleaned = en_text.strip()
    for pat in BOILERPLATE_PATTERNS:
        cleaned = pat.sub("", cleaned).strip()
    return cleaned


def distill_visual_query_generic(translated_clauses: list[str], guard: TokenBudgetGuard, max_budget: int = 75) -> str:
    """Distill translated clauses into a concise visual query under the token budget."""
    cleaned_clauses: list[str] = []
    for c in translated_clauses:
        cl = clean_boilerplate_generic(c)
        # Normalize punctuation and casing
        cl = cl.rstrip(". ").strip()
        if cl:
            cleaned_clauses.append(cl)

    # Greedily pack clauses separated by comma/semicolon while within token budget
    distilled_parts: list[str] = []
    current_text = ""

    for cl in cleaned_clauses:
        cand_text = f"{current_text}, {cl}" if current_text else cl
        toks = guard.count_tokens(cand_text)
        if toks <= max_budget + 2:  # count_tokens includes SOT and EOT (+2)
            current_text = cand_text
            distilled_parts.append(cl)
        else:
            # If a single clause is long, try packing what fits or keep going
            break

    if not current_text and cleaned_clauses:
        # Fallback to guarded first clause
        current_text, _, _ = guard.guard_and_compact(cleaned_clauses[0])

    return current_text


def run_p1a1_experiment() -> None:
    print("=" * 140, flush=True)
    print("KIS P1A1: CLAUSE-WISE VISUAL QUERY DISTILLATION PROTOTYPE (QUERY-ONLY)", flush=True)
    print("=" * 140, flush=True)

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    cfg = SessionConfig.from_yaml(yaml_path)

    translator = MarianOfflineTranslator(
        revision=cfg.translation_revision,
        local_files_only=True,
    )
    guard = TokenBudgetGuard()

    thunghiem_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"

    for idx, item in enumerate(FORENSIC_QUERIES, start=1):
        qid = item["qid"]
        topic = item["topic"]
        q_file = thunghiem_dir / item["file"]
        q_vi = q_file.read_text(encoding="utf-8").strip() if q_file.exists() else ""

        # --- ARM P0 (Production baseline) ---
        p0_raw_marian_en = translator.translate(q_vi)
        p0_raw_tokens = guard.count_tokens(p0_raw_marian_en)
        p0_eff_en, p0_eff_tokens, p0_compacted = guard.guard_and_compact(p0_raw_marian_en)

        # --- ARM P1A1 (Clause-wise translation & distillation) ---
        vi_clauses = split_into_clauses_generic(q_vi)
        translated_clauses: list[dict[str, str]] = []
        raw_en_clause_list: list[str] = []

        for c_idx, c_vi in enumerate(vi_clauses, start=1):
            c_en = translator.translate(c_vi)
            translated_clauses.append({"c_idx": str(c_idx), "vi": c_vi, "en": c_en})
            raw_en_clause_list.append(c_en)

        p1a1_distilled_en = distill_visual_query_generic(raw_en_clause_list, guard, max_budget=75)
        p1a1_tokens = guard.count_tokens(p1a1_distilled_en)

        print(f"\n{'=' * 140}", flush=True)
        print(f"[{idx}/6] {qid}: {topic}", flush=True)
        print(f"{'=' * 140}", flush=True)

        print("\n--- [ARM P0: PRODUCTION BASELINE (Full Query MT + Prefix Truncation)] ---", flush=True)
        print(f"  • P0 Raw Marian EN       : \"{p0_raw_marian_en}\" ({p0_raw_tokens} tokens)", flush=True)
        print(f"  • P0 Effective Retrieval : \"{p0_eff_en}\" ({p0_eff_tokens} tokens | Compacted: {'YES ⚠️' if p0_compacted else 'NO ✅'})", flush=True)

        print("\n--- [ARM P1A1: CLAUSE-WISE INDEPENDENT MT BREAKDOWN] ---", flush=True)
        for tc in translated_clauses:
            print(f"  [Clause {tc['c_idx']}]", flush=True)
            print(f"    • VI: \"{tc['vi']}\"", flush=True)
            print(f"    • EN: \"{tc['en']}\"", flush=True)

        print("\n--- [ARM P1A1: DISTILLED VISUAL QUERY (<=77 CLIP Tokens)] ---", flush=True)
        print(f"  • P1A1 Distilled EN      : \"{p1a1_distilled_en}\"", flush=True)
        print(f"  • P1A1 CLIP Tokens       : {p1a1_tokens} tokens (Budget: <=77)", flush=True)


if __name__ == "__main__":
    run_p1a1_experiment()
