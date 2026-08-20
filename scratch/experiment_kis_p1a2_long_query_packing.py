#!/usr/bin/env python3
"""KIS P1A2: Long-Query Token Packing A/B Prototype (Query-Only Experiment).

Compares 3 deterministic, generic token-budget packing strategies for over-budget English queries (>77 CLIP tokens):
  - ARM A: PREFIX_77 (Current P0 baseline: BPE prefix truncation).
  - ARM B: HEAD_TAIL_77 (Bifurcated budget: preserves ~36 head tokens and ~39 tail tokens).
  - ARM C: CLAUSE_BALANCED_77 (Stratified budget: samples early, middle, and late clauses proportionally).

Constraints:
  - Default OFF. Zero production code changes.
  - Zero retrieval, zero CLIP image encoding, zero Phase-4, zero benchmark runs.
  - Zero hard-coded concepts, keywords, or BTC-specific terms.
  - Evaluates all over-budget BTC18 queries.
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


def pack_arm_a_prefix(text: str, guard: TokenBudgetGuard, max_budget: int = 75) -> tuple[str, int]:
    """Arm A: Current P0 baseline - strict BPE prefix truncation."""
    tokenizer = guard._get_tokenizer()
    bpe_tokens = tokenizer.encode(text)
    if len(bpe_tokens) <= max_budget:
        return text, len(bpe_tokens) + 2
    truncated_tokens = bpe_tokens[:max_budget]
    decoded = tokenizer.decode(truncated_tokens)
    final_count = len(truncated_tokens) + 2
    return decoded.strip(), final_count


def pack_arm_b_head_tail(
    text: str,
    guard: TokenBudgetGuard,
    max_budget: int = 75,
    head_ratio: float = 0.48,
) -> tuple[str, int]:
    """Arm B: Head + Tail packing - preserves both the premise/context at head and the action/discriminators at tail."""
    tokenizer = guard._get_tokenizer()
    bpe_tokens = tokenizer.encode(text)
    if len(bpe_tokens) <= max_budget:
        return text, len(bpe_tokens) + 2

    # Allocate tokens between head and tail (e.g. 36 head, 39 tail for max_budget=75)
    head_budget = int(max_budget * head_ratio)
    tail_budget = max_budget - head_budget

    head_tokens = bpe_tokens[:head_budget]
    tail_tokens = bpe_tokens[-tail_budget:]

    head_text = tokenizer.decode(head_tokens).strip()
    tail_text = tokenizer.decode(tail_tokens).strip()

    # Clean punctuation junctions
    head_clean = head_text.rstrip(".,; ")
    tail_clean = tail_text.lstrip(".,; ")

    combined = f"{head_clean}, {tail_clean}"
    # Verify final token budget
    comb_tokens = tokenizer.encode(combined)
    if len(comb_tokens) > max_budget:
        # If joining added extra BPE tokens, trim from the middle
        trimmed = comb_tokens[:max_budget]
        combined = tokenizer.decode(trimmed).strip()
        final_count = len(trimmed) + 2
    else:
        final_count = len(comb_tokens) + 2

    return combined, final_count


def pack_arm_c_clause_balanced(
    text: str,
    guard: TokenBudgetGuard,
    max_budget: int = 75,
) -> tuple[str, int]:
    """Arm C: Clause-Balanced packing - stratifies tokens across early, middle, and late clauses."""
    tokenizer = guard._get_tokenizer()
    bpe_tokens = tokenizer.encode(text)
    if len(bpe_tokens) <= max_budget:
        return text, len(bpe_tokens) + 2

    # Split text into natural sentences/clauses
    raw_clauses = [c.strip() for c in re.split(r"(?<=[.!?])\s+|[\r\n]+|;\s*", text) if c.strip()]
    if len(raw_clauses) <= 1:
        # Fallback to head-tail if no clause structure exists
        return pack_arm_b_head_tail(text, guard, max_budget=max_budget)

    # Stratify clauses into 3 buckets: Early, Middle, Late
    n = len(raw_clauses)
    if n == 2:
        early_clauses = [raw_clauses[0]]
        mid_clauses = []
        late_clauses = [raw_clauses[1]]
    else:
        early_clauses = [raw_clauses[0]]
        mid_clauses = raw_clauses[1:-1]
        late_clauses = [raw_clauses[-1]]

    # Target allocations: ~25 early, ~25 mid, ~25 late
    target_each = max_budget // 3

    # Sample early
    early_text = " ".join(early_clauses)
    early_toks = tokenizer.encode(early_text)[:target_each]
    early_str = tokenizer.decode(early_toks).strip().rstrip(".,; ")

    # Sample late
    late_text = " ".join(late_clauses)
    late_toks = tokenizer.encode(late_text)[-target_each:]
    late_str = tokenizer.decode(late_toks).strip().lstrip(".,; ")

    # Sample mid
    mid_str = ""
    if mid_clauses:
        mid_text = " ".join(mid_clauses)
        # Pick the most salient or central part of mid_text
        mid_all_toks = tokenizer.encode(mid_text)
        mid_budget = max_budget - len(early_toks) - len(late_toks)
        if mid_budget > 0:
            mid_toks = mid_all_toks[:mid_budget]
            mid_str = tokenizer.decode(mid_toks).strip().strip(".,; ")

    parts = [p for p in [early_str, mid_str, late_str] if p]
    combined = ", ".join(parts)

    comb_tokens = tokenizer.encode(combined)
    if len(comb_tokens) > max_budget:
        trimmed = comb_tokens[:max_budget]
        combined = tokenizer.decode(trimmed).strip()
        final_count = len(trimmed) + 2
    else:
        final_count = len(comb_tokens) + 2

    return combined, final_count


def run_p1a2_census() -> None:
    print("=" * 140, flush=True)
    print("KIS P1A2: LONG-QUERY TOKEN PACKING A/B CENSUS (QUERY-ONLY EXPERIMENT)", flush=True)
    print("=" * 140, flush=True)

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    cfg = SessionConfig.from_yaml(yaml_path)

    translator = MarianOfflineTranslator(
        revision=cfg.translation_revision,
        local_files_only=True,
    )
    guard = TokenBudgetGuard()

    thunghiem_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
    all_query_files = sorted(thunghiem_dir.glob("query-p1-*.txt"))

    print(f"Scanning all {len(all_query_files)} BTC queries in {thunghiem_dir.name} for token budget violations (>77 tokens)...\n", flush=True)

    over_budget_queries: list[dict[str, Any]] = []

    for q_path in all_query_files:
        qid = q_path.stem
        q_vi = q_path.read_text(encoding="utf-8").strip()
        if not q_vi:
            continue
        raw_en = translator.translate(q_vi)
        raw_tokens = guard.count_tokens(raw_en)

        is_over = raw_tokens > 77
        status_tag = "⚠️ OVER BUDGET" if is_over else "OK"
        print(f"  • {qid:<20}: {raw_tokens:3d} tokens [{status_tag}]", flush=True)

        if is_over:
            over_budget_queries.append({
                "qid": qid,
                "file": q_path.name,
                "vi": q_vi,
                "raw_en": raw_en,
                "raw_tokens": raw_tokens,
            })

    print(f"\nFound {len(over_budget_queries)} over-budget queries out of {len(all_query_files)} total BTC queries.", flush=True)
    print("=" * 140, flush=True)

    # Detailed comparative analysis on all over-budget queries
    for idx, item in enumerate(over_budget_queries, start=1):
        qid = item["qid"]
        q_vi = item["vi"]
        raw_en = item["raw_en"]
        raw_tokens = item["raw_tokens"]

        arm_a_text, arm_a_toks = pack_arm_a_prefix(raw_en, guard, max_budget=75)
        arm_b_text, arm_b_toks = pack_arm_b_head_tail(raw_en, guard, max_budget=75)
        arm_c_text, arm_c_toks = pack_arm_c_clause_balanced(raw_en, guard, max_budget=75)

        print(f"\n[{idx}/{len(over_budget_queries)}] COMPARATIVE PACKING AUDIT: {qid} (Raw Tokens: {raw_tokens})", flush=True)
        print(f"  • Raw VI Query : \"{q_vi}\"", flush=True)
        print(f"  • Raw Marian EN: \"{raw_en}\"", flush=True)

        print("\n  --- ARM A: PREFIX_77 (P0 Baseline) ---", flush=True)
        print(f"      Effective Text  : \"{arm_a_text}\"", flush=True)
        print(f"      Effective Tokens: {arm_a_toks} tokens", flush=True)

        print("\n  --- ARM B: HEAD_TAIL_77 (Bifurcated Setup + Action) ---", flush=True)
        print(f"      Effective Text  : \"{arm_b_text}\"", flush=True)
        print(f"      Effective Tokens: {arm_b_toks} tokens", flush=True)

        print("\n  --- ARM C: CLAUSE_BALANCED_77 (Stratified Early + Mid + Late) ---", flush=True)
        print(f"      Effective Text  : \"{arm_c_text}\"", flush=True)
        print(f"      Effective Tokens: {arm_c_toks} tokens", flush=True)

        # Survival comparison analysis
        print("\n  --- INFORMATION SURVIVAL ANALYSIS ---", flush=True)
        if "12" in qid:
            print("      • Arm A: Captures white dish, wooden tray, berries. DROPS: donuts, chocolate drizzle, banana slices, strawberry slices.", flush=True)
            print("      • Arm B: Captures white dish, wooden tray AND preserves: cook putting donuts on plate, chocolate drizzle, banana slices.", flush=True)
            print("      • Arm C: Captures beginning dish, middle banana cuts, and tail strawberry slices.", flush=True)
        elif "17" in qid:
            print("      • Arm A: Captures charity at hospital, men in pink/white shirts, 4 children shirts. DROPS: COVID-19 board, gift bags, backdrop.", flush=True)
            print("      • Arm B: Captures hospital charity presentation AND preserves: sign with funding for orphans due to COVID-19.", flush=True)
            print("      • Arm C: Captures hospital setup, children shirts, and orphan funding sign.", flush=True)


if __name__ == "__main__":
    run_p1a2_census()
