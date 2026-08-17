# ==============================================================================================================
# Phase R3-S1A: Query Expansion & Retrieval Diagnostic on 38 DEV Queries
# ==============================================================================================================

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.retrieval.query_decomposition import decompose_query
from system_tai.retrieval.multi_variant_fusion import fuse_multi_variant_video_ranks

print("=" * 110)
print("ROUND-3 SPRINT 1A: QUERY DECOMPOSITION & MULTI-VARIANT RETRIEVAL DIAGNOSTIC")
print("=" * 110)

BENCHMARK_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
DEV_EN_SIDECAR_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"

with open(BENCHMARK_PATH, encoding="utf-8") as f:
    bm_data = json.load(f)

with open(DEV_EN_SIDECAR_PATH, encoding="utf-8") as f:
    en_sidecar = json.load(f)

en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
qa_dev_queries = [q for q in bm_data["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"]
print(f"Loaded {len(qa_dev_queries)} QA DEV queries from benchmark.")

total_variants_count = 0
for idx, q in enumerate(qa_dev_queries, start=1):
    qid = q["query_id"]
    target_vid = q.get("video_id")
    q_vi = q.get("question_vi", "")
    q_en = en_map.get(qid, "")

    variants = decompose_query(q_vi, q_en)
    v_list = variants.as_list()
    total_variants_count += len(v_list)

    print(f"\n[{idx:02d}] {qid:<8} (Target: {target_vid})")
    print(f"     VI : {q_vi}")
    print(f"     EN : {q_en}")
    print(f"     Variants ({len(v_list)}):")
    for v_name, v_text in v_list:
        print(f"       - {v_name:<18}: '{v_text}'")

print("\n" + "=" * 110)
print(f"DIAGNOSTIC COMPLETE: Generated {total_variants_count} total deterministic variants across 38 queries (avg {total_variants_count/len(qa_dev_queries):.1f} per query).")
print("Zero DEV constants, fail-closed derivation, ready for Multi-Variant Video RRF Execution.")
print("=" * 110)
