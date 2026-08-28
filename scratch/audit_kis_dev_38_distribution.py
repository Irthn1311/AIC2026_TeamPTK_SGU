"""Full 38 KIS DEV Benchmark Query Decomposition and Adaptive-K Audit."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(".").resolve()
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.retrieval.semantic_query import (
    decompose_vietnamese_semantic_units,
    SemanticQueryConfig,
    SemanticUnitRole,
)
from system_tai.kis.video_first import compute_adaptive_video_budget_v2

gt_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "kis_dev_gt.json"
sidecar_path = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "q2_kis_dev_en_translation.json"

gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
sidecar_map = {r["query_id"]: r for r in sidecar_data["records"]}

queries = gt_data["queries"]
cfg = SemanticQueryConfig()

results = []
entropies = []

sharp_scores = [0.38, 0.34, 0.32, 0.31, 0.30] + [0.30 - 0.004 * i for i in range(1, 28)]
moderate_scores = [0.28, 0.275, 0.270, 0.265, 0.260] + [0.260 - 0.002 * i for i in range(1, 28)]

for q in queries:
    qid = q["query_id"]
    s_entry = sidecar_map[qid]
    q_vi = s_entry["source_vi"]
    units = decompose_vietnamese_semantic_units(query_id=qid, query_vi=q_vi, config=cfg)
    
    clause_count = len(units)
    has_attr = any(u.role == SemanticUnitRole.SUPPORTING_ATTRIBUTE for u in units)
    
    k_sharp, diag_sharp = compute_adaptive_video_budget_v2(sharp_scores, clause_count=clause_count, has_attributes=has_attr)
    k_mod, diag_mod = compute_adaptive_video_budget_v2(moderate_scores, clause_count=clause_count, has_attributes=has_attr)
    
    entropies.append(diag_sharp.normalized_entropy)
    
    results.append({
        "query_id": qid,
        "video_id": q["video_id"],
        "query_vi": q_vi,
        "clause_count": clause_count,
        "has_attributes": has_attr,
        "roles": [u.role.value for u in units],
        "k_sharp": k_sharp,
        "k_mod": k_mod,
        "sharp_entropy": diag_sharp.normalized_entropy,
        "sharp_delta1_5": diag_sharp.top1_top5_margin,
        "reasons_sharp": diag_sharp.adaptive_reasons,
    })

print("================================================================================")
print(f"38 DEV QUERIES DECOMPOSITION AUDIT")
print("================================================================================")
for r in results:
    print(f"[{r['query_id']}] {r['video_id']} | Clauses: {r['clause_count']} (Attr: {r['has_attributes']}) -> K_sharp={r['k_sharp']}, K_mod={r['k_mod']} | Reasons: {r['reasons_sharp']}")

print("\n================================================================================")
print("AGGREGATE CLAUSE & COMPLEXITY DISTRIBUTION (38 DEV QUERIES)")
print("================================================================================")
c_2 = sum(1 for r in results if r['clause_count'] == 2)
c_3 = sum(1 for r in results if r['clause_count'] == 3)
c_ge_4 = sum(1 for r in results if r['clause_count'] >= 4)
has_attr_count = sum(1 for r in results if r['has_attributes'])

print(f"• Queries with clause_count == 2 (Full + 1 Primary) : {c_2}/38 ({c_2/38*100:.1f}%)")
print(f"• Queries with clause_count == 3 (Full + 2 Units)   : {c_3}/38 ({c_3/38*100:.1f}%)")
print(f"• Queries with clause_count >= 4 (Full + 3+ Units)  : {c_ge_4}/38 ({c_ge_4/38*100:.1f}%)")
print(f"• Queries with Supporting Attributes (has_attr=True): {has_attr_count}/38 ({has_attr_count/38*100:.1f}%)")

print("\n================================================================================")
print("ENTROPY & TEMPERATURE CALIBRATION AUDIT")
print("================================================================================")
ent_sorted = sorted(entropies)
p90_idx = int(0.90 * len(ent_sorted))
print(f"• Sharp Distribution (Delta 1-5 = 0.080):")
print(f"  - Entropy min    : {ent_sorted[0]:.4f}")
print(f"  - Entropy median : {ent_sorted[len(ent_sorted)//2]:.4f}")
print(f"  - Entropy p90    : {ent_sorted[p90_idx]:.4f}")
print(f"  - Entropy max    : {ent_sorted[-1]:.4f}")
print(f"• Budget Selection on Sharp Dist (Confident):")
print(f"  - % K=32 : {sum(1 for r in results if r['k_sharp'] == 32)/38*100:.1f}% ({sum(1 for r in results if r['k_sharp'] == 32)} queries)")
print(f"  - % K=48 : {sum(1 for r in results if r['k_sharp'] == 48)/38*100:.1f}% ({sum(1 for r in results if r['k_sharp'] == 48)} queries)")
print(f"  - % K=64 : {sum(1 for r in results if r['k_sharp'] == 64)/38*100:.1f}% ({sum(1 for r in results if r['k_sharp'] == 64)} queries)")

print(f"\n• Budget Selection on Moderate Flat Dist (Delta 1-5 = 0.020):")
print(f"  - % K=32 : {sum(1 for r in results if r['k_mod'] == 32)/38*100:.1f}% ({sum(1 for r in results if r['k_mod'] == 32)} queries)")
print(f"  - % K=48 : {sum(1 for r in results if r['k_mod'] == 48)/38*100:.1f}% ({sum(1 for r in results if r['k_mod'] == 48)} queries)")
print(f"  - % K=64 : {sum(1 for r in results if r['k_mod'] == 64)/38*100:.1f}% ({sum(1 for r in results if r['k_mod'] == 64)} queries)")
