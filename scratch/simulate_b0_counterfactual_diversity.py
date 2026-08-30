"""
Phase B0: Counterfactual & Temporal Diversity Simulation
========================================================
Runs instant offline simulation on existing manifest/matrix without re-running pipeline.
1. Counterfactual text comparison on L30_V046 (with vs without 'standing', symmetric postures).
2. Complete Top-30 local timestamps to verify/refute cluster monopolization.
3. Candidate depth K in {10, 15, 20, 25, 30, 40} inclusion simulation for GT interval [264.0, 274.0]s.
4. Hard Temporal NMS simulation with gap in {3s, 5s, 10s} (15 candidates per variant).
"""

import json, sys
from pathlib import Path
import numpy as np
import torch, clip

REPO = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("/kaggle/working/AIC2026_TeamPTK_SGU")
OUT = Path("/kaggle/working/output/v2a3_foundation_closure") if Path("/kaggle/working").exists() else REPO / "scratch" / "v2a3_foundation_closure"
SRC = REPO / "systems" / "system_tai" / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from system_tai.features.btc_clip_store import VideoFeatureStoreLoader


def run_b0_simulation() -> None:
    manifest_path = OUT / "feature_manifest.json"
    csv_path = None
    npy_path = None

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next((v for v in manifest.get("videos", []) if v.get("video_id") == "L30_V046"), None)
        if entry:
            csv_path = Path(entry["mapping_csv_path"])
            npy_path = Path(entry["clip_npy_path"])

    if csv_path is None or not csv_path.is_file():
        input_root = Path("/kaggle/input") if Path("/kaggle/input").exists() else REPO / "scratch"
        csv_candidates = list(input_root.glob("**/L30_V046.csv"))
        npy_candidates = list(input_root.glob("**/L30_V046.npy"))
        assert csv_candidates, f"Không tìm thấy L30_V046.csv trong {input_root}"
        assert npy_candidates, f"Không tìm thấy L30_V046.npy trong {input_root}"
        csv_path = csv_candidates[0]
        npy_path = npy_candidates[0]

    # Pipeline exact variants (from previous candidate artifact or canonical pipeline definitions)
    variants = []
    jsonl_path = OUT / "p1-1_top100_breakdown.jsonl"
    if jsonl_path.is_file():
        final_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        def result_signature(rows):
            return [(int(r["rank"]), str(r["video_id"]), int(r["frame_id"])) for r in rows]
        final_sig = result_signature(final_rows)

        matched = []
        for p in (OUT / "requests").glob("audit-top100-p1-1-*/candidates.json"):
            pay = json.loads(p.read_text(encoding="utf-8"))
            if pay.get("query_id") == "query-p1-1-kis" and result_signature(pay.get("records", [])) == final_sig:
                matched.append((p, pay))
        if matched:
            candidate_path, candidate_payload = max(matched, key=lambda item: item[0].stat().st_mtime)
            seen = set()
            for u in candidate_payload.get("translation", {}).get("units", []):
                for s in u.get("segments", []):
                    vid, txt = s.get("variant_id"), s.get("text")
                    if vid and txt and vid not in seen:
                        seen.add(vid)
                        variants.append({"id": vid, "text": txt, "role": u.get("role")})

    if not variants:
        # Canonical pipeline exact translation variants for P1-1
        variants = [
            {"id": "vi_primary", "text": "Cảnh quay một nhóm người trên 5 người đang đứng thành hàng tập thể dục thực hiện động tác hai tay chạm mũi chân, trong nhóm chỉ có một người đeo kính và ba người đội nón màu đỏ", "role": "PRIMARY_QUERY"},
            {"id": "semantic_01", "text": "The scene shows a group of more than 5 people standing in a row to exercise, performing the movement of both hands touching their toes. In the group, only one person wore glasses and three people wore red hats.", "role": "GLOBAL_SUMMARY"},
            {"id": "semantic_02", "text": "A group of more than 5 people doing toe touch exercise", "role": "KEY_ACTION"},
            {"id": "semantic_03", "text": "A group of people standing in a row wearing red hats and glasses doing exercise", "role": "DISTINCT_FEATURE"},
        ]

    assert variants, "Không đọc được pipeline variants"

    # Add Counterfactual & Symmetric Probes
    cf_variants = [
        {"id": "cf_full_without_standing", "text": "The scene shows a group of more than 5 people in a row to exercise, performing the movement of both hands touching their toes. In the group, only one person wore glasses and three people wore red hats.", "role": "COUNTERFACTUAL_NO_STANDING"},
        {"id": "cf_symmetric_standing", "text": "A group of people standing in a row doing toe touch exercise", "role": "SYMMETRIC_PROBE_STANDING"},
        {"id": "cf_symmetric_seated", "text": "A group of people sitting in a row doing toe touch exercise", "role": "SYMMETRIC_PROBE_SEATED"},
        {"id": "cf_oracle_post_hoc", "text": "A group of people sitting on the ground bending forward touching feet", "role": "POST_HOC_ORACLE_PROBE"},
    ]

    all_test_variants = variants + cf_variants

    print("=" * 120)
    print(f"🔬 PHASE B0: OFFLINE COUNTERFACTUAL & TEMPORAL DIVERSITY SIMULATION (L30_V046)")
    print(f"• CSV Source: {csv_path}")
    print(f"• NPY Source: {npy_path}")
    print("=" * 120)

    # Fast single-store loading
    target_store = VideoFeatureStoreLoader(expected_dimension=512, memory_map=True).load(
        video_id="L30_V046",
        mapping_csv_path=csv_path,
        clip_npy_path=npy_path,
    )

    matrix = np.asarray(target_store.matrix, dtype=np.float32)
    if not target_store.descriptor.normalized:
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    texts = [v["text"] for v in all_test_variants]
    with torch.no_grad():
        toks = clip.tokenize(texts, truncate=True).to(device)
        q_vecs = model.encode_text(toks).float()
        q_vecs /= q_vecs.norm(dim=-1, keepdim=True)
        q_vecs = q_vecs.cpu().numpy().astype(np.float32)

    scores = matrix @ q_vecs.T  # shape (97, num_variants)
    gt_rows = [i for i, m in enumerate(target_store.mappings) if 264.0 <= float(m.pts_time) <= 274.0]
    assert gt_rows, "Không có keyframe trong HUMAN-VERIFIED INTERVAL [264,274]s"

    # 1. COUNTERFACTUAL TEXT IMPACT
    print("\n1. COUNTERFACTUAL TEXT IMPACT TRÊN L30_V046 / f6784")
    print(f"{'Variant ID / Description':<38} | {'Best GT Frame':<16} | {'Score':<8} | {'Local Rank':<10} | {'In Top-10?'}")
    print(f"{'-'*38} | {'-'*16} | {'-'*8} | {'-'*10} | {'-'*10}")

    for j, v in enumerate(all_test_variants):
        best_gt = max(gt_rows, key=lambda r: scores[r, j])
        m = target_store.mappings[best_gt]
        sc = float(scores[best_gt, j])
        loc_rank = 1 + int(np.count_nonzero(scores[:, j] > sc))
        in_t10 = "YES ✅" if loc_rank <= 10 else "NO ❌"
        print(f"{v['id']:<38} | f{m.frame_id}@{m.pts_time:.1f}s{'':<5} | {sc:<8.4f} | {loc_rank:>2}/97{'':<5} | {in_t10}")

    # 2. COMPLETE TOP-30 TIMESTAMPS (TEST CLUSTER MONOPOLIZATION)
    print("\n" + "=" * 120)
    print("2. COMPLETE TOP-30 LOCAL TIMESTAMPS OF L30_V046 (ACTUAL VARIANTS)")
    print("=" * 120)
    for j in range(len(variants)):
        v = variants[j]
        sorted_rows = np.argsort(-scores[:, j])[:30]
        print(f"\n• Variant: [{v['id']}] (role={v['role']})")
        row_strs = []
        for rk, r in enumerate(sorted_rows, start=1):
            m = target_store.mappings[int(r)]
            in_gt = "🎯[HUMAN_INTERVAL]" if 264.0 <= float(m.pts_time) <= 274.0 else ""
            row_strs.append(f"#{rk:02d}: f{m.frame_id}@{m.pts_time:.1f}s({scores[r, j]:.3f}) {in_gt}")
        for col_idx in range(10):
            line = f"  {row_strs[col_idx]:<34} | {row_strs[col_idx+10]:<34} | {row_strs[col_idx+20]:<34}"
            print(line)

    # 3. SIMULATE CANDIDATE DEPTH K IN {10, 15, 20, 25, 30, 40}
    print("\n" + "=" * 120)
    print("3. CANDIDATE DEPTH K INCLUSION SIMULATION FOR HUMAN-VERIFIED INTERVAL [264.0, 274.0]s")
    print("=" * 120)
    for k_val in [10, 15, 20, 25, 30, 40]:
        union_frames = set()
        for j in range(len(variants)):
            top_k_rows = np.argsort(-scores[:, j])[:k_val]
            for r in top_k_rows:
                union_frames.add(int(r))
        gt_included = [r for r in gt_rows if r in union_frames]
        status = "INCLUDED ✅" if gt_included else "MISSED ❌"
        gt_fids = [f"f{target_store.mappings[r].frame_id}@{target_store.mappings[r].pts_time:.1f}s" for r in gt_included]
        print(f"  • K={k_val:<2} -> Total Frames Nominated: {len(union_frames):>2}/97 | GT Frames: {gt_fids or '[]'} -> {status}")

    # 4. SIMULATE HARD TEMPORAL NMS (GREEDY SUPPRESSION)
    print("\n" + "=" * 120)
    print("4. HARD TEMPORAL NMS — 15 TOTAL CANDIDATES PER VARIANT")
    print("=" * 120)
    for gap_sec in [3.0, 5.0, 10.0]:
        print(f"\n--- Temporal Gap = {gap_sec:.1f}s ---")
        for j in range(len(variants)):
            v = variants[j]
            sorted_rows = np.argsort(-scores[:, j])
            selected_rows = []
            selected_pts = []
            for r in sorted_rows:
                pts = float(target_store.mappings[int(r)].pts_time)
                if not any(abs(pts - p) < gap_sec for p in selected_pts):
                    selected_rows.append(int(r))
                    selected_pts.append(pts)
                if len(selected_rows) >= 15:
                    break
            gt_hit = [r for r in selected_rows if r in gt_rows]
            status = "INCLUDED ✅" if gt_hit else "MISSED ❌"
            gt_fids = [f"f{target_store.mappings[r].frame_id}@{target_store.mappings[r].pts_time:.1f}s" for r in gt_hit]
            print(f"  • Variant [{v['id'][:28]}]: {len(selected_rows)} candidates -> GT hit: {gt_fids or '[]'} ({status})")

    # 5. SIMULATE HYBRID CANDIDATE POLICY (RAW TOP-10 + TOP DIVERSE FRAMES + TAIL-FILL)
    print("\n" + "=" * 120)
    print("5. EQUAL-BUDGET HYBRID CANDIDATE POLICY (RAW TOP-10 + DIVERSE SLOTS + TAIL-FILL)")
    print("=" * 120)
    for total_budget in [15, 20, 25, 30]:
        diverse_budget = total_budget - 10
        print(f"\n--- Total Budget K={total_budget} (Raw Top-10 + up to {diverse_budget} Diverse Slots with 5.0s Gap + Raw Tail-Fill) ---")
        union_frames = set()
        for j in range(len(variants)):
            v = variants[j]
            sorted_rows = [int(r) for r in np.argsort(-scores[:, j])]
            raw_top10 = sorted_rows[:min(10, len(sorted_rows))]
            selected_pts = [float(target_store.mappings[r].pts_time) for r in raw_top10]
            selected_rows = list(raw_top10)

            for r in sorted_rows[10:]:
                if len(selected_rows) >= total_budget:
                    break
                pts = float(target_store.mappings[r].pts_time)
                if not any(abs(pts - p) < 5.0 for p in selected_pts):
                    selected_rows.append(r)
                    selected_pts.append(pts)

            # Equal-budget tail-fill with remaining raw candidates
            if len(selected_rows) < total_budget:
                selected_set = set(selected_rows)
                for r in sorted_rows:
                    if r not in selected_set:
                        selected_rows.append(r)
                        selected_set.add(r)
                    if len(selected_rows) >= total_budget:
                        break

            assert len(selected_rows) == min(total_budget, len(sorted_rows)), f"Budget mismatch: {len(selected_rows)} != {total_budget}"

            for r in selected_rows:
                union_frames.add(r)

            print(f"    - {v['id'][:28]}: {len(selected_rows)}/{total_budget} candidates allocated")

        gt_included = [r for r in gt_rows if r in union_frames]
        status = "INCLUDED ✅" if gt_included else "MISSED ❌"
        gt_fids = [f"f{target_store.mappings[r].frame_id}@{target_store.mappings[r].pts_time:.1f}s" for r in gt_included]
        print(f"  • Budget K={total_budget:<2} -> Total Unique Frames Nominated: {len(union_frames):>2}/97 | GT Frames: {gt_fids or '[]'} -> {status}")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    run_b0_simulation()
