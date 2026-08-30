"""
Phase B0.1: Counterfactual, Temporal Diversity & Exact Mode A/B Simulation
========================================================================
1. Mode A: Baseline Exact 3 English Variants from 0d4bae4 (semantic_01, 02, 03).
2. Mode B: Experimental Probes (Exact Vietnamese, Counterfactual No-Standing, Posture Probes).
3. CLIP Token count and truncation audit.
4. Strict Mode A-only Candidate Depth and Equal-Budget Hybrid Policy simulations.
"""

import os, json, sys, time, shutil, urllib.request
from pathlib import Path
import numpy as np
import torch, clip

REPO = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("/kaggle/working/AIC2026_TeamPTK_SGU")
OUT = Path("/kaggle/working/output/v2a3_foundation_closure") if Path("/kaggle/working").exists() else REPO / "scratch" / "v2a3_foundation_closure"
SRC = REPO / "systems" / "system_tai" / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from system_tai.features.btc_clip_store import VideoFeatureStoreLoader


def check_token_length(text: str) -> tuple[int, bool]:
    try:
        toks = clip.tokenize([text], truncate=False)
        # Count non-zero tokens excluding SOT (49406) and EOT (49407)
        non_zero = int((toks[0] != 0).sum().item())
        return non_zero, False
    except RuntimeError:
        # Exceeds 77 context length
        return 77, True


def run_b0_simulation() -> None:
    print("🔍 1. Resolving target feature store for L30_V046...", flush=True)
    manifest_path = OUT / "feature_manifest.json"
    csv_path = None
    npy_path = None

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next((v for v in manifest.get("videos", []) if v.get("video_id") == "L30_V046"), None)
            if entry:
                csv_path = Path(entry["mapping_csv_path"])
                npy_path = Path(entry["clip_npy_path"])
        except Exception:
            pass

    if csv_path is None or not csv_path.is_file():
        print("  • Searching /kaggle/input for L30_V046 CSV & NPY...", flush=True)
        input_roots = [Path("/kaggle/input/datasets"), Path("/kaggle/input"), REPO / "scratch"]
        for base in input_roots:
            if not base.exists():
                continue
            for root, dirs, files in os.walk(str(base)):
                dirs[:] = [d for d in dirs if d.lower() not in ("keyframes", "frames", "videos", "video", "raw_videos", "images", "media")]
                for f in files:
                    if f == "L30_V046.csv" and csv_path is None:
                        csv_path = Path(root) / f
                    elif f == "L30_V046.npy" and npy_path is None:
                        npy_path = Path(root) / f
                if csv_path and npy_path and csv_path.is_file() and npy_path.is_file():
                    break
            if csv_path and npy_path and csv_path.is_file() and npy_path.is_file():
                break

        assert csv_path and csv_path.is_file(), f"Không tìm thấy L30_V046.csv trong {input_roots}"
        assert npy_path and npy_path.is_file(), f"Không tìm thấy L30_V046.npy trong {input_roots}"

    print(f"  • CSV Path: {csv_path}", flush=True)
    print(f"  • NPY Path: {npy_path}", flush=True)

    # 1. MODE A: BASELINE EXACT 3 ENGLISH VARIANTS (PINNED FROM COMMIT 0d4bae4)
    baseline_variants = [
        {
            "id": "semantic_01",
            "text": "The scene shows a group of more than 5 people standing in a row to exercise, performing the movement of both hands touching their toes. In the group, only one person wore glasses and three people wore red hats.",
            "role": "GLOBAL_SUMMARY",
        },
        {
            "id": "semantic_02",
            "text": "A group of more than 5 people line up to exercise, performing the movement of both hands touching their toes",
            "role": "PRIMARY_SCENE_ACTION",
        },
        {
            "id": "semantic_03",
            "text": "In the group, only one person wore glasses and three people wore red hats",
            "role": "SUPPORTING_ATTRIBUTES",
        },
    ]

    # 2. MODE B: EXPERIMENTAL & COUNTERFACTUAL PROBES (EVALUATED INDEPENDENTLY)
    experimental_probes = [
        {
            "id": "vi_exact_experimental",
            "text": "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ.",
            "role": "EXACT_ORIGINAL_VIETNAMESE",
        },
        {
            "id": "cf_full_without_standing",
            "text": "The scene shows a group of more than 5 people in a row to exercise, performing the movement of both hands touching their toes. In the group, only one person wore glasses and three people wore red hats.",
            "role": "COUNTERFACTUAL_NO_STANDING",
        },
        {
            "id": "cf_symmetric_standing",
            "text": "A group of people standing in a row doing toe touch exercise",
            "role": "SYMMETRIC_PROBE_STANDING",
        },
        {
            "id": "cf_symmetric_seated",
            "text": "A group of people sitting in a row doing toe touch exercise",
            "role": "SYMMETRIC_PROBE_SEATED",
        },
        {
            "id": "cf_oracle_post_hoc",
            "text": "A group of people sitting on the ground bending forward touching feet",
            "role": "POST_HOC_ORACLE_PROBE",
        },
    ]

    all_prompts = baseline_variants + experimental_probes

    print("=" * 130)
    print("🔬 PHASE B0.1: OFFLINE COUNTERFACTUAL & TEMPORAL DIVERSITY SIMULATION (L30_V046)")
    print("• Provenance: PINNED_EXACT_FROM_0D4BAE4 (3 Baseline English Variants + 5 Decoupled Probes)")
    print("=" * 130)

    # Fast single-store loading
    target_store = VideoFeatureStoreLoader(expected_dimension=512, memory_map=True).load(
        video_id="L30_V046",
        mapping_csv_path=csv_path,
        clip_npy_path=npy_path,
    )

    matrix = np.asarray(target_store.matrix, dtype=np.float32)
    if not target_store.descriptor.normalized:
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    print("⏳ Loading OpenAI CLIP (ViT-B/32)...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Robust model loading / download with retries
    cache_dir = Path.home() / ".cache" / "clip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_file = cache_dir / "ViT-B-32.pt"

    if not target_file.exists() or target_file.stat().st_size < 100_000_000:
        for p in Path("/kaggle/input").glob("**/ViT-B-32.pt"):
            if p.is_file() and p.stat().st_size > 100_000_000:
                print(f"  • Found pre-cached CLIP model in dataset: {p}", flush=True)
                shutil.copy(p, target_file)
                break

    if not target_file.exists() or target_file.stat().st_size < 100_000_000:
        url = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
        print(f"  • Downloading ViT-B-32.pt (~338MB) to {target_file}...", flush=True)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        req = urllib.request.Request(url, headers=headers)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp, open(target_file, "wb") as f:
                    shutil.copyfileobj(resp, f)
                if target_file.stat().st_size > 100_000_000:
                    print("  • Download complete! ✅", flush=True)
                    break
            except Exception as e:
                print(f"  ⚠️ Download attempt {attempt+1}/5 failed ({e}). Retrying in 3s...", flush=True)
                time.sleep(3)

    model_source = str(target_file) if target_file.exists() and target_file.stat().st_size > 100_000_000 else "ViT-B/32"
    model, _ = clip.load(model_source, device=device)
    print(f"✅ OpenAI CLIP Loaded successfully on {device.upper()}!\n", flush=True)

    texts = [v["text"] for v in all_prompts]
    with torch.no_grad():
        toks = clip.tokenize(texts, truncate=True).to(device)
        q_vecs = model.encode_text(toks).float()
        q_vecs /= q_vecs.norm(dim=-1, keepdim=True)
        q_vecs = q_vecs.cpu().numpy().astype(np.float32)

    scores = matrix @ q_vecs.T  # shape (97, num_variants)
    gt_rows = [i for i, m in enumerate(target_store.mappings) if 264.0 <= float(m.pts_time) <= 274.0]
    assert gt_rows, "Không có keyframe trong HUMAN-VERIFIED INTERVAL [264,274]s"

    # =========================================================================
    # SECTION 1: SCORE & LOCAL RANK AUDIT (WITH TOKEN DIAGNOSTICS)
    # =========================================================================
    print("=" * 130)
    print("1. PROMPT SCORE, LOCAL RANK & TOKEN DIAGNOSTICS ON L30_V046")
    print("=" * 130)
    print(f"{'Variant ID':<24} | {'Role / Category':<24} | {'Toks':<5} | {'Trunc?':<7} | {'f6784 Score':<12} | {'f6784 Rank':<11} | {'In Top-10?'}")
    print(f"{'-'*24} | {'-'*24} | {'-'*5} | {'-'*7} | {'-'*12} | {'-'*11} | {'-'*10}")

    for j, v in enumerate(all_prompts):
        tok_cnt, was_trunc = check_token_length(v["text"])
        f6784_row = next(r for r in gt_rows if target_store.mappings[r].frame_id == 6784)
        sc = float(scores[f6784_row, j])
        loc_rank = 1 + int(np.count_nonzero(scores[:, j] > sc))
        in_t10 = "YES ✅" if loc_rank <= 10 else "NO ❌"
        trunc_str = "YES ⚠️" if was_trunc else "NO"
        print(f"{v['id']:<24} | {v['role']:<24} | {tok_cnt:<5} | {trunc_str:<7} | {sc:<12.4f} | {loc_rank:>2}/97{'':<6} | {in_t10}")

    # Print ranks for all 4 keyframes in human interval
    print("\n• Detailed Local Ranks for All 4 Human-Verified Interval Keyframes [264.0, 274.0]s:")
    print(f"{'Keyframe':<16} | {'semantic_01':<14} | {'semantic_02':<14} | {'semantic_03':<14} | {'vi_exact_experimental':<24}")
    print(f"{'-'*16} | {'-'*14} | {'-'*14} | {'-'*14} | {'-'*24}")
    for r in gt_rows:
        m = target_store.mappings[r]
        r_01 = 1 + int(np.count_nonzero(scores[:, 0] > scores[r, 0]))
        r_02 = 1 + int(np.count_nonzero(scores[:, 1] > scores[r, 1]))
        r_03 = 1 + int(np.count_nonzero(scores[:, 2] > scores[r, 2]))
        r_vi = 1 + int(np.count_nonzero(scores[:, 3] > scores[r, 3]))
        print(f"f{m.frame_id}@{m.pts_time:.1f}s{'':<5} | #{r_01:<13} | #{r_02:<13} | #{r_03:<13} | #{r_vi:<23}")

    # =========================================================================
    # SECTION 2: COMPLETE TOP-30 LOCAL TIMESTAMPS OF BASELINE EXACT VARIANTS
    # =========================================================================
    print("\n" + "=" * 130)
    print("2. COMPLETE TOP-30 LOCAL TIMESTAMPS (MODE A: BASELINE EXACT 3 ENGLISH VARIANTS)")
    print("=" * 130)
    for j in range(3):
        v = baseline_variants[j]
        sorted_rows = np.argsort(-scores[:, j])[:30]
        print(f"\n• Baseline Variant: [{v['id']}] (role={v['role']})")
        row_strs = []
        for rk, r in enumerate(sorted_rows, start=1):
            m = target_store.mappings[int(r)]
            in_gt = "🎯[HUMAN_INTERVAL]" if 264.0 <= float(m.pts_time) <= 274.0 else ""
            row_strs.append(f"#{rk:02d}: f{m.frame_id}@{m.pts_time:.1f}s({scores[r, j]:.3f}) {in_gt}")
        for col_idx in range(10):
            line = f"  {row_strs[col_idx]:<34} | {row_strs[col_idx+10]:<34} | {row_strs[col_idx+20]:<34}"
            print(line)

    # Diagnostic Top-30 for Exact Vietnamese Probe
    print(f"\n• Diagnostic Probe: [vi_exact_experimental] (role=EXACT_ORIGINAL_VIETNAMESE)")
    sorted_rows_vi = np.argsort(-scores[:, 3])[:30]
    row_strs_vi = []
    for rk, r in enumerate(sorted_rows_vi, start=1):
        m = target_store.mappings[int(r)]
        in_gt = "🎯[HUMAN_INTERVAL]" if 264.0 <= float(m.pts_time) <= 274.0 else ""
        row_strs_vi.append(f"#{rk:02d}: f{m.frame_id}@{m.pts_time:.1f}s({scores[r, 3]:.3f}) {in_gt}")
    for col_idx in range(10):
        line = f"  {row_strs_vi[col_idx]:<34} | {row_strs_vi[col_idx+10]:<34} | {row_strs_vi[col_idx+20]:<34}"
        print(line)

    # =========================================================================
    # SECTION 3: CANDIDATE DEPTH K SIMULATION (MODE A BASELINE-ONLY)
    # =========================================================================
    print("\n" + "=" * 130)
    print("3. CANDIDATE DEPTH K SIMULATION (MODE A: BASELINE 3 ENGLISH VARIANTS ONLY)")
    print("=" * 130)
    for k_val in [10, 15, 20, 25, 30, 40]:
        union_frames = set()
        for j in range(3):  # baseline 3 only
            top_k_rows = np.argsort(-scores[:, j])[:k_val]
            for r in top_k_rows:
                union_frames.add(int(r))
        gt_included = [r for r in gt_rows if r in union_frames]
        status = "INCLUDED ✅" if gt_included else "MISSED ❌"
        gt_fids = [f"f{target_store.mappings[r].frame_id}@{target_store.mappings[r].pts_time:.1f}s" for r in gt_included]
        print(f"  • Baseline K={k_val:<2} -> Total Unique Frames: {len(union_frames):>2}/97 | GT Frames Captured: {gt_fids or '[]'} -> {status}")

    print("\n--- Mode A + Experimental Exact Vietnamese Probe (Ablation Comparison) ---")
    for k_val in [10, 15, 20, 25, 30]:
        union_frames_vi = set()
        for j in range(4):  # 3 baseline + 1 vi_exact
            top_k_rows = np.argsort(-scores[:, j])[:k_val]
            for r in top_k_rows:
                union_frames_vi.add(int(r))
        gt_included = [r for r in gt_rows if r in union_frames_vi]
        status = "INCLUDED ✅" if gt_included else "MISSED ❌"
        gt_fids = [f"f{target_store.mappings[r].frame_id}@{target_store.mappings[r].pts_time:.1f}s" for r in gt_included]
        print(f"  • Baseline+VI K={k_val:<2} -> Total Unique Frames: {len(union_frames_vi):>2}/97 | GT Frames Captured: {gt_fids or '[]'} -> {status}")

    # =========================================================================
    # SECTION 4: HARD TEMPORAL NMS (MODE A BASELINE & VI PROBE)
    # =========================================================================
    print("\n" + "=" * 130)
    print("4. HARD TEMPORAL NMS — 15 CANDIDATES PER VARIANT")
    print("=" * 130)
    for gap_sec in [3.0, 5.0, 10.0]:
        print(f"\n--- Temporal Gap = {gap_sec:.1f}s ---")
        for j in range(4):  # 3 baseline + 1 vi_exact
            v = all_prompts[j]
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
            print(f"  • Variant [{v['id']:<22}]: {len(selected_rows)} candidates -> GT hit: {gt_fids or '[]'} ({status})")

    # =========================================================================
    # SECTION 5: EQUAL-BUDGET HYBRID CANDIDATE POLICY (MODE A BASELINE ONLY)
    # =========================================================================
    print("\n" + "=" * 130)
    print("5. EQUAL-BUDGET HYBRID CANDIDATE POLICY (MODE A: BASELINE 3 ENGLISH VARIANTS ONLY)")
    print("=" * 130)
    for total_budget in [15, 20, 25, 30]:
        diverse_budget = total_budget - 10
        print(f"\n--- Total Budget K={total_budget} (Raw Top-10 + up to {diverse_budget} Diverse Slots with 5.0s Gap + Raw Tail-Fill) ---")
        union_frames = set()

        for j in range(3):  # Baseline 3 English variants ONLY
            v = baseline_variants[j]
            sorted_rows = [int(r) for r in np.argsort(-scores[:, j])]
            raw_top10 = sorted_rows[:min(10, len(sorted_rows))]
            selected_pts = [float(target_store.mappings[r].pts_time) for r in raw_top10]

            diverse_slots = []
            for r in sorted_rows[10:]:
                if len(diverse_slots) >= diverse_budget:
                    break
                pts = float(target_store.mappings[r].pts_time)
                if not any(abs(pts - p) < 5.0 for p in selected_pts):
                    diverse_slots.append(r)
                    selected_pts.append(pts)

            selected_rows = raw_top10 + diverse_slots

            # Tail-fill with remaining raw candidates
            tail_fill_slots = []
            if len(selected_rows) < total_budget:
                selected_set = set(selected_rows)
                for r in sorted_rows:
                    if r not in selected_set:
                        tail_fill_slots.append(r)
                        selected_set.add(r)
                    if len(selected_rows) + len(tail_fill_slots) >= total_budget:
                        break

            final_selected = selected_rows + tail_fill_slots
            assert len(final_selected) == min(total_budget, len(sorted_rows))

            for r in final_selected:
                union_frames.add(r)

            # Check which bucket captured GT frames
            raw_gt = [r for r in raw_top10 if r in gt_rows]
            div_gt = [r for r in diverse_slots if r in gt_rows]
            tail_gt = [r for r in tail_fill_slots if r in gt_rows]

            gt_notes = []
            if raw_gt:
                gt_notes.append(f"RAW: {[target_store.mappings[r].frame_id for r in raw_gt]}")
            if div_gt:
                gt_notes.append(f"DIVERSE: {[target_store.mappings[r].frame_id for r in div_gt]}")
            if tail_gt:
                gt_notes.append(f"TAIL: {[target_store.mappings[r].frame_id for r in tail_gt]}")

            gt_capture_str = f"🎯 GT in {', '.join(gt_notes)}" if gt_notes else "❌ No GT"
            print(f"    - {v['id']:<14}: {len(raw_top10)} RAW + {len(diverse_slots)} DIVERSE + {len(tail_fill_slots)} TAIL = {len(final_selected)}/{total_budget} | {gt_capture_str}")

        gt_included = [r for r in gt_rows if r in union_frames]
        status = "INCLUDED ✅" if gt_included else "MISSED ❌"
        gt_fids = [f"f{target_store.mappings[r].frame_id}@{target_store.mappings[r].pts_time:.1f}s" for r in gt_included]
        print(f"  • Mode A Baseline Union (K={total_budget}) -> Total Unique Frames: {len(union_frames):>2}/97 | GT Captured: {gt_fids or '[]'} -> {status}")

    print("\n" + "=" * 130)


if __name__ == "__main__":
    run_b0_simulation()
