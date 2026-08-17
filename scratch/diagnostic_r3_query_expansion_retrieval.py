# ==============================================================================================================
# Phase R3-S1A: Full Retrieval Quality Diagnostic (Baseline vs. Multi-Variant Query Expansion)
# ==============================================================================================================

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.data.corpus_discovery import DiscoveryValidation
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig
from system_tai.retrieval.multi_variant_fusion import fuse_multi_variant_video_ranks
from system_tai.retrieval.query_decomposition import QueryVariants, decompose_query


def run_diagnostic(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    manifest_cache_path: Path,
    input_root: Path = Path("/kaggle/input"),
    device: str = "auto",
):
    print("=" * 115)
    print("ROUND-3 SPRINT 1A: RETRIEVAL QUALITY DIAGNOSTIC (BASELINE vs. MULTI-VARIANT EXPANSION)")
    print("=" * 115)

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)

    with open(dev_en_sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    qa_dev_queries = [q for q in bm_data["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"]
    print(f"Loaded {len(qa_dev_queries)} QA DEV queries.")

    # 1. Bootstrap Runtime for retrieval
    print("\nBootstrapping CLIP Vector Retrieval Runtime...")
    session_output = Path("/kaggle/working/output/r3_diagnostic_runtime") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "r3_diag_runtime"
    session_output.mkdir(parents=True, exist_ok=True)

    config = SessionConfig(
        input_root=input_root,
        manifest_cache=manifest_cache_path,
        output_root=session_output,
        device=device,
        allow_model_download=True,
        default_output_top_k=100,
    )
    runtime = OperationalKISRuntime.bootstrap(config)
    retriever = runtime.retriever
    text_encoder = runtime.text_encoder

    print(f"Runtime bootstrapped successfully. Index embedding dim: {retriever.registry.embedding_dimension}")

    # 2. Track variant distributions
    variant_type_counts: Counter[str] = Counter()

    # 3. Evaluate Baseline vs. Expanded Retrieval for all 38 queries
    baseline_r8_count = 0
    baseline_r16_count = 0
    baseline_r32_count = 0

    expanded_r8_count = 0
    expanded_r16_count = 0
    expanded_r32_count = 0

    gained_top8: list[dict] = []
    gained_top16: list[dict] = []
    gained_top32: list[dict] = []

    lost_top8: list[dict] = []
    lost_top16: list[dict] = []
    lost_top32: list[dict] = []

    query_results = []

    print("\nExecuting comparative retrieval across 38 DEV queries...")
    t0 = time.time()

    for idx, q in enumerate(qa_dev_queries, start=1):
        qid = q["query_id"]
        target_vid = q.get("video_id")
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")

        # A. Baseline Retrieval (Literal English only)
        base_vec = text_encoder.encode(q_en if q_en else q_vi)
        base_res = retriever.search_vector(query_id=qid, query_vector=base_vec, top_k=500)

        # Aggregate baseline frames to ranked video list
        base_vid_ranks: list[str] = []
        base_vid_best_frame: dict[str, int] = {}
        for c in base_res.candidates:
            if c.video_id not in base_vid_ranks:
                base_vid_ranks.append(c.video_id)
                base_vid_best_frame[c.video_id] = c.frame_id

        base_target_rank = (base_vid_ranks.index(target_vid) + 1) if target_vid in base_vid_ranks else None

        if base_target_rank and base_target_rank <= 8: baseline_r8_count += 1
        if base_target_rank and base_target_rank <= 16: baseline_r16_count += 1
        if base_target_rank and base_target_rank <= 32: baseline_r32_count += 1

        # B. Multi-Variant Query Expansion Retrieval
        variants = decompose_query(q_vi, q_en)
        v_list = variants.as_list()

        channel_video_rankings: dict[str, list[tuple[str, int, float]]] = {}
        for v_name, v_text in v_list:
            variant_type_counts[v_name] += 1
            v_vec = text_encoder.encode(v_text)
            v_res = retriever.search_vector(query_id=f"{qid}:{v_name}", query_vector=v_vec, top_k=500)

            # Aggregate per-variant frames to video list
            v_vids: list[str] = []
            v_tuples: list[tuple[str, int, float]] = []
            for c in v_res.candidates:
                if c.video_id not in v_vids:
                    v_vids.append(c.video_id)
                    v_tuples.append((c.video_id, c.frame_id, float(c.score)))
            channel_video_rankings[f"clip:{v_name}"] = v_tuples

        fused_res = fuse_multi_variant_video_ranks(
            query_id=qid,
            variants=variants,
            channel_video_rankings=channel_video_rankings,
            baseline_top_video_ids=base_vid_ranks[:16],
            rrf_k=60.0,
        )

        fused_vids = [v.video_id for v in fused_res.ranked_videos]
        exp_target_rank = (fused_vids.index(target_vid) + 1) if target_vid in fused_vids else None

        if exp_target_rank and exp_target_rank <= 8: expanded_r8_count += 1
        if exp_target_rank and exp_target_rank <= 16: expanded_r16_count += 1
        if exp_target_rank and exp_target_rank <= 32: expanded_r32_count += 1

        # Check Novelty
        is_novel_rescue = (
            exp_target_rank is not None
            and exp_target_rank <= 16
            and (base_target_rank is None or base_target_rank > 16)
        )

        # Provenance for target video
        target_cand = next((v for v in fused_res.ranked_videos if v.video_id == target_vid), None)
        best_contrib = None
        if target_cand and target_cand.contributions:
            best_contrib = min(target_cand.contributions, key=lambda c: c.video_rank)

        # Classify delta
        delta_str = "SAME"
        if base_target_rank != exp_target_rank:
            if base_target_rank is None:
                delta_str = f"GAINED (MISS -> {exp_target_rank})"
            elif exp_target_rank is None:
                delta_str = f"LOST ({base_target_rank} -> MISS)"
            elif exp_target_rank < base_target_rank:
                delta_str = f"IMPROVED ({base_target_rank} -> {exp_target_rank})"
            else:
                delta_str = f"REGRESSED ({base_target_rank} -> {exp_target_rank})"

        # Gained / Lost tracking
        if (not base_target_rank or base_target_rank > 16) and (exp_target_rank and exp_target_rank <= 16):
            gained_top16.append({
                "qid": qid, "target": target_vid, "base_rank": base_target_rank,
                "exp_rank": exp_target_rank, "best_variant": best_contrib.variant_name if best_contrib else "N/A",
                "variant_rank": best_contrib.video_rank if best_contrib else "N/A",
            })
        if (base_target_rank and base_target_rank <= 16) and (not exp_target_rank or exp_target_rank > 16):
            lost_top16.append({
                "qid": qid, "target": target_vid, "base_rank": base_target_rank, "exp_rank": exp_target_rank,
            })

        query_results.append({
            "qid": qid,
            "target_vid": target_vid,
            "base_rank": base_target_rank,
            "exp_rank": exp_target_rank,
            "delta": delta_str,
            "is_novel": is_novel_rescue,
            "best_variant": best_contrib.variant_name if best_contrib else "N/A",
            "variant_rank": best_contrib.video_rank if best_contrib else "N/A",
        })

    elapsed = time.time() - t0
    print(f"Diagnostic completed in {elapsed:.2f}s.")

    # 4. Print Summary Report
    print("\n" + "=" * 115)
    print("RETRIEVAL RECALL COMPARISON (N = 38 DEV QUERIES)")
    print("=" * 115)
    print(f"{'Metric':<20} | {'Baseline Single-Query':<25} | {'Multi-Variant Expansion':<25} | {'Delta':<15}")
    print("-" * 90)
    print(f"{'Video Recall @8':<20} | {f'{baseline_r8_count}/38 ({baseline_r8_count/38*100:.1f}%)':<25} | {f'{expanded_r8_count}/38 ({expanded_r8_count/38*100:.1f}%)':<25} | {f'{expanded_r8_count - baseline_r8_count:+d}':<15}")
    print(f"{'Video Recall @16':<20} | {f'{baseline_r16_count}/38 ({baseline_r16_count/38*100:.1f}%)':<25} | {f'{expanded_r16_count}/38 ({expanded_r16_count/38*100:.1f}%)':<25} | {f'{expanded_r16_count - baseline_r16_count:+d}':<15}")
    print(f"{'Video Recall @32':<20} | {f'{baseline_r32_count}/38 ({baseline_r32_count/38*100:.1f}%)':<25} | {f'{expanded_r32_count}/38 ({expanded_r32_count/38*100:.1f}%)':<25} | {f'{expanded_r32_count - baseline_r32_count:+d}':<15}")
    print("=" * 115)

    print("\n--- VARIANT TYPE DISTRIBUTION ACROSS 38 QUERIES ---")
    for v_name, count in variant_type_counts.most_common():
        print(f"  - {v_name:<20}: {count} queries ({count/38*100:.1f}%)")

    print("\n--- TARGET VIDEOS GAINED INTO TOP 16 ---")
    if gained_top16:
        for g in gained_top16:
            print(f"  🎯 {g['qid']:<8} Target: {g['target']} | Base: {g['base_rank']} -> Expanded: {g['exp_rank']} | Winning Variant: {g['best_variant']} (Rank {g['variant_rank']})")
    else:
        print("  None gained.")

    print("\n--- TARGET VIDEOS LOST FROM TOP 16 ---")
    if lost_top16:
        for l in lost_top16:
            print(f"  ⚠️ {l['qid']:<8} Target: {l['target']} | Base: {l['base_rank']} -> Expanded: {l['exp_rank']}")
    else:
        print("  None lost (Zero regression on Top 16 baseline targets! ✅)")

    print("\n--- PER-QUERY DETAILED RETRIEVAL COMPARISON ---")
    print(f"{'Query ID':<8} | {'Target':<10} | {'Base Rank':<12} | {'Exp Rank':<12} | {'Novel Rescue?':<15} | {'Winning Variant':<18} | {'Status'}")
    print("-" * 105)
    for r in query_results:
        novel_mark = "YES 🌟" if r["is_novel"] else "no"
        print(f"{r['qid']:<8} | {r['target_vid']:<10} | {str(r['base_rank']):<12} | {str(r['exp_rank']):<12} | {novel_mark:<15} | {r['best_variant']:<18} | {r['delta']}")
    print("=" * 115)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Retrieval Quality Diagnostic")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--manifest-cache", type=Path, default=Path("/kaggle/working/manifest_cache.json"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_diagnostic(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        manifest_cache_path=args.manifest_cache,
        input_root=args.input_root,
        device=args.device,
    )
