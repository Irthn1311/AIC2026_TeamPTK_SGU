# ==============================================================================================================
# Phase R3-S1A.1: True Champion-Novelty Audit
# ==============================================================================================================

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig
from system_tai.qa.grounding import nominate_qa_videos
from system_tai.retrieval.multi_query import QueryVariant
from system_tai.retrieval.multi_variant_fusion import fuse_multi_variant_video_ranks
from system_tai.retrieval.query_decomposition import QueryVariants, decompose_query
from system_tai.retrieval.vector_search import ExactNumpyRetriever


def run_novelty_audit(
    benchmark_path: Path,
    dev_en_sidecar_path: Path,
    manifest_cache_path: Path,
    input_root: Path = Path("/kaggle/input"),
    device: str = "auto",
):
    print("=" * 115)
    print("ROUND-3 SPRINT 1A.1: TRUE CHAMPION-NOVELTY AUDIT (ACTUAL CHAMPION NOMINATION POOL vs. EXPANSION)")
    print("=" * 115)

    with open(benchmark_path, encoding="utf-8") as f:
        bm_data = json.load(f)

    with open(dev_en_sidecar_path, encoding="utf-8") as f:
        en_sidecar = json.load(f)

    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}
    qa_dev_queries = [q for q in bm_data["queries"] if q.get("task_type") == "qa" and q.get("split") == "DEV"]

    # 1. Bootstrap Runtime
    print("\nBootstrapping Runtime...")
    session_output = Path("/kaggle/working/output/r3_novelty_audit") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "r3_novelty_audit"
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
    retriever: ExactNumpyRetriever = runtime.exact_retriever
    text_encoder = runtime.shared_encoder
    searcher = runtime.video_restricted_searcher

    seven_gains = ["QA-01", "QA-08", "QA-11", "QA-26", "QA-42", "QA-44", "QA-48"]

    print("\nAuditing all 38 DEV queries with focus on 7 Top16 Gains...")

    audit_records = []
    true_novel_count = 0

    for q in qa_dev_queries:
        qid = q["query_id"]
        target_vid = q.get("video_id")
        q_vi = q.get("question_vi", "")
        q_en = en_map.get(qid, "")

        # 1. Actual Champion Nomination Pool
        # In Champion R2G1, question text is translated EN, search maxima, nominate top 16 videos
        champ_text = q_en if q_en else q_vi
        champ_vec = text_encoder.encode(champ_text)
        champ_var = QueryVariant(variant_id=f"{qid}:champ", text=champ_text, weight=1.0)
        maxima = searcher.search_video_maxima(query_ids=[champ_var.variant_id], query_vectors=[champ_vec])
        champ_noms = nominate_qa_videos(variants=[champ_var], maxima=maxima, config=runtime.qa_pipeline.video_conditioned_evidence_config)
        champ_pool = [item.video_id for item in champ_noms]

        # 2. Multi-Variant Expanded Top 16
        variants = decompose_query(q_vi, q_en)
        v_list = variants.as_list()

        channel_video_rankings = {}
        for v_name, v_text in v_list:
            v_vec = text_encoder.encode(v_text)
            res = retriever.search_vector(query_id=f"{qid}:{v_name}", query_vector=v_vec, top_k=500)
            v_vids = []
            v_tuples = []
            for c in res.ranked_candidates:
                if c.video_id not in v_vids:
                    v_vids.append(c.video_id)
                    v_tuples.append((c.video_id, c.frame_id, float(c.score)))
            channel_video_rankings[f"clip:{v_name}"] = v_tuples

        # Baseline list
        base_res = retriever.search_vector(query_id=qid, query_vector=champ_vec, top_k=500)
        base_vids = []
        for c in base_res.ranked_candidates:
            if c.video_id not in base_vids:
                base_vids.append(c.video_id)

        fused_res = fuse_multi_variant_video_ranks(
            query_id=qid,
            variants=variants,
            channel_video_rankings=channel_video_rankings,
            baseline_top_video_ids=base_vids[:16],
            rrf_k=60.0,
        )

        fused_vids = [v.video_id for v in fused_res.ranked_videos]
        exp_target_rank = (fused_vids.index(target_vid) + 1) if target_vid in fused_vids else None

        target_in_champ_pool = (target_vid in champ_pool)
        target_in_exp_top16 = (exp_target_rank is not None and exp_target_rank <= 16)
        is_true_novel = (target_in_exp_top16 and not target_in_champ_pool)

        if is_true_novel:
            true_novel_count += 1

        target_cand = next((v for v in fused_res.ranked_videos if v.video_id == target_vid), None)
        best_contrib = min(target_cand.contributions, key=lambda c: c.video_rank) if (target_cand and target_cand.contributions) else None

        audit_records.append({
            "qid": qid,
            "target": target_vid,
            "champ_pool": champ_pool[:8],
            "champ_pool_len": len(champ_pool),
            "target_in_champ": target_in_champ_pool,
            "exp_rank": exp_target_rank,
            "exp_top16": target_in_exp_top16,
            "true_novel": is_true_novel,
            "winning_variant": best_contrib.variant_name if best_contrib else "N/A",
            "variant_rank": best_contrib.video_rank if best_contrib else "N/A",
        })

    # Print Report for the 7 Focus Gain Queries
    print("\n" + "=" * 115)
    print("AUDIT REPORT: 7 APPARENT TOP-16 GAINS vs. ACTUAL CHAMPION CANDIDATE POOL")
    print("=" * 115)
    print(f"{'Query ID':<8} | {'Target':<10} | {'Exp Rank':<10} | {'In Champ Pool?':<16} | {'True Novel Rescue?':<20} | {'Winning Variant'}")
    print("-" * 105)
    for r in audit_records:
        if r["qid"] in seven_gains:
            in_cp_str = "YES (ALREADY IN)" if r["target_in_champ"] else "NO (MISSED)"
            novel_str = "TRUE RESCUE 🌟" if r["true_novel"] else "NO (NOT NOVEL)"
            print(f"{r['qid']:<8} | {r['target']:<10} | {str(r['exp_rank']):<10} | {in_cp_str:<16} | {novel_str:<20} | {r['winning_variant']} (Rank {r['variant_rank']})")
    print("=" * 115)

    print(f"\nTotal True Novel Target Videos Recovered into Top 16: {true_novel_count} / 38 queries")
    print("=" * 115)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Champion Novelty Audit")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json")
    parser.add_argument("--sidecar", type=Path, default=REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json")
    parser.add_argument("--manifest-cache", type=Path, default=Path("/kaggle/working/manifest_cache.json"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_novelty_audit(
        benchmark_path=args.benchmark,
        dev_en_sidecar_path=args.sidecar,
        manifest_cache_path=args.manifest_cache,
        input_root=args.input_root,
        device=args.device,
    )
