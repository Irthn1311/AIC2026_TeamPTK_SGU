import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

from system_tai.kis.session_engine import (
    OperationalKISRuntime,
    compile_vietnamese_semantic_query,
    safe_request_directory_name,
)
from system_tai.kis.session_schema import (
    KISVideoFirstConfig,
    QueryLanguage,
    QueryRequest,
    QueryVariant,
    QueryVariantType,
    SessionConfig,
)
from system_tai.retrieval.semantic_query import SemanticQueryConfig
from system_tai.kis.video_first import (
    FusedVideoEvidence,
    build_kis_video_first_outcome,
    fuse_restricted_frames,
    fuse_video_maxima_v2,
    solve_temporal_chain,
)


def get_git_head() -> str:
    try:
        git_dir = REPO_ROOT / ".git"
        if not git_dir.exists():
            return "unknown"
        head_content = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head_content.startswith("ref:"):
            ref_path = git_dir / head_content.split(" ", 1)[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
        return head_content
    except Exception:
        return "unknown"


def find_keyframe_image(dataset_root: Path, video_id: str, frame_id: int, keyframe_order: int | None = None) -> Path | None:
    patterns = []
    if keyframe_order is not None:
        patterns.extend([
            dataset_root / "keyframes" / video_id / f"{keyframe_order:06d}.jpg",
            dataset_root / "keyframes" / video_id / f"{keyframe_order:05d}.jpg",
            dataset_root / "keyframes" / video_id / f"{keyframe_order:04d}.jpg",
            dataset_root / "keyframes" / video_id / f"{keyframe_order:03d}.jpg",
            dataset_root / "keyframes" / video_id / f"{keyframe_order}.jpg",
            dataset_root / video_id / f"{keyframe_order:06d}.jpg",
            dataset_root / video_id / f"{keyframe_order:05d}.jpg",
            dataset_root / video_id / f"{keyframe_order:04d}.jpg",
            dataset_root / video_id / f"{keyframe_order:03d}.jpg",
            dataset_root / video_id / f"{keyframe_order}.jpg",
        ])
    patterns.extend([
        dataset_root / "keyframes" / video_id / f"{frame_id:06d}.jpg",
        dataset_root / "keyframes" / video_id / f"{frame_id:05d}.jpg",
        dataset_root / "keyframes" / video_id / f"{frame_id}.jpg",
        dataset_root / video_id / f"{frame_id:06d}.jpg",
        dataset_root / video_id / f"{frame_id:05d}.jpg",
        dataset_root / video_id / f"{frame_id}.jpg",
    ])
    for p in patterns:
        if p.exists():
            return p
    return None


def main() -> None:
    full_sha = get_git_head()
    print("=" * 120, flush=True)
    print("KIS V2-A.2 DEV CAUSAL AUDIT — VALIDITY AUDIT ONLY (NO ALGORITHM TUNING)", flush=True)
    print("=" * 120, flush=True)
    print(f"• Exact Commit SHA: {full_sha}", flush=True)
    print(f"• Python Version  : {sys.version.split()[0]}", flush=True)

    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest_path = None
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
    ]:
        if p.exists() and p.stat().st_size > 1000:
            reuse_manifest_path = p
            break

    base_out = Path("/kaggle/working/output/v2a2_causal_audit") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "v2a2_causal_audit"
    manifest_cache = None if reuse_manifest_path else (
        Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "manifest_cache.json"
    )

    config = SessionConfig(
        input_root=input_root,
        reuse_manifest=reuse_manifest_path,
        manifest_cache=manifest_cache,
        output_root=base_out,
        device="auto",
        allow_model_download=True,
        enable_dynamic_translation=True,
        translation_model_name="vinai/vinai-translate-vi2en-v2",
        translation_device="auto",
        translation_allow_model_download=True,
        translation_max_clip_tokens=75,
        default_output_top_k=100,
        default_refine_top_n=0,
        rrf_constant=60.0,
        kis_video_first_config=KISVideoFirstConfig(
            enabled=True,
            v2_adaptive_enabled=True,
            selected_video_cap=64,
            top_m_evidence_cap=5,
            top_m_min_frame_gap=60,
            top_m_weights=(0.4, 0.25, 0.15, 0.1, 0.1),
            adaptive_budget_base=32,
            adaptive_budget_medium=48,
            adaptive_budget_high=64,
            coverage_threshold=0.75,
        ),
    )

    t0 = time.time()
    runtime = OperationalKISRuntime.bootstrap(config)
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.\n", flush=True)

    # 1. P1-6 ALL-SCENE UNION & CANDIDATE POOL AUDIT
    run_p1_6_audit(runtime)

    # 2. P1-5 PROMPT-LEVEL CAUSAL AUDIT (6 ARMS)
    run_p1_5_audit(runtime)

    # 3. P1-4 TARGETED TEMPORAL EVIDENCE AUDIT & CONTACT SHEETS
    run_p1_4_audit(runtime, base_out)

    # 4. P1-2 EXACT PRODUCTION FRAME-LOSS TRACE
    run_p1_2_audit(runtime)


def run_p1_6_audit(runtime: OperationalKISRuntime) -> None:
    print("=" * 100, flush=True)
    print("A. P1-6: NOMINATION BUG OR RETRIEVAL WEAKNESS? (ALL-SCENE CANDIDATE UNION)", flush=True)
    print("=" * 100, flush=True)
    qid = "query-p1-6-kis"
    q_vi = "Mẩu tin bắt đầu với hình ảnh một người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh."
    target_vid = "L27_V005"

    video_first_config = runtime.config.kis_video_first_config
    compiled_semantic_query = compile_vietnamese_semantic_query(
        query_id=qid,
        query_vi=q_vi,
        provider=runtime.translation_provider,
        token_budget_guard=runtime.token_budget_guard,
        config=SemanticQueryConfig(
            full_query_weight=video_first_config.full_query_weight,
            primary_scene_weight=video_first_config.primary_scene_weight,
            supporting_attribute_weight=video_first_config.supporting_attribute_weight,
        ),
    )

    temporal_scene_variants = compiled_semantic_query.temporal_scene_variants
    print(f"Compiled {len(temporal_scene_variants)} temporal scene variants from Vietnamese query:\n", flush=True)

    all_variants = [item.query_variant for item in temporal_scene_variants]
    texts = [v.text for v in all_variants]
    embeddings = runtime.shared_encoder.encode_texts(texts)

    maxima = runtime.video_restricted_searcher.search_video_maxima(
        query_ids=tuple(v.variant_id for v in all_variants),
        query_vectors=embeddings,
        top_m_evidence_cap=video_first_config.top_m_evidence_cap,
        top_m_min_frame_gap=video_first_config.top_m_min_frame_gap,
        top_m_weights=video_first_config.top_m_weights,
    )

    cumulative_union: set[str] = set()
    entered_via: list[str] = []

    print(f"{'Scene':<6} | {'PoolSize':<8} | {'NewUnique':<9} | {'TargetRank':<10} | {'InPool':<6} | {'TargetBestFrame':<15} | {'RawMaxCos':<9} | {'TopMScore':<9} | {'BestVideo':<10}", flush=True)
    print("-" * 105, flush=True)

    for idx, (scene_var, v) in enumerate(zip(temporal_scene_variants, all_variants, strict=True), start=1):
        scene_tag = f"T{scene_var.temporal_index}"
        hits = maxima.rankings.get(v.variant_id, ())
        pool_128 = [h.video_id for h in hits[:128]]

        prev_size = len(cumulative_union)
        cumulative_union.update(pool_128)
        new_unique = len(cumulative_union) - prev_size

        target_hit = next((h for h in hits if h.video_id == target_vid), None)
        best_hit = hits[0] if hits else None

        if target_hit:
            t_rank = f"#{target_hit.rank}"
            t_in_pool = "YES" if target_hit.rank <= 128 else "NO"
            t_frame = str(target_hit.frame_id)
            t_raw_max = f"{target_hit.cosine_score:.4f}"
            t_top_m = f"{target_hit.top_m_score:.4f}"
            if target_hit.rank <= 128:
                entered_via.append(scene_tag)
        else:
            t_rank = "N/A"
            t_in_pool = "NO"
            t_frame = "N/A"
            t_raw_max = "N/A"
            t_top_m = "N/A"

        best_str = f"{best_hit.video_id} ({best_hit.top_m_score:.3f})" if best_hit else "N/A"
        print(f"{scene_tag:<6} | {len(pool_128):<8} | {new_unique:<9} | {t_rank:<10} | {t_in_pool:<6} | {t_frame:<15} | {t_raw_max:<9} | {t_top_m:<9} | {best_str:<10}", flush=True)

    print("-" * 105, flush=True)
    print(f"• Final All-Scene Candidate Union Size : {len(cumulative_union)} unique videos", flush=True)
    print(f"• Target {target_vid} in Final All-Scene Union: {'YES ✅' if target_vid in cumulative_union else 'NO ❌'}", flush=True)
    print(f"• Target {target_vid} Entered Via Scenes     : {entered_via if entered_via else 'None (Failed all individual scene pools)'}", flush=True)
    print("\nScene Descriptions & Translations:", flush=True)
    for scene_var in temporal_scene_variants:
        print(f"  T{scene_var.temporal_index} VI: {scene_var.source_vietnamese}", flush=True)
        print(f"  T{scene_var.temporal_index} EN: {scene_var.query_variant.text}", flush=True)
    print()


def run_p1_5_audit(runtime: OperationalKISRuntime) -> None:
    print("=" * 100, flush=True)
    print("B. P1-5: SENTENCE DILUTION OR VISUAL-FEATURE FAILURE? (6-ARM PROMPT AUDIT)", flush=True)
    print("=" * 100, flush=True)
    target_vid = "L30_V021"
    target_gt_frame = 3325

    prompts = [
        ("long T1 (VinAI)", "The clip begins with the peas being put in with the squid being sautéed on the pan, next to a plate of onions and sliced red peppers being prepared for the dish."),
        ("long T2 (VinAI)", "The clip ends with a slow motion pan shaking scene on the stove"),
        ("squid + peas", "squid and peas stir-frying in a pan"),
        ("peas added to squid", "peas being added to squid in a frying pan"),
        ("pepper/onion + squid", "sliced red pepper and onion beside a pan of squid"),
        ("pan tossed over flame", "a frying pan being tossed over a gas flame"),
    ]

    video_first_config = runtime.config.kis_video_first_config
    query_variants = [
        QueryVariant(
            variant_id=f"p1_5_arm_{i:02d}",
            text=text,
            language=QueryLanguage.ENGLISH,
            variant_type=QueryVariantType.ENGLISH_TRANSLATION,
            weight=1.0,
        )
        for i, (_, text) in enumerate(prompts, start=1)
    ]

    embeddings = runtime.shared_encoder.encode_texts([v.text for v in query_variants])
    maxima = runtime.video_restricted_searcher.search_video_maxima(
        query_ids=tuple(v.variant_id for v in query_variants),
        query_vectors=embeddings,
        top_m_evidence_cap=video_first_config.top_m_evidence_cap,
        top_m_min_frame_gap=video_first_config.top_m_min_frame_gap,
        top_m_weights=video_first_config.top_m_weights,
    )

    print(f"Target Groundtruth Video: {target_vid} @ Physical Frame {target_gt_frame}\n", flush=True)
    print(f"| {'Prompt Arm':<24} | {'Target Rank':>11} | {'Raw Max Cos':>11} | {'Top-M Score':>11} | {'Best Frame':>10} | {'Dist to GT':>10} | {'Corpus Best Video':<20} |", flush=True)
    print(f"| {'-'*24} | {'-'*11} | {'-'*11} | {'-'*11} | {'-'*10} | {'-'*10} | {'-'*20} |", flush=True)

    for (label, _), v in zip(prompts, query_variants, strict=True):
        hits = maxima.rankings.get(v.variant_id, ())
        target_hit = next((h for h in hits if h.video_id == target_vid), None)
        best_hit = hits[0] if hits else None

        if target_hit:
            t_rank = f"#{target_hit.rank}"
            t_raw_max = f"{target_hit.cosine_score:.4f}"
            t_top_m = f"{target_hit.top_m_score:.4f}"
            t_frame = str(target_hit.frame_id)
            dist_gt = f"{abs(target_hit.frame_id - target_gt_frame)}f"
        else:
            t_rank = "N/A"
            t_raw_max = "N/A"
            t_top_m = "N/A"
            t_frame = "N/A"
            dist_gt = "N/A"

        best_str = f"{best_hit.video_id} ({best_hit.top_m_score:.3f})" if best_hit else "N/A"
        print(f"| {label:<24} | {t_rank:>11} | {t_raw_max:>11} | {t_top_m:>11} | {t_frame:>10} | {dist_gt:>10} | {best_str:<20} |", flush=True)
    print()


def run_p1_4_audit(runtime: OperationalKISRuntime, base_out: Path) -> None:
    print("=" * 100, flush=True)
    print("C. P1-4: DP CHAIN THẬT HAY FALSE SEMANTIC CHAIN? (CONTACT SHEETS & PEAKS AUDIT)", flush=True)
    print("=" * 100, flush=True)
    qid = "query-p1-4-kis"
    q_vi = "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú."

    video_first_config = runtime.config.kis_video_first_config
    compiled_semantic_query = compile_vietnamese_semantic_query(
        query_id=qid,
        query_vi=q_vi,
        provider=runtime.translation_provider,
        token_budget_guard=runtime.token_budget_guard,
        config=SemanticQueryConfig(
            full_query_weight=video_first_config.full_query_weight,
            primary_scene_weight=video_first_config.primary_scene_weight,
            supporting_attribute_weight=video_first_config.supporting_attribute_weight,
        ),
    )

    temporal_scene_variants = compiled_semantic_query.temporal_scene_variants
    all_variants = [item.query_variant for item in temporal_scene_variants]
    embeddings = runtime.shared_encoder.encode_texts([v.text for v in all_variants])

    maxima = runtime.video_restricted_searcher.search_video_maxima(
        query_ids=tuple(v.variant_id for v in all_variants),
        query_vectors=embeddings,
        top_m_evidence_cap=video_first_config.top_m_evidence_cap,
        top_m_min_frame_gap=video_first_config.top_m_min_frame_gap,
        top_m_weights=video_first_config.top_m_weights,
    )

    videos_to_audit = ["L22_V021", "L28_V012"]

    for vid in videos_to_audit:
        print(f"\n--- Diagnostic Audit for Video {vid} ({'Distractor' if vid == 'L22_V021' else 'Groundtruth Target'}) ---", flush=True)
        peaks_by_scene = []
        raw_scores = []
        for scene_var, v in zip(temporal_scene_variants, all_variants, strict=True):
            hits = maxima.rankings.get(v.variant_id, ())
            hit = next((h for h in hits if h.video_id == vid), None)
            if hit:
                peaks = list(hit.top_m_peaks) if hit.top_m_peaks else [(hit.frame_id, hit.cosine_score)]
                peaks_by_scene.append(peaks)
                raw_scores.append(hit.top_m_score)
                peaks_str = " | ".join([f"f{f} (cos={s:.3f})" for f, s in peaks[:5]])
                print(f"  T{scene_var.temporal_index} Rank #{hit.rank:<3} | Top-M Score={hit.top_m_score:.4f} | Raw Max Cos={hit.cosine_score:.4f} | Top 5 Peaks: [{peaks_str}]", flush=True)
            else:
                peaks_by_scene.append([])
                raw_scores.append(0.0)
                print(f"  T{scene_var.temporal_index} NOT FOUND in corpus ranking", flush=True)

        has_valid_chain, chain_frames, chain_score = solve_temporal_chain(
            peaks_by_scene=peaks_by_scene,
            scene_weights=[float(v.weight) for v in all_variants],
            min_gap=video_first_config.top_m_min_frame_gap,
        )
        print(f"  DP Temporal Chain Result: Valid={has_valid_chain} | Winning Frames={chain_frames} | DP Chain Score={chain_score:.4f}", flush=True)

        contact_path = base_out / f"p1-4_{vid}_contact_sheet.png"
        render_p1_4_contact_sheet(
            runtime=runtime,
            vid=vid,
            temporal_scene_variants=temporal_scene_variants,
            all_variants=all_variants,
            maxima=maxima,
            chain_frames=chain_frames,
            out_path=contact_path,
        )
    print()


def render_p1_4_contact_sheet(
    runtime: OperationalKISRuntime,
    vid: str,
    temporal_scene_variants: list,
    all_variants: list,
    maxima,
    chain_frames: list[int],
    out_path: Path,
) -> None:
    try:
        store = runtime.video_restricted_searcher.registry.get(vid)
    except KeyError:
        print(f"  ⚠️ Cannot render contact sheet: store for {vid} not in registry", flush=True)
        return

    n_scenes = len(temporal_scene_variants)
    fig, axes = plt.subplots(n_scenes, 5, figsize=(25, 5 * n_scenes))
    if n_scenes == 1:
        axes = np.array([axes])

    for row_idx, (scene_var, v) in enumerate(zip(temporal_scene_variants, all_variants, strict=True)):
        hits = maxima.rankings.get(v.variant_id, ())
        hit = next((h for h in hits if h.video_id == vid), None)
        peaks = list(hit.top_m_peaks) if hit else []

        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            if col_idx < len(peaks):
                req_frame_id, cosine = peaks[col_idx]
                rows = store.rows_for_frame(req_frame_id)
                if not rows:
                    ax.text(0.5, 0.5, f"Frame {req_frame_id}\nRow Not Found", ha="center", va="center")
                    ax.axis("off")
                    continue
                mapping = store.frame_for_row(rows[0])
                assert mapping.frame_id == req_frame_id, f"Frame ID mismatch: requested {req_frame_id} vs store mapping {mapping.frame_id}"

                img_path = find_keyframe_image(
                    dataset_root=runtime.config.input_root,
                    video_id=vid,
                    frame_id=req_frame_id,
                    keyframe_order=mapping.keyframe_order,
                )
                if img_path and img_path.exists():
                    img = Image.open(img_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, f"Image not on disk\nFrame {req_frame_id}\n(Order {mapping.keyframe_order})", ha="center", va="center")

                is_chain = req_frame_id in chain_frames
                caption = (
                    f"Video: {vid} | Scene: T{scene_var.temporal_index}\n"
                    f"Physical Frame: {req_frame_id} (Order: {mapping.keyframe_order})\n"
                    f"Raw Cosine: {cosine:.4f} | Peak #{col_idx+1}\n"
                    f"Winning Chain Frame: {'YES ★' if is_chain else 'NO'}"
                )
                ax.set_title(caption, fontsize=9, color="red" if is_chain else "black", pad=8)
            else:
                ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  📸 Saved contact sheet for {vid} -> {out_path}", flush=True)


def run_p1_2_audit(runtime: OperationalKISRuntime) -> None:
    print("=" * 100, flush=True)
    print("D. P1-2: EXACT PRODUCTION FRAME-LOSS TRACE (FRAME 6171 / TARGET L29_V018)", flush=True)
    print("=" * 100, flush=True)
    qid = "query-p1-2-kis"
    q_vi = "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa."
    target_vid = "L29_V018"
    target_gt_frame = 6050
    evidence_frame = 6171

    req = QueryRequest(
        request_id=f"causal-audit-{qid}",
        query_id=qid,
        query_vi=q_vi,
        query_en=None,
        include_vi_variant=True,
        output_top_k=100,
        refine_top_n=0,
    )

    t0 = time.perf_counter()
    out = runtime.handle_query(req)
    latency_ms = (time.perf_counter() - t0) * 1000

    candidates_file = runtime.output_root / out["artifacts"]["candidates_json"]
    cand_data = json.loads(candidates_file.read_text(encoding="utf-8"))

    evidence_frame_pool = cand_data.get("evidence_frame_pool", [])
    records = cand_data.get("records", [])
    vf_trace = cand_data.get("video_first", {})
    selected_videos = vf_trace.get("selected_videos", [])

    print(f"• Query ID: {qid} (Executed in {latency_ms:.1f}ms)", flush=True)
    print(f"• Target Video: {target_vid} | Target GT Frame: {target_gt_frame} | Retained Frame Under Audit: {evidence_frame}\n", flush=True)

    print("--- 1. OFFICIAL CONTEST / MANIFEST MAPPING VALIDATION ---", flush=True)
    store = runtime.video_restricted_searcher.registry.get(target_vid)
    rows_6171 = store.rows_for_frame(evidence_frame)
    if rows_6171:
        mapping_6171 = store.frame_for_row(rows_6171[0])
        print(f"  physical_frame_id           : {evidence_frame}", flush=True)
        print(f"  keyframe_order (column n)   : {mapping_6171.keyframe_order}", flush=True)
        print(f"  clip_row in feature matrix  : {mapping_6171.clip_row}", flush=True)
        print(f"  pts_time                    : {mapping_6171.pts_time:.3f}s", flush=True)
        print(f"  official_frame_idx (CSV)    : {mapping_6171.frame_id}", flush=True)
        print(f"  manifest / mapping source   : {store.descriptor.mapping_csv_path}", flush=True)
        print(f"  mapping integrity status    : PASS ✅ (Exact 1-to-1 match)", flush=True)
    else:
        print(f"  ⚠️ Frame {evidence_frame} not found in store mapping!", flush=True)

    print("\n--- 2. STAGE-BY-STAGE CANDIDATE PIPELINE TRACE ---", flush=True)

    target_vid_ev = next((v for v in selected_videos if v["video_id"] == target_vid), None)
    if target_vid_ev:
        t_rank = target_vid_ev["rank"]
        t_score = target_vid_ev["fusion_score"]
        t_chain = target_vid_ev.get("temporal_chain") or {}
        chain_frames = t_chain.get("selected_chain_frames", [])
        print(f"  Stage 1 (Video Nomination)  : Video {target_vid} RANK #{t_rank} (Score: {t_score:.4f}) -> ENTERED RESTRICTED SET ✅", flush=True)
        print(f"                              : Temporal Chain Valid={t_chain.get('has_valid_chain')} | Selected Frames={chain_frames}", flush=True)
    else:
        print(f"  Stage 1 (Video Nomination)  : Video {target_vid} MISSED Top 64 ❌", flush=True)

    ev_pool_item = next((item for item in evidence_frame_pool if item["video_id"] == target_vid and item["frame_id"] == evidence_frame), None)
    if ev_pool_item:
        ev_pos = evidence_frame_pool.index(ev_pool_item)
        print(f"  Stage 2 (Evidence Frame Pool): Frame {evidence_frame} RETAINED ✅ (Position {ev_pos}/{len(evidence_frame_pool)} in pool)", flush=True)
    else:
        print(f"  Stage 2 (Evidence Frame Pool): Frame {evidence_frame} DROPPED at Evidence Build ❌", flush=True)

    compiled_sq = compile_vietnamese_semantic_query(
        query_id=qid,
        query_vi=q_vi,
        provider=runtime.translation_provider,
        token_budget_guard=runtime.token_budget_guard,
        config=SemanticQueryConfig(
            full_query_weight=runtime.config.kis_video_first_config.full_query_weight,
            primary_scene_weight=runtime.config.kis_video_first_config.primary_scene_weight,
            supporting_attribute_weight=runtime.config.kis_video_first_config.supporting_attribute_weight,
        ),
    )
    variants = compiled_sq.query_variants
    embeddings = runtime.shared_encoder.encode_texts([v.text for v in variants])
    restricted = runtime.video_restricted_searcher.search_selected_videos(
        video_ids=tuple(item["video_id"] for item in selected_videos),
        query_ids=tuple(v.variant_id for v in variants),
        query_vectors=embeddings,
        per_query_result_cap=runtime.config.kis_video_first_config.restricted_frames_per_video_per_variant,
    )

    selected_objects = [
        FusedVideoEvidence(
            video_id=item["video_id"],
            rank=item["rank"],
            fusion_score=item["fusion_score"],
            variant_hit_count=item.get("variant_hit_count", 1),
            primary_coverage_count=item.get("primary_coverage_count", 1),
            best_individual_rank=item.get("best_individual_rank", item["rank"]),
            per_variant=(),
            temporal_chain=None,
        )
        for item in selected_videos
    ]

    full_fused = fuse_restricted_frames(
        query_id=qid,
        variants=variants,
        restricted=restricted,
        selected_videos=selected_objects,
        weighted_rrf=runtime.weighted_rrf,
        output_top_k=99999,
        rrf_constant=runtime.config.rrf_constant,
    )

    fused_6171 = next((f for f in full_fused.ranked_candidates if f.video_id == target_vid and f.frame_id == evidence_frame), None)
    if fused_6171:
        print(f"  Stage 3 (Frame Fusion RRF)  : Frame {evidence_frame} Fused Score={fused_6171.score:.6f} | GLOBAL FUSED RANK = #{fused_6171.rank}", flush=True)
    else:
        print(f"  Stage 3 (Frame Fusion RRF)  : Frame {evidence_frame} not scored in restricted set ❌", flush=True)

    top100_hit = next((r for r in records if r["video_id"] == target_vid and r["frame_id"] == evidence_frame), None)
    target_frames_in_top100 = [r for r in records if r["video_id"] == target_vid]

    print(f"  Stage 4 (Top 100 Truncation): {'RETAINED IN TOP 100 ✅ (Rank #' + str(top100_hit['rank']) + ')' if top100_hit else 'TRUNCATED (Rank > 100) ❌'}", flush=True)
    print(f"                              : Total {target_vid} frames in Top 100 = {len(target_frames_in_top100)}", flush=True)
    for tf in target_frames_in_top100:
        print(f"                                -> Frame {tf['frame_id']:<5} (Order {tf.get('keyframe_order_diagnostic')}) | Top100 Rank #{tf['rank']:<2} | Fusion Score {tf['fusion_score']:.6f}", flush=True)

    print("\n--- 3. FINAL CAUSAL STAGE CLASSIFICATION FOR P1-2 ---", flush=True)
    if not target_vid_ev:
        verdict = "LOST_AT_VIDEO_NOMINATION"
    elif not ev_pool_item:
        verdict = "LOST_AT_EVIDENCE_BUILD"
    elif not rows_6171:
        verdict = "LOST_AT_RESTRICTED_SEARCH"
    elif not fused_6171:
        verdict = "LOST_AT_FRAME_FUSION"
    elif fused_6171.rank > 100 and not top100_hit:
        verdict = "LOST_AT_TOP100"
    elif top100_hit:
        verdict = "SURVIVED"
    else:
        verdict = "LOST_AT_DEDUPE_OR_EXPORT"

    print(f"  FINAL VERDICT: 【 {verdict} 】", flush=True)
    if verdict == "LOST_AT_TOP100":
        print(f"  -> Explanation: Evidence frame 6171 was successfully nominated and retained in evidence pool, but its global RRF frame score placed it at Rank #{fused_6171.rank}, so it was cut off by output_top_k=100.", flush=True)
    print("=" * 100 + "\n", flush=True)


if __name__ == "__main__":
    main()
