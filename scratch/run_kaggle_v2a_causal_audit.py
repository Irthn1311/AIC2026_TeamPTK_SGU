import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig
from system_tai.kis.video_first import KISVideoFirstConfig
from system_tai.kis.contest_schema import KISQuery

def get_git_head():
    try:
        head = Path(os.path.join(os.path.dirname(__file__), "..", ".git", "HEAD")).read_text().strip()
        if head.startswith("ref:"):
            ref_path = head.split(" ")[1]
            return Path(os.path.join(os.path.dirname(__file__), "..", ".git", ref_path)).read_text().strip()[:7]
        return head[:7]
    except Exception:
        return "unknown"

def main():
    print("=" * 100)
    print("FINAL V2-A.2 CAUSAL AUDIT — NO ALGORITHM TUNING")
    print(f"Git SHA: {get_git_head()}")
    print("=" * 100)

    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest_path = None
    for p in [Path("/kaggle/working/manifest_cache.json"), Path("/kaggle/input/system-tai-manifest/feature_manifest.json"), Path("/kaggle/input/datasets/manifest_cache.json"), Path("/kaggle/input/manifest_cache.json")]:
        if p.exists() and p.stat().st_size > 1000:
            reuse_manifest_path = p
            break
            
    base_out = Path("/kaggle/working/output/v2a2_causal_audit") if Path("/kaggle/working").exists() else Path(__file__).parent / "v2a2_causal_audit"
    manifest_cache = None if reuse_manifest_path else (Path("/kaggle/working/manifest_cache.json") if Path("/kaggle/working").exists() else Path(__file__).parent / "manifest_cache.json")

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
    print(f"Runtime bootstrap completed in {time.time() - t0:.2f}s.")

    run_p1_6_audit(runtime)
    run_p1_4_audit(runtime, base_out)
    run_p1_2_audit(runtime)
    run_p1_5_audit(runtime)

def run_p1_6_audit(runtime: OperationalKISRuntime):
    print("\n" + "="*80)
    print("1 & 2. VERIFY ALL-SCENE CANDIDATE UNION & OFFLINE TARGET AUDIT: P1-6")
    print("="*80)
    q = KISQuery(
        query_id="query-p1-6-kis",
        source_vietnamese="Mẩu tin bắt đầu với hình ảnh một người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh."
    )
    target_vid = "L27_V005"
    
    resolved = runtime.translation_service.resolve(q.source_vietnamese)
    units = resolved.units
    temporal_scenes = [u for u in units if u.temporal_index is not None]
    print(f"Query decomposed into {len(temporal_scenes)} temporal scenes.")
    
    variants = [scene.segments[0] for scene in temporal_scenes if scene.segments]
    
    maxima_outcome_full = runtime.video_restricted_searcher.evaluate_video_maxima(
        queries=variants,
        k=99999
    )
    
    union_videos = set()
    for i, (scene, var) in enumerate(zip(temporal_scenes, variants)):
        print(f"\n--- T{scene.temporal_index} ---")
        print(f"VI: {scene.source_vietnamese}")
        print(f"EN: {var.english_text}")
        
        hits_full = maxima_outcome_full.rankings.get(var.variant_id, [])
        target_hit = next((h for h in hits_full if h.video_id == target_vid), None)
        
        if target_hit:
            print(f"Target {target_vid} Full-Corpus Rank: #{target_hit.rank}")
            print(f"Target {target_vid} Raw Max Score   : {target_hit.top_m_score:.4f}")
            print(f"Target {target_vid} Gate Pass (0.15): {'YES' if target_hit.top_m_score >= 0.15 else 'NO'}")
            print(f"Target {target_vid} in Top128 Union : {'YES' if target_hit.rank <= 128 else 'NO'}")
        else:
            print(f"Target {target_vid} NOT FOUND in corpus.")
            
        best_hit = hits_full[0] if hits_full else None
        print(f"Best video in corpus: {best_hit.video_id if best_hit else 'None'} (Score: {best_hit.top_m_score if best_hit else 0.0:.4f})")
        print(f"T{scene.temporal_index} pool size: 128")
        
        vids_128 = [h.video_id for h in hits_full[:128]]
        union_videos.update(vids_128)

    print(f"\nTOTAL Top128 Union Size across all {len(variants)} scenes: {len(union_videos)}")
    print(f"Target {target_vid} inside actual N-scene union: {'YES' if target_vid in union_videos else 'NO'}")

def run_p1_4_audit(runtime: OperationalKISRuntime, base_out: Path):
    print("\n" + "="*80)
    print("3. P1-4 TARGETED TEMPORAL EVIDENCE AUDIT")
    print("="*80)
    q = KISQuery(
        query_id="query-p1-4-kis",
        source_vietnamese="Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú."
    )
    variants = [scene.segments[0] for scene in runtime.translation_service.resolve(q.source_vietnamese).units if scene.temporal_index is not None and scene.segments]
    maxima_outcome_full = runtime.video_restricted_searcher.evaluate_video_maxima(queries=variants, k=99999)
    
    for vid in ["L22_V021", "L28_V012"]:
        print(f"\n--- Video {vid} ---")
        peaks_by_scene = []
        for i, var in enumerate(variants):
            hit = next((h for h in maxima_outcome_full.rankings.get(var.variant_id, []) if h.video_id == vid), None)
            if hit:
                print(f"T{i+1} Top 5 Peaks: {hit.top_m_peaks}")
                print(f"T{i+1} Raw Score  : {hit.top_m_score:.4f}")
                peaks_by_scene.append(hit.top_m_peaks if hit.top_m_peaks else [(hit.frame_id, hit.cosine_score)])
            else:
                peaks_by_scene.append([])
                
        from system_tai.kis.video_first import solve_temporal_chain
        has_valid_chain, chain_frames, chain_score = solve_temporal_chain(peaks_by_scene=peaks_by_scene, scene_weights=[float(v.weight) for v in variants], min_gap=60)
        print(f"Winning DP frames: {chain_frames} (Valid: {has_valid_chain}, Score: {chain_score:.4f})")
        
        render_contact_sheet(runtime, vid, variants, maxima_outcome_full, base_out / f"p1-4_{vid}_contact_sheet.png")

def render_contact_sheet(runtime, vid: str, variants: list, maxima, out_path: Path):
    try:
        store = next((s for s in runtime.video_restricted_searcher.registry.stores if s.video_id == vid), None)
        if not store:
            return
            
        fig, axes = plt.subplots(len(variants), 5, figsize=(20, 4*len(variants)))
        if len(variants) == 1:
            axes = [axes]
            
        for i, var in enumerate(variants):
            hit = next((h for h in maxima.rankings.get(var.variant_id, []) if h.video_id == vid), None)
            peaks = hit.top_m_peaks if hit else []
            for j in range(5):
                ax = axes[i][j]
                if j < len(peaks):
                    frame_id, score = peaks[j]
                    idx = store.frame_index_lookup.get(frame_id)
                    if idx is not None:
                        ax.imshow(store.load_frame_image(idx))
                    ax.set_title(f"T{i+1} Frame {frame_id} (Score {score:.4f})", fontsize=10)
                ax.axis('off')
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved contact sheet to {out_path}")
    except Exception as e:
        print(f"Failed to render contact sheet for {vid}: {e}")

def run_p1_2_audit(runtime: OperationalKISRuntime):
    print("\n" + "="*80)
    print("4. P1-2 FRAME-LOSS TRACE")
    print("="*80)
    q = KISQuery(query_id="query-p1-2-kis", source_vietnamese="Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.")
    target_vid, evidence_frame = "L29_V018", 6171
    req = runtime.process_session_request(query=q)
    
    evidence_pool = req.diagnostic_trace.get("video_first", {}).get("evidence_pool", [])
    ev_frame_hit = next((item for item in evidence_pool if item["video_id"] == target_vid and item["frame_id"] == evidence_frame), None)
    
    if ev_frame_hit:
        print(f"Evidence Frame {evidence_frame} WAS in evidence pool. Pre-fusion score: {ev_frame_hit.get('source_video_score')}")
    else:
        print(f"Evidence Frame {evidence_frame} NOT in evidence pool.")
        
    target_final = [f for f in req.final_candidates if f.video_id == target_vid]
    print(f"Target frames in Final Top100: {len(target_final)}")
    for f in target_final:
        print(f" - Included Frame: {f.frame_id} (Rank #{f.rank}, Final Score: {f.score:.4f})")

def run_p1_5_audit(runtime: OperationalKISRuntime):
    print("\n" + "="*80)
    print("5. P1-5 PROMPT DIAGNOSTIC ONLY")
    print("="*80)
    
    target_vid = "L30_V021"
    prompts = [
        "squid and peas stir-frying in a pan",
        "peas being added to squid in a frying pan",
        "sliced red pepper and onion beside a pan of squid",
        "a frying pan being tossed over a gas flame",
        "The clip begins with the peas being put in with the squid being sautéed on the pan, next to a plate of onions and sliced red peppers being prepared for the dish.",
        "The clip ends with a slow motion pan shaking scene on the stove"
    ]
    
    from system_tai.kis.contest_schema import QueryVariant
    import uuid
    variants = [QueryVariant(variant_id=str(uuid.uuid4()), english_text=p, weight=1.0, original_type="diagnostic") for p in prompts]
    maxima_outcome_full = runtime.video_restricted_searcher.evaluate_video_maxima(queries=variants, k=99999)
    
    for var in variants:
        hit = next((h for h in maxima_outcome_full.rankings.get(var.variant_id, []) if h.video_id == target_vid), None)
        print(f"\nPrompt: '{var.english_text}'")
        if hit:
            print(f"  Target {target_vid} Full-Corpus Rank: #{hit.rank} (Raw Max Score: {hit.top_m_score:.4f})")
        else:
            print(f"  Target {target_vid} NOT FOUND")

if __name__ == "__main__":
    main()
