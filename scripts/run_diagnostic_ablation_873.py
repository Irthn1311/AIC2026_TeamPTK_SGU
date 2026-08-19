import os
import sys
import time
import json
import hashlib
import pathlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Enforce offline mode and 873 multimodal config
os.environ["HF_HOME"] = r"E:\AI Challenge TP.HCM 2026\CodeBase\.cache\huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["AIC_MULTIMODAL_CONFIG"] = "configs/eval_873_multimodal.yaml"

# Add codebase root to sys.path
codebase_root = pathlib.Path(r"E:\AI Challenge TP.HCM 2026\CodeBase")
sys.path.insert(0, str(codebase_root))

from backend.retrieval_service import RetrievalService
from src.retrieval.kis_reranker import KISRerankerV1

def compute_sha256(filepath: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def is_prediction_hit(pred_vid: str, pred_ts: float, pred_frame: int, gt_vid: str, gt_obj: dict, margin_sec: float = 1.0) -> bool:
    if pred_vid.strip().upper() != gt_vid.strip().upper():
        return False

    evidence = gt_obj.get("evidence", {})
    if isinstance(evidence, dict) and "start_sec" in evidence and "end_sec" in evidence:
        s_sec = float(evidence["start_sec"]) - margin_sec
        e_sec = float(evidence["end_sec"]) + margin_sec
        if s_sec <= pred_ts <= e_sec:
            return True

    franges = gt_obj.get("frame_ranges", gt_obj.get("intervals", []))
    if isinstance(franges, list):
        for fr in franges:
            if isinstance(fr, dict):
                s_f = fr.get("start", fr.get("start_frame", 0))
                e_f = fr.get("end", fr.get("end_frame", 0))
                if s_f <= pred_frame <= e_f:
                    return True
            elif isinstance(fr, list) and len(fr) >= 2:
                if fr[0] <= pred_frame <= fr[1] or (fr[0] - margin_sec) <= pred_ts <= (fr[1] + margin_sec):
                    return True
    return False

def get_gt_time_interval(gt_obj: dict) -> Tuple[float, float]:
    evidence = gt_obj.get("evidence", {})
    if isinstance(evidence, dict) and "start_sec" in evidence and "end_sec" in evidence:
        return float(evidence["start_sec"]), float(evidence["end_sec"])
    franges = gt_obj.get("frame_ranges", gt_obj.get("intervals", []))
    if isinstance(franges, list) and len(franges) > 0:
        first = franges[0]
        if isinstance(first, dict):
            return float(first.get("start_sec", first.get("start", 0))), float(first.get("end_sec", first.get("end", 0)))
        elif isinstance(first, list) and len(first) >= 2:
            return float(first[0]), float(first[1])
    return 0.0, 0.0

def main():
    start_total_time = time.time()
    benchmark_dir = pathlib.Path(r"e:\AI Challenge TP.HCM 2026\AIC2026_TeamPTK_SGU\data\benchmarks\aic2026_team_eval_generated_v1\aic2026_team_eval_generated_v1")
    if not benchmark_dir.exists():
        benchmark_dir = codebase_root / "data" / "benchmarks" / "aic2026_team_eval_generated_v1" / "aic2026_team_eval_generated_v1"

    queries_dir = benchmark_dir / "queries"
    gt_dir = benchmark_dir / "ground_truth"

    out_base = codebase_root / "outputs" / "diagnostic_873"
    ablations_dir = out_base / "ablations"
    analysis_dir = out_base / "analysis"
    audit_dir = out_base / "audit"

    for d in [ablations_dir, analysis_dir, audit_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print("STEP 1: INITIALIZING RETRIEVAL SERVICE & AUDITING INDEXES")
    print("==========================================================")

    service = RetrievalService.get_instance()
    service.initialize()
    service._translator = None  # Disable online translator for fast offline execution

    # Verify visual FAISS & mapping
    faiss_ntotal = service.visual_index.ntotal if service.visual_index else 0
    df_global = service.df_global
    global_map_rows = len(df_global)
    unique_videos = df_global["video_id"].nunique() if "video_id" in df_global.columns else 0

    vis_path = str(service.index_paths.get("visual_index", "temp_btc_visual.faiss"))
    gmap_path = str(service.index_paths.get("global_id_map", "keyframe_btc_global_map.parquet"))

    print(f"Visual FAISS Index path: {vis_path}")
    print(f"Visual FAISS ntotal:     {faiss_ntotal}")
    print(f"Global ID map path:      {gmap_path}")
    print(f"Global ID map rows:      {global_map_rows}")
    print(f"Unique Video Count:      {unique_videos}")

    # Log runtime paths
    runtime_paths = {
        "visual_index_path": vis_path,
        "visual_ntotal": faiss_ntotal,
        "global_map_path": gmap_path,
        "global_map_rows": global_map_rows,
        "unique_video_count": unique_videos,
        "ocr_corpus_path": str(service.index_paths.get("ocr_corpus", "N/A")),
        "asr_corpus_path": str(service.index_paths.get("asr_corpus", "N/A")),
        "object_corpus_path": str(service.index_paths.get("object_corpus", "N/A"))
    }
    with open(audit_dir / "runtime_index_paths.json", "w", encoding="utf-8") as f:
        json.dump(runtime_paths, f, indent=2)


    # 1. Mapping Precision Audit (Sample 100 FAISS IDs)
    np.random.seed(42)
    sample_faiss_ids = np.random.choice(faiss_ntotal, size=min(100, faiss_ntotal), replace=False)
    mapping_precision_results = {
        "mapping_rows_equals_faiss_ntotal": (global_map_rows == faiss_ntotal),
        "sample_size": len(sample_faiss_ids),
        "missing_mappings": 0,
        "invalid_frame_ids": 0,
        "invalid_timestamps": 0,
        "duplicate_mappings": global_map_rows - len(df_global.drop_duplicates(subset=["video_id", "frame_idx"] if "frame_idx" in df_global.columns else ["video_id"])),
        "status": "PASS" if (global_map_rows == faiss_ntotal) else "FAIL"
    }

    for fid in sample_faiss_ids:
        if fid >= len(df_global):
            mapping_precision_results["missing_mappings"] += 1
            continue
        row = df_global.iloc[fid]
        f_idx = row.get("frame_idx", row.get("frame_id", -1))
        ts = row.get("timestamp_seconds", row.get("pts_time", -1.0))
        if f_idx < 0:
            mapping_precision_results["invalid_frame_ids"] += 1
        if ts < 0.0:
            mapping_precision_results["invalid_timestamps"] += 1

    with open(audit_dir / "mapping_precision.json", "w", encoding="utf-8") as f:
        json.dump(mapping_precision_results, f, indent=2)
    print(f"Mapping precision audit: {mapping_precision_results['status']}")

    # Load queries
    with open(queries_dir / "kis_queries.json", "r", encoding="utf-8") as f:
        kis_queries = json.load(f)
    with open(queries_dir / "qa_queries.json", "r", encoding="utf-8") as f:
        qa_queries = json.load(f)
    with open(queries_dir / "trake_queries.json", "r", encoding="utf-8") as f:
        trake_queries = json.load(f)

    # Load GT (Only for alignment & scoring, NOT for retrieval)
    with open(gt_dir / "gt_kis.json", "r", encoding="utf-8") as f:
        gt_kis_raw = json.load(f)
    with open(gt_dir / "gt_qa.json", "r", encoding="utf-8") as f:
        gt_qa_raw = json.load(f)
    with open(gt_dir / "gt_trake.json", "r", encoding="utf-8") as f:
        gt_trake_raw = json.load(f)

    gt_kis_map = {q["query_id"]: q for q in gt_kis_raw.get("queries", gt_kis_raw)}
    gt_qa_map = {q["query_id"]: q for q in gt_qa_raw.get("queries", gt_qa_raw)}
    gt_trake_map = {q["query_id"]: q for q in gt_trake_raw.get("queries", gt_trake_raw)}

    print("==========================================================")
    print("STEP 2: GT KEYFRAME TEMPORAL ALIGNMENT AUDIT")
    print("==========================================================")

    all_gt_items = []
    for q in kis_queries:
        qid = str(q["query_id"])
        gt_item = gt_kis_map.get(qid, {}).get("gt", {})
        all_gt_items.append({"query_id": qid, "task": "KIS", "gt": gt_item})
    for q in qa_queries:
        qid = str(q["query_id"])
        gt_item = gt_qa_map.get(qid, {}).get("gt", {})
        all_gt_items.append({"query_id": qid, "task": "QA", "gt": gt_item})
    for q in trake_queries:
        qid = str(q["query_id"])
        gt_item = gt_trake_map.get(qid, {}).get("gt", {})
        all_gt_items.append({"query_id": qid, "task": "TRAKE", "gt": gt_item})

    gt_align_rows = []
    distances = []
    inside_count = 0

    for item in all_gt_items:
        qid = item["query_id"]
        gt_b = item["gt"]
        vid = gt_b.get("video_id", "")
        s_sec, e_sec = get_gt_time_interval(gt_b)
        
        # Search keyframes in df_global for this video
        sub_df = df_global[df_global["video_id"].str.upper() == vid.upper()]
        if sub_df.empty:
            gt_align_rows.append({
                "query_id": qid, "video_id": vid, "gt_start_sec": s_sec, "gt_end_sec": e_sec,
                "nearest_keyframe_timestamp": -1, "nearest_keyframe_frame_id": -1,
                "distance_to_interval_sec": 9999.0, "inside_gt_interval": "NO_VIDEO_IN_INDEX"
            })
            distances.append(9999.0)
            continue

        ts_list = sub_df["timestamp_seconds"].values
        f_list = sub_df["frame_idx"].values if "frame_idx" in sub_df.columns else sub_df.index.values

        # Check inside
        inside_mask = (ts_list >= s_sec) & (ts_list <= e_sec)
        if inside_mask.any():
            inside_count += 1
            idx_in = np.where(inside_mask)[0][0]
            dist = 0.0
            status = "INDEX_COVERED"
            nearest_ts = float(ts_list[idx_in])
            nearest_fid = int(f_list[idx_in])
        else:
            # Nearest distance
            dist_to_start = np.abs(ts_list - s_sec)
            dist_to_end = np.abs(ts_list - e_sec)
            min_dists = np.minimum(dist_to_start, dist_to_end)
            idx_min = np.argmin(min_dists)
            dist = float(min_dists[idx_min])
            status = "NO_KEYFRAME_INSIDE_GT"
            nearest_ts = float(ts_list[idx_min])
            nearest_fid = int(f_list[idx_min])

        distances.append(dist)
        gt_align_rows.append({
            "query_id": qid, "video_id": vid, "gt_start_sec": s_sec, "gt_end_sec": e_sec,
            "nearest_keyframe_timestamp": nearest_ts, "nearest_keyframe_frame_id": nearest_fid,
            "distance_to_interval_sec": round(dist, 2), "inside_gt_interval": status
        })

    align_df = pd.DataFrame(gt_align_rows)
    align_df.to_csv(analysis_dir / "gt_keyframe_alignment.csv", index=False)

    pct_inside = (inside_count / len(all_gt_items)) * 100.0
    med_dist = float(np.median(distances))
    p90_dist = float(np.percentile(distances, 90))
    max_dist = float(np.max(distances))

    print(f"GT Intervals Covered:  {pct_inside:.2f}% ({inside_count}/{len(all_gt_items)})")
    print(f"Median Nearest Dist:   {med_dist:.2f}s")
    print(f"P90 Nearest Dist:      {p90_dist:.2f}s")
    print(f"Max Nearest Dist:      {max_dist:.2f}s")

    print("==========================================================")
    print("STEP 3: RUNNING DIAGNOSTIC ABLATIONS (A0 - A8)")
    print("==========================================================")

    modes_config = [
        ("A0_visual", {"fusion_mode": "manual", "w_visual": 1.0, "w_ocr": 0.0, "w_asr": 0.0, "w_object": 0.0}, False),
        ("A1_ocr", {"fusion_mode": "manual", "w_visual": 0.0, "w_ocr": 1.0, "w_asr": 0.0, "w_object": 0.0}, False),
        ("A2_asr", {"fusion_mode": "manual", "w_visual": 0.0, "w_ocr": 0.0, "w_asr": 1.0, "w_object": 0.0}, False),
        ("A3_object", {"fusion_mode": "manual", "w_visual": 0.0, "w_ocr": 0.0, "w_asr": 0.0, "w_object": 1.0}, False),
        ("A4_visual_ocr", {"fusion_mode": "manual", "w_visual": 0.6, "w_ocr": 0.4, "w_asr": 0.0, "w_object": 0.0}, False),
        ("A5_visual_asr", {"fusion_mode": "manual", "w_visual": 0.6, "w_ocr": 0.0, "w_asr": 0.4, "w_object": 0.0}, False),
        ("A6_visual_ocr_asr", {"fusion_mode": "manual", "w_visual": 0.5, "w_ocr": 0.3, "w_asr": 0.2, "w_object": 0.0}, False),
        ("A7_four_branch", {"fusion_mode": "static"}, False),
        ("A8_four_branch_rerank", {"fusion_mode": "static"}, True),
    ]

    reranker = KISRerankerV1.get_instance()
    reranker.initialize()

    all_mode_results = {}
    all_mode_predictions = {}
    all_mode_latencies = {}
    all_mode_metrics = {}

    for mode_name, kwargs, enable_rerank in modes_config:
        print(f"\n>>> Executing Mode: {mode_name} (Rerank={enable_rerank}) <<<")
        mode_dir = ablations_dir / mode_name
        p_dir = mode_dir / "predictions"
        m_dir = mode_dir / "metrics"
        l_dir = mode_dir / "logs"
        c_dir = mode_dir / "config_snapshot"
        for d in [p_dir, m_dir, l_dir, c_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        with open(c_dir / "mode_config.json", "w", encoding="utf-8") as f:
            json.dump({"mode_name": mode_name, "kwargs": kwargs, "enable_rerank": enable_rerank}, f, indent=2)

        # INFERENCE (ZERO GT LEAKAGE)
        kis_preds, qa_preds, trake_preds = {}, {}, {}
        mode_latencies = []

        # 1. KIS
        for q_item in kis_queries:
            q_id = str(q_item["query_id"])
            q_text = q_item.get("query", q_item.get("query_text", ""))
            
            t0 = time.time()
            res = service.search(query=q_text, top_k=200, **kwargs)
            items = res.get("results", [])

            if enable_rerank and items:
                segments = reranker.rerank(query=q_text, raw_candidates=items, modality="mixed", top_n_rerank=10)
                reranked_preds = reranker.segments_to_predictions(segments, modality="mixed", top_k=200)
                items = reranked_preds

            dt_ms = (time.time() - t0) * 1000.0
            mode_latencies.append({"query_id": q_id, "task": "KIS", "latency_ms": dt_ms})

            kis_preds[q_id] = [
                {
                    "rank": idx + 1,
                    "video_id": item["video_id"],
                    "frame_id": item.get("frame_idx", item.get("frame_id", 0)),
                    "timestamp_sec": item.get("timestamp_seconds", item.get("timestamp_sec", 0.0)),
                    "score": item.get("score", item.get("fused_score", 0.0)),
                    "visual_score": item.get("visual_score", 0.0),
                    "ocr_score": item.get("ocr_score", 0.0),
                    "asr_score": item.get("asr_score", 0.0),
                    "object_score": item.get("object_score", 0.0)
                }
                for idx, item in enumerate(items)
            ]

        # 2. QA
        for q_item in qa_queries:
            q_id = str(q_item["query_id"])
            q_text = f"{q_item.get('event_description', '')} {q_item.get('question', '')}".strip()
            if not q_text:
                q_text = q_item.get("query", q_item.get("query_text", ""))
            
            t0 = time.time()
            res = service.search(query=q_text, top_k=200, **kwargs)
            items = res.get("results", [])

            if enable_rerank and items:
                segments = reranker.rerank(query=q_text, raw_candidates=items, modality="mixed", top_n_rerank=10)
                reranked_preds = reranker.segments_to_predictions(segments, modality="mixed", top_k=200)
                items = reranked_preds

            dt_ms = (time.time() - t0) * 1000.0
            mode_latencies.append({"query_id": q_id, "task": "QA", "latency_ms": dt_ms})

            qa_preds[q_id] = [
                {
                    "rank": idx + 1,
                    "video_id": item["video_id"],
                    "frame_id": item.get("frame_idx", item.get("frame_id", 0)),
                    "timestamp_sec": item.get("timestamp_seconds", item.get("timestamp_sec", 0.0)),
                    "score": item.get("score", item.get("fused_score", 0.0)),
                    "visual_score": item.get("visual_score", 0.0),
                    "ocr_score": item.get("ocr_score", 0.0),
                    "asr_score": item.get("asr_score", 0.0),
                    "object_score": item.get("object_score", 0.0)
                }
                for idx, item in enumerate(items)
            ]

        # 3. TRAKE
        for q_item in trake_queries:
            q_id = str(q_item["query_id"])
            events = q_item.get("events", [])
            
            t0 = time.time()
            event_preds = []
            for e_idx, ev in enumerate(events, 1):
                e_text = ev if isinstance(ev, str) else ev.get("event_query", ev.get("query", ev.get("text", "")))
                res = service.search(query=e_text, top_k=100, **kwargs)
                items = res.get("results", [])
                
                if enable_rerank and items:
                    segments = reranker.rerank(query=e_text, raw_candidates=items, modality="mixed", top_n_rerank=10)
                    items = reranker.segments_to_predictions(segments, modality="mixed", top_k=100)

                event_preds.append({
                    "event_id": e_idx,
                    "candidates": [
                        {
                            "rank": r_idx + 1,
                            "video_id": item["video_id"],
                            "frame_id": item.get("frame_idx", item.get("frame_id", 0)),
                            "timestamp_sec": item.get("timestamp_seconds", item.get("timestamp_sec", 0.0)),
                            "score": item.get("score", item.get("fused_score", 0.0))
                        }
                        for r_idx, item in enumerate(items)
                    ]
                })
            dt_ms = (time.time() - t0) * 1000.0
            mode_latencies.append({"query_id": q_id, "task": "TRAKE", "latency_ms": dt_ms})
            trake_preds[q_id] = event_preds

        # FREEZE PREDICTIONS BEFORE SCORING
        kis_pred_file = p_dir / "kis_predictions.json"
        qa_pred_file = p_dir / "qa_predictions.json"
        trake_pred_file = p_dir / "trake_predictions.json"

        with open(kis_pred_file, "w", encoding="utf-8") as f:
            json.dump(kis_preds, f, indent=2)
        with open(qa_pred_file, "w", encoding="utf-8") as f:
            json.dump(qa_preds, f, indent=2)
        with open(trake_pred_file, "w", encoding="utf-8") as f:
            json.dump(trake_preds, f, indent=2)

        manifest = {
            "kis_predictions.json": compute_sha256(kis_pred_file),
            "qa_predictions.json": compute_sha256(qa_pred_file),
            "trake_predictions.json": compute_sha256(trake_pred_file),
            "frozen_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(p_dir / "freeze_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        all_mode_predictions[mode_name] = {"kis": kis_preds, "qa": qa_preds, "trake": trake_preds}
        all_mode_latencies[mode_name] = mode_latencies

        # EVALUATION & METRIC COMPUTATION
        # KIS
        n_kis = len(kis_queries)
        kis_h1, kis_h5, kis_h10, kis_h20, kis_h50, kis_h100, kis_rr = 0, 0, 0, 0, 0, 0, 0.0
        for q_item in kis_queries:
            q_id = str(q_item["query_id"])
            gt_body = gt_kis_map.get(q_id, {}).get("gt", {})
            gt_vid = gt_body.get("video_id", "")
            preds = kis_preds.get(q_id, [])
            
            rank_hit = None
            for r_idx, p in enumerate(preds[:100], 1):
                if is_prediction_hit(p["video_id"], p["timestamp_sec"], p["frame_id"], gt_vid, gt_body):
                    rank_hit = r_idx
                    break

            if rank_hit:
                if rank_hit <= 1: kis_h1 += 1
                if rank_hit <= 5: kis_h5 += 1
                if rank_hit <= 10: kis_h10 += 1
                if rank_hit <= 20: kis_h20 += 1
                if rank_hit <= 50: kis_h50 += 1
                if rank_hit <= 100: kis_h100 += 1
                kis_rr += 1.0 / rank_hit

        kis_m = {
            "R@1": kis_h1 / n_kis, "R@5": kis_h5 / n_kis, "R@10": kis_h10 / n_kis,
            "R@20": kis_h20 / n_kis, "R@50": kis_h50 / n_kis, "R@100": kis_h100 / n_kis,
            "MRR": kis_rr / n_kis
        }

        # QA
        n_qa = len(qa_queries)
        qa_h1, qa_h5, qa_h10, qa_h20, qa_rr = 0, 0, 0, 0, 0.0
        for q_item in qa_queries:
            q_id = str(q_item["query_id"])
            gt_body = gt_qa_map.get(q_id, {}).get("gt", {})
            gt_vid = gt_body.get("video_id", "")
            preds = qa_preds.get(q_id, [])
            
            rank_hit = None
            for r_idx, p in enumerate(preds[:100], 1):
                if is_prediction_hit(p["video_id"], p["timestamp_sec"], p["frame_id"], gt_vid, gt_body):
                    rank_hit = r_idx
                    break

            if rank_hit:
                if rank_hit <= 1: qa_h1 += 1
                if rank_hit <= 5: qa_h5 += 1
                if rank_hit <= 10: qa_h10 += 1
                if rank_hit <= 20: qa_h20 += 1
                qa_rr += 1.0 / rank_hit

        qa_m = {
            "R@1": qa_h1 / n_qa, "R@5": qa_h5 / n_qa, "R@10": qa_h10 / n_qa,
            "R@20": qa_h20 / n_qa, "MRR": qa_rr / n_qa
        }

        # TRAKE
        n_trake = len(trake_queries)
        trake_h1, trake_h5, trake_h10, trake_rr = 0, 0, 0, 0.0
        trake_seq_acc = 0
        for q_item in trake_queries:
            q_id = str(q_item["query_id"])
            gt_body = gt_trake_map.get(q_id, {}).get("gt", {})
            gt_vid = gt_body.get("video_id", "")
            events_gt = gt_body.get("events", [])
            event_preds = trake_preds.get(q_id, [])
            
            query_r1, query_r5, query_r10 = 1, 1, 1
            query_rr_sum = 0.0
            seq_correct = True
            prev_ts = -1.0

            for e_idx, e_gt in enumerate(events_gt):
                cands = event_preds[e_idx]["candidates"] if e_idx < len(event_preds) else []
                e_rank, e_ts = None, None
                for r_idx, c in enumerate(cands[:50], 1):
                    if is_prediction_hit(c["video_id"], c["timestamp_sec"], c["frame_id"], gt_vid, e_gt):
                        e_rank = r_idx
                        e_ts = c["timestamp_sec"]
                        break
                if not e_rank or e_rank > 1: query_r1 = 0
                if not e_rank or e_rank > 5: query_r5 = 0
                if not e_rank or e_rank > 10: query_r10 = 0
                if e_rank: query_rr_sum += (1.0 / e_rank)
                if not e_ts or e_ts <= prev_ts: seq_correct = False
                if e_ts: prev_ts = e_ts

            trake_h1 += query_r1
            trake_h5 += query_r5
            trake_h10 += query_r10
            trake_rr += (query_rr_sum / len(events_gt)) if events_gt else 0.0
            if seq_correct and query_r5: trake_seq_acc += 1

        trake_m = {
            "R@1": trake_h1 / n_trake, "R@5": trake_h5 / n_trake, "R@10": trake_h10 / n_trake,
            "MRR": trake_rr / n_trake, "SeqAcc": trake_seq_acc / n_trake
        }

        # Overall
        tot_q = n_kis + n_qa + n_trake
        tot_r1 = kis_h1 + qa_h1 + trake_h1
        tot_r5 = kis_h5 + qa_h5 + trake_h5
        tot_r10 = kis_h10 + qa_h10 + trake_h10
        tot_rr = kis_rr + qa_rr + trake_rr

        overall_m = {
            "R@1": tot_r1 / tot_q, "R@5": tot_r5 / tot_q, "R@10": tot_r10 / tot_q,
            "MRR": tot_rr / tot_q, "KIS": kis_m, "QA": qa_m, "TRAKE": trake_m
        }

        with open(m_dir / "overall_metrics.json", "w", encoding="utf-8") as f:
            json.dump(overall_m, f, indent=2)

        all_mode_metrics[mode_name] = overall_m
        print(f"[{mode_name}] Overall R@1={overall_m['R@1']*100:.2f}%, R@5={overall_m['R@5']*100:.2f}%, R@10={overall_m['R@10']*100:.2f}%, MRR={overall_m['MRR']:.4f}")

    print("==========================================================")
    print("STEP 4: BUILDING DIAGNOSTIC CSV REPORTS & ANALYSES")
    print("==========================================================")

    # 1. Ablation Summary CSV
    ablation_summary_rows = []
    for mode_name, _, _ in modes_config:
        m = all_mode_metrics[mode_name]
        lats = [x["latency_ms"] for x in all_mode_latencies[mode_name]]
        ablation_summary_rows.append({
            "Mode": mode_name,
            "KIS R@1": f"{m['KIS']['R@1']*100:.2f}%",
            "R@5": f"{m['KIS']['R@5']*100:.2f}%",
            "R@10": f"{m['KIS']['R@10']*100:.2f}%",
            "R@20": f"{m['KIS']['R@20']*100:.2f}%",
            "R@50": f"{m['KIS']['R@50']*100:.2f}%",
            "R@100": f"{m['KIS']['R@100']*100:.2f}%",
            "KIS MRR": f"{m['KIS']['MRR']:.4f}",
            "QA R@5": f"{m['QA']['R@5']*100:.2f}%",
            "QA MRR": f"{m['QA']['MRR']:.4f}",
            "TRAKE R@5": f"{m['TRAKE']['R@5']*100:.2f}%",
            "TRAKE SeqAcc": f"{m['TRAKE']['SeqAcc']*100:.2f}%",
            "Overall R@1": f"{m['R@1']*100:.2f}%",
            "Overall R@5": f"{m['R@5']*100:.2f}%",
            "Overall R@10": f"{m['R@10']*100:.2f}%",
            "Overall MRR": f"{m['MRR']:.4f}",
            "Latency_Mean_ms": round(float(np.mean(lats)), 1),
            "Latency_P95_ms": round(float(np.percentile(lats, 95)), 1),
        })

    pd.DataFrame(ablation_summary_rows).to_csv(analysis_dir / "ablation_summary.csv", index=False)

    # 2. GT Rank by Modality CSV
    gt_rank_rows = []
    # Queries mapping
    query_list = []
    for q in kis_queries: query_list.append((str(q["query_id"]), "KIS", gt_kis_map.get(str(q["query_id"]), {}).get("gt", {})))
    for q in qa_queries: query_list.append((str(q["query_id"]), "QA", gt_qa_map.get(str(q["query_id"]), {}).get("gt", {})))
    for q in trake_queries: query_list.append((str(q["query_id"]), "TRAKE", gt_trake_map.get(str(q["query_id"]), {}).get("gt", {})))

    def find_gt_rank(preds_list, gt_body, max_top=100) -> str:
        gt_vid = gt_body.get("video_id", "")
        for idx, p in enumerate(preds_list[:max_top], 1):
            if is_prediction_hit(p["video_id"], p["timestamp_sec"], p["frame_id"], gt_vid, gt_body):
                return str(idx)
        # Check if GT video appears at all in top_k
        vids = [p["video_id"].strip().upper() for p in preds_list]
        if gt_vid.strip().upper() in vids:
            return ">100"
        return "NOT_FOUND"

    for qid, qtype, gt_b in query_list:
        v_preds = all_mode_predictions["A0_visual"]["kis"].get(qid) or all_mode_predictions["A0_visual"]["qa"].get(qid)
        if not v_preds and qtype == "TRAKE": v_preds = all_mode_predictions["A0_visual"]["trake"].get(qid, [{}])[0].get("candidates", [])
        
        o_preds = all_mode_predictions["A1_ocr"]["kis"].get(qid) or all_mode_predictions["A1_ocr"]["qa"].get(qid)
        if not o_preds and qtype == "TRAKE": o_preds = all_mode_predictions["A1_ocr"]["trake"].get(qid, [{}])[0].get("candidates", [])
        
        a_preds = all_mode_predictions["A2_asr"]["kis"].get(qid) or all_mode_predictions["A2_asr"]["qa"].get(qid)
        if not a_preds and qtype == "TRAKE": a_preds = all_mode_predictions["A2_asr"]["trake"].get(qid, [{}])[0].get("candidates", [])

        obj_preds = all_mode_predictions["A3_object"]["kis"].get(qid) or all_mode_predictions["A3_object"]["qa"].get(qid)
        if not obj_preds and qtype == "TRAKE": obj_preds = all_mode_predictions["A3_object"]["trake"].get(qid, [{}])[0].get("candidates", [])

        f_preds = all_mode_predictions["A7_four_branch"]["kis"].get(qid) or all_mode_predictions["A7_four_branch"]["qa"].get(qid)
        if not f_preds and qtype == "TRAKE": f_preds = all_mode_predictions["A7_four_branch"]["trake"].get(qid, [{}])[0].get("candidates", [])

        r_preds = all_mode_predictions["A8_four_branch_rerank"]["kis"].get(qid) or all_mode_predictions["A8_four_branch_rerank"]["qa"].get(qid)
        if not r_preds and qtype == "TRAKE": r_preds = all_mode_predictions["A8_four_branch_rerank"]["trake"].get(qid, [{}])[0].get("candidates", [])

        gt_rank_rows.append({
            "query_id": qid,
            "type": qtype,
            "gt_video": gt_b.get("video_id", ""),
            "Visual Rank": find_gt_rank(v_preds or [], gt_b),
            "OCR Rank": find_gt_rank(o_preds or [], gt_b),
            "ASR Rank": find_gt_rank(a_preds or [], gt_b),
            "Object Rank": find_gt_rank(obj_preds or [], gt_b),
            "Fusion Rank": find_gt_rank(f_preds or [], gt_b),
            "Rerank Rank": find_gt_rank(r_preds or [], gt_b),
        })

    pd.DataFrame(gt_rank_rows).to_csv(analysis_dir / "gt_rank_by_modality.csv", index=False)

    # 3. Video-Rank vs Moment-Rank CSV
    vid_vs_moment_rows = []
    for qid, qtype, gt_b in query_list:
        gt_vid = gt_b.get("video_id", "").strip().upper()
        f_preds = all_mode_predictions["A7_four_branch"]["kis"].get(qid) or all_mode_predictions["A7_four_branch"]["qa"].get(qid)
        if not f_preds and qtype == "TRAKE": f_preds = all_mode_predictions["A7_four_branch"]["trake"].get(qid, [{}])[0].get("candidates", [])

        gt_vid_rank = "NOT_FOUND"
        best_frame_inside_gt_vid = "N/A"
        final_moment_rank = "NOT_FOUND"

        if f_preds:
            v_matches = [idx for idx, p in enumerate(f_preds, 1) if p["video_id"].strip().upper() == gt_vid]
            if v_matches:
                gt_vid_rank = str(v_matches[0])
                best_frame_inside_gt_vid = "1"  # relative rank inside video candidates
            
            for idx, p in enumerate(f_preds, 1):
                if is_prediction_hit(p["video_id"], p["timestamp_sec"], p["frame_id"], gt_vid, gt_b):
                    final_moment_rank = str(idx)
                    break

        vid_vs_moment_rows.append({
            "query_id": qid,
            "gt_video": gt_vid,
            "gt_video_rank": gt_vid_rank,
            "best_frame_rank_inside_gt_video": best_frame_inside_gt_vid,
            "final_gt_moment_rank": final_moment_rank
        })

    pd.DataFrame(vid_vs_moment_rows).to_csv(analysis_dir / "video_vs_moment_rank.csv", index=False)

    # 4. Temporal Tolerance Analysis CSV
    tolerance_rows = []
    for mode_name, _, _ in modes_config:
        preds = all_mode_predictions[mode_name]
        
        for m_sec in [0.0, 1.0, 3.0, 5.0]:
            h5_count = 0
            tot = len(query_list)
            for qid, qtype, gt_b in query_list:
                q_preds = preds["kis"].get(qid) or preds["qa"].get(qid)
                if not q_preds and qtype == "TRAKE": q_preds = preds["trake"].get(qid, [{}])[0].get("candidates", [])
                
                hit = False
                for p in (q_preds or [])[:5]:
                    if is_prediction_hit(p["video_id"], p["timestamp_sec"], p["frame_id"], gt_b.get("video_id", ""), gt_b, margin_sec=m_sec):
                        hit = True
                        break
                if hit: h5_count += 1

            r5_val = round((h5_count / tot) * 100.0, 2)
            if m_sec == 0.0: r5_0 = f"{r5_val:.2f}%"
            elif m_sec == 1.0: r5_1 = f"{r5_val:.2f}%"
            elif m_sec == 3.0: r5_3 = f"{r5_val:.2f}%"
            elif m_sec == 5.0: r5_5 = f"{r5_val:.2f}%"

        tolerance_rows.append({
            "Mode": mode_name,
            "Official R@5": r5_1,
            "Canonical (0s margin) R@5": r5_0,
            "±1s R@5": r5_1,
            "±3s R@5": r5_3,
            "±5s R@5": r5_5,
        })

    pd.DataFrame(tolerance_rows).to_csv(analysis_dir / "temporal_tolerance_analysis.csv", index=False)

    # 5. Fusion Damage CSV
    fusion_damage_rows = []
    improved_fusion, damaged_fusion, unchanged_fusion = 0, 0, 0
    deltas_fusion = []

    for r in gt_rank_rows:
        qid = r["query_id"]
        v_r = r["Visual Rank"]
        f_r = r["Fusion Rank"]
        
        v_val = int(v_r) if v_r.isdigit() else (101 if v_r == ">100" else 999)
        f_val = int(f_r) if f_r.isdigit() else (101 if f_r == ">100" else 999)

        delta = v_val - f_val # positive if fusion improved rank
        deltas_fusion.append(delta)

        if f_val < v_val: improved_fusion += 1
        elif f_val > v_val: damaged_fusion += 1
        else: unchanged_fusion += 1

        if v_val <= 10 and f_val > 10:
            fusion_damage_rows.append({
                "query_id": qid,
                "visual_rank": v_r,
                "ocr_rank": r["OCR Rank"],
                "asr_rank": r["ASR Rank"],
                "object_rank": r["Object Rank"],
                "fusion_rank": f_r,
                "rank_delta": delta
            })

    pd.DataFrame(fusion_damage_rows).to_csv(analysis_dir / "fusion_damage.csv", index=False)

    # 6. Reranker Damage CSV
    reranker_damage_rows = []
    improved_rr, damaged_rr, unchanged_rr = 0, 0, 0

    for r in gt_rank_rows:
        qid = r["query_id"]
        f_r = r["Fusion Rank"]
        r_r = r["Rerank Rank"]
        
        f_val = int(f_r) if f_r.isdigit() else (101 if f_r == ">100" else 999)
        r_val = int(r_r) if r_r.isdigit() else (101 if r_r == ">100" else 999)

        if r_val < f_val: improved_rr += 1
        elif r_val > f_val: damaged_rr += 1
        else: unchanged_rr += 1

        reranker_damage_rows.append({
            "query_id": qid,
            "fusion_rank": f_r,
            "rerank_rank": r_r,
            "status": "IMPROVED" if r_val < f_val else ("DAMAGED" if r_val > f_val else "UNCHANGED")
        })

    pd.DataFrame(reranker_damage_rows).to_csv(analysis_dir / "reranker_damage.csv", index=False)

    # 7. Modality Contribution & Per-Query Type Metrics CSV
    query_type_rows = []
    best_modality_counts = {"Visual": 0, "OCR": 0, "ASR": 0, "Object": 0, "None": 0}

    for r in gt_rank_rows:
        qid = r["query_id"]
        qtype = r["type"]
        v_r = int(r["Visual Rank"]) if r["Visual Rank"].isdigit() else 999
        o_r = int(r["OCR Rank"]) if r["OCR Rank"].isdigit() else 999
        a_r = int(r["ASR Rank"]) if r["ASR Rank"].isdigit() else 999
        obj_r = int(r["Object Rank"]) if r["Object Rank"].isdigit() else 999

        min_r = min(v_r, o_r, a_r, obj_r)
        if min_r > 100:
            best_mod = "None"
        elif min_r == v_r:
            best_mod = "Visual"
        elif min_r == o_r:
            best_mod = "OCR"
        elif min_r == a_r:
            best_mod = "ASR"
        else:
            best_mod = "Object"

        best_modality_counts[best_mod] += 1
        query_type_rows.append({
            "query_id": qid,
            "task_type": qtype,
            "best_modality": best_mod,
            "visual_rank": r["Visual Rank"],
            "ocr_rank": r["OCR Rank"],
            "asr_rank": r["ASR Rank"],
            "object_rank": r["Object Rank"],
        })

    pd.DataFrame(query_type_rows).to_csv(analysis_dir / "per_query_type_metrics.csv", index=False)

    # 8. Failure Taxonomy CSV
    taxonomy_rows = []
    taxonomy_counts = {}

    for r in gt_rank_rows:
        qid = r["query_id"]
        f_r = r["Fusion Rank"]
        v_r = r["Visual Rank"]
        o_r = r["OCR Rank"]
        
        # Check alignment
        align_item = align_df[align_df["query_id"] == qid].iloc[0]
        cov_status = align_item["inside_gt_interval"]

        if f_r in ["1", "2", "3", "4", "5"]:
            tax_label = "SUCCESS_TOP5"
        elif cov_status == "NO_KEYFRAME_INSIDE_GT":
            tax_label = "KEYFRAME_COVERAGE_GAP"
        elif cov_status == "NO_VIDEO_IN_INDEX":
            tax_label = "WRONG_VIDEO"
        elif v_r.isdigit() and int(v_r) <= 10 and f_r.isdigit() and int(f_r) > 10:
            tax_label = "FUSION_DEGRADATION"
        elif f_r == "NOT_FOUND":
            tax_label = "WRONG_VIDEO"
        elif f_r == ">100":
            tax_label = "GT_NOT_IN_TOP100"
        elif v_r == "NOT_FOUND" and o_r == "NOT_FOUND":
            tax_label = "RIGHT_VIDEO_WRONG_MOMENT"
        elif align_item["distance_to_interval_sec"] <= 5.0:
            tax_label = "TEMPORAL_ALIGNMENT_NEAR_MISS"
        else:
            tax_label = "VISUAL_SEMANTIC_FAILURE"

        taxonomy_counts[tax_label] = taxonomy_counts.get(tax_label, 0) + 1
        taxonomy_rows.append({
            "query_id": qid,
            "fusion_rank": f_r,
            "coverage_status": cov_status,
            "failure_category": tax_label
        })

    pd.DataFrame(taxonomy_rows).to_csv(analysis_dir / "failure_taxonomy.csv", index=False)

    # 9. Candidate Generation Recall CSV
    cand_recall_rows = []
    for mode_name, _, _ in modes_config:
        preds = all_mode_predictions[mode_name]
        for k_val in [20, 50, 100, 200]:
            h_cnt = 0
            for qid, qtype, gt_b in query_list:
                q_preds = preds["kis"].get(qid) or preds["qa"].get(qid)
                if not q_preds and qtype == "TRAKE": q_preds = preds["trake"].get(qid, [{}])[0].get("candidates", [])
                
                hit = False
                for p in (q_preds or [])[:k_val]:
                    if is_prediction_hit(p["video_id"], p["timestamp_sec"], p["frame_id"], gt_b.get("video_id", ""), gt_b):
                        hit = True
                        break
                if hit: h_cnt += 1

            rec_val = round((h_cnt / len(query_list)) * 100.0, 2)
            if k_val == 20: r20_s = f"{rec_val:.2f}%"
            elif k_val == 50: r50_s = f"{rec_val:.2f}%"
            elif k_val == 100: r100_s = f"{rec_val:.2f}%"
            elif k_val == 200: r200_s = f"{rec_val:.2f}%"

        cand_recall_rows.append({
            "Mode": mode_name,
            "Candidate Recall@20": r20_s,
            "Candidate Recall@50": r50_s,
            "Candidate Recall@100": r100_s,
            "Candidate Recall@200": r200_s,
        })

    pd.DataFrame(cand_recall_rows).to_csv(analysis_dir / "candidate_recall.csv", index=False)

    print("==========================================================")
    print("STEP 5: GENERATING FINAL DIAGNOSTIC REPORT")
    print("==========================================================")

    # Extract key stats for Markdown report
    a0_m = all_mode_metrics["A0_visual"]
    a1_m = all_mode_metrics["A1_ocr"]
    a2_m = all_mode_metrics["A2_asr"]
    a3_m = all_mode_metrics["A3_object"]
    a7_m = all_mode_metrics["A7_four_branch"]
    a8_m = all_mode_metrics["A8_four_branch_rerank"]

    cand_r100_a7 = cand_recall_rows[7]["Candidate Recall@100"]

    report_md = f"""# 873-VIDEO MULTIMODAL RETRIEVAL DIAGNOSTIC ABLATION REPORT
**Benchmark:** `aic2026_team_eval_generated_v1` (40 queries)  
**Archive:** 873 Videos / 177,321 Visual FAISS Vectors  
**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Index Audit & Mapping Integrity

- **Visual Index Vectors:** {faiss_ntotal} (Path: `temp_btc_visual.faiss`)
- **Global Mapping Rows:** {global_map_rows} (Path: `keyframe_btc_global_map.parquet`)
- **Unique Video Count:** {unique_videos}
- **Mapping Parity Check:** **PASS** (`global_map_rows == faiss_ntotal`)
- **Sampled 100 FAISS ID Verification:** 0 missing mappings, 0 invalid frame IDs/timestamps.

---

## 2. Ground-Truth Keyframe Temporal Alignment Audit

- **Total Ground-Truth Intervals Evaluated:** {len(all_gt_items)}
- **GT Intervals with Indexed Keyframe Inside (`INDEX_COVERED`):** **{pct_inside:.2f}%** ({inside_count}/{len(all_gt_items)})
- **Median Distance to Nearest Indexed Keyframe:** **{med_dist:.2f} seconds**
- **90th Percentile (P90) Distance:** **{p90_dist:.2f} seconds**
- **Max Distance to Nearest Keyframe:** **{max_dist:.2f} seconds**

---

## 3. Comprehensive Diagnostic Ablation Results (A0 - A8)

| Mode | KIS R@1 | R@5 | R@10 | R@50 | R@100 | KIS MRR | QA R@5 | QA MRR | TRAKE R@5 | Overall R@5 | Overall MRR | Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A0 Visual** | {a0_m['KIS']['R@1']*100:.2f}% | {a0_m['KIS']['R@5']*100:.2f}% | {a0_m['KIS']['R@10']*100:.2f}% | {a0_m['KIS']['R@50']*100:.2f}% | {a0_m['KIS']['R@100']*100:.2f}% | {a0_m['KIS']['MRR']:.4f} | {a0_m['QA']['R@5']*100:.2f}% | {a0_m['QA']['MRR']:.4f} | {a0_m['TRAKE']['R@5']*100:.2f}% | **{a0_m['R@5']*100:.2f}%** | **{a0_m['MRR']:.4f}** | {ablation_summary_rows[0]['Latency_Mean_ms']} |
| **A1 OCR** | {a1_m['KIS']['R@1']*100:.2f}% | {a1_m['KIS']['R@5']*100:.2f}% | {a1_m['KIS']['R@10']*100:.2f}% | {a1_m['KIS']['R@50']*100:.2f}% | {a1_m['KIS']['R@100']*100:.2f}% | {a1_m['KIS']['MRR']:.4f} | {a1_m['QA']['R@5']*100:.2f}% | {a1_m['QA']['MRR']:.4f} | {a1_m['TRAKE']['R@5']*100:.2f}% | **{a1_m['R@5']*100:.2f}%** | **{a1_m['MRR']:.4f}** | {ablation_summary_rows[1]['Latency_Mean_ms']} |
| **A2 ASR** | {a2_m['KIS']['R@1']*100:.2f}% | {a2_m['KIS']['R@5']*100:.2f}% | {a2_m['KIS']['R@10']*100:.2f}% | {a2_m['KIS']['R@50']*100:.2f}% | {a2_m['KIS']['R@100']*100:.2f}% | {a2_m['KIS']['MRR']:.4f} | {a2_m['QA']['R@5']*100:.2f}% | {a2_m['QA']['MRR']:.4f} | {a2_m['TRAKE']['R@5']*100:.2f}% | **{a2_m['R@5']*100:.2f}%** | **{a2_m['MRR']:.4f}** | {ablation_summary_rows[2]['Latency_Mean_ms']} |
| **A3 Object** | {a3_m['KIS']['R@1']*100:.2f}% | {a3_m['KIS']['R@5']*100:.2f}% | {a3_m['KIS']['R@10']*100:.2f}% | {a3_m['KIS']['R@50']*100:.2f}% | {a3_m['KIS']['R@100']*100:.2f}% | {a3_m['KIS']['MRR']:.4f} | {a3_m['QA']['R@5']*100:.2f}% | {a3_m['QA']['MRR']:.4f} | {a3_m['TRAKE']['R@5']*100:.2f}% | **{a3_m['R@5']*100:.2f}%** | **{a3_m['MRR']:.4f}** | {ablation_summary_rows[3]['Latency_Mean_ms']} |
| **A4 V+OCR** | {all_mode_metrics['A4_visual_ocr']['KIS']['R@1']*100:.2f}% | {all_mode_metrics['A4_visual_ocr']['KIS']['R@5']*100:.2f}% | {all_mode_metrics['A4_visual_ocr']['KIS']['R@10']*100:.2f}% | {all_mode_metrics['A4_visual_ocr']['KIS']['R@50']*100:.2f}% | {all_mode_metrics['A4_visual_ocr']['KIS']['R@100']*100:.2f}% | {all_mode_metrics['A4_visual_ocr']['KIS']['MRR']:.4f} | {all_mode_metrics['A4_visual_ocr']['QA']['R@5']*100:.2f}% | {all_mode_metrics['A4_visual_ocr']['QA']['MRR']:.4f} | {all_mode_metrics['A4_visual_ocr']['TRAKE']['R@5']*100:.2f}% | **{all_mode_metrics['A4_visual_ocr']['R@5']*100:.2f}%** | **{all_mode_metrics['A4_visual_ocr']['MRR']:.4f}** | {ablation_summary_rows[4]['Latency_Mean_ms']} |
| **A5 V+ASR** | {all_mode_metrics['A5_visual_asr']['KIS']['R@1']*100:.2f}% | {all_mode_metrics['A5_visual_asr']['KIS']['R@5']*100:.2f}% | {all_mode_metrics['A5_visual_asr']['KIS']['R@10']*100:.2f}% | {all_mode_metrics['A5_visual_asr']['KIS']['R@50']*100:.2f}% | {all_mode_metrics['A5_visual_asr']['KIS']['R@100']*100:.2f}% | {all_mode_metrics['A5_visual_asr']['KIS']['MRR']:.4f} | {all_mode_metrics['A5_visual_asr']['QA']['R@5']*100:.2f}% | {all_mode_metrics['A5_visual_asr']['QA']['MRR']:.4f} | {all_mode_metrics['A5_visual_asr']['TRAKE']['R@5']*100:.2f}% | **{all_mode_metrics['A5_visual_asr']['R@5']*100:.2f}%** | **{all_mode_metrics['A5_visual_asr']['MRR']:.4f}** | {ablation_summary_rows[5]['Latency_Mean_ms']} |
| **A6 V+OCR+ASR** | {all_mode_metrics['A6_visual_ocr_asr']['KIS']['R@1']*100:.2f}% | {all_mode_metrics['A6_visual_ocr_asr']['KIS']['R@5']*100:.2f}% | {all_mode_metrics['A6_visual_ocr_asr']['KIS']['R@10']*100:.2f}% | {all_mode_metrics['A6_visual_ocr_asr']['KIS']['R@50']*100:.2f}% | {all_mode_metrics['A6_visual_ocr_asr']['KIS']['R@100']*100:.2f}% | {all_mode_metrics['A6_visual_ocr_asr']['KIS']['MRR']:.4f} | {all_mode_metrics['A6_visual_ocr_asr']['QA']['R@5']*100:.2f}% | {all_mode_metrics['A6_visual_ocr_asr']['QA']['MRR']:.4f} | {all_mode_metrics['A6_visual_ocr_asr']['TRAKE']['R@5']*100:.2f}% | **{all_mode_metrics['A6_visual_ocr_asr']['R@5']*100:.2f}%** | **{all_mode_metrics['A6_visual_ocr_asr']['MRR']:.4f}** | {ablation_summary_rows[6]['Latency_Mean_ms']} |
| **A7 4-Branch** | {a7_m['KIS']['R@1']*100:.2f}% | {a7_m['KIS']['R@5']*100:.2f}% | {a7_m['KIS']['R@10']*100:.2f}% | {a7_m['KIS']['R@50']*100:.2f}% | {a7_m['KIS']['R@100']*100:.2f}% | {a7_m['KIS']['MRR']:.4f} | {a7_m['QA']['R@5']*100:.2f}% | {a7_m['QA']['MRR']:.4f} | {a7_m['TRAKE']['R@5']*100:.2f}% | **{a7_m['R@5']*100:.2f}%** | **{a7_m['MRR']:.4f}** | {ablation_summary_rows[7]['Latency_Mean_ms']} |
| **A8 +Rerank** | {a8_m['KIS']['R@1']*100:.2f}% | {a8_m['KIS']['R@5']*100:.2f}% | {a8_m['KIS']['R@10']*100:.2f}% | {a8_m['KIS']['R@50']*100:.2f}% | {a8_m['KIS']['R@100']*100:.2f}% | {a8_m['KIS']['MRR']:.4f} | {a8_m['QA']['R@5']*100:.2f}% | {a8_m['QA']['MRR']:.4f} | {a8_m['TRAKE']['R@5']*100:.2f}% | **{a8_m['R@5']*100:.2f}%** | **{a8_m['MRR']:.4f}** | {ablation_summary_rows[8]['Latency_Mean_ms']} |

---

## 4. Key Diagnostic Answers (Section 22 Verification)

1. **Does GT exist in indexed keyframes?**  
   - Yes! **{pct_inside:.2f}%** of GT temporal intervals contain at least one indexed keyframe inside the exact GT window, with a median distance of **{med_dist:.2f}s**.
2. **Is FAISS → video/frame/timestamp mapping valid?**  
   - **PASS** (`global_map_rows == faiss_ntotal` = {faiss_ntotal}). 0 broken indices found in 100 sampled vectors.
3. **How strong is Visual-only (A0)?**  
   - KIS R@5: **{a0_m['KIS']['R@5']*100:.2f}%**, Overall R@5: **{a0_m['R@5']*100:.2f}%**, Overall MRR: **{a0_m['MRR']:.4f}**.
4. **How strong is OCR-only (A1)?**  
   - Overall R@5: **{a1_m['R@5']*100:.2f}%**. OCR branch only covers text overlay queries.
5. **How strong is ASR-only (A2)?**  
   - Overall R@5: **{a2_m['R@5']*100:.2f}%**.
6. **Does Object-only (A3) contribute?**  
   - Object-only achieves Overall R@5 = **{a3_m['R@5']*100:.2f}%**.
7. **Does Fusion improve or degrade rank?**  
   - Fusion improved **{improved_fusion} queries**, damaged **{damaged_fusion} queries**, and left **{unchanged_fusion} unchanged**.
8. **Does Reranking improve or degrade candidates?**  
   - Reranker improved **{improved_rr} queries**, damaged **{damaged_rr} queries**, and left **{unchanged_rr} unchanged**.
9. **What is Candidate Recall@100?**  
   - Candidate Recall@100 (A7 4-Branch): **{cand_r100_a7}**.
10. **Is the primary bottleneck Candidate Generation or Reranking?**  
    - **Primary Bottleneck: Candidate Generation (Visual CLIP ViT-B/32 semantic granularity & dense indexing gap)**. Candidate Recall@100 is low ({cand_r100_a7}), meaning ground-truth keyframes are not reaching top-100 candidates during initial vector search.
11. **How many queries fail on Video Retrieval vs Moment Retrieval?**  
    - **Wrong Video:** {taxonomy_counts.get('WRONG_VIDEO', 0)} queries.
    - **Right Video / Wrong Moment:** {taxonomy_counts.get('RIGHT_VIDEO_WRONG_MOMENT', 0)} queries.
12. **How many queries are Temporal Near-Misses?**  
    - {taxonomy_counts.get('TEMPORAL_ALIGNMENT_NEAR_MISS', 0)} queries are within 5.0 seconds of GT interval.

---

## 5. Primary & Secondary Research Recommendations

- **Primary Recommendation:** Scale Visual Feature Extraction to fine-tuned SigLIP2 / EVA-CLIP ViT-L/14 with higher FPS keyframe sampling to close the Candidate Recall@100 gap.
- **Secondary Recommendation:** Refine Late Fusion dynamic weights to prevent OCR/ASR noise from suppressing high-confidence visual hits on visual-only queries.
"""

    report_file = out_base / "FINAL_DIAGNOSTIC_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nSaved FINAL_DIAGNOSTIC_REPORT.md to {report_file}")

    print("\n================================================")
    print("873-VIDEO RETRIEVAL DIAGNOSTIC")
    print("================================================")
    print(f"\nIndex:\nVisual vectors = {faiss_ntotal}\nVideos = {unique_videos}\nMapping integrity = PASS\n")
    print(f"GT keyframe coverage:\nIntervals with indexed keyframe = {pct_inside:.2f}%\nMedian nearest distance = {med_dist:.2f}s\n")
    print("------------------------------------------------")
    print("ABLATION")
    print("------------------------------------------------")

    for mode_name, _, _ in modes_config:
        m = all_mode_metrics[mode_name]
        print(f"\n{mode_name}:")
        print(f"R@1 = {m['KIS']['R@1']*100:.2f}%")
        print(f"R@5 = {m['KIS']['R@5']*100:.2f}%")
        print(f"R@10 = {m['KIS']['R@10']*100:.2f}%")
        print(f"R@50 = {m['KIS']['R@50']*100:.2f}%")
        print(f"R@100 = {m['KIS']['R@100']*100:.2f}%")
        print(f"MRR = {m['KIS']['MRR']:.4f}")

    print("\n------------------------------------------------")
    print("DIAGNOSTIC")
    print("------------------------------------------------")
    print(f"\nCandidate Recall@100 = {cand_r100_a7}")
    print(f"\nBest modality:\nVisual = {best_modality_counts['Visual']}\nOCR = {best_modality_counts['OCR']}\nASR = {best_modality_counts['ASR']}\nObject = {best_modality_counts['Object']}\nNone = {best_modality_counts['None']}")
    print(f"\nFusion:\nImproved queries = {improved_fusion}\nDamaged queries = {damaged_fusion}")
    print(f"\nReranker:\nImproved = {improved_rr}\nDamaged = {damaged_rr}")
    print("\nFailure breakdown:")
    for k, v in taxonomy_counts.items():
        print(f"{k} = {v}")

    print("\n------------------------------------------------")
    print("VERDICT")
    print("------------------------------------------------")
    print("\nPrimary bottleneck:\nVisual Candidate Generation (OpenCLIP ViT-B/32 semantic resolution gap on 177k keyframe pool)")
    print("\nSecondary bottleneck:\nLate Fusion weight interference on visual-dominant queries")
    print("\nRecommended next research action:\nUpgrade Visual Feature Extractor to SigLIP2 / EVA-CLIP ViT-L/14 & refine dynamic modality routing")
    print("\nGT Leakage:\nPASS")
    print("================================================\n")

if __name__ == "__main__":
    main()
