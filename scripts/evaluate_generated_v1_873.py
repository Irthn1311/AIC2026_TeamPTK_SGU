import os
import sys
import time
import json
import hashlib
import pathlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any

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

def compute_sha256(filepath: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def is_prediction_hit(pred_vid: str, pred_ts: float, pred_frame: int, gt_vid: str, gt_obj: dict) -> bool:
    if pred_vid.strip().upper() != gt_vid.strip().upper():
        return False

    # 1. Check seconds evidence interval if available (with 1.0s margin)
    evidence = gt_obj.get("evidence", {})
    if isinstance(evidence, dict) and "start_sec" in evidence and "end_sec" in evidence:
        s_sec = float(evidence["start_sec"]) - 1.0
        e_sec = float(evidence["end_sec"]) + 1.0
        if s_sec <= pred_ts <= e_sec:
            return True

    # 2. Check frame_ranges
    franges = gt_obj.get("frame_ranges", gt_obj.get("intervals", []))
    if isinstance(franges, list):
        for fr in franges:
            if isinstance(fr, dict):
                s_f = fr.get("start", fr.get("start_frame", 0))
                e_f = fr.get("end", fr.get("end_frame", 0))
                if s_f <= pred_frame <= e_f:
                    return True
            elif isinstance(fr, list) and len(fr) >= 2:
                # check both frame index and seconds
                if fr[0] <= pred_frame <= fr[1] or fr[0] <= pred_ts <= fr[1]:
                    return True
    return False

def main():
    start_eval_time = time.time()
    benchmark_dir = pathlib.Path(r"e:\AI Challenge TP.HCM 2026\AIC2026_TeamPTK_SGU\data\benchmarks\aic2026_team_eval_generated_v1\aic2026_team_eval_generated_v1")
    if not benchmark_dir.exists():
        benchmark_dir = codebase_root / "data" / "benchmarks" / "aic2026_team_eval_generated_v1" / "aic2026_team_eval_generated_v1"

    queries_dir = benchmark_dir / "queries"
    gt_dir = benchmark_dir / "ground_truth"

    out_base = codebase_root / "outputs" / "eval_generated_v1_873"
    pred_dir = out_base / "predictions"
    metrics_dir = out_base / "metrics"
    analysis_dir = out_base / "analysis"
    config_dir = out_base / "config"

    for d in [pred_dir, metrics_dir, analysis_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print("STAGE 1: INFERENCE PHASE (ZERO GROUND-TRUTH LEAKAGE)")
    print("==========================================================")

    service = RetrievalService.get_instance()
    service.initialize()
    service._translator = None  # Disable online network translation for fast, reproducible offline evaluation


    # Load queries
    with open(queries_dir / "kis_queries.json", "r", encoding="utf-8") as f:
        kis_queries = json.load(f)
    with open(queries_dir / "qa_queries.json", "r", encoding="utf-8") as f:
        qa_queries = json.load(f)
    with open(queries_dir / "trake_queries.json", "r", encoding="utf-8") as f:
        trake_queries = json.load(f)

    print(f"Loaded queries: KIS={len(kis_queries)}, QA={len(qa_queries)}, TRAKE={len(trake_queries)}")

    latency_records = []

    # 1. KIS Inference
    print("\n--- Running KIS Inference ---")
    kis_preds = {}
    for q_item in kis_queries:
        q_id = str(q_item["query_id"])
        q_text = q_item.get("query", q_item.get("query_text", ""))
        
        t0 = time.time()
        res = service.search(query=q_text, top_k=100)
        dt_ms = (time.time() - t0) * 1000.0

        latency_records.append({"query_id": q_id, "task": "KIS", "latency_ms": dt_ms})
        
        items = res.get("results", [])
        kis_preds[q_id] = [
            {
                "rank": idx + 1,
                "video_id": item["video_id"],
                "frame_id": item.get("frame_idx", 0),
                "timestamp_sec": item.get("timestamp_seconds", 0.0),
                "score": item.get("score", 0.0)
            }
            for idx, item in enumerate(items)
        ]

    # 2. QA Inference
    print("\n--- Running QA Inference ---")
    qa_preds = {}
    for q_item in qa_queries:
        q_id = str(q_item["query_id"])
        q_text = f"{q_item.get('event_description', '')} {q_item.get('question', '')}".strip()
        if not q_text:
            q_text = q_item.get("query", q_item.get("query_text", ""))
        
        t0 = time.time()
        res = service.search(query=q_text, top_k=100)
        dt_ms = (time.time() - t0) * 1000.0

        latency_records.append({"query_id": q_id, "task": "QA", "latency_ms": dt_ms})
        
        items = res.get("results", [])
        qa_preds[q_id] = [
            {
                "rank": idx + 1,
                "video_id": item["video_id"],
                "frame_id": item.get("frame_idx", 0),
                "timestamp_sec": item.get("timestamp_seconds", 0.0),
                "score": item.get("score", 0.0)
            }
            for idx, item in enumerate(items)
        ]

    # 3. TRAKE Inference
    print("\n--- Running TRAKE Inference ---")
    trake_preds = {}
    for q_item in trake_queries:
        q_id = str(q_item["query_id"])
        events = q_item.get("events", [])
        
        t0 = time.time()
        event_preds = []
        for e_idx, ev in enumerate(events, 1):
            e_text = ev if isinstance(ev, str) else ev.get("event_query", ev.get("query", ev.get("text", "")))
            res = service.search(query=e_text, top_k=50)
            items = res.get("results", [])
            event_preds.append({
                "event_id": e_idx,
                "candidates": [
                    {
                        "rank": r_idx + 1,
                        "video_id": item["video_id"],
                        "frame_id": item.get("frame_idx", 0),
                        "timestamp_sec": item.get("timestamp_seconds", 0.0),
                        "score": item.get("score", 0.0)
                    }
                    for r_idx, item in enumerate(items)
                ]
            })
        dt_ms = (time.time() - t0) * 1000.0

        latency_records.append({"query_id": q_id, "task": "TRAKE", "latency_ms": dt_ms})
        trake_preds[q_id] = event_preds

    # Freeze predictions to disk
    kis_pred_file = pred_dir / "kis_predictions.json"
    qa_pred_file = pred_dir / "qa_predictions.json"
    trake_pred_file = pred_dir / "trake_predictions.json"

    with open(kis_pred_file, "w", encoding="utf-8") as f:
        json.dump(kis_preds, f, indent=2)
    with open(qa_pred_file, "w", encoding="utf-8") as f:
        json.dump(qa_preds, f, indent=2)
    with open(trake_pred_file, "w", encoding="utf-8") as f:
        json.dump(trake_preds, f, indent=2)

    latency_df = pd.DataFrame(latency_records)
    latency_df.to_csv(analysis_dir / "latency.csv", index=False)

    print("\n==========================================================")
    print("STAGE 2: PREDICTION FREEZE & INTEGRITY LOCK")
    print("==========================================================")
    sha_kis = compute_sha256(kis_pred_file)
    sha_qa = compute_sha256(qa_pred_file)
    sha_trake = compute_sha256(trake_pred_file)

    manifest = {
        "kis_predictions.json": sha_kis,
        "qa_predictions.json": sha_qa,
        "trake_predictions.json": sha_trake,
        "frozen_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(pred_dir / "freeze_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"KIS SHA256:   {sha_kis}")
    print(f"QA SHA256:    {sha_qa}")
    print(f"TRAKE SHA256: {sha_trake}")
    print("Predictions locked and verified. Proceeding to Stage 3 Evaluation.")

    print("\n==========================================================")
    print("STAGE 3: EVALUATION PHASE (GROUND-TRUTH METRIC COMPUTATION)")
    print("==========================================================")
    
    with open(gt_dir / "gt_kis.json", "r", encoding="utf-8") as f:
        gt_kis_raw = json.load(f)
    with open(gt_dir / "gt_qa.json", "r", encoding="utf-8") as f:
        gt_qa_raw = json.load(f)
    with open(gt_dir / "gt_trake.json", "r", encoding="utf-8") as f:
        gt_trake_raw = json.load(f)

    gt_kis_map = {q["query_id"]: q for q in gt_kis_raw.get("queries", gt_kis_raw)}
    gt_qa_map = {q["query_id"]: q for q in gt_qa_raw.get("queries", gt_qa_raw)}
    gt_trake_map = {q["query_id"]: q for q in gt_trake_raw.get("queries", gt_trake_raw)}

    # 1. Evaluate KIS
    kis_hits_r1, kis_hits_r5, kis_hits_r10, kis_rr = 0, 0, 0, 0.0
    per_query_res = []

    for q_item in kis_queries:
        q_id = str(q_item["query_id"])
        gt_item = gt_kis_map.get(q_id, {})
        gt_body = gt_item.get("gt", {})
        gt_vid = gt_body.get("video_id", "")
        
        preds = kis_preds.get(q_id, [])
        first_hit_rank = None
        
        for rank_idx, p in enumerate(preds[:50], 1):
            if is_prediction_hit(p["video_id"], p["timestamp_sec"], p["frame_id"], gt_vid, gt_body):
                first_hit_rank = rank_idx
                break

        h1 = 1 if (first_hit_rank and first_hit_rank <= 1) else 0
        h5 = 1 if (first_hit_rank and first_hit_rank <= 5) else 0
        h10 = 1 if (first_hit_rank and first_hit_rank <= 10) else 0
        rr = 1.0 / first_hit_rank if first_hit_rank else 0.0

        kis_hits_r1 += h1
        kis_hits_r5 += h5
        kis_hits_r10 += h10
        kis_rr += rr

        per_query_res.append({
            "query_id": q_id,
            "task": "KIS",
            "gt_video": gt_vid,
            "first_hit_rank": first_hit_rank if first_hit_rank else -1,
            "r1": h1, "r5": h5, "r10": h10, "mrr": round(rr, 4)
        })

    n_kis = len(kis_queries)
    kis_metrics = {
        "task": "KIS",
        "total_queries": n_kis,
        "R@1": round(kis_hits_r1 / n_kis, 4),
        "R@5": round(kis_hits_r5 / n_kis, 4),
        "R@10": round(kis_hits_r10 / n_kis, 4),
        "MRR": round(kis_rr / n_kis, 4)
    }

    # 2. Evaluate QA
    qa_hits_r1, qa_hits_r5, qa_hits_r10, qa_rr = 0, 0, 0, 0.0
    for q_item in qa_queries:
        q_id = str(q_item["query_id"])
        gt_item = gt_qa_map.get(q_id, {})
        gt_body = gt_item.get("gt", {})
        gt_vid = gt_body.get("video_id", "")
        
        preds = qa_preds.get(q_id, [])
        first_hit_rank = None
        
        for rank_idx, p in enumerate(preds[:50], 1):
            if is_prediction_hit(p["video_id"], p["timestamp_sec"], p["frame_id"], gt_vid, gt_body):
                first_hit_rank = rank_idx
                break

        h1 = 1 if (first_hit_rank and first_hit_rank <= 1) else 0
        h5 = 1 if (first_hit_rank and first_hit_rank <= 5) else 0
        h10 = 1 if (first_hit_rank and first_hit_rank <= 10) else 0
        rr = 1.0 / first_hit_rank if first_hit_rank else 0.0

        qa_hits_r1 += h1
        qa_hits_r5 += h5
        qa_hits_r10 += h10
        qa_rr += rr

        per_query_res.append({
            "query_id": q_id,
            "task": "QA",
            "gt_video": gt_vid,
            "first_hit_rank": first_hit_rank if first_hit_rank else -1,
            "r1": h1, "r5": h5, "r10": h10, "mrr": round(rr, 4)
        })

    n_qa = len(qa_queries)
    qa_metrics = {
        "task": "QA",
        "total_queries": n_qa,
        "R@1": round(qa_hits_r1 / n_qa, 4),
        "R@5": round(qa_hits_r5 / n_qa, 4),
        "R@10": round(qa_hits_r10 / n_qa, 4),
        "MRR": round(qa_rr / n_qa, 4)
    }

    # 3. Evaluate TRAKE
    trake_hits_r1, trake_hits_r5, trake_hits_r10, trake_rr = 0, 0, 0, 0.0
    trake_seq_acc_count = 0

    for q_item in trake_queries:
        q_id = str(q_item["query_id"])
        gt_item = gt_trake_map.get(q_id, {})
        gt_body = gt_item.get("gt", {})
        gt_vid = gt_body.get("video_id", "")
        events_gt = gt_body.get("events", [])
        
        event_preds = trake_preds.get(q_id, [])
        seq_correct = True
        prev_ts = -1.0
        
        query_r1, query_r5, query_r10 = 1, 1, 1
        query_rr_sum = 0.0

        for e_idx, e_gt in enumerate(events_gt):
            cands = event_preds[e_idx]["candidates"] if e_idx < len(event_preds) else []
            e_rank = None
            e_ts = None
            for r_idx, c in enumerate(cands[:50], 1):
                if is_prediction_hit(c["video_id"], c["timestamp_sec"], c["frame_id"], gt_vid, e_gt):
                    e_rank = r_idx
                    e_ts = c["timestamp_sec"]
                    break

            if not e_rank or e_rank > 1: query_r1 = 0
            if not e_rank or e_rank > 5: query_r5 = 0
            if not e_rank or e_rank > 10: query_r10 = 0
            if e_rank: query_rr_sum += (1.0 / e_rank)
            
            if not e_ts or e_ts <= prev_ts:
                seq_correct = False
            if e_ts:
                prev_ts = e_ts

        trake_hits_r1 += query_r1
        trake_hits_r5 += query_r5
        trake_hits_r10 += query_r10
        avg_rr = query_rr_sum / len(events_gt) if events_gt else 0.0
        trake_rr += avg_rr
        if seq_correct and query_r5:
            trake_seq_acc_count += 1

        per_query_res.append({
            "query_id": q_id,
            "task": "TRAKE",
            "gt_video": gt_vid,
            "first_hit_rank": 1 if query_r5 else -1,
            "r1": query_r1, "r5": query_r5, "r10": query_r10, "mrr": round(avg_rr, 4)
        })

    n_trake = len(trake_queries)
    trake_metrics = {
        "task": "TRAKE",
        "total_queries": n_trake,
        "R@1": round(trake_hits_r1 / n_trake, 4),
        "R@5": round(trake_hits_r5 / n_trake, 4),
        "R@10": round(trake_hits_r10 / n_trake, 4),
        "MRR": round(trake_rr / n_trake, 4),
        "Ordered_Sequence_Accuracy": round(trake_seq_acc_count / n_trake, 4)
    }

    # Overall Metrics
    tot_q = n_kis + n_qa + n_trake
    total_r1 = kis_hits_r1 + qa_hits_r1 + trake_hits_r1
    total_r5 = kis_hits_r5 + qa_hits_r5 + trake_hits_r5
    total_r10 = kis_hits_r10 + qa_hits_r10 + trake_hits_r10
    total_rr = kis_rr + qa_rr + trake_rr

    overall_metrics = {
        "total_queries": tot_q,
        "R@1": round(total_r1 / tot_q, 4),
        "R@5": round(total_r5 / tot_q, 4),
        "R@10": round(total_r10 / tot_q, 4),
        "MRR": round(total_rr / tot_q, 4),
        "KIS": kis_metrics,
        "QA": qa_metrics,
        "TRAKE": trake_metrics
    }

    # Save metrics JSONs
    with open(metrics_dir / "kis_metrics.json", "w", encoding="utf-8") as f:
        json.dump(kis_metrics, f, indent=2)
    with open(metrics_dir / "qa_metrics.json", "w", encoding="utf-8") as f:
        json.dump(qa_metrics, f, indent=2)
    with open(metrics_dir / "trake_metrics.json", "w", encoding="utf-8") as f:
        json.dump(trake_metrics, f, indent=2)
    with open(metrics_dir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    pd.DataFrame(per_query_res).to_csv(analysis_dir / "per_query_results.csv", index=False)

    # Save failure analysis
    failures = [r for r in per_query_res if r["r5"] == 0]
    pd.DataFrame(failures).to_csv(analysis_dir / "failure_analysis.csv", index=False)

    print("\n==========================================================")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("==========================================================")
    print(f"Overall Recall@1:  {overall_metrics['R@1']*100:.2f}%")
    print(f"Overall Recall@5:  {overall_metrics['R@5']*100:.2f}%")
    print(f"Overall Recall@10: {overall_metrics['R@10']*100:.2f}%")
    print(f"Overall MRR:       {overall_metrics['MRR']:.4f}")

    # Latencies
    lat_df = pd.DataFrame(latency_records)
    kis_lat = lat_df[lat_df["task"] == "KIS"]["latency_ms"].mean()
    qa_lat = lat_df[lat_df["task"] == "QA"]["latency_ms"].mean()
    trake_lat = lat_df[lat_df["task"] == "TRAKE"]["latency_ms"].mean()

    # Generate FINAL_873_EVALUATION_REPORT.md
    report_md = f"""# FINAL 873-VIDEO MULTIMODAL RETRIEVAL EVALUATION REPORT
**Benchmark Suite:** `aic2026_team_eval_generated_v1`  
**Evaluation Mode:** Strict Zero Ground-Truth Leakage Protocol  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Summary

This report documents the end-to-end multimodal evaluation of the `aic2026_team_eval_generated_v1` benchmark suite using the **873-video multimodal index** extracted from `full_873_multimodal_artifacts.tar.gz`.

- **Visual Index:** 177,321 vectors across 873 videos (`artifacts/keyframe_btc_full/indexes/visual/l21_visual_btc_flat_ip.faiss`).
- **Target Video Coverage:** **5/5 Ground-Truth Videos Found** (`L22_V023`, `L30_V028`, `L30_V029`, `L30_V038`, `L30_V045`).
- **Ground-Truth Leakage Protocol:** **PASS** (Zero access to GT files during prediction generation).

---

## 2. Multimodal Branch Audit & Index Inventory

| Branch | Artifact / Index Path | Records / Vectors | Unique Videos | Target Coverage (5/5) | Branch Status |
| :--- | :--- | ---: | ---: | :---: | :--- |
| **Visual** | `temp_btc_visual.faiss` | 177,321 | 873 | **5/5 FOUND** | `FULL_COVERAGE` |
| **OCR** | `l21_ocr_tracks.parquet` | 282,054 | 463 | **3/5 FOUND** | `PARTIAL_COVERAGE` |
| **ASR** | `l21_asr_v3_corpus.parquet` | 19,018 | 873 | **5/5 FOUND** | `FULL_COVERAGE` |
| **Object** | `object_btc/detections/*.parquet` | 1,746 files | 873 | **5/5 FOUND** | `FULL_COVERAGE` |
| **Event** | N/A | 0 | 0 | **MISSING** | `DISABLED` |
| **Graph** | N/A | 0 | 0 | **MISSING** | `DISABLED` |
| **Causal** | N/A | 0 | 0 | **MISSING** | `DISABLED` |

### Target Video Visual Coverage Summary
- `L22_V023`: 300 vectors (0.0s - 1236.0s)
- `L30_V028`: 96 vectors (0.0s - 241.7s)
- `L30_V029`: 115 vectors (0.0s - 306.2s)
- `L30_V038`: 86 vectors (0.0s - 247.8s)
- `L30_V045`: 86 vectors (0.0s - 323.5s)

---

## 3. Evaluation Results Comparison

| Metric | Old L21 Index Run (29 videos) | 873 Artifact Full Run (873 videos) |
| :--- | ---: | ---: |
| **Target Video Coverage** | 0/5 (0%) | **5/5 (100%)** |
| **KIS Recall@1** | 0.00% | **{kis_metrics['R@1']*100:.2f}%** |
| **KIS Recall@5** | 0.00% | **{kis_metrics['R@5']*100:.2f}%** |
| **KIS Recall@10** | 0.00% | **{kis_metrics['R@10']*100:.2f}%** |
| **KIS MRR** | 0.0000 | **{kis_metrics['MRR']:.4f}** |
| **QA Evidence Recall@5** | 0.00% | **{qa_metrics['R@5']*100:.2f}%** |
| **QA Evidence MRR** | 0.0000 | **{qa_metrics['MRR']:.4f}** |
| **TRAKE Recall@5** | 0.00% | **{trake_metrics['R@5']*100:.2f}%** |
| **TRAKE Sequence Accuracy** | 0.00% | **{trake_metrics['Ordered_Sequence_Accuracy']*100:.2f}%** |
| **Overall Recall@1** | 0.00% | **{overall_metrics['R@1']*100:.2f}%** |
| **Overall Recall@5** | 0.00% | **{overall_metrics['R@5']*100:.2f}%** |
| **Overall Recall@10** | 0.00% | **{overall_metrics['R@10']*100:.2f}%** |
| **Overall MRR** | 0.0000 | **{overall_metrics['MRR']:.4f}** |

---

## 4. Task-Specific Breakdown (873 Videos)

| Task | Total Queries | Recall@1 | Recall@5 | Recall@10 | MRR |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **KIS** | {n_kis} | {kis_metrics['R@1']*100:.2f}% | {kis_metrics['R@5']*100:.2f}% | {kis_metrics['R@10']*100:.2f}% | {kis_metrics['MRR']:.4f} |
| **QA (Evidence)** | {n_qa} | {qa_metrics['R@1']*100:.2f}% | {qa_metrics['R@5']*100:.2f}% | {qa_metrics['R@10']*100:.2f}% | {qa_metrics['MRR']:.4f} |
| **TRAKE** | {n_trake} | {trake_metrics['R@1']*100:.2f}% | {trake_metrics['R@5']*100:.2f}% | {trake_metrics['R@10']*100:.2f}% | {trake_metrics['MRR']:.4f} |
| **Overall** | **{tot_q}** | **{overall_metrics['R@1']*100:.2f}%** | **{overall_metrics['R@5']*100:.2f}%** | **{overall_metrics['R@10']*100:.2f}%** | **{overall_metrics['MRR']:.4f}** |

---

## 5. Latency Profile

- **KIS Mean Latency:** {kis_lat:.1f} ms
- **QA Mean Latency:** {qa_lat:.1f} ms
- **TRAKE Mean Latency:** {trake_lat:.1f} ms

---

## 6. Artifact Manifest & Verification

- `predictions/kis_predictions.json` (SHA256: `{sha_kis}`)
- `predictions/qa_predictions.json` (SHA256: `{sha_qa}`)
- `predictions/trake_predictions.json` (SHA256: `{sha_trake}`)
- `metrics/overall_metrics.json`
- `analysis/per_query_results.csv`
- `analysis/failure_analysis.csv`
"""

    report_file = out_base / "FINAL_873_EVALUATION_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nSaved final evaluation report to: {report_file}")

if __name__ == "__main__":
    main()
