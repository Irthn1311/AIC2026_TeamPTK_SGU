"""
OCR V3 Final CER & WER Evaluator (Audited Frame-level & Segment-level Evaluation)
==================================================================================
Performs strict, independent evaluation of OCR accuracy:
  - Evaluation A (Frame-level): Config A (Raw), Config B (Padded), Config C (Upscaled)
  - Evaluation B (Segment-level): Config D (Temporal Consensus V3)

Exports:
  - outputs/evaluation/ocr_v3/cer_wer_final.csv
  - outputs/evaluation/ocr_v3/cer_wer_final_report.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.preprocessing.ocr_temporal_merger import levenshtein_distance, normalize_text_search


def word_levenshtein_distance(w1: list[str], w2: list[str]) -> int:
    if len(w1) < len(w2):
        return word_levenshtein_distance(w2, w1)
    if len(w2) == 0:
        return len(w1)

    previous_row = range(len(w2) + 1)
    for i, c1 in enumerate(w1):
        current_row = [i + 1]
        for j, c2 in enumerate(w2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_cer_wer(gt_texts: list[str], pred_texts: list[str]) -> tuple[float, float]:
    total_char_dist = 0
    total_char_len = 0
    total_word_dist = 0
    total_word_len = 0

    for gt, pred in zip(gt_texts, pred_texts):
        gt_norm = normalize_text_search(str(gt))
        pred_norm = normalize_text_search(str(pred))

        char_dist = levenshtein_distance(gt_norm, pred_norm)
        total_char_dist += char_dist
        total_char_len += max(1, len(gt_norm))

        w_gt = gt_norm.split()
        w_pred = pred_norm.split()
        word_dist = word_levenshtein_distance(w_gt, w_pred)
        total_word_dist += word_dist
        total_word_len += max(1, len(w_gt))

    cer = (total_char_dist / total_char_len) * 100.0 if total_char_len > 0 else 0.0
    wer = (total_word_dist / total_word_len) * 100.0 if total_word_len > 0 else 0.0
    return round(cer, 2), round(wer, 2)


def run_final_cer_wer_audit():
    print("=" * 80)
    print(" 📊 RUNNING OCR V3 FINAL AUDITED CER & WER EVALUATION")
    print("=" * 80)

    out_dir = PROJECT_ROOT / "outputs" / "evaluation" / "ocr_v3"
    csv_frame = out_dir / "ocr_ground_truth_frame_level.csv"
    csv_seg = out_dir / "ocr_ground_truth_segment_level.csv"

    if not csv_frame.exists() or not csv_seg.exists():
        print("❌ Ground Truth CSVs missing. Generating ground truth first...")
        from scripts.export_ocr_ground_truth_tool import generate_ground_truth_datasets
        generate_ground_truth_datasets()

    df_frame = pd.read_csv(csv_frame)
    df_seg = pd.read_csv(csv_seg)

    # PHẦN 1: Ground Truth Distribution Audit
    print("\n--- PHẦN 1: phân bố GROUND TRUTH DATASET ---")
    print(f"Total Frame Samples: {len(df_frame)}")
    print(f"headline samples   : {len(df_frame[df_frame['region_type'] == 'headline'])}")
    print(f"ticker samples     : {len(df_frame[df_frame['region_type'] == 'ticker'])}")
    print(f"scene_text samples : {len(df_frame[df_frame['region_type'] == 'scene_text'])}")
    print(f"logo_channel samples: {len(df_frame[df_frame['region_type'] == 'logo_channel'])}")
    print(f"clock_time samples : {len(df_frame[df_frame['region_type'] == 'clock_time'])}")

    # PHẦN 2: Headline CER Audit Sample Table
    print("\n--- PHẦN 2: BẢNG KIỂM TRA HEADLINE CER SAMPLE (20 SAMPLES) ---")
    headline_df = df_frame[df_frame["region_type"] == "headline"].copy()
    if len(headline_df) < 20:
        # Augment with top headline-like ticker samples if fewer than 20 headlines
        headline_df = pd.concat([headline_df, df_frame[df_frame["region_type"] == "ticker"]]).head(20)

    print(f"{'video_id':<10} | {'frame_id':<8} | {'CER_A':<6} | {'CER_C':<6} | {'Ground Truth':<35} | {'Config C (Lanczos 2.5x)':<35}")
    print("-" * 110)

    sample_rows = []
    for _, row in headline_df.head(20).iterrows():
        gt = str(row["ground_truth"])
        cA = str(row["ocr_raw"])
        cC = str(row["ocr_v2"])
        cer_a, _ = compute_cer_wer([gt], [cA])
        cer_c, _ = compute_cer_wer([gt], [cC])
        print(f"{row['video_id']:<10} | f{row['frame_id']:<7} | {cer_a:<6.1f} | {cer_c:<6.1f} | {gt[:33]:<35} | {cC[:33]:<35}")
        sample_rows.append({
            "video_id": row["video_id"],
            "frame_id": row["frame_id"],
            "ground_truth": gt,
            "config_A": cA,
            "config_B": row["ocr_padded"],
            "config_C": cC,
            "CER_A": cer_a,
            "CER_C": cer_c,
        })
    print("-" * 110)

    # PHẦN 3 & 4: Multi-frame Consensus Segment Audit
    print("\n--- PHẦN 3 & 4: SEGMENT TEMPORAL CONSENSUS AUDIT (Config D) ---")
    multi_frame_segs = df_seg[df_seg["num_frames"] >= 2].copy()
    print(f"Total Segments: {len(df_seg)} | Segments with >=2 frames: {len(multi_frame_segs)}")

    consensus_sample_rows = []
    for _, row in multi_frame_segs.head(20).iterrows():
        gt = str(row["ground_truth"])
        cons = str(row["text_consensus"])
        cands = json.loads(row["text_candidates"])
        cand1 = cands[0] if len(cands) > 0 else ""
        cand2 = cands[1] if len(cands) > 1 else ""

        cer_cand1, _ = compute_cer_wer([gt], [cand1])
        cer_cons, _ = compute_cer_wer([gt], [cons])

        consensus_sample_rows.append({
            "ocr_segment_id": row["ocr_segment_id"],
            "num_frames": row["num_frames"],
            "region_type": row["region_type"],
            "candidate_1": cand1,
            "candidate_2": cand2,
            "text_consensus": cons,
            "ground_truth": gt,
            "CER_cand1": cer_cand1,
            "CER_consensus": cer_cons,
        })

    # Calculate Evaluation A (Frame-level: Config A, B, C)
    results_A = {}
    configs = {"Config A (Baseline Raw)": "ocr_raw", "Config B (Padded 4px)": "ocr_padded", "Config C (Padded+2.5x Lanczos)": "ocr_v2"}
    for name, col in configs.items():
        cer, wer = compute_cer_wer(df_frame["ground_truth"].tolist(), df_frame[col].tolist())
        results_A[name] = {"CER": cer, "WER": wer}

    # Calculate Evaluation B (Segment-level: Config D Consensus)
    cer_D, wer_D = compute_cer_wer(df_seg["ground_truth"].tolist(), df_seg["text_consensus"].tolist())
    results_B = {"Config D (Temporal Consensus V3)": {"CER": cer_D, "WER": wer_D}}

    # Breakdown by ROI for Frame-level
    roi_breakdown = {}
    for roi in ["overall", "headline", "ticker", "scene_text", "logo_channel", "clock_time"]:
        if roi == "overall":
            sub_df = df_frame
        else:
            sub_df = df_frame[df_frame["region_type"] == roi]
        if len(sub_df) == 0:
            continue

        roi_res = {}
        for name, col in configs.items():
            cer, wer = compute_cer_wer(sub_df["ground_truth"].tolist(), sub_df[col].tolist())
            roi_res[name] = {"CER": cer, "WER": wer}
        roi_breakdown[roi] = roi_res

    # PHẦN 5: Print Summary Tables & Export CSV/HTML
    print("\n" + "=" * 70)
    print(" 🏆 FINAL AUDITED CER / WER SUMMARY (FRAME-LEVEL EVALUATION A)")
    print("=" * 70)
    print(f"{'Configuration':<35} | {'CER (%)':<10} | {'WER (%)':<10}")
    print("-" * 70)
    for name, metrics in results_A.items():
        print(f"{name:<35} | {metrics['CER']:<10.2f} | {metrics['WER']:<10.2f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(" 🏆 FINAL AUDITED CER / WER SUMMARY (SEGMENT-LEVEL EVALUATION B)")
    print("=" * 70)
    print(f"{'Configuration':<35} | {'CER (%)':<10} | {'WER (%)':<10}")
    print("-" * 70)
    print(f"{'Config D (Temporal Consensus V3)':<35} | {cer_D:<10.2f} | {wer_D:<10.2f}")
    print("=" * 70)

    # Save CSV & HTML reports
    export_df_rows = []
    for roi, cfgs in roi_breakdown.items():
        for cfg_name, metrics in cfgs.items():
            export_df_rows.append({"ROI": roi, "Configuration": cfg_name, "CER": metrics["CER"], "WER": metrics["WER"]})
    export_df_rows.append({"ROI": "segment_overall", "Configuration": "Config D (Temporal Consensus)", "CER": cer_D, "WER": wer_D})

    out_csv = out_dir / "cer_wer_final.csv"
    pd.DataFrame(export_df_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_html = out_dir / "cer_wer_final_report.html"
    html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>OCR V3 Final Audited CER/WER Report</title></head>
<body style="font-family: system-ui, sans-serif; padding: 20px; background: #f8f9fa;">
    <h2>📊 OCR V3 Final Audited CER & WER Evaluation Report</h2>
    <h3>Evaluation A: Frame-level (Config A vs B vs C)</h3>
    <table border="1" cellpadding="6" style="border-collapse: collapse; width: 80%; background: white;">
        <tr style="background: #1a73e8; color: white;"><th>Configuration</th><th>CER (%)</th><th>WER (%)</th></tr>
        {"".join(f"<tr><td>{k}</td><td>{v['CER']}</td><td>{v['WER']}</td></tr>" for k, v in results_A.items())}
    </table>
    <h3>Evaluation B: Segment-level (Config D Temporal Consensus)</h3>
    <table border="1" cellpadding="6" style="border-collapse: collapse; width: 80%; background: white;">
        <tr style="background: #188038; color: white;"><th>Configuration</th><th>CER (%)</th><th>WER (%)</th></tr>
        <tr><td>Config D (Temporal Consensus V3)</td><td>{cer_D}</td><td>{wer_D}</td></tr>
    </table>
</body>
</html>"""
    out_html.write_text(html_content, encoding="utf-8")
    print(f"\n📄 Saved Final CER/WER CSV: {out_csv}")
    print(f"🌐 Saved Final CER/WER HTML: {out_html}")
    return results_A, results_B


if __name__ == "__main__":
    run_final_cer_wer_audit()
