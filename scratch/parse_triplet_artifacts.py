# ==============================================================================================================
# Quick Parser to display Triplet Forensic Probe results from existing artifacts on Kaggle
# ==============================================================================================================

import json
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path("/kaggle/working/output/triplet_probe") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "triplet_probe"
BENCHMARK_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
SIDECAR_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "l21_150_diagnostic" / "qa_dev_translations_en.json"


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()


def interval_distance(f: int, start: int, end: int) -> int:
    if start <= f <= end:
        return 0
    return start - f if f < start else f - end


def parse_existing_artifacts():
    print("=" * 135)
    print("PARSING SAVED TRIPLET FORENSIC ARTIFACTS FROM:", OUTPUT_DIR)
    print("=" * 135)

    if not OUTPUT_DIR.exists():
        print(f"Output directory {OUTPUT_DIR} does not exist!")
        return

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        bm = json.load(f)
    all_qa = {q["query_id"]: q for q in bm["queries"] if q.get("task_type") == "qa"}

    with open(SIDECAR_PATH, encoding="utf-8") as f:
        en_sidecar = json.load(f)
    en_map = {e["query_id"]: e.get("question_en", "") for e in en_sidecar.get("entries", [])}

    # Find all evidence files
    ev_files = sorted(OUTPUT_DIR.rglob("*qa_evidence*.json"))
    pred_files = sorted(OUTPUT_DIR.rglob("*predictions*.json"))

    print(f"Found {len(ev_files)} evidence files and {len(pred_files)} prediction files.\n")

    # Map request_id / qid -> files
    queries_to_inspect = ["QA-31", "QA-01", "QA-26", "QA-23", "QA-30", "QA-02"]

    for qid in queries_to_inspect:
        q = all_qa.get(qid, {})
        target_vid = q.get("video_id")
        start_f, end_f = int(q.get("proposed_interval", [-1, -1])[0]), int(q.get("proposed_interval", [-1, -1])[1])
        gt_answers = [normalize_text(a) for a in q.get("accepted_answers", [])]

        matching_ev = [f for f in ev_files if qid in f.name]
        matching_pred = [f for f in pred_files if qid in f.name]

        print("-" * 110)
        print(f"QUERY: {qid} [Target: {target_vid}, GT: [{start_f}..{end_f}], Expected Ans: {gt_answers}]")

        if not matching_ev and not matching_pred:
            print("  No artifact files found for this query.")
            continue

        diags = {}
        if matching_ev:
            with open(matching_ev[0], encoding="utf-8") as ef:
                diags = json.load(ef)

        preds = []
        if matching_pred:
            with open(matching_pred[0], encoding="utf-8") as pf:
                data = json.load(pf)
                preds = data.get("predictions", data) if isinstance(data, dict) else data

        selected_videos = diags.get("selected_video_ids", [])
        target_in_pool = target_vid in selected_videos
        vid_rank = selected_videos.index(target_vid) + 1 if target_in_pool else "ABSENT"

        print(f"  1. Selected Videos (Top 16): {selected_videos}")
        print(f"     -> Target Video {target_vid} Rank: {vid_rank}")

        scored_ev = diags.get("evidence", [])
        target_ev = [r for r in scored_ev if r.get("video_id") == target_vid]

        target_ev_records = []
        for ev in target_ev:
            f_id = ev.get("candidate_frame_id") or ev.get("evidence_frame_id") or ev.get("frame_id")
            if f_id is not None:
                ans = normalize_text(str(ev.get("answer", "")))
                score = float(ev.get("answer_score") or 0.0)
                source = ev.get("evidence_source", "UNKNOWN")
                ref_status = ev.get("refinement_status", "NOT_REFINED")
                target_ev_records.append({
                    "frame_id": int(f_id),
                    "answer": ans,
                    "score": score,
                    "source": source,
                    "refinement": ref_status,
                })

        print(f"  2. Target Evidence Records ({len(target_ev_records)} records):")
        for i, rec in enumerate(target_ev_records, 1):
            in_gt = "IN_GT ✅" if start_f <= rec["frame_id"] <= end_f else "OUT ❌"
            dist = interval_distance(rec["frame_id"], start_f, end_f)
            print(f"     [{i}] f={rec['frame_id']:<6} ({in_gt}, dist={dist:<5}) | src={rec['source']:<16} | ref={rec['refinement']:<12} | ans='{rec['answer']}' (score={rec['score']:.3f})")

        # Top 100 predictions check
        hit_rank = None
        for p in preds:
            p_vid = p.get("video_id")
            p_frame = int(p.get("frame_id", -1))
            p_ans = normalize_text(str(p.get("answer", "")))
            if p_vid == target_vid and start_f <= p_frame <= end_f and p_ans in gt_answers:
                hit_rank = p.get("rank")
                break

        print(f"  3. Final Top 100 Hit: {'Rank ' + str(hit_rank) + ' ✅' if hit_rank else 'NO ❌'}")

        # Dispersion metrics
        frame_list = [r["frame_id"] for r in target_ev_records]
        if frame_list:
            min_f, max_f = min(frame_list), max(frame_list)
            spread = max_f - min_f
            nearest_dist = min([interval_distance(f, start_f, end_f) for f in frame_list])
            nearest_f = min(frame_list, key=lambda f: interval_distance(f, start_f, end_f))

            sorted_frames = sorted(set(frame_list))
            clusters = [[sorted_frames[0]]]
            for f in sorted_frames[1:]:
                if f - clusters[-1][-1] <= 500:
                    clusters[-1].append(f)
                else:
                    clusters.append([f])

            if nearest_dist <= 250:
                diag = f"BOUNDED NEAR MISS (Distance {nearest_dist} <= 250 frames -> Ideal candidate for Local Window Expansion)"
            elif len(clusters) == 1 and spread < 500:
                diag = f"WRONG SEGMENT MODE COLLAPSE (Distance {nearest_dist} frames, all candidates clustered at ~{min_f}..{max_f} -> Needs Diverse Temporal Anchors)"
            else:
                diag = f"CATASTROPHIC WRONG SEGMENT / MULTI-CLUSTER (Distance {nearest_dist} frames, spread {spread} frames)"

            print(f"  4. Dispersion & Diagnosis:")
            print(f"     - Clusters: {len(clusters)} -> {[f'{min(c)}..{max(c)} (n={len(c)})' for c in clusters]}")
            print(f"     - Min/Max: {min_f} .. {max_f} (Spread: {spread} frames)")
            print(f"     - Nearest Frame: {nearest_f} (Distance: {nearest_dist} frames)")
            print(f"     - Diagnosis: {diag}")


if __name__ == "__main__":
    parse_existing_artifacts()
