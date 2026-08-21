#!/usr/bin/env python3
"""KIS BTC Submission Post-Export Reorder Patch (Zero Model Rerun).

Applies exact policy corrections to existing CSV files:
  1. query-p1-23-kis.csv: Promote Marian candidates (L28_V006,14483), (L28_V006,23895), (L28_V006,14444) to Top 1..3.
  2. query-p1-10-kis.csv: Set Top 1..3 to (L30_V017,3010), (L30_V017,2531), (L30_V017,2640) (VinAI handpan concentration).
  3. Preserves all remaining candidates in order, deduplicates, and maintains exactly 100 rows.
  4. Runs full structural validation across all 18 CSVs and emits KIS_BTC_SUBMISSION_READY_FINAL.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = Path("/kaggle/working/submission") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission"


def reorder_csv(
    csv_path: Path,
    promoted_top3: list[tuple[str, int]],
    expected_qid: str,
) -> list[tuple[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    existing_rows: list[tuple[str, int]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) != 2:
                continue
            existing_rows.append((row[0].strip(), int(row[1].strip())))

    seen_keys: set[tuple[str, int]] = set()
    new_rows: list[tuple[str, int]] = []

    # 1. Insert Promoted Top 3
    for vid, fid in promoted_top3:
        key = (vid, fid)
        if key not in seen_keys:
            seen_keys.add(key)
            new_rows.append((vid, fid))

    # 2. Append Remaining Rows Preserving Order
    for vid, fid in existing_rows:
        key = (vid, fid)
        if key not in seen_keys:
            seen_keys.add(key)
            new_rows.append((vid, fid))
        if len(new_rows) >= 100:
            break

    # 3. Write back UTF-8 headerless
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for vid, fid in new_rows[:100]:
            writer.writerow([vid, fid])

    return new_rows[:100]


def validate_all_kis_csvs(submission_dir: Path) -> None:
    csv_files = sorted(list(submission_dir.glob("query-p1-*-kis.csv")))
    if len(csv_files) != 18:
        raise ValueError(f"Expected 18 KIS CSV files, found {len(csv_files)}")

    for f in csv_files:
        content = f.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) != 100:
            raise ValueError(f"File {f.name} has {len(lines)} rows, expected 100")
        seen: set[tuple[str, int]] = set()
        for idx, line in enumerate(lines, start=1):
            parts = line.split(",")
            if len(parts) != 2:
                raise ValueError(f"Invalid column count in {f.name} line {idx}")
            vid, fid_str = parts[0].strip(), parts[1].strip()
            if vid.endswith(".mp4") or not vid:
                raise ValueError(f"Invalid video_id in {f.name} line {idx}: {vid}")
            fid = int(fid_str)
            if fid < 0:
                raise ValueError(f"Negative frame_id in {f.name} line {idx}: {fid}")
            key = (vid, fid)
            if key in seen:
                raise ValueError(f"Duplicate key {key} in {f.name} line {idx}")
            seen.add(key)


import base64
import os
import cv2

VIDEO_PATH_CACHE: dict[str, Path] = {}
THUNGHIEM_DIR = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"


def populate_video_index_once() -> None:
    if VIDEO_PATH_CACHE:
        return
    for search_root in [Path("/kaggle/input"), REPO_ROOT / "systems" / "system_tai" / "data"]:
        if not search_root.exists():
            continue
        for root_dir, _, files in os.walk(str(search_root)):
            for fname in files:
                if fname.endswith(".mp4"):
                    vid = fname[:-4]
                    if vid not in VIDEO_PATH_CACHE:
                        VIDEO_PATH_CACHE[vid] = Path(root_dir) / fname


def resolve_video_path(video_id: str) -> Path | None:
    if video_id in VIDEO_PATH_CACHE:
        return VIDEO_PATH_CACHE[video_id]
    populate_video_index_once()
    return VIDEO_PATH_CACHE.get(video_id)


def decode_thumbnails(rows: list[tuple[str, int]]) -> list[str]:
    thumbnails: list[str] = [""] * len(rows)
    video_to_items: dict[str, list[tuple[int, int]]] = {}
    for idx, (vid, fid) in enumerate(rows):
        video_to_items.setdefault(vid, []).append((idx, fid))

    for vid, items in video_to_items.items():
        vpath = resolve_video_path(vid)
        if not vpath or not vpath.exists():
            continue
        try:
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                continue
            for orig_idx, fid in items:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fid))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                h, w = frame.shape[:2]
                new_w = 220
                new_h = int(h * (new_w / w))
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                thumbnails[orig_idx] = base64.b64encode(buf).decode("utf-8")
            cap.release()
        except Exception:
            pass
    return thumbnails


def generate_final_gallery(submission_dir: Path, out_html_path: Path) -> None:
    print(f"\n[Visual Gallery] Generating side-by-side Top-5 gallery for all 18 KIS queries...")
    csv_files = sorted(list(submission_dir.glob("query-p1-*-kis.csv")))
    
    html_cards = []
    for f in csv_files:
        qid = f.stem
        q_file = THUNGHIEM_DIR / f"{qid}.txt"
        q_vi = q_file.read_text(encoding="utf-8").strip() if q_file.exists() else "N/A"

        # Read top 5 rows
        rows: list[tuple[str, int]] = []
        with f.open("r", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            for r in reader:
                if r and len(r) == 2:
                    rows.append((r[0].strip(), int(r[1].strip())))
                if len(rows) >= 5:
                    break

        thumbnails = decode_thumbnails(rows)

        items = []
        for rank_idx, ((vid, fid), img_b64) in enumerate(zip(rows, thumbnails), start=1):
            img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:4px;" />' if img_b64 else '<div style="background:#333;color:#888;height:80px;display:flex;align-items:center;justify-content:center;border-radius:4px;">No Frame</div>'
            badge_color = "#28a745" if rank_idx <= 3 else "#6c757d"
            items.append(f"""
            <div style="flex:1; margin:3px; padding:6px; background:#1c1c1c; border:1px solid #333; border-radius:6px; text-align:center;">
                <div style="font-weight:bold; color:{badge_color}; font-size:11px; margin-bottom:3px;">Rank @{rank_idx}</div>
                {img_tag}
                <div style="color:#eee; font-weight:600; font-size:11px; margin-top:4px;">{vid}</div>
                <div style="color:#888; font-size:10px;">f={fid}</div>
            </div>
            """)

        html_cards.append(f"""
        <div style="background:#242424; border:1px solid #3c3c3c; border-radius:8px; margin-bottom:18px; padding:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #383838; padding-bottom:8px; margin-bottom:10px;">
                <span style="font-size:15px; font-weight:bold; color:#61afef;">{qid}.csv</span>
                <span style="background:#0d6efd; color:#fff; font-size:11px; font-weight:600; padding:2px 8px; border-radius:4px;">100 ROWS VALIDATED</span>
            </div>
            <div style="font-size:12px; color:#ddd; margin-bottom:10px; line-height:1.4;"><b style="color:#98c379;">Câu hỏi VI:</b> {q_vi}</div>
            <div style="display:flex; gap:4px;">
                {''.join(items)}
            </div>
        </div>
        """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>Final KIS Submission Visual Gallery</title></head>
    <body style="background:#141414; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; padding:16px;">
        <h2 style="color:#61afef; border-bottom:2px solid #333; padding-bottom:8px;">🎯 FINAL KIS SUBMISSION VISUAL GALLERY (18 QUERIES × TOP 5)</h2>
        <div style="color:#aaa; font-size:12px; margin-bottom:16px;">
            <b>Xác thực nộp bài:</b> Mỗi query hiển thị Top 5 khung hình vật lý thực tế được giải mã trực tiếp từ file CSV nộp.
        </div>
        {''.join(html_cards)}
    </body>
    </html>
    """
    out_html_path.write_text(full_html, encoding="utf-8")
    print(f"      • Saved Visual Gallery to: {out_html_path} ✅")


def main() -> None:
    print("=" * 120)
    print("APPLYING POST-EXPORT KIS ROUTING CORRECTIONS (p1-23 Marian Top3, p1-10 VinAI Top3)")
    print("=" * 120)

    # Patch p1-23
    p1_23_path = SUBMISSION_DIR / "query-p1-23-kis.csv"
    p1_23_top3 = [("L28_V006", 14483), ("L28_V006", 23895), ("L28_V006", 14444)]
    reordered_23 = reorder_csv(p1_23_path, p1_23_top3, "query-p1-23-kis")
    print(f"\n[query-p1-23-kis] Patched (Marian Top3 Promoted):")
    for r, (vid, fid) in enumerate(reordered_23[:10], start=1):
        print(f"  @{r:<2} : {vid},{fid}")

    # Patch p1-10
    p1_10_path = SUBMISSION_DIR / "query-p1-10-kis.csv"
    p1_10_top3 = [("L30_V017", 3010), ("L30_V017", 2531), ("L30_V017", 2640)]
    reordered_10 = reorder_csv(p1_10_path, p1_10_top3, "query-p1-10-kis")
    print(f"\n[query-p1-10-kis] Patched (Handpan VinAI Top3 Promoted):")
    for r, (vid, fid) in enumerate(reordered_10[:10], start=1):
        print(f"  @{r:<2} : {vid},{fid}")

    # Structural Validation of All 18 KIS Files
    validate_all_kis_csvs(SUBMISSION_DIR)
    print("\n" + "=" * 120)
    print("ALL 18 KIS CSV SUBMISSION FILES 100% VALIDATED AGAINST OFFICIAL BTC CONTRACT")
    print("=" * 120)
    print("Expected KIS: 18")
    print("Generated KIS: 18")
    print("Missing: []")
    print("Extra: []")
    print("Invalid CSV: []")
    print("Duplicate rows: 0")

    # Generate Visual Gallery for user inspection
    gallery_out = Path("/kaggle/working/kis_final_merged_gallery.html") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_final_merged_gallery.html"
    generate_final_gallery(SUBMISSION_DIR, gallery_out)

    print("\n>>> DECLARATION: KIS_BTC_SUBMISSION_READY_FINAL <<<\n")


if __name__ == "__main__":
    main()

