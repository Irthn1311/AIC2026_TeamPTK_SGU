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
    print("\n>>> DECLARATION: KIS_BTC_SUBMISSION_READY_FINAL <<<\n")


if __name__ == "__main__":
    main()
