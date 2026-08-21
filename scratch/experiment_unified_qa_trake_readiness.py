#!/usr/bin/env python3
"""Unified QA + TRAKE BTC Operational Readiness & Full Submission Packager.

Strict Constraints:
  - 100% FROZEN RUNTIME: Zero algorithmic changes to QA Champion or TRAKE Pipeline.
  - Dynamically discovers all queries in THUNGHIEM_20-8:
      • KIS queries (*-kis.txt)
      • TRAKE queries (*-trake.txt)
      • QA queries (*-qa.txt)
  - Executes TRAKE queries:
      • video_id,f1,f2,...,fN (count of frame IDs == N exactly).
      • Single video, non-decreasing frames, <= 100 rows.
      • Empty output (0 rows) is flagged as OPERATIONAL_FAIL_EMPTY_OUTPUT.
  - Executes QA queries:
      • video_id,official_frame_id,answer (answer <= 100 chars, standard CSV writer quoting).
      • <= 100 rows.
      • Empty output is flagged as OPERATIONAL_FAIL_EMPTY_OUTPUT.
  - Preserves already-generated KIS CSV files in submission/.
  - Performs Global Coverage & Integrity Audit (Expected == Generated).
  - Automatically creates submission.zip when all queries are 100% valid and non-empty.
  - Emits BTC_P1_FULL_SUBMISSION_READY upon complete success.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

print("=" * 150, flush=True)
print("UNIFIED QA + TRAKE BTC OPERATIONAL READINESS & SUBMISSION PACKAGER (100% FROZEN RUNTIME)", flush=True)
print("=" * 150, flush=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

try:
    import clip
except ImportError:
    print("Installing official openai-clip dependency ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/openai/CLIP.git", "ftfy", "regex"], check=True)
    import clip

try:
    import cv2
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"], check=False)
    import cv2

import torch
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    QAQueryRequest,
    SessionConfig,
    TRAKEQueryRequest,
)

THUNGHIEM_DIR = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"


def get_reuse_manifest() -> Path | None:
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
        REPO_ROOT / "systems" / "system_tai" / "data" / "feature_manifest.json",
    ]:
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def parse_trake_query(file_path: Path) -> tuple[str, list[dict[str, str]]]:
    content = file_path.read_text(encoding="utf-8").strip()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    events: list[dict[str, str]] = []
    for line in lines:
        match = re.match(r"^(?:E\d+|Sự kiện \d+|\d+\.)\s*:\s*(.+)$", line, re.IGNORECASE)
        if match:
            events.append({"description": match.group(1).strip()})
        elif line.startswith("E") and ":" in line:
            parts = line.split(":", 1)
            events.append({"description": parts[1].strip()})
        elif not events and ("tìm các sự kiện" in line.lower() or "gồm các khoảnh khắc" in line.lower()):
            continue
        elif events:
            events.append({"description": line})

    if not events:
        for line in lines:
            events.append({"description": line})

    return file_path.stem, events


def parse_qa_query(file_path: Path) -> tuple[str, str, str]:
    content = file_path.read_text(encoding="utf-8").strip()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    full_text = " ".join(lines)
    
    # Identify question part vs event description part
    # Look for 'Hỏi ' or sentence ending with '?'
    q_part = full_text
    desc_part = full_text
    
    match = re.search(r"(Hỏi\s+.+\?.*)$", full_text, re.IGNORECASE)
    if match:
        q_part = match.group(1).strip()
        desc_part = full_text[:match.start()].strip()
    elif "?" in full_text:
        parts = full_text.split("?")
        q_part = parts[-2].strip() + "?" if len(parts) >= 2 else full_text
        desc_part = "?".join(parts[:-2]).strip() if len(parts) >= 3 else parts[0].strip()

    if not desc_part:
        desc_part = full_text

    return file_path.stem, desc_part, q_part


def validate_trake_csv(csv_path: Path, expected_event_count: int) -> list[str]:
    errors: list[str] = []
    if not csv_path.exists():
        return [f"File does not exist: {csv_path}"]
    content = csv_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) == 0:
        return [f"EMPTY OUTPUT: File {csv_path.name} has 0 rows (OPERATIONAL_FAIL_EMPTY_OUTPUT)"]
    if len(lines) > 100:
        errors.append(f"Exceeds 100 rows ({len(lines)} rows)")

    seen: set[tuple[str, tuple[int, ...]]] = set()
    expected_cols = 1 + expected_event_count
    for idx, line in enumerate(lines, start=1):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) != expected_cols:
            errors.append(f"Line {idx}: Column count {len(parts)} != expected {expected_cols}")
            continue
        vid = parts[0]
        if vid.endswith(".mp4") or not vid:
            errors.append(f"Line {idx}: Invalid video_id '{vid}'")
        fids: list[int] = []
        for f_str in parts[1:]:
            try:
                fid = int(f_str)
                if fid < 0:
                    errors.append(f"Line {idx}: Negative frame_id {fid}")
                fids.append(fid)
            except ValueError:
                errors.append(f"Line {idx}: Invalid integer '{f_str}'")
        for i in range(len(fids) - 1):
            if fids[i] > fids[i + 1]:
                errors.append(f"Line {idx}: Temporal order violation: {fids}")
        key = (vid, tuple(fids))
        if key in seen:
            errors.append(f"Line {idx}: Duplicate chain {key}")
        seen.add(key)

    return errors


def validate_qa_csv(csv_path: Path) -> list[str]:
    errors: list[str] = []
    if not csv_path.exists():
        return [f"File does not exist: {csv_path}"]
    content = csv_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) == 0:
        return [f"EMPTY OUTPUT: File {csv_path.name} has 0 rows (OPERATIONAL_FAIL_EMPTY_OUTPUT)"]
    if len(lines) > 100:
        errors.append(f"Exceeds 100 rows ({len(lines)} rows)")

    seen: set[tuple[str, int, str]] = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, parts in enumerate(reader, start=1):
            if not parts:
                continue
            if len(parts) != 3:
                errors.append(f"Line {idx}: Column count {len(parts)} != 3")
                continue
            vid, fid_str, ans = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if vid.endswith(".mp4") or not vid:
                errors.append(f"Line {idx}: Invalid video_id '{vid}'")
            try:
                fid = int(fid_str)
                if fid < 0:
                    errors.append(f"Line {idx}: Negative frame_id {fid}")
            except ValueError:
                errors.append(f"Line {idx}: Invalid integer frame_id '{fid_str}'")
            if len(ans) > 100:
                errors.append(f"Line {idx}: Answer exceeds 100 chars ({len(ans)} chars)")
            key = (vid, fid, ans)
            if key in seen:
                errors.append(f"Line {idx}: Duplicate row {key}")
            seen.add(key)

    return errors


def validate_kis_csv(csv_path: Path) -> list[str]:
    errors: list[str] = []
    if not csv_path.exists():
        return [f"File does not exist: {csv_path}"]
    content = csv_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) == 0:
        return [f"EMPTY OUTPUT: File {csv_path.name} has 0 rows"]
    if len(lines) > 100:
        errors.append(f"Exceeds 100 rows ({len(lines)} rows)")
    seen: set[tuple[str, int]] = set()
    for idx, line in enumerate(lines, start=1):
        parts = line.split(",")
        if len(parts) != 2:
            errors.append(f"Line {idx}: Column count != 2")
            continue
        vid, fid_str = parts[0].strip(), parts[1].strip()
        if vid.endswith(".mp4") or not vid:
            errors.append(f"Line {idx}: Invalid video_id")
        try:
            fid = int(fid_str)
            if fid < 0:
                errors.append(f"Line {idx}: Negative frame_id")
        except ValueError:
            errors.append(f"Line {idx}: Invalid integer frame_id")
        key = (vid, fid)
        if key in seen:
            errors.append(f"Line {idx}: Duplicate {key}")
        seen.add(key)
    return errors


def run_unified_readiness() -> None:
    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest = get_reuse_manifest()
    out_dir = Path("/kaggle/working/output/unified_readiness_session") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "unified_readiness_session"

    cfg = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_dir,
        reuse_manifest=reuse_manifest,
    )

    # 1. Dynamic Discovery of ALL BTC Queries
    kis_files = sorted(list(THUNGHIEM_DIR.glob("*kis*.txt")))
    trake_files = sorted(list(THUNGHIEM_DIR.glob("*trake*.txt")))
    qa_files = sorted(list(THUNGHIEM_DIR.glob("*qa*.txt")))

    all_expected_files = kis_files + trake_files + qa_files
    expected_qids = [f.stem for f in all_expected_files]

    print(f"\n[Dynamic Discovery] Total {len(expected_qids)} queries found in {THUNGHIEM_DIR}:")
    print(f"  • KIS Queries   : {len(kis_files)} ({', '.join([f.stem for f in kis_files])})")
    print(f"  • TRAKE Queries : {len(trake_files)} ({', '.join([f.stem for f in trake_files])})")
    print(f"  • QA Queries    : {len(qa_files)} ({', '.join([f.stem for f in qa_files])})")
    print(f"  • TOTAL         : {len(expected_qids)}")

    # 2. Bootstrap Operational Runtime
    print("\n[1/3] Bootstrapping OperationalKISRuntime (Frozen QA Champion & TRAKE Pipeline)...", flush=True)
    t0_rt = time.time()
    runtime = OperationalKISRuntime.bootstrap(cfg)
    device = runtime.shared_encoder.identifiers.get("device", "cpu")
    if torch.cuda.is_available():
        device = "cuda"
    print(f"      • Runtime Bootstrapped in {time.time() - t0_rt:.2f}s (device={device}) ✅", flush=True)

    submission_dir = Path("/kaggle/working/submission") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    operational_failures: list[dict[str, Any]] = []

    # 3. Execute TRAKE Queries
    print("\n" + "=" * 150, flush=True)
    print("EXECUTING FROZEN TRAKE PIPELINE (READ-ONLY OPERATIONAL AUDIT)", flush=True)
    print("=" * 150, flush=True)

    for f in trake_files:
        qid, events = parse_trake_query(f)
        t0_q = time.time()
        req = TRAKEQueryRequest(
            request_id=f"req-{qid}",
            query_id=qid,
            events=tuple(events),
            include_vi_variant=True,
            top_k_per_variant=100,
            event_candidate_top_k=100,
            output_top_k=100,
            beam_width=100,
            refine_top_n=3,
        )

        resp = runtime.handle_trake_query(req)
        lat_q = time.time() - t0_q

        req_dir_name = qid
        query_out_dir = out_dir / "requests" / req_dir_name
        predictions_jsonl = query_out_dir / "trake_predictions.jsonl"

        predictions: list[dict[str, Any]] = []
        if predictions_jsonl.exists():
            with predictions_jsonl.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        predictions.append(json.loads(line))

        csv_path = submission_dir / f"{qid}.csv"
        if len(predictions) == 0:
            zero_reason = resp.get("diagnostics", {}).get("zero_output_reason", "unknown_zero_output")
            print(f"❌ [TRAKE FAIL] {qid}: OPERATIONAL_FAIL_EMPTY_OUTPUT (zero_output_reason={zero_reason})", flush=True)
            operational_failures.append({"qid": qid, "task": "TRAKE", "reason": f"zero_output_reason={zero_reason}"})
            # Do not create empty CSV
        else:
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                for p in predictions[:100]:
                    vid = str(p["video_id"]).removesuffix(".mp4")
                    fids = [int(fid) for fid in p["frame_ids"]]
                    writer.writerow([vid, *fids])

            val_errs = validate_trake_csv(csv_path, expected_event_count=len(events))
            if val_errs:
                print(f"❌ [TRAKE INVALID] {qid}: {val_errs}", flush=True)
                operational_failures.append({"qid": qid, "task": "TRAKE", "reason": str(val_errs)})
            else:
                top3_str = " | ".join([f"@{r}:{p['video_id']}({','.join(map(str, p['frame_ids']))})" for r, p in enumerate(predictions[:3], start=1)])
                print(f"✅ [{qid}] (N={len(events)} events | Rows: {len(predictions)}/100 | Lat: {lat_q*1000:.0f}ms)")
                print(f"    - Top 3 Chains: {top3_str}")
                print(f"    - File Written: {csv_path} (Validated ✅)")

    # 4. Execute QA Queries
    print("\n" + "=" * 150, flush=True)
    print("EXECUTING FROZEN QA CHAMPION PIPELINE (READ-ONLY OPERATIONAL AUDIT)", flush=True)
    print("=" * 150, flush=True)

    for f in qa_files:
        qid, desc, question = parse_qa_query(f)
        t0_q = time.time()
        req = QAQueryRequest(
            request_id=f"req-{qid}",
            query_id=qid,
            event_description=desc,
            question=question,
            include_vi_variant=True,
            top_k_per_variant=100,
            output_top_k=100,
            refine_top_n=3,
        )

        resp = runtime.handle_qa_query(req)
        lat_q = time.time() - t0_q

        req_dir_name = qid
        query_out_dir = out_dir / "requests" / req_dir_name
        predictions_jsonl = query_out_dir / "qa_predictions.jsonl"

        predictions: list[dict[str, Any]] = []
        if predictions_jsonl.exists():
            with predictions_jsonl.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        predictions.append(json.loads(line))

        csv_path = submission_dir / f"{qid}.csv"
        if len(predictions) == 0:
            unsupported_reason = resp.get("unsupported_reason") or resp.get("diagnostics", {}).get("zero_output_reason", "unknown_unsupported")
            print(f"❌ [QA FAIL] {qid}: OPERATIONAL_FAIL_EMPTY_OUTPUT (unsupported_reason={unsupported_reason})", flush=True)
            operational_failures.append({"qid": qid, "task": "QA", "reason": f"unsupported_reason={unsupported_reason}"})
        else:
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                for p in predictions[:100]:
                    vid = str(p["video_id"]).removesuffix(".mp4")
                    fid = int(p["frame_id"])
                    ans = str(p["answer"]).strip()[:100]
                    writer.writerow([vid, fid, ans])

            val_errs = validate_qa_csv(csv_path)
            if val_errs:
                print(f"❌ [QA INVALID] {qid}: {val_errs}", flush=True)
                operational_failures.append({"qid": qid, "task": "QA", "reason": str(val_errs)})
            else:
                top3_str = " | ".join([f"@{r}:{p['video_id']}(f={p['frame_id']}, ans='{p['answer']}')" for r, p in enumerate(predictions[:3], start=1)])
                print(f"✅ [{qid}] (Rows: {len(predictions)}/100 | Lat: {lat_q*1000:.0f}ms)")
                print(f"    - Top 3 Predictions: {top3_str}")
                print(f"    - File Written     : {csv_path} (Validated ✅)")

    # 5. Global Submission Package Audit & Verification
    print("\n" + "=" * 150, flush=True)
    print("GLOBAL SUBMISSION PACKAGE INTEGRITY AUDIT", flush=True)
    print("=" * 150, flush=True)

    csv_files = sorted(list(submission_dir.glob("query-p1-*.csv")))
    generated_qids = [f.stem for f in csv_files]

    missing_qids = set(expected_qids) - set(generated_qids)
    extra_qids = set(generated_qids) - set(expected_qids)

    # Validate all existing CSV files in submission directory
    all_invalid: list[str] = []
    for f in csv_files:
        if "-kis" in f.name:
            errs = validate_kis_csv(f)
        elif "-trake" in f.name:
            # find event count
            tr_file = THUNGHIEM_DIR / f"{f.stem}.txt"
            _, evs = parse_trake_query(tr_file)
            errs = validate_trake_csv(f, expected_event_count=len(evs))
        elif "-qa" in f.name:
            errs = validate_qa_csv(f)
        else:
            errs = [f"Unknown query format in {f.name}"]
        if errs:
            all_invalid.append(f"{f.name}: {errs}")

    print(f"Expected Queries : {len(expected_qids)}")
    print(f"Generated CSVs   : {len(generated_qids)}")
    print(f"Missing Queries  : {sorted(list(missing_qids))}")
    print(f"Extra CSVs       : {sorted(list(extra_qids))}")
    print(f"Invalid / Empty  : {all_invalid}")
    print(f"Failures Logged  : {operational_failures}")

    if (
        len(missing_qids) == 0
        and len(extra_qids) == 0
        and len(all_invalid) == 0
        and len(operational_failures) == 0
        and len(generated_qids) == len(expected_qids)
    ):
        # Create submission.zip
        zip_path = Path("/kaggle/working/submission.zip") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "submission.zip"
        print(f"\n[Packaging] Creating final submission.zip at {zip_path} ...", flush=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for csv_f in csv_files:
                arcname = f"submission/{csv_f.name}"
                zipf.write(csv_f, arcname=arcname)
        print(f"      • Successfully created {zip_path} containing {len(csv_files)} CSV files! ✅", flush=True)
        print("\n" + "=" * 150, flush=True)
        print(">>> DECLARATION: BTC_P1_FULL_SUBMISSION_READY <<<", flush=True)
        print("=" * 150, flush=True)
    else:
        print("\n" + "=" * 150, flush=True)
        print(">>> DECLARATION: BTC_P1_SUBMISSION_NOT_READY <<<", flush=True)
        print("=" * 150, flush=True)


if __name__ == "__main__":
    run_unified_readiness()
