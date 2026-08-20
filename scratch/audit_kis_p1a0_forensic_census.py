#!/usr/bin/env python3
"""KIS P1A0: Forensic Failure Census & Deep Analysis.

Audits the 6 targeted BTC queries:
  - query-p1-12-kis (long query control - donuts)
  - query-p1-13-kis (camera cleaning)
  - query-p1-17-kis (charity gift at hospital)
  - query-p1-21-kis (beetle flight robotics at Lausanne)
  - query-p1-23-kis (Spielberg 1975 marine animal - shark)
  - query-p1-24-kis (cycling 3-rider top-down composition)

Strictly diagnostic:
  - Zero algorithm changes.
  - Generates contact sheet HTML for Ranks 4-30.
  - Extracts exact dropped text due to token compaction.
  - Formulates draft clause-aware compacted queries for p1-12 and p1-17.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Purge stale system_tai modules
for mod in list(sys.modules.keys()):
    if mod.startswith("system_tai"):
        del sys.modules[mod]

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

import cv2
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.translation.provider import MarianOfflineTranslator, TokenBudgetGuard

FORENSIC_QUERIES = [
    {
        "qid": "query-p1-12-kis",
        "name": "Trang trí bánh rán, rưới chocolate dâu tây (Long Query Control)",
        "file": "query-p1-12-kis.txt",
        "hypothesized_mechanism": "TOKEN_TRUNCATION_FAILURE (Long sequential narrative)",
    },
    {
        "qid": "query-p1-13-kis",
        "name": "Vệ sinh máy ảnh, tháo ống kính đặt trên khăn hồng tím",
        "file": "query-p1-13-kis.txt",
        "hypothesized_mechanism": "TRANSLATION_FAILURE (Lexical: 'vệ sinh máy ảnh' -> 'camera toilet')",
    },
    {
        "qid": "query-p1-17-kis",
        "name": "Trao quà từ thiện tại bệnh viện cho 4 em nhỏ nhận biển COVID-19",
        "file": "query-p1-17-kis.txt",
        "hypothesized_mechanism": "TRANSLATION_FAILURE + TOKEN_TRUNCATION_FAILURE (99 tokens)",
    },
    {
        "qid": "query-p1-21-kis",
        "name": "Nghiên cứu cơ chế bay của bọ để chế tạo robot ở ĐH Lausanne",
        "file": "query-p1-21-kis.txt",
        "hypothesized_mechanism": "TRANSLATION_ERROR (Lausanne->Larissa) + RETRIEVAL_RECALL_FAILURE",
    },
    {
        "qid": "query-p1-23-kis",
        "name": "Động vật biển nguy hiểm phim Steven Spielberg 1975 (Cá mập Jaws)",
        "file": "query-p1-23-kis.txt",
        "hypothesized_mechanism": "SEMANTIC_REASONING_FAILURE (Implicit cultural reference)",
    },
    {
        "qid": "query-p1-24-kis",
        "name": "Đua xe đạp quay từ trên cao, 3 tay đua thẳng hàng",
        "file": "query-p1-24-kis.txt",
        "hypothesized_mechanism": "FINE_GRAINED_TRANSLATION_FAILURE ('từ trên cao' -> 'top-up shot')",
    },
]


def resolve_video_path(video_id: str) -> Path | None:
    search_dirs = [
        Path("/kaggle/input/datasets/videos"),
        Path("/kaggle/input/datasets"),
        Path("/kaggle/input"),
        REPO_ROOT / "systems" / "system_tai" / "data",
    ]
    for root in search_dirs:
        if not root.exists():
            continue
        direct = root / f"{video_id}.mp4"
        if direct.exists():
            return direct
        batch_prefix = video_id.split("_")[0] if "_" in video_id else ""
        if batch_prefix:
            sub = root / batch_prefix / f"{video_id}.mp4"
            if sub.exists():
                return sub
            sub_batch = root / f"videos_{batch_prefix}" / f"{video_id}.mp4"
            if sub_batch.exists():
                return sub_batch
        # Glob fallback
        matches = list(root.rglob(f"{video_id}.mp4"))
        if matches:
            return matches[0]
    return None


def extract_thumbnail_base64(video_id: str, frame_id: int) -> str:
    vpath = resolve_video_path(video_id)
    if not vpath or not vpath.exists():
        return ""
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return ""
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return ""
    # Resize to thumbnail
    h, w = frame.shape[:2]
    new_w = 280
    new_h = int(h * (new_w / w))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buf).decode("utf-8")


def run_forensic_census() -> None:
    print("=" * 150, flush=True)
    print("KIS P1A0: FORENSIC FAILURE CENSUS & DEEP RECALL AUDIT (6 BTC CASES)", flush=True)
    print("=" * 150, flush=True)

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    cfg = SessionConfig.from_yaml(yaml_path)

    translator = MarianOfflineTranslator(
        revision=cfg.translation_revision,
        local_files_only=True,
    )
    guard = TokenBudgetGuard()

    # Initialize runtime if needed for candidates
    session_output = Path("/kaggle/working/output/kis_forensic_census") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_forensic_census"
    session_output.mkdir(parents=True, exist_ok=True)

    reuse_manifest_path: Path | None = None
    for p in [
        Path("/kaggle/working/manifest_cache.json"),
        Path("/kaggle/input/system-tai-manifest/feature_manifest.json"),
        Path("/kaggle/input/datasets/manifest_cache.json"),
        Path("/kaggle/input/manifest_cache.json"),
    ]:
        if p.exists() and p.stat().st_size > 1000:
            reuse_manifest_path = p
            break

    runtime_instance = None
    gate3_output_dir = Path("/kaggle/working/output/kis_release_gates")
    thunghiem_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"

    for item in FORENSIC_QUERIES:
        qid = item["qid"]
        name = item["name"]
        q_file = thunghiem_dir / item["file"]
        q_vi = q_file.read_text(encoding="utf-8").strip() if q_file.exists() else ""

        raw_marian_en = translator.translate(q_vi)
        raw_tokens = guard.count_tokens(raw_marian_en)
        effective_en, effective_tokens, was_compacted = guard.guard_and_compact(raw_marian_en)

        # Compute exact dropped content if compacted
        dropped_text = ""
        if was_compacted:
            if len(raw_marian_en) > len(effective_en):
                dropped_text = raw_marian_en[len(effective_en):].strip()

        # Try to locate Top100 candidates from jsonl
        candidates = []
        for cand_path in [
            gate3_output_dir / f"gate3-{qid}.top100.jsonl",
            gate3_output_dir / f"prod-smoke-{qid}.top100.jsonl",
            session_output / f"forensic-{qid}.top100.jsonl",
            Path(f"/kaggle/working/output/{qid}.top100.jsonl"),
        ]:
            if cand_path.exists():
                for line in cand_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        candidates.append(json.loads(line.strip()))
                if candidates:
                    break

        if not candidates:
            if runtime_instance is None:
                print("Bootstrapping OperationalKISRuntime for on-demand candidate generation...", flush=True)
                exec_cfg = SessionConfig.from_yaml(
                    yaml_path,
                    input_root=Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input"),
                    output_root=session_output,
                    reuse_manifest=reuse_manifest_path,
                )
                runtime_instance = OperationalKISRuntime.bootstrap(exec_cfg)

            req = QueryRequest(
                request_id=f"forensic-{qid}",
                query_id=qid,
                query_vi=q_vi,
                query_en=None,
                output_top_k=100,
                refine_top_n=3,
            )
            res = runtime_instance.handle_query(req)
            top100_rel = res["artifacts"].get("refined_top100_jsonl", res["artifacts"]["top100_jsonl"])
            top100_path = runtime_instance.output_root / top100_rel
            candidates = [
                json.loads(line)
                for line in top100_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        forensic_results.append({
            "qid": qid,
            "name": name,
            "query_vi": q_vi,
            "raw_marian_en": raw_marian_en,
            "raw_tokens": raw_tokens,
            "effective_en": effective_en,
            "effective_tokens": effective_tokens,
            "was_compacted": was_compacted,
            "dropped_text": dropped_text,
            "hypothesized_mechanism": item["hypothesized_mechanism"],
            "candidates": candidates,
            "num_candidates": len(candidates),
        })

        print(f"\n--- [{qid}] {name} ---", flush=True)
        print(f"• Raw VI Query                 : \"{q_vi}\"", flush=True)
        print(f"• Untouched Marian EN          : \"{raw_marian_en}\"", flush=True)
        print(f"• Raw CLIP Tokens              : {raw_tokens} tokens", flush=True)
        print(f"• Effective Retrieval Text     : \"{effective_en}\"", flush=True)
        print(f"• Effective CLIP Tokens        : {effective_tokens} tokens (Compacted? {'YES ⚠️' if was_compacted else 'NO ✅'})", flush=True)
        if was_compacted:
            print(f"• DROPPED TEXT / TAIL DETAILS : \"{dropped_text}\"", flush=True)
        cand_desc = [f"@{c['rank']}: {c['video_id']} (f={c['frame_id']})" for c in candidates[:10]]
        print(f"• Top 10 Candidates            : {cand_desc}", flush=True)

    # Output draft clause-aware compacted queries for p1-12 and p1-17
    print("\n" + "=" * 150, flush=True)
    print("DRAFT CLAUSE-AWARE COMPACTED QUERIES FOR OVER-BUDGET CASES (ILLUSTRATIVE ONLY - NOT INTEGRATED)", flush=True)
    print("=" * 150, flush=True)

    draft_p1_12 = "two donuts on a white plate, chef decorating them with chocolate drizzle, banana slices, and strawberry slices, on a wooden table"
    draft_p1_17 = "charity gift presentation ceremony at a hospital with children receiving sponsorship boards for COVID-19 orphans, holding medical gift bags against a red backdrop"

    tok_12 = guard.count_tokens(draft_p1_12)
    tok_17 = guard.count_tokens(draft_p1_17)

    print(f"• [p1-12 Draft Clause-Aware EN] : \"{draft_p1_12}\"", flush=True)
    print(f"  CLIP Tokens: {tok_12} tokens (Preserves: donuts, chef decorating, chocolate drizzle, banana & strawberry slices) vs P0 lost tail", flush=True)

    print(f"\n• [p1-17 Draft Clause-Aware EN] : \"{draft_p1_17}\"", flush=True)
    print(f"  CLIP Tokens: {tok_17} tokens (Preserves: charity gift at hospital, children receiving COVID-19 orphan sponsorship boards) vs P0 lost tail", flush=True)

    # Generate HTML Contact Sheet for Ranks 4-30
    out_html = Path("/kaggle/working/kis_p1a0_forensic_gallery.html")
    generate_forensic_gallery_html(forensic_results, out_html)
    print(f"\nSaved Ranks 4-30 forensic gallery to: {out_html}", flush=True)


def generate_forensic_gallery_html(results: list[dict[str, Any]], out_path: Path) -> None:
    html_cards = []
    for r in results:
        qid = r["qid"]
        name = r["name"]
        q_vi = r["query_vi"]
        raw_en = r["raw_marian_en"]
        eff_en = r["effective_en"]
        cands = r["candidates"]
        mech = r["hypothesized_mechanism"]

        # Ranks 4 to 30
        shortlist = cands[3:30] if len(cands) >= 4 else cands

        thumb_items = []
        for c in shortlist:
            rank = c["rank"]
            vid = c["video_id"]
            fid = c["frame_id"]
            img_b64 = extract_thumbnail_base64(vid, fid)
            img_tag = (
                f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:4px;" />'
                if img_b64
                else '<div style="background:#333;color:#888;height:90px;display:flex;align-items:center;justify-content:center;font-size:11px;">No Frame</div>'
            )
            thumb_items.append(f"""
            <div style="width:180px; margin:4px; padding:4px; background:#1e1e1e; border:1px solid #333; border-radius:6px; font-size:11px; text-align:center;">
                <div style="font-weight:bold; color:#ffc107; margin-bottom:2px;">Rank @{rank}</div>
                {img_tag}
                <div style="color:#eee; font-weight:600; margin-top:2px;">{vid}</div>
                <div style="color:#aaa; font-size:10px;">Frame {fid}</div>
            </div>
            """)

        thumbs_grid = "".join(thumb_items) if thumb_items else "<div style='color:#888;'>No candidate frames available</div>"

        html_cards.append(f"""
        <div style="background:#252526; border:1px solid #3c3c3c; border-radius:8px; margin-bottom:24px; padding:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #333; padding-bottom:8px; margin-bottom:12px;">
                <span style="background:#0d6efd; color:#fff; font-weight:bold; padding:4px 10px; border-radius:4px; font-size:14px;">{qid}</span>
                <span style="color:#ffc107; font-weight:bold; font-size:13px;">{name}</span>
                <span style="background:#495057; color:#f8f9fa; padding:3px 8px; border-radius:4px; font-size:11px;">{mech}</span>
            </div>
            <div style="margin-bottom:8px; font-size:12px; color:#ddd;">
                <b style="color:#61afef;">Raw VI:</b> {q_vi}
            </div>
            <div style="margin-bottom:8px; font-size:12px; color:#ddd;">
                <b style="color:#98c379;">Marian EN:</b> "{raw_en}" <span style="color:#888;">(Tokens: {r['raw_tokens']})</span>
            </div>
            {f'<div style="margin-bottom:8px; font-size:12px; color:#e5c07b;"><b>Compacted Effective Retrieval:</b> "{eff_en}" <span style="color:#888;">(Tokens: {r["effective_tokens"]})</span></div>' if r["was_compacted"] else ''}
            {f'<div style="margin-bottom:12px; font-size:12px; color:#e06c75; background:#332222; padding:6px; border-radius:4px;"><b>Dropped Tail Details:</b> "{r["dropped_text"]}"</div>' if r["was_compacted"] else ''}
            <div style="font-weight:bold; color:#9cdcfe; margin-bottom:8px; font-size:13px;">Ranks 4..30 Visual Shortlist:</div>
            <div style="display:flex; flex-wrap:wrap; max-height:480px; overflow-y:auto; background:#181818; padding:8px; border-radius:6px;">
                {thumbs_grid}
            </div>
        </div>
        """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>KIS P1A0 Forensic Failure Census</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background:#121212; color:#fff; padding:20px; }}
        </style>
    </head>
    <body>
        <h2 style="color:#61afef; border-bottom:2px solid #333; padding-bottom:10px;">🔬 KIS P1A0: FORENSIC FAILURE CENSUS & RANKS 4..30 AUDIT</h2>
        <div style="color:#aaa; margin-bottom:20px; font-size:13px;">Detailed visual inspection of 6 failure/at-risk queries (p1-12, p1-13, p1-17, p1-21, p1-23, p1-24) across Ranks 4 to 30.</div>
        {''.join(html_cards)}
    </body>
    </html>
    """

    out_path.write_text(full_html, encoding="utf-8")


if __name__ == "__main__":
    run_forensic_census()
