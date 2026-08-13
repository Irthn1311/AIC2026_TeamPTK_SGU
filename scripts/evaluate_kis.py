"""
Evaluate KIS retrieval results against JsonTest/gt_kis*.json.

The evaluator uses the existing backend RetrievalService instead of rebuilding
retrieval. Ground truth is read-only: video_id + gt.semantic_frame are the only
target fields used for scoring.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd


import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce offline mode and store all caches strictly on Drive E (no writes to C:)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")
os.environ["TRANSFORMERS_CACHE"] = str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")
os.environ["PIP_CACHE_DIR"] = str(PROJECT_ROOT / ".cache" / "pip")

from backend.retrieval_service import RetrievalService
from backend.schemas import FusionWeights


TOLERANCES_SEC = (1.0, 3.0, 5.0)
HIT_KS = (1, 5, 10, 20, 30, 50)
DEFAULT_GT_CANDIDATES = (
    PROJECT_ROOT / "JsonTest" / "gt_kis(2).json",
    PROJECT_ROOT / "JsonTest" / "gt_kis.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KIS retrieval with temporal tolerance and an HTML debug report."
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground-truth JSON path. Defaults to JsonTest/gt_kis(2).json if present, otherwise JsonTest/gt_kis.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "kis",
        help="Directory for metrics.json, per_query_results.csv, failures.csv, and report.html.",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Number of predictions to evaluate/render.")
    parser.add_argument(
        "--fusion-mode",
        choices=("static", "dynamic", "manual"),
        default="static",
        help="RetrievalService fusion mode.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Manual weights as visual,ocr,asr,object. Only used with --fusion-mode manual.",
    )
    parser.add_argument(
        "--dedup-window-seconds",
        type=float,
        default=4.0,
        help="Temporal dedup window passed to RetrievalService.search.",
    )
    parser.add_argument(
        "--no-temporal-dedup",
        action="store_true",
        help="Disable RetrievalService temporal deduplication.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional query limit for smoke tests.")
    parser.add_argument(
        "--skip-thumbnails",
        action="store_true",
        help="Skip copying/extracting thumbnails. Metrics are still produced.",
    )
    return parser.parse_args()


def resolve_gt_path(path_arg: Path | None) -> Path:
    if path_arg is not None:
        path = path_arg if path_arg.is_absolute() else PROJECT_ROOT / path_arg
        if not path.exists():
            raise FileNotFoundError(f"Ground-truth JSON not found: {path}")
        return path
    for candidate in DEFAULT_GT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find JsonTest/gt_kis(2).json or JsonTest/gt_kis.json")


def load_gt(gt_path: Path, limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gt_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"GT JSON must contain a list field named 'queries': {gt_path}")
    if limit is not None:
        queries = queries[: max(0, limit)]
    return payload, queries


def parse_weights(value: str | None) -> FusionWeights | None:
    if value is None:
        return None
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--weights must contain 4 comma-separated values: visual,ocr,asr,object")
    return FusionWeights(visual=parts[0], ocr=parts[1], asr=parts[2], object=parts[3])


def find_video_path(video_id: str) -> Path | None:
    for video_root in sorted((PROJECT_ROOT / "datasets_L21").glob("Videos_*")):
        for ext in (".mp4", ".mkv", ".avi", ".mov"):
            path = video_root / "video" / f"{video_id}{ext}"
            if path.exists():
                return path
    return None


class VideoMetadata:
    def __init__(self) -> None:
        self._fps: dict[str, float] = {}
        self._paths: dict[str, Path | None] = {}

    def video_path(self, video_id: str) -> Path | None:
        if video_id not in self._paths:
            self._paths[video_id] = find_video_path(video_id)
        return self._paths[video_id]

    def fps(self, video_id: str) -> float:
        if video_id not in self._fps:
            self._fps[video_id] = self._read_fps(video_id)
        return self._fps[video_id]

    def _read_fps(self, video_id: str) -> float:
        path = self.video_path(video_id)
        if path is not None:
            try:
                import cv2

                cap = cv2.VideoCapture(str(path))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                if fps > 0 and math.isfinite(fps):
                    return fps
            except Exception:
                pass

        map_path = (
            PROJECT_ROOT
            / "datasets_L21"
            / "map-keyframes-aic25-b1"
            / "map-keyframes"
            / f"{video_id}.csv"
        )
        if map_path.exists():
            try:
                df = pd.read_csv(map_path, usecols=["fps"])
                fps = float(df["fps"].dropna().iloc[0])
                if fps > 0 and math.isfinite(fps):
                    return fps
            except Exception:
                pass
        raise ValueError(f"Could not determine FPS for {video_id}. Video path: {path}")


def format_time(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return ""
    s = max(0.0, float(seconds))
    hh = int(s // 3600)
    mm = int((s % 3600) // 60)
    ss = s % 60
    if hh:
        return f"{hh:02d}:{mm:02d}:{ss:05.2f}"
    return f"{mm:02d}:{ss:05.2f}"


def safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def extract_gt_thumbnail(video_path: Path | None, frame_id: int, out_path: Path) -> str:
    if video_path is None:
        return ""
    try:
        import cv2

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return ""
        h, w = frame.shape[:2]
        max_w = 420
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if cv2.imwrite(str(out_path), frame):
            return out_path.as_posix()
    except Exception:
        return ""
    return ""


def copy_prediction_thumbnail(src_path: Path | None, out_path: Path) -> str:
    if src_path is None or not src_path.exists():
        return ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, out_path)
    return out_path.as_posix()


def rel_asset(path_str: str, report_dir: Path) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    try:
        return path.relative_to(report_dir).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_result(
    raw: dict[str, Any],
    gt_video_id: str,
    gt_time_sec: float,
    fps_lookup: VideoMetadata,
    service: RetrievalService,
    query_id: str,
    output_dir: Path,
    skip_thumbnails: bool,
) -> dict[str, Any]:
    video_id = str(raw.get("video_id", ""))
    fps = fps_lookup.fps(video_id)
    pred_time = float(raw.get("timestamp_seconds", raw.get("timestamp_sec", raw.get("frame_idx", 0) / fps)))
    frame_id = int(raw.get("actual_frame_id", raw.get("frame_idx", 0)))
    temporal_error = abs(pred_time - gt_time_sec) if video_id == gt_video_id else None

    kf_name = str(raw.get("keyframe_name", ""))
    pred_thumb = ""
    if not skip_thumbnails:
        src = service.get_keyframe_image_path(video_id, kf_name) if kf_name else None
        thumb_name = f"{safe_name(query_id)}_r{int(raw.get('rank', 0)):02d}_{safe_name(video_id)}_{safe_name(kf_name)}.jpg"
        pred_thumb = copy_prediction_thumbnail(src, output_dir / "thumbnails" / "pred" / thumb_name)

    scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    return {
        "rank": int(raw.get("rank", 0)),
        "score": float(raw.get("score", 0.0)),
        "video_id": video_id,
        "frame_id": frame_id,
        "fps": fps,
        "timestamp_sec": pred_time,
        "timestamp_text": format_time(pred_time),
        "api_timestamp_sec": float(raw.get("timestamp_seconds", pred_time)),
        "keyframe_name": kf_name,
        "thumbnail": pred_thumb,
        "temporal_error_sec": temporal_error,
        "hit_1s": temporal_error is not None and temporal_error <= 1.0,
        "hit_3s": temporal_error is not None and temporal_error <= 3.0,
        "hit_5s": temporal_error is not None and temporal_error <= 5.0,
        "scores": {
            "visual": float(scores.get("visual", 0.0)),
            "ocr": float(scores.get("ocr", 0.0)),
            "asr": float(scores.get("asr", 0.0)),
            "object": float(scores.get("object", 0.0)),
        },
        "ocr_text": str(raw.get("ocr_text", "")),
        "asr_text": str(raw.get("asr_text", "")),
    }


def first_hit_rank(predictions: list[dict[str, Any]], tolerance: float, k: int) -> int | None:
    for pred in predictions[:k]:
        err = pred.get("temporal_error_sec")
        if err is not None and err <= tolerance:
            return int(pred["rank"])
    return None


def compute_metric_block(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        res: dict[str, Any] = {
            "num_queries": 0,
            "mrr": 0.0,
            "mean_temporal_error_sec": None,
            "median_temporal_error_sec": None,
            "temporal_error_support": 0,
        }
        for k in HIT_KS:
            res[f"hit_at_{k}"] = 0.0
        return res

    out: dict[str, Any] = {"num_queries": n}
    reciprocal_ranks = []
    for k in HIT_KS:
        hits = 0
        for row in rows:
            hit_rank = row.get(f"first_hit_rank_{int(tolerance)}s")
            if hit_rank is not None and int(hit_rank) <= k:
                hits += 1
        out[f"hit_at_{k}"] = round(hits / n, 6)

    for row in rows:
        rank = row.get(f"first_hit_rank_{int(tolerance)}s")
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / float(rank))
    out["mrr"] = round(sum(reciprocal_ranks) / n, 6)

    errors = [
        float(row["first_same_video_temporal_error_sec"])
        for row in rows
        if row.get("first_same_video_temporal_error_sec") is not None
    ]
    out["mean_temporal_error_sec"] = round(mean(errors), 6) if errors else None
    out["median_temporal_error_sec"] = round(median(errors), 6) if errors else None
    out["temporal_error_support"] = len(errors)
    return out


def group_metric_blocks(rows: list[dict[str, Any]], key: str, tolerance: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown") or "unknown")].append(row)
    return {name: compute_metric_block(items, tolerance) for name, items in sorted(grouped.items())}


def compute_metrics(rows: list[dict[str, Any]], gt_payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    by_tolerance = {}
    for tol in TOLERANCES_SEC:
        tol_key = f"tolerance_{int(tol)}s"
        by_tolerance[tol_key] = {
            "overall": compute_metric_block(rows, tol),
            "by_difficulty": group_metric_blocks(rows, "difficulty", tol),
            "by_modality": group_metric_blocks(rows, "modality", tol),
            "by_video": group_metric_blocks(rows, "gt_video_id", tol),
        }

    return {
        "dataset": gt_payload.get("dataset", ""),
        "gt_version": gt_payload.get("version", ""),
        "task": gt_payload.get("task", "kis"),
        "frame_reference": gt_payload.get("frame_reference", ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "gt_path": str(args.gt_path),
            "output_dir": str(args.output_dir),
            "top_k": args.top_k,
            "fusion_mode": args.fusion_mode,
            "temporal_dedup": not args.no_temporal_dedup,
            "dedup_window_seconds": args.dedup_window_seconds,
            "rule": "pred.video_id == gt.video_id and abs(pred.frame_id/fps - gt.semantic_frame/fps) <= tolerance",
            "primary_tolerance_sec": 3.0,
            "temporal_error_definition": "Temporal error is measured on the first same-video prediction within Top-K; queries with no same-video prediction are excluded from mean/median support.",
        },
        "by_tolerance": by_tolerance,
        "primary": by_tolerance["tolerance_3s"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bar_chart(title: str, values: dict[str, dict[str, Any]], metric: str = "hit_at_20") -> str:
    rows = []
    for name, block in values.items():
        value = float(block.get(metric, 0.0) or 0.0)
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{html.escape(name)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{value * 100:.2f}%"></div></div>
              <div class="bar-value">{value:.1%}</div>
            </div>
            """
        )
    return f"""
    <section class="panel">
      <h2>{html.escape(title)}</h2>
      {''.join(rows)}
    </section>
    """


def temporal_histogram(rows: list[dict[str, Any]]) -> str:
    bins = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 30), (30, math.inf)]
    labels = ["0-1s", "1-3s", "3-5s", "5-10s", "10-30s", ">30s"]
    counts = [0 for _ in bins]
    for row in rows:
        value = row.get("first_same_video_temporal_error_sec")
        if value is None:
            continue
        for i, (lo, hi) in enumerate(bins):
            if lo <= float(value) < hi:
                counts[i] += 1
                break
    max_count = max(counts) if counts else 1
    parts = []
    for label, count in zip(labels, counts):
        pct = 0 if max_count == 0 else count / max_count * 100
        parts.append(
            f"""
            <div class="hist-row">
              <div class="bar-label">{label}</div>
              <div class="bar-track"><div class="bar-fill hist" style="width:{pct:.2f}%"></div></div>
              <div class="bar-value">{count}</div>
            </div>
            """
        )
    return f"""
    <section class="panel">
      <h2>Temporal Error Distribution</h2>
      {''.join(parts)}
    </section>
    """


def render_img(src: str, alt: str) -> str:
    if not src:
        return '<div class="missing-img">missing thumbnail</div>'
    return f'<img src="{html.escape(src)}" alt="{html.escape(alt)}" loading="lazy">'


def generate_html_report(
    output_path: Path,
    rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    primary = metrics.get("primary", {})
    overall = primary.get("overall", {})

    summary_headers = "".join(f"<th>Hit@{k}</th>" for k in HIT_KS)
    summary_cells = "".join(
        f"<td><strong>{overall.get(f'hit_at_{k}', 0.0):.1%}</strong></td>" for k in HIT_KS
    )
    summary_table = f"""
      <table class="summary-table">
        <thead><tr>{summary_headers}<th>MRR</th><th>Mean Error</th><th>Median Error</th></tr></thead>
        <tbody>
          <tr>{summary_cells}<td><strong>{overall.get('mrr', 0.0):.4f}</strong></td><td>{overall.get('mean_temporal_error_sec', '')}s</td><td>{overall.get('median_temporal_error_sec', '')}s</td></tr>
        </tbody>
      </table>
    """

    # Build interactive table rows
    table_rows = []
    for idx, (row, detail) in enumerate(zip(rows, details)):
        q_id = row.get("query_id", "")
        q_text = row.get("query", "")
        diff = row.get("difficulty", "unknown")
        mod = row.get("modality", "unknown")
        gt = detail.get("gt", {})
        preds = detail.get("predictions", [])
        top1 = preds[0] if preds else {}
        hit_rank = row.get("first_hit_rank_3s")

        gt_thumb = detail.get("asset_paths", {}).get(gt.get("thumbnail", ""), "")
        top1_thumb = detail.get("asset_paths", {}).get(top1.get("thumbnail", ""), "")

        # Hit status logic
        has_hit = hit_rank is not None and str(hit_rank).strip() != ""
        if has_hit:
            hit_rank_int = int(hit_rank)
            if hit_rank_int == 1:
                t_err = top1.get("temporal_error_sec")
                t_err_str = f"{float(t_err):.2f}s" if t_err is not None and str(t_err).strip() != "" else ""
                status_badge = f'<span class="badge badge-hit">✅ Hit #1 ({t_err_str})</span>'
                status_filter = "hit hit1"
            else:
                status_badge = f'<span class="badge badge-hit-k">✅ Hit #{hit_rank_int}</span>'
                status_filter = "hit hitk"
        else:
            if top1.get("video_id") == gt.get("video_id"):
                err = top1.get("temporal_error_sec")
                err_str = f"+{float(err):.1f}s" if err is not None and str(err).strip() != "" else ""
                status_badge = f'<span class="badge badge-diff-time">⚠️ Cùng Video ({err_str})</span>'
            else:
                status_badge = f'<span class="badge badge-miss">❌ Khác Video ({top1.get("video_id")})</span>'
            status_filter = "failed miss"

        scores = top1.get("scores", {})
        score_preview = f"V {scores.get('visual', 0):.2f} · O {scores.get('ocr', 0):.2f} · A {scores.get('asr', 0):.2f} · Obj {scores.get('object', 0):.2f}"

        table_rows.append(
            f"""
            <tr class="query-row {status_filter} mod-{mod} diff-{diff}" data-query-index="{idx}" onclick="openModal({idx})">
              <td class="col-id">
                <div class="qid">#{idx+1} {html.escape(q_id)}</div>
                <div class="row-tags">
                  <span class="tag-pill tag-{diff}">{html.escape(diff)}</span>
                  <span class="tag-pill tag-{mod}">{html.escape(mod)}</span>
                </div>
              </td>
              <td class="col-query">
                <div class="q-text">{html.escape(q_text)}</div>
                <div class="q-sub-score">{score_preview}</div>
              </td>
              <td class="col-thumb">
                <div class="mini-card gt-mini">
                  <div class="mini-img-wrap">{render_img(gt_thumb, 'GT')}</div>
                  <div class="mini-label">🎯 {html.escape(gt.get('video_id', ''))} : f{gt.get('frame_id', '')} ({html.escape(gt.get('timestamp_text', ''))})</div>
                </div>
              </td>
              <td class="col-thumb">
                <div class="mini-card pred-mini">
                  <div class="mini-img-wrap">{render_img(top1_thumb, 'Top1')}</div>
                  <div class="mini-label">🏆 {html.escape(top1.get('video_id', ''))} : f{top1.get('frame_id', '')} ({html.escape(top1.get('timestamp_text', ''))})</div>
                </div>
              </td>
              <td class="col-status">
                <div>{status_badge}</div>
                <div class="score-val">Score: <strong>{top1.get('score', 0):.4f}</strong></div>
              </td>
              <td class="col-action">
                <button class="btn-inspect" onclick="event.stopPropagation(); openModal({idx});">
                  🔍 So sánh Visual
                </button>
              </td>
            </tr>
            """
        )

    # Detailed query cards
    query_cards = []
    for idx, item in enumerate(details):
        gt = item["gt"]
        gt_thumb = item["asset_paths"].get(gt["thumbnail"], "")
        pred_cards = []
        for pred in item["predictions"]:
            err = pred.get("temporal_error_sec")
            err_text = "" if err is None else f"{err:.2f}s"
            is_hit = bool(pred.get("hit_3s"))
            status_icon = "✅" if is_hit else "❌"
            status_class = "hit" if is_hit else "miss"
            p_thumb = item["asset_paths"].get(pred.get("thumbnail", ""), "")
            p_rank = pred.get("rank", 1)
            pred_cards.append(
                f"""
                <div class="pred-card {status_class}" onclick="openModalWithRank({idx}, {p_rank})">
                  <div class="thumb">{render_img(p_thumb, 'prediction')}</div>
                  <div class="pred-meta">
                    <div><strong>#{p_rank}</strong> {status_icon} score <strong>{pred['score']:.4f}</strong></div>
                    <div>{html.escape(pred['video_id'])} / f{pred['frame_id']} ({html.escape(pred['timestamp_text'])})</div>
                    <div>Δt: <strong>{html.escape(err_text)}</strong></div>
                    <div class="score-line">V {pred['scores']['visual']:.2f} · OCR {pred['scores']['ocr']:.2f} · ASR {pred['scores']['asr']:.2f} · Obj {pred['scores']['object']:.2f}</div>
                  </div>
                </div>
                """
            )
        query_cards.append(
            f"""
            <section class="query-card" id="card-{idx}">
              <div class="query-head">
                <div>
                  <h3>#{idx+1} {html.escape(item['query_id'])}</h3>
                  <p class="q-main-text">{html.escape(item['query'])}</p>
                </div>
                <div class="head-right">
                  <div class="tags">
                    <span class="tag-pill tag-{item['difficulty']}">{html.escape(item['difficulty'])}</span>
                    <span class="tag-pill tag-{item['modality']}">{html.escape(item['modality'])}</span>
                  </div>
                  <button class="btn-inspect btn-inspect-sm" onclick="openModal({idx})">🔍 So sánh Visual</button>
                </div>
              </div>
              <div class="compare-row">
                <div class="gt-card" onclick="openModal({idx})">
                  <div class="thumb large">{render_img(gt_thumb, 'ground truth')}</div>
                  <div class="gt-meta">
                    <strong>🎯 Ground Truth</strong>
                    <div>{html.escape(gt['video_id'])} / frame {gt['frame_id']}</div>
                    <div>{html.escape(gt['timestamp_text'])} | FPS {gt['fps']:.1f}</div>
                  </div>
                </div>
                <div class="pred-grid">{''.join(pred_cards)}</div>
              </div>
            </section>
            """
        )

    # Encode JSON details safely for client-side inspector modal
    json_details = json.dumps(details, ensure_ascii=False)

    html_doc = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KIS Multimodal Evaluation & Visual Inspector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f19;
      --panel: #111827;
      --panel-hover: #1e293b;
      --panel-alt: #162032;
      --text: #f3f4f6;
      --text-dim: #9ca3af;
      --muted: #64748b;
      --border: #2d3748;
      --border-focus: #3b82f6;
      --blue: #3b82f6;
      --blue-glow: rgba(59, 130, 246, 0.25);
      --green: #10b981;
      --green-bg: rgba(16, 185, 129, 0.15);
      --red: #ef4444;
      --red-bg: rgba(239, 68, 68, 0.15);
      --amber: #f59e0b;
      --amber-bg: rgba(245, 158, 11, 0.15);
      --purple: #a855f7;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 14px;
      line-height: 1.5;
    }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    
    /* Headers */
    .header-bar {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 4px;
    }}
    .subtle {{ color: var(--muted); font-size: 13px; }}
    h2 {{ font-size: 16px; font-weight: 700; color: #e2e8f0; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
    h3 {{ font-size: 15px; font-weight: 700; color: #f1f5f9; }}
    
    /* Panels */
    .panel, .query-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 20px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }}
    
    /* Summary Metrics Table */
    .summary-table {{ width: 100%; border-collapse: collapse; }}
    .summary-table th, .summary-table td {{
      padding: 12px 14px;
      text-align: center;
      border: 1px solid var(--border);
    }}
    .summary-table th {{
      background: var(--panel-alt);
      color: #94a3b8;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .summary-table td {{
      font-size: 18px;
      color: #60a5fa;
      font-family: 'JetBrains Mono', monospace;
    }}
    
    /* Charts Grid */
    .summary-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }}
    .bar-row, .hist-row {{
      display: grid;
      grid-template-columns: 90px 1fr 60px;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
    }}
    .bar-label {{ color: var(--text-dim); font-size: 12px; font-weight: 500; }}
    .bar-track {{ height: 10px; border-radius: 5px; background: #1e293b; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 5px; }}
    .bar-fill.hist {{ background: linear-gradient(90deg, #f59e0b, #fbbf24); }}
    .bar-value {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; text-align: right; color: #f3f4f6; }}
    
    /* Interactive Filter & Controls Bar */
    .controls-panel {{
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 18px;
      margin-bottom: 16px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    .filter-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .tab-btn {{
      background: #1e293b;
      border: 1px solid var(--border);
      color: #cbd5e1;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .tab-btn:hover {{ background: #334155; color: #fff; border-color: #64748b; }}
    .tab-btn.active {{
      background: var(--blue);
      color: #ffffff;
      border-color: var(--blue);
      box-shadow: 0 0 12px var(--blue-glow);
    }}
    .search-input-wrap {{ position: relative; min-width: 280px; }}
    .search-input {{
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 12px 7px 32px;
      color: #f8fafc;
      font-size: 13px;
      outline: none;
      transition: border-color 0.2s;
    }}
    .search-input:focus {{ border-color: var(--blue); }}
    .search-icon {{
      position: absolute;
      left: 10px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--muted);
      font-size: 12px;
    }}
    
    /* Interactive Explorer Table */
    .table-container {{
      overflow-x: auto;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--panel);
    }}
    .eval-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .eval-table th {{
      background: #162032;
      color: #94a3b8;
      padding: 10px 14px;
      text-align: left;
      font-size: 12px;
      font-weight: 700;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}
    .eval-table td {{
      padding: 10px 14px;
      border-bottom: 1px solid #1e293b;
      vertical-align: middle;
    }}
    .query-row {{
      cursor: pointer;
      transition: background 0.15s ease, border-left 0.15s ease;
    }}
    .query-row:hover {{
      background: #1e293b;
    }}
    .query-row.selected {{
      background: #233554;
    }}
    
    /* Table Columns */
    .col-id {{ width: 150px; }}
    .qid {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 12px; color: #93c5fd; }}
    .row-tags {{ display: flex; gap: 4px; margin-top: 4px; }}
    .tag-pill {{
      font-size: 10px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      text-transform: uppercase;
    }}
    .tag-easy {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .tag-medium {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .tag-hard {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
    .tag-visual {{ background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }}
    .tag-ocr {{ background: rgba(168, 85, 247, 0.2); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.4); }}
    .tag-object {{ background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.4); }}
    .tag-mixed {{ background: rgba(20, 184, 166, 0.2); color: #5eead4; border: 1px solid rgba(20, 184, 166, 0.4); }}
    
    .col-query {{ min-width: 260px; max-width: 380px; }}
    .q-text {{ font-weight: 500; color: #f1f5f9; line-height: 1.4; }}
    .q-sub-score {{ font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace; margin-top: 3px; }}
    
    /* Table Mini Thumbnails */
    .col-thumb {{ width: 180px; }}
    .mini-card {{
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      width: 160px;
    }}
    .mini-img-wrap {{
      width: 100%;
      height: 90px;
      background: #020617;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .mini-img-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}
    .mini-label {{
      padding: 3px 6px;
      font-size: 11px;
      color: #94a3b8;
      font-family: 'JetBrains Mono', monospace;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      background: #1e293b;
    }}
    
    /* Badges */
    .col-status {{ width: 170px; }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .badge-hit {{ background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }}
    .badge-hit-k {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }}
    .badge-miss {{ background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }}
    .badge-diff-time {{ background: var(--amber-bg); color: var(--amber); border: 1px solid var(--amber); }}
    .score-val {{ font-size: 11px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }}
    
    .col-action {{ width: 130px; text-align: center; }}
    .btn-inspect {{
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: white;
      border: none;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }}
    .btn-inspect:hover {{
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.5);
    }}
    .btn-inspect-sm {{ padding: 4px 10px; font-size: 11px; }}
    
    /* Query Cards Section */
    .cards-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 32px;
      margin-bottom: 16px;
    }}
    .query-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }}
    .head-right {{ display: flex; align-items: center; gap: 12px; }}
    .tags {{ display: flex; gap: 6px; }}
    .q-main-text {{ color: #cbd5e1; font-size: 14px; margin-top: 4px; font-weight: 500; }}
    .compare-row {{ display: grid; grid-template-columns: 280px 1fr; gap: 14px; }}
    .gt-card {{
      border: 2px solid #3b82f6;
      border-radius: 8px;
      background: #131d2e;
      overflow: hidden;
      cursor: pointer;
      transition: transform 0.15s ease;
    }}
    .gt-card:hover {{ transform: scale(1.01); }}
    .pred-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 10px;
    }}
    .pred-card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #162032;
      overflow: hidden;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .pred-card:hover {{
      transform: translateY(-2px);
      border-color: #60a5fa;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }}
    .pred-card.hit {{ border-color: rgba(16, 185, 129, 0.8); background: #0f241d; }}
    .pred-card.miss {{ border-color: rgba(239, 68, 68, 0.3); }}
    .thumb {{ width: 100%; aspect-ratio: 16 / 9; background: #020617; display: flex; align-items: center; justify-content: center; }}
    .thumb.large {{ aspect-ratio: 16 / 10; }}
    img {{ width: 100%; height: 100%; object-fit: contain; display: block; background: #000; }}
    .missing-img {{ padding: 20px; color: var(--muted); font-size: 12px; }}
    .gt-meta, .pred-meta {{ padding: 8px 10px; font-size: 11px; line-height: 1.4; }}
    .score-line {{ color: var(--muted); margin-top: 3px; font-family: 'JetBrains Mono', monospace; font-size: 10px; }}
    
    /* ==========================================
       SIDE-BY-SIDE VISUAL INSPECTOR MODAL
       ========================================== */
    .modal-overlay {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(4, 7, 13, 0.88);
      backdrop-filter: blur(8px);
      z-index: 9999;
      justify-content: center;
      align-items: center;
      padding: 20px;
      opacity: 0;
      transition: opacity 0.2s ease;
    }}
    .modal-overlay.active {{
      display: flex;
      opacity: 1;
    }}
    .modal-container {{
      background: #111827;
      border: 1px solid #374151;
      border-radius: 16px;
      width: 100%;
      max-width: 1380px;
      max-height: 94vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
      overflow: hidden;
      animation: modalPop 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    @keyframes modalPop {{
      from {{ transform: scale(0.95) translateY(10px); opacity: 0; }}
      to {{ transform: scale(1) translateY(0); opacity: 1; }}
    }}
    
    /* Modal Header */
    .modal-header {{
      background: #162032;
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}
    .modal-title-wrap {{ flex: 1; min-width: 0; }}
    .modal-title-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
    .modal-qid {{ font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 800; color: #60a5fa; }}
    .modal-query-text {{ font-size: 15px; font-weight: 600; color: #f8fafc; overflow-wrap: break-word; }}
    
    .modal-nav-controls {{ display: flex; align-items: center; gap: 8px; }}
    .modal-btn-nav {{
      background: #1e293b;
      border: 1px solid var(--border);
      color: #e2e8f0;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s;
    }}
    .modal-btn-nav:hover {{ background: #334155; color: #fff; }}
    .modal-counter {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--muted); padding: 0 4px; }}
    .modal-btn-close {{
      background: #374151;
      border: none;
      color: #e2e8f0;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s;
    }}
    .modal-btn-close:hover {{ background: #ef4444; color: #fff; }}
    
    /* Modal Main Body: Side-by-Side Comparison */
    .modal-body {{
      padding: 20px;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .comparison-stage {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      align-items: stretch;
    }}
    
    /* Inspector Cards (Left GT vs Right Pred) */
    .inspector-card {{
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }}
    .inspector-card.gt-stage {{ border-color: #3b82f6; }}
    .inspector-card.pred-stage.hit {{ border-color: #10b981; }}
    .inspector-card.pred-stage.miss {{ border-color: #ef4444; }}
    
    .card-head-bar {{
      padding: 10px 14px;
      background: #1e293b;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      font-size: 13px;
      border-bottom: 1px solid var(--border);
    }}
    .gt-head {{ color: #60a5fa; }}
    .pred-head {{ display: flex; align-items: center; gap: 8px; }}
    
    .stage-image-wrap {{
      width: 100%;
      height: 310px;
      background: #000;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .stage-image-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    .img-zoom-btn {{
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(0, 0, 0, 0.6);
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 11px;
      cursor: pointer;
      opacity: 0.8;
      transition: opacity 0.15s;
    }}
    .img-zoom-btn:hover {{ opacity: 1; background: rgba(0, 0, 0, 0.9); }}
    
    .stage-info {{
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      flex: 1;
      font-size: 13px;
    }}
    
    .info-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      background: #162032;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
    }}
    .info-item span {{ color: var(--muted); }}
    .info-item strong {{ color: #f1f5f9; }}
    
    /* Multimodal Score Breakdown Bars */
    .score-breakdown-box {{
      background: #162032;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .score-title {{
      font-size: 12px;
      font-weight: 700;
      color: #cbd5e1;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }}
    .branch-bar-row {{
      display: grid;
      grid-template-columns: 80px 1fr 45px;
      gap: 8px;
      align-items: center;
      margin: 4px 0;
      font-size: 11px;
    }}
    .branch-name {{ color: #94a3b8; font-weight: 600; }}
    .branch-track {{ height: 8px; background: #0f172a; border-radius: 4px; overflow: hidden; }}
    .branch-fill {{ height: 100%; border-radius: 4px; }}
    .fill-visual {{ background: #3b82f6; }}
    .fill-ocr {{ background: #a855f7; }}
    .fill-asr {{ background: #10b981; }}
    .fill-obj {{ background: #ec4899; }}
    .branch-val {{ font-family: 'JetBrains Mono', monospace; text-align: right; color: #e2e8f0; font-size: 11px; }}
    
    /* Text / ASR Insights Box */
    .text-insights-box {{
      background: #0b1220;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      color: #94a3b8;
    }}
    .insight-label {{ font-size: 11px; font-weight: 700; color: #cbd5e1; margin-bottom: 2px; }}
    .insight-content {{ color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 11px; max-height: 48px; overflow-y: auto; }}
    
    /* Filmstrip Carousel of Candidates */
    .filmstrip-panel {{
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 16px;
    }}
    .filmstrip-title {{
      font-size: 12px;
      font-weight: 700;
      color: #94a3b8;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .filmstrip-row {{
      display: flex;
      gap: 10px;
      overflow-x: auto;
      padding-bottom: 6px;
      scrollbar-width: thin;
      scrollbar-color: #3b82f6 #1e293b;
    }}
    .filmstrip-item {{
      flex: 0 0 130px;
      background: #1e293b;
      border: 2px solid transparent;
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .filmstrip-item:hover {{
      transform: translateY(-2px);
      border-color: #60a5fa;
    }}
    .filmstrip-item.active {{
      border-color: #3b82f6;
      box-shadow: 0 0 10px var(--blue-glow);
    }}
    .filmstrip-item.hit {{ border-color: rgba(16, 185, 129, 0.7); }}
    .filmstrip-thumb {{ width: 100%; height: 75px; background: #000; }}
    .filmstrip-thumb img {{ width: 100%; height: 100%; object-fit: cover; }}
    .filmstrip-meta {{
      padding: 4px 6px;
      font-size: 10px;
      font-family: 'JetBrains Mono', monospace;
      background: #111827;
      display: flex;
      justify-content: space-between;
      color: #94a3b8;
    }}
    
    @media (max-width: 1024px) {{
      .summary-grid, .comparison-stage, .compare-row {{ grid-template-columns: 1fr; }}
      .col-thumb {{ width: 120px; }}
      .mini-card {{ width: 120px; }}
      .stage-image-wrap {{ height: 240px; }}
    }}
  </style>
</head>
<body>
<main>
  <div class="header-bar">
    <div>
      <h1>KIS Multimodal Evaluation & Visual Inspector</h1>
      <div class="subtle">Benchmark độ chính xác Known-Item Search trên L21 Dataset | Dung sai chuẩn: ±3.0s | Nhấp vào bất kỳ dòng nào để xem so sánh trực quan Ground Truth vs Predictions</div>
    </div>
  </div>

  <section class="panel">
    <h2>📊 Tổng Quan Hiệu Năng (Overall Summary)</h2>
    {summary_table}
  </section>

  <div class="summary-grid">
    {bar_chart("🎯 Độ Chính Xác Theo Modality (Hit@20, ±3s)", primary.get("by_modality", {}))}
    {bar_chart("⚡ Độ Chính Xác Theo Độ Khó (Hit@20, ±3s)", primary.get("by_difficulty", {}))}
    {temporal_histogram(rows)}
  </div>

  <!-- Interactive Filter & Search Bar -->
  <div class="controls-panel">
    <div class="filter-tabs">
      <button class="tab-btn active" onclick="filterTable('failed', this)">❌ Chỉ Xem Thất Bại ({len([r for r in rows if r.get('first_hit_rank_3s') is None])})</button>
      <button class="tab-btn" onclick="filterTable('hit', this)">✅ Thành Công ({len([r for r in rows if r.get('first_hit_rank_3s') is not None])})</button>
      <button class="tab-btn" onclick="filterTable('all', this)">📋 Tất Cả ({len(rows)})</button>
      <button class="tab-btn" onclick="filterTable('mod-visual', this)">🎨 Visual</button>
      <button class="tab-btn" onclick="filterTable('mod-ocr', this)">🔤 OCR</button>
      <button class="tab-btn" onclick="filterTable('mod-object', this)">📦 Object</button>
      <button class="tab-btn" onclick="filterTable('mod-mixed', this)">🔀 Mixed</button>
    </div>
    <div class="search-input-wrap">
      <span class="search-icon">🔍</span>
      <input type="text" class="search-input" id="search-box" placeholder="Tìm kiếm câu query, query ID, video..." oninput="handleSearch(this.value)">
    </div>
  </div>

  <!-- Interactive Table -->
  <div class="table-container">
    <table class="eval-table" id="eval-table">
      <thead>
        <tr>
          <th>Query ID</th>
          <th>Câu Truy Vấn (Query Text)</th>
          <th>🎯 Ground Truth</th>
          <th>🏆 Top-1 Prediction</th>
          <th>Kết Quả / Sai Số (Δt)</th>
          <th>Thao Tác</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>

  <!-- Full Query Cards Section -->
  <div class="cards-header">
    <h2>📑 Chi Tiết Từng Câu Truy Vấn & Top Candidates</h2>
    <div class="subtle">Nhấp vào thẻ hoặc nút "So sánh Visual" để mở bộ xem so sánh chi tiết</div>
  </div>
  {''.join(query_cards)}
</main>

<!-- Side-by-Side Visual Comparison Inspector Modal -->
<div class="modal-overlay" id="modal-overlay" onclick="closeModalOnOverlay(event)">
  <div class="modal-container" id="modal-container">
    <div class="modal-header">
      <div class="modal-title-wrap">
        <div class="modal-title-row">
          <span class="modal-qid" id="modal-qid">KIS_L21_V001_001</span>
          <span class="tag-pill" id="modal-tag-diff">MEDIUM</span>
          <span class="tag-pill" id="modal-tag-mod">VISUAL</span>
          <span class="badge" id="modal-status-badge">❌ MISSED</span>
        </div>
        <div class="modal-query-text" id="modal-query-text">Query text here...</div>
      </div>
      <div class="modal-nav-controls">
        <button class="modal-btn-nav" onclick="prevQuery()">◀ Trước (←)</button>
        <span class="modal-counter" id="modal-counter">1 / 56</span>
        <button class="modal-btn-nav" onclick="nextQuery()">Sau (→) ▶</button>
        <button class="modal-btn-close" onclick="closeModal()" title="Đóng (Esc)">✕</button>
      </div>
    </div>

    <div class="modal-body">
      <div class="comparison-stage">
        <!-- Left: Ground Truth Stage -->
        <div class="inspector-card gt-stage">
          <div class="card-head-bar gt-head">
            <span>🎯 GROUND TRUTH (MỤC TIÊU CHUẨN)</span>
            <span id="gt-badge-info">L21_V001 / f2700</span>
          </div>
          <div class="stage-image-wrap">
            <img id="gt-stage-img" src="" alt="Ground Truth">
            <button class="img-zoom-btn" onclick="openFullImage('gt')">🔍 Xem Full</button>
          </div>
          <div class="stage-info">
            <div class="info-grid">
              <div class="info-item"><span>Video ID:</span> <strong id="gt-info-video">L21_V001</strong></div>
              <div class="info-item"><span>Frame ID:</span> <strong id="gt-info-frame">2700</strong></div>
              <div class="info-item"><span>Thời gian:</span> <strong id="gt-info-time">01:30.00</strong></div>
              <div class="info-item"><span>FPS:</span> <strong id="gt-info-fps">30.0</strong></div>
            </div>
            <div class="text-insights-box">
              <div class="insight-label">📌 Ghi chú Ground Truth:</div>
              <div class="insight-content" id="gt-info-note">Semantic Frame chuẩn được xác thực từ tập gán nhãn gt_kis.json.</div>
            </div>
          </div>
        </div>

        <!-- Right: Prediction Candidate Stage -->
        <div class="inspector-card pred-stage" id="pred-card-stage">
          <div class="card-head-bar">
            <div class="pred-head">
              <span id="pred-stage-rank">🏆 PREDICTION (TOP #1)</span>
              <span class="badge" id="pred-hit-tag">MISS</span>
            </div>
            <div id="pred-score-text" style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #60a5fa;">Score: 0.6556</div>
          </div>
          <div class="stage-image-wrap">
            <img id="pred-stage-img" src="" alt="Prediction">
            <button class="img-zoom-btn" onclick="openFullImage('pred')">🔍 Xem Full</button>
          </div>
          <div class="stage-info">
            <div class="info-grid">
              <div class="info-item"><span>Video ID:</span> <strong id="pred-info-video">L21_V007</strong></div>
              <div class="info-item"><span>Frame ID:</span> <strong id="pred-info-frame">6268</strong></div>
              <div class="info-item"><span>Thời gian:</span> <strong id="pred-info-time">03:28.93</strong></div>
              <div class="info-item"><span>Sai số Δt:</span> <strong id="pred-info-err" style="color: #f87171;">Khác Video</strong></div>
            </div>

            <!-- Branch Score Progress Bars -->
            <div class="score-breakdown-box">
              <div class="score-title">
                <span>Trọng Số Phân Nhánh (Multimodal Scores)</span>
                <span id="pred-total-score">Fused: 0.6556</span>
              </div>
              <div class="branch-bar-row">
                <span class="branch-name">🎨 Visual</span>
                <div class="branch-track"><div class="branch-fill fill-visual" id="bar-visual" style="width: 90%;"></div></div>
                <span class="branch-val" id="val-visual">0.966</span>
              </div>
              <div class="branch-bar-row">
                <span class="branch-name">🔤 OCR</span>
                <div class="branch-track"><div class="branch-fill fill-ocr" id="bar-ocr" style="width: 0%;"></div></div>
                <span class="branch-val" id="val-ocr">0.000</span>
              </div>
              <div class="branch-bar-row">
                <span class="branch-name">🎙️ ASR</span>
                <div class="branch-track"><div class="branch-fill fill-asr" id="bar-asr" style="width: 95%;"></div></div>
                <span class="branch-val" id="val-asr">0.976</span>
              </div>
              <div class="branch-bar-row">
                <span class="branch-name">📦 Object</span>
                <div class="branch-track"><div class="branch-fill fill-obj" id="bar-obj" style="width: 25%;"></div></div>
                <span class="branch-val" id="val-obj">0.251</span>
              </div>
            </div>

            <!-- Extracted OCR & Speech Texts -->
            <div class="text-insights-box">
              <div class="insight-label">🔤 Văn bản OCR nhận diện:</div>
              <div class="insight-content" id="pred-ocr-text">Không có văn bản</div>
            </div>
            <div class="text-insights-box">
              <div class="insight-label">🎙️ Giọng nói ASR bóc tách:</div>
              <div class="insight-content" id="pred-asr-text">Không có lời thoại</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Bottom Filmstrip: Top-20 Candidates Carousel -->
      <div class="filmstrip-panel">
        <div class="filmstrip-title">
          <span>🎞️ TẤT CẢ ỨNG VIÊN TOP-20 (Nhấp vào ứng viên để so sánh trực tiếp với Ground Truth)</span>
          <span style="font-size: 11px; color: var(--muted);">Phím tắt: 1-9 để chọn nhanh Top 1-9</span>
        </div>
        <div class="filmstrip-row" id="filmstrip-row">
          <!-- Dynamically populated -->
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Embedded Dataset -->
<script id="eval-dataset" type="application/json">
{json_details}
</script>

<script>
  let detailsData = [];
  try {{
    detailsData = JSON.parse(document.getElementById('eval-dataset').textContent);
  }} catch(e) {{
    console.error("Failed to parse evaluation details:", e);
  }}

  let currentQueryIndex = 0;
  let currentCandidateRank = 1;
  let activeFilter = 'failed';

  // Apply default filter on load
  document.addEventListener('DOMContentLoaded', () => {{
    applyFilter('failed');
  }});

  function filterTable(filterType, btnElem) {{
    if (btnElem) {{
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btnElem.classList.add('active');
    }}
    activeFilter = filterType;
    applyFilter(filterType);
  }}

  function applyFilter(filterType) {{
    const rows = document.querySelectorAll('.eval-table tbody tr');
    const searchVal = (document.getElementById('search-box').value || '').toLowerCase().trim();

    rows.forEach(row => {{
      let show = true;
      if (filterType === 'failed') {{
        show = row.classList.contains('failed');
      }} else if (filterType === 'hit') {{
        show = row.classList.contains('hit');
      }} else if (filterType.startsWith('mod-')) {{
        show = row.classList.contains(filterType);
      }}

      if (show && searchVal) {{
        const text = row.textContent.toLowerCase();
        show = text.includes(searchVal);
      }}
      row.style.display = show ? '' : 'none';
    }});
  }}

  function handleSearch(val) {{
    applyFilter(activeFilter);
  }}

  function openModal(queryIdx) {{
    currentQueryIndex = queryIdx;
    currentCandidateRank = 1;
    renderModal();
    const overlay = document.getElementById('modal-overlay');
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
  }}

  function openModalWithRank(queryIdx, rank) {{
    currentQueryIndex = queryIdx;
    currentCandidateRank = rank;
    renderModal();
    const overlay = document.getElementById('modal-overlay');
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
  }}

  function closeModal() {{
    const overlay = document.getElementById('modal-overlay');
    overlay.classList.remove('active');
    document.body.style.overflow = '';
  }}

  function closeModalOnOverlay(e) {{
    if (e.target.id === 'modal-overlay') {{
      closeModal();
    }}
  }}

  function prevQuery() {{
    if (detailsData.length === 0) return;
    currentQueryIndex = (currentQueryIndex - 1 + detailsData.length) % detailsData.length;
    currentCandidateRank = 1;
    renderModal();
  }}

  function nextQuery() {{
    if (detailsData.length === 0) return;
    currentQueryIndex = (currentQueryIndex + 1) % detailsData.length;
    currentCandidateRank = 1;
    renderModal();
  }}

  function selectCandidate(rank) {{
    currentCandidateRank = rank;
    renderModalPredictionOnly();
  }}

  function renderModal() {{
    if (!detailsData || !detailsData[currentQueryIndex]) return;
    const item = detailsData[currentQueryIndex];
    const gt = item.gt;
    const assetPaths = item.asset_paths || {{}};

    // Header info
    document.getElementById('modal-qid').textContent = `#${{currentQueryIndex + 1}} ${{item.query_id}}`;
    document.getElementById('modal-query-text').textContent = item.query;
    document.getElementById('modal-counter').textContent = `${{currentQueryIndex + 1}} / ${{detailsData.length}}`;

    const tagDiff = document.getElementById('modal-tag-diff');
    tagDiff.textContent = item.difficulty.toUpperCase();
    tagDiff.className = `tag-pill tag-${{item.difficulty}}`;

    const tagMod = document.getElementById('modal-tag-mod');
    tagMod.textContent = item.modality.toUpperCase();
    tagMod.className = `tag-pill tag-${{item.modality}}`;

    // Overall hit status in Top-20
    const hasHit = item.predictions.some(p => p.hit_3s);
    const modalStatus = document.getElementById('modal-status-badge');
    if (hasHit) {{
      const firstHit = item.predictions.find(p => p.hit_3s);
      modalStatus.textContent = `✅ HIT (#${{firstHit.rank}})`;
      modalStatus.className = 'badge badge-hit';
    }} else {{
      modalStatus.textContent = '❌ MISSED (Top-20)';
      modalStatus.className = 'badge badge-miss';
    }}

    // Render GT Stage
    const gtImgSrc = assetPaths[gt.thumbnail] || gt.thumbnail || '';
    document.getElementById('gt-stage-img').src = gtImgSrc;
    document.getElementById('gt-badge-info').textContent = `${{gt.video_id}} / f${{gt.frame_id}}`;
    document.getElementById('gt-info-video').textContent = gt.video_id;
    document.getElementById('gt-info-frame').textContent = gt.frame_id;
    document.getElementById('gt-info-time').textContent = gt.timestamp_text;
    document.getElementById('gt-info-fps').textContent = `${{gt.fps}} FPS`;

    // Render Filmstrip
    const filmstrip = document.getElementById('filmstrip-row');
    filmstrip.innerHTML = '';
    item.predictions.forEach(pred => {{
      const pThumb = assetPaths[pred.thumbnail] || pred.thumbnail || '';
      const isHit = Boolean(pred.hit_3s);
      const hitClass = isHit ? 'hit' : '';
      const activeClass = pred.rank === currentCandidateRank ? 'active' : '';

      const itemDiv = document.createElement('div');
      itemDiv.className = `filmstrip-item ${{hitClass}} ${{activeClass}}`;
      itemDiv.onclick = () => selectCandidate(pred.rank);
      itemDiv.innerHTML = `
        <div class="filmstrip-thumb"><img src="${{pThumb}}" alt="Rank ${{pred.rank}}" loading="lazy"></div>
        <div class="filmstrip-meta">
          <span>#${{pred.rank}} ${{isHit ? '✅' : ''}}</span>
          <span>${{pred.score.toFixed(3)}}</span>
        </div>
      `;
      filmstrip.appendChild(itemDiv);
    }});

    // Render Prediction Stage
    renderModalPredictionOnly();
  }}

  function renderModalPredictionOnly() {{
    const item = detailsData[currentQueryIndex];
    if (!item) return;
    const gt = item.gt;
    const assetPaths = item.asset_paths || {{}};
    const pred = item.predictions.find(p => p.rank === currentCandidateRank) || item.predictions[0];
    if (!pred) return;

    // Highlight filmstrip item
    document.querySelectorAll('.filmstrip-item').forEach((elem, idx) => {{
      elem.classList.toggle('active', (idx + 1) === currentCandidateRank);
    }});

    // Prediction Stage Card Border
    const predStageCard = document.getElementById('pred-card-stage');
    predStageCard.className = `inspector-card pred-stage ${{pred.hit_3s ? 'hit' : 'miss'}}`;

    // Pred details
    const predImgSrc = assetPaths[pred.thumbnail] || pred.thumbnail || '';
    document.getElementById('pred-stage-img').src = predImgSrc;
    document.getElementById('pred-stage-rank').textContent = `🏆 CANDIDATE #${{pred.rank}}`;
    document.getElementById('pred-score-text').textContent = `Score: ${{pred.score.toFixed(4)}}`;
    document.getElementById('pred-total-score').textContent = `Fused: ${{pred.score.toFixed(4)}}`;

    const predHitTag = document.getElementById('pred-hit-tag');
    if (pred.hit_3s) {{
      predHitTag.textContent = `✅ HIT (Δt ${{pred.temporal_error_sec ? pred.temporal_error_sec.toFixed(2) + 's' : ''}})`;
      predHitTag.className = 'badge badge-hit';
    }} else if (pred.video_id === gt.video_id) {{
      predHitTag.textContent = `⚠️ CÙNG VIDEO (Δt ${{pred.temporal_error_sec ? pred.temporal_error_sec.toFixed(2) + 's' : ''}})`;
      predHitTag.className = 'badge badge-diff-time';
    }} else {{
      predHitTag.textContent = `❌ KHÁC VIDEO (${{pred.video_id}})`;
      predHitTag.className = 'badge badge-miss';
    }}

    document.getElementById('pred-info-video').textContent = pred.video_id;
    document.getElementById('pred-info-frame').textContent = pred.frame_id;
    document.getElementById('pred-info-time').textContent = pred.timestamp_text;

    const errElem = document.getElementById('pred-info-err');
    if (pred.video_id === gt.video_id) {{
      errElem.textContent = `${{pred.temporal_error_sec.toFixed(2)}}s (Cùng video)`;
      errElem.style.color = pred.hit_3s ? '#34d399' : '#fbbf24';
    }} else {{
      errElem.textContent = `Khác video (${{pred.video_id}} vs ${{gt.video_id}})`;
      errElem.style.color = '#f87171';
    }}

    // Scores
    const sc = pred.scores || {{ visual: 0, ocr: 0, asr: 0, object: 0 }};
    setBar('visual', sc.visual || 0);
    setBar('ocr', sc.ocr || 0);
    setBar('asr', sc.asr || 0);
    setBar('obj', sc.object || 0);

    // Text snippets
    document.getElementById('pred-ocr-text').textContent = pred.ocr_text || '(Không có văn bản OCR trên frame này)';
    document.getElementById('pred-asr-text').textContent = pred.asr_text || '(Không có lời thoại ASR tại thời điểm này)';
  }}

  function setBar(type, val) {{
    const pct = Math.min(100, Math.max(0, val * 100));
    document.getElementById(`bar-${{type}}`).style.width = `${{pct}}%`;
    document.getElementById(`val-${{type}}`).textContent = val.toFixed(3);
  }}

  function openFullImage(target) {{
    const item = detailsData[currentQueryIndex];
    if (!item) return;
    const assetPaths = item.asset_paths || {{}};
    let src = '';
    if (target === 'gt') {{
      src = assetPaths[item.gt.thumbnail] || item.gt.thumbnail;
    }} else {{
      const pred = item.predictions.find(p => p.rank === currentCandidateRank) || item.predictions[0];
      if (pred) src = assetPaths[pred.thumbnail] || pred.thumbnail;
    }}
    if (src) window.open(src, '_blank');
  }}

  // Keyboard Navigation
  document.addEventListener('keydown', (e) => {{
    const modalActive = document.getElementById('modal-overlay').classList.contains('active');
    if (!modalActive) return;

    if (e.key === 'Escape') {{
      closeModal();
    }} else if (e.key === 'ArrowLeft') {{
      prevQuery();
    }} else if (e.key === 'ArrowRight') {{
      nextQuery();
    }} else if (e.key >= '1' && e.key <= '9') {{
      const rank = parseInt(e.key, 10);
      selectCandidate(rank);
    }}
  }});
</script>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    args.gt_path = resolve_gt_path(args.gt)
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    args.top_k = max(args.top_k, max(HIT_KS))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_payload, queries = load_gt(args.gt_path, args.limit)
    weights = parse_weights(args.weights)

    print(f"[KIS] GT: {args.gt_path}")
    print(f"[KIS] Output: {args.output_dir}")
    print(f"[KIS] Queries: {len(queries)} | top_k={args.top_k} | fusion_mode={args.fusion_mode}")

    service = RetrievalService.get_instance()
    service.initialize()
    fps_lookup = VideoMetadata()

    per_query_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for idx, q in enumerate(queries, start=1):
        query_id = str(q.get("query_id", f"query_{idx:03d}"))
        query_text = str(q.get("query", "")).strip()
        gt = q.get("gt") or {}
        metadata = q.get("metadata") or {}
        gt_video_id = str(gt.get("video_id", "")).strip()
        gt_frame = int(gt.get("semantic_frame"))
        gt_fps = fps_lookup.fps(gt_video_id)
        gt_time = gt_frame / gt_fps
        difficulty = str(metadata.get("difficulty", "unknown") or "unknown")
        modality = str(metadata.get("primary_modality", "unknown") or "unknown")

        print(f"[KIS] {idx:03d}/{len(queries):03d} {query_id}: {query_text[:80]}")
        search_res = service.search(
            query=query_text,
            top_k=args.top_k,
            fusion_mode=args.fusion_mode,
            weights=weights,
            temporal_dedup=not args.no_temporal_dedup,
            dedup_window_seconds=args.dedup_window_seconds,
        )

        gt_thumb = ""
        if not args.skip_thumbnails:
            gt_thumb = extract_gt_thumbnail(
                fps_lookup.video_path(gt_video_id),
                gt_frame,
                args.output_dir / "thumbnails" / "gt" / f"{safe_name(query_id)}.jpg",
            )

        preds = [
            normalize_result(
                raw,
                gt_video_id=gt_video_id,
                gt_time_sec=gt_time,
                fps_lookup=fps_lookup,
                service=service,
                query_id=query_id,
                output_dir=args.output_dir,
                skip_thumbnails=args.skip_thumbnails,
            )
            for raw in search_res.get("results", [])[: args.top_k]
        ]

        ranks_by_tol = {
            int(tol): first_hit_rank(preds, tolerance=tol, k=args.top_k) for tol in TOLERANCES_SEC
        }
        first_same_video = next((p for p in preds if p["video_id"] == gt_video_id), None)
        first_same_video_error = (
            None if first_same_video is None else float(first_same_video["temporal_error_sec"])
        )
        best_same_video_error = None
        same_video_errors = [
            float(p["temporal_error_sec"])
            for p in preds
            if p.get("temporal_error_sec") is not None
        ]
        if same_video_errors:
            best_same_video_error = min(same_video_errors)

        top1 = preds[0] if preds else {}
        row = {
            "query_id": query_id,
            "query": query_text,
            "gt_video_id": gt_video_id,
            "gt_semantic_frame": gt_frame,
            "gt_fps": gt_fps,
            "gt_time_sec": round(gt_time, 6),
            "gt_timestamp_text": format_time(gt_time),
            "difficulty": difficulty,
            "modality": modality,
            "first_hit_rank_1s": ranks_by_tol[1],
            "first_hit_rank_3s": ranks_by_tol[3],
            "first_hit_rank_5s": ranks_by_tol[5],
            "first_same_video_rank": None if first_same_video is None else first_same_video["rank"],
            "first_same_video_temporal_error_sec": None
            if first_same_video_error is None
            else round(first_same_video_error, 6),
            "best_same_video_temporal_error_top20_sec": None
            if best_same_video_error is None
            else round(best_same_video_error, 6),
            "top1_video_id": top1.get("video_id", ""),
            "top1_frame_id": top1.get("frame_id", ""),
            "top1_time_sec": top1.get("timestamp_sec", ""),
            "top1_score": top1.get("score", ""),
            "top1_temporal_error_sec": top1.get("temporal_error_sec", ""),
            "retrieval_elapsed_ms": search_res.get("elapsed_ms", ""),
            "fusion_mode": search_res.get("fusion_mode", args.fusion_mode),
            "effective_weights_json": json.dumps(search_res.get("effective_weights", {}), ensure_ascii=False),
            "top_predictions_json": json.dumps(preds, ensure_ascii=False),
        }
        for tol in TOLERANCES_SEC:
            tol_i = int(tol)
            for k in HIT_KS:
                row[f"hit_at_{k}_{tol_i}s"] = (
                    ranks_by_tol[tol_i] is not None and int(ranks_by_tol[tol_i]) <= k
                )
        per_query_rows.append(row)

        asset_paths = {gt_thumb: rel_asset(gt_thumb, args.output_dir)} if gt_thumb else {}
        for pred in preds:
            if pred.get("thumbnail"):
                asset_paths[pred["thumbnail"]] = rel_asset(pred["thumbnail"], args.output_dir)
        detail_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "difficulty": difficulty,
                "modality": modality,
                "gt": {
                    "video_id": gt_video_id,
                    "frame_id": gt_frame,
                    "fps": gt_fps,
                    "timestamp_sec": gt_time,
                    "timestamp_text": format_time(gt_time),
                    "thumbnail": gt_thumb,
                },
                "predictions": preds,
                "asset_paths": asset_paths,
            }
        )

    metrics = compute_metrics(per_query_rows, gt_payload, args)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = [
        "query_id",
        "query",
        "gt_video_id",
        "gt_semantic_frame",
        "gt_fps",
        "gt_time_sec",
        "gt_timestamp_text",
        "difficulty",
        "modality",
        "first_hit_rank_1s",
        "first_hit_rank_3s",
        "first_hit_rank_5s",
        "first_same_video_rank",
        "first_same_video_temporal_error_sec",
        "best_same_video_temporal_error_top20_sec",
        "top1_video_id",
        "top1_frame_id",
        "top1_time_sec",
        "top1_score",
        "top1_temporal_error_sec",
        "retrieval_elapsed_ms",
        "fusion_mode",
        "effective_weights_json",
    ]
    for tol in TOLERANCES_SEC:
        for k in HIT_KS:
            fields.append(f"hit_at_{k}_{int(tol)}s")
    fields.append("top_predictions_json")
    write_csv(args.output_dir / "per_query_results.csv", per_query_rows, fields)

    failure_rows = [row for row in per_query_rows if row.get("first_hit_rank_3s") is None]
    write_csv(args.output_dir / "failures.csv", failure_rows, fields)
    generate_html_report(args.output_dir / "report.html", per_query_rows, detail_rows, metrics)

    def format_row(name: str, block: dict[str, Any], total_w: int = 24) -> str:
        num_q = block.get("num_queries", "")
        num_str = f"({num_q} q)" if num_q != "" else ""
        hits = " | ".join(f"{block.get(f'hit_at_{k}', 0.0):6.1%}" for k in HIT_KS)
        mrr = block.get("mrr", 0.0)
        return f"  {name:<{total_w}} {num_str:<8} | {hits} | {mrr:.4f}"

    header_hits = " | ".join(f"Hit@{k:<2}" for k in HIT_KS)
    print("\n" + "=" * 96)
    print(" 📊 KIS EVALUATION SUMMARY (HIT@1 / 5 / 10 / 20 / 30 / 50 / MRR)")
    print("=" * 96)

    print(f"\n--- 1. OVERALL BY TEMPORAL TOLERANCE ---")
    print(f"  {'Tolerance':<24} {'Queries':<8} | {header_hits} | MRR")
    print("  " + "-" * 92)
    for tol in TOLERANCES_SEC:
        tol_key = f"tolerance_{int(tol)}s"
        blk = metrics["by_tolerance"][tol_key]["overall"]
        print(format_row(f"±{int(tol)}s Tolerance", blk))

    print(f"\n--- 2. BREAKDOWN BY MODALITY (Primary ±3s) ---")
    print(f"  {'Modality':<24} {'Queries':<8} | {header_hits} | MRR")
    print("  " + "-" * 92)
    for mod_name, blk in metrics["primary"]["by_modality"].items():
        print(format_row(mod_name, blk))

    print(f"\n--- 3. BREAKDOWN BY DIFFICULTY (Primary ±3s) ---")
    print(f"  {'Difficulty':<24} {'Queries':<8} | {header_hits} | MRR")
    print("  " + "-" * 92)
    for diff_name, blk in metrics["primary"]["by_difficulty"].items():
        print(format_row(diff_name, blk))

    print(f"\n--- 4. BREAKDOWN BY VIDEO (Primary ±3s) ---")
    print(f"  {'Video ID':<24} {'Queries':<8} | {header_hits} | MRR")
    print("  " + "-" * 92)
    for vid_name, blk in metrics["primary"]["by_video"].items():
        print(format_row(vid_name, blk))

    # 5. Miss@20 but Hit@30 / Hit@50 Analysis
    print("\n" + "=" * 96)
    print(" 🔍 CANDIDATE POOL ANALYSIS: MISS@20 BUT HIT@30 / HIT@50 (Tolerance ±3s)")
    print("=" * 96)

    miss20_hit30 = []
    miss20_hit50 = []

    for row in per_query_rows:
        r = row.get("first_hit_rank_3s")
        if r is not None:
            r = int(r)
            if 20 < r <= 30:
                miss20_hit30.append((row["query_id"], row["query"], r, row["difficulty"], row["modality"]))
            elif 30 < r <= 50:
                miss20_hit50.append((row["query_id"], row["query"], r, row["difficulty"], row["modality"]))

    total_q = len(per_query_rows)
    h20_val = metrics['primary']['overall']['hit_at_20']
    h30_val = metrics['primary']['overall']['hit_at_30']
    h50_val = metrics['primary']['overall']['hit_at_50']

    print(f"Total queries evaluated: {total_q}")
    print(f"Hit@20: {h20_val:6.1%} ({round(h20_val * total_q):2d}/{total_q})")
    print(f"Hit@30: {h30_val:6.1%} ({round(h30_val * total_q):2d}/{total_q}) -> Gain vs Top-20: +{len(miss20_hit30)} queries (+{len(miss20_hit30)/total_q:.1%})")
    print(f"Hit@50: {h50_val:6.1%} ({round(h50_val * total_q):2d}/{total_q}) -> Gain vs Top-20: +{len(miss20_hit30) + len(miss20_hit50)} queries (+{(len(miss20_hit30) + len(miss20_hit50))/total_q:.1%})")

    if miss20_hit30:
        print(f"\n[Queries Missed in Top-20, Found in Rank 21-30]: ({len(miss20_hit30)} queries)")
        for qid, qtext, rank, diff, mod in miss20_hit30:
            print(f"  • Rank #{rank:2d} | [{diff:<6}] [{mod:<6}] {qid}: {qtext}")

    if miss20_hit50:
        print(f"\n[Queries Missed in Top-30, Found in Rank 31-50]: ({len(miss20_hit50)} queries)")
        for qid, qtext, rank, diff, mod in miss20_hit50:
            print(f"  • Rank #{rank:2d} | [{diff:<6}] [{mod:<6}] {qid}: {qtext}")

    if not miss20_hit30 and not miss20_hit50:
        print("\nNo additional queries found between rank 21 and 50.")
    print("=" * 96 + "\n")

    print("[KIS] Done.")
    print(f"[KIS] Wrote: {args.output_dir / 'metrics.json'}")
    print(f"[KIS] Wrote: {args.output_dir / 'per_query_results.csv'}")
    print(f"[KIS] Wrote: {args.output_dir / 'failures.csv'}")
    print(f"[KIS] Wrote: {args.output_dir / 'report.html'}")


if __name__ == "__main__":
    evaluate(parse_args())
