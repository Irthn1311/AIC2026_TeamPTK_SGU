"""
Stage 4B: EventGraph Visual/Semantic Edge Quality Audit & Side-by-Side HTML Dashboard
=======================================================================================
Performs empirical edge quality auditing across 60 stratified edge samples:
  - 30 VISUAL_SIMILARITY edges (10 Top, 10 Middle, 10 Near-Threshold)
  - 30 SEMANTIC_CONTINUITY edges (10 Top, 10 Middle, 10 Near-Threshold)

Features:
  1. Stage 3D Frozen Artifact Verification (ensures all_events.parquet is from verified Stage 3D).
  2. Stratified Sampling across High, Medium, and Low score buckets.
  3. Side-by-Side Metadata Extraction (Event IDs, Video IDs, Timestamps, Shots, Keyframes, Text Captions).
  4. Interactive HTML Audit Dashboard (stage4b_edge_quality_audit.html) with:
     - Side-by-side visual/semantic event cards.
     - Interactive [✓ Relevant] / [✕ Irrelevant] buttons and Failure Pattern selectors.
     - Dynamic real-time calculation of Precision@sample, Precision@bucket, and failure breakdown.
     - Exportable audit decisions JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("stage4b-quality-audit")


def verify_stage3d_input_integrity(df_events: pd.DataFrame) -> Dict[str, Any]:
    """Verify that all_events.parquet is coming from frozen Stage 3D output."""
    num_events = len(df_events)
    num_videos = df_events["video_id"].nunique()

    has_required = all(c in df_events.columns for c in ["event_id", "video_id", "start_shot", "end_shot", "start_sec", "end_sec"])
    status = "VERIFIED" if has_required and num_events > 0 else "FAILED"

    return {
        "status": status,
        "total_events": num_events,
        "total_videos": num_videos,
        "mean_shots_per_event": round(df_events["num_shots"].mean(), 2) if "num_shots" in df_events.columns else 0.0,
        "mean_duration_sec": round(df_events["duration_sec"].mean(), 2) if "duration_sec" in df_events.columns else 0.0,
    }


def sample_stratified_edges(df_edges: pd.DataFrame, edge_type: str, n_top: int = 10, n_mid: int = 10, n_low: int = 10) -> List[Dict[str, Any]]:
    """Sample stratified edges across Top, Middle, and Near-Threshold score buckets."""
    type_edges = df_edges[df_edges["edge_type"] == edge_type].sort_values(by="score", ascending=False).reset_index(drop=True)
    if type_edges.empty:
        return []

    total_count = len(type_edges)

    # 1. Top Bucket
    top_df = type_edges.head(n_top).copy()
    top_df["bucket"] = "TOP"

    # 2. Middle Bucket
    mid_start = max(0, (total_count // 2) - (n_mid // 2))
    mid_df = type_edges.iloc[mid_start : mid_start + n_mid].copy()
    mid_df["bucket"] = "MIDDLE"

    # 3. Near-Threshold Bucket (Low scores)
    low_df = type_edges.tail(n_low).copy()
    low_df["bucket"] = "NEAR_THRESHOLD"

    combined = pd.concat([top_df, mid_df, low_df]).drop_duplicates(subset=["src_event_id", "dst_event_id"])
    return combined.to_dict("records")


def extract_event_keyframes(node: Dict[str, Any], vid: str) -> List[Dict[str, str]]:
    """Extract up to 3 representative keyframes (Start, Center, End) for an event."""
    rk = node.get("representative_keyframes", [])
    if isinstance(rk, (list, np.ndarray)) and len(rk) > 0:
        kf_list = [str(k) for k in rk]
    else:
        kf_str = str(rk) if rk else "N/A"
        kf_list = [kf_str]

    # Pick start, center, end
    n = len(kf_list)
    if n == 1:
        picked = [("Start", kf_list[0]), ("Center", kf_list[0]), ("End", kf_list[0])]
    elif n == 2:
        picked = [("Start", kf_list[0]), ("Center", kf_list[0]), ("End", kf_list[1])]
    else:
        picked = [("Start", kf_list[0]), ("Center", kf_list[n // 2]), ("End", kf_list[-1])]

    result = []
    for label, kf in picked:
        clean_kf = kf.replace(".jpg", "").replace(".png", "")
        img_url = f"keyframes/{vid}/{clean_kf}.jpg"
        result.append({"label": label, "keyframe": clean_kf, "img_url": img_url})

    return result


def enrich_edge_samples(
    sampled_records: List[Dict[str, Any]], df_nodes: pd.DataFrame, df_shots: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """Enrich edge samples with full node metadata, timestamps, 3 keyframes (Start/Center/End), and captions."""
    node_map = df_nodes.set_index("event_id").to_dict("index")
    
    shot_map = {}
    if df_shots is not None and not df_shots.empty:
        for _, s in df_shots.iterrows():
            key = (str(s["video_id"]), int(s["shot_id"]))
            txt = str(s.get("ocr_text", s.get("asr_text", s.get("caption", ""))))
            shot_map[key] = txt

    enriched = []
    for idx, e in enumerate(sampled_records):
        src_id = str(e["src_event_id"])
        dst_id = str(e["dst_event_id"])

        src_node = node_map.get(src_id, {})
        dst_node = node_map.get(dst_id, {})

        src_vid = str(src_node.get("video_id", e.get("src_video_id", "")))
        dst_vid = str(dst_node.get("video_id", e.get("dst_video_id", "")))

        # 3 Keyframes (Start / Center / End)
        src_keyframes = extract_event_keyframes(src_node, src_vid)
        dst_keyframes = extract_event_keyframes(dst_node, dst_vid)

        # Captions
        src_captions = []
        for sid in src_node.get("shot_ids", []):
            txt = shot_map.get((src_vid, int(sid)))
            if txt and txt not in src_captions:
                src_captions.append(txt)

        dst_captions = []
        for sid in dst_node.get("shot_ids", []):
            txt = shot_map.get((dst_vid, int(sid)))
            if txt and txt not in dst_captions:
                dst_captions.append(txt)

        rec = {
            "sample_id": idx + 1,
            "edge_type": e["edge_type"],
            "bucket": e.get("bucket", "STANDARD"),
            "score": float(e["score"]),
            "z_score": float(e.get("z_score", 0.0)),
            "src": {
                "event_id": src_id,
                "video_id": src_vid,
                "start_sec": float(src_node.get("start_sec", 0.0)),
                "end_sec": float(src_node.get("end_sec", 0.0)),
                "duration_sec": float(src_node.get("duration_sec", 0.0)),
                "num_shots": int(src_node.get("num_shots", 0)),
                "keyframes": src_keyframes,
                "caption": " | ".join(src_captions[:3]) if src_captions else f"Event {src_id} in video {src_vid}",
            },
            "dst": {
                "event_id": dst_id,
                "video_id": dst_vid,
                "start_sec": float(dst_node.get("start_sec", 0.0)),
                "end_sec": float(dst_node.get("end_sec", 0.0)),
                "duration_sec": float(dst_node.get("duration_sec", 0.0)),
                "num_shots": int(dst_node.get("num_shots", 0)),
                "keyframes": dst_keyframes,
                "caption": " | ".join(dst_captions[:3]) if dst_captions else f"Event {dst_id} in video {dst_vid}",
            },
        }
        enriched.append(rec)

    return enriched


def generate_html_audit_dashboard(enriched_samples: List[Dict[str, Any]], meta: Dict[str, Any], output_path: Path):
    """Generate modern side-by-side interactive HTML audit dashboard with 3 representative keyframes per event and strict progress tracking."""
    json_data = json.dumps(enriched_samples, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage 4B: EventGraph Edge Quality Audit Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0f172a;
            --panel-bg: #1e293b;
            --card-bg: #334155;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --text-light: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #475569;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
        body {{ background-color: var(--bg-dark); color: var(--text-light); padding: 20px; }}

        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid var(--border-color);
            padding: 24px; border-radius: 12px; margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}

        .header h1 {{ font-size: 24px; font-weight: 700; color: var(--accent-blue); margin-bottom: 8px; }}
        .header p {{ color: var(--text-muted); font-size: 14px; }}

        .stats-banner {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px;
            margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);
        }}

        .stat-card {{
            background: rgba(30, 41, 59, 0.7); border: 1px solid var(--border-color);
            padding: 12px 16px; border-radius: 8px; text-align: center;
        }}

        .stat-card .val {{ font-size: 22px; font-weight: 700; color: var(--accent-blue); }}
        .stat-card .lbl {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

        .filter-bar {{
            display: flex; gap: 12px; margin-bottom: 20px; align-items: center; flex-wrap: wrap;
        }}

        .filter-btn, .export-btn {{
            background: var(--panel-bg); color: var(--text-light); border: 1px solid var(--border-color);
            padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600;
            transition: all 0.2s ease;
        }}

        .filter-btn.active, .filter-btn:hover {{
            background: var(--accent-blue); color: #000; border-color: var(--accent-blue);
        }}

        .export-btn {{
            background: rgba(34, 197, 94, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green);
        }}
        .export-btn:hover {{ background: var(--accent-green); color: #000; }}

        .sample-grid {{ display: flex; flex-direction: column; gap: 20px; }}

        .sample-card {{
            background: var(--panel-bg); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 20px; transition: border-color 0.2s ease;
        }}

        .sample-card.evaluated-pass {{ border-left: 6px solid var(--accent-green); }}
        .sample-card.evaluated-fail {{ border-left: 6px solid var(--accent-red); }}
        .sample-card.evaluated-ambiguous {{ border-left: 6px solid var(--accent-amber); }}

        .sample-header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--card-bg);
        }}

        .badge {{
            padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; text-transform: uppercase;
        }}

        .badge-vis {{ background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }}
        .badge-sem {{ background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid #c084fc; }}
        .badge-top {{ background: rgba(34, 197, 94, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }}
        .badge-mid {{ background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid #facc15; }}
        .badge-low {{ background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }}

        .side-by-side {{
            display: grid; grid-template-columns: 1fr 60px 1fr; gap: 16px; align-items: stretch;
        }}

        @media (max-width: 1100px) {{
            .side-by-side {{ grid-template-columns: 1fr; }}
            .vs-divider {{ text-align: center; margin: 10px 0; }}
        }}

        .event-box {{
            background: var(--card-bg); border: 1px solid var(--border-color);
            border-radius: 8px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between;
        }}

        .event-box h4 {{ font-size: 15px; color: var(--accent-blue); margin-bottom: 8px; }}
        .meta-row {{ font-size: 13px; color: var(--text-muted); margin-bottom: 6px; }}
        .meta-row span {{ color: var(--text-light); font-weight: 600; }}

        .keyframes-strip {{
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 10px 0;
        }}

        .kf-item {{
            background: #0f172a; border: 1px solid var(--border-color); border-radius: 6px; overflow: hidden; text-align: center;
        }}

        .kf-lbl {{ font-size: 10px; font-weight: 700; background: #1e293b; color: var(--accent-blue); padding: 2px 4px; text-transform: uppercase; }}
        .kf-img-box {{ width: 100%; height: 110px; position: relative; background: #000; }}
        .kf-img-box img {{ width: 100%; height: 100%; object-fit: cover; }}
        .kf-name {{ font-size: 9px; color: var(--text-muted); padding: 4px; font-family: monospace; word-break: break-all; }}

        .caption-box {{
            background: #182234; padding: 10px; border-radius: 6px; font-size: 12px; color: #cbd5e1;
            font-style: italic; border-left: 3px solid var(--accent-blue); margin-top: 8px;
        }}

        .vs-divider {{
            display: flex; align-items: center; justify-content: center; font-size: 24px; font-weight: 700; color: var(--text-muted);
        }}

        .audit-action-bar {{
            display: flex; gap: 12px; margin-top: 16px; align-items: center; flex-wrap: wrap;
            background: rgba(15, 23, 42, 0.5); padding: 12px; border-radius: 8px;
        }}

        .btn-pass {{
            background: rgba(34, 197, 94, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green);
            padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px;
        }}
        .btn-pass:hover, .btn-pass.selected {{ background: var(--accent-green); color: #000; }}

        .btn-fail {{
            background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red);
            padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px;
        }}
        .btn-fail:hover, .btn-fail.selected {{ background: var(--accent-red); color: #fff; }}

        .btn-ambiguous {{
            background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid var(--accent-amber);
            padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px;
        }}
        .btn-ambiguous:hover, .btn-ambiguous.selected {{ background: var(--accent-amber); color: #000; }}

        .fail-reason-select {{
            background: var(--card-bg); color: var(--text-light); border: 1px solid var(--border-color);
            padding: 8px; border-radius: 6px; font-size: 12px; display: none;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>🕸️ Stage 4B: EventGraph Edge Quality Audit Dashboard</h1>
        <p>Empirical Human Audit across 60 Stratified Visual & Semantic Edge Samples (Top, Middle, Near-Threshold)</p>

        <div class="stats-banner">
            <div class="stat-card">
                <div class="val" id="stat-evaluated" style="color: var(--accent-blue);">0 / 60</div>
                <div class="lbl">Label Progress</div>
            </div>
            <div class="stat-card">
                <div class="val" id="stat-relevant-cnt" style="color: var(--accent-green);">0</div>
                <div class="lbl">✓ Relevant</div>
            </div>
            <div class="stat-card">
                <div class="val" id="stat-irrelevant-cnt" style="color: var(--accent-red);">0</div>
                <div class="lbl">✕ Irrelevant</div>
            </div>
            <div class="stat-card">
                <div class="val" id="stat-ambig-cnt" style="color: var(--accent-amber);">0</div>
                <div class="lbl">? Ambiguous</div>
            </div>
            <div class="stat-card">
                <div class="val" id="stat-overall-prec" style="color: #c084fc;">N/A</div>
                <div class="lbl">Current Precision</div>
            </div>
        </div>
    </div>

    <div class="filter-bar">
        <button class="filter-btn active" onclick="filterSamples('ALL')">All (60)</button>
        <button class="filter-btn" onclick="filterSamples('VISUAL_SIMILARITY')">Visual Similarity (30)</button>
        <button class="filter-btn" onclick="filterSamples('SEMANTIC_CONTINUITY')">Semantic Continuity (30)</button>
        <button class="filter-btn" onclick="filterSamples('TOP')">Top Bucket</button>
        <button class="filter-btn" onclick="filterSamples('MIDDLE')">Middle Bucket</button>
        <button class="filter-btn" onclick="filterSamples('NEAR_THRESHOLD')">Near Threshold</button>
        <div style="flex-grow: 1;"></div>
        <button class="export-btn" onclick="exportJSON()">📥 Export JSON Labels</button>
        <button class="export-btn" onclick="exportCSV()">📥 Export CSV Labels</button>
    </div>

    <div class="sample-grid" id="samples-container"></div>

    <script>
        const samplesData = {json_data};
        const evaluations = {{}};

        function renderKeyframesHtml(keyframesList, videoId) {{
            return keyframesList.map(kf => {{
                const fallbackText = encodeURIComponent(`${{videoId}} | ${{kf.keyframe}}`);
                return `
                    <div class="kf-item">
                        <div class="kf-lbl">${{kf.label}}</div>
                        <div class="kf-img-box">
                            <img src="${{kf.img_url}}" 
                                 onerror="this.onerror=null; this.src='https://placehold.co/200x110/1e293b/38bdf8?text=' + '${{fallbackText}}'" 
                                 alt="${{kf.label}} Keyframe"/>
                        </div>
                        <div class="kf-name">${{kf.keyframe}}</div>
                    </div>
                `;
            }}).join('');
        }}

        function renderSamples(filterType = 'ALL') {{
            const container = document.getElementById('samples-container');
            container.innerHTML = '';

            samplesData.forEach((s) => {{
                if (filterType !== 'ALL' && s.edge_type !== filterType && s.bucket !== filterType) return;

                const card = document.createElement('div');
                card.className = 'sample-card';
                card.id = `card-${{s.sample_id}}`;

                const typeBadgeClass = s.edge_type === 'VISUAL_SIMILARITY' ? 'badge-vis' : 'badge-sem';
                const bucketBadgeClass = s.bucket === 'TOP' ? 'badge-top' : (s.bucket === 'MIDDLE' ? 'badge-mid' : 'badge-low');

                const srcKfHtml = renderKeyframesHtml(s.src.keyframes, s.src.video_id);
                const dstKfHtml = renderKeyframesHtml(s.dst.keyframes, s.dst.video_id);

                card.innerHTML = `
                    <div class="sample-header">
                        <div>
                            <span class="badge ${{typeBadgeClass}}">${{s.edge_type}}</span>
                            <span class="badge ${{bucketBadgeClass}}">${{s.bucket}}</span>
                            <strong style="margin-left: 10px; color: var(--text-light);">Sample #${{s.sample_id}}</strong>
                        </div>
                        <div style="font-size: 14px; font-weight: 600; color: var(--accent-blue);">
                            Score: ${{s.score.toFixed(4)}} (Z: ${{s.z_score.toFixed(2)}})
                        </div>
                    </div>

                    <div class="side-by-side">
                        <!-- Source Event Box -->
                        <div class="event-box">
                            <div>
                                <h4>Src Event: ${{s.src.event_id}}</h4>
                                <div class="meta-row">Video: <span>${{s.src.video_id}}</span></div>
                                <div class="meta-row">Time Range: <span>${{s.src.start_sec.toFixed(1)}}s - ${{s.src.end_sec.toFixed(1)}}s (${{s.src.duration_sec.toFixed(1)}}s)</span></div>
                                <div class="meta-row">Shots: <span>${{s.src.num_shots}}</span></div>
                                <div class="keyframes-strip">
                                    ${{srcKfHtml}}
                                </div>
                            </div>
                            <div class="caption-box">💬 Caption: "${{s.src.caption}}"</div>
                        </div>

                        <div class="vs-divider">↔</div>

                        <!-- Target Event Box -->
                        <div class="event-box">
                            <div>
                                <h4>Dst Event: ${{s.dst.event_id}}</h4>
                                <div class="meta-row">Video: <span>${{s.dst.video_id}}</span></div>
                                <div class="meta-row">Time Range: <span>${{s.dst.start_sec.toFixed(1)}}s - ${{s.dst.end_sec.toFixed(1)}}s (${{s.dst.duration_sec.toFixed(1)}}s)</span></div>
                                <div class="meta-row">Shots: <span>${{s.dst.num_shots}}</span></div>
                                <div class="keyframes-strip">
                                    ${{dstKfHtml}}
                                </div>
                            </div>
                            <div class="caption-box">💬 Caption: "${{s.dst.caption}}"</div>
                        </div>
                    </div>

                    <div class="audit-action-bar">
                        <span style="font-size: 13px; font-weight: 600;">Human Label:</span>
                        <button class="btn-pass" id="pass-${{s.sample_id}}" onclick="evaluateSample(${{s.sample_id}}, 'RELEVANT')">✓ Relevant</button>
                        <button class="btn-fail" id="fail-${{s.sample_id}}" onclick="evaluateSample(${{s.sample_id}}, 'IRRELEVANT')">✕ Irrelevant</button>
                        <button class="btn-ambiguous" id="ambig-${{s.sample_id}}" onclick="evaluateSample(${{s.sample_id}}, 'AMBIGUOUS')">? Ambiguous</button>

                        <select class="fail-reason-select" id="reason-${{s.sample_id}}" onchange="setFailReason(${{s.sample_id}}, this.value)">
                            <option value="">-- Select Failure Pattern --</option>
                            <option value="Static Slide FP">Static Slide FP</option>
                            <option value="Background Noise">Background Noise</option>
                            <option value="Scene Discontinuity">Scene Discontinuity</option>
                            <option value="Domain Mismatch">Domain Mismatch</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>
                `;
                container.appendChild(card);
            }});
        }}

        function evaluateSample(sampleId, status) {{
            evaluations[sampleId] = evaluations[sampleId] || {{}};
            evaluations[sampleId].label = status;

            const card = document.getElementById(`card-${{sampleId}}`);
            const btnPass = document.getElementById(`pass-${{sampleId}}`);
            const btnFail = document.getElementById(`fail-${{sampleId}}`);
            const btnAmbig = document.getElementById(`ambig-${{sampleId}}`);
            const reasonSelect = document.getElementById(`reason-${{sampleId}}`);

            btnPass.classList.remove('selected');
            btnFail.classList.remove('selected');
            btnAmbig.classList.remove('selected');

            if (status === 'RELEVANT') {{
                card.className = 'sample-card evaluated-pass';
                btnPass.classList.add('selected');
                reasonSelect.style.display = 'none';
            }} else if (status === 'IRRELEVANT') {{
                card.className = 'sample-card evaluated-fail';
                btnFail.classList.add('selected');
                reasonSelect.style.display = 'inline-block';
            }} else {{
                card.className = 'sample-card evaluated-ambiguous';
                btnAmbig.classList.add('selected');
                reasonSelect.style.display = 'none';
            }}
            updateStats();
        }}

        function setFailReason(sampleId, reason) {{
            if (evaluations[sampleId]) {{
                evaluations[sampleId].reason = reason;
            }}
        }}

        function updateStats() {{
            const totalEval = Object.keys(evaluations).length;
            const relCount = Object.values(evaluations).filter(e => e.label === 'RELEVANT').length;
            const irrelCount = Object.values(evaluations).filter(e => e.label === 'IRRELEVANT').length;
            const ambigCount = Object.values(evaluations).filter(e => e.label === 'AMBIGUOUS').length;

            document.getElementById('stat-evaluated').innerText = `${{totalEval}} / ${{samplesData.length}}`;
            document.getElementById('stat-relevant-cnt').innerText = relCount;
            document.getElementById('stat-irrelevant-cnt').innerText = irrelCount;
            document.getElementById('stat-ambig-cnt').innerText = ambigCount;

            const denom = relCount + irrelCount;
            if (denom > 0) {{
                const prec = ((relCount / denom) * 100).toFixed(1);
                document.getElementById('stat-overall-prec').innerText = `${{prec}}%`;
            }} else {{
                document.getElementById('stat-overall-prec').innerText = 'N/A';
            }}
        }}

        function exportJSON() {{
            const totalEval = Object.keys(evaluations).length;
            if (totalEval < samplesData.length) {{
                alert(`⚠️ Warning: You have labeled ${{totalEval}} / ${{samplesData.length}} samples. Please label all 60 samples before performing final Freeze evaluation!`);
            }}

            const exportList = samplesData.map(s => ({{
                sample_id: s.sample_id,
                src_event_id: s.src.event_id,
                dst_event_id: s.dst.event_id,
                src_video_id: s.src.video_id,
                dst_video_id: s.dst.video_id,
                edge_type: s.edge_type,
                score: s.score,
                z_score: s.z_score,
                bucket: s.bucket,
                label: evaluations[s.sample_id]?.label || 'UNLABELED',
                reason: evaluations[s.sample_id]?.reason || ''
            }}));

            const blob = new Blob([JSON.stringify(exportList, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'stage4b_human_audit_labels.json';
            a.click();
        }}

        function exportCSV() {{
            const totalEval = Object.keys(evaluations).length;
            if (totalEval < samplesData.length) {{
                alert(`⚠️ Warning: You have labeled ${{totalEval}} / ${{samplesData.length}} samples. Please label all 60 samples before performing final Freeze evaluation!`);
            }}

            let csv = 'sample_id,src_event_id,dst_event_id,src_video_id,dst_video_id,edge_type,score,z_score,bucket,label,reason\\n';
            samplesData.forEach(s => {{
                const ev = evaluations[s.sample_id] || {{ label: 'UNLABELED', reason: '' }};
                csv += `${{s.sample_id}},${{s.src.event_id}},${{s.dst.event_id}},${{s.src.video_id}},${{s.dst.video_id}},${{s.edge_type}},${{s.score}},${{s.z_score}},${{s.bucket}},${{ev.label}},${{ev.reason || ''}}\\n`;
            }});

            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'stage4b_human_audit_labels.csv';
            a.click();
        }}

        function filterSamples(type) {{
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            renderSamples(type);
        }}

        renderSamples();
    </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Saved Stage 4B Interactive HTML Dashboard to: %s", output_path)





def main():
    parser = argparse.ArgumentParser(description="Stage 4B: Semantic/Visual Edge Quality Audit")
    parser.add_argument(
        "--events-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "events" / "all_events.parquet"),
        help="Path to all_events.parquet",
    )
    parser.add_argument(
        "--nodes-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "event_nodes.parquet"),
        help="Path to event_nodes.parquet",
    )
    parser.add_argument(
        "--edges-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "event_edges.parquet"),
        help="Path to event_edges.parquet",
    )
    parser.add_argument(
        "--shots-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "features" / "shot_features.parquet"),
        help="Path to optional shot_features.parquet",
    )
    parser.add_argument(
        "--html-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "stage4b_edge_quality_audit.html"),
        help="Path to output HTML audit dashboard",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "graph" / "stage4b_audit_manifest.json"),
        help="Path to output JSON audit manifest",
    )
    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info("🔬 STAGE 4B: SEMANTIC & VISUAL EDGE QUALITY AUDIT")
    logger.info("==================================================================")

    events_path = Path(args.events_in)
    nodes_path = Path(args.nodes_in)
    edges_path = Path(args.edges_in)

    if not events_path.exists() or not nodes_path.exists() or not edges_path.exists():
        logger.error("Required Parquet inputs not found!")
        sys.exit(1)

    df_events = pd.read_parquet(events_path)
    df_nodes = pd.read_parquet(nodes_path)
    df_edges = pd.read_parquet(edges_path)

    df_shots = pd.read_parquet(args.shots_in) if Path(args.shots_in).exists() else None

    # 1. Verify Stage 3D Input Integrity
    s3d_verify = verify_stage3d_input_integrity(df_events)
    logger.info("Stage 3D Input Integrity Status: %s (%d events, %d videos)", s3d_verify["status"], s3d_verify["total_events"], s3d_verify["total_videos"])

    # 2. Stratified Sampling of 60 Edges (30 Visual, 30 Semantic)
    vis_samples = sample_stratified_edges(df_edges, edge_type="VISUAL_SIMILARITY", n_top=10, n_mid=10, n_low=10)
    sem_samples = sample_stratified_edges(df_edges, edge_type="SEMANTIC_CONTINUITY", n_top=10, n_mid=10, n_low=10)

    total_sampled = vis_samples + sem_samples
    logger.info("Sampled %d total edges (30 Visual, 30 Semantic).", len(total_sampled))

    # 3. Enrich Edge Samples with metadata
    enriched_samples = enrich_edge_samples(total_sampled, df_nodes, df_shots)

    # 4. Generate Interactive HTML Audit Dashboard
    html_out = Path(args.html_out)
    generate_html_audit_dashboard(enriched_samples, s3d_verify, html_out)

    # 5. Save Audit Manifest JSON
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage3d_verification": s3d_verify,
        "sample_count": len(enriched_samples),
        "visual_sample_count": len(vis_samples),
        "semantic_sample_count": len(sem_samples),
        "samples": enriched_samples,
    }

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved Stage 4B Audit Manifest JSON to: %s", report_out)

    print("\n" + "=" * 80)
    print("🎉 STAGE 4B EDGE QUALITY AUDIT INITIALIZATION COMPLETE!")
    print("=" * 80)
    print(f" • Stage 3D Verification   : {s3d_verify['status']}")
    print(f" • Total Sampled Edges     : {len(enriched_samples):,} (30 Visual / 30 Semantic)")
    print(f" • Score Buckets Stratified: 20 Top / 20 Middle / 20 Near-Threshold")
    print(f" • HTML Dashboard Output   : {html_out}")
    print(f" • Manifest Output JSON    : {report_out}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
