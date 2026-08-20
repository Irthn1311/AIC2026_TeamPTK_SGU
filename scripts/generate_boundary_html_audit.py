"""
Stage 3C HTML Visual Audit Generator (AI Challenge 2026 EventGraph)
====================================================================
Senior ML Engineer Tool: Generates an interactive, standalone HTML Visual Inspection
Dashboard from boundary CSV samples (e.g. boundary_samples_150.csv).

Key Features:
1. Visual Cards for each sampled boundary (Shot i -> Shot i+1).
2. Contact Sheet Layout:
   [3 Before Frames (Shots i-2, i-1, i)]  ||  ⚡ BOUNDARY  ||  [3 After Frames (Shots i+1, i+2, i+3)]
3. Resolves keyframe IDs to local images (or base64/SVG placeholders).
4. Highlights central boundary line with color-coded evidence indicators.
5. Interactive UI with Dark Mode, Glassmorphism, Image Lightbox Modal.
6. Client-side JS filtering (All / is_boundary=True / is_boundary=False) and sorting (Score Desc/Asc).
7. Interactive Manual Labeling (Correct / False / Ambiguous) with exportable JSON/CSV.
8. Fixes inconsistency between legacy is_boundary and calibrated boundary_score.

Usage:
  python scripts/generate_boundary_html_audit.py \
    --input-csv boundary_samples_150.csv \
    --shots artifacts/event_graph/features/shot_features.parquet \
    --output artifacts/event_graph/validation/boundary_visual_audit.html
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("boundary-html-audit")


def create_svg_keyframe_placeholder(video_id: str, shot_id: Any, kf_name: str, bg_color: str = "#1e293b") -> str:
    """Generate an inline Data URI SVG placeholder for missing keyframe images."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="240" height="135" viewBox="0 0 240 135">
      <rect width="240" height="135" fill="{bg_color}" rx="8"/>
      <rect x="2" y="2" width="236" height="131" fill="none" stroke="#475569" stroke-width="1.5" stroke-dasharray="4,4" rx="6"/>
      <text x="120" y="45" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="12" font-weight="bold" fill="#94a3b8" text-anchor="middle">{video_id}</text>
      <text x="120" y="70" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="15" font-weight="extrabold" fill="#f8fafc" text-anchor="middle">Shot {shot_id}</text>
      <text x="120" y="95" font-family="monospace" font-size="10" fill="#64748b" text-anchor="middle">KF: {kf_name[:22]}</text>
    </svg>"""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def resolve_image_to_data_uri(img_path: Path) -> Optional[str]:
    """Convert a local image file to base64 Data URI."""
    if not img_path.is_file():
        return None
    try:
        ext = img_path.suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.debug("Failed to read image %s: %s", img_path, e)
        return None


def resolve_keyframe_src(video_id: str, shot_id: Any, kf_id: str, keyframe_dirs: List[Path]) -> str:
    """Find keyframe image across search paths, convert to Data URI or return SVG fallback."""
    if not kf_id or str(kf_id).strip() == "" or kf_id == "None":
        return create_svg_keyframe_placeholder(video_id, shot_id, f"Shot_{shot_id}")

    clean_kf = str(kf_id).strip()
    candidates = []

    for kdir in keyframe_dirs:
        # Standard layout: keyframe_dir/video_id/kf_id.jpg
        candidates.append(kdir / video_id / f"{clean_kf}.jpg")
        candidates.append(kdir / video_id / f"{clean_kf}.png")
        candidates.append(kdir / f"{clean_kf}.jpg")
        candidates.append(kdir / f"{clean_kf}.png")
        if clean_kf.isdigit():
            candidates.append(kdir / video_id / f"{int(clean_kf):05d}.jpg")
            candidates.append(kdir / video_id / f"{int(clean_kf):06d}.jpg")

    for cand in candidates:
        data_uri = resolve_image_to_data_uri(cand)
        if data_uri:
            return data_uri

    # Fallback to SVG placeholder if image file not on disk
    return create_svg_keyframe_placeholder(video_id, shot_id, clean_kf)


def get_context_shots_for_boundary(
    video_id: str,
    shot_i: int,
    shot_next: int,
    df_shots: pd.DataFrame,
    keyframe_dirs: List[Path],
) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve 3 shots before (shot_i-2, shot_i-1, shot_i) and 3 shots after (shot_next, shot_next+1, shot_next+2)."""
    if df_shots.empty or "video_id" not in df_shots.columns:
        # Return fallback items
        before_items = [
            {"shot_id": shot_i - 2, "kf_id": f"{shot_i-2}", "src": create_svg_keyframe_placeholder(video_id, shot_i - 2, f"Shot_{shot_i-2}", "#0f172a")},
            {"shot_id": shot_i - 1, "kf_id": f"{shot_i-1}", "src": create_svg_keyframe_placeholder(video_id, shot_i - 1, f"Shot_{shot_i-1}", "#1e293b")},
            {"shot_id": shot_i,     "kf_id": f"Shot_{shot_i}", "src": create_svg_keyframe_placeholder(video_id, shot_i, f"Shot_{shot_i}", "#1e3a8a")},
        ]
        after_items = [
            {"shot_id": shot_next,     "kf_id": f"Shot_{shot_next}", "src": create_svg_keyframe_placeholder(video_id, shot_next, f"Shot_{shot_next}", "#831843")},
            {"shot_id": shot_next + 1, "kf_id": f"{shot_next+1}", "src": create_svg_keyframe_placeholder(video_id, shot_next + 1, f"Shot_{shot_next+1}", "#1e293b")},
            {"shot_id": shot_next + 2, "kf_id": f"{shot_next+2}", "src": create_svg_keyframe_placeholder(video_id, shot_next + 2, f"Shot_{shot_next+2}", "#0f172a")},
        ]
        return {"before": before_items, "after": after_items}

    v_shots = df_shots[df_shots["video_id"] == video_id].sort_values("start_sec").reset_index(drop=True)
    if v_shots.empty:
        return get_context_shots_for_boundary(video_id, shot_i, shot_next, pd.DataFrame(), keyframe_dirs)

    # Find index of shot_i
    idx_i_list = v_shots.index[v_shots["shot_id"] == shot_i].tolist()
    idx_i = idx_i_list[0] if idx_i_list else 0

    # Extract 3 before: idx_i-2, idx_i-1, idx_i
    before_indices = [max(0, idx_i - 2), max(0, idx_i - 1), idx_i]
    # Unique indices while preserving order
    seen = set()
    before_indices = [x for x in before_indices if not (x in seen or seen.add(x))]

    before_items = []
    for bi in before_indices:
        r = v_shots.iloc[bi]
        s_id = int(r["shot_id"])
        kf_id = str(r.get("representative_keyframe", f"Shot_{s_id}"))
        src = resolve_keyframe_src(video_id, s_id, kf_id, keyframe_dirs)
        before_items.append({"shot_id": s_id, "kf_id": kf_id, "src": src, "start_sec": float(r.get("start_sec", 0.0))})

    # Extract 3 after: idx_next, idx_next+1, idx_next+2
    idx_next_list = v_shots.index[v_shots["shot_id"] == shot_next].tolist()
    idx_next = idx_next_list[0] if idx_next_list else min(len(v_shots) - 1, idx_i + 1)

    after_indices = [idx_next, min(len(v_shots) - 1, idx_next + 1), min(len(v_shots) - 1, idx_next + 2)]
    seen_a = set()
    after_indices = [x for x in after_indices if not (x in seen_a or seen_a.add(x))]

    after_items = []
    for ai in after_indices:
        r = v_shots.iloc[ai]
        s_id = int(r["shot_id"])
        kf_id = str(r.get("representative_keyframe", f"Shot_{s_id}"))
        src = resolve_keyframe_src(video_id, s_id, kf_id, keyframe_dirs)
        after_items.append({"shot_id": s_id, "kf_id": kf_id, "src": src, "start_sec": float(r.get("start_sec", 0.0))})

    return {"before": before_items, "after": after_items}


def generate_standalone_html(
    df_samples: pd.DataFrame,
    df_shots: pd.DataFrame,
    keyframe_dirs: List[Path],
    threshold: float = 0.70,
) -> str:
    """Build modern dark-mode HTML dashboard with Contact Sheets & Interactive Annotation."""
    cards_data = []

    for idx, row in df_samples.iterrows():
        vid = str(row["video_id"])
        shot_i = int(row["shot_i"])
        shot_next = int(row["shot_next"])

        b_score = float(row.get("boundary_score", 0.0))
        f_sim = float(row.get("fused_similarity", 0.0))

        # Dynamically recalculate is_boundary consistency check!
        is_b_consistent = bool(b_score > threshold)
        is_b_csv = bool(row.get("is_boundary", is_b_consistent))

        v_sim = float(row.get("visual_similarity", 0.0))
        s_sim = float(row.get("semantic_similarity", 0.0))
        v_evd = float(row.get("visual_boundary_evidence", 0.0))
        s_evd = float(row.get("semantic_boundary_evidence", 0.0))

        # Retrieve 3 before + 3 after context shots
        context = get_context_shots_for_boundary(vid, shot_i, shot_next, df_shots, keyframe_dirs)

        cards_data.append({
            "id": idx + 1,
            "video_id": vid,
            "shot_i": shot_i,
            "shot_next": shot_next,
            "boundary_score": b_score,
            "fused_similarity": f_sim,
            "is_boundary_calibrated": is_b_consistent,
            "is_boundary_csv": is_b_csv,
            "visual_similarity": v_sim,
            "semantic_similarity": s_sim,
            "visual_evidence": v_evd,
            "semantic_evidence": s_evd,
            "shots_before": context["before"],
            "shots_after": context["after"],
        })

    cards_json = json.dumps(cards_data, ensure_ascii=False)

    html_template = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stage 3C Event Boundary Visual Audit Dashboard — AIC2026</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-dark: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --accent-cyan: #06b6d4;
      --accent-pink: #ec4899;
      --accent-green: #10b981;
      --accent-yellow: #f59e0b;
      --boundary-glow: rgba(236, 72, 153, 0.4);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: 'Inter', sans-serif;
      padding: 24px;
      line-height: 1.5;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(31, 41, 55, 0.9) 100%);
      backdrop-filter: blur(12px);
      padding: 20px 28px;
      border-radius: 16px;
      border: 1px solid var(--card-border);
      margin-bottom: 24px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }}
    .header h1 {{
      font-size: 1.5rem;
      font-weight: 800;
      background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .header p {{ font-size: 0.875rem; color: var(--text-muted); margin-top: 4px; }}
    
    .controls-bar {{
      display: flex;
      gap: 16px;
      align-items: center;
      background-color: var(--card-bg);
      padding: 16px 24px;
      border-radius: 12px;
      border: 1px solid var(--card-border);
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}
    .control-group {{ display: flex; align-items: center; gap: 8px; font-size: 0.875rem; color: var(--text-muted); }}
    select, input, button {{
      background-color: #1f2937;
      color: #f9fafb;
      border: 1px solid #374151;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 0.875rem;
      outline: none;
      transition: all 0.2s;
    }}
    select:hover, button:hover {{ border-color: var(--accent-cyan); cursor: pointer; }}
    
    .stat-badge {{
      background: rgba(6, 182, 212, 0.15);
      color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.3);
      padding: 4px 10px;
      border-radius: 20px;
      font-weight: 600;
      font-size: 0.8rem;
    }}

    .cards-grid {{
      display: flex;
      flex-direction: column;
      gap: 28px;
    }}
    
    .card {{
      background-color: var(--card-bg);
      border-radius: 16px;
      border: 1px solid var(--card-border);
      padding: 20px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      transition: transform 0.2s, border-color 0.2s;
    }}
    .card:hover {{ border-color: #374151; transform: translateY(-2px); }}
    
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1f2937;
      padding-bottom: 12px;
      margin-bottom: 16px;
    }}
    .video-title {{ font-size: 1.1rem; font-weight: 700; color: #f3f4f6; }}
    .shot-range {{ font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #38bdf8; }}
    
    .metrics-pills {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .pill {{
      font-size: 0.75rem;
      font-family: 'JetBrains Mono', monospace;
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 600;
    }}
    .pill-score {{ background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.4); font-size: 0.85rem; }}
    .pill-sim {{ background: rgba(55, 65, 81, 0.5); color: #d1d5db; border: 1px solid #4b5563; }}
    .pill-is-b {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .pill-not-b {{ background: rgba(107, 114, 128, 0.2); color: #9ca3af; border: 1px solid rgba(107, 114, 128, 0.4); }}

    /* Contact Sheet Layout */
    .contact-sheet {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      position: relative;
      margin: 16px 0;
    }}
    .frames-group {{
      display: flex;
      gap: 8px;
    }}
    .frame-box {{
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 170px;
      background: #0f172a;
      border-radius: 10px;
      padding: 6px;
      border: 1px solid #1e293b;
    }}
    .frame-box.boundary-left {{ border: 2px solid #38bdf8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.3); }}
    .frame-box.boundary-right {{ border: 2px solid #ec4899; box-shadow: 0 0 12px rgba(236, 72, 153, 0.3); }}
    
    .frame-box img {{
      width: 100%;
      height: 96px;
      object-fit: cover;
      border-radius: 6px;
      cursor: zoom-in;
      transition: transform 0.2s;
    }}
    .frame-box img:hover {{ transform: scale(1.04); }}
    .frame-label {{
      font-size: 0.72rem;
      font-family: 'JetBrains Mono', monospace;
      color: #94a3b8;
      margin-top: 4px;
    }}

    /* Boundary Divider Wall */
    .boundary-divider {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 0 12px;
      min-width: 110px;
    }}
    .divider-line {{
      width: 3px;
      height: 90px;
      background: linear-gradient(to bottom, #ec4899, #818cf8);
      border-radius: 2px;
      box-shadow: 0 0 10px var(--boundary-glow);
    }}
    .divider-badge {{
      background: #ec4899;
      color: white;
      font-size: 0.7rem;
      font-weight: 800;
      padding: 4px 8px;
      border-radius: 12px;
      margin: 6px 0;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    /* Annotation Footer */
    .annotation-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px dashed #1f2937;
    }}
    .btn-anno {{
      padding: 6px 14px;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: 8px;
      border: 1px solid transparent;
      cursor: pointer;
    }}
    .btn-correct {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.4); }}
    .btn-correct.active {{ background: #10b981; color: white; }}
    .btn-false {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: rgba(239, 68, 68, 0.4); }}
    .btn-false.active {{ background: #ef4444; color: white; }}
    .btn-ambiguous {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.4); }}
    .btn-ambiguous.active {{ background: #f59e0b; color: white; }}

    /* Lightbox Modal */
    .modal {{
      display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%;
      background-color: rgba(0,0,0,0.9); backdrop-filter: blur(8px);
      align-items: center; justify-content: center;
    }}
    .modal img {{ max-width: 90%; max-height: 90%; border-radius: 12px; border: 2px solid #374151; }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>🎬 Stage 3C Event Boundary Visual Audit Dashboard</h1>
      <p>Systematic inspection of candidate boundaries with 3-shot temporal context (AIC2026 Multimodal EventGraph)</p>
    </div>
    <div class="stat-badge" id="total-badge">Total Samples: 0</div>
  </div>

  <div class="controls-bar">
    <div class="control-group">
      <label>Filter is_boundary:</label>
      <select id="filter-is-b" onchange="applyFilters()">
        <option value="all">All (True & False)</option>
        <option value="true">is_boundary = True</option>
        <option value="false">is_boundary = False</option>
      </select>
    </div>
    <div class="control-group">
      <label>Sort By:</label>
      <select id="sort-by" onchange="applyFilters()">
        <option value="score-desc">Boundary Score (High → Low)</option>
        <option value="score-asc">Boundary Score (Low → High)</option>
        <option value="video">Video ID</option>
      </select>
    </div>
    <div class="control-group" style="margin-left: auto;">
      <button onclick="exportAnnotationsCSV()" style="background: #10b981; color: white; border: none; font-weight: 600;">
        📥 Export Annotations CSV
      </button>
    </div>
  </div>

  <div class="cards-grid" id="cards-container"></div>

  <!-- Lightbox Modal -->
  <div id="modal" class="modal" onclick="closeModal()">
    <img id="modal-img" src="" alt="Zoomed Keyframe">
  </div>

  <script>
    const RAW_CARDS = {cards_json};
    let userAnnotations = {{}}; // cardId -> label

    function renderCards(cards) {{
      const container = document.getElementById('cards-container');
      document.getElementById('total-badge').innerText = `Displaying: ${{cards.length}} Samples`;
      container.innerHTML = '';

      if (cards.length === 0) {{
        container.innerHTML = '<div style="text-align:center; padding: 40px; color: #9ca3af;">No boundary samples match the filter criteria.</div>';
        return;
      }}

      cards.forEach(card => {{
        const cardEl = document.createElement('div');
        cardEl.className = 'card';
        
        const labelState = userAnnotations[card.id] || '';

        // Context before HTML
        const beforeHtml = card.shots_before.map((s, i) => {{
          const isLastBefore = i === card.shots_before.length - 1;
          const borderClass = isLastBefore ? 'boundary-left' : '';
          return `
            <div class="frame-box ${{borderClass}}">
              <img src="${{s.src}}" onclick="zoomImage('${{s.src}}')" alt="Shot ${{s.shot_id}}">
              <div class="frame-label">Shot ${{s.shot_id}} (Before)</div>
            </div>
          `;
        }}).join('');

        // Context after HTML
        const afterHtml = card.shots_after.map((s, i) => {{
          const isFirstAfter = i === 0;
          const borderClass = isFirstAfter ? 'boundary-right' : '';
          return `
            <div class="frame-box ${{borderClass}}">
              <img src="${{s.src}}" onclick="zoomImage('${{s.src}}')" alt="Shot ${{s.shot_id}}">
              <div class="frame-label">Shot ${{s.shot_id}} (After)</div>
            </div>
          `;
        }}).join('');

        cardEl.innerHTML = `
          <div class="card-header">
            <div>
              <span class="video-title">${{card.video_id}}</span>
              <span class="shot-range" style="margin-left: 12px;">Shot ${{card.shot_i}} &rarr; Shot ${{card.shot_next}}</span>
            </div>
            <div class="metrics-pills">
              <span class="pill pill-score">Boundary Score: ${{card.boundary_score.toFixed(4)}}</span>
              <span class="pill pill-sim">Fused Sim: ${{card.fused_similarity.toFixed(4)}}</span>
              <span class="pill pill-sim">Vis: ${{card.visual_similarity.toFixed(4)}} (Evd: ${{card.visual_evidence.toFixed(2)}})</span>
              <span class="pill pill-sim">Sem: ${{card.semantic_similarity.toFixed(4)}} (Evd: ${{card.semantic_evidence.toFixed(2)}})</span>
              <span class="pill ${{card.is_boundary_calibrated ? 'pill-is-b' : 'pill-not-b'}}">
                ${{card.is_boundary_calibrated ? 'is_boundary: True' : 'is_boundary: False'}}
              </span>
            </div>
          </div>

          <div class="contact-sheet">
            <div class="frames-group">${{beforeHtml}}</div>
            
            <div class="boundary-divider">
              <div class="divider-line"></div>
              <div class="divider-badge">⚡ BOUNDARY</div>
              <div class="divider-line"></div>
            </div>
            
            <div class="frames-group">${{afterHtml}}</div>
          </div>

          <div class="annotation-bar">
            <div style="font-size: 0.85rem; color: #94a3b8;">
              Manual Inspection Audit Label:
            </div>
            <div style="display: flex; gap: 8px;">
              <button class="btn-anno btn-correct ${{labelState === 'correct' ? 'active' : ''}}" onclick="annotateCard(${{card.id}}, 'correct')">
                ✓ Correct Boundary
              </button>
              <button class="btn-anno btn-false ${{labelState === 'false' ? 'active' : ''}}" onclick="annotateCard(${{card.id}}, 'false')">
                ✕ False Boundary
              </button>
              <button class="btn-anno btn-ambiguous ${{labelState === 'ambiguous' ? 'active' : ''}}" onclick="annotateCard(${{card.id}}, 'ambiguous')">
                ? Ambiguous
              </button>
            </div>
          </div>
        `;
        container.appendChild(cardEl);
      }});
    }}

    function applyFilters() {{
      const filterIsB = document.getElementById('filter-is-b').value;
      const sortBy = document.getElementById('sort-by').value;

      let filtered = RAW_CARDS.filter(c => {{
        if (filterIsB === 'true') return c.is_boundary_calibrated === true;
        if (filterIsB === 'false') return c.is_boundary_calibrated === false;
        return true;
      }});

      if (sortBy === 'score-desc') {{
        filtered.sort((a, b) => b.boundary_score - a.boundary_score);
      }} else if (sortBy === 'score-asc') {{
        filtered.sort((a, b) => a.boundary_score - b.boundary_score);
      }} else if (sortBy === 'video') {{
        filtered.sort((a, b) => a.video_id.localeCompare(b.video_id));
      }}

      renderCards(filtered);
    }}

    function annotateCard(cardId, label) {{
      userAnnotations[cardId] = label;
      applyFilters();
    }}

    function zoomImage(src) {{
      document.getElementById('modal-img').src = src;
      document.getElementById('modal').style.display = 'flex';
    }}

    function closeModal() {{
      document.getElementById('modal').style.display = 'none';
    }}

    function exportAnnotationsCSV() {{
      const rows = [['card_id', 'video_id', 'shot_i', 'shot_next', 'boundary_score', 'user_label']];
      RAW_CARDS.forEach(c => {{
        const label = userAnnotations[c.id] || 'unannotated';
        rows.push([c.id, c.video_id, c.shot_i, c.shot_next, c.boundary_score, label]);
      }});
      
      let csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\\n");
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "manual_boundary_annotations.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }}

    // Initial render
    applyFilters();
  </script>
</body>
</html>"""
    return html_template


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3C Standalone HTML Visual Audit Generator"
    )
    parser.add_argument(
        "--input-csv",
        default="boundary_samples_150.csv",
        help="Input boundary CSV file (e.g. boundary_samples_150.csv)",
    )
    parser.add_argument(
        "--shots",
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "features" / "shot_features.parquet"),
        help="Path to shot_features.parquet",
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "artifacts" / "event_graph" / "validation" / "boundary_visual_audit.html"
        ),
        help="Path for output HTML file",
    )
    parser.add_argument(
        "--keyframe-dirs",
        nargs="*",
        default=[],
        help="Directories containing keyframe images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Calibrated boundary score threshold (default: 0.70)",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        candidates = [
            PROJECT_ROOT / args.input_csv,
            PROJECT_ROOT / "artifacts" / "event_graph" / "validation" / "boundary_samples_for_manual_check.csv",
            Path("artifacts/event_graph/validation/boundary_samples_for_manual_check.csv"),
            Path("/kaggle/working/boundary_samples.csv"),
            Path("/kaggle/working/boundary_samples_150.csv"),
            Path("/kaggle/working/AIC2026_TeamPTK_SGU/artifacts/event_graph/validation/boundary_samples_for_manual_check.csv"),
            Path("/kaggle/working/AIC2026_TeamPTK_SGU/boundary_samples_150.csv"),
            Path("../boundary_samples_150.csv"),
        ]
        for cand in candidates:
            if cand.exists():
                input_csv = cand
                logger.info("Auto-resolved input CSV path to: %s", input_csv)
                break

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input boundary CSV not found at '{args.input_csv}'. "
            f"Please pass --input-csv path/to/file.csv or generate samples first using interactive_boundary_annotator.py!"
        )

    logger.info("==================================================================")
    logger.info("🖼️ STAGE 3C HTML VISUAL AUDIT DASHBOARD GENERATOR")
    logger.info("==================================================================")
    logger.info("Loading input CSV from: %s", input_csv)
    df_samples = pd.read_csv(input_csv)

    df_shots = pd.DataFrame()
    shots_path = Path(args.shots)
    if shots_path.exists():
        logger.info("Loading shot features for context from: %s", shots_path)
        df_shots = pd.read_parquet(shots_path)
    else:
        logger.warning("Shot features parquet not found at %s. Context will use fallback IDs.", shots_path)

    keyframe_dirs = [Path(d) for d in args.keyframe_dirs]
    # Common default keyframe paths
    keyframe_dirs.extend([
        PROJECT_ROOT / "artifacts" / "keyframe_btc_full",
        PROJECT_ROOT / "keyframes",
        PROJECT_ROOT / "media",
    ])

    output_html_path = Path(args.output)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating standalone HTML dashboard with 3-shot context sheets...")
    html_content = generate_standalone_html(
        df_samples=df_samples,
        df_shots=df_shots,
        keyframe_dirs=keyframe_dirs,
        threshold=args.threshold,
    )

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n" + "=" * 80)
    print("🎉 STAGE 3C HTML VISUAL AUDIT GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f" • Output File Path  : {output_html_path}")
    print(f" • Total Boundaries  : {len(df_samples):,}")
    print(f" • Score Threshold   : {args.threshold:.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
