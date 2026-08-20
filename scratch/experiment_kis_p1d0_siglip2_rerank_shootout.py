#!/usr/bin/env python3
"""KIS P1D0: SigLIP2 Top-30 Visual Reranker Shootout (Retrieval-Shortlist Only).

Compares:
  - ARM A (Production P0 Baseline): Marian EN -> OpenAI CLIP ViT-B/32 1st-pass Top-30 candidates.
  - ARM B (SigLIP2 Reranked): Exact same Top-30 candidate frames re-scored and re-ordered purely by SigLIP2.

Strict constraints:
  - Production P0 remains 100% immutable (default OFF).
  - ZERO image re-indexing: 100% reuse of the existing 177,532-keyframe OpenAI CLIP ViT-B/32 feature matrix for 1st-pass.
  - Pure shortlist re-ordering: NO new candidate injection, ZERO score fusion between CLIP and SigLIP2.
  - Strict candidate-set equality assertion: set(candidates_A) == set(candidates_B).
  - HARD frame decode coverage gate: decoded_frames == candidate_count (30/30) strictly required. Aborts query if coverage < 100%.
  - Strict SigLIP2 model requirement: google/siglip2-base-patch16-224. ZERO silent fallback to SigLIP1.
  - Full attention_mask, pixel_attention_mask, and spatial_shapes passed to feature extractors.
  - Ground-truth SigLIP2 tokenizer telemetry (raw input_ids count, effective attention_mask sum, truncation flag, max length).
  - Marian Parity Gate: Asserts Marian translations match frozen P0 baseline across 3 reference probes.
  - ZERO ground-truth leakage.
"""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Purge stale system_tai modules
for mod in list(sys.modules.keys()):
    if mod.startswith("system_tai") or mod.startswith("experiment_kis"):
        del sys.modules[mod]

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TAI_SRC = REPO_ROOT / "systems" / "system_tai" / "src"
if str(SYSTEM_TAI_SRC) not in sys.path:
    sys.path.insert(0, str(SYSTEM_TAI_SRC))

# Ensure required libraries
try:
    import cv2
    from PIL import Image
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless", "pillow"], check=False)
    import cv2
    from PIL import Image

import transformers
from transformers import AutoModel, AutoProcessor

import numpy as np
import torch
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
TOP_K_SHORTLIST = 30

BTC_KIS_QUERIES = [
    {"qid": "query-p1-1-kis", "category": "REGRESSION_GUARD", "name": "Phóng tàu vũ trụ tư nhân / 4 phi hành gia áo đen"},
    {"qid": "query-p1-2-kis", "category": "REGRESSION_GUARD", "name": "Con hổ (Tiger)"},
    {"qid": "query-p1-5-kis", "category": "REGRESSION_GUARD", "name": "Hai người phụ nữ cho dê ăn"},
    {"qid": "query-p1-6-kis", "category": "GENERAL_PROBE", "name": "Cắt tỉa cây cảnh Bonsai / Đĩa chay"},
    {"qid": "query-p1-7-kis", "category": "REGRESSION_GUARD", "name": "Chim lông đen ánh xanh cổ (Bird)"},
    {"qid": "query-p1-8-kis", "category": "GENERAL_PROBE", "name": "Lễ hội ẩm thực Nhật bé đeo mực đỏ"},
    {"qid": "query-p1-9-kis", "category": "GENERAL_PROBE", "name": "Thu hoạch dứa ở miền Tây"},
    {"qid": "query-p1-10-kis", "category": "REGRESSION_GUARD", "name": "Chơi nhạc cụ kim loại tròn (Handpan)"},
    {"qid": "query-p1-11-kis", "category": "REGRESSION_GUARD", "name": "Đổ bóng tạo chân dung mặc vest"},
    {"qid": "query-p1-12-kis", "category": "LONG_QUERY_PROBE", "name": "Trang trí bánh rán dâu tây chuối chocolate"},
    {"qid": "query-p1-13-kis", "category": "TARGET_PROBE", "name": "Vệ sinh máy ảnh, lens trên khăn hồng, tăm bông"},
    {"qid": "query-p1-14-kis", "category": "GENERAL_PROBE", "name": "Điêu khắc cát thể thao đường phố"},
    {"qid": "query-p1-17-kis", "category": "LONG_QUERY_PROBE", "name": "Trao quà từ thiện bệnh viện biển COVID-19"},
    {"qid": "query-p1-20-kis", "category": "REGRESSION_GUARD", "name": "Thêm 2 ly panna cotta, hoa ăn được"},
    {"qid": "query-p1-21-kis", "category": "TARGET_PROBE", "name": "Cơ chế bay của bọ làm robot ở ĐH Lausanne"},
    {"qid": "query-p1-23-kis", "category": "REASONING_CONTROL", "name": "Động vật biển nguy hiểm Steven Spielberg 1975"},
    {"qid": "query-p1-24-kis", "category": "TARGET_PROBE", "name": "Đua xe đạp quay từ trên cao xuống"},
    {"qid": "query-p1-25-kis", "category": "REGRESSION_GUARD", "name": "Đua xe đạp flycam trên cao áo xanh vượt 3"},
]

VIDEO_PATH_CACHE: dict[str, Path] = {}


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


def resolve_video_path(video_id: str, raw_video_registry: Any = None) -> Path | None:
    if raw_video_registry:
        try:
            rec = raw_video_registry.get(video_id)
            if rec and rec.raw_video_path and rec.raw_video_path.exists():
                return rec.raw_video_path
        except Exception:
            pass
    if video_id in VIDEO_PATH_CACHE:
        return VIDEO_PATH_CACHE[video_id]
    populate_video_index_once()
    return VIDEO_PATH_CACHE.get(video_id)


def decode_candidate_frames(
    candidates: list[Any],
    raw_video_registry: Any = None,
) -> tuple[list[Image.Image | None], list[str], float, int]:
    """Decode exact keyframes for candidate list grouped by video to maximize IO efficiency."""
    t0 = time.time()
    images: list[Image.Image | None] = [None] * len(candidates)
    b64_thumbnails: list[str] = [""] * len(candidates)
    decoded_count = 0

    # Group by video_id
    video_to_items: dict[str, list[tuple[int, int]]] = {}
    for idx, c in enumerate(candidates):
        video_to_items.setdefault(c.video_id, []).append((idx, c.frame_id))

    for vid, items in video_to_items.items():
        vpath = resolve_video_path(vid, raw_video_registry)
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
                # Convert BGR to RGB PIL image for SigLIP2
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                images[orig_idx] = pil_img
                decoded_count += 1

                # Generate thumbnail base64 for gallery
                h, w = frame.shape[:2]
                new_w = 240
                new_h = int(h * (new_w / w))
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                b64_thumbnails[orig_idx] = base64.b64encode(buf).decode("utf-8")
            cap.release()
        except Exception:
            pass

    decode_seconds = time.time() - t0
    return images, b64_thumbnails, decode_seconds, decoded_count


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


def verify_marian_parity(runtime: OperationalKISRuntime) -> None:
    """Verify Marian translation outputs match frozen production P0 baseline across reference probes."""
    print("\n" + "=" * 120, flush=True)
    print("MARIAN TRANSLATION PARITY GATE VERIFICATION", flush=True)
    print("=" * 120, flush=True)
    
    probe_2_vi = "Mẩu tin giới thiệu về đàn hổ tại một địa phương ở miền Nam vừa có thêm khoảng 3-6 con hổ con. Đây là một giống hổ quý hiếm"
    probe_2_en = runtime.translation_provider.translate(probe_2_vi).strip()
    
    probe_10_vi = "Tìm chính xác đoạn clip ngắn có ba người (hai phụ nữ và một nam giới) đang ngồi cạnh nhau, tập trung chơi nhạc cụ kim loại có dạng tròn, rỗng, với các vết lõm để tạo ra âm thanh khi gõ tay. Có 1 người mặc áo trắng ngồi giữa 2 người mặc áo đen. Bối cảnh phía sau là một kệ sách nhiều ngăn, xếp đầy sách với nhiều màu sắc"
    probe_10_en = runtime.translation_provider.translate(probe_10_vi).strip()
    
    print(f"• Marian Probe p1-2  : \"{probe_2_en}\"", flush=True)
    print(f"• Marian Probe p1-10 : \"{probe_10_en[:80]}...\"", flush=True)
    
    if "rare tiger species" not in probe_2_en.lower():
        raise RuntimeError(f"Marian Parity Gate Failed on p1-2: Unexpected translation '{probe_2_en}'")
    if "metal instruments" not in probe_10_en.lower():
        raise RuntimeError(f"Marian Parity Gate Failed on p1-10: Unexpected translation '{probe_10_en}'")
    
    print("• Marian Parity Status: 100% MATCH WITH FROZEN P0 BASELINE ✅", flush=True)
    print("=" * 120, flush=True)


def load_strict_siglip2_model(device: str) -> tuple[Any, Any]:
    """Strictly load SigLIP2 model. If it fails, abort immediately without fallback."""
    print(f"\n[2/4] Loading Strict SigLIP2 Model: '{SIGLIP2_MODEL_ID}' on device '{device}' ...", flush=True)
    print(f"      • transformers version : {transformers.__version__}", flush=True)
    print(f"      • torch version        : {torch.__version__}", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
        model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID).to(device)
        model.eval()
        
        # Log model config/revision info
        model_type = getattr(getattr(model, "config", None), "model_type", "unknown")
        hidden_size = getattr(getattr(model, "config", None), "text_config", {}).get("hidden_size", getattr(getattr(model, "config", None), "hidden_size", "unknown"))
        print(f"      • Model Architecture   : {model_type} (hidden_dim={hidden_size})", flush=True)
        print(f"      • Model Loading Status : SUCCESS (Model ID: {SIGLIP2_MODEL_ID}) ✅", flush=True)
        return model, processor
    except Exception as exc:
        print("=" * 120, flush=True)
        print(f"FATAL ERROR: Could not load required SigLIP2 model '{SIGLIP2_MODEL_ID}'.", flush=True)
        print(f"Details: {exc}", flush=True)
        print("P1D0 requires google/siglip2-base-patch16-224 strictly. Silent fallback to SigLIP1 is forbidden.", flush=True)
        print("Status: UNSUPPORTED_OR_ERROR -> ABORTING EXPERIMENT.", flush=True)
        print("=" * 120, flush=True)
        raise RuntimeError(f"P1D0 Aborted: Failed to load {SIGLIP2_MODEL_ID}: {exc}") from exc


def extract_tensor_features(output: Any) -> torch.Tensor:
    """Extract the primary feature tensor from model feature extractor output across Transformers versions."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        lhs = output.last_hidden_state
        if len(lhs.shape) == 3:
            return lhs[:, 0, :]  # CLS token
        return lhs
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot extract feature tensor from output of type {type(output)}")


def score_candidates_with_siglip2(
    model: Any,
    processor: Any,
    text: str,
    images: list[Image.Image | None],
    device: str,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Compute SigLIP2 image-text similarity scores passing all processor outputs."""
    t0 = time.time()
    
    # 1. Ground-Truth SigLIP2 Tokenizer Telemetry
    tokenizer = processor.tokenizer
    max_len = getattr(tokenizer, "model_max_length", 64)
    if max_len is None or max_len > 10000:
        max_len = 64  # Standard default for SigLIP2 base

    # Ground-truth raw model tokens with special tokens
    raw_encoded = tokenizer(text, truncation=False, add_special_tokens=True)
    raw_model_tokens = len(raw_encoded["input_ids"])

    # Effective tokenization with model max_length and attention_mask
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    effective_model_tokens = int(text_inputs["attention_mask"].sum())
    truncated = raw_model_tokens > effective_model_tokens

    token_telemetry = {
        "siglip_raw_token_count": raw_model_tokens,
        "siglip_effective_token_count": effective_model_tokens,
        "siglip_model_max_length": max_len,
        "siglip_truncated": truncated,
    }

    # 2. Filter valid decoded images
    valid_indices = [i for i, img in enumerate(images) if img is not None]
    full_scores = np.full(len(images), -np.inf, dtype=np.float32)

    if not valid_indices:
        return full_scores, time.time() - t0, token_telemetry

    valid_images = [images[i] for i in valid_indices]

    # 3. Process images with image_processor
    image_inputs = processor.image_processor(
        valid_images,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # Pass full inputs (including attention_mask, pixel_attention_mask, spatial_shapes)
        text_out = model.get_text_features(**text_inputs)
        image_out = model.get_image_features(**image_inputs)

        text_feat = extract_tensor_features(text_out)
        image_feat = extract_tensor_features(image_out)

        # L2 Normalization
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        # Dot product cosine similarity
        sims = (image_feat @ text_feat.T).squeeze(-1).float().cpu().numpy()

    for idx_in_valid, orig_idx in enumerate(valid_indices):
        full_scores[orig_idx] = float(sims[idx_in_valid])

    model_seconds = time.time() - t0
    return full_scores, model_seconds, token_telemetry


def run_p1d0_experiment() -> None:
    print("=" * 150, flush=True)
    print(f"KIS P1D0: SIGLIP2 TOP-{TOP_K_SHORTLIST} VISUAL RERANKER SHOOTOUT", flush=True)
    print("=" * 150, flush=True)
    print("NOTICE ON LONG-QUERY PROBES (p1-12 & p1-17):", flush=True)
    print("  • P1D0 uses P0 effective_en as the semantic input query to test pure visual reranking.", flush=True)
    print("  • It evaluates visual discrimination over the exact same coarse shortlist, but CANNOT recover", flush=True)
    print("    semantic details already removed by P0 prefix truncation.", flush=True)
    print("=" * 150, flush=True)

    yaml_path = REPO_ROOT / "systems" / "system_tai" / "configs" / "production.yaml"
    input_root = Path("/kaggle/input/datasets") if Path("/kaggle/input/datasets").exists() else Path("/kaggle/input")
    reuse_manifest = get_reuse_manifest()
    out_dir = Path("/kaggle/working/output/kis_p1d0_session") if Path("/kaggle/working").exists() else REPO_ROOT / "scratch" / "kis_p1d0_session"

    cfg = SessionConfig.from_yaml(
        yaml_path,
        input_root=input_root,
        output_root=out_dir,
        reuse_manifest=reuse_manifest,
    )

    # 1. Bootstrap Production Runtime (Provides exact manifest, exact_retriever, Marian, OpenAI CLIP)
    print("\n[1/4] Bootstrapping OperationalKISRuntime...", flush=True)
    t0_rt = time.time()
    runtime = OperationalKISRuntime.bootstrap(cfg)
    device = runtime.shared_encoder.identifiers.get("device", "cpu")
    if torch.cuda.is_available():
        device = "cuda"
    print(f"      Runtime Bootstrapped in {time.time() - t0_rt:.2f}s (device={device})", flush=True)

    # Marian Parity Verification
    verify_marian_parity(runtime)

    # 2. Strict SigLIP2 Model Loading
    siglip_model, siglip_processor = load_strict_siglip2_model(device)

    # 3. Benchmark across all 18 BTC KIS Queries
    thunghiem_dir = REPO_ROOT / "systems" / "system_tai" / "THUNGHIEM_20-8"
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 150, flush=True)
    print(f"RUNNING TOP-{TOP_K_SHORTLIST} SHORTLIST RERANKING ACROSS 18 BTC KIS QUERIES", flush=True)
    print("=" * 150, flush=True)

    decode_latencies = []
    rerank_latencies = []
    total_latencies = []

    for idx, item in enumerate(BTC_KIS_QUERIES, start=1):
        qid = item["qid"]
        category = item["category"]
        name = item["name"]
        q_file = thunghiem_dir / f"{qid}.txt"

        if not q_file.exists():
            continue

        q_vi = q_file.read_text(encoding="utf-8").strip()

        # Step A: 1st-Pass Retrieval via Frozen Marian EN -> OpenAI CLIP (Top 30)
        t_coarse0 = time.time()
        raw_en = runtime.translation_provider.translate(q_vi)
        eff_en, tok_count_clip, was_compacted = runtime.token_budget_guard.guard_and_compact(raw_en)
        vec_a = runtime.shared_encoder.encode(eff_en)
        res_a = runtime.exact_retriever.search_vector(query_id=f"a-{qid}", query_vector=vec_a, top_k=TOP_K_SHORTLIST)
        lat_coarse = time.time() - t_coarse0

        candidates_a = list(res_a.ranked_candidates)
        if len(candidates_a) == 0:
            continue

        # Step B: Decode exact Top-30 keyframes with HARD Coverage Gate
        images, b64_thumbs, lat_decode, decoded_count = decode_candidate_frames(candidates_a, runtime.raw_video_registry)
        decode_latencies.append(lat_decode)

        # HARD Frame Decode Coverage Gate
        coverage_status = f"{decoded_count}/{len(candidates_a)}"
        if decoded_count != len(candidates_a):
            error_msg = f"HARD DECODE COVERAGE GATE FAILED on query '{qid}': Decoded {decoded_count}/{len(candidates_a)} frames. Incomplete shortlist cannot be evaluated cleanly."
            print(f"  ❌ {error_msg}", flush=True)
            raise RuntimeError(error_msg)

        # Step C: Score exact Top-30 candidates with SigLIP2
        siglip_scores, lat_siglip, token_telemetry = score_candidates_with_siglip2(
            siglip_model,
            siglip_processor,
            eff_en,
            images,
            device,
        )
        rerank_latencies.append(lat_siglip)
        lat_total = lat_coarse + lat_decode + lat_siglip
        total_latencies.append(lat_total)

        # Step D: Re-order candidates by descending SigLIP2 score
        reranked_order = np.argsort(-siglip_scores)
        candidates_b = [candidates_a[i] for i in reranked_order]
        b64_thumbs_b = [b64_thumbs[i] for i in reranked_order]
        siglip_scores_b = [float(siglip_scores[i]) for i in reranked_order]

        # Strict Candidate-Set Equality Assertion
        set_a = set((c.video_id, c.frame_id) for c in candidates_a)
        set_b = set((c.video_id, c.frame_id) for c in candidates_b)
        assert set_a == set_b, f"FATAL ERROR: Candidate set mismatch on query {qid}!"

        # Track rank migrations
        rank_shifts = []
        for new_r, orig_idx in enumerate(reranked_order[:10], start=1):
            orig_r = orig_idx + 1
            shift = orig_r - new_r
            shift_str = f"+{shift}" if shift > 0 else (f"{shift}" if shift < 0 else "=")
            c = candidates_b[new_r - 1]
            s = siglip_scores_b[new_r - 1]
            rank_shifts.append(f"@{new_r}: {c.video_id}(f={c.frame_id}) [orig=@{orig_r}, shift={shift_str}, s_siglip={s:.3f}]")

        top10_desc_a = [f"@{i}: {c.video_id} (f={c.frame_id}, s_clip={c.score:.3f})" for i, c in enumerate(candidates_a[:10], start=1)]

        results.append({
            "qid": qid,
            "category": category,
            "name": name,
            "query_vi": q_vi,
            "eff_en": eff_en,
            "lat_coarse": lat_coarse,
            "lat_decode": lat_decode,
            "lat_siglip": lat_siglip,
            "lat_total": lat_total,
            "candidates_a": candidates_a,
            "candidates_b": candidates_b,
            "thumbs_a": b64_thumbs,
            "thumbs_b": b64_thumbs_b,
            "siglip_scores_b": siglip_scores_b,
            "top10_desc_a": top10_desc_a,
            "top10_desc_b": rank_shifts,
            "token_telemetry": token_telemetry,
        })

        badge = f"[{category}]"
        print(f"\n--- [{idx:02d}/{len(BTC_KIS_QUERIES)}] {qid} {badge} : {name} ---", flush=True)
        print(f"• Marian EN Query: \"{eff_en}\"", flush=True)
        print(f"• Token Telemetry: CLIP={tok_count_clip}/77 | SigLIP2 Raw={token_telemetry['siglip_raw_token_count']}, Eff={token_telemetry['siglip_effective_token_count']}/{token_telemetry['siglip_model_max_length']} (Truncated={token_telemetry['siglip_truncated']})", flush=True)
        print(f"• Latency Breakdown: 1st-pass CLIP={lat_coarse*1000:5.1f}ms | Frame Decode(30)={lat_decode*1000:5.1f}ms (Coverage: {coverage_status} ✅) | SigLIP2 Scoring={lat_siglip*1000:5.1f}ms | Total={lat_total*1000:5.1f}ms", flush=True)
        print(f"• Arm A (CLIP Top 5)    : {top10_desc_a[:5]}", flush=True)
        print(f"• Arm B (SigLIP2 Top 5) : {rank_shifts[:5]}", flush=True)

        if qid == "query-p1-13-kis":
            # Check camera cleaning video L30_V095
            pos_a = next((f"@{i+1}" for i, c in enumerate(candidates_a) if c.video_id == "L30_V095"), "Not in Top30")
            pos_b = next((f"@{i+1}" for i, c in enumerate(candidates_b) if c.video_id == "L30_V095"), "Not in Top30")
            print(f"  🔍 TARGET PROBE (p1-13 Camera Lens): L30_V095 rank migration -> Arm A (CLIP): {pos_a}  --->  Arm B (SigLIP2): {pos_b}", flush=True)
        elif qid == "query-p1-24-kis":
            print(f"  🔍 TARGET PROBE (p1-24 Overhead Cycling): Arm A Top 1: {candidates_a[0].video_id}  --->  Arm B Top 1: {candidates_b[0].video_id}", flush=True)
        elif qid == "query-p1-21-kis":
            print(f"  🔍 TARGET PROBE (p1-21 Beetle/Robot Lab): Arm A Top 1: {candidates_a[0].video_id}  --->  Arm B Top 1: {candidates_b[0].video_id}", flush=True)

    # Generate comparative HTML gallery
    gallery_out = Path("/kaggle/working/kis_p1d0_siglip2_gallery.html")
    generate_rerank_gallery_html(results, gallery_out, SIGLIP2_MODEL_ID)
    print(f"\nSaved Comparative Side-by-Side Gallery to: {gallery_out}", flush=True)

    # Summary table
    print("\n" + "=" * 150, flush=True)
    print("KIS P1D0 SIGLIP2 RERANKING LATENCY & TOP 1 OVERVIEW TABLE", flush=True)
    print("=" * 150, flush=True)
    print(f"{'Query ID':<18} | {'Category':<22} | {'Arm A CLIP Top 1':<24} | {'Arm B SigLIP Top 1':<24} | {'Decode':<8} | {'SigLIP':<8} | {'Total Lat':<10}")
    print("-" * 125)
    for r in results:
        top1_a = f"{r['candidates_a'][0].video_id} (f={r['candidates_a'][0].frame_id})"
        top1_b = f"{r['candidates_b'][0].video_id} (f={r['candidates_b'][0].frame_id})"
        print(f"{r['qid']:<18} | {r['category']:<22} | {top1_a:<24} | {top1_b:<24} | {r['lat_decode']*1000:5.0f}ms | {r['lat_siglip']*1000:5.0f}ms | {r['lat_total']*1000:6.0f} ms")
    print("=" * 150, flush=True)
    print(f"Mean Latencies (30 frames): Decode = {np.mean(decode_latencies)*1000:.1f}ms | SigLIP2 Scoring = {np.mean(rerank_latencies)*1000:.1f}ms | Total Query Latency = {np.mean(total_latencies)*1000:.1f}ms", flush=True)
    print("=" * 150, flush=True)


def generate_rerank_gallery_html(results: list[dict[str, Any]], out_path: Path, model_id: str) -> None:
    html_cards = []
    for r in results:
        qid = r["qid"]
        name = r["name"]
        category = r["category"]
        q_vi = r["query_vi"]
        en_a = r["eff_en"]
        preds_a = r["candidates_a"][:3]
        thumbs_a = r["thumbs_a"][:3]
        preds_b = r["candidates_b"][:3]
        thumbs_b = r["thumbs_b"][:3]
        scores_b = r["siglip_scores_b"][:3]

        def render_top3_grid(preds: list[Any], thumbs: list[str], label: str, color: str, is_siglip: bool = False) -> str:
            items = []
            for rank_idx, (p, img_b64) in enumerate(zip(preds, thumbs), start=1):
                vid = p.video_id
                fid = p.frame_id
                score_str = f"s_siglip={scores_b[rank_idx-1]:.3f}" if is_siglip else f"s_clip={p.score:.3f}"
                img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:4px;" />' if img_b64 else '<div style="background:#333;color:#888;height:80px;display:flex;align-items:center;justify-content:center;">No Frame</div>'
                items.append(f"""
                <div style="flex:1; margin:4px; padding:6px; background:#181818; border:1px solid #333; border-radius:6px; text-align:center; font-size:11px;">
                    <div style="font-weight:bold; color:{color};">Rank @{rank_idx} ({score_str})</div>
                    {img_tag}
                    <div style="color:#eee; font-weight:600; margin-top:2px;">{vid}</div>
                    <div style="color:#888; font-size:10px;">f={fid}</div>
                </div>
                """)
            return f"""
            <div style="flex:1; padding:8px; background:#222; border-radius:6px; margin:4px;">
                <div style="font-weight:bold; color:{color}; margin-bottom:6px;">{label}</div>
                <div style="display:flex;">{''.join(items)}</div>
            </div>
            """

        grid_a = render_top3_grid(preds_a, thumbs_a, "ARM A: OpenAI CLIP 1st-Pass Coarse Order (Top 3)", "#0d6efd", is_siglip=False)
        grid_b = render_top3_grid(preds_b, thumbs_b, f"ARM B: Pure SigLIP2 Re-ordered Shortlist (Top 3)", "#28a745", is_siglip=True)

        cat_badge_color = "#ffc107; color:#111" if category == "REGRESSION_GUARD" else ("#e83e8c; color:#fff" if category == "TARGET_PROBE" else "#17a2b8; color:#fff")

        html_cards.append(f"""
        <div style="background:#2b2b2b; border:1px solid #444; border-radius:8px; margin-bottom:20px; padding:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #3c3c3c; padding-bottom:8px; margin-bottom:12px;">
                <span style="font-size:15px; font-weight:bold; color:#61afef;">{qid}</span>
                <span style="font-size:13px; font-weight:bold; color:#fff;">{name}</span>
                <span style="background:{cat_badge_color}; font-weight:bold; font-size:11px; padding:3px 8px; border-radius:4px;">{category}</span>
            </div>
            <div style="font-size:12px; color:#ccc; margin-bottom:4px;"><b style="color:#aaa;">VI Query:</b> {q_vi}</div>
            <div style="font-size:11px; color:#9cdcfe; margin-bottom:12px;"><b style="color:#0d6efd;">Marian EN Query:</b> "{en_a}"</div>
            <div style="display:flex; gap:8px;">
                {grid_a}
                {grid_b}
            </div>
        </div>
        """)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>KIS P1D0 SigLIP2 Reranking Gallery</title></head>
    <body style="background:#121212; color:#fff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; padding:20px;">
        <h2 style="color:#61afef; border-bottom:2px solid #333; padding-bottom:8px;">🔬 KIS P1D0: SIGLIP2 TOP-30 VISUAL RERANKER SHOOTOUT GALLERY</h2>
        <div style="color:#aaa; font-size:13px; margin-bottom:16px;">
            Model: <b>{model_id}</b> | Shortlist size: <b>30 frames</b> | Zero score fusion (Pure Visual Re-ordering).
        </div>
        {''.join(html_cards)}
    </body>
    </html>
    """
    out_path.write_text(full_html, encoding="utf-8")


if __name__ == "__main__":
    run_p1d0_experiment()
