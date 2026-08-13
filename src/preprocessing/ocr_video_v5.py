from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.preprocessing.ocr_temporal_merger import bbox_iou, remove_vietnamese_accents, text_similarity


ROLE_HEADLINE = "HEADLINE"
ROLE_TICKER = "TICKER"
ROLE_SCENE = "SCENE_TEXT"
ROLE_LOGO = "CHANNEL_LOGO"
ROLE_CLOCK = "CLOCK"
ROLE_SCOREBOARD = "SCOREBOARD"
ROLE_OTHER = "OTHER"


VIETNAMESE_CORRECTION_RULES: list[tuple[str, str]] = [
    (r"\bNGHI\s+LE\b", "NGHỈ LỄ"),
    (r"\bNGHỈ\s+LÊ\b", "NGHỈ LỄ"),
    (r"\bNGHI\s+L\b", "NGHỈ LỄ"),
    (r"\bQU[ÕO]?C\b", "QUỐC"),
    (r"\bLIÊN\s+TI[ËE]?P\b", "LIÊN TIẾP"),
    (r"\bVI[ÊE]T\s+NAM\b", "VIỆT NAM"),
    (r"\bHÀN\s+QUỐC\b", "HÀN QUỐC"),
    (r"\bMT\s+BÀNG\b", "MẶT BẰNG"),
    (r"\bNHÀ\s+PH[ÕO]\b", "NHÀ PHỐ"),
    (r"\bNHÀ\s+PHỐ\s+[ÊE]?\s*AM\b", "NHÀ PHỐ Ế ẨM"),
    (r"\bSHOPHOUSE\s+SÔI\s+D[ÔO]NG\b", "SHOPHOUSE SÔI ĐỘNG"),
    (r"\bLÂN\s+D[ÃA]U\b", "LẦN ĐẦU"),
    (r"\bGI[ÁAÅ]M\s+L[ÃA]I\s+SU[ÃA]T\b", "GIẢM LÃI SUẤT"),
    (r"\bSAU\s+HON\s+4\s+N[ÃA]M\b", "SAU HƠN 4 NĂM"),
    (r"\bKHAI\s+MAC\b", "KHAI MẠC"),
    (r"\bTRI[ÊE]N\s+L[ÃA]M\b", "TRIỂN LÃM"),
    (r"\bY\s+DUC\b", "Y DƯỢC"),
    (r"\bQUỐC\s+T\b", "QUỐC TẾ"),
    (r"\bHÀN\s+QUỐC\s+CHÍNH\s+TH[ÚU]C\s+NH[ÂA]P\s+QU[ÀA]\s+BUI\s+CỦA\s+VIỆT\s+NAM\b", "HÀN QUỐC CHÍNH THỨC NHẬP QUẢ BƯỞI CỦA VIỆT NAM"),
    (r"\bD[ÁA]K\s+L[ÁA]K\b", "ĐẮK LẮK"),
    (r"\bDI\s+VÀO\s+DI[ÊE]M\s+MÙ\b", "ĐI VÀO ĐIỂM MÙ"),
    (r"\bXE\s+D[ÃA]U\s+K[ÉE]O\b", "XE ĐẦU KÉO"),
    (r"\bTHÀNH\s+PH[ÕO]\s+THUNG\s+HÀI\s+\(TRUNG\s+QUỐC\)", "THÀNH PHỐ THƯỢNG HẢI (TRUNG QUỐC)"),
    (r"\bROBOT\s+DON\s+C[ÓO]\s+VÀ\s+KI[ÊE]M\s+SO[ÁA]T\s+CÂY\s+TR[ÔO]NG\s+T[ÚU]\s+D[ÔO]NG\b", "ROBOT DỌN CỎ VÀ KIỂM SOÁT CÂY TRỒNG TỰ ĐỘNG"),
    (r"\bTRU\s+[ÄAĂ]NG\s+TEN\s+DÀI\b", "TRỤ ĂNG TEN ĐÀI"),
    (r"\bCHÂU\s+ÂU\s+HAN\s+CH[ÊE]\s+S[ÕO]\s+T\s+XU[ÃA]T\s+HUY[ÊE]T\b", "CHÂU ÂU HẠN CHẾ SỐT XUẤT HUYẾT"),
    (r"\bDINH\s+CHI\s+TÀI\s+X[ÊE]\s+XE\s+BU[ÝY]T\s+VUT\s+D[ÊE]N\s+D[ÒO]\s+TR[ÊE]N\s+QUỐC\s+L[ÔO]\s+22\b", "ĐÌNH CHỈ TÀI XẾ XE BUÝT VƯỢT ĐÈN ĐỎ TRÊN QUỐC LỘ 22"),
]


def normalize_light(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.strip().split())


def normalize_for_match(text: str, no_accent: bool = False) -> str:
    text = normalize_light(text).lower()
    text = text.replace("–", "-").replace("—", "-").replace("：", ":")
    text = re.sub(r"\s+", " ", text)
    if no_accent:
        text = remove_vietnamese_accents(text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    text = normalize_for_match(text, no_accent=True)
    return re.findall(r"[\w%:/.-]+", text, flags=re.UNICODE)


def vietnamese_quality_score(text: str) -> float:
    text = normalize_light(text)
    if not text:
        return 0.0
    upper = text.upper()
    penalties = 0.0
    suspicious_patterns = [
        r"\bQUC\b",
        r"\bPHÕ\b",
        r"\bMT\s+BÀNG\b",
        r"\bSUÃT\b",
        r"\bDÃU\b",
        r"\bVIÊT\s+NAM\b",
        r"\bHON\b",
        r"\bNÃM\b",
        r"\bDÔNG\b",
        r"\bTRIÊN\b",
        r"\bDUC\b",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, upper):
            penalties += 0.10
    tokens = tokenize(text)
    if tokens:
        one_char_ratio = sum(1 for token in tokens if len(token) <= 1) / len(tokens)
        penalties += min(0.25, one_char_ratio * 0.25)
    if looks_like_vietocr_garbage(text):
        penalties += 0.35
    accent_chars = sum(1 for ch in text if unicodedata.normalize("NFD", ch) != ch or ch in "đĐ")
    if len(text) >= 12 and accent_chars == 0:
        penalties += 0.08
    return round(max(0.0, min(1.0, 1.0 - penalties)), 4)


def correct_vietnamese_text(text: str, role: str) -> dict[str, Any]:
    original = normalize_light(text)
    if role in {ROLE_CLOCK, ROLE_LOGO, ROLE_SCOREBOARD}:
        return {
            "text": original,
            "changed": False,
            "source": "not_applicable",
            "reasons": [],
            "quality_score": vietnamese_quality_score(original),
            "needs_human_review": False,
        }

    corrected = original
    reasons = []
    for pattern, replacement in VIETNAMESE_CORRECTION_RULES:
        new_text = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        if new_text != corrected:
            reasons.append(pattern)
            corrected = new_text
    corrected = normalize_light(corrected)
    score = vietnamese_quality_score(corrected)
    return {
        "text": corrected,
        "changed": corrected != original,
        "source": "vietnamese_rules" if corrected != original else "none",
        "reasons": reasons,
        "quality_score": score,
        "needs_human_review": bool(score < 0.78 and role in {ROLE_HEADLINE, ROLE_TICKER}),
    }


def parse_bbox(value: Any) -> list[int]:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return [0, 0, 0, 0]
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", value)]
            return [int(round(x)) for x in nums[:4]] if len(nums) >= 4 else [0, 0, 0, 0]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [int(round(float(v))) for v in value]
    return [0, 0, 0, 0]


def bbox_normalized(bbox: list[int], width: int, height: int) -> list[float]:
    w = max(1, int(width))
    h = max(1, int(height))
    return [
        round(float(bbox[0]) / w, 6),
        round(float(bbox[1]) / h, 6),
        round(float(bbox[2]) / w, 6),
        round(float(bbox[3]) / h, 6),
    ]


def bbox_center_norm(bbox: list[int], width: int, height: int) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_normalized(bbox, width, height)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance_norm(a: list[int], b: list[int], width: int, height: int) -> float:
    ax, ay = bbox_center_norm(a, width, height)
    bx, by = bbox_center_norm(b, width, height)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def bbox_area_norm(bbox: list[int], width: int, height: int) -> float:
    x1, y1, x2, y2 = bbox_normalized(bbox, width, height)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_width_norm(bbox: list[int], width: int) -> float:
    return max(0.0, float(bbox[2] - bbox[0]) / max(1, width))


def bbox_height_norm(bbox: list[int], height: int) -> float:
    return max(0.0, float(bbox[3] - bbox[1]) / max(1, height))


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[:./-]\d+)*", normalize_light(text))


def looks_like_clock(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", normalize_light(text)))


def looks_like_scoreboard(text: str) -> bool:
    text = normalize_light(text).upper()
    return bool(re.search(r"\b[A-Z]{2,4}\b\s+\d+\s*[-:]\s*\d+\s+\b[A-Z]{2,4}\b", text))


def looks_like_vietocr_garbage(text: str) -> bool:
    text = normalize_light(text)
    compact = re.sub(r"[^A-Za-z]", "", text).upper()
    if re.fullmatch(r"\d{7,}", re.sub(r"\D", "", text)):
        return True
    if not compact:
        return False
    suspicious_suffixes = (
        "TION",
        "TIONS",
        "ALITY",
        "ALITIES",
        "ISM",
        "ISMS",
        "IZED",
        "IZING",
        "ATIONAL",
        "PRESSION",
        "CONTRACTION",
        "MENTALIZED",
        "TENTIATION",
    )
    if len(compact) >= 7 and (compact.endswith(suspicious_suffixes) or "CONTRACTION" in compact):
        return True
    vowels = sum(1 for ch in compact if ch in "AEIOUY")
    return len(compact) >= 10 and vowels / max(1, len(compact)) < 0.18


def looks_like_channel_logo(text: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", normalize_light(text)).upper()
    if compact in {"HTV", "HTV7", "HTV9", "HTVHD", "VTV", "VTV1", "VTV3", "VTV9", "THVL", "HD"}:
        return True
    return bool(re.fullmatch(r"(?:HTV|VTV|THVL)\d{0,2}(?:HD)?", compact))


def is_short_valid_text(text: str) -> bool:
    text = normalize_light(text)
    return bool(re.fullmatch(r"(?:[A-ZĐ]{1,5}\d{0,2}|\d+[Gg]?|\d+\s*[-:]\s*\d+)", text))


def text_is_gibberish(text: str) -> bool:
    text = normalize_light(text)
    if not text:
        return True
    tokens = tokenize(text)
    if len(tokens) >= 4:
        short = sum(1 for t in tokens if len(t) <= 1)
        if short / len(tokens) >= 0.55:
            return True
    alnum = sum(ch.isalnum() for ch in text)
    if len(text) >= 8 and alnum / max(1, len(text)) < 0.45:
        return True
    return False


def _best_preferred_candidate(
    cleaned: list[dict[str, Any]],
    preferred_engine: str | None,
    min_similarity: float,
) -> dict[str, Any] | None:
    if not preferred_engine:
        return None
    preferred = [c for c in cleaned if c.get("engine") == preferred_engine]
    fallback = [c for c in cleaned if c.get("engine") != preferred_engine]
    if not preferred:
        return None

    fallback_text = ""
    if fallback:
        fallback_counts = Counter(c["norm"] for c in fallback)
        fallback_text = fallback_counts.most_common(1)[0][0]

    best = None
    best_score = -1.0
    for cand in preferred:
        text = cand["text"]
        if looks_like_vietocr_garbage(text) or text_is_gibberish(text):
            continue
        pref_numbers = extract_numbers(text)
        fallback_numbers = extract_numbers(fallback_text)
        if pref_numbers and fallback_numbers and pref_numbers != fallback_numbers:
            continue
        sim = text_similarity(cand["norm"], fallback_text) if fallback_text else 1.0
        has_vietnamese_signal = any(ch in text for ch in "ăâđêôơưĂÂĐÊÔƠƯàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ")
        if fallback_text and sim < min_similarity and not has_vietnamese_signal:
            continue
        score = sim + (0.20 if has_vietnamese_signal else 0.0) + min(len(text), 80) / 400.0
        if score > best_score:
            best = cand
            best_score = score
    return best


def consensus_from_candidates(
    candidates: list[dict[str, Any]],
    preferred_engine: str | None = None,
    preferred_min_similarity: float = 0.45,
) -> dict[str, Any]:
    cleaned = []
    for cand in candidates:
        text = normalize_light(str(cand.get("text", "")))
        if text:
            cleaned.append({**cand, "text": text, "norm": normalize_for_match(text, no_accent=True)})
    if not cleaned:
        return {"text": "", "numeric_conflict": False, "engine_agreement": 0.0, "reason": "empty"}

    number_votes: Counter[str] = Counter()
    number_signatures: Counter[tuple[str, ...]] = Counter()
    for cand in cleaned:
        nums = tuple(extract_numbers(cand["text"]))
        if nums:
            number_signatures[nums] += 1
        for num in nums:
            number_votes[num] += 1
    numeric_conflict = len(number_signatures) > 1

    exact_votes: Counter[str] = Counter()
    confidence_by_text: dict[str, list[float]] = {}
    original_by_norm: dict[str, str] = {}
    for cand in cleaned:
        norm = cand["norm"]
        exact_votes[norm] += 1
        confidence_by_text.setdefault(norm, []).append(float(cand.get("confidence") or 0.0))
        original_by_norm.setdefault(norm, cand["text"])

    preferred = _best_preferred_candidate(cleaned, preferred_engine, preferred_min_similarity)
    if preferred is not None and not numeric_conflict:
        return {
            "text": preferred["text"],
            "numeric_conflict": False,
            "engine_agreement": 1.0,
            "reason": f"{preferred_engine}_primary",
        }

    best_norm = ""
    best_score = -1.0
    for norm, freq in exact_votes.items():
        mean_conf = sum(confidence_by_text[norm]) / max(1, len(confidence_by_text[norm]))
        sim_sum = sum(text_similarity(norm, other) for other in exact_votes if other != norm)
        score = (freq * 2.5) + (mean_conf * 1.5) + (sim_sum * 0.35) + (len(norm) * 0.02)
        if score > best_score:
            best_score = score
            best_norm = norm

    agreement_items = []
    seen_norms = set()
    for norm, _freq in exact_votes.most_common(24):
        agreement_items.append({"norm": norm})
        seen_norms.add(norm)
    if len(agreement_items) < 2:
        for cand in cleaned[:24]:
            if cand["norm"] not in seen_norms:
                agreement_items.append(cand)
                seen_norms.add(cand["norm"])
            if len(agreement_items) >= 24:
                break

    engine_agreement = 0.0
    if len(agreement_items) >= 2:
        pairs = 0
        sim_total = 0.0
        for i in range(len(agreement_items)):
            for j in range(i + 1, len(agreement_items)):
                pairs += 1
                sim_total += text_similarity(agreement_items[i]["norm"], agreement_items[j]["norm"])
        engine_agreement = sim_total / max(1, pairs)
    else:
        engine_agreement = 1.0

    if number_votes:
        top_num, top_count = number_votes.most_common(1)[0]
        if top_count >= 2 and all(c["norm"] in number_votes or True for c in cleaned):
            numeric_texts = [c["text"] for c in cleaned if normalize_for_match(c["text"]) == top_num or top_num in extract_numbers(c["text"])]
            if numeric_texts and all(re.fullmatch(r"[\d\s:./-]+", t) for t in numeric_texts):
                return {
                    "text": numeric_texts[0],
                    "numeric_conflict": numeric_conflict,
                    "engine_agreement": round(engine_agreement, 4),
                    "reason": "numeric_majority",
                }

    return {
        "text": original_by_norm[best_norm],
        "numeric_conflict": numeric_conflict,
        "engine_agreement": round(engine_agreement, 4),
        "reason": "weighted_temporal_vote",
    }


def build_raw_observations(frame_records: list[dict[str, Any]], video_meta: dict[str, Any]) -> list[dict[str, Any]]:
    width = int(video_meta.get("width") or 0)
    height = int(video_meta.get("height") or 0)
    rows = []
    counter = 0
    for rec in frame_records:
        video_id = str(rec.get("video_id", ""))
        frame_idx = int(rec.get("frame_idx", 0))
        timestamp = float(rec.get("timestamp_seconds", rec.get("timestamp_sec", 0.0)))
        for det_idx, det in enumerate(rec.get("detections", [])):
            bbox = parse_bbox(det.get("bbox", det.get("box", [0, 0, 0, 0])))
            paddle_text = normalize_light(str(det.get("paddle_text", det.get("text", ""))))
            paddle_conf = float(det.get("confidence", det.get("rec_conf", det.get("det_conf", 0.0))) or 0.0)
            if not paddle_text:
                continue
            counter += 1
            rows.append({
                "observation_id": f"{video_id}_obs_{counter:08d}",
                "video_id": video_id,
                "timestamp": round(timestamp, 4),
                "frame_id": frame_idx,
                "frame_idx": frame_idx,
                "shot_id": int(rec.get("shot_id", -1)),
                "sample_stage": str(rec.get("sample_stage", "")),
                "det_idx": int(det.get("det_idx", det_idx)),
                "bbox": bbox,
                "bbox_normalized": bbox_normalized(bbox, width, height),
                "paddle_text": paddle_text,
                "paddle_conf": round(paddle_conf, 4),
                "vietocr_text": normalize_light(str(det.get("vietocr_text", ""))) or None,
                "vietocr_conf": det.get("vietocr_conf", None),
                "source_width": width,
                "source_height": height,
                "region_hint": str(det.get("region_type", "scene_text")),
                "crop_path": str(det.get("crop_path", "")),
            })
    return rows


def merge_horizontal_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for obs in observations:
        grouped.setdefault(int(obs["frame_id"]), []).append(obs)

    merged_rows = []
    for frame_id, rows in grouped.items():
        rows = sorted(rows, key=lambda r: (r["bbox"][1], r["bbox"][0]))
        used = [False] * len(rows)
        for i, row in enumerate(rows):
            if used[i]:
                continue
            used[i] = True
            members = [row]
            bbox = list(row["bbox"])
            texts = [row["paddle_text"]]
            confs = [float(row["paddle_conf"])]
            for j in range(i + 1, len(rows)):
                if used[j]:
                    continue
                nxt = rows[j]
                cur_h = max(1, bbox[3] - bbox[1])
                nxt_h = max(1, nxt["bbox"][3] - nxt["bbox"][1])
                y_overlap = max(0, min(bbox[3], nxt["bbox"][3]) - max(bbox[1], nxt["bbox"][1]))
                x_gap = nxt["bbox"][0] - bbox[2]
                same_line = y_overlap / max(1, min(cur_h, nxt_h)) >= 0.35
                close_x = -30 <= x_gap <= max(cur_h, nxt_h) * 2.5
                if same_line and close_x:
                    used[j] = True
                    members.append(nxt)
                    bbox = [
                        min(bbox[0], nxt["bbox"][0]),
                        min(bbox[1], nxt["bbox"][1]),
                        max(bbox[2], nxt["bbox"][2]),
                        max(bbox[3], nxt["bbox"][3]),
                    ]
                    texts.append(nxt["paddle_text"])
                    confs.append(float(nxt["paddle_conf"]))
            width = int(row["source_width"])
            height = int(row["source_height"])
            merged_rows.append({
                "line_observation_id": f"{row['video_id']}_line_{frame_id:08d}_{len(merged_rows):05d}",
                "video_id": row["video_id"],
                "timestamp": row["timestamp"],
                "frame_id": frame_id,
                "frame_idx": frame_id,
                "bbox": bbox,
                "bbox_normalized": bbox_normalized(bbox, width, height),
                "paddle_text": normalize_light(" ".join(texts)),
                "paddle_conf": round(sum(confs) / max(1, len(confs)), 4),
                "source_width": width,
                "source_height": height,
                "region_hint": row["region_hint"],
                "member_observation_ids": [m["observation_id"] for m in members],
                "raw_observations": members,
            })
    return sorted(merged_rows, key=lambda r: (r["frame_id"], r["bbox"][1], r["bbox"][0]))


def _track_match_score(det: dict[str, Any], track: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[bool, float]:
    last = track[-1]
    width = int(det.get("source_width") or last.get("source_width") or 1)
    height = int(det.get("source_height") or last.get("source_height") or 1)
    max_gap = float(cfg.get("max_gap_seconds", 3.0))
    if float(det["timestamp"]) - float(last["timestamp"]) > max_gap:
        return False, 0.0
    iou = bbox_iou(det["bbox"], last["bbox"])
    center_dist = center_distance_norm(det["bbox"], last["bbox"], width, height)
    sim = text_similarity(det["paddle_text"], last["paddle_text"])
    min_iou = float(cfg.get("bbox_iou_threshold", 0.30))
    min_sim = float(cfg.get("min_text_similarity", 0.70))
    max_dist = float(cfg.get("center_distance_threshold", 0.12))
    role_hint = str(det.get("region_hint", ""))
    is_clock = looks_like_clock(det["paddle_text"]) and looks_like_clock(last["paddle_text"])
    if iou >= min_iou and (sim >= 0.35 or is_clock or role_hint in {"logo_channel", "clock_time"}):
        return True, (iou * 0.55) + ((1.0 - min(center_dist / max_dist, 1.0)) * 0.20) + (sim * 0.25)
    if iou >= 0.10 and center_dist <= max_dist and sim >= min_sim:
        return True, (iou * 0.35) + ((1.0 - min(center_dist / max_dist, 1.0)) * 0.25) + (sim * 0.40)
    return False, 0.0


def build_temporal_tracks(line_observations: list[dict[str, Any]], cfg: dict[str, Any]) -> list[list[dict[str, Any]]]:
    active: list[list[dict[str, Any]]] = []
    finished: list[list[dict[str, Any]]] = []
    max_gap = float(cfg.get("max_gap_seconds", 3.0))
    for obs in line_observations:
        next_active = []
        for tr in active:
            if float(obs["timestamp"]) - float(tr[-1]["timestamp"]) <= max_gap:
                next_active.append(tr)
            else:
                finished.append(tr)
        active = next_active
        best_idx = -1
        best_score = 0.0
        for idx, tr in enumerate(active):
            ok, score = _track_match_score(obs, tr, cfg)
            if ok and score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0:
            active[best_idx].append(obs)
        else:
            active.append([obs])
    finished.extend(active)
    return finished


def classify_track_role(track: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[str, float, str]:
    first = track[0]
    width = int(first.get("source_width") or 1)
    height = int(first.get("source_height") or 1)
    boxes = [o["bbox"] for o in track]
    bbox_mean = [int(round(sum(b[i] for b in boxes) / max(1, len(boxes)))) for i in range(4)]
    x1, y1, x2, y2 = bbox_normalized(bbox_mean, width, height)
    w_n = x2 - x1
    h_n = y2 - y1
    area = w_n * h_n
    texts = [o["paddle_text"] for o in track]
    joined = " ".join(texts[:80])
    duration = max(0.0, float(track[-1]["timestamp"]) - float(track[0]["timestamp"]))
    hint_counts = Counter(str(o.get("region_hint", "")) for o in track)
    top_hint = hint_counts.most_common(1)[0][0] if hint_counts else ""

    if all(looks_like_clock(t) for t in texts[: min(5, len(texts))]) and y1 <= float(cfg.get("overlay_corner_y_max", 0.30)):
        return ROLE_CLOCK, 0.97, "clock_pattern_corner"
    if looks_like_scoreboard(joined):
        return ROLE_SCOREBOARD, 0.93, "scoreboard_pattern"
    if top_hint == "ticker" or (y1 >= float(cfg.get("ticker_y_min", 0.82)) and w_n >= float(cfg.get("ticker_min_width", 0.30))):
        return ROLE_TICKER, 0.90, "bottom_long_text_region"
    headline_min_width = float(cfg.get("headline_min_width", 0.35))
    if (
        top_hint == "headline"
        and w_n >= headline_min_width
    ) or (
        float(cfg.get("headline_y_min", 0.45)) <= y1 <= float(cfg.get("headline_y_max", 0.84))
        and w_n >= headline_min_width
    ):
        return ROLE_HEADLINE, 0.88, "lower_third_layout"
    logo_like = any(looks_like_channel_logo(t) for t in texts[: min(20, len(texts))])
    persistent_corner = (
        y1 <= float(cfg.get("overlay_corner_y_max", 0.30))
        and area <= float(cfg.get("logo_max_area", 0.06))
        and duration >= float(cfg.get("channel_persistence_seconds", 20.0))
    )
    if logo_like or persistent_corner:
        return ROLE_LOGO, 0.84, "persistent_corner_overlay"
    if top_hint == "scene_text":
        return ROLE_SCENE, 0.78, "scene_text_layout"
    if len(joined) <= 2 and not is_short_valid_text(joined):
        return ROLE_OTHER, 0.62, "short_uncertain"
    return ROLE_SCENE, 0.58, "default_scene_text"


def spatial_stability(track: list[dict[str, Any]]) -> float:
    if len(track) <= 1:
        return 0.6
    first = track[0]
    width = int(first.get("source_width") or 1)
    height = int(first.get("source_height") or 1)
    centers = [bbox_center_norm(o["bbox"], width, height) for o in track]
    mean_x = sum(x for x, _ in centers) / len(centers)
    mean_y = sum(y for _, y in centers) / len(centers)
    avg_dist = sum(math.sqrt((x - mean_x) ** 2 + (y - mean_y) ** 2) for x, y in centers) / len(centers)
    return max(0.0, min(1.0, 1.0 - avg_dist / 0.12))


def reliability_score(track: list[dict[str, Any]], consensus: dict[str, Any], cfg: dict[str, Any]) -> float:
    weights = cfg.get("reliability_weights", {})
    obs_count = len(track)
    duration = max(0.0, float(track[-1]["timestamp"]) - float(track[0]["timestamp"]))
    temporal_support = min(1.0, (obs_count / 4.0) * 0.65 + min(duration / 4.0, 1.0) * 0.35)
    paddle_conf = sum(float(o.get("paddle_conf", 0.0)) for o in track) / max(1, obs_count)
    obs_score = min(1.0, obs_count / 6.0)
    stability = spatial_stability(track)
    engine = float(consensus.get("engine_agreement", 0.0))
    score = (
        float(weights.get("engine_agreement", 0.25)) * engine
        + float(weights.get("temporal_support", 0.25)) * temporal_support
        + float(weights.get("paddle_confidence", 0.20)) * paddle_conf
        + float(weights.get("observation_count", 0.15)) * obs_score
        + float(weights.get("spatial_stability", 0.15)) * stability
    )
    if consensus.get("numeric_conflict"):
        score -= 0.12
    return round(max(0.0, min(1.0, score)), 4)


def quality_status(text: str, role: str, score: float, obs_count: int, cfg: dict[str, Any]) -> tuple[str, str]:
    text = normalize_light(text)
    if not text:
        return "REJECT", "empty_text"
    if text_is_gibberish(text) and score < float(cfg.get("keep_threshold", 0.58)):
        return "REJECT", "gibberish_low_reliability"
    if len(text) <= 2 and not is_short_valid_text(text):
        if obs_count >= int(cfg.get("min_observations_for_short_text", 1)) and role in {ROLE_LOGO, ROLE_SCOREBOARD, ROLE_CLOCK}:
            return "KEEP", "short_but_valid_role"
        return "LOW_QUALITY", "short_uncertain"
    if role in {ROLE_LOGO, ROLE_CLOCK}:
        return "KEEP", "low_semantic_overlay_metadata"
    if score >= float(cfg.get("keep_threshold", 0.58)):
        return "KEEP", "reliable"
    if score >= float(cfg.get("low_quality_threshold", 0.35)):
        return "LOW_QUALITY", "low_reliability"
    return "REJECT", "very_low_reliability"


def tracks_to_records(tracks: list[list[dict[str, Any]]], cfg: dict[str, Any], video_id: str) -> list[dict[str, Any]]:
    records = []
    temporal_cfg = cfg.get("consensus", {})
    role_cfg = cfg.get("roles", {})
    quality_cfg = cfg.get("quality", {})
    for idx, track in enumerate(tracks, start=1):
        candidates = []
        raw_obs = []
        for obs in track:
            candidates.append({
                "engine": "paddle",
                "text": obs["paddle_text"],
                "confidence": obs["paddle_conf"],
                "frame_id": obs["frame_id"],
            })
            if obs.get("vietocr_text"):
                candidates.append({
                    "engine": "vietocr",
                    "text": obs["vietocr_text"],
                    "confidence": obs.get("vietocr_conf"),
                    "frame_id": obs["frame_id"],
                })
            raw_obs.extend(obs.get("raw_observations", []))
        ocr_cfg = cfg.get("ocr", {})
        priority = str(ocr_cfg.get("recognition_priority", "temporal_majority"))
        preferred_engine = "vietocr" if priority in {"vietocr_primary", "v2_vietocr_primary"} else None
        cons = consensus_from_candidates(
            candidates,
            preferred_engine=preferred_engine,
            preferred_min_similarity=float(ocr_cfg.get("vietocr_primary_min_similarity", 0.45)),
        )
        role, role_conf, role_reason = classify_track_role(track, role_cfg)
        score = reliability_score(track, cons, temporal_cfg)
        q_status, q_reason = quality_status(cons["text"], role, score, len(track), quality_cfg)
        width = int(track[0].get("source_width") or 1)
        height = int(track[0].get("source_height") or 1)
        boxes = [o["bbox"] for o in track]
        bbox_mean = [int(round(sum(b[i] for b in boxes) / max(1, len(boxes)))) for i in range(4)]
        text_values = [o["paddle_text"] for o in track if o.get("paddle_text")]
        canonical = cons["text"]
        if role == ROLE_CLOCK and len(text_values) > 1:
            canonical = f"{text_values[0]} -> {text_values[-1]}"
        raw_canonical = canonical
        correction = correct_vietnamese_text(raw_canonical, role)
        canonical = str(correction["text"])
        record = {
            "track_id": f"{video_id}_ocrv5_track_{idx:06d}",
            "video_id": video_id,
            "start_time": round(float(track[0]["timestamp"]), 3),
            "end_time": round(float(track[-1]["timestamp"]), 3),
            "start_frame": int(track[0]["frame_id"]),
            "end_frame": int(track[-1]["frame_id"]),
            "bbox_mean": bbox_mean,
            "bbox_mean_normalized": bbox_normalized(bbox_mean, width, height),
            "raw_observation_ids": [o["line_observation_id"] for o in track],
            "raw_observations": raw_obs,
            "paddle_candidates": list(dict.fromkeys([c["text"] for c in candidates if c["engine"] == "paddle"])),
            "vietocr_candidates": list(dict.fromkeys([c["text"] for c in candidates if c["engine"] == "vietocr"])),
            "temporal_consensus_text": cons["text"],
            "canonical_text_raw": raw_canonical,
            "canonical_text": canonical,
            "canonical_text_corrected": canonical,
            "vietnamese_quality_score": correction["quality_score"],
            "correction_source": correction["source"],
            "correction_reasons": correction["reasons"],
            "needs_human_review": correction["needs_human_review"],
            "role": role,
            "role_confidence": round(role_conf, 4),
            "role_reason": role_reason,
            "reliability_score": score,
            "quality_status": q_status,
            "quality_reason": q_reason,
            "numeric_conflict": bool(cons.get("numeric_conflict")),
            "engine_agreement": float(cons.get("engine_agreement", 0.0)),
            "consensus_reason": cons.get("reason", ""),
            "recognition_priority": priority,
            "observations": len(track),
            "temporal_support": min(1.0, len(track) / 6.0),
            "source_frames": sorted({int(o["frame_id"]) for o in track}),
            "qwen_used": False,
            "qwen_changed": False,
            "qwen_status": "not_requested",
        }
        records.append(record)
    return records


def searchable_fields(track: dict[str, Any]) -> dict[str, str]:
    role = str(track.get("role", ROLE_OTHER))
    text = normalize_light(str(track.get("canonical_text", "")))
    fields = {
        "headline_text": "",
        "scene_text": "",
        "ticker_text": "",
        "scoreboard_text": "",
        "other_text": "",
        "overlay_text": "",
    }
    if role == ROLE_HEADLINE:
        fields["headline_text"] = text
    elif role == ROLE_SCENE:
        fields["scene_text"] = text
    elif role == ROLE_TICKER:
        fields["ticker_text"] = text
    elif role == ROLE_SCOREBOARD:
        fields["scoreboard_text"] = text
    elif role in {ROLE_LOGO, ROLE_CLOCK}:
        fields["overlay_text"] = text
    else:
        fields["other_text"] = text
    return fields


def build_ocr_corpus(tracks: list[dict[str, Any]], role_weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    role_weights = role_weights or {}
    rows = []
    for track in tracks:
        fields = searchable_fields(track)
        text = normalize_light(str(track.get("canonical_text", "")))
        row = {
            "track_id": track["track_id"],
            "video_id": track["video_id"],
            "timestamp": track["start_time"],
            "end_time": track["end_time"],
            "frame_id": track["start_frame"],
            "text": text,
            "text_no_accent": normalize_for_match(text, no_accent=True),
            "role": track["role"],
            "role_weight": float(role_weights.get(track["role"], 0.5)),
            "reliability_score": track["reliability_score"],
            "quality_status": track["quality_status"],
            "bbox_mean": track["bbox_mean"],
            "mapped_global_id": track.get("mapped_global_id"),
            "mapped_frame_id": track.get("mapped_frame_id"),
            "mapped_frame_idx": track.get("mapped_frame_idx"),
            "mapped_keyframe_name": track.get("mapped_keyframe_name", ""),
            "mapped_keyframe_path": track.get("mapped_keyframe_path", ""),
            **fields,
        }
        rows.append(row)
    return rows


def detect_headline_boundaries(tracks: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    min_duration = float(cfg.get("min_persistence_seconds", 1.5))
    min_obs = int(cfg.get("min_observations", 2))
    sim_threshold = float(cfg.get("headline_similarity_threshold", 0.58))
    headlines = [
        t for t in tracks
        if t.get("role") == ROLE_HEADLINE
        and t.get("quality_status") != "REJECT"
        and int(t.get("observations", 0)) >= min_obs
        and float(t.get("end_time", 0.0)) - float(t.get("start_time", 0.0)) >= min_duration
    ]
    headlines.sort(key=lambda t: (float(t["start_time"]), float(t["end_time"])))
    out = []
    prev = None
    for track in headlines:
        text = str(track.get("canonical_text", ""))
        if prev is None:
            out.append({
                "timestamp": track["start_time"],
                "signal": "headline_appear",
                "boundary_type": "headline_appear",
                "confidence": float(cfg.get("appear_confidence", 0.78)) * float(track.get("role_confidence", 0.8)),
                "old_text": "",
                "new_text": text,
                "old_track_id": "",
                "new_track_id": track["track_id"],
                "headline_similarity": 0.0,
                "persistence": round(float(track["end_time"]) - float(track["start_time"]), 3),
                "accepted": True,
            })
            prev = track
            continue
        sim = text_similarity(str(prev.get("canonical_text", "")), text)
        accepted = sim < sim_threshold
        if accepted:
            out.append({
                "timestamp": track["start_time"],
                "signal": "headline_change",
                "boundary_type": "headline_change",
                "confidence": round(float(cfg.get("change_confidence", 0.88)) * (1.0 - sim) * float(track.get("role_confidence", 0.8)), 4),
                "old_text": str(prev.get("canonical_text", "")),
                "new_text": text,
                "old_track_id": prev["track_id"],
                "new_track_id": track["track_id"],
                "headline_similarity": round(sim, 4),
                "persistence": round(float(track["end_time"]) - float(track["start_time"]), 3),
                "accepted": True,
            })
        prev = track
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
