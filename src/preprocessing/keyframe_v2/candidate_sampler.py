from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetSpec:
    target_id: int
    target_ratio: float
    target_frame: int
    segment_start: int
    segment_end: int


def make_targets(start_frame: int, end_frame: int, fps: float, cfg: dict) -> list[TargetSpec]:
    n = max(1, end_frame - start_frame + 1)
    duration = n / fps
    targets: list[TargetSpec] = []
    if duration < float(cfg["short_max_seconds"]):
        ratios = [0.50]
        spans = [(start_frame, end_frame)]
    elif duration < float(cfg["medium_max_seconds"]):
        ratios = [0.50]
        spans = [(start_frame, end_frame)]
    elif duration < float(cfg["long_max_seconds"]):
        ratios = [float(x) for x in cfg["long_candidate_ratios"]]
        spans = [(start_frame, end_frame)] * len(ratios)
    else:
        seg_frames = max(1, int(round(float(cfg["long_segment_seconds"]) * fps)))
        spans = []
        cursor = start_frame
        while cursor <= end_frame:
            seg_end = min(end_frame, cursor + seg_frames - 1)
            spans.append((cursor, seg_end))
            cursor = seg_end + 1
        ratios = [0.50] * len(spans)

    for idx, (ratio, span) in enumerate(zip(ratios, spans)):
        seg_start, seg_end = span
        target_frame = int(round(seg_start + (seg_end - seg_start) * ratio))
        targets.append(TargetSpec(idx, ratio, target_frame, seg_start, seg_end))
    return targets


def margin_guard(start_frame: int, end_frame: int, cfg: dict) -> int:
    n = max(1, end_frame - start_frame + 1)
    base = max(int(cfg.get("min_guard_frames", 3)), int(round(n * float(cfg.get("guard_ratio", 0.05)))))
    cap = int(round(n * float(cfg.get("max_guard_fraction", 0.20))))
    return max(0, min(base, cap))


def make_candidate_frames(target: TargetSpec, shot_start: int, shot_end: int, fps: float, candidate_cfg: dict, guard_frames: int) -> list[int]:
    count = max(1, int(candidate_cfg.get("candidate_count", 7)))
    half_window = max(1, int(round(float(candidate_cfg.get("candidate_window_seconds", 0.5)) * fps)))
    if count == 1:
        raw = [target.target_frame]
    else:
        step = max(1, int(round((2 * half_window) / float(count - 1))))
        raw = [target.target_frame - half_window + i * step for i in range(count)]
    lo = shot_start + guard_frames
    hi = shot_end - guard_frames
    if lo > hi:
        lo, hi = shot_start, shot_end
    frames = sorted({max(lo, min(hi, int(x))) for x in raw})
    return frames
