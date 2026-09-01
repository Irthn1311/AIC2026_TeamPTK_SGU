"""Frozen Canonical Projection Serializer and Digest Helper for KIS Retrieval."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from typing import Any


def float_to_ieee754_hex(val: float) -> str:
    """Convert 64-bit IEEE-754 float to exact uppercase 16-hex string."""
    return f"{struct.unpack('>Q', struct.pack('>d', float(val)))[0]:016X}"


def canonical_projection_digest(candidates: Sequence[dict[str, Any]]) -> str:
    """Compute deterministic SHA256 digest of top-100 (rank, video_id, frame_id, score_bits).

    Each row is serialized as:
        {rank}:{video_id}:{frame_id}:{score_bits_hex}
    joined by newlines.
    """
    rows = []
    for c in candidates:
        rank = int(c["rank"])
        vid = str(c["video_id"]).strip()
        fid = int(c.get("frame_id", c.get("actual_frame_id", 0)))
        score_val = float(c.get("fusion_score", c.get("score", 0.0)))
        s_hex = float_to_ieee754_hex(score_val)
        rows.append(f"{rank}:{vid}:{fid}:{s_hex}")
    payload = "\n".join(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
