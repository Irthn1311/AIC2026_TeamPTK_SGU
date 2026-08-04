from __future__ import annotations

import unittest
from pathlib import Path

from system_tai.common.schemas import (
    BenchmarkVideoRecord,
    FrameIndexBase,
    FrameRecord,
    RankedKISRecord,
)


class SchemaTests(unittest.TestCase):
    def test_video_record_requires_known_frame_index_base(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_index_base"):
            BenchmarkVideoRecord(
                video_id="L21_V001",
                video_path=Path("videos/L21_V001.mp4"),
                fps=30.0,
                duration_seconds=10.0,
                total_frames=300,
                frame_index_base=FrameIndexBase.UNKNOWN,
            )

    def test_frame_record_rejects_negative_actual_frame_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "actual_frame_id"):
            FrameRecord(
                video_id="L21_V001",
                actual_frame_id=-1,
                keyframe_order=1,
                clip_row=0,
                pts_time=0.0,
                fps=30.0,
                mapping_version="mapping-v1",
            )

    def test_ranked_record_requires_one_based_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            RankedKISRecord(
                query_id="Q001",
                rank=0,
                video_id="L21_V001",
                actual_frame_id=411,
                score=0.5,
            )


if __name__ == "__main__":
    unittest.main()
