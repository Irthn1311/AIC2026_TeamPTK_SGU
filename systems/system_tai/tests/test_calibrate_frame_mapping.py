from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_frame_mapping.py"
SPEC = importlib.util.spec_from_file_location("system_tai_calibrate_frame_mapping", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load calibration CLI from {SCRIPT_PATH}")
CALIBRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CALIBRATION
SPEC.loader.exec_module(CALIBRATION)


class FrameMappingCalibrationTests(unittest.TestCase):
    @staticmethod
    def row(order: int, frame: int, physical_row: int = 0):
        return CALIBRATION.CalibrationMappingRow(
            keyframe_order=order,
            actual_frame_id=frame,
            physical_row=physical_row,
            pts_time=float(frame),
            fps=1.0,
        )

    @staticmethod
    def video_frames(count: int) -> dict[int, np.ndarray]:
        return {frame: np.full((4, 5, 3), frame * 20, dtype=np.uint8) for frame in range(count)}

    def test_sampling_always_includes_beginning_middle_and_end(self) -> None:
        rows = tuple(self.row(index + 1, index, index) for index in range(9))
        selected = CALIBRATION.select_mapping_rows(rows, sample_count=3)
        self.assertEqual([row.physical_row for row in selected], [0, 4, 8])

    def test_explicit_orders_are_preserved(self) -> None:
        rows = tuple(self.row(index + 1, index, index) for index in range(5))
        selected = CALIBRATION.select_mapping_rows(rows, sample_count=2, explicit_orders=[5, 1, 3])
        self.assertEqual([row.keyframe_order for row in selected], [5, 1, 3])

    def test_zero_offset_passes_for_exact_synthetic_frames(self) -> None:
        frames = self.video_frames(6)
        rows = tuple(self.row(index, index, index - 1) for index in (1, 2, 4))
        references = {row.keyframe_order: frames[row.actual_frame_id] for row in rows}
        report = CALIBRATION.evaluate_frame_offsets(
            rows,
            references,
            frames.__getitem__,
            total_frames=len(frames),
        )
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["zero_best_ratio"], 1.0)
        self.assertFalse(report["frame_policy"]["automatic_offset_correction_applied"])

    def test_consistent_positive_offset_fails_without_applying_correction(self) -> None:
        frames = self.video_frames(7)
        rows = tuple(self.row(index, index, index - 1) for index in (1, 2, 4))
        references = {row.keyframe_order: frames[row.actual_frame_id + 1] for row in rows}
        report = CALIBRATION.evaluate_frame_offsets(
            rows,
            references,
            frames.__getitem__,
            total_frames=len(frames),
        )
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["superior_nonzero_offsets"][0]["offset"], 1)
        self.assertFalse(report["frame_policy"]["automatic_offset_correction_applied"])

    def test_invalid_boundary_offsets_are_reported_not_decoded(self) -> None:
        frames = self.video_frames(3)
        row = self.row(1, 0)
        report = CALIBRATION.evaluate_frame_offsets(
            [row],
            {1: frames[0]},
            frames.__getitem__,
            total_frames=3,
        )
        negative = report["samples"][0]["comparisons"][0]
        self.assertFalse(negative["valid"])
        self.assertEqual(negative["decoded_frame_id"], -1)

    def test_batch_manifest_supports_at_least_three_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "batch.json"
            path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "video_id": f"L21_V00{index}",
                                "video_path": f"/input/L21_V00{index}.mp4",
                                "mapping_csv": f"/input/L21_V00{index}.csv",
                                "keyframes": f"/input/L21_V00{index}",
                            }
                            for index in range(1, 4)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cases = CALIBRATION._load_batch_cases(path)
        self.assertEqual(len(cases), 3)


if __name__ == "__main__":
    unittest.main()
