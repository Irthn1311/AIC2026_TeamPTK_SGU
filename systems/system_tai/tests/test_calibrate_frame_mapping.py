from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    def row(
        order: int,
        frame: int,
        physical_row: int = 0,
        *,
        pts_time: str | float | None = None,
        fps: str | float = 1,
    ):
        return CALIBRATION.CalibrationMappingRow(
            keyframe_order=order,
            actual_frame_id=frame,
            physical_row=physical_row,
            pts_time=float(frame) if pts_time is None else pts_time,
            fps=fps,
        )

    @staticmethod
    def video_frames(count: int) -> dict[int, np.ndarray]:
        return {frame: np.full((4, 5, 3), frame * 20, dtype=np.uint8) for frame in range(count)}

    @staticmethod
    def comparison(offset: int, similarity: float) -> dict:
        return {
            "offset": offset,
            "decoded_frame_id": offset + 10,
            "valid": True,
            "scores": {"similarity": similarity},
        }

    def test_verified_small_margins_are_ambiguous(self) -> None:
        for margin in (0.000007, 0.000014, 0.000017):
            with self.subTest(margin=margin):
                decision = CALIBRATION.classify_visual_decision(
                    [
                        self.comparison(0, 0.9),
                        self.comparison(1, 0.9 - margin),
                        self.comparison(-1, 0.5),
                    ],
                    superiority_margin=0.0001,
                )
                self.assertEqual(decision["visual_decision_status"], "AMBIGUOUS")
                self.assertAlmostEqual(decision["best_minus_second_margin"], margin)
                self.assertEqual(decision["tied_offset_candidates"], [0, 1])

    def test_margin_exactly_at_threshold_is_ambiguous(self) -> None:
        decision = CALIBRATION.classify_visual_decision(
            [self.comparison(0, 0.9), self.comparison(1, 0.8999)],
            superiority_margin=0.0001,
        )
        self.assertEqual(decision["visual_decision_status"], "AMBIGUOUS")

    def test_margin_clearly_above_threshold_is_decisive(self) -> None:
        decision = CALIBRATION.classify_visual_decision(
            [self.comparison(0, 0.9), self.comparison(1, 0.8998)],
            superiority_margin=0.0001,
        )
        self.assertEqual(decision["visual_decision_status"], "DECISIVE")
        self.assertEqual(decision["tied_offset_candidates"], [0])

    def test_only_ambiguous_contradiction_is_explained_case(self) -> None:
        frames = self.video_frames(5)
        row = self.row(1, 2)

        def fake_compare(_reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
            frame_id = int(candidate[0, 0, 0] // 20)
            similarity = {1: 0.5, 2: 0.899993, 3: 0.9}[frame_id]
            return {"similarity": similarity}

        with mock.patch.object(CALIBRATION, "compare_images", side_effect=fake_compare):
            report = CALIBRATION.evaluate_frame_offsets(
                [row],
                {1: frames[2]},
                frames.__getitem__,
                total_frames=len(frames),
                superiority_margin=0.0001,
            )
        self.assertEqual(report["status"], "VISUAL_ALIGNMENT_EXPLAINED")
        self.assertEqual(report["decisive_sample_count"], 0)
        self.assertEqual(report["ambiguous_sample_count"], 1)
        self.assertEqual(report["explained_decisive_ratio"], 1.0)
        self.assertEqual(report["contradictory_decisive_sample_count"], 0)

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
        self.assertEqual(report["status"], "VISUAL_ALIGNMENT_EXPLAINED")
        self.assertEqual(
            report["mapping_coordinate_validation"]["status"], "MAPPING_POLICY_PASSED"
        )
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
        self.assertEqual(report["status"], "MAPPING_POLICY_FAILED")
        self.assertEqual(report["systematic_unexplained_offsets"][0]["offset"], 1)
        self.assertFalse(report["frame_policy"]["automatic_offset_correction_applied"])

    def test_positive_offset_explained_by_round_half_up_is_not_failure(self) -> None:
        frames = self.video_frames(7)
        rows = tuple(
            self.row(
                index,
                index,
                index - 1,
                pts_time=f"{index}.99",
            )
            for index in (1, 2, 4)
        )
        references = {row.keyframe_order: frames[row.actual_frame_id + 1] for row in rows}
        report = CALIBRATION.evaluate_frame_offsets(
            rows,
            references,
            frames.__getitem__,
            total_frames=len(frames),
        )
        self.assertEqual(report["status"], "VISUAL_ALIGNMENT_EXPLAINED")
        self.assertEqual(
            report["visual_artifact_agreement"]["visual_best_offset_distribution"]["1"],
            3,
        )
        self.assertTrue(
            all(
                sample["visual_best_matches_round_half_up_prediction"]
                for sample in report["samples"]
            )
        )

    def test_non_systematic_unexplained_offset_is_inconclusive(self) -> None:
        frames = self.video_frames(7)
        rows = tuple(self.row(index, index, index - 1) for index in (1, 2, 4))
        references = {
            rows[0].keyframe_order: frames[rows[0].actual_frame_id],
            rows[1].keyframe_order: frames[rows[1].actual_frame_id],
            rows[2].keyframe_order: frames[rows[2].actual_frame_id + 1],
        }
        report = CALIBRATION.evaluate_frame_offsets(
            rows,
            references,
            frames.__getitem__,
            total_frames=len(frames),
        )
        self.assertEqual(report["status"], "VISUAL_ALIGNMENT_INCONCLUSIVE")

    def test_random_and_sequential_decoder_disagreement_fails(self) -> None:
        frames = self.video_frames(6)
        rows = (self.row(1, 2),)
        references = {1: frames[2]}

        def disagreeing_decoder(frame_id: int) -> np.ndarray:
            return np.full_like(frames[frame_id], 255)

        report = CALIBRATION.evaluate_frame_offsets(
            rows,
            references,
            frames.__getitem__,
            sequential_decode_frame=disagreeing_decoder,
            total_frames=len(frames),
        )
        self.assertEqual(report["status"], "MAPPING_POLICY_FAILED")
        self.assertEqual(report["decoder_agreement"]["status"], "DISAGREEMENT")

    def test_out_of_bounds_frame_idx_fails_mapping_policy(self) -> None:
        report = CALIBRATION.validate_mapping_coordinates(
            [self.row(1, 6)], total_frames=6
        )
        self.assertEqual(report["status"], "MAPPING_POLICY_FAILED")
        self.assertEqual(report["out_of_bounds_rows"][0]["actual_frame_id"], 6)

    def test_decimal_floor_difference_does_not_fail_mapping_policy(self) -> None:
        row = self.row(63, 7811, pts_time="260.4", fps="30")
        numeric = CALIBRATION.timestamp_rounding_diagnostic(row)
        mapping = CALIBRATION.validate_mapping_coordinates([row], total_frames=8000)
        self.assertEqual(numeric["frame_idx_minus_decimal_floor"], -1)
        self.assertEqual(numeric["frame_idx_minus_binary_float_truncation"], 0)
        self.assertEqual(numeric["predicted_visual_offset"], 1)
        self.assertEqual(mapping["status"], "MAPPING_POLICY_PASSED")
        self.assertFalse(numeric["numeric_rule_modifies_mapping_validity"])

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
