from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "identify_btc_clip_pipeline.py"
SPEC = importlib.util.spec_from_file_location("system_tai_identify_btc_clip", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load CLIP identification CLI from {SCRIPT_PATH}")
IDENTIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTIFY)


class BTCClipPipelineIdentificationTests(unittest.TestCase):
    def test_exact_embeddings_have_perfect_row_and_self_match_metrics(self) -> None:
        matrix = np.eye(4, dtype=np.float32)
        metrics = IDENTIFY.compute_alignment_metrics(matrix, matrix.copy())
        self.assertAlmostEqual(metrics["row_wise_cosine"]["mean"], 1.0)
        self.assertAlmostEqual(metrics["normalized_l2_distance"]["mean"], 0.0)
        self.assertAlmostEqual(metrics["maximum_absolute_difference"], 0.0)
        self.assertEqual(metrics["self_match_top1_accuracy"], 1.0)
        self.assertEqual(metrics["mean_self_match_rank"], 1.0)

    def test_permuted_embeddings_fail_self_match(self) -> None:
        original = np.eye(3, dtype=np.float32)
        candidate = original[[1, 2, 0]]
        metrics = IDENTIFY.compute_alignment_metrics(original, candidate)
        self.assertEqual(metrics["self_match_top1_accuracy"], 0.0)
        self.assertGreater(metrics["mean_self_match_rank"], 1.0)

    def test_dimension_match_is_required_but_not_sufficient(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            IDENTIFY.compute_alignment_metrics(
                np.ones((2, 4), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32),
            )

    def test_unavailable_optional_backend_is_skipped(self) -> None:
        def unavailable(_paths, *, allow_download):
            del allow_download
            raise IDENTIFY.BackendUnavailable("weights unavailable")

        report = IDENTIFY.run_candidate_backend(
            "optional-test",
            [],
            np.eye(2, dtype=np.float32),
            allow_download=False,
            encoder=unavailable,
        )
        self.assertEqual(report["status"], "SKIPPED")
        self.assertIn("weights unavailable", report["reason"])

    def test_non_monotonic_keyframe_order_rejects_feature_row_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mapping.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["n", "pts_time", "fps", "frame_idx"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"n": 2, "pts_time": 0, "fps": 30, "frame_idx": 0},
                        {"n": 1, "pts_time": 1, "fps": 30, "frame_idx": 30},
                    ]
                )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                IDENTIFY._load_mapping(path)

    def test_identification_requires_reproduction_across_three_videos(self) -> None:
        metrics = IDENTIFY.compute_alignment_metrics(
            np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)
        )
        cases = [
            {
                "video_id": f"L21_V00{index}",
                "feature_row_mapping_validated": True,
                "backends": {
                    "openai_clip": {
                        "status": "MEASURED",
                        "metrics": metrics,
                    }
                },
            }
            for index in range(1, 4)
        ]
        two_video = IDENTIFY.classify_backends(cases[:2], minimum_videos=3)
        three_video = IDENTIFY.classify_backends(cases, minimum_videos=3)
        self.assertEqual(two_video["openai_clip"]["status"], "UNVERIFIED")
        self.assertEqual(three_video["openai_clip"]["status"], "IDENTIFIED")

    def test_unvalidated_feature_row_mapping_cannot_be_identified(self) -> None:
        metrics = IDENTIFY.compute_alignment_metrics(
            np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)
        )
        cases = [
            {
                "video_id": f"L21_V00{index}",
                "feature_row_mapping_validated": index != 2,
                "backends": {
                    "openai_clip": {"status": "MEASURED", "metrics": metrics}
                },
            }
            for index in range(1, 4)
        ]
        summary = IDENTIFY.classify_backends(cases, minimum_videos=3)
        self.assertEqual(summary["openai_clip"]["status"], "UNVERIFIED")
        self.assertFalse(summary["openai_clip"]["all_feature_row_mappings_validated"])


if __name__ == "__main__":
    unittest.main()
