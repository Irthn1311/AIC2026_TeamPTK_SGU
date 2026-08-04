from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from system_tai.common.schemas import FrameRecord
from system_tai.features.btc_clip_store import BTCClipFeatureStore


class BTCClipFeatureStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    @staticmethod
    def records(count: int) -> tuple[FrameRecord, ...]:
        return tuple(
            FrameRecord(
                video_id="L21_V001",
                actual_frame_id=row * 30,
                keyframe_order=row + 1,
                clip_row=row,
                pts_time=float(row),
                fps=30.0,
                mapping_version="v1",
                physical_row=row,
            )
            for row in range(count)
        )

    def save(self, matrix: np.ndarray) -> Path:
        path = self.root / "features.npy"
        np.save(path, matrix)
        return path

    def test_valid_matrix_exposes_stats_and_explicit_frame_access(self) -> None:
        path = self.save(np.eye(3, dtype=np.float16))
        store = BTCClipFeatureStore(normalization_tolerance=1e-3)
        stats = store.load(path, self.records(3), encoder_id="btc-test")
        self.assertEqual((stats.row_count, stats.dimension), (3, 3))
        self.assertEqual(stats.dtype, "float16")
        self.assertTrue(stats.appears_l2_normalized)
        self.assertEqual(store.frame_for_row(2).actual_frame_id, 60)
        self.assertNotEqual(store.frame_for_row(2).actual_frame_id, 2)

    def test_dimension_is_observed_not_hard_coded(self) -> None:
        path = self.save(np.ones((2, 7), dtype=np.float32))
        stats = BTCClipFeatureStore().load(path, self.records(2), encoder_id="btc-test")
        self.assertEqual(stats.dimension, 7)

    def test_non_two_dimensional_matrix_is_rejected(self) -> None:
        path = self.save(np.ones(3, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            BTCClipFeatureStore().load(path, self.records(3), encoder_id="btc-test")

    def test_nan_and_infinity_are_rejected(self) -> None:
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                path = self.save(np.array([[1.0, invalid]], dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "NaN"):
                    BTCClipFeatureStore().load(
                        path, self.records(1), encoder_id="btc-test"
                    )

    def test_row_count_mismatch_is_rejected(self) -> None:
        path = self.save(np.ones((2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "row-count mismatch"):
            BTCClipFeatureStore().load(path, self.records(1), encoder_id="btc-test")

    def test_expected_dimension_is_checked_when_configured(self) -> None:
        path = self.save(np.ones((2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            BTCClipFeatureStore().load(
                path,
                self.records(2),
                encoder_id="btc-test",
                expected_dimension=512,
            )

    def test_normalized_working_copy_does_not_modify_source(self) -> None:
        matrix = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float16)
        path = self.save(matrix)
        store = BTCClipFeatureStore()
        store.load(
            path,
            self.records(2),
            encoder_id="btc-test",
            normalize_working_copy=True,
        )
        np.testing.assert_array_equal(store.source_matrix, matrix)
        np.testing.assert_allclose(
            np.linalg.norm(store.working_matrix, axis=1), np.ones(2), atol=1e-6
        )
        self.assertEqual(store.working_matrix.dtype, np.float32)

    def test_zero_norm_row_cannot_be_normalized(self) -> None:
        path = self.save(np.array([[0.0, 0.0]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "zero norm"):
            BTCClipFeatureStore().load(
                path,
                self.records(1),
                encoder_id="btc-test",
                normalize_working_copy=True,
            )


if __name__ == "__main__":
    unittest.main()
