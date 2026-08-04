from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from system_tai.common.schemas import FrameIndexBase
from system_tai.data.video_catalog import BenchmarkVideoCatalog


class BenchmarkVideoCatalogAcceptanceTests(unittest.TestCase):
    fieldnames = [
        "video_id",
        "video_path",
        "fps",
        "duration_seconds",
        "total_frames",
        "frame_index_base",
    ]

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def write_catalog(self, rows: list[dict[str, object]]) -> Path:
        path = self.root / "catalog.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def valid_row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "video_id": "L21_V001",
            "video_path": "videos/L21_V001.mp4",
            "fps": 30,
            "duration_seconds": 10,
            "total_frames": 300,
            "frame_index_base": "zero_based",
        }
        row.update(overrides)
        return row

    def test_duplicate_video_id_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row(), self.valid_row()])
        with self.assertRaisesRegex(ValueError, "duplicate video_id"):
            BenchmarkVideoCatalog().load(path)

    def test_missing_file_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row()])
        with self.assertRaises(FileNotFoundError):
            BenchmarkVideoCatalog(strict_paths=True).load(path)

    def test_invalid_fps_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row(fps=0)])
        with self.assertRaisesRegex(ValueError, "fps must be positive"):
            BenchmarkVideoCatalog().load(path)

    def test_non_positive_total_frames_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row(total_frames=0)])
        with self.assertRaisesRegex(ValueError, "total_frames must be positive"):
            BenchmarkVideoCatalog().load(path)

    def test_non_positive_duration_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row(duration_seconds=0)])
        with self.assertRaisesRegex(ValueError, "duration_seconds must be positive"):
            BenchmarkVideoCatalog().load(path)

    def test_duration_frame_count_inconsistency_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row(duration_seconds=100)])
        catalog = BenchmarkVideoCatalog(duration_tolerance_seconds=0.1)
        with self.assertRaisesRegex(ValueError, "inconsistency"):
            catalog.load(path)

    def test_unknown_frame_index_base_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row(frame_index_base="unknown")])
        with self.assertRaisesRegex(ValueError, "frame_index_base"):
            BenchmarkVideoCatalog().load(path)

    def test_malformed_video_id_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row(video_id="bad/video")])
        with self.assertRaisesRegex(ValueError, "malformed video_id"):
            BenchmarkVideoCatalog().load(path)

    def test_empty_video_id_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row(video_id="")])
        with self.assertRaisesRegex(ValueError, "malformed video_id"):
            BenchmarkVideoCatalog().load(path)

    def test_non_finite_fps_is_rejected(self) -> None:
        path = self.write_catalog([self.valid_row(fps="nan")])
        with self.assertRaisesRegex(ValueError, "fps must be positive"):
            BenchmarkVideoCatalog().load(path)

    def test_zero_based_bounds_are_enforced(self) -> None:
        path = self.write_catalog([self.valid_row()])
        catalog = BenchmarkVideoCatalog()
        records = catalog.load(path)
        self.assertEqual(records[0].frame_index_base, FrameIndexBase.ZERO)
        catalog.validate_actual_frame_id("L21_V001", 0)
        catalog.validate_actual_frame_id("L21_V001", 299)
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            catalog.validate_actual_frame_id("L21_V001", 300)

    def test_one_based_bounds_are_enforced_without_assumption(self) -> None:
        path = self.write_catalog([self.valid_row(frame_index_base="one_based")])
        catalog = BenchmarkVideoCatalog()
        catalog.load(path)
        catalog.validate_actual_frame_id("L21_V001", 1)
        catalog.validate_actual_frame_id("L21_V001", 300)
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            catalog.validate_actual_frame_id("L21_V001", 0)


if __name__ == "__main__":
    unittest.main()
