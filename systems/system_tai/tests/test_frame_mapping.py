from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from system_tai.data.frame_mapping import FrameMappingLoader
from system_tai.data.video_catalog import BenchmarkVideoCatalog


class FrameMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        catalog_path = self.root / "catalog.csv"
        with catalog_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "video_id",
                    "video_path",
                    "fps",
                    "duration_seconds",
                    "total_frames",
                    "frame_index_base",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "video_id": "L21_V001",
                    "video_path": "L21_V001.mp4",
                    "fps": 30,
                    "duration_seconds": 10,
                    "total_frames": 300,
                    "frame_index_base": "zero_based",
                }
            )
        self.catalog = BenchmarkVideoCatalog()
        self.catalog.load(catalog_path)

    def write_mapping(
        self,
        rows: list[dict[str, object]],
        *,
        include_video_id: bool = False,
        include_clip_row: bool = False,
    ) -> Path:
        fields = ["n", "pts_time", "fps", "frame_idx"]
        if include_video_id:
            fields.insert(0, "video_id")
        if include_clip_row:
            fields.append("clip_row")
        path = self.root / "mapping.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_frame_idx_is_actual_frame_and_physical_row_is_clip_row(self) -> None:
        path = self.write_mapping(
            [
                {"n": 1, "pts_time": 0, "fps": 30, "frame_idx": 0},
                {"n": 2, "pts_time": 3, "fps": 30, "frame_idx": 90},
            ]
        )
        records = FrameMappingLoader().load(
            path,
            self.catalog,
            mapping_version="v1",
            video_id="L21_V001",
            use_physical_clip_rows=True,
        )
        self.assertEqual([record.actual_frame_id for record in records], [0, 90])
        self.assertEqual([record.clip_row for record in records], [0, 1])
        self.assertNotEqual(records[1].keyframe_order, records[1].actual_frame_id)

    def test_video_id_column_is_supported(self) -> None:
        path = self.write_mapping(
            [
                {
                    "video_id": "L21_V001",
                    "n": 1,
                    "pts_time": 0,
                    "fps": 30,
                    "frame_idx": 0,
                }
            ],
            include_video_id=True,
        )
        records = FrameMappingLoader().load(
            path,
            self.catalog,
            mapping_version="v1",
            use_physical_clip_rows=True,
        )
        self.assertEqual(records[0].video_id, "L21_V001")

    def test_missing_video_id_requires_explicit_argument(self) -> None:
        path = self.write_mapping(
            [{"n": 1, "pts_time": 0, "fps": 30, "frame_idx": 0}]
        )
        with self.assertRaisesRegex(ValueError, "explicit video_id"):
            FrameMappingLoader().load(path, self.catalog, mapping_version="v1")

    def test_out_of_bounds_actual_frame_is_rejected(self) -> None:
        path = self.write_mapping(
            [{"n": 1, "pts_time": 10, "fps": 30, "frame_idx": 300}]
        )
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            FrameMappingLoader().load(
                path,
                self.catalog,
                mapping_version="v1",
                video_id="L21_V001",
                use_physical_clip_rows=True,
            )

    def test_duplicate_keyframe_mapping_is_rejected(self) -> None:
        path = self.write_mapping(
            [
                {"n": 1, "pts_time": 0, "fps": 30, "frame_idx": 0},
                {"n": 1, "pts_time": 1, "fps": 30, "frame_idx": 30},
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate keyframe"):
            FrameMappingLoader().load(
                path,
                self.catalog,
                mapping_version="v1",
                video_id="L21_V001",
                use_physical_clip_rows=True,
            )

    def test_duplicate_actual_frame_mapping_is_rejected(self) -> None:
        path = self.write_mapping(
            [
                {"n": 1, "pts_time": 0, "fps": 30, "frame_idx": 0},
                {"n": 2, "pts_time": 0, "fps": 30, "frame_idx": 0},
            ]
        )
        with self.assertRaisesRegex(ValueError, "ambiguous duplicate"):
            FrameMappingLoader().load(
                path,
                self.catalog,
                mapping_version="v1",
                video_id="L21_V001",
                use_physical_clip_rows=True,
            )

    def test_n_minus_one_is_not_enforced_as_dataset_invariant(self) -> None:
        path = self.write_mapping(
            [{"n": 10, "pts_time": 1, "fps": 30, "frame_idx": 30}]
        )
        records = FrameMappingLoader().load(
            path,
            self.catalog,
            mapping_version="v1",
            video_id="L21_V001",
            use_physical_clip_rows=True,
        )
        self.assertEqual(records[0].clip_row, 0)
        self.assertNotEqual(records[0].clip_row, records[0].keyframe_order - 1)

    def test_explicit_clip_row_is_preserved(self) -> None:
        path = self.write_mapping(
            [{"n": 7, "pts_time": 1, "fps": 30, "frame_idx": 30, "clip_row": 0}],
            include_clip_row=True,
        )
        records = FrameMappingLoader().load(
            path,
            self.catalog,
            mapping_version="v1",
            video_id="L21_V001",
            use_physical_clip_rows=False,
        )
        self.assertEqual(records[0].clip_row, 0)

    def test_physical_clip_row_fallback_can_be_disabled(self) -> None:
        path = self.write_mapping(
            [{"n": 1, "pts_time": 0, "fps": 30, "frame_idx": 0}]
        )
        with self.assertRaisesRegex(ValueError, "physical-row fallback disabled"):
            FrameMappingLoader().load(
                path,
                self.catalog,
                mapping_version="v1",
                video_id="L21_V001",
                use_physical_clip_rows=False,
            )

    def test_mapping_fps_must_match_catalog(self) -> None:
        path = self.write_mapping(
            [{"n": 1, "pts_time": 0, "fps": 25, "frame_idx": 0}]
        )
        with self.assertRaisesRegex(ValueError, "FPS mismatch"):
            FrameMappingLoader().load(
                path,
                self.catalog,
                mapping_version="v1",
                video_id="L21_V001",
                use_physical_clip_rows=True,
            )

    def test_video_id_argument_must_match_explicit_column(self) -> None:
        path = self.write_mapping(
            [
                {
                    "video_id": "L21_V001",
                    "n": 1,
                    "pts_time": 0,
                    "fps": 30,
                    "frame_idx": 0,
                }
            ],
            include_video_id=True,
        )
        with self.assertRaisesRegex(ValueError, "video_id mismatch"):
            FrameMappingLoader().load(
                path,
                self.catalog,
                mapping_version="v1",
                video_id="L99_V999",
                use_physical_clip_rows=True,
            )

    def test_missing_required_column_is_rejected(self) -> None:
        path = self.root / "mapping.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps"])
            writer.writeheader()
            writer.writerow({"n": 1, "pts_time": 0, "fps": 30})
        with self.assertRaisesRegex(ValueError, "missing columns: frame_idx"):
            FrameMappingLoader().load(
                path,
                self.catalog,
                mapping_version="v1",
                video_id="L21_V001",
                use_physical_clip_rows=True,
            )

    def test_duplicate_explicit_clip_row_is_rejected(self) -> None:
        path = self.write_mapping(
            [
                {"n": 1, "pts_time": 0, "fps": 30, "frame_idx": 0, "clip_row": 0},
                {"n": 2, "pts_time": 1, "fps": 30, "frame_idx": 30, "clip_row": 0},
            ],
            include_clip_row=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate clip_row"):
            FrameMappingLoader().load(
                path,
                self.catalog,
                mapping_version="v1",
                video_id="L21_V001",
            )

    def test_non_finite_timestamp_is_rejected(self) -> None:
        path = self.write_mapping(
            [{"n": 1, "pts_time": "nan", "fps": 30, "frame_idx": 0}]
        )
        with self.assertRaisesRegex(ValueError, "pts_time must be non-negative"):
            FrameMappingLoader().load(
                path,
                self.catalog,
                mapping_version="v1",
                video_id="L21_V001",
                use_physical_clip_rows=True,
            )


if __name__ == "__main__":
    unittest.main()
