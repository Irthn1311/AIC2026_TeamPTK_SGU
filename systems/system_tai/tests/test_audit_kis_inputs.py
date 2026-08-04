from __future__ import annotations

import csv
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_kis_inputs.py"
SPEC = importlib.util.spec_from_file_location("system_tai_audit_kis_inputs", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load audit CLI from {SCRIPT_PATH}")
AUDIT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT_MODULE)


class AuditKISInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.catalog = self.root / "catalog.csv"
        with self.catalog.open("w", encoding="utf-8", newline="") as stream:
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
        self.mapping = self.root / "mapping.csv"
        with self.mapping.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["n", "pts_time", "fps", "frame_idx"]
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"n": 1, "pts_time": 0, "fps": 30, "frame_idx": 0},
                    {"n": 2, "pts_time": 3, "fps": 30, "frame_idx": 90},
                ]
            )
        self.features = self.root / "features.npy"
        np.save(self.features, np.eye(2, dtype=np.float32))

    def run_cli(self, *extra: str) -> tuple[int, dict[str, object]]:
        args = [
            "--video-catalog",
            str(self.catalog),
            "--video-id",
            "L21_V001",
            "--mapping-csv",
            str(self.mapping),
            "--clip-npy",
            str(self.features),
            *extra,
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = AUDIT_MODULE.main(args)
        return return_code, json.loads(output.getvalue())

    def test_valid_inputs_produce_structured_report(self) -> None:
        return_code, report = self.run_cli("--expected-dimension", "2")
        self.assertEqual(return_code, 0)
        self.assertTrue(report["valid"])
        self.assertEqual(report["mapping"]["row_count"], 2)
        self.assertTrue(report["mapping"]["feature_row_alignment_validated"])
        self.assertEqual(report["features"]["shape"], [2, 2])
        self.assertEqual(report["features"]["mapping_coverage"]["ratio"], 1.0)

    def test_invalid_expected_dimension_returns_nonzero(self) -> None:
        return_code, report = self.run_cli("--expected-dimension", "512")
        self.assertEqual(return_code, 1)
        self.assertFalse(report["valid"])
        self.assertEqual(report["errors"][0]["stage"], "features")
        self.assertFalse(report["mapping"]["feature_row_alignment_validated"])


if __name__ == "__main__":
    unittest.main()
