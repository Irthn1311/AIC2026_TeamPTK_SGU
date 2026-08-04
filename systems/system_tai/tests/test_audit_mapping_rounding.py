from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_mapping_rounding.py"
SPEC = importlib.util.spec_from_file_location("system_tai_audit_mapping_rounding", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load mapping-rounding audit from {SCRIPT_PATH}")
ROUNDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUNDING)


class MappingRoundingAuditTests(unittest.TestCase):
    @staticmethod
    def diagnostic(product: str, *, fps: str = "1", frame_idx: int = 30):
        pts_time = Decimal(product) / Decimal(fps)
        return ROUNDING.calculate_rounding_diagnostic(
            keyframe_order=1,
            pts_time=pts_time,
            fps=Decimal(fps),
            frame_idx=frame_idx,
            physical_row=0,
        )

    def test_exact_integer_timestamp(self) -> None:
        row = self.diagnostic("30")
        self.assertEqual((row["decimal_floor"], row["decimal_round_half_up"]), (30, 30))
        self.assertTrue(row["matches_decimal_floor"])
        self.assertTrue(row["matches_binary_float_truncation"])
        self.assertTrue(row["matches_decimal_nearest"])

    def test_fraction_001_above_integer(self) -> None:
        row = self.diagnostic("30.001")
        self.assertEqual((row["decimal_floor"], row["decimal_round_half_up"]), (30, 30))
        self.assertEqual(row["decimal_ceil"], 31)

    def test_fraction_049_rounds_down(self) -> None:
        row = self.diagnostic("30.49")
        self.assertEqual(row["decimal_round_half_up"], 30)
        self.assertEqual(row["decimal_nearest_minus_frame_idx"], 0)

    def test_fraction_050_rounds_up(self) -> None:
        row = self.diagnostic("30.50")
        self.assertEqual(row["decimal_floor"], 30)
        self.assertEqual(row["decimal_round_half_up"], 31)
        self.assertEqual(row["decimal_nearest_minus_frame_idx"], 1)

    def test_fraction_099_rounds_up(self) -> None:
        row = self.diagnostic("30.99")
        self.assertEqual((row["decimal_floor"], row["decimal_round_half_up"]), (30, 31))

    def test_non_30_fps_uses_decimal_product(self) -> None:
        row = self.diagnostic("25.50", fps="25", frame_idx=25)
        self.assertEqual(row["pts_time"], "1.02")
        self.assertEqual(row["decimal_round_half_up"], 26)
        self.assertEqual(row["decimal_nearest_minus_frame_idx"], 1)

    def test_binary_float_truncation_regressions_match_verified_rows(self) -> None:
        cases = (
            (63, "260.4", "30", 7811, "7812"),
            (248, "1024.1", "30", 30722, "30723"),
            (249, "1031.1", "30", 30932, "30933"),
            (254, "1058.6", "30", 31757, "31758"),
        )
        for order, pts_time, fps, frame_idx, decimal_product in cases:
            with self.subTest(keyframe_order=order):
                row = ROUNDING.calculate_rounding_diagnostic(
                    keyframe_order=order,
                    pts_time=pts_time,
                    fps=fps,
                    frame_idx=frame_idx,
                    physical_row=0,
                )
                self.assertEqual(
                    Decimal(row["decimal_exact_product"]), Decimal(decimal_product)
                )
                self.assertEqual(row["frame_idx_minus_decimal_floor"], -1)
                self.assertEqual(row["frame_idx_minus_binary_float_truncation"], 0)
                self.assertEqual(row["decimal_nearest_minus_frame_idx"], 1)
                self.assertFalse(row["matches_decimal_floor"])
                self.assertTrue(row["matches_binary_float_truncation"])
                self.assertFalse(row["numeric_rule_unresolved"])

    def test_summary_reports_distributions_and_observed_floor_rule(self) -> None:
        rows = (
            self.diagnostic("30.00"),
            {
                **self.diagnostic("40.99", frame_idx=40),
                "physical_row": 1,
                "keyframe_order": 2,
            },
        )
        report = ROUNDING.summarize_rounding(rows, video_id="L21_V001")
        self.assertEqual(report["frame_idx_equals_decimal_floor"]["ratio"], 1.0)
        self.assertEqual(
            report["decimal_nearest_minus_frame_idx_distribution"], {"0": 1, "1": 1}
        )
        self.assertEqual(report["observed_rule_summary"], "DECIMAL_FLOOR_OBSERVED")
        self.assertFalse(
            report["coordinate_policy"]["rounding_diagnostics_modify_shared_frame_id"]
        )

    def test_binary_float_match_is_not_an_unresolved_numeric_rule(self) -> None:
        rows = (
            self.diagnostic("30.00"),
            ROUNDING.calculate_rounding_diagnostic(
                keyframe_order=2,
                pts_time="260.4",
                fps="30",
                frame_idx=7811,
                physical_row=1,
            ),
        )
        report = ROUNDING.summarize_rounding(rows, video_id="L21_V001")
        self.assertEqual(report["status"], "BINARY_FLOAT_TRUNCATION_OBSERVED")
        self.assertEqual(report["numeric_rule_unresolved_row_count"], 0)
        self.assertEqual(
            report["frame_idx_minus_decimal_floor_distribution"], {"-1": 1, "0": 1}
        )
        self.assertEqual(
            report["frame_idx_minus_binary_float_truncation_distribution"], {"0": 2}
        )

    def test_malformed_and_non_finite_values_are_rejected(self) -> None:
        for value in ("bad", "NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "mapping.csv"
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=["n", "pts_time", "fps", "frame_idx"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {"n": 1, "pts_time": value, "fps": "30", "frame_idx": 0}
                    )
                with self.assertRaisesRegex(ValueError, "pts_time"):
                    ROUNDING.load_rounding_diagnostics(path)


if __name__ == "__main__":
    unittest.main()
