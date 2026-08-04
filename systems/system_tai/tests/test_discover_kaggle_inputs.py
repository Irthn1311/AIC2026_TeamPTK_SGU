from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "discover_kaggle_inputs.py"
SPEC = importlib.util.spec_from_file_location("system_tai_discover_kaggle_inputs", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load discovery CLI from {SCRIPT_PATH}")
DISCOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISCOVERY)


class KaggleInputDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.input_root = Path(self.temp_dir.name) / "input"
        self.input_root.mkdir()

    def create_dataset(self, slug: str = "private-runtime-slug") -> Path:
        root = self.input_root / slug
        paths = {
            "video": root / "Videos_L21_a" / "L21_V001.mp4",
            "mapping": root / "map-keyframes-aic25-b1" / "L21_V001.csv",
            "clip": root / "clip-features-32-aic25-b1" / "L21_V001.npy",
            "keyframe": root / "keyframes" / "L21" / "L21_V001" / "001.jpg",
            "media": root / "media-info-aic25-b1" / "L21_V001.json",
            "object": root / "objects-aic25-b1" / "L21_V001.json",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")
        return root

    def test_discovers_dynamic_slug_and_optional_artifacts_without_copying(self) -> None:
        dataset_root = self.create_dataset("slug-chosen-by-kaggle")
        report = DISCOVERY.discover(self.input_root, "L21_V001")
        self.assertEqual(report["status"], "DISCOVERED")
        self.assertEqual(Path(report["dataset_root"]), dataset_root.resolve())
        self.assertTrue(report["artifacts"]["original_video"].endswith("L21_V001.mp4"))
        self.assertEqual(report["artifacts"]["keyframe_image_count"], 1)
        self.assertIn("keyframes", report["matched_discovery_hints"])
        self.assertIsNotNone(report["artifacts"]["media_info"])
        self.assertIsNotNone(report["artifacts"]["object_json"])
        self.assertFalse(report["copied_artifacts"])

    def test_zero_matching_dataset_roots_fails_clearly(self) -> None:
        (self.input_root / "unrelated-dataset").mkdir()
        with self.assertRaisesRegex(DISCOVERY.DiscoveryError, "no Dataset_AIC2026"):
            DISCOVERY.discover(self.input_root, "L21_V001")

    def test_multiple_matching_dataset_roots_are_ambiguous(self) -> None:
        self.create_dataset("first-slug")
        self.create_dataset("second-slug")
        with self.assertRaisesRegex(DISCOVERY.DiscoveryError, "ambiguous dataset root"):
            DISCOVERY.discover(self.input_root, "L21_V001")

    def test_multiple_mapping_files_are_ambiguous(self) -> None:
        root = self.create_dataset()
        duplicate = root / "map-keyframes-aic25-b1" / "nested" / "L21_V001.csv"
        duplicate.parent.mkdir()
        duplicate.write_text("n,pts_time,fps,frame_idx\n", encoding="utf-8")
        with self.assertRaisesRegex(DISCOVERY.DiscoveryError, "ambiguous mapping_csv"):
            DISCOVERY.discover(self.input_root, "L21_V001")

    def test_missing_required_clip_array_fails(self) -> None:
        root = self.create_dataset()
        (root / "clip-features-32-aic25-b1" / "L21_V001.npy").unlink()
        with self.assertRaisesRegex(DISCOVERY.DiscoveryError, "missing required clip_npy"):
            DISCOVERY.discover(self.input_root, "L21_V001")


if __name__ == "__main__":
    unittest.main()
