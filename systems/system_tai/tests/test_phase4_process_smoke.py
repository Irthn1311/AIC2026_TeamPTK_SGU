from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from system_tai.data.corpus_discovery import CorpusManifest, DiscoveredVideo, _fingerprint

FAKE_TORCH = """
import numpy as np

class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def to(self, device): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value

class _Cuda:
    @staticmethod
    def is_available(): return False
cuda = _Cuda()

class _NoGrad:
    def __enter__(self): return self
    def __exit__(self, *args): return False
def no_grad(): return _NoGrad()
def stack(items): return Tensor(np.asarray([item.value for item in items]))
"""

FAKE_CLIP = """
import os
import numpy as np
import torch
__version__ = "synthetic"

def available_models(): return ["ViT-B/32"]
def tokenize(texts, truncate=True): return torch.Tensor(np.ones((len(texts), 1)))

class Model:
    def eval(self): return self
    def encode_text(self, tokens):
        rows = len(tokens.value)
        output = np.zeros((rows, 512), dtype=np.float32)
        output[:, 0] = 1
        return torch.Tensor(output)
    def encode_image(self, batch):
        frame_ids = batch.value[:, 0].astype(np.float32)
        output = np.zeros((len(frame_ids), 512), dtype=np.float32)
        output[:, 0] = 1 / (1 + np.abs(frame_ids - 55))
        output[:, 1] = 0.1
        return torch.Tensor(output)

def load(name, device, jit, download_root):
    marker = os.environ["FAKE_MODEL_LOAD_MARKER"]
    with open(marker, "a", encoding="utf-8") as stream: stream.write("load\\n")
    def preprocess(image):
        return torch.Tensor([np.asarray(image)[0, 0, 0]])
    return Model(), preprocess
"""

FAKE_CV2 = """
import numpy as np
CAP_PROP_FPS = 1
CAP_PROP_FRAME_COUNT = 2
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_POS_FRAMES = 5

class VideoCapture:
    def __init__(self, path): self.position = 0
    def isOpened(self): return True
    def get(self, key):
        if key == 5: return self.position
        return {1: 10, 2: 100, 3: 4, 4: 4}[key]
    def set(self, key, value): self.position = int(value); return True
    def read(self):
        value = self.position
        self.position += 1
        return True, np.full((4, 4, 3), value, dtype=np.uint8)
    def release(self): pass
"""


def _make_phase3_run(tmp_path: Path) -> Path:
    run = tmp_path / "phase3"
    run.mkdir()
    mapping = tmp_path / "L21_V001.csv"
    mapping.write_text("n,pts_time,fps,frame_idx\n1,5,10,50\n", encoding="utf-8")
    clip = tmp_path / "L21_V001.npy"
    np.save(clip, np.asarray([[1.0, 0.0]], dtype=np.float32))
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
    (keyframes / "1.jpg").touch()
    video_path = tmp_path / "L21_V001.mp4"
    video_path.touch()
    discovered = DiscoveredVideo(
        "L21_V001",
        mapping,
        clip,
        keyframes,
        video_path,
        1,
        2,
        mapping.stat().st_size,
        clip.stat().st_size,
        1,
    )
    CorpusManifest(tmp_path, tmp_path, _fingerprint((discovered,)), (discovered,)).write(
        run / "feature_manifest.json"
    )
    (run / "top100.jsonl").write_text(
        '{"query_id":"Q1","rank":1,"video_id":"L21_V001","frame_id":50}\n',
        encoding="utf-8",
    )
    (run / "candidates.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "query_id": "Q1",
                        "rank": 1,
                        "video_id": "L21_V001",
                        "frame_id": 50,
                        "fusion_score": 0.2,
                        "variant_hit_count": 1,
                        "best_individual_rank": 1,
                        "per_variant": [],
                        "clip_row_diagnostic": 0,
                        "keyframe_order_diagnostic": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "Q1",
                        "status": "SUCCESS",
                        "variants": [
                            {
                                "variant_id": "vi",
                                "text": "target",
                                "language": "vi",
                                "variant_type": "vietnamese_direct",
                                "weight": 1.0,
                            },
                            {
                                "variant_id": "en",
                                "text": "target translated",
                                "language": "en",
                                "variant_type": "english_translation",
                                "weight": 1.5,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return run


def test_process_level_synthetic_refine_cli(tmp_path: Path) -> None:
    fake_modules = tmp_path / "fake_modules"
    fake_modules.mkdir()
    (fake_modules / "torch.py").write_text(FAKE_TORCH, encoding="utf-8")
    (fake_modules / "clip.py").write_text(FAKE_CLIP, encoding="utf-8")
    (fake_modules / "cv2.py").write_text(FAKE_CV2, encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ViT-B-32.pt").touch()
    marker = tmp_path / "model_loads.txt"
    run = _make_phase3_run(tmp_path)
    output = tmp_path / "phase4"
    source_root = Path(__file__).parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(fake_modules), str(source_root), environment.get("PYTHONPATH", ""))
    )
    environment["FAKE_MODEL_LOAD_MARKER"] = str(marker)
    command = [
        sys.executable,
        "-m",
        "system_tai.kis.refine",
        "--run-directory",
        str(run),
        "--output-directory",
        str(output),
        "--top-candidates-to-refine",
        "1",
        "--window-before-seconds",
        "1",
        "--window-after-seconds",
        "1",
        "--coarse-stride-frames",
        "5",
        "--coarse-top-n",
        "1",
        "--fine-radius-frames",
        "2",
        "--fine-stride-frames",
        "1",
        "--image-batch-size",
        "2",
        "--max-decoded-frames-per-candidate",
        "30",
        "--output-top-k",
        "1",
        "--device",
        "cpu",
        "--clip-cache-dir",
        str(cache),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False, timeout=30
    )
    print("PROCESS_COMMAND:", subprocess.list2cmdline(command))
    print("PROCESS_STDOUT:", completed.stdout.strip())
    print("PROCESS_STDERR:", completed.stderr.strip())
    assert completed.returncode == 0, completed.stderr
    record = json.loads((output / "refined_top100.jsonl").read_text(encoding="utf-8").strip())
    assert record == {
        "query_id": "Q1",
        "rank": 1,
        "video_id": "L21_V001",
        "frame_id": 55,
    }
    validation = json.loads(
        (output / "refinement_validation_report.json").read_text(encoding="utf-8")
    )
    assert validation["valid"] is True
    assert marker.read_text(encoding="utf-8").splitlines() == ["load"]
    trace = json.loads((output / "refinement_trace.json").read_text(encoding="utf-8"))
    case = trace["queries"][0]["candidates"][0]
    assert case["window_start_frame"] == 40 and case["window_end_frame"] == 60
    assert case["candidate_frame_id"] == 50 and case["refined_frame_id"] == 55
    assert "embedding" not in json.dumps(trace) and '"image"' not in json.dumps(trace)
    assert not any(
        path.suffix.casefold() in {".mp4", ".avi", ".mkv", ".mov", ".webm", ".npy"}
        for path in output.rglob("*")
        if path.is_file()
    )
    print(
        "PROCESS_FILES:",
        sorted(
            str(path.relative_to(output)).replace("\\", "/")
            for path in output.rglob("*")
            if path.is_file()
        ),
    )
