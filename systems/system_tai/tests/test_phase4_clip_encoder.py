from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from system_tai.refinement.clip_encoder import OpenAIClipRefinementEncoder


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeModel:
    def __init__(self):
        self.eval_calls = 0
        self.text_calls = 0
        self.image_calls = 0

    def eval(self):
        self.eval_calls += 1

    def encode_text(self, tokens):
        self.text_calls += 1
        rows = len(tokens.value)
        return FakeTensor(np.tile([[3.0, 4.0]], (rows, 1)))

    def encode_image(self, batch):
        self.image_calls += 1
        rows = len(batch.value)
        return FakeTensor(np.tile([[3.0, 4.0]], (rows, 1)))


class FakeClip:
    __file__ = "/fake/clip.py"
    __version__ = "test"

    def __init__(self, model):
        self.model = model
        self.load_calls = 0

    @staticmethod
    def available_models():
        return ["ViT-B/32"]

    def load(self, name, *, device, jit, download_root):
        assert (name, device, jit) == ("ViT-B/32", "cpu", False)
        assert download_root
        self.load_calls += 1
        return self.model, lambda image: FakeTensor([image.size[0], image.size[1]])

    @staticmethod
    def tokenize(texts, truncate):
        assert truncate is True
        return FakeTensor(np.ones((len(texts), 2)))


def test_refinement_encoder_model_once_text_image_batch_and_normalization(tmp_path) -> None:
    (tmp_path / "ViT-B-32.pt").touch()
    model = FakeModel()
    clip = FakeClip(model)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        no_grad=nullcontext,
        stack=lambda items: FakeTensor([item.value for item in items]),
    )
    image_api = SimpleNamespace(fromarray=lambda array: SimpleNamespace(size=(2, 2)))
    encoder = OpenAIClipRefinementEncoder(
        device="cpu",
        allow_model_download=False,
        cache_dir=tmp_path,
        expected_dimension=2,
        clip_module=clip,
        torch_module=torch,
        image_module=image_api,
    )
    texts = encoder.encode_texts(("vi", "en"))
    images = encoder.encode_images(
        tuple(np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(5)), batch_size=2
    )
    assert clip.load_calls == 1 and model.eval_calls == 1
    assert model.text_calls == 1 and model.image_calls == 3
    assert np.allclose(np.linalg.norm(texts, axis=1), 1)
    assert np.allclose(np.linalg.norm(images, axis=1), 1)


def test_refinement_encoder_no_implicit_download_and_explicit_cuda(tmp_path) -> None:
    clip = FakeClip(FakeModel())
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), no_grad=nullcontext)
    with pytest.raises(RuntimeError, match="not cached"):
        OpenAIClipRefinementEncoder(
            device="cpu",
            allow_model_download=False,
            cache_dir=tmp_path,
            expected_dimension=2,
            clip_module=clip,
            torch_module=torch,
        )
    (tmp_path / "ViT-B-32.pt").touch()
    with pytest.raises(RuntimeError, match="CUDA"):
        OpenAIClipRefinementEncoder(
            device="cuda",
            allow_model_download=False,
            cache_dir=tmp_path,
            expected_dimension=2,
            clip_module=clip,
            torch_module=torch,
        )
