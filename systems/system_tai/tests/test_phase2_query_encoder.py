from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from system_tai.features.query_encoder import OpenAIClipTextEncoder, TextEncoderUnavailable


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
    context_length = 77

    def __init__(self):
        self.eval_calls = 0
        self.encode_calls = 0

    def eval(self):
        self.eval_calls += 1

    def encode_text(self, _tokens):
        self.encode_calls += 1
        output = np.zeros((1, 512), dtype=np.float32)
        output[0, :2] = [3.0, 4.0]
        return FakeTensor(output)


class FakeClip:
    __file__ = "/fake/clip.py"
    __version__ = "test"

    def __init__(self, model):
        self.model = model
        self.load_calls = 0

    @staticmethod
    def available_models():
        return ["RN50", "ViT-B/32"]

    def load(self, name, *, device, jit, download_root):
        assert (name, device, jit) == ("ViT-B/32", "cpu", False)
        assert download_root
        self.load_calls += 1
        return self.model, "fake-preprocess"

    @staticmethod
    def tokenize(texts, truncate):
        assert texts and truncate is True
        return FakeTensor([[1]])


def test_official_adapter_uses_only_public_api_and_reuses_model(tmp_path) -> None:
    (tmp_path / "ViT-B-32.pt").touch()
    model = FakeModel()
    clip = FakeClip(model)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), no_grad=nullcontext)
    encoder = OpenAIClipTextEncoder(cache_dir=tmp_path, clip_module=clip, torch_module=torch)
    first = encoder.encode("xe máy trong mưa")
    second = encoder.encode("motorcycle in rain")
    assert clip.load_calls == 1
    assert model.eval_calls == 1
    assert model.encode_calls == 2
    assert np.allclose(first[:2], [0.6, 0.8])
    assert np.allclose(second, first)
    assert encoder.identifiers["model"] == "ViT-B/32"
    assert encoder.dimension == 512


def test_adapter_does_not_download_or_fall_back_silently(tmp_path) -> None:
    clip = FakeClip(FakeModel())
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), no_grad=nullcontext)
    with pytest.raises(TextEncoderUnavailable, match="weights are not cached"):
        OpenAIClipTextEncoder(cache_dir=tmp_path, clip_module=clip, torch_module=torch)


def test_adapter_rejects_invalid_encoded_vector(tmp_path) -> None:
    (tmp_path / "ViT-B-32.pt").touch()
    model = FakeModel()
    model.encode_text = lambda _tokens: FakeTensor(np.zeros((1, 512), dtype=np.float32))
    clip = FakeClip(model)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), no_grad=nullcontext)
    encoder = OpenAIClipTextEncoder(cache_dir=tmp_path, clip_module=clip, torch_module=torch)
    with pytest.raises(ValueError, match="non-zero norm"):
        encoder.encode("query")
