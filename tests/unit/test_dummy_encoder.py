import numpy as np

from triage_eg.common.schemas import FrameRecord
from triage_eg.features.dummy_encoder import DeterministicDummyEncoder


def test_dummy_encoder_is_deterministic_and_normalized():
    frame = FrameRecord("f", "v", 1, 40, "x.jpg", "SHOT_CENTER", None, "v1", "e1")
    encoder = DeterministicDummyEncoder(16)
    first = encoder.encode_frames([frame])
    second = encoder.encode_frames([frame])
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
