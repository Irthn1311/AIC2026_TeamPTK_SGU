import numpy as np
import pytest

from triage_eg.retrieval.numpy_index import NumPyFlatCosineIndex


def test_numpy_cosine_index_returns_best_match():
    index = NumPyFlatCosineIndex()
    index.build(np.array([[1, 0], [0, 1]], dtype=np.float32), ["a", "b"])
    scores, ids = index.search(np.array([[0.9, 0.1]], dtype=np.float32), 2)
    assert ids.tolist() == [["a", "b"]]
    assert scores[0, 0] > scores[0, 1]


def test_numpy_index_validates_dimension():
    index = NumPyFlatCosineIndex()
    index.build(np.ones((2, 3), dtype=np.float32), ["a", "b"])
    with pytest.raises(ValueError, match="shape"):
        index.search(np.ones((1, 2), dtype=np.float32), 1)
