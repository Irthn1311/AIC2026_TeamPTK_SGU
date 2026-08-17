from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from triage_eg.diagnostics.sca1_siglip2_complementarity import (
    Siglip2ExactBackend,
    Siglip2GroundingPipeline,
    l2_normalize,
)
from triage_eg.diagnostics.sca1_siglip2_complementarity import backend as backend_module


def test_s1_scores_cross_module_without_mutating_production_runtime(monkeypatch) -> None:
    vectors = l2_normalize(np.eye(768, dtype=np.float32)[:3]).astype(np.float16)
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1).astype(np.float32)
    monkeypatch.setattr(
        backend_module,
        "validate_siglip2_index",
        lambda *args, **kwargs: {"manifest": {}, "vectors": vectors, "norms": norms},
    )
    grounding_backend = Siglip2ExactBackend("synthetic")
    production_backend, production_encoder = object(), object()
    runtime = SimpleNamespace(
        backend=production_backend,
        encoder=production_encoder,
        encode_requests=lambda requests: SimpleNamespace(
            encodings=({"clip_input_text": "frozen opus english"},)
        ),
    )
    pipeline = object.__new__(Siglip2GroundingPipeline)
    pipeline.runtime = runtime
    pipeline.grounding_backend = grounding_backend
    pipeline.grounding_encoder = SimpleNamespace(
        encode_text=lambda texts: l2_normalize(np.ones((len(texts), 768), dtype=np.float32))
    )
    pipeline._encoded_text = {}
    pipeline._score_cache = {}
    pipeline.text_identity_records = {}

    vector, scores, provenance = pipeline._scores("nguon tieng Viet", "vi", "CB-KIS-001__grounding")

    assert vector.shape == (768,)
    assert scores.shape == (3,)
    assert provenance["sca1_clip_input_text"] == "frozen opus english"
    assert runtime.backend is production_backend
    assert runtime.encoder is production_encoder
    assert "_frame_embedding" not in Siglip2GroundingPipeline.__dict__
    assert "_answer_embeddings" not in Siglip2GroundingPipeline.__dict__
