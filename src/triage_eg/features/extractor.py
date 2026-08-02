"""Feature extraction orchestration."""

from pathlib import Path

import numpy as np

from triage_eg.common.schemas import FeatureRecord, FrameRecord
from triage_eg.features.interfaces import MultimodalEncoder


def extract_frame_features(
    frames: list[FrameRecord],
    encoder: MultimodalEncoder,
    artifact_path: str | Path,
) -> tuple[np.ndarray, list[FeatureRecord]]:
    """Encode frames and build their row-level feature manifest."""

    vectors = encoder.encode_frames(frames)
    expected_shape = (len(frames), encoder.dimension)
    if vectors.shape != expected_shape:
        raise ValueError(f"Encoder returned {vectors.shape}; expected {expected_shape}")
    records = [
        FeatureRecord(
            feature_uid=f"{encoder.model_name}:{encoder.model_version}:{frame.frame_uid}",
            frame_uid=frame.frame_uid,
            model_name=encoder.model_name,
            model_version=encoder.model_version,
            feature_row=row,
            dimension=encoder.dimension,
            normalized=True,
            artifact_path=str(Path(artifact_path) / "vectors.npy"),
        )
        for row, frame in enumerate(frames)
    ]
    return vectors, records
