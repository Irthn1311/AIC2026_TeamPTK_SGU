"""Opt-in raw-video exact-frame refinement for Textual KIS."""

from system_tai.refinement.models import (
    CandidateFailurePolicy,
    MissingRawVideoPolicy,
    RefinementConfig,
    RefinementStatus,
)

__all__ = [
    "CandidateFailurePolicy",
    "MissingRawVideoPolicy",
    "RefinementConfig",
    "RefinementStatus",
]
