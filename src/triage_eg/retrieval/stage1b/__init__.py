"""Stage 1B bounded CLIP encoder compatibility validation."""

from triage_eg.retrieval.stage1b.contracts import (
    CandidateContract,
    CompatibilityGate,
    Stage1BConfig,
)
from triage_eg.retrieval.stage1b.pipeline import Stage1BResult, run_stage1b
from triage_eg.retrieval.stage1b.writers import create_stage1b_report_bundle

__all__ = [
    "CandidateContract",
    "CompatibilityGate",
    "Stage1BConfig",
    "Stage1BResult",
    "create_stage1b_report_bundle",
    "run_stage1b",
]
