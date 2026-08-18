"""BCF-1 protected late rank fusion diagnostic."""

from .attribution import fusion_diagnostics, paired_evaluation
from .contracts import BCF1Settings
from .fusion import candidate_key, fuse_predictions, fuse_query, normalize_qa_answer
from .preparation import (
    BCF1Preparation,
    load_post_gt_design_sanity,
    load_preparation_freeze,
)
from .report import formal_report
from .runner import (
    create_bundle,
    evaluate_post_gt,
    fuse_l21,
    promotion_decision,
    reproduce_cross,
    run_l21_arm,
    validate_all_hashes_before_gt,
    validate_frozen_index,
    write_manifests,
)

__all__ = [
    "BCF1Preparation",
    "BCF1Settings",
    "candidate_key",
    "create_bundle",
    "evaluate_post_gt",
    "formal_report",
    "fuse_l21",
    "fuse_predictions",
    "fuse_query",
    "fusion_diagnostics",
    "load_post_gt_design_sanity",
    "load_preparation_freeze",
    "normalize_qa_answer",
    "paired_evaluation",
    "promotion_decision",
    "reproduce_cross",
    "run_l21_arm",
    "validate_all_hashes_before_gt",
    "validate_frozen_index",
    "write_manifests",
]
