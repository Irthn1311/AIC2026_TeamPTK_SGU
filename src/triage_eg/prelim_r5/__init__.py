"""Conservative Prelim R5 query-ensemble policy and final evaluation helpers."""

from .evidence import fuse_asr_multiview, qa_evidence_from_asr
from .fusion import R5Settings, build_r5_query_candidates, fuse_multiview_branch
from .qa import build_deterministic_qa_rows
from .runner import (
    evaluate_frozen_arms,
    finalize_pre_gt_predictions,
    select_production_policy,
    write_r5_artifacts,
)
from .views import VIEW_NAMES, build_query_views, materialize_view_queries

__all__ = [
    "R5Settings",
    "VIEW_NAMES",
    "build_deterministic_qa_rows",
    "build_query_views",
    "build_r5_query_candidates",
    "evaluate_frozen_arms",
    "finalize_pre_gt_predictions",
    "fuse_asr_multiview",
    "fuse_multiview_branch",
    "materialize_view_queries",
    "qa_evidence_from_asr",
    "select_production_policy",
    "write_r5_artifacts",
]
