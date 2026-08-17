"""SCA-1 diagnostic-only SigLIP2 complementarity experiment."""

from .assets import (
    create_asset_zip,
    local_only_load_smoke,
    prepare_offline_asset,
    validate_offline_asset,
)
from .attribution import (
    classify_complementarity,
    oracle_union_diagnostics,
    paired_unit_deltas,
    summarize_paired,
)
from .backend import Siglip2ExactBackend
from .contracts import SCA1Settings
from .encoder import Siglip2OfflineEncoder, l2_normalize
from .index import build_siglip2_index, run_index_smoke, validate_siglip2_index
from .pipeline import Siglip2GroundingPipeline
from .preparation import PreparationFreeze, load_preparation_freeze
from .report import formal_report
from .runner import (
    create_bundle,
    evaluate_post_gt,
    run_pre_gt_arm,
    validate_pre_gt_integrity,
    write_manifests,
)

__all__ = [
    "PreparationFreeze",
    "SCA1Settings",
    "Siglip2ExactBackend",
    "Siglip2GroundingPipeline",
    "Siglip2OfflineEncoder",
    "build_siglip2_index",
    "classify_complementarity",
    "create_asset_zip",
    "create_bundle",
    "evaluate_post_gt",
    "formal_report",
    "l2_normalize",
    "load_preparation_freeze",
    "local_only_load_smoke",
    "oracle_union_diagnostics",
    "paired_unit_deltas",
    "prepare_offline_asset",
    "run_index_smoke",
    "run_pre_gt_arm",
    "summarize_paired",
    "validate_offline_asset",
    "validate_pre_gt_integrity",
    "validate_siglip2_index",
    "write_manifests",
]
