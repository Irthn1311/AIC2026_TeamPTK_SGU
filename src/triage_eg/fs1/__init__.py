"""FS1 final-shot experimental integration layer.

This package is deliberately isolated from production retrieval defaults.
"""

from .contracts import FS1Settings, RouteDecision
from .fusion import fuse_tail, reciprocal_rank_fusion
from .router import route_query

__all__ = ["FS1Settings", "RouteDecision", "fuse_tail", "reciprocal_rank_fusion", "route_query"]
