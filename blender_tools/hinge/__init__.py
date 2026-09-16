"""hinge — geometry-first hinge insertion for HSSD furniture.

Stage 1 (this file's siblings) is pure numpy: classify topology between a
door OBB and a panel OBB and enumerate every geometrically valid
HingePlacement. Stages 2-5 (policy, snap, multihinge, validate) live in
sibling modules and are added in later passes.
"""

from .geometry import (
    BoxRole,
    FaceRef,
    HingePlacement,
    HingeType,
    OBB,
    Topology,
    axis_with_canonical_sign,
    classify_topology,
    corner_ownership,
    enumerate_placements,
)
from .policy import (
    FitScore,
    TemplateLeafMeta,
    load_template_metadata,
    rank_by_fit,
)

__all__ = [
    "BoxRole",
    "FaceRef",
    "FitScore",
    "HingePlacement",
    "HingeType",
    "OBB",
    "TemplateLeafMeta",
    "Topology",
    "axis_with_canonical_sign",
    "classify_topology",
    "corner_ownership",
    "enumerate_placements",
    "load_template_metadata",
    "rank_by_fit",
]
