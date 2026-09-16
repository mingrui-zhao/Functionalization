"""
mount_component_picker.py

Given a cabinet's body decomposed into oriented bounding boxes, pick the
most likely mounting panel for each movable joint, then classify the
door-vs-panel relationship into the hinge style it calls for.

Cases (for revolute joints / doors):

  Case 1 — exterior (overlay):   door face parallel to panel face, door
                                 sits in FRONT of panel. Use hinge2 default.
  Case 2 — tuck-in (parallel):   door face parallel to panel face, door
                                 sits BESIDE panel. Use hinge2 rotated 90°
                                 around the hinge axis.
  Case 3 — interior (perpend.):  door face perpendicular to panel face.
                                 Use hinge1 (interior mount).

Pure-Python (numpy only); no Blender / bpy dependency. Intended to be
imported from a driver script or from the adapted hinge-insertion scripts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# =====================================================================
# OBB data model
# =====================================================================

@dataclass
class OBB:
    """Oriented bounding box. Full extents (not half-extents) in `size`."""
    center: np.ndarray      # (3,)
    R: np.ndarray           # (3,3) rotation, columns are local axes in parent frame
    size: np.ndarray        # (3,) full extents along local axes

    @property
    def half(self) -> np.ndarray:
        return self.size / 2.0

    def thinnest_axis_idx(self) -> int:
        return int(np.argmin(self.size))

    def volume(self) -> float:
        return float(np.prod(self.size))


# =====================================================================
# OBB geometry helpers
# =====================================================================

def point_to_obb_distance(p: np.ndarray, obb: OBB) -> float:
    """Analytic point-to-OBB-surface distance. Returns 0 if inside."""
    local = obb.R.T @ (p - obb.center)
    clamped = np.clip(local, -obb.half, obb.half)
    return float(np.linalg.norm(local - clamped))


def outward_normal(obb: OBB, anchor: np.ndarray) -> tuple[np.ndarray, int]:
    """Unit normal along the OBB's thinnest axis, oriented away from `anchor`."""
    i = obb.thinnest_axis_idx()
    n = obb.R[:, i].copy()
    if np.dot(obb.center - anchor, n) < 0:
        n = -n
    return n, i


def aggregate_body_center(body_boxes: list[OBB]) -> np.ndarray:
    """Volume-weighted centroid of all body boxes."""
    if not body_boxes:
        return np.zeros(3)
    vols = np.array([b.volume() for b in body_boxes])
    cs = np.array([b.center for b in body_boxes])
    total = vols.sum()
    if total <= 0:
        return cs.mean(axis=0)
    return (cs * vols[:, None]).sum(axis=0) / total


# =====================================================================
# Classification
# =====================================================================

CASE_EXTERIOR = "case1_exterior"            # hinge2 default
CASE_TUCKIN_PARALLEL = "case2_tuckin"       # hinge2 rotated 90° around axis
CASE_INTERIOR = "case3_interior"            # hinge1

PARALLEL_COS_THRESHOLD = 0.7      # |n_d . n_p| above this = "parallel"
MIN_PANEL_THICKNESS = 1e-3        # 1 mm; filter out near-degenerate slabs
MAX_PANEL_THICKNESS_RATIO = 0.6   # skip "chunks" where thin axis is not
                                  #   meaningfully thin (size_min / size_max > this)


@dataclass
class PanelPick:
    idx: int
    obb: OBB
    dist_hinge_to_surface: float


@dataclass
class Classification:
    case: str
    panel: PanelPick
    n_door: np.ndarray
    n_panel: np.ndarray
    cos_theta: float                # abs dot of outward normals
    # how far door center is from panel face plane (along n_panel):
    offset_along_panel_normal: float
    # projected in-plane offset normalized by panel half-extent (u, v in panel face plane):
    in_plane_overlap_u: float
    in_plane_overlap_v: float
    # geometric vectors useful for orienting a tuck-in hinge in body frame
    outward_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inward_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))
    panel_to_door_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # list of candidate hinge strategies (primary first). Case 2 returns TWO
    # (tuck-in + interior fallback); Cases 1/3 return one.
    hinge_strategies: list = field(default_factory=list)


def _is_panel_like(box: OBB) -> bool:
    """Reject boxes that are effectively cubes/blocks — we want slab-like panels."""
    s = np.sort(box.size)
    if s[0] < MIN_PANEL_THICKNESS:
        return False
    return (s[0] / s[2]) < MAX_PANEL_THICKNESS_RATIO


def aabb_of_obb(obb: OBB) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box (in whatever frame obb is in) of the OBB."""
    aabb_half = np.zeros(3)
    for i in range(3):
        for j in range(3):
            aabb_half[i] += abs(obb.R[i, j]) * obb.half[j]
    return obb.center - aabb_half, obb.center + aabb_half


def aabb_overlap(a: OBB, b: OBB, tol: float = 0.0) -> bool:
    """True if the axis-aligned bounding boxes of two OBBs overlap, optionally
    shrunk/expanded by `tol` on each side. AABB overlap is a conservative
    collision proxy — a real rotation sweeps a larger volume, so if AABBs
    at rest already overlap, the swept door almost certainly collides
    with the panel."""
    a_min, a_max = aabb_of_obb(a)
    b_min, b_max = aabb_of_obb(b)
    for i in range(3):
        if a_min[i] - tol > b_max[i] or b_min[i] - tol > a_max[i]:
            return False
    return True


def support_extent(obb: OBB, direction: np.ndarray) -> float:
    """Max signed projection of the OBB onto `direction` (how far the OBB
    reaches in that direction). direction need not be unit length, though
    interpretation of returned scalar depends on |direction|."""
    c = float(obb.center @ direction)
    h = 0.0
    for i in range(3):
        h += abs(float(obb.R[:, i] @ direction)) * obb.half[i]
    return c + h


def pick_nearest_panel(
    query_point: np.ndarray,
    body_boxes: list[OBB],
    require_panel_like: bool = True,
) -> Optional[PanelPick]:
    """Pick the body box whose surface is closest to `query_point`."""
    best: Optional[PanelPick] = None
    for i, b in enumerate(body_boxes):
        if require_panel_like and not _is_panel_like(b):
            continue
        d = point_to_obb_distance(query_point, b)
        if best is None or d < best.dist_hinge_to_surface:
            best = PanelPick(idx=i, obb=b, dist_hinge_to_surface=d)
    if best is None and body_boxes:
        # fallback: drop the panel-like filter
        for i, b in enumerate(body_boxes):
            d = point_to_obb_distance(query_point, b)
            if best is None or d < best.dist_hinge_to_surface:
                best = PanelPick(idx=i, obb=b, dist_hinge_to_surface=d)
    return best


def classify_door_panel(
    door_obb_body: OBB,
    panel: PanelPick,
    body_center: np.ndarray,
    hinge_axis_body: np.ndarray,
    parallel_cos_threshold: float = PARALLEL_COS_THRESHOLD,
) -> Classification:
    """Classify the door-vs-panel relationship into Case 1 / 2 / 3."""
    n_panel, _ = outward_normal(panel.obb, body_center)
    n_door, _ = outward_normal(door_obb_body, body_center)

    cos_theta = abs(float(np.dot(n_door, n_panel)))

    # Offset of door center from panel face plane (signed).
    delta = door_obb_body.center - panel.obb.center
    offset_along_np = float(np.dot(delta, n_panel))

    # In-plane offset, normalized by panel half-extent along each panel in-plane axis.
    p_thin = panel.obb.thinnest_axis_idx()
    in_plane_axes = [a for a in range(3) if a != p_thin]
    u = panel.obb.R[:, in_plane_axes[0]]
    v = panel.obb.R[:, in_plane_axes[1]]
    du = float(np.dot(delta, u))
    dv = float(np.dot(delta, v))
    hu = float(panel.obb.half[in_plane_axes[0]])
    hv = float(panel.obb.half[in_plane_axes[1]])
    overlap_u = abs(du) / max(hu, 1e-9)
    overlap_v = abs(dv) / max(hv, 1e-9)

    subcase = ""
    if cos_theta >= parallel_cos_threshold:
        # Parallel: distinguish Case 1 (in front of) from Case 2 (beside).
        if overlap_u <= 1.0 and overlap_v <= 1.0:
            case = CASE_EXTERIOR
            subcase = "parallel_overlay"
        else:
            case = CASE_TUCKIN_PARALLEL
            subcase = "parallel_beside"
    else:
        # Perpendicular L-shape. Default: interior mount (case 3). Fall back
        # to exterior (case 1) or tuck-in (case 2) only when interior would
        # collide: whichever of door/panel extends farther outward owns the
        # outer corner, and that decides the fallback.
        #   door owns corner  -> case 1 exterior (pin sits on the door's
        #                        exterior face, hinge is visible on outside)
        #   panel owns corner -> case 2 tuck-in  (pin tucks behind panel's
        #                        outer edge)
        if not aabb_overlap(door_obb_body, panel.obb):
            case = CASE_INTERIOR
            subcase = "perpendicular_no_collision"
        else:
            # n_door is already the door's outward direction (away from
            # body centroid), so the exterior corner lies on the +n_door
            # side of the assembly. Whichever OBB extends farther along
            # +n_door owns the outer corner.
            forward = n_door
            ext_door = support_extent(door_obb_body, forward)
            ext_panel = support_extent(panel.obb, forward)
            if ext_door >= ext_panel:
                case = CASE_EXTERIOR
                subcase = "perpendicular_door_owns_corner"
            else:
                case = CASE_TUCKIN_PARALLEL
                subcase = "perpendicular_panel_owns_corner"

    # Shared outward direction (only meaningful for parallel cases, but we
    # compute it always so visualizations can draw an inward arrow).
    outward = n_door + n_panel
    if np.linalg.norm(outward) > 1e-6:
        outward = outward / np.linalg.norm(outward)
    else:
        outward = n_door
    inward = -outward

    # Unit direction from panel-center to door-center, projected onto the
    # plane perpendicular to the hinge axis. This is the "which side of
    # the pin is the door on" direction used to resolve the Case-2 tuck
    # rotation sign in downstream mount code.
    pd = door_obb_body.center - panel.obb.center
    axis_unit = hinge_axis_body / (np.linalg.norm(hinge_axis_body) + 1e-12)
    pd_perp = pd - float(np.dot(pd, axis_unit)) * axis_unit
    pd_norm = float(np.linalg.norm(pd_perp))
    pd_dir = pd_perp / pd_norm if pd_norm > 1e-9 else np.zeros(3)

    strategies = suggest_hinge_strategies(
        case=case,
        hinge_axis_body=axis_unit,
        outward_dir=outward,
        inward_dir=inward,
        panel_to_door_dir=pd_dir,
    )

    cls = Classification(
        case=case,
        panel=panel,
        n_door=n_door,
        n_panel=n_panel,
        cos_theta=cos_theta,
        offset_along_panel_normal=offset_along_np,
        in_plane_overlap_u=overlap_u,
        in_plane_overlap_v=overlap_v,
        outward_dir=outward,
        inward_dir=inward,
        panel_to_door_dir=pd_dir,
        hinge_strategies=strategies,
    )
    # Smuggle the subcase through the strategies so downstream code can
    # distinguish parallel_overlay vs perpendicular_door_owns_corner etc.
    for s in cls.hinge_strategies:
        s["subcase"] = subcase
    return cls


def suggest_hinge_strategies(
    case: str,
    hinge_axis_body: np.ndarray,
    outward_dir: np.ndarray,
    inward_dir: np.ndarray,
    panel_to_door_dir: np.ndarray,
) -> list:
    """Map a classified case to one or more placement strategies.

    Case 1: [hinge2 default].
    Case 2: [hinge2 tuck-in, hinge1 interior-alternative] — Case 2
            geometry is also admissible for a Case 3 style interior mount.
    Case 3: [hinge1 default].

    For Case 2 tuck-in, we do NOT fix the rotation angle sign here — the
    downstream mount code owns the sign, determined at placement time by the
    rule: after aligning the hinge's motion axis to the body-frame hinge
    axis, rotate about that axis so that (a) the dynamic-leaf anchor plane
    is on the door side (`panel_to_door_dir` direction from the pin), and
    (b) both leaves tuck along `inward_dir` with the pin sitting on the
    outward side. We expose the required geometric vectors so the caller
    can solve the sign analytically.
    """
    axis = hinge_axis_body
    out = outward_dir
    ind = inward_dir
    pd = panel_to_door_dir

    if case == CASE_EXTERIOR:
        return [{
            "template": "hinge2",
            "variant": "default_overlay",
            "motion_axis": axis.tolist(),
            "outward_dir": out.tolist(),
            "notes": "panel is parallel to door and door covers panel face",
        }]
    if case == CASE_TUCKIN_PARALLEL:
        return [
            {
                "template": "hinge2",
                "variant": "tuckin_perpendicular_leaves",
                "motion_axis": axis.tolist(),
                "inward_dir": ind.tolist(),
                "outward_dir": out.tolist(),
                "panel_to_door_dir": pd.tolist(),
                "notes": ("rotate 90 around motion_axis such that dynamic "
                          "anchor plane ends up on door side (+panel_to_door_dir) "
                          "and static anchor plane on panel side "
                          "(-panel_to_door_dir); both leaves tuck along "
                          "inward_dir, pin sits on outward_dir"),
            },
            {
                "template": "hinge1",
                "variant": "interior_alternative",
                "motion_axis": axis.tolist(),
                "inward_dir": ind.tolist(),
                "notes": "fallback: Case-2 geometry also admits a Case-3 style interior mount",
            },
        ]
    if case == CASE_INTERIOR:
        return [{
            "template": "hinge1",
            "variant": "interior_default",
            "motion_axis": axis.tolist(),
            "notes": "door perpendicular to mounting panel; interior mount",
        }]
    return []


# Keep the old singular name callable for backward compat if anything imports it.
# =====================================================================
# Per-URDF pipeline (revolute joints only for this prototype)
# =====================================================================

