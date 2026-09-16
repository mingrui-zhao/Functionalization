"""Stage 4: post-snap validation. Sweep the door through the URDF swing
range and check actual mesh penetration against body fragments.

Three-tier verdict:
    fail  -- door penetrates a body fragment by > 5mm at any frame.
             Door physically can't open; reject this candidate.
    warn  -- door penetrates by 0.5-5mm OR mechanism penetrates at all.
             Hinge will bind/rub but door clears. Accept; note in manifest.
    pass  -- < 0.5mm anywhere. Mesh tessellation noise.

Implementation:
    Per swing frame, compute the door's world-space rotated mesh, build a
    BVH from those polygons, and BVH-overlap against each body fragment.
    For overlapping pairs, estimate penetration depth as the world-AABB
    intersection thickness in the fragment-normal direction.

    BVH overlap is the binary detector (precise); AABB intersection is
    the severity proxy (approximate). Keeping them separate means a frame
    with a few stray triangle pairs (BVH overlap, tiny AABB depth) reads
    as "pass" while a real interpenetration reads "warn" or "fail".

Excluded fragments (caller must filter before calling):
    - the mounting panel of the chosen placement (always overlaps the
      hinge mech by definition)
    - the door's own merged mesh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import bpy  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore

from .geometry import HingePlacement


_FAIL_DEPTH_MM = 5.0
_WARN_DEPTH_MM = 0.5


@dataclass
class ValidationVerdict:
    status: Literal["pass", "warn", "fail"]
    max_door_penetration_mm: float
    max_mechanism_penetration_mm: float
    worst_frame: int | None
    diagnostic: dict = field(default_factory=dict)


def validate_snap(
    door_obj: bpy.types.Object,
    mechanism_objs: list[bpy.types.Object],
    body_fragment_objs: list[bpy.types.Object],
    placement: HingePlacement,
    swing_samples: int = 20,
) -> ValidationVerdict:
    """Sweep door + mechanism through swing_range, return verdict."""
    bpy.context.view_layer.update()

    # ---- per-fragment BVHs (built once; fragments don't move) ----
    frag_data = []
    for frag in body_fragment_objs:
        if frag.type != "MESH" or frag.data is None or len(frag.data.vertices) == 0:
            continue
        verts_world = [frag.matrix_world @ v.co for v in frag.data.vertices]
        polys = [list(p.vertices) for p in frag.data.polygons]
        if not polys:
            continue
        bvh = BVHTree.FromPolygons(verts_world, polys)
        frag_min, frag_max = _world_aabb_from_verts(verts_world)
        frag_data.append({
            "name": frag.name,
            "bvh": bvh,
            "aabb_min": frag_min,
            "aabb_max": frag_max,
        })

    # ---- door + mechanism rest meshes (verts in world coords) ----
    door_rest_verts = _verts_world(door_obj)
    door_polys = [list(p.vertices) for p in door_obj.data.polygons]
    mech_rest = []
    for m in mechanism_objs:
        if m.type != "MESH" or m.data is None or len(m.data.vertices) == 0:
            continue
        mech_rest.append({
            "name": m.name,
            "verts": _verts_world(m),
            "polys": [list(p.vertices) for p in m.data.polygons],
        })

    axis = np.asarray(placement.axis_direction, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    origin = np.asarray(placement.axis_origin, dtype=float)
    lower, upper = placement.swing_range
    angles = np.linspace(lower, upper, swing_samples)

    max_door = 0.0
    max_mech = 0.0
    worst_frame: int | None = None
    per_frame: list[dict] = []

    for i, ang in enumerate(angles):
        R = _rodrigues_np(axis, ang)
        # Rotate door verts about (origin, axis).
        door_verts_rot = _rotate_verts(door_rest_verts, R, origin)
        if not door_polys:
            door_overlap_mm = 0.0
        else:
            door_overlap_mm = _max_penetration(door_verts_rot, door_polys, frag_data)
        # Rotate each mechanism part. Take max over all.
        mech_overlap_mm = 0.0
        for m in mech_rest:
            verts_rot = _rotate_verts(m["verts"], R, origin)
            d = _max_penetration(verts_rot, m["polys"], frag_data)
            if d > mech_overlap_mm:
                mech_overlap_mm = d

        per_frame.append({
            "frame": i,
            "angle_rad": float(ang),
            "door_penetration_mm": door_overlap_mm,
            "mech_penetration_mm": mech_overlap_mm,
        })

        worst_at_this_frame = max(door_overlap_mm, mech_overlap_mm)
        if worst_at_this_frame > max(max_door, max_mech):
            worst_frame = i
        if door_overlap_mm > max_door:
            max_door = door_overlap_mm
        if mech_overlap_mm > max_mech:
            max_mech = mech_overlap_mm

    status: Literal["pass", "warn", "fail"]
    if max_door > _FAIL_DEPTH_MM:
        status = "fail"
    elif max_door > _WARN_DEPTH_MM or max_mech > _WARN_DEPTH_MM:
        status = "warn"
    else:
        status = "pass"

    return ValidationVerdict(
        status=status,
        max_door_penetration_mm=max_door,
        max_mechanism_penetration_mm=max_mech,
        worst_frame=worst_frame,
        diagnostic={
            "fragment_count": len(frag_data),
            "swing_samples": swing_samples,
            "fail_threshold_mm": _FAIL_DEPTH_MM,
            "warn_threshold_mm": _WARN_DEPTH_MM,
            "per_frame": per_frame,
        },
    )


# ============================================================ helpers ==

def _verts_world(obj: bpy.types.Object) -> list[Vector]:
    mw = obj.matrix_world
    return [mw @ v.co for v in obj.data.vertices]


def _world_aabb_from_verts(verts: list[Vector]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array([(v.x, v.y, v.z) for v in verts])
    return arr.min(axis=0), arr.max(axis=0)


def _rodrigues_np(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotate_verts(verts: list[Vector], R: np.ndarray, origin: np.ndarray) -> list[Vector]:
    """Rotate a list of mathutils.Vector verts around (origin, R) and return
    a new list of Vector. Implementation in numpy for speed."""
    arr = np.array([(v.x, v.y, v.z) for v in verts])
    rotated = (arr - origin) @ R.T + origin
    return [Vector(tuple(p)) for p in rotated]


def _max_penetration(
    moving_verts_world: list[Vector],
    moving_polys: list[list[int]],
    frag_data: list[dict],
) -> float:
    """Return max penetration depth (mm) of the moving mesh into any fragment.

    Two-pass:
        1. AABB-AABB pre-screen per fragment. Skip if no overlap.
        2. BVH-overlap on screen-passing fragments. If overlap, severity =
           AABB intersection's smallest-axis thickness.
    """
    arr = np.array([(v.x, v.y, v.z) for v in moving_verts_world])
    moving_min = arr.min(axis=0)
    moving_max = arr.max(axis=0)

    max_depth_mm = 0.0
    moving_bvh: BVHTree | None = None  # built lazily

    for frag in frag_data:
        # AABB screen.
        overlap = np.minimum(moving_max, frag["aabb_max"]) - np.maximum(moving_min, frag["aabb_min"])
        if (overlap <= 0).any():
            continue
        # BVH check.
        if moving_bvh is None:
            moving_bvh = BVHTree.FromPolygons(moving_verts_world, moving_polys)
        pairs = moving_bvh.overlap(frag["bvh"])
        if not pairs:
            continue
        # Severity = AABB intersection's smallest dimension in mm.
        depth_mm = float(overlap.min()) * 1000.0
        if depth_mm > max_depth_mm:
            max_depth_mm = depth_mm
    return max_depth_mm
