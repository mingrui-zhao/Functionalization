"""Cobordering: align the dynamic snap-plane's pin-side axial edge with
the door dynamic face's pin-side outer edge.

Why: snap.py + refinement land the snap planes ON their target face
planes, but the in-plane positioning along the perpendicular-to-axis
direction is unconstrained -- so the leaf can sit anywhere along the
door's width. For EXT and FLAT installations (face-to-face geometry),
that means the leaf can end up offset from the door's actual hinge edge
and the hinge knuckle floats off the door corner.

Cobordering picks the in-plane translation that makes the dynamic snap
plane's edge nearest the pin coincide with the door dynamic face's edge
nearest the pin -- i.e. snaps the leaf flush to the door's outer hinge
border. Since the translation is in the in-plane perp-to-axis direction
and EXT/FLAT have parallel face normals (door and panel), the same
translation also keeps the static snap plane on its target panel face
-- nothing else gets broken.

INTERIOR is a no-op: its leaves sit at the corner edge naturally
because the static and dynamic faces are perpendicular, not parallel.
"""
from __future__ import annotations

import numpy as np
import bpy  # type: ignore
from mathutils import Vector  # type: ignore

from .geometry import HingePlacement, HingeType, OBB
from .snap import SnapResult, _to_list


def _obb_face_corners_world(obb: OBB, face_id: int) -> list[Vector]:
    """4 corners of OBB face `face_id` in world coords.
    face_id encoding: 2*axis + (sign==+ ? 1 : 0)."""
    axis = face_id // 2
    sign = +1 if (face_id % 2 == 1) else -1
    e = obb.half_extents
    R = obb.R
    c = obb.center
    other = [i for i in range(3) if i != axis]
    corners = []
    for sa in (-1, +1):
        for sb in (-1, +1):
            local = np.zeros(3)
            local[axis] = sign * e[axis]
            local[other[0]] = sa * e[other[0]]
            local[other[1]] = sb * e[other[1]]
            world = c + R @ local
            corners.append(Vector((float(world[0]), float(world[1]), float(world[2]))))
    return corners


def _seam_coborder_one_side(
    placement: HingePlacement,
    snap_plane_obj,
    this_obb: OBB,
    this_face_id: int,
    this_face_normal_vec: Vector,
    other_obb: OBB,
    other_face_id: int,
) -> Vector:
    """Coborder one side: translate the leaf along `inplane_perp` so the
    snap_plane's pin-near edge coincides with `this_obb`'s face edge that
    is closest to the placement pin.

    The previous "closest in absolute distance to other_obb's face" rule
    picked the wrong end of the door whenever the panel was taller than
    the door in the perpendicular direction — the door's FAR-from-pin
    edge happens to be the closer pair to a panel edge, so the algorithm
    pulled the hinge to the far end. The pin position (placement.axis_
    origin) is the trustworthy anchor: predictions encode it directly
    via the hinge_border face, and built-from-URDF placements derive it
    from joint geometry. `other_obb`/`other_face_id` are kept in the
    signature for API stability but are no longer used.
    """
    _ = (other_obb, other_face_id)   # intentionally unused; see docstring
    pin_axis = Vector(_to_list(placement.axis_direction))
    if pin_axis.length < 1e-9:
        return Vector((0.0, 0.0, 0.0))
    pin_axis.normalize()
    inplane_perp = pin_axis.cross(this_face_normal_vec)
    if inplane_perp.length < 1e-9:
        return Vector((0.0, 0.0, 0.0))
    inplane_perp.normalize()

    if snap_plane_obj is None or snap_plane_obj.data is None \
            or len(snap_plane_obj.data.vertices) == 0:
        return Vector((0.0, 0.0, 0.0))
    perps = [(snap_plane_obj.matrix_world @ v.co).dot(inplane_perp)
             for v in snap_plane_obj.data.vertices]
    pin_origin = Vector(_to_list(placement.axis_origin))
    pin_perp = pin_origin.dot(inplane_perp)
    p_max, p_min = max(perps), min(perps)
    plane_pin_side = p_max if abs(p_max - pin_perp) < abs(p_min - pin_perp) else p_min

    this_corners = _obb_face_corners_world(this_obb, this_face_id)
    this_perps = [c.dot(inplane_perp) for c in this_corners]
    t_max, t_min = max(this_perps), min(this_perps)

    # Pick this_obb's face edge closest to the pin (in projection along
    # inplane_perp). That edge IS the hinge-side seam — the placement
    # already located which end of the door the hinge mounts on.
    box_target = t_max if abs(t_max - pin_perp) < abs(t_min - pin_perp) else t_min
    return (box_target - plane_pin_side) * inplane_perp


def _face_edges_u(obb: OBB, face_id: int, u: Vector) -> tuple[float, float]:
    """(min, max) projections along `u` of the `face_id` face corners."""
    perps = [c.dot(u) for c in _obb_face_corners_world(obb, face_id)]
    return min(perps), max(perps)


def _seam_mid_u(door_obb: OBB, door_face: int,
                panel_obb: OBB, panel_face: int, u: Vector) -> float:
    """Midpoint of the door↔panel SEAM along `u`: the pair of face edges
    (one from each part) that lie closest to each other. Selecting by
    mutual proximity — not by distance to the pin — matters because the
    pre-coborder pin position is arbitrary along `u` (snap leaves this
    direction unconstrained for parallel-normal mounts), so a nearest-
    to-pin rule can grab the door's opposite edge."""
    d_lo, d_hi = _face_edges_u(door_obb, door_face, u)
    p_lo, p_hi = _face_edges_u(panel_obb, panel_face, u)
    best = min(((de, pe) for de in (d_lo, d_hi) for pe in (p_lo, p_hi)),
               key=lambda t: abs(t[0] - t[1]))
    return 0.5 * (best[0] + best[1])


def apply_coborder_to_multihinge(
    snap_results: list[SnapResult],
    placement: HingePlacement,
    door_obb: OBB,
    panel_obb: OBB | None = None,
) -> dict:
    """Compute cobordering delta from the first hinge and apply it to
    each hinge's root.

    EXT — seam-aligned leaf rule: the dynamic snap-plane's pin-near edge
    is aligned to the door face's edge nearest the pin, so the leaf sits
    flush at the door's hinge border (the knuckle correctly protrudes
    past the border on an exterior mount).

    FLAT — PIN-CENTERED seam rule: the knuckle must sit ON the seam
    between the two coplanar parts, i.e. between the door border and the
    panel border. Aligning leaf EDGES to the borders (the previous
    averaged rule) only centres the pin when both leaves carry identical
    knuckle clearances — real templates are asymmetric, which pushed the
    pin a constant offset outside the seam. Instead, translate along the
    in-plane perpendicular so the template's rotation axis lands at the
    midpoint between the two borders' pin-near edges.

    INTERIOR is a no-op: its leaves sit at the corner edge naturally.
    """
    if not snap_results:
        return {"applied": False, "reason": "no snap_results"}
    if placement.hinge_type is HingeType.INTERIOR:
        return {"applied": False, "reason": "interior"}
    sr0 = snap_results[0]

    d_n = Vector(_to_list(placement.dynamic_face.normal))
    s_n = Vector(_to_list(placement.static_face.normal))
    if d_n.length >= 1e-9:
        d_n.normalize()
    if s_n.length >= 1e-9:
        s_n.normalize()

    delta_dyn = Vector((0.0, 0.0, 0.0))
    delta_sta = Vector((0.0, 0.0, 0.0))
    pin_centered = False
    if placement.hinge_type is HingeType.FLAT and panel_obb is not None:
        pin_axis = Vector(_to_list(placement.axis_direction))
        u = pin_axis.cross(d_n)
        if u.length >= 1e-9:
            u.normalize()
            seam_mid = _seam_mid_u(
                door_obb, placement.dynamic_face.face_id,
                panel_obb, placement.static_face.face_id, u)
            # Current pin position: the snapped template's rotation-axis
            # empty (authoritative); placement.axis_origin as fallback.
            if sr0.axis is not None:
                pin_u = Vector(sr0.axis.matrix_world.translation).dot(u)
            else:
                pin_u = Vector(_to_list(placement.axis_origin)).dot(u)
            delta = (seam_mid - pin_u) * u
            pin_centered = True
        else:
            delta = Vector((0.0, 0.0, 0.0))
    else:
        if panel_obb is not None and sr0.snap_dynamic is not None:
            delta_dyn = _seam_coborder_one_side(
                placement, sr0.snap_dynamic,
                door_obb, placement.dynamic_face.face_id, d_n,
                panel_obb, placement.static_face.face_id,
            )
        delta = delta_dyn

    if delta.length < 1e-9:
        return {"applied": False, "delta_mm": [0.0, 0.0, 0.0],
                "reason": "delta below threshold or interior/degenerate"}
    for sr in snap_results:
        sr.root.location = sr.root.location + delta
    bpy.context.view_layer.update()
    return {
        "applied": True,
        "seam_aligned": True,
        "pin_centered": pin_centered,
        "delta_dyn_mm": [round(float(delta_dyn.x)*1000, 2),
                          round(float(delta_dyn.y)*1000, 2),
                          round(float(delta_dyn.z)*1000, 2)],
        "delta_mm": [round(float(delta.x)*1000, 2),
                     round(float(delta.y)*1000, 2),
                     round(float(delta.z)*1000, 2)],
        "delta_magnitude_mm": round(float(delta.length)*1000, 2),
    }
