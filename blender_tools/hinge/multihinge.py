"""Multi-hinge layout: replicate one HingePlacement along the door axis,
snap each hinge, parent the merged door to one driver hinge.

Count rule (per-template):
    INTERIOR / EXTERIOR (small concealed and overlay hinges):
        door_axial_extent < 600mm  -> 2 hinges
        door_axial_extent < 1500mm -> 3 hinges
        otherwise                  -> 4 hinges
    FLAT (wraparound utility hinges, intrinsically tall ~140mm each):
        door_axial_extent <= 2000mm -> 1 hinge
        otherwise                   -> 2 hinges

Hinge positions along the axis: spread evenly between fractional offsets
0.10 and 0.90 of the door's axial extent (measured from the lowest-axial
door corner). The MIDDLE hinge is the driver -- the merged door is
parented to its hinge_dynamic_end. The other hinges' dynamic leaves
animate cosmetically via their baked actions but don't affect the door's
transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import bpy  # type: ignore

from .geometry import HingePlacement, HingeType, OBB
from .snap import SnapResult, snap_template_to_placement


@dataclass
class MultiHingeResult:
    placements: list[HingePlacement]
    snap_results: list[SnapResult]
    driver_index: int                 # which entry drives the merged door
    diagnostic: dict = field(default_factory=dict)


def hinge_count_for(hinge_type: HingeType, door_axial_extent_mm: float) -> int:
    if hinge_type is HingeType.FLAT:
        return 1 if door_axial_extent_mm <= 2000.0 else 2
    if door_axial_extent_mm < 600.0:
        return 2
    if door_axial_extent_mm < 1500.0:
        return 3
    return 4


def door_axial_extent_mm(door_obb: OBB, axis_direction: np.ndarray) -> float:
    """Project the door OBB onto axis_direction and return its full extent in mm."""
    axis = np.asarray(axis_direction, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    e = np.asarray(door_obb.half_extents, dtype=float)
    R = np.asarray(door_obb.R, dtype=float)
    # Project each local axis onto the world axis_direction.
    contributions = np.abs(R.T @ axis) * e
    return float(2.0 * contributions.sum() * 1000.0)


def _axial_anchor_world(
    door_obb: OBB, axis_direction: np.ndarray, axis_origin: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return (door_axial_extent_meters, lowest_axial_point_world).

    The "lowest axial point" is the door OBB corner that minimises the
    projection onto axis_direction. We anchor multi-hinge offsets from
    this point so the layout is deterministic across runs.
    """
    axis = np.asarray(axis_direction, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    pin = np.asarray(axis_origin, dtype=float)
    e = np.asarray(door_obb.half_extents, dtype=float)
    R = np.asarray(door_obb.R, dtype=float)
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    corners = door_obb.center + (signs * e) @ R.T
    along = corners @ axis
    extent = float(along.max() - along.min())
    # Project the lowest-axial corner onto the line through pin parallel to axis.
    lowest_corner = corners[int(np.argmin(along))]
    rel = lowest_corner - pin
    along_pin = float(rel @ axis)
    anchor = pin + along_pin * axis
    return extent, anchor


def _axial_offsets(count: int, extent_m: float) -> list[float]:
    """Hinge positions as offsets-from-anchor along the axis, in meters."""
    if count == 1:
        return [0.5 * extent_m]
    fractions = np.linspace(0.10, 0.90, count)
    return [float(f * extent_m) for f in fractions]


def _shelf_avoiding_offsets(
    count: int,
    extent_m: float,
    anchor: np.ndarray,
    axis_direction: np.ndarray,
    body_obbs: list[OBB],
    margin_frac: float = 0.04,
) -> list[float]:
    """Distribute `count` hinge offsets along [0.05*extent, 0.95*extent] but
    avoid the projected axial spans of "shelf-like" body fragments.

    A body fragment is treated as a shelf-obstacle if its axial projection
    covers LESS THAN 50 percent of the door axial extent (i.e. it's not
    one of the long side panels that span the full door axis -- those are
    just the cabinet sides and don't actually obstruct the hinge in the
    axial direction).

    Each detected obstacle is widened by `margin_frac * extent_m` on each
    side. Free intervals = [0.05, 0.95] minus blocked intervals. Hinges
    are placed at the midpoints of the largest `count` free intervals (or
    fall back to uniform spacing if there isn't enough free space).
    """
    if count <= 0:
        return []
    axis = np.asarray(axis_direction, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    anchor_axial = float(np.asarray(anchor, dtype=float) @ axis)
    margin = margin_frac * extent_m

    blocked: list[tuple[float, float]] = []
    for obb in body_obbs:
        # Project OBB onto axis.
        e = np.asarray(obb.half_extents, dtype=float)
        R = np.asarray(obb.R, dtype=float)
        center_axial = float(np.asarray(obb.center, dtype=float) @ axis)
        half_axial = float(np.sum(np.abs(R.T @ axis) * e))
        # If the OBB's axial coverage is >= 50% of the door extent, treat
        # it as a side panel (not a shelf) and ignore.
        if half_axial * 2 >= 0.5 * extent_m:
            continue
        lo = (center_axial - half_axial - anchor_axial) - margin
        hi = (center_axial + half_axial - anchor_axial) + margin
        # Intersect with [0, extent_m] window
        lo = max(0.0, lo); hi = min(extent_m, hi)
        if hi > lo:
            blocked.append((lo, hi))

    # Compute free intervals = [0.05*extent, 0.95*extent] minus blocked
    window = (0.05 * extent_m, 0.95 * extent_m)
    blocked.sort()
    free: list[tuple[float, float]] = []
    cursor = window[0]
    for (lo, hi) in blocked:
        if hi < cursor:
            continue
        if lo > cursor:
            free.append((cursor, min(lo, window[1])))
        cursor = max(cursor, hi)
        if cursor >= window[1]:
            break
    if cursor < window[1]:
        free.append((cursor, window[1]))

    # If no free space (everything blocked), fall back to uniform.
    if not free or sum(hi - lo for lo, hi in free) < 1e-6:
        return _axial_offsets(count, extent_m)

    # If only one free interval and we need many hinges: distribute uniformly within it.
    if len(free) == 1:
        lo, hi = free[0]
        if count == 1:
            return [0.5 * (lo + hi)]
        fractions = np.linspace(0.10, 0.90, count)
        return [float(lo + f * (hi - lo)) for f in fractions]

    # Multiple free intervals: rank by length, take the largest `count`,
    # place hinge at each interval's midpoint.
    free_sorted = sorted(free, key=lambda iv: -(iv[1] - iv[0]))
    if len(free_sorted) >= count:
        chosen = sorted(free_sorted[:count], key=lambda iv: iv[0])  # restore axial order
        return [float(0.5 * (lo + hi)) for lo, hi in chosen]
    # More hinges than free intervals: place one per interval, then top up
    # by adding extra hinges in the largest interval (subdivide).
    placed = [0.5 * (lo + hi) for lo, hi in free_sorted]
    extra = count - len(placed)
    largest_lo, largest_hi = free_sorted[0]
    extra_offsets = np.linspace(largest_lo + 0.1*(largest_hi - largest_lo),
                                largest_hi - 0.1*(largest_hi - largest_lo),
                                extra + 1)[1:] if extra > 0 else []
    placed.extend(float(x) for x in extra_offsets)
    return sorted(placed)


def _make_offset_placement(
    placement: HingePlacement, offset_m: float
) -> HingePlacement:
    """Return a copy of placement with axis_origin shifted along axis_direction
    by `offset_m` meters."""
    axis = np.asarray(placement.axis_direction, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    new_origin = np.asarray(placement.axis_origin, dtype=float) + offset_m * axis
    return HingePlacement(
        axis_origin=new_origin,
        axis_direction=placement.axis_direction,
        static_face=placement.static_face,
        dynamic_face=placement.dynamic_face,
        closed_angle=placement.closed_angle,
        hinge_type=placement.hinge_type,
        swing_range=placement.swing_range,
    )


def insert_multihinge(
    template_path: Path,
    placement: HingePlacement,
    door_obb: OBB,
    suffix_base: str,
    refine: bool = True,
    refine_threshold_mm: float = 0.5,
    scale_factor: float = 1.0,
    hinge_count: int | None = None,
    body_obbs_for_avoidance: list[OBB] | None = None,
) -> MultiHingeResult:
    """Distribute hinges along the door axis and snap each one.

    The MIDDLE hinge is the designated driver (caller responsible for
    parenting the merged door to its hinge_dynamic_end). All other hinges
    instantiate as cosmetic copies that animate via their own baked actions.

    hinge_count: explicit number of hinges (>=1). None falls back to the
        per-type auto rule (hinge_count_for).
    body_obbs_for_avoidance: if provided AND placement.hinge_type is
        INTERIOR, hinge axial offsets are nudged to avoid the projected
        axial spans of these OBBs (e.g. shelves) so the hinge mechanism
        doesn't intersect internal cabinet structures.
    """
    extent_m, anchor = _axial_anchor_world(
        door_obb, placement.axis_direction, placement.axis_origin
    )
    extent_mm = extent_m * 1000.0
    count = hinge_count if hinge_count is not None else hinge_count_for(
        placement.hinge_type, extent_mm
    )
    if body_obbs_for_avoidance is not None and placement.hinge_type is HingeType.INTERIOR:
        offsets = _shelf_avoiding_offsets(
            count, extent_m, anchor,
            np.asarray(placement.axis_direction, dtype=float),
            body_obbs_for_avoidance,
        )
    else:
        offsets = _axial_offsets(count, extent_m)
    driver_index = count // 2  # middle (count=2 -> index 1, count=3 -> 1, count=4 -> 2)

    placements: list[HingePlacement] = []
    snap_results: list[SnapResult] = []
    axis = np.asarray(placement.axis_direction, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)

    # Convert "offset from anchor along axis" into "offset from placement.axis_origin
    # along axis" so _make_offset_placement can shift correctly.
    pin = np.asarray(placement.axis_origin, dtype=float)
    pin_to_anchor_along = float((anchor - pin) @ axis)

    for i, offset_m in enumerate(offsets):
        absolute_along_axis = pin_to_anchor_along + offset_m
        per_hinge_placement = _make_offset_placement(placement, absolute_along_axis)
        placements.append(per_hinge_placement)
        result = snap_template_to_placement(
            template_path=template_path,
            placement=per_hinge_placement,
            suffix=f"{suffix_base}__h{i}",
            refine=refine,
            refine_threshold_mm=refine_threshold_mm,
            scale_factor=scale_factor,
        )
        snap_results.append(result)

    return MultiHingeResult(
        placements=placements,
        snap_results=snap_results,
        driver_index=driver_index,
        diagnostic={
            "door_axial_extent_mm": extent_mm,
            "hinge_count": count,
            "axial_offsets_mm": [float(o * 1000.0) for o in offsets],
            "anchor_world": anchor.tolist(),
        },
    )


def parent_door_to_driver(
    door_obj: bpy.types.Object,
    multi: MultiHingeResult,
) -> None:
    """Parent the merged door object under the driver hinge's dynamic_leaf.

    Preserves the door's world transform.
    """
    driver = multi.snap_results[multi.driver_index]
    bpy.context.view_layer.update()
    door_world = door_obj.matrix_world.copy()
    door_obj.parent = driver.dynamic_leaf
    door_obj.matrix_parent_inverse = driver.dynamic_leaf.matrix_world.inverted()
    door_obj.matrix_world = door_world
    bpy.context.view_layer.update()
