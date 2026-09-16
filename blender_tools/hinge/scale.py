"""Auto uniform-scale computation for hinge templates.

The pipeline applies a single scalar to root.scale of the loaded template
before snap reads any matrix_world transforms. The scale is uniform on all
3 axes so the hinge geometry stays proportional.

Per-type rules (with margin and clamp):

  EXT  -- snap_plane edges:
            along-axis edge represents "door axial length" footprint
            perp-axis edge represents "panel thickness" footprint
          scale = min(0.95 * door_axial_extent  / template_along_edge,
                     0.95 * panel_thickness     / template_perp_edge)

  FLAT -- snap_plane perp-axis edge represents "panel/door perpendicular
          width" footprint. Margin is TIGHTER (0.6) here because the FLAT
          template is already large.
          scale = 0.6 * min(panel_perp_width, door_perp_width)
                       / template_perp_edge

  INT  -- the static plane, when extended, cuts the dynamic plane along
          the pin axis. The smaller of the two resulting strips is the
          "overlay" portion that would poke past the panel face plane.
          scale = 0.95 * min(door_thickness, panel_thickness)
                       / dynamic_smaller_strip_perp_extent

All scales additionally capped by a door-axial proportionality rule
(post-scale hinge extent along the pin axis <= DOOR_AXIAL_FRAC * door
extent along that axis), and clamped to [MIN_SCALE, MAX_SCALE] = [0.03, 2.5].

EXT only, final no-protrusion check: after every rule above (including the
clamp), the leaf plates must not be thicker than the panels they mount on
-- static plate <= panel thickness, dynamic plate <= door thickness,
measured along each leaf's surface normal. If violated, the whole hinge is
uniformly shrunk by the worst ratio. Nothing feeds back into the earlier
rules, and the result is intentionally NOT re-clamped to MIN_SCALE: the
no-protrusion guarantee wins over the size floor.

Public entry: compute_auto_scale(template_path, hinge_type, door_obb,
panel_obb, placement) -> tuple[float, dict].

The function probe-loads the template to read snap_plane mesh dimensions,
computes the scale, then deletes the probe template before returning.
Cost is one extra template load per call, negligible compared to the rest
of the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import bpy  # type: ignore
from mathutils import Vector  # type: ignore

from .geometry import HingePlacement, HingeType, OBB
from .snap import _snap_plane_normal_world, _to_list
from .templates import load_template


# ============================================================ constants ==

MIN_SCALE = 0.03
MAX_SCALE = 2.5
DEFAULT_MARGIN = 0.95
INTERIOR_SIZE_BOOST = 1.15   # multiplier on the interior auto-scale (see rule below)
FLAT_MARGIN = 0.6
# Per-hinge axial extent should be at most this fraction of the door's
# axial extent. Caps every per-type rule so multi-hinge layouts can't
# overhang the door height.
DOOR_AXIAL_FRAC = 0.12


# ============================================================ helpers ==

def _obb_extent_along_world(obb: OBB, world_dir: np.ndarray) -> float:
    """Total extent of an OBB along an arbitrary world direction."""
    n = float(np.linalg.norm(world_dir))
    if n < 1e-12:
        return 0.0
    d = np.asarray(world_dir, dtype=float) / n
    return 2.0 * float(np.sum(np.abs(obb.R.T @ d) * obb.half_extents))


def _obb_thickness(obb: OBB) -> float:
    """Smallest OBB extent (= thickness perpendicular to the broad face)."""
    return 2.0 * float(min(obb.half_extents))


def _snap_plane_extents_in_axes(
    snap_obj: bpy.types.Object,
    template_pin_axis_world: Vector,
) -> tuple[float, float]:
    """(along_axis_extent, perp_axis_extent) of the snap_plane mesh
    rectangle, measured in template-world along the template pin axis and
    perpendicular to it in the leaf surface."""
    M = snap_obj.matrix_world
    verts = [M @ v.co for v in snap_obj.data.vertices]
    leaf_n = _snap_plane_normal_world(snap_obj)
    perp_dir = template_pin_axis_world.cross(leaf_n)
    if perp_dir.length < 1e-9:
        helper = Vector((1, 0, 0)) if abs(template_pin_axis_world.x) < 0.9 else Vector((0, 1, 0))
        perp_dir = template_pin_axis_world.cross(helper)
    perp_dir.normalize()
    along = [float(v.dot(template_pin_axis_world)) for v in verts]
    perp  = [float(v.dot(perp_dir)) for v in verts]
    return (max(along) - min(along)), (max(perp) - min(perp))


def _snap_plane_smaller_strip(
    snap_obj: bpy.types.Object,
    template_pin_axis_world: Vector,
    template_pin_origin_world: Vector,
) -> float:
    """For an INT-type snap_plane: the pin axis line lies within the
    rectangle and divides it along the perp-to-axis direction into two
    strips. Return the smaller strip's perp-to-axis extent."""
    M = snap_obj.matrix_world
    verts = [M @ v.co for v in snap_obj.data.vertices]
    leaf_n = _snap_plane_normal_world(snap_obj)
    perp_dir = template_pin_axis_world.cross(leaf_n)
    if perp_dir.length < 1e-9:
        return 0.0
    perp_dir.normalize()
    pin_perp = float(template_pin_origin_world.dot(perp_dir))
    rel = [float(v.dot(perp_dir)) - pin_perp for v in verts]
    pos = [r for r in rel if r > 0]
    neg = [r for r in rel if r < 0]
    pos_extent = max(pos) if pos else 0.0
    neg_extent = -min(neg) if neg else 0.0
    if pos_extent > 0 and neg_extent > 0:
        return min(pos_extent, neg_extent)
    return max(pos_extent, neg_extent)


def _mesh_extent_along(obj: bpy.types.Object, world_dir: Vector) -> float:
    """Total extent of a mesh object's vertices along a world direction."""
    if obj.type != "MESH" or obj.data is None or not obj.data.vertices:
        return 0.0
    M = obj.matrix_world
    vals = [float((M @ v.co).dot(world_dir)) for v in obj.data.vertices]
    return max(vals) - min(vals)


def _delete_probe_objects(suffix: str) -> None:
    """Remove every object whose name ends with `__{suffix}` (probe load)."""
    tag = f"__{suffix}"
    to_remove = [o for o in bpy.data.objects if o.name.endswith(tag)]
    for o in to_remove:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.context.view_layer.update()


# ============================================================ public api ==

def compute_auto_scale(
    template_path: Path,
    hinge_type: HingeType,
    door_obb: OBB,
    panel_obb: OBB,
    placement: HingePlacement,
    probe_suffix: str = "scale_probe",
) -> tuple[float, dict]:
    """Probe-load the template, measure leaf/snap-plane geometry, and
    compute the uniform scale factor per the per-type rules above.

    Returns (clamped_scale, diagnostic_dict). The probe template is
    appended into the active scene for measurement, then deleted before
    return -- the active scene state is restored to whatever it was before
    the call (modulo the probe load/delete which is transparent).
    """
    refs = load_template(template_path, suffix=probe_suffix)
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()

    template_pin_axis = (refs.axis.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    template_pin_origin = refs.axis.matrix_world.to_translation()
    placement_axis_np = np.asarray(_to_list(placement.axis_direction), dtype=float)
    s_face_n = np.asarray(_to_list(placement.static_face.normal), dtype=float)
    d_face_n = np.asarray(_to_list(placement.dynamic_face.normal), dtype=float)

    diag: dict = {"hinge_type": hinge_type.value}

    # All types are also capped by the door-axial proportionality rule:
    # the post-scale hinge extent along the pin axis must not exceed
    # DOOR_AXIAL_FRAC of the door's extent along that same axis.
    static_along, static_perp = _snap_plane_extents_in_axes(refs.snap_static, template_pin_axis)
    door_axial_for_cap = _obb_extent_along_world(door_obb, placement_axis_np)
    s_door_cap = (DOOR_AXIAL_FRAC * door_axial_for_cap / static_along) if static_along > 1e-9 else MAX_SCALE

    if hinge_type is HingeType.EXTERIOR:
        along, perp = static_along, static_perp
        door_axial = door_axial_for_cap
        panel_thick = _obb_thickness(panel_obb)
        s_along = (DEFAULT_MARGIN * door_axial / along) if along > 1e-9 else MAX_SCALE
        s_perp  = (DEFAULT_MARGIN * panel_thick / perp)  if perp  > 1e-9 else MAX_SCALE
        raw = min(s_along, s_perp, s_door_cap)
        diag.update(template_along_mm=along*1000, template_perp_mm=perp*1000,
                    door_axial_mm=door_axial*1000, panel_thick_mm=panel_thick*1000,
                    margin=DEFAULT_MARGIN, door_axial_frac=DOOR_AXIAL_FRAC,
                    s_along=s_along, s_perp=s_perp, s_door_cap=s_door_cap,
                    binding=("along" if s_along == raw else
                             "perp" if s_perp == raw else "door_axial_frac"))

    elif hinge_type is HingeType.FLAT:
        along, perp = static_along, static_perp
        s_perp_dir = np.cross(s_face_n, placement_axis_np)
        d_perp_dir = np.cross(d_face_n, placement_axis_np)
        if np.linalg.norm(s_perp_dir) > 1e-9:
            s_perp_dir = s_perp_dir / np.linalg.norm(s_perp_dir)
        if np.linalg.norm(d_perp_dir) > 1e-9:
            d_perp_dir = d_perp_dir / np.linalg.norm(d_perp_dir)
        panel_w = _obb_extent_along_world(panel_obb, s_perp_dir) if s_perp_dir.any() else _obb_thickness(panel_obb)
        door_w  = _obb_extent_along_world(door_obb,  d_perp_dir) if d_perp_dir.any() else _obb_thickness(door_obb)
        constraint = FLAT_MARGIN * min(panel_w, door_w)
        s_perp_rule = (constraint / perp) if perp > 1e-9 else MAX_SCALE
        raw = min(s_perp_rule, s_door_cap)
        diag.update(template_perp_mm=perp*1000, panel_perp_w_mm=panel_w*1000,
                    door_perp_w_mm=door_w*1000, margin=FLAT_MARGIN,
                    door_axial_frac=DOOR_AXIAL_FRAC,
                    s_perp_rule=s_perp_rule, s_door_cap=s_door_cap,
                    binding=("panel_w" if (s_perp_rule == raw and panel_w < door_w) else
                             "door_w"   if (s_perp_rule == raw) else
                             "door_axial_frac"))

    elif hinge_type is HingeType.INTERIOR:
        smaller = _snap_plane_smaller_strip(refs.snap_dynamic, template_pin_axis, template_pin_origin)
        door_thick = _obb_thickness(door_obb)
        panel_thick = _obb_thickness(panel_obb)
        constraint = DEFAULT_MARGIN * min(door_thick, panel_thick)
        s_thick_rule = (constraint / smaller) if smaller > 1e-9 else MAX_SCALE
        # Visual-size boost tuned on inspection builds (49188 ladder,
        # x0.9..x1.2): strictly thickness-contained plates read undersized;
        # 15% larger is the preferred look and stays collision-clean.
        raw = min(s_thick_rule, s_door_cap) * INTERIOR_SIZE_BOOST
        diag.update(template_smaller_strip_mm=smaller*1000,
                    door_thick_mm=door_thick*1000, panel_thick_mm=panel_thick*1000,
                    margin=DEFAULT_MARGIN, door_axial_frac=DOOR_AXIAL_FRAC,
                    s_thick_rule=s_thick_rule, s_door_cap=s_door_cap,
                    binding=("door_thick"      if (s_thick_rule == raw and door_thick < panel_thick) else
                             "panel_thick"     if (s_thick_rule == raw) else
                             "door_axial_frac"))
    else:
        raw = 1.0

    clamped = max(MIN_SCALE, min(MAX_SCALE, raw))
    diag["raw_scale"] = raw
    diag["clamped_scale"] = clamped
    diag["was_clamped"] = abs(clamped - raw) > 1e-6

    # Final no-protrusion check (EXT only). Runs after every other rule and
    # the clamp; nothing downstream re-derives scale from it.
    final = clamped
    if hinge_type is HingeType.EXTERIOR:
        t_static = _mesh_extent_along(
            refs.static_leaf, _snap_plane_normal_world(refs.snap_static))
        t_dynamic = _mesh_extent_along(
            refs.dynamic_leaf, _snap_plane_normal_world(refs.snap_dynamic))
        panel_thick = _obb_thickness(panel_obb)
        door_thick = _obb_thickness(door_obb)
        cap = 1.0
        if t_static > 1e-6 and panel_thick > 1e-6:
            cap = min(cap, panel_thick / (t_static * clamped))
        if t_dynamic > 1e-6 and door_thick > 1e-6:
            cap = min(cap, door_thick / (t_dynamic * clamped))
        if cap < 1.0:
            final = clamped * cap
        diag.update(plate_static_mm=t_static * clamped * 1000,
                    plate_dynamic_mm=t_dynamic * clamped * 1000,
                    door_thick_mm=door_thick * 1000,
                    plate_cap_factor=cap,
                    plate_capped=cap < 1.0)
    diag["final_scale"] = final

    _delete_probe_objects(probe_suffix)
    return final, diag
