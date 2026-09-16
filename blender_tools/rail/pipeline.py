"""rail pipeline. Per drawer joint:
  * Build the LEFT RailPlacement with the right anchor for the template's
    style:
       CORNER  -> drawer-bottom corner on this side
       CENTER  -> drawer-side midpoint (vertical middle) on this side
  * Snap the left rail with snap_rail_to_placement
  * MIRROR the left rail across the drawer's median (sagittal) plane to
    produce the right rail (Householder reflection + flip_normals to
    restore outward winding), then coborder-refine both sides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import bpy  # type: ignore
from mathutils import Matrix, Vector  # type: ignore

from .snap import RailPlacement, RailSnapResult, snap_rail_to_placement
from .templates import (RailTemplateRefs, _NAME_LOOKUP_OPTIONAL,
                        _NAME_LOOKUP_REQUIRED, load_rail_template)


@dataclass
class DrawerInstallResult:
    joint_name: str
    style:      str   # "corner" | "center"
    left:       RailSnapResult
    right:      RailSnapResult | None
    diagnostic: dict = field(default_factory=dict)


def _build_anchor(rec: dict, side: str, style: str,
                  body_region) -> np.ndarray:
    """Anchor world position where the rail's snap_static centroid lands.

    The slide-axial position is the BACK end of the drawer body (= the
    farthest-from-pulling-board end). For left/right side and bottom/middle
    height, anchor is at the body-back corner (corner style) or body-back
    midpoint (center style).
    """
    drawer_center  = np.asarray(rec["drawer_center"],     dtype=float)
    slide_world    = np.asarray(rec["slide_axis_world"],  dtype=float)
    width_world    = np.asarray(rec["width_axis_world"],  dtype=float)
    height_world   = np.asarray(rec["height_axis_world"], dtype=float)
    half_width     = float(rec["half_width_m"])
    half_height    = float(rec["half_height_m"])
    sign = -1.0 if side == "left" else +1.0

    # Slide-axial back (= LOWEST projection on slide_outward axis)
    back_along_slide = body_region.slide_min_world
    drawer_center_along_slide = float(drawer_center @ slide_world)
    slide_offset = back_along_slide - drawer_center_along_slide

    # In-plane offset (left/right + bottom/middle)
    perp_offset_world = sign * half_width * width_world
    if style == "corner":
        perp_offset_world = perp_offset_world - half_height * height_world
    return drawer_center + slide_offset * slide_world + perp_offset_world


def _build_placement(rec: dict, side: str, style: str, body_region) -> RailPlacement:
    return RailPlacement(
        slide_axis_world=np.asarray(rec["slide_axis_world"],  dtype=float),
        width_axis_world=np.asarray(rec["width_axis_world"],  dtype=float),
        height_axis_world=np.asarray(rec["height_axis_world"], dtype=float),
        drawer_center=np.asarray(rec["drawer_center"], dtype=float),
        half_slide=float(rec["half_slide_m"]),
        half_width=float(rec["half_width_m"]),
        half_height=float(rec["half_height_m"]),
        side=side,
        anchor_world=_build_anchor(rec, side, style, body_region),
    )


def _detect_style(template_path: Path) -> str:
    """Probe-load the template once to detect corner vs center style."""
    refs = load_rail_template(template_path, suffix="_probe")
    style = refs.style
    # Cleanup probe
    for o in list(bpy.data.objects):
        if o.name.endswith("___probe") or o.name.endswith("__probe"):
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.context.view_layer.update()
    return style


def install_rails_for_drawer(
    template_path: Path,
    drawer_record: dict,
    drawer_mesh_objs: list[bpy.types.Object] | None = None,
) -> DrawerInstallResult:
    """drawer_mesh_objs: pass the drawer's mesh fragments from the open
    scene so we can:
      1. detect body region (= largest fragment slide-extent)
      2. parent the drawer to the rail's dynamic_end (so motion follows)
    If None, falls back to drawer OBB extents (less accurate)."""
    from .body_region import detect_body_region
    from .scale import compute_rail_scale

    joint_name = drawer_record["joint"]
    style = _detect_style(template_path)

    # Body region detection (or fallback to OBB)
    slide_world = np.asarray(drawer_record["slide_axis_world"], dtype=float)
    drawer_center = np.asarray(drawer_record["drawer_center"], dtype=float)
    body_region = (detect_body_region(drawer_mesh_objs, slide_world)
                   if drawer_mesh_objs else None)
    if body_region is None:
        center_proj = float(drawer_center @ slide_world)
        half = float(drawer_record["half_slide_m"])
        from .body_region import BodyRegion
        body_region = BodyRegion(
            body_obj_name="<obb_fallback>",
            slide_min_world=center_proj - half,
            slide_max_world=center_proj + half,
            body_length=2 * half,
            body_center_along_slide=center_proj,
            diagnostic={"fallback": "obb_only"},
        )

    # Compute uniform scale + snap_static offset from rail back
    scale_factor, snap_static_offset_back, scale_diag = compute_rail_scale(
        template_path, body_region.body_length)

    # Snap LEFT rail (snap_static.+normal aligns with cabinet-LEFT-wall direction = +width)
    placement_l = _build_placement(drawer_record, side="left", style=style, body_region=body_region)
    left  = snap_rail_to_placement(
        template_path, placement_l,
        suffix=f"{joint_name}_L",
        scale_factor=scale_factor,
        snap_static_offset_from_back=snap_static_offset_back,
    )

    # Mirror LEFT rail across drawer's median plane (perpendicular to width
    # axis through drawer_center) to produce the RIGHT rail. Reflection
    # preserves the rail's "up" direction (no top-down flip) AND flips
    # snap_static normal to point at the right cabinet wall.
    right = _mirror_rail_across_drawer_median(left, drawer_record, suffix=f"{joint_name}_R")

    # Raycast-based perpendicular refinement using coborder (median-based
    # body-W detection, robust to single-geometry drawers and thick pulling
    # boards).
    # Right rail uses mirrored placement; coborder raycasts and the wall
    # thickness cap both need the flipped inward direction.
    right.placement.side = "right"
    coborder_diag = {}
    if drawer_mesh_objs:
        from .coborder import coborder_rail_to_drawer
        body_slide_range = (body_region.slide_min_world, body_region.slide_max_world)
        coborder_diag["left"] = coborder_rail_to_drawer(
            left, drawer_mesh_objs, body_slide_range=body_slide_range)
        coborder_diag["right"] = coborder_rail_to_drawer(
            right, drawer_mesh_objs, body_slide_range=body_slide_range)

    # Thickness cap (must run BEFORE the drawer is parented to the rail,
    # so the anisotropic scale never touches drawer geometry). The carve
    # cutter ships inside the template and follows the root through snap,
    # mirror, and this cap.
    from .carve import CARVE_PREFIX, cap_rail_thickness_to_wall
    cap_diag = {
        "left": cap_rail_thickness_to_wall(left, drawer_mesh_objs or []),
        "right": cap_rail_thickness_to_wall(right, drawer_mesh_objs or []),
    }
    # Eye-hide the template cutters now that mirroring and capping are done.
    # hide_set keeps them in the depsgraph (booleans still evaluate them,
    # matrices stay fresh), unlike hide_viewport which would not.
    for res in (left, right):
        for o in res.root.children_recursive:
            if o.name.startswith(CARVE_PREFIX):
                try:
                    o.hide_set(True)
                except RuntimeError:
                    pass

    # Parent drawer mesh fragments to LEFT dynamic_end (one driver, like
    # multihinge convention) so the drawer follows the rail's slide motion.
    if drawer_mesh_objs:
        _parent_drawer_to_rail(drawer_mesh_objs, left.refs.dynamic_end)

    return DrawerInstallResult(
        joint_name=joint_name, style=style, left=left, right=right,
        diagnostic={"body_region": {
            "body_obj": body_region.body_obj_name,
            "body_length_m": body_region.body_length,
            "slide_min_world": body_region.slide_min_world,
            "slide_max_world": body_region.slide_max_world,
        }, "scale": scale_diag,
        "thickness_cap": cap_diag,
        "drawer_parented": bool(drawer_mesh_objs),
        "coborder": coborder_diag},
    )


def _mirror_rail_across_drawer_median(
    left: RailSnapResult,
    drawer_record: dict,
    suffix: str,
) -> RailSnapResult:
    """Duplicate the left rail's hierarchy and reflect across the plane
    perpendicular to width_axis through drawer_center."""
    width = Vector((float(drawer_record["width_axis_world"][0]),
                    float(drawer_record["width_axis_world"][1]),
                    float(drawer_record["width_axis_world"][2]))).normalized()
    center = Vector((float(drawer_record["drawer_center"][0]),
                     float(drawer_record["drawer_center"][1]),
                     float(drawer_record["drawer_center"][2])))
    # Householder reflection about plane through `center` perpendicular to `width`
    n = width
    refl3 = Matrix.Identity(3) - 2.0 * Matrix(((n.x*n.x, n.x*n.y, n.x*n.z),
                                               (n.y*n.x, n.y*n.y, n.y*n.z),
                                               (n.z*n.x, n.z*n.y, n.z*n.z)))
    refl4 = (Matrix.Translation(center) @ refl3.to_4x4()
             @ Matrix.Translation(-center))

    # Duplicate hierarchy
    bpy.ops.object.select_all(action="DESELECT")
    def select_recursive(o):
        o.select_set(True)
        for c in o.children:
            select_recursive(c)
    select_recursive(left.root)
    bpy.context.view_layer.objects.active = left.root
    bpy.ops.object.duplicate(linked=False)
    new_root = bpy.context.view_layer.objects.active
    new_root.name = left.root.name.replace("__", "__mirror_") + f"__{suffix}"

    # Apply reflection to root's matrix_world; children inherit.
    new_root.matrix_world = refl4 @ new_root.matrix_world
    bpy.context.view_layer.update()

    # Reflection flips face winding -> normals point inward. Recalculate
    # outside on every duplicated mesh so shading/raycast normals stay sane.
    for obj in [new_root, *new_root.children_recursive]:
        if obj.type == "MESH" and obj.data is not None:
            obj.data.flip_normals()
    bpy.context.view_layer.update()

    new_refs = _rebuild_refs_after_duplicate(new_root, left.refs)
    # A COPY of the left placement: the caller flips .side to "right" for
    # coborder, which must not mutate the left rail's record by reference.
    from dataclasses import replace as _dc_replace
    return RailSnapResult(
        root=new_root, refs=new_refs,
        placement=_dc_replace(left.placement),
        diagnostic={"mirrored_from": left.root.name, "suffix": suffix},
    )


def _rebuild_refs_after_duplicate(new_root, original_refs: RailTemplateRefs) -> RailTemplateRefs:
    """After bpy.ops.object.duplicate, rebuild refs by matching base names
    (= template object name before per-instance suffix) in the new
    hierarchy. Needed so coborder reads centroids from the mirrored rail
    rather than the original LEFT rail."""
    candidates = [new_root, *new_root.children_recursive]

    def base_name(name: str) -> str:
        if "." in name and name.rsplit(".", 1)[-1].isdigit():
            name = name.rsplit(".", 1)[0]
        return name.split("__", 1)[0]

    by_base: dict[str, list] = {}
    for o in candidates:
        by_base.setdefault(base_name(o.name), []).append(o)

    def find(canonical: str, variants: tuple[str, ...]):
        for v in variants:
            if v in by_base and by_base[v]:
                return by_base[v][0]
        return None

    snap_support = (find("snap_support", _NAME_LOOKUP_OPTIONAL["snap_support"])
                    if original_refs.snap_support is not None else None)
    return RailTemplateRefs(
        root=new_root,
        axis=find("axis", _NAME_LOOKUP_REQUIRED["axis"]) or original_refs.axis,
        static_end=find("static_end", _NAME_LOOKUP_REQUIRED["static_end"]) or original_refs.static_end,
        dynamic_end=find("dynamic_end", _NAME_LOOKUP_REQUIRED["dynamic_end"]) or original_refs.dynamic_end,
        snap_static=find("snap_static", _NAME_LOOKUP_REQUIRED["snap_static"]) or original_refs.snap_static,
        snap_dynamic=find("snap_dynamic", _NAME_LOOKUP_REQUIRED["snap_dynamic"]) or original_refs.snap_dynamic,
        snap_support=snap_support,
        style=original_refs.style,
        template_path=original_refs.template_path,
    )


def _parent_drawer_to_rail(drawer_objs: list, dynamic_end_obj) -> None:
    """Parent every drawer mesh fragment to dynamic_end, preserving world
    positions via parent_inverse_matrix."""
    for d in drawer_objs:
        d.parent = dynamic_end_obj
        d.matrix_parent_inverse = dynamic_end_obj.matrix_world.inverted()
    bpy.context.view_layer.update()
