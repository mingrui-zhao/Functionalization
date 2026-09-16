"""Rail snap algorithm.

Two rail-template styles are supported (templates.py reports `style`):

  CORNER (template_01: sliding_rail_annotated):
      3 snap planes -- static, dynamic, support
      One rail per side (left + right at drawer corners)
      Anchor: drawer bottom corner on the chosen side

  CENTER (template_02: sliding_rail_02_annotated):
      2 snap planes -- static, dynamic
      One rail per side, mounted at the vertical MIDDLE of drawer side
      (no support plane required)
      Anchor: drawer-side midpoint on the chosen side

Per-rail snap algorithm (rigid root transform):

  step 0  load template at frame 0 with `suffix`
  step 1  q_align: rotate root so sliding_axis empty's local +Z aligns
          with placement.slide_axis_world (= drawer's outward direction)
  step 2  q_roll : rotate around the slide axis so snap_plane_static_end
          MESH face normal aligns with the cabinet-wall-toward-drawer
          direction (= +width on left, -width on right)
  step 3  translate: shift root so snap_plane_static_end's MESH centroid
          (corner style) or MESH centroid (center style) lands at the
          chosen anchor point in world.

For CORNER style, anchor = drawer-bottom corner on the chosen side.
For CENTER style, anchor = drawer-side midpoint (vertical middle).

Both cases are governed by RailPlacement.anchor_world; pipeline computes
the right anchor per style.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import bpy  # type: ignore
from mathutils import Quaternion, Vector  # type: ignore

from .templates import RailTemplateRefs, load_rail_template


_EPS = 1e-9


@dataclass
class RailPlacement:
    slide_axis_world:    np.ndarray    # unit, OUTWARD
    width_axis_world:    np.ndarray    # unit, viewer-RIGHT
    height_axis_world:   np.ndarray    # unit, world-up direction
    drawer_center:       np.ndarray
    half_slide:          float
    half_width:          float
    half_height:         float
    side:                str            # "left" | "right"
    anchor_world:        np.ndarray     # world point where the rail's snap_static centroid lands


@dataclass
class RailSnapResult:
    root: bpy.types.Object
    refs: RailTemplateRefs
    placement: RailPlacement
    diagnostic: dict = field(default_factory=dict)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v));  return v / n if n > _EPS else v


def _vec(v) -> Vector:
    return Vector((float(v[0]), float(v[1]), float(v[2])))


def _mesh_face_normal_world(obj: bpy.types.Object) -> Vector:
    R = obj.matrix_world.to_3x3()
    acc = Vector((0, 0, 0))
    for p in obj.data.polygons:
        acc += (R @ p.normal) * p.area
    return acc.normalized() if acc.length > _EPS else Vector((0, 0, 1))


def _mesh_face_centroid_world(obj: bpy.types.Object) -> Vector:
    M = obj.matrix_world
    acc = Vector((0, 0, 0));  a = 0.0
    for p in obj.data.polygons:
        acc += (M @ p.center) * p.area;  a += p.area
    return (acc / a) if a > _EPS else Vector(M.to_translation())


def _quat_from_to(a: Vector, b: Vector) -> Quaternion:
    a = a.normalized();  b = b.normalized()
    d = float(a.dot(b))
    if d > 1.0 - 1e-9:
        return Quaternion()
    if d < -1.0 + 1e-9:
        helper = Vector((1, 0, 0)) if abs(a.x) < 0.9 else Vector((0, 1, 0))
        axis = a.cross(helper).normalized()
        return Quaternion(axis, math.pi)
    axis = a.cross(b).normalized()
    return Quaternion(axis, math.acos(max(-1.0, min(1.0, d))))


def _project_perp(v: Vector, axis: Vector) -> Vector:
    return v - axis * float(v.dot(axis))


def snap_rail_to_placement(
    template_path: Path,
    placement: RailPlacement,
    suffix: str,
    scale_factor: float = 1.0,
    snap_static_offset_from_back: float = 0.0,
) -> RailSnapResult:
    """snap_static_offset_from_back: distance (template-coords, will be
    multiplied by scale_factor internally) from the rail-mesh's back-most
    extent to the snap_static centroid along the slide axis. Used so we
    can position the rail's BACK extent at placement.anchor_world (rather
    than placing snap_static centroid there) -- this guarantees the rail
    doesn't protrude past the cabinet-back edge of the drawer body.
    """
    refs = load_rail_template(template_path, suffix=suffix)
    if scale_factor != 1.0:
        refs.root.scale = (refs.root.scale[0] * scale_factor,
                           refs.root.scale[1] * scale_factor,
                           refs.root.scale[2] * scale_factor)
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()

    target_axis     = _vec(placement.slide_axis_world).normalized()
    width           = _vec(placement.width_axis_world).normalized()
    # cabinet-wall-toward-drawer direction = +width on left side, -width on right
    target_static_n = width if placement.side == "left" else -width

    # Step 1: align template axis with target slide axis
    t_axis_world = (refs.axis.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    q_align = _quat_from_to(t_axis_world, target_axis)

    # Step 2: roll around target axis to align snap_static MESH face normal
    t_static_n = _mesh_face_normal_world(refs.snap_static)
    post_align_static = (q_align @ t_static_n).normalized()
    n_proj = _project_perp(post_align_static, target_axis)
    t_proj = _project_perp(target_static_n, target_axis)
    if n_proj.length < 0.05 or t_proj.length < 0.05:
        q_roll = Quaternion()
    else:
        n_proj.normalize();  t_proj.normalize()
        cos_r = float(n_proj.dot(t_proj))
        sin_r = float(target_axis.dot(n_proj.cross(t_proj)))
        q_roll = Quaternion(target_axis, math.atan2(sin_r, cos_r))

    refs.root.rotation_mode = "QUATERNION"
    refs.root.rotation_quaternion = q_roll @ q_align @ refs.root.rotation_quaternion
    bpy.context.view_layer.update()

    # Step 3: translate so the rail's BACK extent (along slide axis) lands
    # at placement.anchor_world. anchor_world is the body-back position.
    # cur_static_centroid is currently at some point. We want to put it at
    # anchor_world + snap_static_offset_from_back * scale_factor * slide_axis
    # (because rail-back = anchor_world; snap_static is offset_from_back ahead).
    cur_static_centroid = _mesh_face_centroid_world(refs.snap_static)
    target_static_centroid = _vec(placement.anchor_world) + \
        target_axis * (snap_static_offset_from_back * scale_factor)
    refs.root.location = refs.root.location + (target_static_centroid - cur_static_centroid)
    bpy.context.view_layer.update()

    return RailSnapResult(root=refs.root, refs=refs, placement=placement, diagnostic={
        "side": placement.side,
        "style": refs.style,
        "q_align_angle_deg": math.degrees(q_align.angle),
        "q_roll_angle_deg":  math.degrees(q_roll.angle),
        "snap_static_after": list(_mesh_face_normal_world(refs.snap_static)),
    })
