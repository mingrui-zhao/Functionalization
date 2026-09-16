"""Stage 3: snap a hinge template onto a HingePlacement.

One hinge per call. No mesh merging, no door parenting, no multi-hinge --
those belong to Stage 4.

Algorithm: closed-form rigid transform on the template root, optionally
followed by a BVH-based linear-system refinement on translation only.

Read-once / compute / apply-once / read-once pattern:

    bpy.data.libraries.load() and any subsequent setattr on transforms do
    NOT trigger depsgraph evaluation. Reading children's matrix_world
    without an intervening view_layer.update() returns stale values. To
    avoid silent staleness bugs, snap_template_to_placement does the
    transform math entirely in numpy/mathutils, then applies it as a
    single batched setattr block, then calls view_layer.update() exactly
    once before reading residual back. Never interleave set/read cycles.

Closed-form 3-step transform:

    1. Align: rotate root so the rotation_axis empty's world +Z direction
       equals placement.axis_direction. Quaternion via cross product.
    2. Roll: rotate root around placement.axis_direction so snap_static's
       (post-align) world +Z aligns with placement.static_face.normal,
       projected into the plane perpendicular to the axis.
    3. Translate: solve root.location so the rotation_axis empty's world
       position equals placement.axis_origin, accounting for parent scale.

Edge cases for the roll step:

    - template_static_normal parallel to axis (within tolerance):
      template is mis-annotated for this placement -- the static leaf
      plane should contain the axis. Do NOT fall back to identity roll;
      that hides the bug. SnapResult returns residual_mm=inf and a string
      diagnostic. Caller skips the candidate or re-tries.
    - projected normals anti-parallel: 180-deg rotation needed. Two valid
      perpendicular axes; pick either.

BVH refinement (optional, refine=True):

    Solves a 3x3 linear system on Delta-translation:
        (snap_static_world  + delta - static_face.point ) . static_face.n  = 0
        (snap_dynamic_world + delta - dynamic_face.point) . dynamic_face.n = 0
        delta . axis_direction = 0   (no slide along axis)
    Singular when the three normals are linearly dependent (axis parallel
    to a face normal -- topology error upstream); diagnostic notes the
    skip and the original residual is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import bpy  # type: ignore
from mathutils import Matrix, Quaternion, Vector  # type: ignore

from .geometry import HingePlacement
from .templates import TemplateRefs, load_template


_AXIS_PARALLEL_DOT = 0.99   # |projected_normal . axis| above this -> parallel
_NUMERIC_EPS = 1e-9


@dataclass
class SnapResult:
    root: bpy.types.Object
    dynamic_leaf: bpy.types.Object
    static_leaf: bpy.types.Object
    residual_mm: float
    refined: bool
    diagnostic: dict = field(default_factory=dict)
    snap_static:  bpy.types.Object | None = None
    snap_dynamic: bpy.types.Object | None = None
    axis: bpy.types.Object | None = None


def snap_template_to_placement(
    template_path: Path,
    placement: HingePlacement,
    suffix: str,
    refine: bool = False,
    refine_threshold_mm: float = 2.0,
    scale_factor: float = 1.0,
) -> SnapResult:
    """Append template, compute the unique root transform mapping (template
    axis frame) -> (placement frame), apply, return refs + residual.

    Args:
        template_path: .blend file to append from.
        placement: target hinge placement in body frame.
        suffix: name suffix for collision-avoidance (e.g. joint name).
        refine: if True and residual exceeds threshold, run the
            3-equation BVH-style linear refinement on translation.
        refine_threshold_mm: minimum residual to trigger refinement.
        scale_factor: uniform multiplier applied to root.scale BEFORE the
            snap math reads template state. 1.0 = no change. Caller is
            responsible for choosing a sensible factor (see scale.py).
    """
    refs = load_template(template_path, suffix=suffix)
    if scale_factor != 1.0:
        refs.root.scale = (refs.root.scale[0] * scale_factor,
                           refs.root.scale[1] * scale_factor,
                           refs.root.scale[2] * scale_factor)

    # ===== READ ONCE: capture template state before any modification =====
    # Force frame 0 first. snap_dynamic is parented to the dynamic_leaf,
    # which has rotation_euler fcurves (the door swing animation). Reading
    # snap_dynamic.matrix_world at any other frame yields the leaf's pose
    # mid-swing, not the closed-pose annotation, which corrupts both the
    # residual measurement and the BVH-refinement linear solve. Frame 0 is
    # the convention for "template at closed pose".
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    template_axis_dir = _normalize_v(
        refs.axis.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
    )
    # snap_static is a mesh rectangle (4 verts, 1 polygon). Its TRUE plane
    # normal is the mesh face normal, not the object transform's local +Z.
    # In INT/EXT these happen to agree; in FLAT they differ by 90 deg
    # because the rectangle was authored in the leaf's surface plane but the
    # object transform was rotated independently. Read the face normal.
    template_static_normal_world = _snap_plane_normal_world(refs.snap_static)
    # Local position of the axis empty in root's local frame -- used to
    # compute root.location after orientation is set.
    axis_local_pos = refs.root.matrix_world.inverted() @ refs.axis.matrix_world.to_translation()
    # Convert root.rotation_mode to QUATERNION safely. Blender does NOT
    # auto-convert keys/properties when rotation_mode changes; if root was
    # in XYZ mode, rotation_quaternion may be stale. Sync it explicitly
    # from the active mode's value before reading.
    original_root_q = _ensure_quaternion_mode(refs.root)
    # Snapshot scale matrix for the translation solve. matrix_world.to_scale()
    # is wrong for negative determinants (returns flipped signs); we read
    # the .scale property directly which is always correct.
    root_scale_diag = Vector(refs.root.scale)

    # ===== COMPUTE: axis-align quaternion (step 1) =====
    target_axis_dir = Vector(_to_list(placement.axis_direction))
    q_align = _quat_from_to(template_axis_dir, target_axis_dir)

    # ===== COMPUTE: roll quaternion around target axis (step 2) =====
    # Apply q_align to template_static_normal_world to get the post-align
    # direction, then project both into the plane perpendicular to target.
    post_align_static_normal = _normalize_v(q_align @ template_static_normal_world)
    target_static_normal = Vector(_to_list(placement.static_face.normal))

    diagnostic: dict = {}
    n_proj = _project_perp(post_align_static_normal, target_axis_dir)
    t_proj = _project_perp(target_static_normal, target_axis_dir)
    if n_proj.length < 0.05 or t_proj.length < 0.05:
        # template_static_normal parallel to axis -- annotation mismatch.
        diagnostic["error"] = (
            "template_static_normal parallel to placement axis; "
            "cannot determine roll. template likely mis-annotated for "
            "this placement type."
        )
        return SnapResult(
            root=refs.root,
            dynamic_leaf=refs.dynamic_leaf,
            static_leaf=refs.static_leaf,
            residual_mm=float("inf"),
            refined=False,
            diagnostic=diagnostic,
        )
    n_proj.normalize()
    t_proj.normalize()
    cos_roll = float(n_proj.dot(t_proj))
    sin_roll = float(target_axis_dir.dot(n_proj.cross(t_proj)))
    roll_angle = float(np.arctan2(sin_roll, cos_roll))
    q_roll = Quaternion(target_axis_dir, roll_angle)

    # ===== COMPUTE: full root quaternion =====
    q_full = q_roll @ q_align @ original_root_q

    # ===== COMPUTE: translation =====
    # axis_world = root.location + R_root @ S_root @ axis_local_pos
    # We want axis_world = placement.axis_origin, given the new R_root.
    R_full_mat = q_full.to_matrix()
    scaled_axis_local = Vector((
        axis_local_pos.x * root_scale_diag.x,
        axis_local_pos.y * root_scale_diag.y,
        axis_local_pos.z * root_scale_diag.z,
    ))
    new_location = (
        Vector(_to_list(placement.axis_origin)) - R_full_mat @ scaled_axis_local
    )

    # ===== APPLY ONCE =====
    refs.root.rotation_mode = "QUATERNION"
    refs.root.rotation_quaternion = q_full
    refs.root.location = new_location
    bpy.context.view_layer.update()

    # ===== READ ONCE: residual on dynamic-side snap plane =====
    residual_mm, residual_diag = _measure_residual(refs, placement)
    diagnostic.update(residual_diag)

    refined = False
    static_residual_mm = diagnostic.get("static_residual_mm", 0.0)
    needs_refine = max(residual_mm, static_residual_mm) > refine_threshold_mm
    if refine and needs_refine:
        delta, refine_diag = _solve_refine_translation(refs, placement)
        diagnostic["refine"] = refine_diag
        if delta is not None:
            refs.root.location = refs.root.location + delta
            bpy.context.view_layer.update()
            residual_mm, residual_diag2 = _measure_residual(refs, placement)
            diagnostic["residual_after_refine"] = residual_diag2
            refined = True

    diagnostic["residual_mm"] = residual_mm
    return SnapResult(
        root=refs.root,
        dynamic_leaf=refs.dynamic_leaf,
        static_leaf=refs.static_leaf,
        residual_mm=residual_mm,
        refined=refined,
        diagnostic=diagnostic,
        snap_static=refs.snap_static,
        snap_dynamic=refs.snap_dynamic,
        axis=refs.axis,
    )


# ============================================================ helpers ==

def _normalize_v(v: Vector) -> Vector:
    n = v.length
    return Vector((0.0, 0.0, 0.0)) if n < _NUMERIC_EPS else v / n


def _snap_plane_normal_world(obj: bpy.types.Object) -> Vector:
    """Return the world-space normal of a snap_plane object.

    snap_plane objects are mesh rectangles (4 verts, 1 polygon). Their
    geometric plane normal is the mesh face normal, NOT the object's local
    +Z. These differ if the object was rotated independently of the mesh
    geometry -- which is the case in the FLAT template where the object
    +Z is 90 deg off from the actual rectangle's normal.

    Falls back to the object transform's local +Z if obj is not a mesh or
    has no polygons (e.g. an Empty placeholder)."""
    if obj.type == "MESH" and obj.data is not None and len(obj.data.polygons) > 0:
        R = obj.matrix_world.to_3x3()
        acc = Vector((0.0, 0.0, 0.0))
        for p in obj.data.polygons:
            acc += (R @ p.normal) * p.area
        return _normalize_v(acc)
    return _normalize_v(obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0)))


def _snap_plane_position_world(obj: bpy.types.Object) -> Vector:
    """Return the world-space position of a snap_plane object's mesh face.

    For a mesh rectangle, the geometrically meaningful "position" is the
    mesh face centroid, NOT the object's transform origin. They diverge
    when the rectangle's vertices were authored offset from the object
    origin -- which is the case in the FLAT template where the mesh
    centroid is ~14mm offset from the object origin. Using the object
    origin in residual / refinement causes the mesh to land that far off
    the target face plane (the "half-sunk hinge" symptom).

    Falls back to the object transform's translation if obj is not a mesh
    or has no polygons."""
    if obj.type == "MESH" and obj.data is not None and len(obj.data.polygons) > 0:
        M = obj.matrix_world
        acc = Vector((0.0, 0.0, 0.0))
        total_area = 0.0
        for p in obj.data.polygons:
            acc += (M @ p.center) * p.area
            total_area += p.area
        return acc / total_area if total_area > 0 else Vector(M.to_translation())
    return Vector(obj.matrix_world.to_translation())


def _ensure_quaternion_mode(obj: bpy.types.Object) -> Quaternion:
    """Switch obj.rotation_mode to QUATERNION, syncing rotation_quaternion
    from whichever mode was active. Returns the current quaternion.

    Blender does NOT auto-convert when rotation_mode changes -- without
    the explicit sync, switching from XYZ to QUATERNION leaves
    rotation_quaternion at whatever stale value the .blend file stored
    (often identity), losing the actual orientation."""
    if obj.rotation_mode == "QUATERNION":
        return obj.rotation_quaternion.copy()
    if obj.rotation_mode == "AXIS_ANGLE":
        # axis_angle is (angle, x, y, z); build a quat from it.
        aa = obj.rotation_axis_angle
        q = Quaternion((aa[1], aa[2], aa[3]), aa[0])
    else:
        # XYZ / XZY / YXZ / etc -- all eulers.
        q = obj.rotation_euler.to_quaternion()
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = q
    return q.copy()


def _project_perp(v: Vector, axis: Vector) -> Vector:
    """Component of v perpendicular to (unit) axis."""
    return v - axis * float(v.dot(axis))


def _quat_from_to(v_from: Vector, v_to: Vector) -> Quaternion:
    """Shortest-arc rotation taking v_from to v_to. Both must be unit."""
    d = float(v_from.dot(v_to))
    if d > 1.0 - 1e-9:
        return Quaternion()  # identity
    if d < -1.0 + 1e-9:
        # Anti-parallel: 180-deg rotation around any axis perpendicular to v_from.
        helper = Vector((1.0, 0.0, 0.0))
        if abs(v_from.dot(helper)) > 0.9:
            helper = Vector((0.0, 1.0, 0.0))
        axis = _normalize_v(v_from.cross(helper))
        return Quaternion(axis, np.pi)
    axis = _normalize_v(v_from.cross(v_to))
    angle = float(np.arccos(np.clip(d, -1.0, 1.0)))
    return Quaternion(axis, angle)


def _to_list(arr) -> list[float]:
    """Coerce numpy array / mathutils Vector / sequence to a plain list."""
    return [float(x) for x in arr]


def _measure_residual(refs: TemplateRefs, placement: HingePlacement) -> tuple[float, dict]:
    """Distance from snap_dynamic world position to placement.dynamic_face plane,
    in mm. Also reports static-side residual and pin-position error for
    diagnosis."""
    # Use mesh face centroid, not object origin -- they diverge for the FLAT
    # template by ~14mm and that offset is what causes the "half-sunk hinge"
    # symptom when the object origin lands on the panel face but the actual
    # mesh rectangle ends up inside the panel volume.
    dyn_world = _snap_plane_position_world(refs.snap_dynamic)
    dyn_face_point = Vector(_to_list(placement.dynamic_face.point))
    dyn_face_n = Vector(_to_list(placement.dynamic_face.normal))
    dyn_residual = abs(float((dyn_world - dyn_face_point).dot(dyn_face_n))) * 1000.0

    static_world = _snap_plane_position_world(refs.snap_static)
    static_face_point = Vector(_to_list(placement.static_face.point))
    static_face_n = Vector(_to_list(placement.static_face.normal))
    static_residual = abs(float((static_world - static_face_point).dot(static_face_n))) * 1000.0

    pin_world = refs.axis.matrix_world.to_translation()
    target_pin = Vector(_to_list(placement.axis_origin))
    pin_err = float((pin_world - target_pin).length) * 1000.0

    axis_world_dir = _normalize_v(refs.axis.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0)))
    target_axis = Vector(_to_list(placement.axis_direction))
    axis_dot = float(axis_world_dir.dot(target_axis))

    return dyn_residual, {
        "dynamic_residual_mm": dyn_residual,
        "static_residual_mm": static_residual,
        "pin_position_error_mm": pin_err,
        "axis_alignment_dot": axis_dot,
    }


def _solve_refine_translation(
    refs: TemplateRefs,
    placement: HingePlacement,
) -> tuple[Vector | None, dict]:
    """Solve the 3x3 linear system on translation delta via lstsq.

    System (always 3 equations on 3 unknowns):
        delta . static_face.normal  = (static_face.point - snap_static_world) . static_face.normal
        delta . dynamic_face.normal = (dynamic_face.point - snap_dynamic_world) . dynamic_face.normal
        delta . axis_direction      = 0

    For closed-angle-0 hinges (true exterior) the static and dynamic normals
    are anti-parallel, making the matrix singular. lstsq returns the
    minimum-norm least-squares solution -- best-fit delta given the
    underdetermined constraints. Residuals will be nonzero in this case
    (template leaf gap doesn't match placement leaf gap exactly), which
    is the expected limitation, not a bug.
    """
    s_n = np.asarray(_to_list(placement.static_face.normal), dtype=float)
    d_n = np.asarray(_to_list(placement.dynamic_face.normal), dtype=float)
    a_d = np.asarray(_to_list(placement.axis_direction), dtype=float)

    # Mesh face centroid, not object origin (see _snap_plane_position_world).
    s_w = np.asarray(_to_list(_snap_plane_position_world(refs.snap_static)), dtype=float)
    d_w = np.asarray(_to_list(_snap_plane_position_world(refs.snap_dynamic)), dtype=float)
    s_p = np.asarray(_to_list(placement.static_face.point), dtype=float)
    d_p = np.asarray(_to_list(placement.dynamic_face.point), dtype=float)

    A = np.stack([s_n, d_n, a_d], axis=0)
    b = np.array([
        float((s_p - s_w) @ s_n),
        float((d_p - d_w) @ d_n),
        0.0,
    ])
    delta, _resids, rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    rank = int(rank)
    diag: dict = {
        "delta_mm": [float(x) * 1000.0 for x in delta],
        "system_rank": rank,
        "system_det": float(np.linalg.det(A)),
    }
    if rank < 3:
        # Underdetermined system. Most common case: closed-angle-0 (true
        # exterior) hinge where static and dynamic face normals are
        # anti-parallel, collapsing to one effective face constraint.
        diag["rank_deficient_reason"] = (
            f"rank-{rank} system: face normal constraints linearly "
            f"dependent. lstsq returns minimum-norm best-fit; residuals "
            f"will be bounded by half the template/placement leaf-gap "
            f"mismatch."
        )
    # Consistency gate: accept the delta only if it actually SATISFIES the
    # constraints. A large-but-consistent delta is a legitimate correction
    # (e.g. translating from the pin hint on the door's outer face to the
    # door-back/panel seam — one full door thickness). An inconsistent
    # system (large post-fit residual) means the two face targets disagree
    # — e.g. one of them was corrupted by a refinement ray tunneling
    # through a face-frame opening — and the min-norm compromise would
    # trade the verified pin position for contact with a bogus plane.
    # In that case keep the pin-exact placement.
    fit_res = A @ np.asarray(delta, dtype=float) - b
    diag["fit_residual_mm"] = [float(x) * 1000.0 for x in fit_res]
    _FIT_TOL_M = 0.005
    _ABS_CAP_M = 0.10
    if float(np.max(np.abs(fit_res))) > _FIT_TOL_M:
        diag["rejected"] = (
            "constraints inconsistent (post-fit residual "
            f"{float(np.max(np.abs(fit_res)))*1000:.1f}mm > {_FIT_TOL_M*1000:.0f}mm); "
            "keeping pin-exact placement")
        return None, diag
    if float(np.linalg.norm(delta)) > _ABS_CAP_M:
        diag["rejected"] = "delta exceeds 10cm sanity cap; keeping pin-exact placement"
        return None, diag
    return Vector((float(delta[0]), float(delta[1]), float(delta[2]))), diag
