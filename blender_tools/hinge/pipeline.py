"""Stage 5: end-to-end pipeline. Parse URDF, classify, rank, snap, validate.

Per HSSD revolute joint:
    1. Build OBBs for door + panel from URDF.
    2. enumerate_placements -> all geometrically valid candidates.
    3. rank_by_fit -> sorted by geometry fit (footprint, coplanarity,
       axis coverage). No body_fragments, no swept-OBB collision.
    4. Snap the top-ranked candidate. refine=True by default.
    5. Merge door fragments to single mesh; parent under driver leaf.
    6. Run validate_snap as a post-snap diagnostic only -- its verdict is
       surfaced in the manifest but no longer drives candidate selection.

Defaults:
    refine=True. The HSSD URDF places joint axes at the cabinet's outer
    corner, ~18mm offset from where the actual hinge axis lives. Without
    refinement, every placement carries that 18mm error in its static
    snap. See test_snap_inside_blender.py and Stage 3 review notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import bpy  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore

from .geometry import (
    FaceRef, HingePlacement, HingeType, OBB, enumerate_placements,
)
from .merge import merge_fragments_to_single_mesh
from .multihinge import (
    MultiHingeResult,
    insert_multihinge,
    parent_door_to_driver,
)
from .policy import FitScore, rank_by_fit
from .validate import ValidationVerdict, validate_snap


_TEMPLATE_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "annotated_mechanical_parts"
_TEMPLATE_FOR_TYPE = {
    HingeType.INTERIOR: "hinge_interior_01.blend",
    HingeType.EXTERIOR: "hinge_exterior_01.blend",
    HingeType.FLAT:     "hinge_flat_01.blend",
}


@dataclass
class JointResult:
    joint_name: str
    chosen_type: HingeType
    candidates_ranked: list[FitScore]   # full ranked list, top = result[0]
    multi: MultiHingeResult
    validation: ValidationVerdict


# ─── Mesh-aware placement refinement ─────────────────────────────────────────

def _refine_face_against_mesh(face: FaceRef,
                                mesh_obj: bpy.types.Object,
                                max_dist: float = 1.0,
                                max_translation: float = 0.03) -> FaceRef:
    """Refine an OBB-derived face's POINT onto the actual mesh surface.

    Updates only `point`, NOT `normal`. The OBB-derived normal encodes the
    door/panel's principal mounting direction; replacing it with a hit
    polygon's normal is fragile (a beveled, routed, or rounded mesh face
    can produce a normal arbitrarily far from the cabinet's intended
    mounting direction, which then rolls the template into a wrong
    orientation and visibly breaks the door's swing axis).

    Strategy: ray-cast a small in-plane grid from outside the OBB face
    inward along -face.normal and keep the OUTERMOST hit within
    max_translation of the OBB face. A single center ray is not safe:
    on a face frame the face center is the opening, so the ray tunnels
    through and lands on whatever lies behind (back panel, shelf) —
    which then drags the whole hinge into the cabinet body via the
    refinement solve. The grid finds the stile surface instead, and the
    tight max_translation (leaf corrections are never more than a panel
    thickness) rejects tunnel hits even if every ray falls through.
    """
    if mesh_obj is None or mesh_obj.type != "MESH" or mesh_obj.data is None:
        return face
    if len(mesh_obj.data.polygons) == 0:
        return face
    deps = bpy.context.evaluated_depsgraph_get()
    try:
        bvh = BVHTree.FromObject(mesh_obj, deps)
    except Exception:
        return face
    n = np.asarray(face.normal, dtype=float)
    p = np.asarray(face.point, dtype=float)
    # In-plane orthonormal basis for the sample grid.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(n @ helper)) > 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    u = np.cross(n, helper); u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    direction = Vector((-float(n[0]), -float(n[1]), -float(n[2])))
    moves = []
    offsets = (0.0, 0.02, -0.02, 0.06, -0.06)
    for du in offsets:
        for dv in offsets:
            s = p + u * du + v * dv
            origin = Vector((float(s[0] + n[0] * 0.05),
                             float(s[1] + n[1] * 0.05),
                             float(s[2] + n[2] * 0.05)))
            location, _normal, _idx, _dist = bvh.ray_cast(origin, direction, max_dist)
            if location is None:
                continue
            hit = np.array([location.x, location.y, location.z], dtype=float)
            move = float(np.dot(hit - s, -n))   # positive = inward of OBB face
            if move < 0.0 or move > max_translation:
                continue
            moves.append(move)
    if not moves:
        return face
    # MEDIAN of the sampled surface depths: the outermost hit overshoots
    # onto proud trim/ornament lips, a single center ray falls into frame
    # openings — the median tracks the dominant (true mounting) surface
    # under the leaf footprint in both cases.
    med = float(np.median(moves))
    # Keep the face point on the original face-center line so downstream
    # plane math sees only the along-normal shift.
    return FaceRef(box_role=face.box_role, face_id=face.face_id,
                   point=p - n * med, normal=face.normal)


def _refine_placement_against_meshes(
    placement: HingePlacement,
    static_mesh_obj: bpy.types.Object | None,
    dynamic_mesh_obj: bpy.types.Object | None,
) -> HingePlacement:
    """Return a new HingePlacement whose static_face / dynamic_face are
    refined onto the actual panel / door mesh surfaces. axis_origin and
    axis_direction are unchanged (caller's pin_hint already encodes the
    hinge edge; mesh refinement only fixes the LEAF MOUNTING PLANES)."""
    new_static = _refine_face_against_mesh(placement.static_face,
                                            static_mesh_obj) \
                  if static_mesh_obj is not None else placement.static_face
    new_dynamic = _refine_face_against_mesh(placement.dynamic_face,
                                              dynamic_mesh_obj) \
                  if dynamic_mesh_obj is not None else placement.dynamic_face
    return HingePlacement(
        axis_origin=placement.axis_origin,
        axis_direction=placement.axis_direction,
        static_face=new_static,
        dynamic_face=new_dynamic,
        closed_angle=placement.closed_angle,
        hinge_type=placement.hinge_type,
        swing_range=placement.swing_range,
    )


def insert_hinges_for_joint(
    joint_name: str,
    door_obb: OBB,
    panel_obb: OBB,
    body_fragment_objs: list[bpy.types.Object],
    door_fragment_prefix: str,
    axis_hint: np.ndarray,
    pin_hint: np.ndarray,
    body_centroid: np.ndarray,
    swing_range: tuple[float, float],
    template_dir: Path = _TEMPLATE_DIR_DEFAULT,
    swing_samples: int = 20,
    force_hinge_type: HingeType | None = None,
    scale_factor: float | None = None,
    hinge_count: int | None = 2,
    body_obbs_for_avoidance: list[OBB] | None = None,
    static_mesh_obj: bpy.types.Object | None = None,
    dynamic_mesh_obj: bpy.types.Object | None = None,
) -> JointResult:
    """Run the full hinge-insertion pipeline for one revolute joint.

    body_fragment_objs is consumed only by validate.py for post-snap door-vs-
    body collision diagnostics. The geometry-fit ranker does NOT take body
    fragments -- fit is judged from door+panel geometry alone.

    force_hinge_type: if set, restricts candidates to only that hinge type
        before ranking. Raises if the geometry produces no candidate of
        that type.

    scale_factor: uniform multiplier on the hinge template's root.scale.
        - None (default): compute auto-scale per scale.compute_auto_scale
          (fits hinge to door/panel dimensions per per-type rules).
        - float: use this value directly. Useful when you want a specific
          visual size or are debugging.
    """
    placements = enumerate_placements(
        door_obb, panel_obb, axis_hint, pin_hint, body_centroid,
        swing_range=swing_range, tol_mm=10.0,
    )
    if force_hinge_type is not None:
        placements = [p for p in placements if p.hinge_type is force_hinge_type]
    if not placements:
        raise RuntimeError(
            f"joint {joint_name}: enumerate_placements returned no candidates"
            + (f" of type {force_hinge_type.value}" if force_hinge_type else "")
        )

    # Mesh-aware refinement: replace each candidate's static_face /
    # dynamic_face with the actual mesh surface (BVH ray-cast). Falls back
    # to the OBB face if the mesh isn't available or the ray misses.
    if static_mesh_obj is not None or dynamic_mesh_obj is not None:
        placements = [
            _refine_placement_against_meshes(
                p, static_mesh_obj, dynamic_mesh_obj
            ) for p in placements
        ]

    ranked = rank_by_fit(placements, door_obb, panel_obb)
    if not ranked:
        raise RuntimeError(
            f"joint {joint_name}: rank_by_fit returned empty (no template metadata?)"
        )

    # Snap the top-ranked candidate. validate.py runs once as a diagnostic.
    chosen = ranked[0]
    cand_idx = 0

    # Resolve scale_factor: None -> auto, float -> use as-is.
    chosen_template_path = template_dir / _TEMPLATE_FOR_TYPE[chosen.placement.hinge_type]
    if scale_factor is None:
        from .scale import compute_auto_scale
        scale_factor, scale_diag = compute_auto_scale(
            template_path=chosen_template_path,
            hinge_type=chosen.placement.hinge_type,
            door_obb=door_obb,
            panel_obb=panel_obb,
            placement=chosen.placement,
            probe_suffix=f"{joint_name}_scaleprobe",
        )
    else:
        scale_diag = {"hinge_type": chosen.placement.hinge_type.value,
                      "raw_scale": scale_factor, "clamped_scale": scale_factor,
                      "was_clamped": False, "source": "user_override"}

    cand_attempt = _try_one_candidate(
        cand=chosen,
        door_obb=door_obb,
        door_fragment_prefix=door_fragment_prefix,
        body_fragment_objs=body_fragment_objs,
        template_dir=template_dir,
        joint_name=joint_name,
        candidate_idx=cand_idx,
        swing_samples=swing_samples,
        scale_factor=scale_factor,
        hinge_count=hinge_count,
        body_obbs_for_avoidance=body_obbs_for_avoidance,
        panel_obb=panel_obb,
    )
    _chosen_score, multi, _door_obj, verdict = cand_attempt
    multi.diagnostic["scale"] = scale_diag

    return JointResult(
        joint_name=joint_name,
        chosen_type=chosen.placement.hinge_type,
        candidates_ranked=ranked,
        multi=multi,
        validation=verdict,
    )


# ============================================================ internals ==

def _try_one_candidate(
    cand: FitScore,
    door_obb: OBB,
    door_fragment_prefix: str,
    body_fragment_objs: list[bpy.types.Object],
    template_dir: Path,
    joint_name: str,
    candidate_idx: int,
    swing_samples: int,
    scale_factor: float = 1.0,
    hinge_count: int | None = 2,
    body_obbs_for_avoidance: list[OBB] | None = None,
    panel_obb: OBB | None = None,
) -> tuple[FitScore, MultiHingeResult, bpy.types.Object | None, ValidationVerdict]:
    """Snap, merge, parent, validate. Returns the result quad.

    No rollback in the new pipeline -- candidate selection happens once,
    validation is purely diagnostic. The function name is preserved in
    case a future fallback path is reintroduced."""
    template_path = template_dir / _TEMPLATE_FOR_TYPE[cand.placement.hinge_type]
    suffix = f"{joint_name}_c{candidate_idx}"

    multi = insert_multihinge(
        template_path=template_path,
        placement=cand.placement,
        door_obb=door_obb,
        suffix_base=suffix,
        refine=True,
        refine_threshold_mm=0.5,
        scale_factor=scale_factor,
        hinge_count=hinge_count,
        body_obbs_for_avoidance=body_obbs_for_avoidance,
    )

    # Cobordering: align dynamic snap-plane's pin-side axial edge with the
    # door dynamic face's pin-side outer edge. For FLAT we additionally
    # apply the static-side equivalent and average the two — this avoids
    # the leaf snapping all the way to the door's thin-edge border.
    from .coborder import apply_coborder_to_multihinge
    coborder_diag = apply_coborder_to_multihinge(
        multi.snap_results, cand.placement, door_obb, panel_obb=panel_obb,
    )
    multi.diagnostic["coborder"] = coborder_diag

    merged_door = merge_fragments_to_single_mesh(door_fragment_prefix)
    if merged_door is not None:
        parent_door_to_driver(merged_door, multi)

    mechanism_objs: list[bpy.types.Object] = []
    for sr in multi.snap_results:
        mechanism_objs.extend(_descendants_of(sr.root))

    verdict = validate_snap(
        door_obj=merged_door,
        mechanism_objs=mechanism_objs,
        body_fragment_objs=body_fragment_objs,
        placement=cand.placement,
        swing_samples=swing_samples,
    ) if merged_door is not None else _no_door_verdict()
    return cand, multi, merged_door, verdict


def _no_door_verdict() -> ValidationVerdict:
    return ValidationVerdict(
        status="warn",
        max_door_penetration_mm=0.0,
        max_mechanism_penetration_mm=0.0,
        worst_frame=None,
        diagnostic={"warning": "no door mesh found for prefix; validation skipped"},
    )


def _descendants_of(root: bpy.types.Object) -> list[bpy.types.Object]:
    out: list[bpy.types.Object] = []
    stack = [root]
    while stack:
        o = stack.pop()
        out.append(o)
        for child in o.children:
            stack.append(child)
    return out


# ============================================================ manifest ==

def joint_result_to_manifest_entry(jr: JointResult) -> dict:
    """JSON-safe dict for the per-joint manifest entry."""
    return {
        "joint": jr.joint_name,
        "chosen_hinge_type": jr.chosen_type.value,
        "candidates_considered_in_rank_order": [
            c.placement.hinge_type.value for c in jr.candidates_ranked
        ],
        "fit_scores": [
            {
                "hinge_type": c.placement.hinge_type.value,
                "composite_score": c.composite_score,
                "footprint_ratio": c.footprint_ratio,
                "coplanarity_residual_mm": c.coplanarity_residual_mm,
                "axis_coverage_ratio": c.axis_coverage_ratio,
            }
            for c in jr.candidates_ranked
        ],
        "multihinge": {
            "hinge_count": jr.multi.diagnostic.get("hinge_count"),
            "door_axial_extent_mm": jr.multi.diagnostic.get("door_axial_extent_mm"),
            "axial_offsets_mm": jr.multi.diagnostic.get("axial_offsets_mm"),
            "driver_index": jr.multi.driver_index,
            "snap_residuals_mm": [
                float(sr.residual_mm) for sr in jr.multi.snap_results
            ],
            "snap_refined": [bool(sr.refined) for sr in jr.multi.snap_results],
            "scale": jr.multi.diagnostic.get("scale"),
        },
        "validation": {
            "status": jr.validation.status,
            "max_door_penetration_mm": jr.validation.max_door_penetration_mm,
            "max_mech_penetration_mm": jr.validation.max_mechanism_penetration_mm,
            "worst_frame": jr.validation.worst_frame,
        },
    }
