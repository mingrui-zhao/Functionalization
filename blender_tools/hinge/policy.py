"""Stage 2 (rewritten): rank candidates by geometry fit, not collision.

Per-candidate fit on three criteria (priority order):

  a) footprint_ratio = max(leaf_axial/face_axial, leaf_perp/face_perp).
     > 1 means the leaf overhangs the available mounting surface.

  b) coplanarity_residual_mm = distance from each leaf's snap plane to its
     target face plane in the closed pose, BEFORE refinement. Smaller =
     template lands more naturally; bigger = needs more refinement (refine
     can null it for full-rank cases, bound it for rank-deficient cases).

  c) axis_coverage_ratio = door_axial_extent / (hinge_count * leaf_axial).
     >= 1 means door is long enough for the planned hinge layout.

Composite ordering: hard penalty when footprint_ratio > 1.0; primary sort
ascending on coplanarity_residual_mm; tiebreak descending on axis_coverage.

No body_fragments. No swept volumes. No collision concept anywhere.
Validate.py remains a post-snap diagnostic; its verdicts surface in the
manifest but no longer feed back to candidate selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import HingePlacement, HingeType, OBB


# ---- inlined from multihinge.py to keep policy.py pure-numpy (no bpy dep) ----
# multihinge.py imports bpy; policy.py must not. Same logic, kept in sync.
def _hinge_count_for(hinge_type: HingeType, door_axial_extent_mm: float) -> int:
    if hinge_type is HingeType.FLAT:
        return 1 if door_axial_extent_mm <= 2000.0 else 2
    if door_axial_extent_mm < 600.0:
        return 2
    if door_axial_extent_mm < 1500.0:
        return 3
    return 4


_TEMPLATE_FOOTPRINTS_JSON_DEFAULT = (
    Path(__file__).resolve().parent / "template_footprints.json"
)
_TEMPLATE_NAME_TO_TYPE = {
    "hinge_interior_01": HingeType.INTERIOR,
    "hinge_exterior_01": HingeType.EXTERIOR,
    "hinge_flat_01":     HingeType.FLAT,
}


# ============================================================ dataclasses ==

@dataclass(frozen=True)
class TemplateLeafMeta:
    """Per-template leaf geometry, expressed in the pin frame.

    Pin frame: origin at the rotation_axis empty's world position, +Z along
    the axis direction, +X/+Y arbitrary orthonormal basis perpendicular.

    sizes are FULL extents in meters along (perp_x, perp_y, axial_z).
    snap_*_offset_pin: position of the snap plane center, from pin, in pin frame.
    snap_*_normal_pin: snap plane local +Z direction, projected into pin frame.
    """
    static_leaf_size_pin:    np.ndarray   # (3,) perp_x, perp_y, axial_z
    dynamic_leaf_size_pin:   np.ndarray
    snap_static_offset_pin:  np.ndarray   # (3,)
    snap_static_normal_pin:  np.ndarray   # (3,) unit
    snap_dynamic_offset_pin: np.ndarray
    snap_dynamic_normal_pin: np.ndarray


@dataclass
class FitScore:
    placement: HingePlacement
    footprint_ratio: float
    coplanarity_residual_mm: float
    axis_coverage_ratio: float
    composite_score: float
    diagnostic: dict = field(default_factory=dict)


# =================================================== template metadata ==

def load_template_metadata(
    json_path: Path | None = None,
) -> dict[HingeType, TemplateLeafMeta]:
    """Project template_footprints.json into TemplateLeafMeta records keyed
    by HingeType."""
    path = json_path or _TEMPLATE_FOOTPRINTS_JSON_DEFAULT
    raw = json.loads(path.read_text())
    out: dict[HingeType, TemplateLeafMeta] = {}
    for tname, htype in _TEMPLATE_NAME_TO_TYPE.items():
        if tname not in raw:
            continue
        rec = raw[tname]
        axis_world = np.asarray(rec["axis_world_direction"], dtype=float)
        # Reconstruct pin-frame basis used by measure_template_footprints.py.
        # (helper-cross-z convention; same construction must be used to
        # interpret the JSON's pin-frame quantities.)
        pin_x_world, pin_y_world, pin_z_world = _pin_frame_basis(axis_world)
        snap_s_world_n = np.asarray(rec["snap_static_world_normal"], dtype=float)
        snap_d_world_n = np.asarray(rec["snap_dynamic_world_normal"], dtype=float)
        snap_s_normal_pin = _world_to_pin(snap_s_world_n, pin_x_world, pin_y_world, pin_z_world)
        snap_d_normal_pin = _world_to_pin(snap_d_world_n, pin_x_world, pin_y_world, pin_z_world)
        # Use OBJECT ORIGIN (matrix_world.to_translation()), not AABB center.
        # snap.py's residual is computed from object origin, and for FLAT
        # the snap-plane mesh origins differ from their AABB centers.
        out[htype] = TemplateLeafMeta(
            static_leaf_size_pin=np.asarray(rec["static_leaf"]["size"], dtype=float),
            dynamic_leaf_size_pin=np.asarray(rec["dynamic_leaf"]["size"], dtype=float),
            snap_static_offset_pin=np.asarray(
                rec["snap_static"]["object_origin_pin_frame"], dtype=float),
            snap_static_normal_pin=_normalize(snap_s_normal_pin),
            snap_dynamic_offset_pin=np.asarray(
                rec["snap_dynamic"]["object_origin_pin_frame"], dtype=float),
            snap_dynamic_normal_pin=_normalize(snap_d_normal_pin),
        )
    return out


# Loaded lazily on first use; default arg of rank_by_fit pulls from this.
TEMPLATE_METADATA: dict[HingeType, TemplateLeafMeta] | None = None


def _ensure_metadata_loaded() -> dict[HingeType, TemplateLeafMeta]:
    global TEMPLATE_METADATA
    if TEMPLATE_METADATA is None:
        TEMPLATE_METADATA = load_template_metadata()
    return TEMPLATE_METADATA


# ============================================================ public entry ==

def rank_by_fit(
    candidates: list[HingePlacement],
    door_obb: OBB,
    panel_obb: OBB,
    template_metadata: dict[HingeType, TemplateLeafMeta] | None = None,
) -> list[FitScore]:
    """Score candidates by geometric fit. Returns ascending by composite_score
    (lower = better)."""
    metadata = template_metadata or _ensure_metadata_loaded()
    out: list[FitScore] = []
    for placement in candidates:
        meta = metadata.get(placement.hinge_type)
        if meta is None:
            # No metadata for this hinge type -- can't score. Skip.
            continue
        score = _score_one(placement, door_obb, panel_obb, meta)
        out.append(score)
    out.sort(key=lambda s: s.composite_score)
    return out


# ============================================================ scoring ==

def _score_one(
    placement: HingePlacement,
    door_obb: OBB,
    panel_obb: OBB,
    meta: TemplateLeafMeta,
) -> FitScore:
    fp_ratio, fp_diag = _footprint_ratio(placement, door_obb, panel_obb, meta)
    cop_residual_mm, cop_diag = _coplanarity_residual_mm(placement, meta)
    cov_ratio, cov_diag = _axis_coverage_ratio(placement, door_obb, meta)
    composite = _composite(fp_ratio, cop_residual_mm, cov_ratio)
    return FitScore(
        placement=placement,
        footprint_ratio=fp_ratio,
        coplanarity_residual_mm=cop_residual_mm,
        axis_coverage_ratio=cov_ratio,
        composite_score=composite,
        diagnostic={
            "footprint": fp_diag,
            "coplanarity": cop_diag,
            "axis_coverage": cov_diag,
        },
    )


def _footprint_ratio(
    placement: HingePlacement, door_obb: OBB, panel_obb: OBB,
    meta: TemplateLeafMeta,
) -> tuple[float, dict]:
    """Max ratio of leaf extent / available face extent, taken per-axis
    over (axial, in-plane perp) and over (static, dynamic) leaves."""
    axis = _normalize(np.asarray(placement.axis_direction, dtype=float))
    static_face_n = _normalize(np.asarray(placement.static_face.normal, dtype=float))
    dynamic_face_n = _normalize(np.asarray(placement.dynamic_face.normal, dtype=float))

    # Static leaf on panel's static_face.
    s_axial, s_perp = _leaf_face_extents(meta.static_leaf_size_pin, meta.snap_static_normal_pin)
    pf_axis_extent, pf_perp_extent = _face_extents_along_axis_and_perp(
        panel_obb, static_face_n, axis,
    )
    s_ratio_axial = s_axial / max(pf_axis_extent, 1e-9)
    s_ratio_perp = s_perp / max(pf_perp_extent, 1e-9)
    s_ratio = max(s_ratio_axial, s_ratio_perp)

    # Dynamic leaf on door's dynamic_face.
    d_axial, d_perp = _leaf_face_extents(meta.dynamic_leaf_size_pin, meta.snap_dynamic_normal_pin)
    df_axis_extent, df_perp_extent = _face_extents_along_axis_and_perp(
        door_obb, dynamic_face_n, axis,
    )
    d_ratio_axial = d_axial / max(df_axis_extent, 1e-9)
    d_ratio_perp = d_perp / max(df_perp_extent, 1e-9)
    d_ratio = max(d_ratio_axial, d_ratio_perp)

    overall = max(s_ratio, d_ratio)
    return overall, {
        "static_leaf_axial_m":  float(s_axial),
        "static_leaf_perp_m":   float(s_perp),
        "static_face_axial_m":  float(pf_axis_extent),
        "static_face_perp_m":   float(pf_perp_extent),
        "dynamic_leaf_axial_m": float(d_axial),
        "dynamic_leaf_perp_m":  float(d_perp),
        "dynamic_face_axial_m": float(df_axis_extent),
        "dynamic_face_perp_m":  float(df_perp_extent),
        "static_ratio":  float(s_ratio),
        "dynamic_ratio": float(d_ratio),
    }


def _coplanarity_residual_mm(
    placement: HingePlacement, meta: TemplateLeafMeta,
) -> tuple[float, dict]:
    """Distance from each snap-plane to its target face plane in the closed
    pose, expressed in mm.

    Identity:
        residual = | D_snap_along_normal_pin - D_pin_to_face |

    Both terms are scalar projections; no full rotation needed at runtime.
    """
    pin = np.asarray(placement.axis_origin, dtype=float)
    s_face_pt = np.asarray(placement.static_face.point, dtype=float)
    s_face_n = _normalize(np.asarray(placement.static_face.normal, dtype=float))
    d_face_pt = np.asarray(placement.dynamic_face.point, dtype=float)
    d_face_n = _normalize(np.asarray(placement.dynamic_face.normal, dtype=float))

    d_pin_to_static_face = float((s_face_pt - pin) @ s_face_n)
    d_pin_to_dynamic_face = float((d_face_pt - pin) @ d_face_n)

    d_snap_static_along_normal_pin = float(
        meta.snap_static_offset_pin @ meta.snap_static_normal_pin
    )
    d_snap_dynamic_along_normal_pin = float(
        meta.snap_dynamic_offset_pin @ meta.snap_dynamic_normal_pin
    )

    static_residual_m = abs(d_snap_static_along_normal_pin - d_pin_to_static_face)
    dynamic_residual_m = abs(d_snap_dynamic_along_normal_pin - d_pin_to_dynamic_face)

    overall_mm = max(static_residual_m, dynamic_residual_m) * 1000.0
    return overall_mm, {
        "static_residual_mm":  static_residual_m * 1000.0,
        "dynamic_residual_mm": dynamic_residual_m * 1000.0,
        "d_pin_to_static_face_mm":  d_pin_to_static_face * 1000.0,
        "d_pin_to_dynamic_face_mm": d_pin_to_dynamic_face * 1000.0,
    }


def _axis_coverage_ratio(
    placement: HingePlacement, door_obb: OBB, meta: TemplateLeafMeta,
) -> tuple[float, dict]:
    """door axial extent / (hinge_count * dynamic_leaf_axial). >= 1 = door
    long enough."""
    axis = _normalize(np.asarray(placement.axis_direction, dtype=float))
    door_extent_m = _box_extent_along(door_obb, axis)
    door_extent_mm = door_extent_m * 1000.0
    count = _hinge_count_for(placement.hinge_type, door_extent_mm)
    leaf_axial_m = float(meta.dynamic_leaf_size_pin[2])
    needed_m = count * leaf_axial_m
    if needed_m < 1e-9:
        return 1.0, {"hinge_count": count, "needed_m": 0.0,
                     "door_extent_m": door_extent_m, "ratio": 1.0}
    ratio = door_extent_m / needed_m
    return ratio, {
        "hinge_count": count,
        "leaf_axial_m": leaf_axial_m,
        "needed_m": needed_m,
        "door_extent_m": door_extent_m,
        "ratio": ratio,
    }


def _composite(footprint_ratio: float, coplanarity_mm: float,
               axis_coverage: float) -> float:
    """Sortable scalar: hard penalty for overhang, then coplanarity_mm
    primary (mm scale), then 1/coverage tiebreaker (sub-mm influence)."""
    overhang_penalty = 1e6 if footprint_ratio > 1.0 else 0.0
    coverage_inv = 1.0 / max(axis_coverage, 0.01)
    return overhang_penalty + coplanarity_mm + coverage_inv * 0.001


# ============================================================ helpers ==

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _pin_frame_basis(axis_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same construction as measure_template_footprints.py."""
    z = _normalize(axis_world)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = helper - z * float(helper @ z)
    x = _normalize(x)
    y = np.cross(z, x)
    return x, y, z


def _world_to_pin(v_world: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.array([float(v_world @ x), float(v_world @ y), float(v_world @ z)])


def _leaf_face_extents(
    leaf_size_pin: np.ndarray, snap_normal_pin: np.ndarray,
) -> tuple[float, float]:
    """For an axis-aligned-in-pin-frame leaf with extents (perp_x, perp_y,
    axial_z), and a snap normal direction (mostly perpendicular to axis),
    return (axial_extent, in_plane_perp_extent) -- the leaf's footprint
    dimensions on the surface it mounts to."""
    axial_extent = float(leaf_size_pin[2])
    # In-plane perp = the perpendicular direction NOT along snap_normal.
    # snap_normal_pin's XY components tell us which pin axis it aligns with.
    nx = abs(float(snap_normal_pin[0]))
    ny = abs(float(snap_normal_pin[1]))
    if nx > ny:
        # snap_normal mostly along pin X -> in-plane perp dim is pin Y.
        in_plane_perp = float(leaf_size_pin[1])
    else:
        in_plane_perp = float(leaf_size_pin[0])
    return axial_extent, in_plane_perp


def _box_extent_along(box: OBB, direction: np.ndarray) -> float:
    """Full extent of the OBB projected onto a unit `direction`."""
    e = np.asarray(box.half_extents, dtype=float)
    R = np.asarray(box.R, dtype=float)
    contributions = np.abs(R.T @ direction) * e
    return float(2.0 * contributions.sum())


def _face_extents_along_axis_and_perp(
    box: OBB, face_normal: np.ndarray, axis_dir: np.ndarray,
) -> tuple[float, float]:
    """For a box and one of its faces (identified by face_normal), return
    (extent_along_axis, extent_perp_to_both_normal_and_axis).

    Both directions lie in the face plane (perpendicular to face_normal).
    For axis-aligned boxes (R = I) these reduce to the box's full extents
    along the relevant world axes."""
    fn = _normalize(face_normal)
    axis = _normalize(axis_dir)
    # Project axis onto the face plane (subtract its component along fn).
    axis_in_face = axis - fn * float(axis @ fn)
    axis_in_face_n = _normalize(axis_in_face)
    if float(np.linalg.norm(axis_in_face)) < 1e-9:
        # axis is parallel to face normal -- shouldn't happen for a valid
        # mount face; pick an arbitrary in-plane direction.
        helper = np.array([1.0, 0.0, 0.0]) if abs(fn[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis_in_face_n = _normalize(helper - fn * float(helper @ fn))
    perp_in_face = np.cross(fn, axis_in_face_n)
    extent_along_axis = _box_extent_along(box, axis_in_face_n)
    extent_perp = _box_extent_along(box, perp_in_face)
    return extent_along_axis, extent_perp
