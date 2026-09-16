"""Stage 1: geometry-only classification and placement enumeration.

Pure numpy. No bpy imports. Operates in the body link's local frame, which
equals the world frame inside the decomposed .blend.

Convention (canonical):
    axis_direction is signed such that, by the right-hand rule, positive
    delta moves the door's far edge AWAY from body_centroid. All callers
    must pass axes through axis_with_canonical_sign() before storing them
    on a HingePlacement.

Stage 2 (policy.py) ranks the placements this module emits by collision
rate. This module emits *all* geometrically valid candidates with no
preference filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


# --------------------------------------------------------------- enums --

class HingeType(Enum):
    EXTERIOR = "exterior"   # closed-state leaf angle 0   (coplanar, opposite normals)
    INTERIOR = "interior"   # closed-state leaf angle pi/2 (perpendicular)
    FLAT     = "flat"       # closed-state leaf angle pi   (coplanar, same normals)


class Topology(Enum):
    EDGE_CONTACT = "edge"   # cases 1, 2, 4 -- D and P share an edge
    FACE_CONTACT = "face"   # case 3        -- D's face flush against P's face
    UNKNOWN      = "unknown"


class BoxRole(Enum):
    DOOR  = "door"
    PANEL = "panel"


# Closed-state leaf-to-leaf angles (radians) per hinge type.
_CLOSED_ANGLE = {
    HingeType.EXTERIOR: 0.0,
    HingeType.INTERIOR: np.pi / 2.0,
    HingeType.FLAT:     np.pi,
}


# ---------------------------------------------------------- dataclasses --

@dataclass(frozen=True)
class OBB:
    """Oriented bounding box in body frame.

    center: (3,) -- box center in body frame.
    half_extents: (3,) -- half-sizes along the box's local axes.
    R: (3, 3) -- columns are the box's local axes expressed in body frame.
                 Identity for the common axis-aligned case.
    """
    center: np.ndarray
    half_extents: np.ndarray
    R: np.ndarray


@dataclass(frozen=True)
class FaceRef:
    """One of an OBB's six faces, with cached plane equation.

    box_role: which side of the hinge this face belongs to.
    face_id:  0..5, encoding (axis, sign) as 2*axis + (1 if positive face else 0).
    point:    (3,) -- the face center in body frame.
    normal:   (3,) -- unit outward normal in body frame.
    """
    box_role: BoxRole
    face_id: int
    point: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class HingePlacement:
    """A complete spec for one hinge insertion."""
    axis_origin: np.ndarray
    axis_direction: np.ndarray
    static_face: FaceRef
    dynamic_face: FaceRef
    closed_angle: float
    hinge_type: HingeType
    swing_range: tuple[float, float]


# ============================================================ helpers ==

_PARALLEL_DOT = 0.985    # |dot| above this -> treat as parallel (~10 deg slop)
_PERP_DOT     = 0.15     # |dot| below this -> treat as perpendicular
_TOL_EPS      = 1e-6     # absolute fudge added to tol for boundary comparisons


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _box_faces(box: OBB, role: BoxRole) -> list[FaceRef]:
    """Six FaceRefs for an OBB in canonical (axis, sign) order.

    face_id = 2*axis + (1 if sign==+1 else 0).
    """
    out = []
    for axis in range(3):
        for sign in (-1, +1):
            normal = sign * box.R[:, axis]
            point = box.center + sign * box.half_extents[axis] * box.R[:, axis]
            face_id = 2 * axis + (1 if sign == +1 else 0)
            out.append(FaceRef(box_role=role, face_id=face_id, point=point, normal=normal))
    return out


def _face_axes(box: OBB, face: FaceRef) -> tuple[int, int]:
    """Indices of the two in-plane local axes for a face."""
    axis = face.face_id // 2
    return tuple(i for i in range(3) if i != axis)


def _is_broad_face(box: OBB, face: FaceRef) -> bool:
    """True when this face is perpendicular to the box's smallest extent.

    For a slab, the broad face is the largest face -- the one whose normal
    points along the slab thickness direction. Used to distinguish true
    face-to-face contact (case 3, both broad) from end-to-end / corner
    contact between non-broad faces (cases 1, 2, 4).
    """
    axis = face.face_id // 2
    return axis == int(np.argmin(box.half_extents))


def _face_corners(box: OBB, face: FaceRef) -> np.ndarray:
    """Return (4, 3) array of face corners in body frame, CCW around outward normal."""
    a, b = _face_axes(box, face)
    ea = box.half_extents[a] * box.R[:, a]
    eb = box.half_extents[b] * box.R[:, b]
    return np.array([
        face.point - ea - eb,
        face.point + ea - eb,
        face.point + ea + eb,
        face.point - ea + eb,
    ])


def _face_extents(box: OBB, face: FaceRef) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Return (u_axis_world, u_half, v_axis_world, v_half) for a face."""
    a, b = _face_axes(box, face)
    return box.R[:, a], box.half_extents[a], box.R[:, b], box.half_extents[b]


def _project_to_plane_2d(
    points3d: np.ndarray,
    plane_origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> np.ndarray:
    """Return 2D coords (N, 2) of points projected onto the (u, v) basis at origin."""
    rel = points3d - plane_origin
    return np.column_stack([rel @ u_axis, rel @ v_axis])


def _polygon_distance_2d(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Signed-style separation between two convex 2D polygons by SAT.

    Returns:
        <= 0  -- polygons overlap (negative = overlap depth, magnitude not exact).
        > 0   -- polygons disjoint, value is the maximum gap across separating axes.
    """
    max_sep = -np.inf
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            nn = np.linalg.norm(normal)
            if nn < 1e-12:
                continue
            normal /= nn
            pa = poly_a @ normal
            pb = poly_b @ normal
            sep = max(pb.min() - pa.max(), pa.min() - pb.max())
            if sep > max_sep:
                max_sep = sep
    return float(max_sep)


def _line_distance_to_face_2d(
    box: OBB, face: FaceRef, line_point: np.ndarray, line_dir: np.ndarray,
) -> float:
    """Distance from a line (assumed lying in face's plane) to face's rectangle.

    Returns 0 if the line crosses the rectangle, else the perpendicular gap
    in the plane. Caller must guarantee the line lies in face.plane (within tol).
    """
    u_axis, u_half, v_axis, v_half = _face_extents(box, face)
    # Project line into face's local 2D frame.
    rel = line_point - face.point
    p2 = np.array([rel @ u_axis, rel @ v_axis])
    d2 = np.array([line_dir @ u_axis, line_dir @ v_axis])
    nd = np.linalg.norm(d2)
    if nd < 1e-9:
        # Line direction is normal to the face -- treat as point-to-rect distance.
        dx = max(0.0, abs(p2[0]) - u_half)
        dy = max(0.0, abs(p2[1]) - v_half)
        return float(np.hypot(dx, dy))
    d2 /= nd
    # Build a 2D normal to the line and compute signed distance per corner.
    n2 = np.array([-d2[1], d2[0]])
    corners = np.array([
        [-u_half, -v_half],
        [+u_half, -v_half],
        [+u_half, +v_half],
        [-u_half, +v_half],
    ])
    signed = (corners - p2) @ n2
    if signed.min() <= 0.0 <= signed.max():
        # Line crosses the rectangle's interior.
        return 0.0
    return float(min(abs(signed.min()), abs(signed.max())))


def _plane_intersection_line(
    p1: np.ndarray, n1: np.ndarray, p2: np.ndarray, n2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersection line of two perpendicular planes.

    Returns (point_on_line, unit_direction). n1 must be perpendicular to n2.
    """
    direction = _normalize(np.cross(n1, n2))
    # n2 lies in plane1 (since n2 perp n1 and plane1 perp n1). Walk from p1
    # along n2 until we hit plane2.
    alpha = float((p2 - p1) @ n2)
    point = p1 + alpha * n2
    return point, direction


def _orthonormal_basis_in_plane(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) -- two orthonormal vectors spanning the plane perp to normal."""
    n = _normalize(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(np.cross(n, helper))
    v = np.cross(n, u)
    return u, v


# ============================================ topology classification ==

def classify_topology(
    door_obb: OBB,
    panel_obb: OBB,
    tol_mm: float = 10.0,
) -> Topology:
    """Detect EDGE_CONTACT, FACE_CONTACT, or UNKNOWN.

    EDGE_CONTACT:
        Some face of D and some face of P have perpendicular normals, AND
        their plane-intersection line lies (within tol) inside both face
        rectangles. Covers cases 1, 2.
        Also: parallel-same-direction coplanar faces whose footprints are
        adjacent without overlapping (case 4).

    FACE_CONTACT:
        Some face of D and some face of P have anti-parallel normals, are
        coplanar (within tol), and their 2D footprints overlap.
        Covers case 3.
    """
    tol = tol_mm * 1e-3  # meters

    door_faces = _box_faces(door_obb, BoxRole.DOOR)
    panel_faces = _box_faces(panel_obb, BoxRole.PANEL)

    found_edge = False
    found_face = False

    for fD in door_faces:
        for fP in panel_faces:
            dot = float(np.clip(fD.normal @ fP.normal, -1.0, 1.0))

            if dot <= -_PARALLEL_DOT:
                # Anti-parallel: candidate FACE_CONTACT.
                signed = float((fP.point - fD.point) @ fD.normal)
                if abs(signed) > tol:
                    continue
                # Both faces in (approximately) the same plane. Check footprint overlap.
                u, v = _orthonormal_basis_in_plane(fD.normal)
                pa = _project_to_plane_2d(_face_corners(door_obb, fD), fD.point, u, v)
                pb = _project_to_plane_2d(_face_corners(panel_obb, fP), fD.point, u, v)
                sep = _polygon_distance_2d(pa, pb)
                if sep <= tol:
                    if sep < -tol:
                        # Real area overlap. Distinguish case 3 (broad-on-broad,
                        # true face contact) from case 4 / corner geometries
                        # where one or both contact faces are end faces of a slab.
                        if _is_broad_face(door_obb, fD) and _is_broad_face(panel_obb, fP):
                            found_face = True
                        else:
                            found_edge = True
                    else:
                        # Touching only (no interior overlap) -- treat as edge.
                        found_edge = True
                continue

            if dot >= _PARALLEL_DOT:
                # Same direction parallel: candidate case-4 edge contact.
                signed = float((fP.point - fD.point) @ fD.normal)
                if abs(signed) > tol:
                    continue
                u, v = _orthonormal_basis_in_plane(fD.normal)
                pa = _project_to_plane_2d(_face_corners(door_obb, fD), fD.point, u, v)
                pb = _project_to_plane_2d(_face_corners(panel_obb, fP), fD.point, u, v)
                sep = _polygon_distance_2d(pa, pb)
                # Adjacent without interior overlap = sep ~0 with no negative depth.
                if abs(sep) <= tol:
                    found_edge = True
                continue

            if abs(dot) <= _PERP_DOT:
                # Perpendicular: candidate cases 1, 2.
                line_pt, _line_dir = _plane_intersection_line(
                    fD.point, fD.normal, fP.point, fP.normal,
                )
                # Edge direction is along cross(nD, nP).
                line_dir = _normalize(np.cross(fD.normal, fP.normal))
                if np.linalg.norm(line_dir) < 1e-9:
                    continue
                dD = _line_distance_to_face_2d(door_obb, fD, line_pt, line_dir)
                dP = _line_distance_to_face_2d(panel_obb, fP, line_pt, line_dir)
                if dD <= tol + _TOL_EPS and dP <= tol + _TOL_EPS:
                    found_edge = True

    if found_face:
        return Topology.FACE_CONTACT
    if found_edge:
        return Topology.EDGE_CONTACT
    return Topology.UNKNOWN


# ====================================== axis sign canonicalization ==

def axis_with_canonical_sign(
    axis: np.ndarray,
    pin: np.ndarray,
    door_obb: OBB,
    body_centroid: np.ndarray,
) -> np.ndarray:
    """Return axis (possibly negated) so RH-rule rotation opens the door.

    'Open' means the door's far edge moves away from body_centroid. Algorithm:
      1. Find the door's far corner -- the corner of door_obb farthest from
         pin in the plane perpendicular to axis through pin.
      2. Motion direction for infinitesimal positive delta:
             motion = cross(axis, far_edge - pin)
      3. Outward direction from body:
             outward = far_edge - body_centroid
      4. If dot(motion, outward) > 0, return axis. Else return -axis.
    """
    axis_n = _normalize(axis)

    # Generate the 8 corners of door_obb in body frame.
    e = door_obb.half_extents
    R = door_obb.R
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    corners = door_obb.center + signs * e @ R.T  # (8, 3)

    # Project each corner onto the plane perpendicular to axis through pin,
    # then measure radial distance from the pin.
    rel = corners - pin
    along = rel @ axis_n
    in_plane = rel - np.outer(along, axis_n)
    radial = np.linalg.norm(in_plane, axis=1)
    far_idx = int(np.argmax(radial))
    far_corner = corners[far_idx]

    motion = np.cross(axis_n, far_corner - pin)
    outward = far_corner - body_centroid
    if motion @ outward > 0.0:
        return axis_n
    return -axis_n


# =============================================== corner ownership ==

def corner_ownership(
    door_obb: OBB,
    panel_obb: OBB,
    shared_edge_dir: np.ndarray,
    tol_mm: float = 10.0,
) -> str:
    """For EDGE_CONTACT geometry, decide which box reaches further past
    the shared edge. Returns 'door' (case 1) or 'panel' (case 2 / case 4).

    Heuristic: project each box's centroid onto the plane perpendicular to
    shared_edge_dir, then compare which one's outer extent reaches further
    past the contact point.
    """
    edge_n = _normalize(shared_edge_dir)
    # Vector from panel center to door center, projected out of edge direction.
    delta = door_obb.center - panel_obb.center
    delta_in_plane = delta - (delta @ edge_n) * edge_n

    # Panel's max extent along delta_in_plane direction:
    direction = _normalize(delta_in_plane)
    if np.linalg.norm(direction) < 1e-9:
        return "panel"  # degenerate; concentric in-plane.

    panel_extent = float(np.sum(np.abs(panel_obb.R.T @ direction) * panel_obb.half_extents))
    door_extent = float(np.sum(np.abs(door_obb.R.T @ direction) * door_obb.half_extents))
    # Whichever box has a larger reach along the inter-center direction "owns the corner".
    return "door" if door_extent >= panel_extent else "panel"


# ============================================= candidate construction ==

def _select_contact_faces(
    door_obb: OBB,
    panel_obb: OBB,
    pin: np.ndarray,
    axis: np.ndarray,
    tol_mm: float,
) -> tuple[FaceRef, FaceRef] | None:
    """Pick the (door_face, panel_face) pair that defines the contact edge.

    Search order:
      1. Perpendicular face pairs whose plane-intersection line is within
         both face rectangles (within tol) and parallel to axis. Score by
         pin-to-line distance plus the in-plane line-to-rect gaps.
      2. If no perpendicular contact, anti-parallel face pairs whose planes
         are coplanar and footprints have real area overlap (sep < -tol/2).
         Score by pin-to-plane distance.

    Perpendicular contacts are preferred even when an anti-parallel pair
    scores lower, because anti-parallel pairs can show up coincidentally
    when the back of the door happens to be coplanar with the back of the
    panel without representing the geometric hinge contact.
    """
    tol = tol_mm * 1e-3
    axis_n = _normalize(axis)

    door_faces = _box_faces(door_obb, BoxRole.DOOR)
    panel_faces = _box_faces(panel_obb, BoxRole.PANEL)

    # ---------- pass 1: perpendicular pairs ----------
    best = None
    best_score = np.inf
    for fD in door_faces:
        for fP in panel_faces:
            dot = float(np.clip(fD.normal @ fP.normal, -1.0, 1.0))
            if abs(dot) > _PERP_DOT:
                continue
            edge_dir = _normalize(np.cross(fD.normal, fP.normal))
            if np.linalg.norm(edge_dir) < 1e-9:
                continue
            if abs(abs(edge_dir @ axis_n) - 1.0) > 0.1:
                continue
            line_pt, _ = _plane_intersection_line(
                fD.point, fD.normal, fP.point, fP.normal,
            )
            dD = _line_distance_to_face_2d(door_obb, fD, line_pt, edge_dir)
            dP = _line_distance_to_face_2d(panel_obb, fP, line_pt, edge_dir)
            if dD > tol + _TOL_EPS or dP > tol + _TOL_EPS:
                continue
            rel = pin - line_pt
            pin_to_line = float(np.linalg.norm(rel - (rel @ edge_dir) * edge_dir))
            score = pin_to_line + dD + dP
            if score < best_score:
                best_score = score
                best = (fD, fP)
    if best is not None:
        return best

    # ---------- pass 2: anti-parallel coplanar pairs with real overlap ----------
    for fD in door_faces:
        for fP in panel_faces:
            dot = float(np.clip(fD.normal @ fP.normal, -1.0, 1.0))
            if dot > -_PARALLEL_DOT:
                continue
            signed = float((fP.point - fD.point) @ fD.normal)
            if abs(signed) > tol:
                continue
            u, v = _orthonormal_basis_in_plane(fD.normal)
            pa = _project_to_plane_2d(_face_corners(door_obb, fD), fD.point, u, v)
            pb = _project_to_plane_2d(_face_corners(panel_obb, fP), fD.point, u, v)
            sep = _polygon_distance_2d(pa, pb)
            if sep >= -tol * 0.5:
                continue  # require real area overlap, not just coincidental coplanarity
            pin_to_plane = abs((pin - fD.point) @ fD.normal)
            if pin_to_plane < best_score:
                best_score = pin_to_plane
                best = (fD, fP)

    return best


def _opposite_face(box: OBB, face: FaceRef, role: BoxRole) -> FaceRef:
    """Return the face of `box` diametrically opposite `face`."""
    axis = face.face_id // 2
    new_sign = -1 if (face.face_id % 2 == 1) else +1
    new_id = 2 * axis + (1 if new_sign == +1 else 0)
    point = box.center + new_sign * box.half_extents[axis] * box.R[:, axis]
    normal = new_sign * box.R[:, axis]
    return FaceRef(box_role=role, face_id=new_id, point=point, normal=normal)


def _adjacent_face_along_edge(
    box: OBB,
    contact_face: FaceRef,
    edge_dir: np.ndarray,
    reference_point: np.ndarray,
    role: BoxRole,
) -> FaceRef | None:
    """Return the box's face that is adjacent to contact_face along edge_dir.

    The adjacent face's local axis must be (a) different from contact_face's
    axis and (b) not parallel to edge_dir. The sign is picked so the face's
    plane lies closest to reference_point (which should be a point on the
    contact edge -- typically the URDF pin).
    """
    contact_axis = contact_face.face_id // 2
    edge_n = _normalize(edge_dir)
    candidates = []
    for axis in range(3):
        if axis == contact_axis:
            continue
        local_axis_world = box.R[:, axis]
        if abs(local_axis_world @ edge_n) > _PARALLEL_DOT:
            continue
        candidates.append(axis)
    if len(candidates) != 1:
        return None
    adj_axis = candidates[0]
    # Pick the sign whose face plane is closest to reference_point along
    # the adjacent axis.
    axis_vec = box.R[:, adj_axis]
    proj = float((reference_point - box.center) @ axis_vec)
    sign = +1 if proj >= 0.0 else -1
    new_id = 2 * adj_axis + (1 if sign == +1 else 0)
    point = box.center + sign * box.half_extents[adj_axis] * box.R[:, adj_axis]
    normal = sign * box.R[:, adj_axis]
    return FaceRef(box_role=role, face_id=new_id, point=point, normal=normal)


def _enumerate_perp_faces(
    obb: OBB, axis_dir: np.ndarray, role: BoxRole,
) -> list[FaceRef]:
    """Return the 4 faces of the OBB perpendicular to axis_dir (i.e. the
    faces whose plane CONTAINS the axis line)."""
    along_axis_idx = _local_axis_along_world(obb, axis_dir)
    out: list[FaceRef] = []
    for axis_idx in range(3):
        if axis_idx == along_axis_idx:
            continue
        for sign in (-1, +1):
            face_id = 2 * axis_idx + (1 if sign == +1 else 0)
            normal = sign * obb.R[:, axis_idx]
            point = obb.center + sign * obb.half_extents[axis_idx] * obb.R[:, axis_idx]
            out.append(FaceRef(box_role=role, face_id=face_id, point=point, normal=normal))
    return out


def _is_inward(face: FaceRef, body_centroid: np.ndarray) -> bool:
    return float((np.asarray(body_centroid, dtype=float) - face.point) @ face.normal) > 0.0


def _face_area_perp_to_axis(obb: OBB, face: FaceRef, axis_dir: np.ndarray) -> float:
    """Area of the face. The face is a rectangle in the plane perp to its
    own normal; its dimensions are the box's extents in the two perpendicular
    local axes."""
    face_axis = face.face_id // 2
    return float(4.0 * obb.half_extents[(face_axis + 1) % 3] * obb.half_extents[(face_axis + 2) % 3])


def _local_axis_along_world(obb: OBB, world_dir: np.ndarray) -> int:
    """Return the index (0/1/2) of the OBB local axis whose world direction
    is most aligned with world_dir."""
    proj = np.abs(np.asarray(obb.R, dtype=float).T @ _normalize(world_dir))
    return int(np.argmax(proj))


def _make_placement(
    static_face: FaceRef,
    dynamic_face: FaceRef,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    hinge_type: HingeType,
    swing_range: tuple[float, float],
) -> HingePlacement:
    return HingePlacement(
        axis_origin=axis_origin,
        axis_direction=axis_direction,
        static_face=static_face,
        dynamic_face=dynamic_face,
        closed_angle=_CLOSED_ANGLE[hinge_type],
        hinge_type=hinge_type,
        swing_range=swing_range,
    )


# =================================================== public entry ==

def enumerate_placements(
    door_obb: OBB,
    panel_obb: OBB,
    axis_hint: np.ndarray,
    pin_hint: np.ndarray,
    body_centroid: np.ndarray,
    swing_range: tuple[float, float],
    tol_mm: float = 10.0,
) -> list[HingePlacement]:
    """Return every geometrically valid HingePlacement for this (D, P, axis).

    EDGE_CONTACT  -> up to 3 candidates (one per HingeType).
    FACE_CONTACT  -> exactly 1 candidate (EXTERIOR only).
    UNKNOWN       -> empty list.

    The returned axis_direction is always re-signed so that RH-rule positive
    delta moves the door's far corner away from body_centroid.
    """
    axis_n = axis_with_canonical_sign(axis_hint, pin_hint, door_obb, body_centroid)
    topology = classify_topology(door_obb, panel_obb, tol_mm=tol_mm)

    if topology is Topology.UNKNOWN:
        return []

    # v3 enumerate-and-pick logic. Treats both EDGE_CONTACT and FACE_CONTACT
    # uniformly: enumerate the 4 perp-to-axis faces of each box, then for
    # each hinge type apply that type's selection rule. Gives correct
    # results across cases that the older topology-branching code couldn't
    # generalize (microwave, fridge, stove, chest, bench, etc).
    if topology in (Topology.EDGE_CONTACT, Topology.FACE_CONTACT):
        return _enumerate_candidates(
            door_obb, panel_obb, axis_n, pin_hint, body_centroid,
            swing_range, tol_mm,
        )

    # Unreachable -- topology is one of the three enum values.
    return []


# Mountable-face rules for the door's dynamic mounting face (rules from
# the user's geometric intuition, applied uniformly to EXT / INT / FLAT):
#
#   Rule A — face normal NOT parallel to motion axis. The two end-cap
#            faces (whose normal aligns with the rotation axis ± direction)
#            cannot host a hinge: the leaves there would lie perpendicular
#            to the axis, so the door cannot pivot.
#   Rule B — face NOT opposite to the pin. The face whose outward normal
#            points across the door from the rotation axis (the "far" face
#            on the opposite side from the hinge edge) is geometrically
#            invalid for mounting. Implemented as
#               dot(face.normal, normalize(pin - door_center)) >= -0.5
#            so faces that point ~120°+ away from the pin direction are
#            excluded.
_FLUSH_TOL_M = 0.004    # FLAT plane-gap tolerance: leaves must lie within
                        # this of coplanar to qualify as a valid flat
                        # mount. Real flush seams in normalized HSSD/PNM
                        # geometry carry 1-2mm of authoring slop (a 1mm
                        # bound rejected genuinely flush inset doors by
                        # 0.02mm); 4mm accepts those while still excluding
                        # overlay fronts (offset >= one door thickness).


def _is_valid_dynamic_face(face: FaceRef,
                            door_obb: OBB,
                            axis: np.ndarray,
                            pin: np.ndarray,
                            opposite_threshold: float = -0.5,
                            ) -> tuple[bool, str]:
    """Apply rules A (not axis-parallel) + B (not opposite to pin) to the
    door's dynamic mounting face. Returns (ok, reason)."""
    n = np.asarray(face.normal, dtype=float)
    a = np.asarray(axis, dtype=float)
    a_norm = float(np.linalg.norm(a))
    if a_norm < 1e-9:
        return True, "axis_zero"
    a_n = a / a_norm
    if abs(float(n @ a_n)) > _PARALLEL_DOT:
        return False, "axis_parallel"
    door_center = np.asarray(door_obb.center, dtype=float)
    pin_dir = np.asarray(pin, dtype=float) - door_center
    pin_norm = float(np.linalg.norm(pin_dir))
    if pin_norm < 1e-9:
        return True, "pin_at_center"
    pin_dir_n = pin_dir / pin_norm
    if float(n @ pin_dir_n) < opposite_threshold:
        return False, "opposite_to_pin"
    return True, "ok"


def _enumerate_candidates(
    door_obb: OBB,
    panel_obb: OBB,
    axis_n: np.ndarray,
    pin_hint: np.ndarray,
    body_centroid: np.ndarray,
    swing_range: tuple[float, float],
    tol_mm: float,
) -> list[HingePlacement]:
    """v3 face-selection rules (validated against 11+ samples):

    EXT  = anti-parallel face pair (perp to axis) where both perpendicular-
           axis face-rect overlaps/gaps are within tol_mm. Among valid,
           smallest plane gap (closest contact).

    INT  = perpendicular pair. Panel face: when door & panel share the same
           world-thinnest axis (STACKED), pick face whose normal is most
           aligned with panel->door direction; otherwise (PERPENDICULAR
           L-corner), pick face whose normal points most toward body. Door
           face: broad-perp face whose normal points most toward body.

    FLAT = parallel-same OUTWARD pair on a different local axis from the
           EXT pair (= adjacent to EXT in the cuboid sense), smallest plane
           gap. Rejected if the smallest gap exceeds _FLUSH_TOL_M (the
           leaves must lie within ~5 mm of coplanar to be a valid flat
           mount).

    All three types additionally check the door's dynamic_face against
    rules A (not axis-parallel) and B (not opposite to pin).
    """
    tol = tol_mm * 1e-3

    door_perp = _enumerate_perp_faces(door_obb, axis_n, BoxRole.DOOR)
    panel_perp = _enumerate_perp_faces(panel_obb, axis_n, BoxRole.PANEL)

    placements: list[HingePlacement] = []

    # ---- EXTERIOR ----
    ext_candidates: list[tuple[float, FaceRef, FaceRef]] = []
    for p in panel_perp:
        for d in door_perp:
            if float(p.normal @ d.normal) > -_PARALLEL_DOT:
                continue
            face_axis = p.face_id // 2
            other_axes = [i for i in range(3) if i != face_axis]
            p_min = panel_obb.center - panel_obb.half_extents
            p_max = panel_obb.center + panel_obb.half_extents
            d_min = door_obb.center - door_obb.half_extents
            d_max = door_obb.center + door_obb.half_extents
            gap_a = max(p_min[other_axes[0]], d_min[other_axes[0]]) - min(p_max[other_axes[0]], d_max[other_axes[0]])
            gap_b = max(p_min[other_axes[1]], d_min[other_axes[1]]) - min(p_max[other_axes[1]], d_max[other_axes[1]])
            if gap_a > tol or gap_b > tol:
                continue
            plane_gap = abs(float((p.point - d.point) @ d.normal))
            ext_candidates.append((plane_gap, p, d))

    ext_chosen: tuple[FaceRef, FaceRef] | None = None
    if ext_candidates:
        ext_candidates.sort(key=lambda t: t[0])
        _, ext_p, ext_d = ext_candidates[0]
        ok, _ = _is_valid_dynamic_face(ext_d, door_obb, axis_n, pin_hint)
        if ok:
            ext_chosen = (ext_p, ext_d)
            placements.append(_make_placement(
                static_face=ext_p, dynamic_face=ext_d,
                axis_origin=pin_hint.copy(), axis_direction=axis_n,
                hinge_type=HingeType.EXTERIOR, swing_range=swing_range,
            ))

    # ---- INTERIOR ----
    # An INT hinge sits at an orthogonal corner where two boxes touch.
    # The two mounting faces are each box's BROAD face (= perpendicular to
    # the box's thinnest axis = the largest-area face). For each box, pick
    # the broad face whose outward normal points TOWARD the other box's
    # centroid -- that's the inner face at the corner. Symmetric, locally
    # defined (no body_centroid dependency), robust to multi-cabinet rows.
    door_center_np  = np.asarray(door_obb.center,  dtype=float)
    panel_center_np = np.asarray(panel_obb.center, dtype=float)
    door_thin_idx  = int(np.argmin(door_obb.half_extents))
    panel_thin_idx = int(np.argmin(panel_obb.half_extents))

    panel_all_faces = _box_faces(panel_obb, BoxRole.PANEL)
    door_all_faces  = _box_faces(door_obb,  BoxRole.DOOR)
    panel_broad = [f for f in panel_all_faces if (f.face_id // 2) == panel_thin_idx]
    door_broad  = [f for f in door_all_faces  if (f.face_id // 2) == door_thin_idx]
    if not panel_broad:
        panel_broad = list(panel_perp)  # fallback
    if not door_broad:
        door_broad = list(door_perp)    # fallback

    panel_inner = max(
        panel_broad,
        key=lambda f: float(np.asarray(f.normal, dtype=float)
                            @ (door_center_np - panel_center_np)),
    )
    door_inner = max(
        door_broad,
        key=lambda f: float(np.asarray(f.normal, dtype=float)
                            @ (panel_center_np - door_center_np)),
    )

    # Topology gate: INTERIOR only makes sense at a PERPENDICULAR L-corner
    # (door's broad face is at ~90° to panel's broad face). For parallel
    # overlay (door directly in front of/behind panel) the inner-broad
    # face pair would put the leaves on stacked parallel surfaces, which
    # is the EXT topology, not INT. Skip INT when the broad-face normals
    # are parallel within _PARALLEL_DOT.
    int_cos = abs(float(np.asarray(door_inner.normal, dtype=float)
                          @ np.asarray(panel_inner.normal, dtype=float)))
    int_topology_ok = int_cos < _PARALLEL_DOT
    ok, _ = _is_valid_dynamic_face(door_inner, door_obb, axis_n, pin_hint)
    if ok and int_topology_ok:
        placements.append(_make_placement(
            static_face=panel_inner, dynamic_face=door_inner,
            axis_origin=pin_hint.copy(), axis_direction=axis_n,
            hinge_type=HingeType.INTERIOR, swing_range=swing_range,
        ))

    # ---- FLAT ----
    if ext_chosen is not None:
        ext_p, ext_d = ext_chosen
        ext_p_axis = ext_p.face_id // 2
        ext_d_axis = ext_d.face_id // 2

        flat_pairs: list[tuple[float, FaceRef, FaceRef]] = []
        for p in panel_perp:
            if (p.face_id // 2) == ext_p_axis:
                continue
            if _is_inward(p, body_centroid):
                continue
            for d in door_perp:
                if (d.face_id // 2) == ext_d_axis:
                    continue
                if _is_inward(d, body_centroid):
                    continue
                if float(p.normal @ d.normal) < _PARALLEL_DOT:
                    continue
                plane_gap = abs(float((p.point - d.point) @ d.normal))
                flat_pairs.append((plane_gap, p, d))

        # Reject FLAT candidates whose plane gap exceeds the flush
        # tolerance — leaves must be coplanar within _FLUSH_TOL_M to make
        # geometric sense as a flat mount.
        flat_pairs = [(g, p, d) for g, p, d in flat_pairs if g <= _FLUSH_TOL_M]
        if flat_pairs:
            flat_pairs.sort(key=lambda t: t[0])
            _, flat_p, flat_d = flat_pairs[0]
            ok, _ = _is_valid_dynamic_face(flat_d, door_obb, axis_n, pin_hint)
            if ok:
                placements.append(_make_placement(
                    static_face=flat_p, dynamic_face=flat_d,
                    axis_origin=pin_hint.copy(), axis_direction=axis_n,
                    hinge_type=HingeType.FLAT, swing_range=swing_range,
                ))

    return placements


# Note: the previous EDGE_CONTACT/FACE_CONTACT-branched face selectors were
# replaced by _enumerate_candidates above and have been removed.
