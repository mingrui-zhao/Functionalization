"""Support block / divider generator (v3).

Per rail, decides between two modes based on the local geometry:

  SLAB mode (v2-style)
      Used when the cabinet wall is close behind the rail (cast distance
      is at most LONG_EDGE / 3, where LONG_EDGE is the longer edge of
      snap_plane_static). The block is a thin extrusion from the snap
      plane to the wall.

  DIVIDER mode (new)
      Used when the wall is far away or absent. A vertical panel is
      maximally inscribed in the cabinet's local interior at the rail's
      snap-plane location. The rail mounts on the divider's front face.

Geometry-derived constants -- nothing hardcoded:

      SLAB_MAX_DEPTH        = LONG_EDGE  / 3
      DIVIDER_THICKNESS_TGT = SHORT_EDGE      (= snap plane short edge)

Divider sizing
  * In-plane footprint inscribed via raycasts from the snap plane center
    along +/- plane_x and +/- plane_y until they hit cabinet body (statics
    only; doors/handles are excluded). If a direction misses, that side is
    open -- the divider is bounded by the RAIL's own extent there (+5 mm),
    never by the cabinet's outer envelope.
  * Thickness defaults to DIVIDER_THICKNESS_TGT, but is reduced if any
    ray in a dense grid over the divider's full footprint hits cabinet
    body within target thickness. Thickness = min(target, min_hit - clearance).

Merging
  Facing rail pairs across a gap each keep their own divider; both
  thicknesses are capped to half the gap so the panels meet without
  interpenetrating. Same-side stacked dividers (within 5 mm along the
  normal) merge into one panel whose footprint is the union of the
  contributors, so a column of drawers emits one divider, not N.

Existing slab logic (rail-already-mating skip; bolts past snap plane
allowed to tuck into body) is preserved.

Imported by the rail pipeline; not a standalone script.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import bpy  # type: ignore
import bmesh  # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # blender_tools/ on sys.path

# ===== helpers ==========================================================
_STATIC_NODE_NAME = "static_node"

_RAIL_MECH_BASES = {
    "rail_master", "rail",
    "sliding_axis", "rail_static_end", "rail_dynamic_end",
    "snap_plane_static_end", "snap_plane_dynamic_end",
    "snap_plane_rail_dynamic_end", "snap_plane_drawer_support",
}


def _base_name(name: str) -> str:
    if "__" in name: name = name.split("__", 1)[0]
    if "." in name and name.rsplit(".", 1)[-1].isdigit():
        name = name.rsplit(".", 1)[0]
    return name


def _is_in_rail_hierarchy(obj) -> bool:
    cur = obj
    while cur is not None:
        if _base_name(cur.name) in _RAIL_MECH_BASES:
            return True
        cur = cur.parent
    return False


def _bvh_from_objects(objs, depsgraph) -> tuple[BVHTree | None, list[Vector]]:
    bm = bmesh.new(); n = 0
    for o in objs:
        if o.type != "MESH" or o.data is None: continue
        eo = o.evaluated_get(depsgraph)
        try: me = eo.to_mesh()
        except Exception: continue
        try:
            tmp = bmesh.new(); tmp.from_mesh(me); tmp.transform(eo.matrix_world)
            verts_lookup = {tv.index: bm.verts.new(tv.co) for tv in tmp.verts}
            bm.verts.ensure_lookup_table()
            for tf in tmp.faces:
                try: bm.faces.new([verts_lookup[v.index] for v in tf.verts]); n += 1
                except ValueError: pass
            tmp.free()
        finally: eo.to_mesh_clear()
    if n == 0: bm.free(); return None, []
    bm.faces.ensure_lookup_table(); bm.normal_update()
    normals = [Vector(f.normal) for f in bm.faces]
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh, normals


_RAIL_BODY_MESH_BASES = {"rail_static_end", "rail_dynamic_end"}


def _split_rail_faces_by_plane(rail_root, plane_co: Vector, plane_n: Vector,
                               depsgraph) -> tuple[BVHTree | None, BVHTree | None]:
    """Walk descendants of rail_root, split faces by side of (plane_co, plane_n).
    Only considers true rail-mechanism meshes (rail_static_end / rail_dynamic_end);
    drawer fragments parented under the rail's dynamic_end after install are
    EXCLUDED, otherwise the LEFT rail (the parenting target) inherits the
    drawer's faces and looks asymmetric vs. the RIGHT rail."""
    bm_before = bmesh.new(); n_before = 0
    bm_after = bmesh.new(); n_after = 0
    for d in [rail_root, *rail_root.children_recursive]:
        if d.type != "MESH" or d.data is None: continue
        # Only true rail mech meshes count.
        if _base_name(d.name) not in _RAIL_BODY_MESH_BASES:
            continue
        eo = d.evaluated_get(depsgraph)
        try: me = eo.to_mesh()
        except Exception: continue
        try:
            tmp = bmesh.new(); tmp.from_mesh(me); tmp.transform(eo.matrix_world)
            for tf in tmp.faces:
                co = sum((v.co for v in tf.verts), Vector((0,0,0))) / len(tf.verts)
                d_signed = (co - plane_co).dot(plane_n)
                target_bm = bm_after if d_signed > _BEFORE_EPS else bm_before
                target_n_inc = 1
                # Add this face to the chosen bm
                vs = [target_bm.verts.new(v.co) for v in tf.verts]
                try: target_bm.faces.new(vs)
                except ValueError: pass
                if target_bm is bm_before: n_before += target_n_inc
                else: n_after += target_n_inc
            tmp.free()
        finally: eo.to_mesh_clear()
    out_before = None
    if n_before > 0:
        bm_before.faces.ensure_lookup_table()
        out_before = BVHTree.FromBMesh(bm_before)
    bm_before.free()
    out_after = None
    if n_after > 0:
        bm_after.faces.ensure_lookup_table()
        out_after = BVHTree.FromBMesh(bm_after)
    bm_after.free()
    return out_before, out_after


def _snap_plane_world(plane_obj, depsgraph) -> tuple[Vector, Vector, Vector, Vector] | None:
    """Return (center, normal, in_plane_x, in_plane_y, half_x, half_y) for
    the snap_plane mesh at frame-0 evaluated state. The normal points
    away from the rail body (i.e., into the cabinet)."""
    eo = plane_obj.evaluated_get(depsgraph)
    me = eo.to_mesh()
    try:
        if not me.polygons: return None
        # Plane normal in world coords.
        M3 = eo.matrix_world.to_3x3()
        n = Vector((0, 0, 0))
        for p in me.polygons:
            n += M3 @ p.normal
        if n.length < 1e-9: return None
        normal = n.normalized()
        # Center (vertex average).
        verts = [eo.matrix_world @ v.co for v in me.vertices]
        center = sum(verts, Vector((0,0,0))) / len(verts)
        # In-plane axes via PCA-ish: pick edge of first polygon for x.
        p0 = me.polygons[0]
        v_idx = list(p0.vertices)
        if len(v_idx) < 2: return None
        a = eo.matrix_world @ me.vertices[v_idx[0]].co
        b = eo.matrix_world @ me.vertices[v_idx[1]].co
        plane_x = (b - a) - normal * (b - a).dot(normal)
        if plane_x.length < 1e-9: return None
        plane_x = plane_x.normalized()
        plane_y = normal.cross(plane_x).normalized()
        # Half extents along plane_x, plane_y.
        xs = [(v - center).dot(plane_x) for v in verts]
        ys = [(v - center).dot(plane_y) for v in verts]
        half_x = max(abs(min(xs)), abs(max(xs)))
        half_y = max(abs(min(ys)), abs(max(ys)))
        return (center, normal, plane_x, plane_y, half_x, half_y)
    finally:
        eo.to_mesh_clear()


def _get_or_create_static_node():
    n = bpy.data.objects.get(_STATIC_NODE_NAME)
    if n is not None: return n
    n = bpy.data.objects.new(_STATIC_NODE_NAME, None)
    n.empty_display_type = "PLAIN_AXES"; n.empty_display_size = 0.25
    bpy.context.scene.collection.objects.link(n)
    return n


def _get_or_create_misc_material():
    m = bpy.data.materials.get(_MATERIAL_NAME)
    if m is not None: return m
    m = bpy.data.materials.new(_MATERIAL_NAME)
    return m



# ===== tunables =========================================================
_RAY_GRID_SLAB     = (5, 5)        # for slab depth check (within snap plane)
_RAY_GRID_DIVIDER  = (9, 9)        # for divider thickness pruning (within full footprint)
_INSCRIBE_RAYS     = 5             # rays per direction for divider inscribing
_RAY_MAX_DIST      = 2.0           # 2 m: enough for any cabinet
_BEFORE_EPS        = 0.0005
_BLOCK_CLEARANCE   = 0.0002
_INSCRIBE_CLEARANCE = 0.001        # 1 mm gap inside cabinet
_GROUP_TOL_NORMAL  = 0.005         # 5 mm: rails within this share a divider
_GROUP_TOL_NORMAL_DIR_DOT = 0.97   # cos(threshold) for normal alignment
_PAIR_LATERAL_TOL  = 0.020         # 20 mm: paired dividers must be laterally aligned within this
_PAIR_GAP_MAX      = 1.0           # 1 m: max gap to consider as paired
_FRONT_CLIP_MARGIN = 0.005         # 5 mm: divider may extend this far past rail front (back of pulling board)
_MIN_THICKNESS     = 0.002         # 2 mm floor; below this we skip
_MATERIAL_NAME     = "misc"
_BLOCK_NAME_SLAB    = "supporting_block"
_BLOCK_NAME_DIVIDER = "supporting_divider"


# ===== plane raycast helpers ============================================
def _cast_inplane_distance(body_bvh, origin: Vector, direction: Vector,
                           max_dist: float) -> float | None:
    """Single ray; returns distance to first hit or None."""
    res = body_bvh.ray_cast(origin, direction.normalized(), max_dist)
    if res is None or res[0] is None:
        return None
    return (res[0] - origin).length


def _grid_min_hit(body_bvh, center: Vector,
                  axis_a: Vector, axis_b: Vector,
                  half_a: float, half_b: float,
                  cast_dir: Vector, max_dist: float,
                  grid: tuple[int, int]) -> float | None:
    """Cast (na x nb) rays from a rectangle (center, axis_a*half_a, axis_b*half_b)
    along cast_dir up to max_dist. Returns min hit distance or None."""
    na, nb = grid
    hits = []
    for i in range(na):
        ta = (i + 0.5) / na
        for j in range(nb):
            tb = (j + 0.5) / nb
            sa = (ta - 0.5) * 2 * half_a
            sb = (tb - 0.5) * 2 * half_b
            origin = center + axis_a * sa + axis_b * sb
            d = _cast_inplane_distance(body_bvh, origin, cast_dir, max_dist)
            if d is not None: hits.append(d)
    return min(hits) if hits else None


def _inscribe_distance(body_bvh, snap_center: Vector,
                       cast_dir: Vector, perp_axis: Vector,
                       perp_half: float, max_dist: float,
                       n_rays: int = _INSCRIBE_RAYS) -> tuple[float | None, int, int]:
    """Cast `n_rays` rays toward `cast_dir` from points evenly spaced along
    `perp_axis` over [-perp_half, +perp_half] passing through snap_center.
    Returns (min_hit_distance_or_None, n_hits, n_total).

    Multi-ray inscribing protects against a single centred ray that grazes
    a corner: a strip of rays catches near-edge obstacles and yields a
    safer "guaranteed clearance" for sizing the divider footprint.
    """
    hits: list[float] = []
    if n_rays <= 1:
        d = _cast_inplane_distance(body_bvh, snap_center, cast_dir, max_dist)
        return ((d, 1, 1) if d is not None else (None, 0, 1))
    for i in range(n_rays):
        # t in [-1, +1]
        t = (i / (n_rays - 1) - 0.5) * 2.0
        origin = snap_center + perp_axis * (t * perp_half)
        d = _cast_inplane_distance(body_bvh, origin, cast_dir, max_dist)
        if d is not None:
            hits.append(d)
    return (min(hits) if hits else None), len(hits), n_rays



# ===== block builder ====================================================
def _build_box(name: str, center: Vector, normal: Vector,
               plane_x: Vector, plane_y: Vector,
               half_x: float, half_y: float, depth: float):
    """Front face is centered at `center` (local origin in plane basis).
    Box extends `depth` along +normal."""
    front = [center + plane_x * sx * half_x + plane_y * sy * half_y
             for sx, sy in [(-1,-1),(+1,-1),(+1,+1),(-1,+1)]]
    back = [p + normal * depth for p in front]
    verts = [tuple(v) for v in front + back]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5),
             (0, 4, 5, 1), (1, 5, 6, 2),
             (2, 6, 7, 3), (3, 7, 4, 0)]
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], faces); mesh.update()
    # The hand-listed windings above point inward for a right-handed plane
    # basis; recalc so the closed box is consistently outward regardless of
    # the basis handedness.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(_get_or_create_misc_material())
    parent = _get_or_create_static_node()
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()
    obj["generated_by"] = "support_block"
    obj["depth_m"] = float(depth)
    return obj


# ===== per-rail proposal ================================================
def _propose_for_rail(rail_root, body_objs, body_bvh, depsgraph,
                      enable_divider: bool = True) -> dict:
    diag = {"rail": rail_root.name}
    snap_static = None
    for d in [rail_root, *rail_root.children_recursive]:
        if d.type == "MESH" and _base_name(d.name) == "snap_plane_static_end":
            snap_static = d; break
    if snap_static is None:
        diag["skip"] = "no snap_plane_static_end"; return diag

    plane_info = _snap_plane_world(snap_static, depsgraph)
    if plane_info is None:
        diag["skip"] = "could not derive plane basis"; return diag
    center, normal, plane_x, plane_y, half_x, half_y = plane_info

    # Re-orient normal so it points away from rail body. Only consider
    # actual rail mechanism meshes -- the drawer (parented to LEFT rail's
    # dynamic_end) would otherwise dominate the centroid and flip the
    # normal direction asymmetrically.
    rc = Vector((0, 0, 0)); n_v = 0
    for d in rail_root.children_recursive:
        if d.type != "MESH" or d.data is None: continue
        if _base_name(d.name) not in _RAIL_BODY_MESH_BASES: continue
        eo = d.evaluated_get(depsgraph)
        for v in eo.data.vertices:
            rc += eo.matrix_world @ v.co; n_v += 1
    if n_v > 0:
        rc /= n_v
        if (rc - center).dot(normal) > 0:
            normal = -normal
            plane_y = -plane_y

    # Geometry-derived tunables
    short_edge = 2 * min(half_x, half_y)
    long_edge = 2 * max(half_x, half_y)
    slab_max_depth = long_edge / 3.0
    target_thickness = short_edge

    # BEFORE intersection check -> skip if rail already mates
    before_bvh, _ = _split_rail_faces_by_plane(rail_root, center, normal, depsgraph)
    if before_bvh is not None:
        try:
            if before_bvh.overlap(body_bvh):
                diag["action"] = "skip"; diag["reason"] = "rail body already mating"
                diag["short_edge"] = short_edge; diag["long_edge"] = long_edge
                return diag
        except Exception:
            pass

    # Slab depth check: cast grid over snap-plane footprint
    min_hit_slab = _grid_min_hit(
        body_bvh, center, plane_x, plane_y, half_x, half_y,
        normal, _RAY_MAX_DIST, _RAY_GRID_SLAB)

    diag["short_edge"] = short_edge
    diag["long_edge"] = long_edge
    diag["slab_max_depth"] = slab_max_depth
    diag["target_thickness"] = target_thickness
    diag["min_hit_slab"] = min_hit_slab

    if min_hit_slab is not None and min_hit_slab <= slab_max_depth:
        # ---- SLAB mode ----
        depth = max(0.0, min_hit_slab - _BLOCK_CLEARANCE)
        if depth < _MIN_THICKNESS:
            diag["action"] = "skip"; diag["reason"] = f"slab depth too small ({depth*1000:.1f}mm)"
            return diag
        return {
            "mode": "slab", "rail": rail_root.name,
            "center": center, "normal": normal,
            "plane_x": plane_x, "plane_y": plane_y,
            "half_x": half_x, "half_y": half_y,
            "depth": depth, "diag": diag,
            "action": "build_slab",
        }

    # Slab not feasible. If the caller asked us NOT to fall back to a
    # divider, skip this rail entirely -- the user prefers a missing
    # support to a divider that can protrude through the cabinet body.
    if not enable_divider:
        diag["action"] = "skip"
        diag["reason"] = ("slab depth over threshold "
                          f"({min_hit_slab})/{slab_max_depth:.3f}; "
                          "divider disabled")
        return diag

    # ---- DIVIDER mode ----
    # Inscribed in-plane extents via a STRIP of rays (rather than one
    # centred ray) so we catch near-edge obstacles. Each direction's
    # rays span the snap plane's perpendicular half-extent.
    front_dist, fh_n, fh_t = _inscribe_distance(
        body_bvh, center,  plane_x,  plane_y, half_y, _RAY_MAX_DIST)
    back_dist,  bh_n, bh_t = _inscribe_distance(
        body_bvh, center, -plane_x,  plane_y, half_y, _RAY_MAX_DIST)
    up_dist,    uh_n, uh_t = _inscribe_distance(
        body_bvh, center,  plane_y,  plane_x, half_x, _RAY_MAX_DIST)
    down_dist,  dh_n, dh_t = _inscribe_distance(
        body_bvh, center, -plane_y,  plane_x, half_x, _RAY_MAX_DIST)

    # Rail extents along all four in-plane directions. These are the
    # MISS fallbacks: a ray that finds no cabinet geometry means that
    # direction is open (open front, no ceiling), and a divider exists
    # only to support the rail -- so where the cabinet gives no bound,
    # the rail itself is the bound (+ a small margin). This replaces the
    # old outer-bbox fallback, which grew the divider flush with the
    # cabinet's outer envelope (through the top panel, past the face
    # frame), and the old +plane_x-only front clip, which relied on an
    # axis identity the snap-plane basis does not guarantee.
    rail_ext = {"+x": 0.0, "-x": 0.0, "+y": 0.0, "-y": 0.0}
    for d in rail_root.children_recursive:
        if d.type != "MESH" or d.data is None: continue
        # True rail-mechanism meshes only: after install the DRAWER is
        # parented under the left rail's dynamic_end, and counting its
        # pulling board would push the fallback past the face frame.
        if _base_name(d.name) not in _RAIL_BODY_MESH_BASES: continue
        eo = d.evaluated_get(depsgraph)
        for v in eo.data.vertices:
            w = eo.matrix_world @ v.co - center
            px = w.dot(plane_x); py = w.dot(plane_y)
            if px > rail_ext["+x"]: rail_ext["+x"] = px
            if -px > rail_ext["-x"]: rail_ext["-x"] = -px
            if py > rail_ext["+y"]: rail_ext["+y"] = py
            if -py > rail_ext["-y"]: rail_ext["-y"] = -py
    if front_dist is None: front_dist = rail_ext["+x"] + _FRONT_CLIP_MARGIN
    if back_dist  is None: back_dist  = rail_ext["-x"] + _FRONT_CLIP_MARGIN
    if up_dist    is None: up_dist    = rail_ext["+y"] + _FRONT_CLIP_MARGIN
    if down_dist  is None: down_dist  = rail_ext["-y"] + _FRONT_CLIP_MARGIN
    diag["inscribe_hits"] = {
        "front": (fh_n, fh_t), "back":  (bh_n, bh_t),
        "up":    (uh_n, uh_t), "down":  (dh_n, dh_t),
    }
    diag["rail_ext"] = dict(rail_ext)
    diag["inscribe"] = dict(front=front_dist, back=back_dist, up=up_dist, down=down_dist)

    # Footprint half-extents (with clearance), allowing asymmetric center shift
    div_half_x = max(0.0, (front_dist + back_dist) * 0.5 - _INSCRIBE_CLEARANCE)
    div_half_y = max(0.0, (up_dist + down_dist) * 0.5 - _INSCRIBE_CLEARANCE)
    if div_half_x < short_edge * 0.25 or div_half_y < short_edge * 0.25:
        diag["action"] = "skip"; diag["reason"] = "inscribed footprint too small"
        return diag
    center_offset_x = (front_dist - back_dist) * 0.5
    center_offset_y = (up_dist - down_dist) * 0.5
    div_center = center + plane_x * center_offset_x + plane_y * center_offset_y

    # Thickness: cast dense grid over divider footprint along +normal and
    # take min hit. If something is between snap plane and target thickness,
    # shrink to avoid it.
    min_hit_div = _grid_min_hit(
        body_bvh, div_center, plane_x, plane_y, div_half_x, div_half_y,
        normal, _RAY_MAX_DIST, _RAY_GRID_DIVIDER)
    if min_hit_div is not None:
        thickness = min(target_thickness, max(0.0, min_hit_div - _BLOCK_CLEARANCE))
    else:
        thickness = target_thickness
    diag["min_hit_divider_grid"] = min_hit_div
    diag["thickness"] = thickness
    if thickness < _MIN_THICKNESS:
        diag["action"] = "skip"; diag["reason"] = f"divider thickness too small ({thickness*1000:.1f}mm)"
        return diag
    return {
        "mode": "divider", "rail": rail_root.name,
        "center": div_center, "normal": normal,
        "plane_x": plane_x, "plane_y": plane_y,
        "half_x": div_half_x, "half_y": div_half_y,
        "depth": thickness, "diag": diag,
        "action": "build_divider",
        # Merge key components (filled later)
        "snap_center": center,
        # Snap-plane footprint (for pair bridge slabs, which must be
        # rail-sized, not wall-sized)
        "snap_half_x": half_x, "snap_half_y": half_y,
    }


# ===== merging ==========================================================

def _resolve_paired_dividers(proposals: list[dict]) -> list[dict]:
    """Two divider proposals that face each other across a gap (antiparallel
    snap normals, laterally aligned, gap below _PAIR_GAP_MAX) are rails on
    facing columns. Design rule: each drawer keeps ITS OWN divider hugging
    its rail; the only adjustment is capping both thicknesses to half the
    gap (minus clearance) so the two panels meet in the middle instead of
    interpenetrating. No central panel, no bridge slabs."""
    divs = [p for p in proposals if p["mode"] == "divider"]
    others = [p for p in proposals if p["mode"] != "divider"]

    for i, p1 in enumerate(divs):
        for j in range(i + 1, len(divs)):
            p2 = divs[j]
            # Antiparallel normals
            if p1["normal"].dot(p2["normal"]) > -_GROUP_TOL_NORMAL_DIR_DOT:
                continue
            # p2 in the +N1 direction of p1 (they face each other)
            v = p2["snap_center"] - p1["snap_center"]
            v_along = v.dot(p1["normal"])
            if v_along <= 0 or v_along > _PAIR_GAP_MAX:
                continue
            # Laterally aligned within tolerance
            v_perp = v - p1["normal"] * v_along
            if v_perp.length > _PAIR_LATERAL_TOL:
                continue
            cap = max(_MIN_THICKNESS, v_along * 0.5 - _BLOCK_CLEARANCE)
            for pk in (p1, p2):
                if pk["depth"] > cap:
                    pk["depth"] = cap
                    pk["capped_by_facing_pair"] = True
    return others + divs



def _merge_dividers(proposals: list[dict]) -> list[dict]:
    """Group divider proposals by:
      - similar normal direction  (|dot| > GROUP_TOL_NORMAL_DIR_DOT)
      - similar position along that normal axis  (|delta| <= GROUP_TOL_NORMAL)
    Within a group, take footprint UNION (min/max along plane_x and plane_y),
    keep min thickness (most conservative against collision)."""
    divs = [p for p in proposals if p.get("action") == "build_divider"]
    others = [p for p in proposals if p.get("action") != "build_divider"]

    groups: list[list[dict]] = []
    for d in divs:
        placed = False
        for g in groups:
            ref = g[0]
            if abs(d["normal"].dot(ref["normal"])) < _GROUP_TOL_NORMAL_DIR_DOT:
                continue
            # Project center onto normal axis (using the snap_plane center as
            # the ON-plane position, so stacked drawers with same X group together)
            d_pos = d["snap_center"].dot(ref["normal"])
            r_pos = ref["snap_center"].dot(ref["normal"])
            if abs(d_pos - r_pos) <= _GROUP_TOL_NORMAL:
                g.append(d); placed = True; break
        if not placed:
            groups.append([d])

    merged: list[dict] = []
    for g in groups:
        ref = g[0]
        # Union extents in plane basis (relative to ref.center)
        plane_x = ref["plane_x"]; plane_y = ref["plane_y"]
        normal = ref["normal"]
        # Use the LCS centered at ref.snap_center for stable arithmetic
        ref_center = ref["snap_center"]
        x_min, x_max = float("inf"), float("-inf")
        y_min, y_max = float("inf"), float("-inf")
        thicknesses = []
        contributors = []
        for d in g:
            # Support-projected halfs: correct even when a member's plane
            # basis has different axis roles than the reference (mirrored
            # rails swap slide and vertical).
            cx = (d["center"] - ref_center).dot(plane_x)
            cy = (d["center"] - ref_center).dot(plane_y)
            hx = (abs(d["plane_x"].dot(plane_x)) * d["half_x"]
                  + abs(d["plane_y"].dot(plane_x)) * d["half_y"])
            hy = (abs(d["plane_x"].dot(plane_y)) * d["half_x"]
                  + abs(d["plane_y"].dot(plane_y)) * d["half_y"])
            x_min = min(x_min, cx - hx)
            x_max = max(x_max, cx + hx)
            y_min = min(y_min, cy - hy)
            y_max = max(y_max, cy + hy)
            thicknesses.append(d["depth"])
            contributors.append(d["rail"])
        new_half_x = (x_max - x_min) * 0.5
        new_half_y = (y_max - y_min) * 0.5
        new_center_offset_x = (x_max + x_min) * 0.5
        new_center_offset_y = (y_max + y_min) * 0.5
        new_center = (ref_center
                      + plane_x * new_center_offset_x
                      + plane_y * new_center_offset_y)
        merged.append({
            "mode": "divider",
            "center": new_center, "normal": normal,
            "plane_x": plane_x, "plane_y": plane_y,
            "half_x": new_half_x, "half_y": new_half_y,
            "depth": min(thicknesses),
            "contributors": contributors,
        })

    return merged + others


# ===== runner ===========================================================
def _enumerate_objects(drawer_link_names: list[str]):
    """(rails, body). Body must be STATIC cabinet structure only: doors,
    handles, and hinge mechanisms move with the animation, so including
    them poisons both the inscribe raycasts and the bbox fallback (a
    divider would inherit the door/handle's front plane and protrude past
    the face frame)."""
    drawer_prefixes = [dl + "__mesh" for dl in drawer_link_names]
    body, rails = [], []
    for o in bpy.context.scene.objects:
        if _base_name(o.name) in ("rail_master", "rail") and o.parent is None:
            rails.append(o); continue
        if o.type != "MESH": continue
        if any(o.name.startswith(p) for p in drawer_prefixes): continue
        if _is_in_rail_hierarchy(o): continue
        low = o.name.lower()
        if "door" in low or "handle" in low or "hinge" in low:
            continue
        body.append(o)
    return rails, body


def process_current_scene(drawer_link_names: list[str],
                          enable_divider: bool = True) -> dict:
    """Add support blocks / dividers to the CURRENT loaded scene.
    Caller is responsible for opening the blend, frame-setting, and
    saving. Returns a per-rail diagnostic dict.

    When `enable_divider=False`, the divider fallback is suppressed --
    if the slab can't span to a nearby wall within threshold, the rail
    gets no support at all (use this when a possibly-protruding divider
    is worse than no support).
    """
    bpy.context.scene.frame_set(0); bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rails, body = _enumerate_objects(drawer_link_names)
    body_bvh, _ = _bvh_from_objects(body, depsgraph)

    diags = []
    proposals = []
    for r in rails:
        try:
            if body_bvh is None:
                diags.append({"rail": r.name, "skip": "no cabinet body"}); continue
            p = _propose_for_rail(r, body, body_bvh, depsgraph,
                                  enable_divider=enable_divider)
            if "diag" in p:
                diags.append(p["diag"])
            else:
                diags.append(p)
            if p.get("action") in ("build_slab", "build_divider"):
                proposals.append(p)
        except Exception as exc:
            traceback.print_exc()
            diags.append({"rail": r.name, "error": str(exc)})

    proposals = _resolve_paired_dividers(proposals)
    final = _merge_dividers(proposals)

    n_slab = n_div = 0
    for p in final:
        if p["mode"] == "slab":
            obj = _build_box(_BLOCK_NAME_SLAB, p["center"], p["normal"],
                             p["plane_x"], p["plane_y"],
                             p["half_x"], p["half_y"], p["depth"])
            obj["rail_root"] = p.get("rail", "")
            obj["mode"] = "slab"
            n_slab += 1
        else:
            obj = _build_box(_BLOCK_NAME_DIVIDER, p["center"], p["normal"],
                             p["plane_x"], p["plane_y"],
                             p["half_x"], p["half_y"], p["depth"])
            obj["mode"] = "divider"
            obj["contributors"] = ",".join(p.get("contributors", []))
            n_div += 1

    return {
        "n_rails": len(rails),
        "n_slabs_built": n_slab,
        "n_dividers_built": n_div,
        "n_skipped": sum(1 for d in diags if d.get("action") == "skip" or d.get("skip")),
        "rails": diags,
    }


