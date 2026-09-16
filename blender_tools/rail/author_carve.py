"""One-off authoring of `carve_rail_clearance` volumes into rail templates.

Bakes a boolean cutter into each rail template .blend, mirroring how hinge
templates ship `carve_body_clearance`. The pipeline then just loads it with
the template: it scales, snaps, mirrors, and thickness-caps together with
the rail root, and install_from_pred subtracts it from static body meshes
when --carve is on. Nothing is built at install time.

Cutter construction, per template: the CONVEX HULL of the static member
plus the sliding member stretched along the slide axis over its whole
closed-to-open travel (frames 0..50; the members are extruded profiles
along that axis, so the stretch is an exact swept volume).

A hull, deliberately, and not the exact rail surface: the pocket must
evacuate the SPACE the assembly sweeps through. An exact-shape cut leaves
panel material inside every rail hole and groove, which collides the
moment the rail slides (and the pocket is invisible because the rail
fills it perfectly). For these straight channel rails the hull is the
tight wrapping of the outer silhouette with all concavities bridged --
zero added clearance, watertight, no folds, boolean-fast, and immune to
the corner template's non-manifold source meshes (hulls only read verts).

Run from the repo root (re-runnable; replaces any existing cutter):

    /usr/share/blender/blender --background --factory-startup \
        --python-exit-code 1 --python blender_tools/rail/author_carve.py

Originals are backed up next to each template as <name>.pre_carve.blend
(first run only).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import bmesh  # type: ignore
import bpy  # type: ignore
import numpy as np
from mathutils import Matrix, Vector  # type: ignore

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = [
    REPO / "annotated_mechanical_parts" / "sliding_rail_annotated.blend",
    REPO / "annotated_mechanical_parts" / "sliding_rail_02_annotated.blend",
]
CARVE_NAME = "carve_rail_clearance"
_OPEN_FRAME = 50
_MEMBERS = ("rail_static_end", "rail_dynamic_end")
_MOUNT_PIERCE = 0.004         # m, template space: hull extension through the
                              # mounting face on the drawer side (open air)


def _world_mesh(obj):
    tm = obj.data.copy()
    tm.transform(obj.matrix_world)
    return tm


def _interval(obj, axis: np.ndarray) -> tuple[float, float]:
    M = obj.matrix_world
    vals = [float(np.dot(np.asarray(M @ v.co, dtype=float), axis))
            for v in obj.data.vertices]
    return min(vals), max(vals)


def _stretch_matrix(axis: np.ndarray, lo: float, hi: float,
                    t_lo: float, t_hi: float) -> Matrix:
    """Affine map stretching interval [lo, hi] along `axis` to [t_lo, t_hi]."""
    k = (t_hi - t_lo) / max(hi - lo, 1e-12)
    L = np.eye(3) + (k - 1.0) * np.outer(axis, axis)
    t = ((1.0 - k) * lo + (t_lo - lo)) * axis
    return Matrix(((L[0][0], L[0][1], L[0][2], t[0]),
                   (L[1][0], L[1][1], L[1][2], t[1]),
                   (L[2][0], L[2][1], L[2][2], t[2]),
                   (0.0, 0.0, 0.0, 1.0)))


def _boundary_edges(me) -> int:
    bm = bmesh.new()
    bm.from_mesh(me)
    n = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    bm.free()
    return n


def author(template: Path) -> None:
    backup = template.with_suffix(".pre_carve.blend")
    if not backup.exists():
        shutil.copy2(template, backup)
        print(f"[{template.name}] backup -> {backup.name}")

    bpy.ops.wm.open_mainfile(filepath=str(template))
    root = bpy.data.objects.get("rail_master") or bpy.data.objects.get("rail")
    axis_obj = bpy.data.objects["sliding_axis"]
    old = bpy.data.objects.get(CARVE_NAME)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    s_axis = np.asarray(axis_obj.matrix_world.to_3x3() @ Vector((0, 0, 1)),
                        dtype=float)
    s_axis = s_axis / max(np.linalg.norm(s_axis), 1e-12)

    dyn = bpy.data.objects["rail_dynamic_end"]
    lo0, hi0 = _interval(dyn, s_axis)
    bpy.context.scene.frame_set(_OPEN_FRAME)
    bpy.context.view_layer.update()
    lo1, hi1 = _interval(dyn, s_axis)
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    t_lo, t_hi = min(lo0, lo1), max(hi0, hi1)

    bm = bmesh.new()
    for name in _MEMBERS:
        o = bpy.data.objects[name]
        tm = _world_mesh(o)
        if name == "rail_dynamic_end" and (t_hi - t_lo) > (hi0 - lo0) + 1e-9:
            tm.transform(_stretch_matrix(s_axis, lo0, hi0, t_lo, t_hi))
        for v in tm.vertices:
            bm.verts.new(v.co)
        bpy.data.meshes.remove(tm)
    bm.verts.ensure_lookup_table()
    hull = bmesh.ops.convex_hull(bm, input=bm.verts[:])
    inner = list({g for g in (hull.get("geom_interior", [])
                              + hull.get("geom_unused", []))
                  if isinstance(g, bmesh.types.BMVert)})
    if inner:
        bmesh.ops.delete(bm, geom=inner, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    # Extend the hull through the mounting face, on the drawer side (open
    # air). The rail's inner extent can land within a tenth of a millimetre
    # of the wall's inner surface; a boolean across that near-coplanar
    # margin is unstable and can seal the pocket with a skin (a closed
    # void inside the panel). Piercing decisively through the surface
    # costs nothing: the added volume lies outside the wall, so the pocket
    # shape is unchanged and the cross-section keeps zero clearance.
    snap = bpy.data.objects["snap_plane_static_end"]
    Rn = snap.matrix_world.to_3x3()
    acc = Vector((0.0, 0.0, 0.0))
    for poly in snap.data.polygons:
        acc += (Rn @ poly.normal) * poly.area
    n_axis = np.asarray(acc.normalized(), dtype=float)
    # The snap plane RECTANGLE lies on the wall-side face; its centroid
    # tells us which extreme of the hull is the wall side, independent of
    # which way the plane's normal was authored.
    Ms = snap.matrix_world
    sc = Vector((0.0, 0.0, 0.0))
    sa = 0.0
    for poly in snap.data.polygons:
        sc += (Ms @ poly.center) * poly.area
        sa += poly.area
    wall_proj = float(np.dot(np.asarray(sc / max(sa, 1e-12), dtype=float),
                             n_axis))
    projs = [float(np.dot(np.asarray(v.co, dtype=float), n_axis))
             for v in bm.verts]
    mn, mx = min(projs), max(projs)
    width = max(mx - mn, 1e-9)
    k = (width + _MOUNT_PIERCE) / width
    anchor = mx if wall_proj > (mn + mx) * 0.5 else mn
    for v, pr in zip(bm.verts, projs):
        v.co += Vector((k - 1.0) * (pr - anchor) * n_axis)
    me = bpy.data.meshes.new(f"{CARVE_NAME}_mesh")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(CARVE_NAME, me)
    bpy.context.scene.collection.objects.link(obj)
    mode = ["convex-hull"]
    n_boundary = _boundary_edges(me)
    if n_boundary:
        print(f"[{template.name}] WARNING: cutter has "
              f"{n_boundary} boundary edges")
    obj.parent = root
    obj.matrix_parent_inverse = root.matrix_world.inverted()
    # Wire + no-render, but NOT hide_viewport: disabled-in-viewport objects
    # drop out of the depsgraph, which makes boolean operands evaluate
    # empty, matrices go stale, and duplicate() skip them. The pipeline
    # eye-hides (hide_set) instances after install instead.
    obj.display_type = "WIRE"
    obj.hide_render = True

    bpy.ops.wm.save_mainfile(filepath=str(template))
    print(f"[{template.name}] carve authored: {', '.join(mode)}  "
          f"verts={len(me.vertices)}  travel={t_hi - t_lo - (hi0 - lo0):.3f}m")


def main() -> None:
    for t in TEMPLATES:
        if not t.exists():
            print(f"[skip] {t} (missing)")
            continue
        author(t)
    print("AUTHOR-CARVE-DONE")


if __name__ == "__main__":
    main()
