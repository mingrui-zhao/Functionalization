"""coborder v2 -- robust body detection from full drawer mesh.

Fixes two known failure modes of v1:

  1. Single-geometry drawer: body fragment heuristic picks the whole drawer
     including the pulling board. Outer-W cast then snaps the rail to the
     pulling board's edge, offset from the actual drawer body.

  2. Thick pulling board that overlaps tray fragments along the slide
     axis. Body fragment slide range still hits the pulling board, same
     offset failure.

Approach:

  Take the FULL drawer mesh (no fragment selection). For each rail side:

    a. Cast inward at a dense (slide x height) grid -- the same cast
       layout as v1.

    b. For each slide position, take the OUTERMOST W hit across heights.
       Call this curve outer_W(slide).

    c. The body's outer W is the MEDIAN of outer_W(slide). The pulling
       board contributes a few outliers (wider/thicker) that the median
       ignores. For a uniform single-geometry drawer with NO pulling
       board, median == body W trivially.

    d. The body slide range is the contiguous run of slide positions
       whose outer_W is within tolerance of the median.

    e. Snap snap_dynamic centroid to the median W (not the extremum).

For the height refinement (corner support / center vertical centering),
restrict slide samples to the body slide range so pulling-board geometry
can't bias the result.

This module is a parallel implementation alongside coborder.py; the v1
script remains the baseline. To switch a pipeline to v2, replace the
import.
"""
from __future__ import annotations

import bmesh
import bpy  # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore

from .snap import RailSnapResult, _mesh_face_centroid_world


_N_SLIDE_DENSE   = 25       # along slide axis (dense, for body slide-range detection)
_N_HEIGHT        = 11
_SAFE_MARGIN     = 0.10
_BODY_W_TOL      = 0.010    # 10mm: how close to the median to count as "body" slide
_MIN_BODY_FRAC   = 0.30     # require at least 30% of slide samples to be body


def _build_bvh_from_objects(objs):
    bm = bmesh.new()
    aabb_min = None; aabb_max = None
    for o in objs:
        if o.type != "MESH" or o.data is None: continue
        tmp = bmesh.new()
        tmp.from_mesh(o.data)
        tmp.transform(o.matrix_world)
        verts_lookup = {tv.index: bm.verts.new(tv.co) for tv in tmp.verts}
        bm.verts.ensure_lookup_table()
        for tf in tmp.faces:
            try: bm.faces.new([verts_lookup[v.index] for v in tf.verts])
            except ValueError: pass
        for v in tmp.verts:
            p = np.array([v.co.x, v.co.y, v.co.z])
            aabb_min = p if aabb_min is None else np.minimum(aabb_min, p)
            aabb_max = p if aabb_max is None else np.maximum(aabb_max, p)
        tmp.free()
    bm.faces.ensure_lookup_table()
    if len(bm.faces) == 0:
        bm.free(); return None, None, None
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh, aabb_min, aabb_max


def _aabb_extent_along(aabb_min, aabb_max, direction):
    corners = np.array(np.meshgrid([aabb_min[0], aabb_max[0]],
                                    [aabb_min[1], aabb_max[1]],
                                    [aabb_min[2], aabb_max[2]])).T.reshape(-1, 3)
    projs = corners @ direction
    return float(projs.min()), float(projs.max())


def _outer_w_per_slide(bvh, slide_samples, height_samples,
                       width, slide, height, cast_w_proj, cast_dir, dist):
    """For each slide sample, return (slide_pos, outermost_W) across height samples.
    None if no hit at this slide."""
    out = []
    for sp in slide_samples:
        hits = []
        for hp in height_samples:
            origin = cast_w_proj * width + sp * slide + hp * height
            origin_v = Vector((float(origin[0]), float(origin[1]), float(origin[2])))
            hit_pos, _, _, _ = bvh.ray_cast(origin_v, cast_dir, dist)
            if hit_pos is not None:
                hits.append(float(np.asarray(hit_pos).dot(width)))
        out.append((float(sp), hits))
    return out


def coborder_rail_to_drawer(snap_result: RailSnapResult,
                                drawer_mesh_objs: list,
                                body_slide_range=None) -> dict:
    """v2 entrypoint -- same signature as v1 (body_slide_range kept for
    interface compatibility but ignored: v2 detects body slide range
    itself from the drawer mesh)."""
    refs = snap_result.refs
    placement = snap_result.placement
    diag: dict = {"side": placement.side, "style": refs.style, "version": "v2"}

    if not drawer_mesh_objs:
        diag["skipped"] = "no drawer mesh"; return diag
    bvh, aabb_min, aabb_max = _build_bvh_from_objects(drawer_mesh_objs)
    if bvh is None:
        diag["skipped"] = "no faces"; return diag

    width  = np.asarray([float(x) for x in placement.width_axis_world]); width  /= max(np.linalg.norm(width), 1e-12)
    height = np.asarray([float(x) for x in placement.height_axis_world]); height /= max(np.linalg.norm(height), 1e-12)
    slide  = np.asarray([float(x) for x in placement.slide_axis_world]); slide  /= max(np.linalg.norm(slide), 1e-12)

    w_min, w_max = _aabb_extent_along(aabb_min, aabb_max, width)
    h_min, h_max = _aabb_extent_along(aabb_min, aabb_max, height)
    s_min, s_max = _aabb_extent_along(aabb_min, aabb_max, slide)

    if placement.side == "left":
        cast_w_proj = w_min - _SAFE_MARGIN
        cast_dir = Vector(list(width))
        side_sign_inward = +1   # toward drawer center
    else:
        cast_w_proj = w_max + _SAFE_MARGIN
        cast_dir = Vector(list(-width))
        side_sign_inward = -1
    cast_dist = (w_max - w_min) + 2 * _SAFE_MARGIN

    sample_slide = np.linspace(s_min + 0.02*(s_max-s_min),
                               s_max - 0.02*(s_max-s_min),
                               _N_SLIDE_DENSE)
    sample_height = np.linspace(h_min + 0.05*(h_max-h_min),
                                h_max - 0.05*(h_max-h_min),
                                _N_HEIGHT)

    # ----- (a)(b) per-slide outermost W -----
    per_slide = _outer_w_per_slide(
        bvh, sample_slide, sample_height,
        width, slide, height, cast_w_proj, cast_dir, cast_dist)

    # outer_w_at_slide: list of (slide_pos, outer_w) for slides that had any hit
    outer_w_at_slide: list[tuple[float, float]] = []
    for sp, hits in per_slide:
        if not hits: continue
        outer = min(hits) if placement.side == "left" else max(hits)
        outer_w_at_slide.append((sp, outer))

    if len(outer_w_at_slide) < 3:
        diag["skipped"] = f"too few hits ({len(outer_w_at_slide)})"; return diag

    outer_ws = np.array([w for _, w in outer_w_at_slide])

    # ----- (c) body W via robust median -----
    body_W = float(np.median(outer_ws))

    # ----- (d) body slide range = contiguous run near body_W -----
    flags = [abs(w - body_W) < _BODY_W_TOL for w in outer_ws]
    # Largest contiguous run of body slides
    best_run = (0, 0); cur_start = None; cur_len = 0
    runs = []
    for i, f in enumerate(flags):
        if f:
            if cur_start is None: cur_start = i
            cur_len += 1
            if cur_len > best_run[1] - best_run[0]:
                best_run = (cur_start, i + 1)
        else:
            if cur_start is not None:
                runs.append((cur_start, cur_start + cur_len))
            cur_start = None; cur_len = 0
    if best_run == (0, 0):
        # No sample within tolerance of the median (bimodal outer_w): fall
        # back to the full slide range instead of a meaningless [0, -1] span.
        diag["warn_body_run"] = "no contiguous run near median; using full range"
        best_run = (0, len(outer_w_at_slide))
    body_idx_range = best_run
    body_slide_min = outer_w_at_slide[body_idx_range[0]][0]
    body_slide_max = outer_w_at_slide[body_idx_range[1] - 1][0]

    body_frac = (body_idx_range[1] - body_idx_range[0]) / max(len(outer_ws), 1)
    diag["body_W"] = body_W
    diag["body_slide_range"] = (body_slide_min, body_slide_max)
    diag["body_frac"] = body_frac
    diag["n_slide_hits"] = len(outer_w_at_slide)

    if body_frac < _MIN_BODY_FRAC:
        # Very few "body" samples -- something off; fall through to
        # plain median anyway, but note in diag.
        diag["warn"] = f"body_frac={body_frac:.2f} below {_MIN_BODY_FRAC}"

    # Heights with hits at any of the BODY slide positions (for vertical
    # centering / cluster-based centering on center style).
    body_hit_heights: list[float] = []
    body_slide_set = {sp for i, (sp, _) in enumerate(outer_w_at_slide)
                      if body_idx_range[0] <= i < body_idx_range[1]}
    # Re-cast at body slide positions only, recording hit height.
    for sp, hits in per_slide:
        if sp not in body_slide_set: continue
        # Re-cast each height to know which heights produced hits.
        for hp in sample_height:
            origin = cast_w_proj * width + sp * slide + hp * height
            origin_v = Vector((float(origin[0]), float(origin[1]), float(origin[2])))
            hit_pos, _, _, _ = bvh.ray_cast(origin_v, cast_dir, cast_dist)
            if hit_pos is not None:
                body_hit_heights.append(float(hp))
    body_hit_heights_set = set(body_hit_heights)

    # ----- (e) translate rail so snap_dynamic centroid_W = body_W -----
    sd_centroid = _mesh_face_centroid_world(refs.snap_dynamic)
    cur_w = float(np.asarray(sd_centroid).dot(width))
    delta_w = body_W - cur_w
    snap_result.root.location = snap_result.root.location + Vector(list(width * delta_w))
    bpy.context.view_layer.update()
    diag["width_delta_mm"] = delta_w * 1000

    # ----- CENTER-style vertical centering on largest hit-height cluster -----
    if refs.snap_support is None and body_hit_heights_set:
        # Walk sample_height in order; longest contiguous run of hits.
        best_run_h: list[float] = []
        cur: list[float] = []
        for hp in sample_height:
            if float(hp) in body_hit_heights_set:
                cur.append(float(hp))
                if len(cur) > len(best_run_h):
                    best_run_h = cur[:]
            else:
                cur = []
        if best_run_h:
            body_h_mid = 0.5 * (best_run_h[0] + best_run_h[-1])
            sd_c = _mesh_face_centroid_world(refs.snap_dynamic)
            cur_h = float(np.asarray(sd_c).dot(height))
            delta_h = body_h_mid - cur_h
            snap_result.root.location = snap_result.root.location + Vector(list(height * delta_h))
            bpy.context.view_layer.update()
            diag["height_delta_mm"] = delta_h * 1000
            diag["body_h_extent"] = (best_run_h[0], best_run_h[-1])

    # ----- CORNER-style support snap to drawer bottom -----
    if refs.snap_support is not None:
        cast_h_proj = h_min - _SAFE_MARGIN
        cast_dir_h = Vector(list(height))
        sample_width = np.linspace(w_min + 0.05*(w_max-w_min),
                                   w_max - 0.05*(w_max-w_min),
                                   _N_HEIGHT)
        # Use the CENTER of body_slide_range for height refinement: drop
        # 10% from each end. Edge slide positions can sit right at the
        # pulling-board / body transition where the cast may hit a
        # transitional feature lower than the bottom plate, dragging the
        # target down by a few mm and breaking snap quality.
        body_idx_lo = body_idx_range[0]
        body_idx_hi = body_idx_range[1]
        body_n = body_idx_hi - body_idx_lo
        trim = max(1, int(0.1 * body_n)) if body_n >= 5 else 0
        body_slide_samples = [outer_w_at_slide[i][0]
                              for i in range(body_idx_lo + trim, body_idx_hi - trim)]
        if not body_slide_samples:  # fallback if trimming exhausted the range
            body_slide_samples = [outer_w_at_slide[i][0]
                                  for i in range(body_idx_lo, body_idx_hi)]
        cast_dist_h = (h_max - h_min) + 2 * _SAFE_MARGIN
        hits_h: list[float] = []
        for sp in body_slide_samples:
            for wp in sample_width:
                origin = wp * width + sp * slide + cast_h_proj * height
                origin_v = Vector((float(origin[0]), float(origin[1]), float(origin[2])))
                hit_pos, _, _, _ = bvh.ray_cast(origin_v, cast_dir_h, cast_dist_h)
                if hit_pos is not None:
                    hits_h.append(float(np.asarray(hit_pos).dot(height)))
        if hits_h:
            target_h = min(hits_h)
            ss_c = _mesh_face_centroid_world(refs.snap_support)
            cur_h = float(np.asarray(ss_c).dot(height))
            delta_h = target_h - cur_h
            snap_result.root.location = snap_result.root.location + Vector(list(height * delta_h))
            bpy.context.view_layer.update()
            diag["height_delta_mm"] = delta_h * 1000
            diag["height_target_world"] = target_h
            diag["height_n_hits"] = len(hits_h)

    return diag
