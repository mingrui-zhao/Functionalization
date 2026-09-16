"""Compute uniform rail scale = body_length / template_rail_slide_extent.

Probe the rail template once to measure its slide-axial extent (along
sliding_axis empty's local +Z direction in template-world), then return
body_length / template_extent. Caller applies this as
refs.root.scale *= factor BEFORE the snap reads template state.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import bpy  # type: ignore
from mathutils import Vector  # type: ignore

from .templates import load_rail_template


_MIN_SCALE = 0.10   # permissive: thin drawer bodies need scales near 0.2
_MAX_SCALE = 3.0


def _measure_template_slide_geometry(template_path: Path,
                                      probe_suffix: str = "rail_scale_probe"
                                      ) -> tuple[float, float, float]:
    """Probe-load template; return (slide_extent, snap_static_proj_in_template,
    rail_min_proj). All projections are onto sliding_axis empty's local +Z
    direction in template world. Probe is removed before return."""
    refs = load_rail_template(template_path, suffix=probe_suffix)
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    sa = (refs.axis.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()

    # Aggregate slide projections of every rail mesh vertex.
    projs = []
    for o in (refs.static_end, refs.dynamic_end, refs.snap_static, refs.snap_dynamic):
        if o is None or o.data is None: continue
        M = o.matrix_world
        for v in o.data.vertices:
            projs.append(float((M @ v.co).dot(sa)))
    if refs.snap_support is not None and refs.snap_support.data is not None:
        M = refs.snap_support.matrix_world
        for v in refs.snap_support.data.vertices:
            projs.append(float((M @ v.co).dot(sa)))
    if not projs:
        rail_min = 0.0; rail_max = 0.0
    else:
        rail_min = min(projs); rail_max = max(projs)

    # snap_static centroid projection
    M = refs.snap_static.matrix_world
    sc_acc = Vector((0, 0, 0));  sc_a = 0.0
    for p in refs.snap_static.data.polygons:
        sc_acc += (M @ p.center) * p.area;  sc_a += p.area
    snap_static_proj = float((sc_acc / sc_a).dot(sa)) if sc_a > 0 else 0.0

    # Cleanup
    tag = f"__{probe_suffix}"
    for o in list(bpy.data.objects):
        if o.name.endswith(tag):
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.context.view_layer.update()
    return (rail_max - rail_min), snap_static_proj - rail_min, rail_min


def compute_rail_scale(template_path: Path, body_length: float,
                       front_margin_frac: float = 0.10
                       ) -> tuple[float, float, dict]:
    """Returns (clamped_scale, snap_static_offset_from_back_in_unscaled_template, diag).

    front_margin_frac: leave this fraction of body_length empty at the
    front (pulling-board side) so the rail doesn't tuck into the pulling
    board. Default 0.10 = 10% margin. Effective rail length =
    body_length * (1 - front_margin_frac). The rail's BACK extent still
    lands at body_back; only the FRONT extent is shortened.
    """
    template_extent, snap_static_offset, _rail_min = _measure_template_slide_geometry(template_path)
    if template_extent < 1e-6 or body_length < 1e-6:
        return 1.0, snap_static_offset, {"reason": "degenerate"}
    target_length = body_length * (1.0 - max(0.0, front_margin_frac))
    raw = target_length / template_extent
    clamped = max(_MIN_SCALE, min(_MAX_SCALE, raw))
    return clamped, snap_static_offset, {
        "template_slide_extent_m": template_extent,
        "snap_static_offset_from_back_m": snap_static_offset,
        "body_length_m": body_length,
        "front_margin_frac": front_margin_frac,
        "target_length_m": target_length,
        "raw_scale": raw, "clamped_scale": clamped,
        "was_clamped": abs(clamped - raw) > 1e-6,
    }
