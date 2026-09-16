"""Detect drawer body region (excluding pulling board) from mesh fragments.

HSSD drawers are typically split into multiple sub-meshes:
  * a large flat bottom-plate fragment (the body) -- many verts, large
    slide-axial extent
  * one or more thin pulling-board / face fragments at the front -- few
    verts, tiny slide-axial extent (just the board's thickness)

Heuristic:
  body = fragment with the LARGEST slide-axial extent.
  Drawer-body bounds along the slide axis are this fragment's
  [min_proj, max_proj] in world.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import bpy  # type: ignore


@dataclass
class BodyRegion:
    body_obj_name: str
    slide_min_world: float          # projection on slide_axis_world
    slide_max_world: float
    body_length: float
    body_center_along_slide: float   # midpoint
    diagnostic: dict


def detect_body_region(
    drawer_mesh_objs: list[bpy.types.Object],
    slide_axis_world: np.ndarray,
) -> BodyRegion | None:
    """Pick the fragment with the largest slide-axial extent and return its
    slide-axial bounds. Returns None if no fragments."""
    if not drawer_mesh_objs:
        return None
    sa = np.asarray(slide_axis_world, dtype=float)
    sa_norm = float(np.linalg.norm(sa))
    if sa_norm < 1e-9:
        return None
    sa = sa / sa_norm

    extents: list[tuple[float, float, float, str]] = []  # (extent, min, max, name)
    for o in drawer_mesh_objs:
        if o.type != "MESH" or o.data is None or len(o.data.vertices) == 0:
            continue
        M = o.matrix_world
        proj = np.array([float((M @ v.co).x * sa[0]
                              + (M @ v.co).y * sa[1]
                              + (M @ v.co).z * sa[2])
                         for v in o.data.vertices])
        if len(proj) == 0:
            continue
        extents.append((proj.max() - proj.min(), float(proj.min()),
                        float(proj.max()), o.name))
    if not extents:
        return None
    # Body = fragment with LARGEST slide-axial extent
    extents.sort(key=lambda t: -t[0])
    ext, lo, hi, name = extents[0]
    return BodyRegion(
        body_obj_name=name,
        slide_min_world=lo,
        slide_max_world=hi,
        body_length=ext,
        body_center_along_slide=(lo + hi) * 0.5,
        diagnostic={"all_fragment_extents": extents},
    )
