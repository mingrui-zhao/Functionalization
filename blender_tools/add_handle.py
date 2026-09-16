"""Attach a handle to a predicted door or drawer.

Designed to be called from install_from_pred when a 'handle' node is
predicted with an `attached` edge to a door/drawer node.

Placement rule:

  DOORS:    handle is on the side OPPOSITE the predicted hinge axis.
            Long-axis orientation follows the hinge axis projected onto the
            door plane: side-flip (vertical hinge) -> vertical bar handle;
            top/bottom-flip (horizontal hinge)     -> horizontal bar handle.
  DRAWERS:  handle is centered horizontally, vertically centered (or
            offset to the front face for cabinets where the slide axis
            normal is the natural top), with vertical margin.

The actual snap+rotation+scale is delegated to the Functionalization
Suite addon's `apply_handle_to_selected_object` — we only construct the
correct `joint_info` (motion axis + origin) so the addon's hinge-side
detection routes the handle to the opposite edge.

Public API:
    add_handle_for_door(parent_obj, hinge_axis, hinge_origin=None, **kwargs)
    add_handle_for_drawer(parent_obj, slide_axis, slide_origin=None, **kwargs)
    add_handle_for_predicted_attachment(parent_obj, motion_axis,
                                        parent_kind='door', ...)
    is_handle_node(node_dict) -> bool
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import bpy
import numpy as np
from mathutils import Vector


# ── addon import boilerplate (mirrors add_top.py) ───────────────────────────

def _resolve_addon_path() -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().parent,   # bundled: blender_tools/functionalization_ui
        Path.home() / ".config/blender/5.0/scripts/addons",
        Path.home() / ".config/blender/4.2/scripts/addons",
    ]
    for parent in candidates:
        pkg = parent / "functionalization_ui"
        if pkg.is_dir() and (pkg / "__init__.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return pkg
    return None


_ADDON_PKG = _resolve_addon_path()
if _ADDON_PKG is None:
    raise ImportError(
        "functionalization_ui addon not found on any expected path. "
        "Install the addon under ~/.config/blender/<ver>/scripts/addons/ "
        "or run from the GraFu checkout (blender_tools/)."
    )

from functionalization_ui.utils.handle_placement import (  # noqa: E402
    apply_handle_to_selected_object,
)


# ── helpers ─────────────────────────────────────────────────────────────────

_DEFAULT_HANDLE_STYLE = "handle3_bar"
_DEFAULT_FRONT_AXIS   = "-Y"
_DEFAULT_UP_AXIS      = "+Z"


def _mat_base(mat: str) -> str:
    return re.sub(r"\.\d+$", "", (mat or "")).strip().lower()


def is_handle_node(node: dict) -> bool:
    return _mat_base(node.get("material", "")) == "handle"


def _to_vector(v) -> Optional[Vector]:
    if v is None:
        return None
    if isinstance(v, Vector):
        return v
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size != 3:
        return None
    return Vector((float(arr[0]), float(arr[1]), float(arr[2])))


def _world_centroid(obj: bpy.types.Object) -> Vector:
    """Geometric centroid of `obj` mesh in world space (vertex-mean).
    Used as the default joint origin if the caller didn't supply one."""
    if obj.type != "MESH" or obj.data is None or len(obj.data.vertices) == 0:
        return Vector(obj.matrix_world.translation)
    s = Vector((0, 0, 0))
    for v in obj.data.vertices:
        s += obj.matrix_world @ v.co
    return s / len(obj.data.vertices)


def _select_only(obj: bpy.types.Object) -> None:
    """Make `obj` the sole selected and active object."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


# ── public api ──────────────────────────────────────────────────────────────

def add_handle_for_door(
    parent_obj: bpy.types.Object,
    hinge_axis,
    hinge_origin=None,
    *,
    handle_style: str = _DEFAULT_HANDLE_STYLE,
    front_axis: str = _DEFAULT_FRONT_AXIS,
    up_axis: str = _DEFAULT_UP_AXIS,
    edge_margin_frac: float = 0.10,
    auto_rescale: bool = True,
    partition_frac: Optional[float] = None,
    n_handles: int = 1,
) -> bool:
    """Place a handle on `parent_obj` (a door mesh) on the side opposite
    the hinge.

    Args:
      parent_obj        — bpy door mesh.
      hinge_axis        — vec3 world-space hinge rotation axis (predicted).
      hinge_origin      — vec3 world-space point on the axis. Defaults to
                          door centroid (the addon only needs *a* point on
                          the hinge line for opposite-side detection).
      handle_style      — file-stem of a handle template (e.g. 'handle3_bar').
      edge_margin_frac  — distance from the side edge as a fraction of the
                          door's right-axis span. 0.10 = 10% from the corner.
      auto_rescale      — let the addon's protrusion-driven scale rule run.

    Returns True on success.
    """
    if parent_obj is None or parent_obj.type != "MESH":
        print("[add_handle/door] parent_obj must be a mesh"); return False

    axis_vec = _to_vector(hinge_axis)
    if axis_vec is None or axis_vec.length < 1e-9:
        print("[add_handle/door] hinge_axis is zero or invalid"); return False
    axis_vec.normalize()

    origin_vec = _to_vector(hinge_origin)
    if origin_vec is None:
        origin_vec = _world_centroid(parent_obj)

    joint_info = {
        "type":      "hinge",
        "origin":    origin_vec,
        "direction": axis_vec,
    }

    # Snapshot prior selection so we can call apply_handle... cleanly.
    prev_active = bpy.context.view_layer.objects.active
    prev_selected = list(bpy.context.selected_objects)
    _select_only(parent_obj)

    try:
        # align_long_to_motion projects the hinge axis onto the door plane:
        #   side-flip doors (vertical hinge axis)  -> long axis vertical
        #   top/bottom-flip doors (horizontal axis) -> long axis horizontal
        # Top-flip cases were previously mis-rotated to vertical because the
        # mode was hardcoded to align_long_to_up.
        ok = apply_handle_to_selected_object(
            handle_style=handle_style,
            target_type="door",
            front_axis=front_axis,
            up_axis=up_axis,
            edge_margin_frac=edge_margin_frac,
            auto_rescale=auto_rescale,
            joint_info=joint_info,
            twist_mode="align_long_to_motion",
            partition_frac=partition_frac,
            n_handles=n_handles,
        )
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for o in prev_selected:
            if o is not None and o.name in bpy.data.objects:
                o.select_set(True)
        if prev_active is not None and prev_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = prev_active

    if not ok:
        print(f"[add_handle/door] failed for {parent_obj.name}")
    return bool(ok)


def add_handle_for_drawer(
    parent_obj: bpy.types.Object,
    slide_axis,
    slide_origin=None,
    *,
    handle_style: str = _DEFAULT_HANDLE_STYLE,
    front_axis: str = _DEFAULT_FRONT_AXIS,
    up_axis: str = _DEFAULT_UP_AXIS,
    edge_margin_frac: float = 0.10,
    auto_rescale: bool = True,
    partition_frac: Optional[float] = None,
    n_handles: int = 1,
) -> bool:
    """Place a handle on `parent_obj` (a drawer face mesh).

    For drawers the placement convention is "centered horizontally,
    centered vertically" — the slide axis tells us which way the drawer
    pulls (it's the front-face normal), and the addon picks 'drawer'
    geometry orientation (horizontal long axis).
    """
    if parent_obj is None or parent_obj.type != "MESH":
        print("[add_handle/drawer] parent_obj must be a mesh"); return False

    axis_vec = _to_vector(slide_axis)
    if axis_vec is None or axis_vec.length < 1e-9:
        print("[add_handle/drawer] slide_axis is zero or invalid"); return False
    axis_vec.normalize()

    origin_vec = _to_vector(slide_origin)
    if origin_vec is None:
        origin_vec = _world_centroid(parent_obj)

    joint_info = {
        "type":      "slider",
        "origin":    origin_vec,
        "direction": axis_vec,
    }

    prev_active = bpy.context.view_layer.objects.active
    prev_selected = list(bpy.context.selected_objects)
    _select_only(parent_obj)

    try:
        ok = apply_handle_to_selected_object(
            handle_style=handle_style,
            target_type="drawer",
            front_axis=front_axis,
            up_axis=up_axis,
            edge_margin_frac=edge_margin_frac,
            auto_rescale=auto_rescale,
            joint_info=joint_info,
            twist_mode="align_long_to_right",
            partition_frac=partition_frac,
            n_handles=n_handles,
        )
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for o in prev_selected:
            if o is not None and o.name in bpy.data.objects:
                o.select_set(True)
        if prev_active is not None and prev_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = prev_active

    if not ok:
        print(f"[add_handle/drawer] failed for {parent_obj.name}")
    return bool(ok)


def add_handle_for_predicted_attachment(
    parent_obj: bpy.types.Object,
    motion_axis,
    *,
    parent_kind: str = "door",
    motion_origin=None,
    handle_style: str = _DEFAULT_HANDLE_STYLE,
    **kwargs,
) -> bool:
    """Single dispatch entry point for install_from_pred.

    parent_kind: 'door' (revolute joint) or 'drawer' (prismatic joint).
    motion_axis: the predicted hinge axis (door) or rail slide axis (drawer).
    motion_origin: optional point on the motion axis.
    """
    kind = parent_kind.strip().lower()
    if kind == "door":
        return add_handle_for_door(parent_obj, motion_axis, motion_origin,
                                   handle_style=handle_style, **kwargs)
    if kind in ("drawer", "slider"):
        return add_handle_for_drawer(parent_obj, motion_axis, motion_origin,
                                     handle_style=handle_style, **kwargs)
    print(f"[add_handle] unknown parent_kind: {parent_kind!r}")
    return False
