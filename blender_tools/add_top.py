"""Add a top panel to a predicted furniture model.

Designed to be called from install_from_pred when a 'top panel' node is
predicted in the graph. The caller passes the list of body objects (the
node objects whose material is NOT door/drawer/handle), and this module
merges them into a temp mesh, runs the addon's top-processor over it,
then renames + materials the resulting top so it integrates with the
rest of the predicted model.

Public API:
    add_top_from_body_meshes(body_objs, **kwargs) -> bpy.types.Object | None
    is_top_panel_node(node_dict) -> bool

Imports the actual top algorithm from the Functionalization Suite addon.
Resolved at module load: the bundled copy next to this file
(blender_tools/functionalization_ui) is tried first, then the installed
copies under ~/.config/blender/<ver>/scripts/addons/.
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Iterable, Optional

import bpy
from mathutils import Vector


# ── addon import boilerplate ────────────────────────────────────────────────

def _resolve_addon_path() -> Optional[Path]:
    """Locate the functionalization_ui package and add its parent to sys.path.
    Returns the package directory, or None if not found."""
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
        "Install the addon (or symlink) under ~/.config/blender/<ver>/scripts/addons/."
    )

from functionalization_ui.utils.top_processor import (  # noqa: E402
    add_top_to_object,
    detect_and_add_top,
    remove_existing_top_objects,
)


# ── helpers ─────────────────────────────────────────────────────────────────

_DEFAULT_EXCLUDE_MATS = ("door", "drawer", "handle")


def _mat_base(mat: str) -> str:
    """Strip Blender's `.001` numeric suffix from a material name."""
    return re.sub(r"\.\d+$", "", (mat or "")).strip().lower()


def is_top_panel_node(node: dict) -> bool:
    """True when a predicted graph node represents a top panel / countertop."""
    base = _mat_base(node.get("material", ""))
    return base in {"top panel", "top", "countertop"}


def _build_combined_temp(body_objs: Iterable[bpy.types.Object],
                         name: str) -> Optional[bpy.types.Object]:
    """Duplicate every body object, join the duplicates, return the result.
    Originals are not modified. Returns None if there are no inputs."""
    body_objs = [o for o in body_objs if o is not None and o.type == "MESH"
                 and o.data is not None and len(o.data.vertices) > 0]
    if not body_objs:
        return None

    bpy.ops.object.select_all(action="DESELECT")
    duplicates = []
    for o in body_objs:
        dup = o.copy()
        dup.data = o.data.copy()
        dup.name = f"{name}_dup_{o.name}"
        # Keep the duplicate flat in the scene (drop parent so transforms
        # bake to its own matrix_world before the join).
        bpy.context.scene.collection.objects.link(dup)
        dup.parent = None
        dup.matrix_world = o.matrix_world.copy()
        duplicates.append(dup)

    bpy.context.view_layer.update()
    for d in duplicates:
        d.select_set(True)
    bpy.context.view_layer.objects.active = duplicates[0]

    if len(duplicates) > 1:
        bpy.ops.object.join()

    joined = bpy.context.active_object
    joined.name = name
    # Apply transforms so the top-processor sees a clean axis-aligned mesh.
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.context.view_layer.update()
    return joined


def _cleanup_temp(obj: Optional[bpy.types.Object]) -> None:
    if obj is None:
        return
    me = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


# ── public api ──────────────────────────────────────────────────────────────

def add_top_from_body_meshes(
    body_objs: Iterable[bpy.types.Object],
    *,
    up_axis: str = "+Z",
    overhang_pct: float = 0.03,
    thickness_abs: float = 0.020,
    footprint_mode: str = "min_area_rect",
    shape_style: str = "RECTANGLE",
    corner_radius: float = 0.15,
    corner_segments: int = 8,
    auto_detect: bool = False,
    coverage_thresh: float = 0.6,
    force_add: bool = True,
    min_generated_top_area_frac: float = 0.30,
    slice_delta_ratio: float = 0.001,
    name: str = "predicted_top_panel",
    material: Optional[bpy.types.Material] = None,
    replace_existing: bool = True,
) -> Optional[bpy.types.Object]:
    """Add a top panel above the union of the given body meshes.

    Inputs:
      body_objs   — iterable of bpy mesh objects to use as the cabinet body.
                    The caller is responsible for excluding doors/drawers/
                    handles (use is_top_panel_node + the predicted graph's
                    material labels to filter).
      up_axis     — door-up axis (default +Z).
      overhang_pct, thickness_abs, shape_style, corner_radius — passed
                    through to the addon's top processor. Defaults match
                    typical cabinet-top conventions.
      auto_detect — if True, only add the top when the silhouette is missing
                    one (coverage < threshold). Default False so a predicted
                    "top panel" node always produces a top.
      replace_existing — remove any prior `<name>_Top` object before adding.

    Returns the new top mesh object, or None on failure (logs the reason).
    """
    body_list = [o for o in body_objs if o is not None and o.type == "MESH"
                 and o.data is not None and len(o.data.vertices) > 0]
    if not body_list:
        print("[add_top] no usable body meshes; skipping")
        return None

    if replace_existing:
        try:
            removed = remove_existing_top_objects(name)
            if removed:
                print(f"[add_top] removed {removed} prior top object(s) for '{name}'")
        except Exception as exc:
            print(f"[add_top] remove_existing_top_objects warning: {exc}")

    # Save selection state, build the temp combined mesh, run the processor.
    prev_active = bpy.context.view_layer.objects.active
    prev_selected = list(bpy.context.selected_objects)

    temp = None
    try:
        temp = _build_combined_temp(body_list, name=f"{name}__combined")
        if temp is None:
            print("[add_top] failed to build combined temp mesh")
            return None

        if auto_detect:
            result = detect_and_add_top(
                obj=temp,
                up_axis=up_axis,
                force_add=force_add,
                coverage_thresh=coverage_thresh,
                overhang_pct=overhang_pct,
                thickness_abs=thickness_abs,
                footprint_mode=footprint_mode,
                min_generated_top_area_frac=min_generated_top_area_frac,
                slice_delta_ratio=slice_delta_ratio,
                shape_style=shape_style,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
                create_separate=True,
            )
            success = result.get("top_added", False)
            err = result.get("error")
            info = result.get("creation_info") or {}
        else:
            success, info = add_top_to_object(
                obj=temp,
                up_axis=up_axis,
                overhang_pct=overhang_pct,
                thickness_abs=thickness_abs,
                footprint_mode=footprint_mode,
                min_generated_top_area_frac=min_generated_top_area_frac,
                slice_delta_ratio=slice_delta_ratio,
                shape_style=shape_style,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
                create_separate=True,
            )
            err = None if success else info.get("reason")

        if not success:
            print(f"[add_top] top processor failed: {err or 'unknown'}; info={info}")
            return None

        # The processor names the new top "<temp.name>_Top". Rename to <name>.
        old_top_name = f"{temp.name}_Top"
        new_top = bpy.data.objects.get(old_top_name)
        if new_top is None:
            # In some paths the processor names it slightly differently;
            # search for any top-suffixed object that we just created.
            new_top = next((o for o in bpy.data.objects
                            if o.name.startswith(temp.name) and o.name.endswith("_Top")),
                           None)
        if new_top is None:
            print(f"[add_top] processor reported success but '_Top' object not found")
            return None

        new_top.name = name
        if new_top.data is not None:
            new_top.data.name = f"{name}_mesh"

        if material is not None:
            new_top.data.materials.clear()
            new_top.data.materials.append(material)

        print(f"[add_top] added '{name}' from {len(body_list)} body meshes "
              f"(thickness={thickness_abs*1000:.0f}mm, shape={shape_style})")
        return new_top

    finally:
        _cleanup_temp(temp)
        bpy.ops.object.select_all(action="DESELECT")
        for o in prev_selected:
            if o is not None and o.name in bpy.data.objects:
                o.select_set(True)
        if prev_active is not None and prev_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = prev_active


def add_top_for_predicted_node(
    pred_node: dict,
    pred_id_to_obj: dict,
    pred_nodes: dict,
    *,
    exclude_mats: tuple = _DEFAULT_EXCLUDE_MATS,
    **kwargs,
) -> Optional[bpy.types.Object]:
    """Convenience: when install_from_pred sees a top-panel-class node,
    select all current scene meshes whose predicted material is NOT in
    exclude_mats, and call add_top_from_body_meshes.

    Args:
      pred_node      — the predicted top-panel node dict (used for material
                       lookup and naming).
      pred_id_to_obj — install_from_pred's mapping from predicted node id
                       to the corresponding bpy mesh object.
      pred_nodes     — pred["nodes"] dict (so we can look up each node's
                       predicted material).
      exclude_mats   — material bases to skip (default: door/drawer/handle).
      **kwargs       — forwarded to add_top_from_body_meshes.
    """
    exclude_set = {m.lower() for m in exclude_mats}
    body_objs = []
    for nid, obj in pred_id_to_obj.items():
        node = pred_nodes.get(nid, {})
        mat = _mat_base(node.get("material", ""))
        if mat in exclude_set:
            continue
        body_objs.append(obj)

    if not body_objs:
        print("[add_top] no body objects after filtering by material")
        return None

    return add_top_from_body_meshes(body_objs, **kwargs)
