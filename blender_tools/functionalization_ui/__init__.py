# ============================================================================
# GraFu Functionalization Suite — Blender Add-on
# ============================================================================
# Graph-driven workflow: load a GraFu prediction (pred + input + meshes),
# install hinges / rails / handles / tops / interiors per predicted edges,
# refine per joint, preview motion, export.
#
# The implementation layer (hinge / rail / add_handle / add_top /
# install_policy_fpc / instantiate_interior) is shared verbatim with the
# headless batch installer blender_tools/install_from_pred.py.
# ============================================================================

bl_info = {
    "name": "GraFu Functionalization Suite",
    "author": "GraFu",
    "version": (3, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Functionalize",
    "description": "Interactive functionalization of GraFu graph predictions "
                   "(hinges, rails, handles, tops, interiors)",
    "warning": "",
    "doc_url": "",
    "category": "Object",
}

from pathlib import Path

import bpy

from bpy.app.handlers import persistent

from .utils import graph_properties as _graph_props
from .utils import install_helpers as _install_helpers
from .operators import graph_operators as _graph_ops
from .ui import graph_panel as _graph_panel
from .ui import overlay as _overlay


@persistent
def _restore_overlay_on_load(_path):
    """Property update callbacks don't fire on file load: a .blend saved
    with the motion overlay enabled would otherwise draw nothing until the
    checkbox is toggled off and on again."""
    try:
        props = bpy.context.scene.graph_func_props
        if getattr(props, "show_overlay", False):
            _overlay.enable()
            _overlay.rebuild_from_props(props)
    except Exception:
        pass


def _detect_repo_root() -> Path | None:
    """Locate the GraFu repo so hinge/rail and the template assets
    resolve. Three layouts:
      (a) bundled — the addon lives at <repo>/blender_tools/functionalization_ui
      (b) installed under Blender's addons/ with a `repo_location.txt`
          (one line: the repo path) written next to this file at deploy time
      (c) installed with neither — 'Repo root' must be set in the panel
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2],   # bundled: <repo>/blender_tools/functionalization_ui
    ]
    pointer = here.parent / "repo_location.txt"
    if pointer.is_file():
        try:
            candidates.insert(0, Path(pointer.read_text().strip()))
        except Exception:
            pass
    for c in candidates:
        if (c / "blender_tools" / "hinge").is_dir():
            return c
    return None


def register():
    repo_root = _detect_repo_root()
    if repo_root is not None:
        _install_helpers.init_paths(repo_root)
        _graph_props.set_hinge_variant_items(
            _install_helpers.available_hinge_variants())
        _graph_props.set_rail_variant_items(
            _install_helpers.available_rail_variants())
    else:
        print("[functionalization_ui] WARNING: repo root not auto-detected; "
              "set 'Repo root' under Manual paths before loading.")

    _graph_props.register()
    _graph_ops.register()
    _graph_panel.register()
    if _restore_overlay_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_restore_overlay_on_load)
    print("GraFu Functionalization Suite: registered.")


def unregister():
    if _restore_overlay_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_restore_overlay_on_load)
    _overlay.disable()
    _graph_panel.unregister()
    _graph_ops.unregister()
    _graph_props.unregister()
    print("GraFu Functionalization Suite: unregistered.")


if __name__ == "__main__":
    register()
