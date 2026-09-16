"""PropertyGroups for the graph-driven workflow (`scene.graph_func_props`).

Single data model for the whole addon (the previous per-panel property
block is gone). Layout:

    repo_root / pred_source / model_enum      model browser (auto-derived paths)
    pred_path / input_path / mesh_dir         manual override paths
    hinges / rails / handle_groups            per-joint record rows
    default_*                                 Apply-All defaults
    interior_*                                interior detect/generate params
    log                                       status-log entries (capped)
    live_update / show_overlay                interaction toggles

Row → viewport selection sync happens via the active-index update callbacks;
live re-apply happens via the per-row property update callbacks (guarded by
`suspend_updates()` during Load so defaults don't trigger installs).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    StringProperty, BoolProperty, IntProperty, FloatProperty,
    EnumProperty, CollectionProperty, PointerProperty,
)


# ─────────────────────────────────────────────────────────── enum inventory ─
# Populated at register time from install_helpers.available_*_variants().
# Module-level so background mode works (dynamic enum_items stay empty there)
# AND so the strings stay referenced (Blender's dynamic-enum gotcha).

_HINGE_VARIANT_ITEMS_BY_CATEGORY: dict[str, list[tuple[str, str, str]]] = {
    "interior": [("hinge_interior_01.blend", "Interior 01", "")],
    "exterior": [("hinge_exterior_01.blend", "Exterior 01", "")],
    "flat":     [("hinge_flat_01.blend",     "Flat 01",     "")],
}
_RAIL_VARIANT_ITEMS: list[tuple[str, str, str]] = [
    ("sliding_rail_02_annotated.blend", "Center (rail 02)", ""),
    ("sliding_rail_annotated.blend",    "Corner (rail 01)", ""),
]

# Model-browser caches (refreshed by graph_func.refresh_sources).
_PRED_SOURCE_ITEMS: list[tuple[str, str, str]] = [("NONE", "(scan first)", "")]
_MODEL_ITEMS: list[tuple[str, str, str]] = [("NONE", "(none found)", "")]


def set_hinge_variant_items(items_by_category: dict[str, list[tuple[str, str]]]):
    global _HINGE_VARIANT_ITEMS_BY_CATEGORY
    merged = {
        cat: [(fn, lbl, "") for lbl, fn in items]
        for cat, items in items_by_category.items()
        if items
    }
    if merged:
        _HINGE_VARIANT_ITEMS_BY_CATEGORY = merged


def set_rail_variant_items(items: list[tuple[str, str]]):
    global _RAIL_VARIANT_ITEMS
    if items:
        _RAIL_VARIANT_ITEMS = [(fn, lbl, "") for lbl, fn in items]


def set_pred_source_items(items: list[tuple[str, str, str]]):
    global _PRED_SOURCE_ITEMS
    _PRED_SOURCE_ITEMS = items or [("NONE", "(no pred dirs found)", "")]


def set_model_items(items: list[tuple[str, str, str]]):
    global _MODEL_ITEMS
    _MODEL_ITEMS = items or [("NONE", "(none found)", "")]


_AUTO_VARIANT_ITEM = [("AUTO", "(per-type defaults)",
                       "Auto mode competes all classes with their default "
                       "templates")]


def _items_for_category(self, context):
    cat = getattr(self, "category", None) or "exterior"
    if cat == "auto":
        return list(_AUTO_VARIANT_ITEM)
    return list(_HINGE_VARIANT_ITEMS_BY_CATEGORY.get(cat,
                                                     [("NONE", "(none)", "")]))


def _items_default_for_default_cat(self, context):
    cat = getattr(self, "default_hinge_cat", None) or "exterior"
    if cat == "auto":
        return list(_AUTO_VARIANT_ITEM)
    return list(_HINGE_VARIANT_ITEMS_BY_CATEGORY.get(cat,
                                                     [("NONE", "(none)", "")]))


def _items_rail(self, context):
    return list(_RAIL_VARIANT_ITEMS) or [("", "(none)", "")]


def _items_pred_source(self, context):
    return list(_PRED_SOURCE_ITEMS)


def _items_model(self, context):
    return list(_MODEL_ITEMS)


# ───────────────────────────────────────────────── update-callback plumbing ─
# Guard so programmatic writes during Load/Apply-All don't trigger installs.

_SUSPEND = 0


@contextmanager
def suspend_updates():
    global _SUSPEND
    _SUSPEND = _SUSPEND + 1
    try:
        yield
    finally:
        _SUSPEND = max(0, _SUSPEND - 1)


def updates_suspended() -> bool:
    return _SUSPEND > 0


def _live_reapply_hinge(self, context):
    props = context.scene.graph_func_props
    if updates_suspended() or not props.loaded or not props.live_update:
        return
    # Only react when this row is the active one (UIList edits always are).
    try:
        idx = list(props.hinges).index(self)
    except ValueError:
        return
    if idx != props.active_hinge_idx:
        props.active_hinge_idx = idx
    try:
        bpy.ops.graph_func.reapply_one_hinge('INVOKE_DEFAULT')
    except RuntimeError:
        pass   # failure is already surfaced via row status + log


def _live_reapply_rail(self, context):
    props = context.scene.graph_func_props
    if updates_suspended() or not props.loaded or not props.live_update:
        return
    try:
        idx = list(props.rails).index(self)
    except ValueError:
        return
    if idx != props.active_rail_idx:
        props.active_rail_idx = idx
    try:
        bpy.ops.graph_func.reapply_one_rail('INVOKE_DEFAULT')
    except RuntimeError:
        pass


def _select_active_hinge(self, context):
    if updates_suspended():
        return
    try:
        bpy.ops.graph_func.select_row_objects(row_kind='HINGE')
    except Exception:
        pass


def _select_active_rail(self, context):
    if updates_suspended():
        return
    try:
        bpy.ops.graph_func.select_row_objects(row_kind='RAIL')
    except Exception:
        pass


def _select_active_handle(self, context):
    if updates_suspended():
        return
    try:
        bpy.ops.graph_func.select_row_objects(row_kind='HANDLE')
    except Exception:
        pass


def _live_reapply_handles(self, context):
    """Re-apply the active handle group when its style/mode/params change —
    only once it has been installed at least once (status ok)."""
    props = context.scene.graph_func_props
    if updates_suspended() or not props.loaded or not props.live_update:
        return
    if not self.status.startswith("ok"):
        return
    try:
        idx = list(props.handle_groups).index(self)
    except ValueError:
        return
    if idx != props.active_hgroup_idx:
        props.active_hgroup_idx = idx
    try:
        bpy.ops.graph_func.add_group_handles('INVOKE_DEFAULT')
    except RuntimeError:
        pass   # failure is already surfaced via row status + log


def _toggle_overlay(self, context):
    from ..ui import overlay
    if self.show_overlay:
        overlay.enable()
        overlay.rebuild_from_props(self)
    else:
        overlay.disable()


def _live_retop(self, context):
    """Re-synthesize the top when its style params change, but only if a
    synthesized top already exists in the scene."""
    props = context.scene.graph_func_props
    if updates_suspended() or not props.loaded or not props.live_update:
        return
    if not any(o.name.startswith("predicted_top") for o in bpy.data.objects):
        return
    try:
        bpy.ops.graph_func.add_predicted_top('INVOKE_DEFAULT')
    except RuntimeError:
        pass


# ────────────────────────────────────────────────────────── per-joint items ─

class GraphHingeItem(PropertyGroup):
    """One row = one predicted hinge joint."""
    joint_name:  StringProperty()
    door_node:   StringProperty()
    panel_node:  StringProperty()
    category: EnumProperty(
        name="Type",
        description="Hinge class. Auto competes all classes and keeps the "
                    "least-colliding (EXTERIOR > FLAT > INTERIOR tiebreak)",
        items=[
            ("auto",     "Auto (collision-tested)",
             "Snap all classes, validate swing collision, keep the winner"),
            ("exterior", "Exterior", "Closed-state coplanar, leaves opposite"),
            ("interior", "Interior", "Closed-state perpendicular (concealed)"),
            ("flat",     "Flat",     "Closed-state coplanar, same normals"),
        ],
        default="auto",
        update=_live_reapply_hinge,
    )
    variant: EnumProperty(
        name="Variant",
        description="Template blend for the selected class",
        items=_items_for_category,
        update=_live_reapply_hinge,
    )
    scale_factor: FloatProperty(
        name="Scale",
        description="Uniform multiplier on the hinge template (1.0 = auto)",
        default=1.0, min=0.05, max=10.0,
        update=_live_reapply_hinge,
    )
    use_auto_scale: BoolProperty(
        name="Auto scale",
        description="Let hinge compute scale from door/panel dimensions",
        default=True,
        update=_live_reapply_hinge,
    )
    hinge_count: IntProperty(
        name="Hinges/door", default=2, min=1, max=4,
        update=_live_reapply_hinge,
    )
    strict_border: BoolProperty(
        name="Strict mounting edge",
        description=(
            "Force the hinge to mount on the door face predicted by the "
            "graph (hinge_border). Off = let hinge pick by minimum "
            "panel gap, which can flip left↔right on face-frame cabinets."),
        default=True,
        update=_live_reapply_hinge,
    )
    flip_side: BoolProperty(
        name="Flip side",
        description=(
            "Mirror the mounting edge to the opposite face on the same "
            "axis (min↔max). Use when the prediction has the door hinged "
            "on the wrong edge."),
        default=False,
        update=_live_reapply_hinge,
    )
    use_decompose: BoolProperty(
        name="Decompose panel",
        description=(
            "Split a fused mounting panel (face_frame) into loose bars and "
            "mount on the bar nearest the hinge axis. Turn off if the "
            "hinge fails to place (falls back to the full panel box)."),
        default=True,
        update=_live_reapply_hinge,
    )
    status:      StringProperty(default="")
    detail:      StringProperty(default="", description="Full diagnostic text")
    last_ok:     StringProperty(default="", options={'HIDDEN'})
    # JSON of the last successfully installed settings — a failed change
    # auto-reverts to these so the joint never loses its working hinge.


class GraphRailItem(PropertyGroup):
    """One row = one predicted drawer joint."""
    joint_name:  StringProperty()
    drawer_node: StringProperty()
    panel_node:  StringProperty()
    variant: EnumProperty(
        name="Rail",
        items=_items_rail,
        update=_live_reapply_rail,
    )
    status:      StringProperty(default="")
    detail:      StringProperty(default="")
    last_ok:     StringProperty(default="", options={'HIDDEN'})


# Handle template stems in annotated_mechanical_parts/ (<stem>.blend).
HANDLE_STYLE_ITEMS = [
    ("handle2_bar",  "Bar 02", ""),
    ("handle3_bar",  "Bar 03 (default)", ""),
    ("handle4_bar",  "Bar 04", ""),
    ("handle5_knob", "Knob 05", ""),
    ("handle6_bar",  "Bar 06", ""),
    ("handle7_bar",  "Bar 07", ""),
    ("handle8_bar",  "Bar 08", ""),
    ("handle9_bar",  "Bar 09", ""),
    ("handle10_bar", "Bar 10", ""),
    ("handle11_bar", "Bar 11", ""),
    ("handle12_bar", "Bar 12", ""),
    ("handle13_bar", "Bar 13", ""),
    ("handle14_bar", "Bar 14", ""),
    ("handle15_bar", "Bar 15", ""),
    ("handle16_knob", "Knob 16", ""),
    ("handle17_bar", "Bar 17", ""),
    ("handle18_bar", "Bar 18", ""),
    ("handle19_cup", "Cup 19", ""),
]


class GraphHandleGroupItem(PropertyGroup):
    """One row = one door/drawer PARENT (every movable part with a motion
    record gets a row, whether or not the model fired a handle for it).

    Grouping per parent (not per handle) is what makes the multi-handle
    partition rule installable and idempotent: Apply replaces the whole
    group — N>1 handles land as knobs at equal partitions.
    """
    parent_id:   StringProperty()
    parent_kind: StringProperty()   # door | drawer | unknown
    n_new:       IntProperty(default=0)      # fired (free-slot) handles
    n_anchored:  IntProperty(default=0)      # handles already in the input
    count: IntProperty(
        name="Count",
        description="How many handles to install on this parent "
                    "(0 = remove; >1 = knobs at equal partitions)",
        default=1, min=0, max=4,
        update=_live_reapply_handles,
    )
    mode: EnumProperty(
        name="Mode",
        description="Additive = external template handle; Recessed = a "
                    "carved pull groove in the panel itself (no attachment)",
        items=[
            ("ADDITIVE", "Additive", "Attach a template handle mesh"),
            ("RECESSED", "Recessed", "Carve a finger groove into the panel"),
        ],
        default="ADDITIVE",
        update=_live_reapply_handles,
    )
    style: EnumProperty(
        name="Style",
        description="Template for this parent's handles (multi-handle "
                    "groups auto-shrink each handle to its partition slot)",
        items=HANDLE_STYLE_ITEMS,
        default="handle3_bar",
        update=_live_reapply_handles,
    )
    margin_edge: FloatProperty(
        name="Edge margin",
        description="Distance from the pull edge (doors: the edge opposite "
                    "the hinge) as a fraction of that span",
        default=0.10, min=0.02, max=0.45, subtype='FACTOR',
        update=_live_reapply_handles,
    )
    pos_frac: FloatProperty(
        name="Position",
        description="Position along the free axis (doors: along the pull "
                    "edge; drawers: across the width). 0.5 = centered. "
                    "Ignored when Count > 1 (partitions take over)",
        default=0.5, min=0.05, max=0.95, subtype='FACTOR',
        update=_live_reapply_handles,
    )
    sub_shape: EnumProperty(
        name="Shape",
        items=[("flat", "Flat pocket", "Rectangular pocket"),
               ("u",    "U-groove",    "Rounded-bottom groove")],
        default="flat",
        update=_live_reapply_handles,
    )
    sub_depth: FloatProperty(
        name="Depth",
        description="Groove depth as a fraction of the panel thickness",
        default=0.5, min=0.1, max=0.9, subtype='FACTOR',
        update=_live_reapply_handles,
    )
    sub_method: EnumProperty(
        name="Method",
        items=[("DEFORM",  "Soft deform", "Vertex displacement — preserves topology"),
               ("BOOLEAN", "Boolean cut", "Sharp CSG cut — may add artifacts")],
        default="DEFORM",
        update=_live_reapply_handles,
    )
    replace_anchored: BoolProperty(
        name="Replace anchored",
        description="Delete the input's own handle meshes when applying "
                    "(they are replaced by the chosen template/groove)",
        default=False,
        update=_live_reapply_handles,
    )
    status:      StringProperty(default="")
    detail:      StringProperty(default="")


class GraphLogEntry(PropertyGroup):
    level: StringProperty(default="INFO")   # INFO | WARN | ERR
    text:  StringProperty(default="")


# ─────────────────────────────────────────────────────────── root props ────

class GraphFunctionalizationProperties(PropertyGroup):
    """All UI state for the graph workflow."""

    # ── model browser ──
    repo_root: StringProperty(
        name="Repo root",
        description="Path to the GraFu repo (auto-detected if blank)",
        subtype="DIR_PATH",
        default="",
    )
    pred_source: EnumProperty(
        name="Predictions",
        description="Prediction set (scanned from results/ and its "
                    "subdirectories, or a custom directory)",
        items=_items_pred_source,
    )
    custom_pred_dir: StringProperty(
        name="Custom pred dir",
        description="Directory of <mid>_pred.json files (used when "
                    "Predictions = Custom)",
        subtype="DIR_PATH",
        default="",
    )
    model_enum: EnumProperty(
        name="Model",
        description="Model id (from <mid>_pred.json files in the selected "
                    "prediction set)",
        items=_items_model,
    )
    use_manual_paths: BoolProperty(
        name="Manual paths",
        description="Type pred/input/mesh paths directly instead of using "
                    "the model browser",
        default=False,
    )
    pred_path: StringProperty(
        name="Pred graph",
        description="Predicted graph JSON (decoder output)",
        subtype="FILE_PATH", default="",
    )
    input_path: StringProperty(
        name="Input graph",
        description="Unfunctional input graph JSON (slot→node mapping + GT OBBs)",
        subtype="FILE_PATH", default="",
    )
    mesh_dir: StringProperty(
        name="Mesh dir",
        description="Directory containing parts.json/objs/ (PNM) or "
                    "<node_id>.ply (FurFun/HSSD)",
        subtype="DIR_PATH", default="",
    )
    drop_hallucinated: BoolProperty(
        name="Drop hallucinated",
        description="Drop fired free-slot nodes outside the completion "
                    "whitelist (handle / top panel / shelf / divider)",
        default=True,
    )

    # ── loaded state ──
    loaded: BoolProperty(default=False)
    state_json: StringProperty(default="", options={'HIDDEN'})
    model_label: StringProperty(default="")
    n_free_nodes: IntProperty(default=0)
    free_summary: StringProperty(default="")   # "2 handle, 1 shelf"
    load_warnings: IntProperty(default=0)

    # ── record rows ──
    hinges: CollectionProperty(type=GraphHingeItem)
    rails:  CollectionProperty(type=GraphRailItem)
    handle_groups: CollectionProperty(type=GraphHandleGroupItem)
    active_hinge_idx: IntProperty(default=0, update=_select_active_hinge)
    active_rail_idx:  IntProperty(default=0, update=_select_active_rail)
    active_hgroup_idx: IntProperty(default=0, update=_select_active_handle)

    # ── defaults for Apply All ──
    default_hinge_cat: EnumProperty(
        name="Default hinge",
        items=[
            ("auto",     "Auto (collision-tested)", ""),
            ("exterior", "Exterior", ""),
            ("interior", "Interior", ""),
            ("flat",     "Flat",     ""),
        ],
        default="auto",
    )
    default_hinge_var: EnumProperty(
        name="Variant",
        items=_items_default_for_default_cat,
    )
    default_rail_var: EnumProperty(
        name="Default rail",
        items=_items_rail,
    )
    default_hinge_count: IntProperty(
        name="Hinges/door", default=2, min=1, max=4,
    )
    add_support_blocks: BoolProperty(
        name="Add support blocks",
        description="Install drawer support slabs after rails (recommended)",
        default=True,
    )
    enable_divider: BoolProperty(
        name="Allow divider fallback",
        description="If a rail has no nearby wall for a support slab, "
                    "inscribe a divider",
        default=True,
    )

    # ── interaction toggles ──
    auto_apply_on_load: BoolProperty(
        name="Auto-functionalize on load",
        description="Run the full automatic pass (hinges, rails, supports, "
                    "fired handles/top/interior) right after loading",
        default=True,
    )
    live_update: BoolProperty(
        name="Live update",
        description="Re-apply the active joint immediately when its type/"
                    "variant/scale changes (off = press Re-apply manually)",
        default=True,
    )
    show_overlay: BoolProperty(
        name="Motion overlay",
        description="Draw predicted motion axes and hinge borders in the "
                    "viewport",
        default=False,
        update=_toggle_overlay,
    )

    # ── tops ──
    top_shape_style: EnumProperty(
        name="Shape",
        description="Footprint style of the synthesized top panel",
        items=[("RECTANGLE", "Rectangle", ""),
               ("ROUNDED",   "Rounded",   "Rounded corners"),
               ("CHAMFERED", "Chamfered", "Cut corners"),
               ("OVAL",      "Oval",      ""),
               ("CAPSULE",   "Capsule",   "Semicircular ends")],
        default="RECTANGLE",
        update=_live_retop,
    )
    top_thickness: FloatProperty(
        name="Thickness", default=0.020, min=0.004, max=0.12,
        subtype='DISTANCE', update=_live_retop,
    )
    top_overhang: FloatProperty(
        name="Overhang",
        description="Footprint enlargement as a fraction of its size",
        default=0.03, min=0.0, max=0.2, subtype='FACTOR',
        update=_live_retop,
    )
    top_corner_radius: FloatProperty(
        name="Corner radius",
        description="Corner radius as a fraction of the footprint's short "
                    "side (rounded/chamfered/oval/capsule)",
        default=0.15, min=0.02, max=0.45, subtype='FACTOR',
        update=_live_retop,
    )

    # ── interior ──
    interior_up_axis: EnumProperty(
        name="Up", items=[("+Z", "+Z", ""), ("-Z", "-Z", ""),
                          ("+Y", "+Y", ""), ("-Y", "-Y", ""),
                          ("+X", "+X", ""), ("-X", "-X", "")],
        default="+Z",
    )
    interior_front_axis: EnumProperty(
        name="Front", items=[("-Y", "-Y", ""), ("+Y", "+Y", ""),
                             ("-X", "-X", ""), ("+X", "+X", ""),
                             ("-Z", "-Z", ""), ("+Z", "+Z", "")],
        default="-Y",
    )
    interior_orientation: EnumProperty(
        name="Orientation",
        items=[("horizontal", "Shelves (horizontal)", ""),
               ("vertical",   "Dividers (vertical)", "")],
        default="horizontal",
    )
    interior_num_panels: IntProperty(
        name="Panels", default=2, min=1, max=12)
    interior_panel_thickness: FloatProperty(
        name="Thickness", default=0.018, min=0.002, max=0.2, subtype='DISTANCE')
    interior_front_inset: FloatProperty(
        name="Front inset", default=0.01, min=0.0, max=0.5, subtype='DISTANCE')
    interior_detected_count: IntProperty(default=0)

    # ── auto-functionalize progress ──
    auto_running: BoolProperty(default=False)
    auto_progress: StringProperty(default="")

    # ── status log ──
    log: CollectionProperty(type=GraphLogEntry)
    active_log_idx: IntProperty(default=0)
    status_message: StringProperty(default="")


# ──────────────────────────────────────────────────────── UI list classes ──

def _status_icon(status: str, default_icon: str) -> str:
    if status.startswith("ok"):
        return "CHECKMARK"
    if status.startswith("warn"):
        return "ERROR"
    if status.startswith("err"):
        return "CANCEL"
    return default_icon


class GRAPH_UL_hinges(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text="", icon=_status_icon(item.status, "CON_PIVOT"))
        row.label(text=item.joint_name)
        row.label(text=f"→ {item.panel_node}")


class GRAPH_UL_rails(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text="", icon=_status_icon(item.status, "CON_TRACKTO"))
        row.label(text=item.joint_name)


class GRAPH_UL_handle_groups(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text="", icon=_status_icon(item.status, "LINKED"))
        row.label(text=f"{item.parent_id} ({item.parent_kind})")
        tag = ("recessed" if item.mode == "RECESSED"
               else f"×{item.count}")
        pred = (f"{item.n_new} fired" if item.n_new
                else f"{item.n_anchored} anch" if item.n_anchored else "—")
        row.label(text=f"{pred} · {tag}")


class GRAPH_UL_log(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        row = layout.row(align=True)
        icon = {"INFO": "INFO", "WARN": "ERROR", "ERR": "CANCEL"}.get(
            item.level, "INFO")
        row.label(text=item.text, icon=icon)


_classes = (
    GraphHingeItem,
    GraphRailItem,
    GraphHandleGroupItem,
    GraphLogEntry,
    GraphFunctionalizationProperties,
    GRAPH_UL_hinges,
    GRAPH_UL_rails,
    GRAPH_UL_handle_groups,
    GRAPH_UL_log,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.graph_func_props = PointerProperty(
        type=GraphFunctionalizationProperties)


def unregister():
    if hasattr(bpy.types.Scene, "graph_func_props"):
        del bpy.types.Scene.graph_func_props
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
