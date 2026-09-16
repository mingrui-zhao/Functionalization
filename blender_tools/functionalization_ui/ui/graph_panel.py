"""GraFu Functionalization panels (View3D > Sidebar > Functionalize).

Panel stack (DESIGN.md §II.2):
  1 · Model      — prediction browser + auto-derived paths + Load
  2 · Review     — fired-node summary, overlay toggle, Auto-Functionalize,
                   motion preview controls
  3 · Hinges     — UIList + per-joint type/variant/scale/border controls
  4 · Rails      — UIList + per-joint variant + supports
  5 · Handles    — per-PARENT groups (multi-handle partitions)
  6 · Tops       — detect coverage + one-click synthesise
  7 · Interior   — instantiate-from-prediction + manual detect/generate
  8 · Finalize   — validate, commit & clean, export manifest
  ▸ Status log   — scrollable history (full messages, copyable via console)
"""
from __future__ import annotations

import bpy
from bpy.types import Panel


def _props(context):
    return context.scene.graph_func_props


class GRAPH_PT_main(Panel):
    bl_idname = "GRAPH_PT_main"
    bl_label = "GraFu Functionalization"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"

    def draw_header(self, context):
        self.layout.label(icon="GRAPH")

    def draw(self, context):
        props = _props(context)
        col = self.layout.column()
        col.scale_y = 0.8
        if props.loaded:
            col.label(text=f"Model: {props.model_label}", icon="OBJECT_DATA")
        else:
            col.label(text="Load a prediction to begin", icon="INFO")


# ───────────────────────────────────────────────────────────── 1 · Model ──

class GRAPH_PT_model(Panel):
    bl_idname = "GRAPH_PT_model"
    bl_label = "1 · Model"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"
    bl_parent_id = "GRAPH_PT_main"

    def draw_header(self, context):
        self.layout.label(icon="FILE_FOLDER")

    def draw(self, context):
        layout = self.layout
        props = _props(context)

        browser = layout.box()
        row = browser.row(align=True)
        row.prop(props, "pred_source", text="")
        row.operator("graph_func.refresh_sources", text="", icon="FILE_REFRESH")
        if props.pred_source == "CUSTOM":
            browser.prop(props, "custom_pred_dir", text="Dir")
        browser.prop(props, "model_enum", text="Model")
        browser.prop(props, "auto_apply_on_load")
        row = browser.row(align=True)
        row.scale_y = 1.5
        row.operator("graph_func.load_model", icon="IMPORT")
        row.operator("graph_func.clear_scene", text="", icon="TRASH")
        # Opened one of infer.py's blenderized results directly? Adopt it
        # for editing without rebuilding the scene.
        browser.operator("graph_func.attach_result", icon="LINKED")

        manual = layout.box()
        manual.prop(props, "use_manual_paths", toggle=True,
                    icon="FILEBROWSER")
        if props.use_manual_paths:
            manual.prop(props, "pred_path", text="Pred")
            manual.prop(props, "input_path", text="Input")
            manual.prop(props, "mesh_dir", text="Meshes")
            manual.prop(props, "repo_root", text="Repo")
            manual.prop(props, "drop_hallucinated")
            row = manual.row(align=True)
            row.scale_y = 1.3
            row.operator("graph_func.load_graph", text="Load Manual",
                         icon="IMPORT")

        if props.loaded:
            info = layout.box()
            col = info.column()
            col.scale_y = 0.85
            col.label(text=f"{props.model_label}: "
                           f"{len(props.hinges)} hinges · "
                           f"{len(props.rails)} rails · "
                           f"{len(props.handle_groups)} handle parents",
                      icon="CHECKMARK")
            if props.n_free_nodes:
                col.label(text=f"fired: {props.free_summary}", icon="PLUS")
            if props.load_warnings:
                w = col.row()
                w.alert = True
                w.label(text=f"{props.load_warnings} load warning(s) — "
                             f"see Status log", icon="ERROR")


# ──────────────────────────────────────────────────────────── 2 · Review ──

class GRAPH_PT_review(Panel):
    bl_idname = "GRAPH_PT_review"
    bl_label = "2 · Review"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"
    bl_parent_id = "GRAPH_PT_main"

    @classmethod
    def poll(cls, context):
        return _props(context).loaded

    def draw_header(self, context):
        self.layout.label(icon="VIEWZOOM")

    def draw(self, context):
        layout = self.layout
        props = _props(context)

        row = layout.row(align=True)
        row.prop(props, "show_overlay", toggle=True, icon="ORIENTATION_GIMBAL")
        row.prop(props, "live_update", toggle=True, icon="FILE_REFRESH")

        auto = layout.row()
        auto.scale_y = 1.7
        auto.enabled = not props.auto_running
        auto.operator("graph_func.auto_functionalize", icon="PLAY")
        if props.auto_running and props.auto_progress:
            prog = layout.row()
            prog.label(text=props.auto_progress, icon="TIME")
            prog.label(text="(ESC to cancel)")

        # Triage summary from row statuses.
        n_ok = sum(1 for c in (props.hinges, props.rails, props.handle_groups)
                   for it in c if it.status.startswith("ok"))
        n_warn = sum(1 for c in (props.hinges, props.rails, props.handle_groups)
                     for it in c if it.status.startswith("warn"))
        n_err = sum(1 for c in (props.hinges, props.rails, props.handle_groups)
                    for it in c if it.status.startswith("err"))
        stat = layout.row(align=True)
        stat.label(text=f"{n_ok} ok", icon="CHECKMARK")
        stat.label(text=f"{n_warn} warn", icon="ERROR")
        stat.label(text=f"{n_err} err", icon="CANCEL")

        mo = layout.row(align=True)
        mo.operator("graph_func.preview_motion", icon="PLAY")
        mo.operator("graph_func.jump_max_extent", icon="FRAME_NEXT")
        layout.operator("graph_func.pick_row_from_object",
                        icon="EYEDROPPER")
        tests = layout.row(align=True)
        tests.operator("graph_func.validate_motion", icon="MOD_PHYSICS")
        tests.operator("graph_func.test_connectivity", icon="SNAP_ON")


# ──────────────────────────────────────────────────────────── 3 · Hinges ──

class GRAPH_PT_hinges(Panel):
    bl_idname = "GRAPH_PT_hinges"
    bl_label = "3 · Hinges"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"
    bl_parent_id = "GRAPH_PT_main"

    @classmethod
    def poll(cls, context):
        return _props(context).loaded

    def draw_header(self, context):
        self.layout.label(icon="CON_PIVOT")

    def draw(self, context):
        layout = self.layout
        props = _props(context)

        defaults = layout.box()
        defaults.label(text="Defaults (Apply All)", icon="OPTIONS")
        defaults.prop(props, "default_hinge_cat", text="Type")
        if props.default_hinge_cat != "auto":
            defaults.prop(props, "default_hinge_var", text="Variant")
        defaults.prop(props, "default_hinge_count")

        row = layout.row()
        row.scale_y = 1.4
        row.enabled = len(props.hinges) > 0
        row.operator("graph_func.apply_all_hinges", icon="PLAY")

        if not props.hinges:
            layout.label(text="No predicted hinges.", icon="INFO")
            return

        layout.template_list("GRAPH_UL_hinges", "", props, "hinges",
                             props, "active_hinge_idx", rows=3)

        idx = max(0, min(props.active_hinge_idx, len(props.hinges) - 1))
        item = props.hinges[idx]

        det = layout.box()
        head = det.row(align=True)
        head.label(text=f"{item.joint_name}", icon="MOD_DATA_TRANSFER")
        sel = head.operator("graph_func.select_row_objects", text="",
                            icon="RESTRICT_SELECT_OFF")
        sel.row_kind = 'HINGE'
        solo = head.operator("graph_func.solo_joint", text="", icon="SOLO_ON")
        solo.row_kind = 'HINGE'
        det.label(text=f"door {item.door_node} → panel {item.panel_node}")

        det.prop(item, "category", text="Type")
        if item.category != "auto":
            var = det.row(align=True)
            op_prev = var.operator("graph_func.cycle_hinge_variant",
                                   text="", icon="TRIA_LEFT")
            op_prev.delta = -1
            var.prop(item, "variant", text="")
            op_next = var.operator("graph_func.cycle_hinge_variant",
                                   text="", icon="TRIA_RIGHT")
            op_next.delta = 1

        scol = det.column(align=True)
        scol.prop(item, "use_auto_scale")
        sub = scol.row()
        sub.enabled = not item.use_auto_scale
        sub.prop(item, "scale_factor", slider=True)
        det.prop(item, "hinge_count")

        adv = det.box()
        adv.label(text="Advanced", icon="PREFERENCES")
        adv.prop(item, "strict_border")
        adv.prop(item, "flip_side")
        adv.prop(item, "use_decompose")

        act = layout.row()
        act.scale_y = 1.3
        act.operator("graph_func.reapply_one_hinge", icon="FILE_REFRESH")

        if item.status:
            info = layout.row()
            info.alert = item.status.startswith("err")
            info.label(text=item.status, icon=(
                "CANCEL" if item.status.startswith("err") else
                "ERROR" if item.status.startswith("warn") else "CHECKMARK"))


# ───────────────────────────────────────────────────────────── 4 · Rails ──

class GRAPH_PT_rails(Panel):
    bl_idname = "GRAPH_PT_rails"
    bl_label = "4 · Rails"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"
    bl_parent_id = "GRAPH_PT_main"

    @classmethod
    def poll(cls, context):
        return _props(context).loaded

    def draw_header(self, context):
        self.layout.label(icon="CON_TRACKTO")

    def draw(self, context):
        layout = self.layout
        props = _props(context)

        defaults = layout.box()
        defaults.prop(props, "default_rail_var", text="Variant")
        defaults.prop(props, "add_support_blocks")
        sub = defaults.row()
        sub.enabled = props.add_support_blocks
        sub.prop(props, "enable_divider")

        row = layout.row(align=True)
        row.scale_y = 1.4
        row.enabled = len(props.rails) > 0
        row.operator("graph_func.apply_all_rails", icon="PLAY")
        row.operator("graph_func.add_support_blocks", text="Supports",
                     icon="MESH_GRID")

        if not props.rails:
            layout.label(text="No predicted rails.", icon="INFO")
            return

        layout.template_list("GRAPH_UL_rails", "", props, "rails",
                             props, "active_rail_idx", rows=3)

        idx = max(0, min(props.active_rail_idx, len(props.rails) - 1))
        item = props.rails[idx]

        det = layout.box()
        head = det.row(align=True)
        head.label(text=f"{item.joint_name}", icon="MOD_DATA_TRANSFER")
        sel = head.operator("graph_func.select_row_objects", text="",
                            icon="RESTRICT_SELECT_OFF")
        sel.row_kind = 'RAIL'
        solo = head.operator("graph_func.solo_joint", text="", icon="SOLO_ON")
        solo.row_kind = 'RAIL'
        var = det.row(align=True)
        op_prev = var.operator("graph_func.cycle_rail_variant",
                               text="", icon="TRIA_LEFT")
        op_prev.delta = -1
        var.prop(item, "variant", text="")
        op_next = var.operator("graph_func.cycle_rail_variant",
                               text="", icon="TRIA_RIGHT")
        op_next.delta = 1

        act = layout.row()
        act.scale_y = 1.3
        act.operator("graph_func.reapply_one_rail", icon="FILE_REFRESH")

        if item.status:
            info = layout.row()
            info.alert = item.status.startswith("err")
            info.label(text=item.status, icon=(
                "CANCEL" if item.status.startswith("err") else
                "ERROR" if item.status.startswith("warn") else "CHECKMARK"))


# ─────────────────────────────────────────────────────────── 5 · Handles ──

class GRAPH_PT_handles(Panel):
    bl_idname = "GRAPH_PT_handles"
    bl_label = "5 · Handles"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"
    bl_parent_id = "GRAPH_PT_main"

    @classmethod
    def poll(cls, context):
        return _props(context).loaded

    def draw_header(self, context):
        self.layout.label(icon="DRIVER_DISTANCE")

    def draw(self, context):
        layout = self.layout
        props = _props(context)

        if not props.handle_groups:
            layout.label(text="No doors/drawers with motion records.",
                         icon="INFO")
            return

        row = layout.row()
        row.scale_y = 1.3
        row.operator("graph_func.add_all_predicted_handles",
                     text="Apply All Handles", icon="SELECT_EXTEND")

        layout.template_list("GRAPH_UL_handle_groups", "", props,
                             "handle_groups", props, "active_hgroup_idx",
                             rows=3)

        idx = max(0, min(props.active_hgroup_idx,
                         len(props.handle_groups) - 1))
        it = props.handle_groups[idx]

        det = layout.box()
        head = det.row(align=True)
        head.label(text=f"{it.parent_id} ({it.parent_kind})",
                   icon="MOD_DATA_TRANSFER")
        sel = head.operator("graph_func.select_row_objects", text="",
                            icon="RESTRICT_SELECT_OFF")
        sel.row_kind = 'HANDLE'
        if it.n_new or it.n_anchored:
            det.label(text=f"predicted: {it.n_new} fired + "
                           f"{it.n_anchored} anchored")
        else:
            det.label(text="no handle predicted — add one manually",
                      icon="INFO")

        det.row().prop(it, "mode", expand=True)

        if it.mode == "ADDITIVE":
            style = det.row(align=True)
            op_prev = style.operator("graph_func.cycle_handle_style",
                                     text="", icon="TRIA_LEFT")
            op_prev.delta = -1
            style.prop(it, "style", text="")
            op_next = style.operator("graph_func.cycle_handle_style",
                                     text="", icon="TRIA_RIGHT")
            op_next.delta = 1
            det.prop(it, "count")
            place = det.column(align=True)
            place.prop(it, "margin_edge", slider=True)
            pos = place.row()
            pos.enabled = it.count <= 1
            pos.prop(it, "pos_frac", slider=True)
            if it.count > 1:
                det.label(text=f"{it.count} × {it.style.split('_')[-1]} at "
                               f"equal partitions, auto-shrunk to fit",
                          icon="INFO")
        else:
            col = det.column(align=True)
            col.prop(it, "sub_shape", text="Shape")
            col.prop(it, "sub_depth", slider=True)
            col.prop(it, "sub_method", text="Method")
            det.label(text="Carves the panel itself (undoable via Remove)",
                      icon="INFO")

        if it.n_anchored:
            det.prop(it, "replace_anchored")

        act = layout.row(align=True)
        act.scale_y = 1.3
        act.operator("graph_func.add_group_handles", text="Apply",
                     icon="CHECKMARK")
        act.operator("graph_func.remove_group_handles", text="Remove",
                     icon="TRASH")

        if it.status:
            info = layout.row()
            info.alert = it.status.startswith("err")
            info.label(text=it.status, icon=(
                "CANCEL" if it.status.startswith("err") else
                "ERROR" if it.status.startswith("warn") else "CHECKMARK"))


# ────────────────────────────────────────────────────────────── 6 · Tops ──

class GRAPH_PT_tops(Panel):
    bl_idname = "GRAPH_PT_tops"
    bl_label = "6 · Tops"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"
    bl_parent_id = "GRAPH_PT_main"

    @classmethod
    def poll(cls, context):
        return _props(context).loaded

    def draw_header(self, context):
        self.layout.label(icon="TRIA_UP_BAR")

    def draw(self, context):
        layout = self.layout
        props = _props(context)
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("graph_func.detect_top", icon="VIEWZOOM")
        row.operator("graph_func.add_predicted_top", icon="MESH_PLANE")
        style = layout.box()
        col = style.column(align=True)
        col.prop(props, "top_shape_style", text="Shape")
        col.prop(props, "top_thickness")
        col.prop(props, "top_overhang", slider=True)
        if props.top_shape_style != "RECTANGLE":
            col.prop(props, "top_corner_radius", slider=True)
        hint = layout.column()
        hint.scale_y = 0.75
        hint.label(text="Synthesised above the body silhouette")
        hint.label(text="(doors/drawers/handles excluded).")


# ─────────────────────────────────────────────────────────── 7 · Interior ─

class GRAPH_PT_interior(Panel):
    bl_idname = "GRAPH_PT_interior"
    bl_label = "7 · Interior"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"
    bl_parent_id = "GRAPH_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _props(context).loaded

    def draw_header(self, context):
        self.layout.label(icon="MESH_CUBE")

    def draw(self, context):
        layout = self.layout
        props = _props(context)

        auto = layout.row()
        auto.scale_y = 1.4
        auto.operator("graph_func.instantiate_interior", icon="SHADERFX")
        col = layout.column()
        col.scale_y = 0.75
        col.label(text="One panel per fired shelf/divider node,")
        col.label(text="connectivity-placed. Idempotent.")

        manual = layout.box()
        manual.label(text="Manual", icon="TOOL_SETTINGS")
        ax = manual.row(align=True)
        ax.prop(props, "interior_up_axis", text="Up")
        ax.prop(props, "interior_front_axis", text="Front")
        row = manual.row()
        row.scale_y = 1.2
        row.operator("graph_func.detect_interior", icon="VIEWZOOM")
        if props.interior_detected_count > 0:
            manual.label(text=f"✓ {props.interior_detected_count} "
                              f"compartment(s)", icon="CHECKMARK")
            manual.operator("graph_func.clear_compartment_boxes",
                            icon="X")
            gen = manual.box()
            col = gen.column(align=True)
            col.prop(props, "interior_orientation")
            col.prop(props, "interior_num_panels")
            col.prop(props, "interior_panel_thickness")
            col.prop(props, "interior_front_inset")
            row = gen.row(align=True)
            row.scale_y = 1.2
            op_sel = row.operator("graph_func.generate_interior_panels",
                                  text="Selected", icon="RESTRICT_SELECT_OFF")
            op_sel.use_all = False
            op_all = row.operator("graph_func.generate_interior_panels",
                                  text="All", icon="SELECT_EXTEND")
            op_all.use_all = True
            act = gen.row(align=True)
            act.operator("graph_func.clear_interior_panels",
                         text="Clear", icon="TRASH")
            act.operator("graph_func.commit_interior",
                         text="Commit", icon="CHECKMARK")


# ─────────────────────────────────────────────────────────── 8 · Finalize ─

class GRAPH_PT_finalize(Panel):
    bl_idname = "GRAPH_PT_finalize"
    bl_label = "8 · Finalize"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"
    bl_parent_id = "GRAPH_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _props(context).loaded

    def draw_header(self, context):
        self.layout.label(icon="CHECKMARK")

    def draw(self, context):
        layout = self.layout
        tests = layout.row(align=True)
        tests.operator("graph_func.validate_motion", icon="MOD_PHYSICS")
        tests.operator("graph_func.test_connectivity", icon="SNAP_ON")
        layout.operator("graph_func.commit_finalize", icon="CHECKMARK")
        layout.operator("graph_func.export_manifest", icon="FILE_TICK")
        layout.operator("graph_func.rebuild_records", icon="RECOVER_LAST")


# ─────────────────────────────────────────────────────────── Status log ────

class GRAPH_PT_status(Panel):
    bl_idname = "GRAPH_PT_status"
    bl_label = "Status Log"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Functionalize"
    bl_parent_id = "GRAPH_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.label(icon="INFO")

    def draw(self, context):
        layout = self.layout
        props = _props(context)
        if not len(props.log):
            layout.label(text="(no recent activity)", icon="INFO")
            return
        layout.template_list("GRAPH_UL_log", "", props, "log",
                             props, "active_log_idx", rows=6)
        layout.operator("graph_func.clear_log", icon="TRASH")


_classes = (
    GRAPH_PT_main,
    GRAPH_PT_model,
    GRAPH_PT_review,
    GRAPH_PT_hinges,
    GRAPH_PT_rails,
    GRAPH_PT_handles,
    GRAPH_PT_tops,
    GRAPH_PT_interior,
    GRAPH_PT_finalize,
    GRAPH_PT_status,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
