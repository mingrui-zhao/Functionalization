"""Operators for the graph-driven functionalization workflow.

Every operator reads/writes `scene.graph_func_props`. Loaded-graph records
are cached in a sidecar JSON next to the .blend (path in `props.state_json`)
and can always be rebuilt from the pred/input paths + scene objects via
`graph_func.rebuild_records` (undo / reload recovery).

Feedback contract: every operator logs through `_log()` — which appends to
the scrollable status log, mirrors to `self.report` (status bar + Info
editor) and the console. No silent failures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import bpy
from bpy.types import Operator
import numpy as np

from ..utils import predicted_graph as pg
from ..utils import install_helpers as ih
from ..utils import interior_scene as iscene
from ..utils import graph_properties as gp
from ..ui import overlay


# ─────────────────────────────────────────────────────────── helpers ───────

def _props(context):
    return context.scene.graph_func_props


_LOG_CAP = 80


def _log(props, level: str, msg: str, op: Optional[Operator] = None) -> None:
    """Central feedback: status log + status bar + console."""
    print(f"[graph_op:{level}] {msg}")
    props.status_message = ("✓ " if level == "INFO" else
                            "⚠ " if level == "WARN" else "✗ ") + msg
    entry = props.log.add()
    entry.level = level
    entry.text = msg
    while len(props.log) > _LOG_CAP:
        props.log.remove(0)
    props.active_log_idx = len(props.log) - 1
    if op is not None:
        op.report({'INFO' if level == "INFO" else
                   'WARNING' if level == "WARN" else 'ERROR'}, msg)


def _serialise_obj_map(pred_id_to_obj: dict) -> dict:
    return {nid: obj.name for nid, obj in pred_id_to_obj.items()
            if obj is not None and obj.name in bpy.data.objects}


def _deserialise_obj_map(name_map: dict) -> dict:
    out = {}
    for nid, name in (name_map or {}).items():
        o = bpy.data.objects.get(name)
        if o is not None:
            out[nid] = o
    return out


def _load_state(props) -> Optional[dict]:
    if not props.loaded or not props.state_json:
        return None
    p = Path(props.state_json)
    if not p.is_file():
        print(f"[graph_op] state file not found: {p}")
        return None
    try:
        s = json.loads(p.read_text())
    except Exception as exc:
        print(f"[graph_op] state file parse error: {exc}")
        return None
    return pg.records_from_scene_state(s)


def _raw_state(props) -> Optional[dict]:
    if not props.loaded or not props.state_json:
        return None
    p = Path(props.state_json)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_state(props, state: dict) -> None:
    import tempfile
    blend_path = bpy.data.filepath
    if blend_path:
        side = Path(blend_path).parent / (Path(blend_path).stem
                                          + ".graph_func_state.json")
    else:
        side = Path(tempfile.gettempdir()) / "graph_func_state_unsaved.json"
    side.write_text(json.dumps(state))
    props.state_json = str(side)
    props.loaded = True


def _state_with_objs(props):
    state = _load_state(props)
    if state is None:
        return None
    raw = _raw_state(props) or {}
    objs = _deserialise_obj_map(raw.get("pred_id_to_obj", {}))
    if not objs:
        # Undo / file-reload recovery: resolve by naming convention.
        for o in bpy.data.objects:
            if "__mesh" in o.name:
                objs[o.name.split("__mesh")[0]] = o
    state["pred_id_to_obj"] = objs
    return state


def _find_record(state: dict, key: str, joint_name: str) -> Optional[dict]:
    for rec in state.get(key, []):
        if rec.get("joint") == joint_name:
            return rec
    return None


# ── repo root + model browser ───────────────────────────────────────────────

def _resolve_repo_root(props) -> Path:
    if props.repo_root:
        p = Path(bpy.path.abspath(props.repo_root))
        if p.exists():
            return p
    # Registration already resolved the repo (bundled layout or the
    # repo_location.txt pointer of an installed copy) — reuse it.
    if ih._REPO_ROOT is not None:
        return ih._REPO_ROOT
    # bundled layout: <repo>/blender_tools/functionalization_ui/operators/
    cand = Path(__file__).resolve().parents[3]
    if (cand / "blender_tools" / "hinge").is_dir():
        return cand
    return Path(__file__).resolve().parents[2]


# Dataset conventions for auto-deriving input graph + mesh dir from a mid.
_DATASET_LAYOUTS = (
    ("datasets/pnm_clean/graphs_unfunctional/{mid}/{mid}.json",
     "datasets/pnm_clean/part_geometries/{mid}"),
    ("datasets/pnm_clean/graphs_unfunctional/{mid}/{mid}.json",
     "datasets/pnm_clean/part_geometries/{mid}"),
    ("datasets/hssd_test/graph_unfunc/{mid}/{mid}.json",
     "datasets/hssd_test/part_geometries/{mid}"),
    ("datasets/furfun/graphs_unfunctional/{mid}/{mid}.json",
     "datasets/furfun/part_geometries/{mid}"),
)


def _derive_paths(repo: Path, mid: str, pred_dir: Path | None = None):
    """Resolve (input graph, mesh dir) for `mid`, in priority order:
      1. inference_manifest.json next to the predictions (written by
         infer.py — the standard path for user data);
      2. a bundle folder next to the predictions: <pred_dir>/<mid>/ holding
         <mid>.json + the part meshes;
      3. the repo's dataset conventions (pnm_clean / hssd_test / furfun).
    """
    if pred_dir is not None:
        man = pred_dir / "inference_manifest.json"
        if man.is_file():
            try:
                entry = json.loads(man.read_text()).get(mid) or {}
            except Exception:
                entry = {}
            inp = Path(entry["input"]) if entry.get("input") else None
            mesh = Path(entry["mesh_dir"]) if entry.get("mesh_dir") else None
            if inp is not None and inp.is_file():
                if mesh is None or not mesh.is_dir():
                    mesh = inp.parent
                return inp, mesh
        bundle = pred_dir / mid
        for name in (f"{mid}.json", f"{mid}_input.json"):
            if (bundle / name).is_file():
                return bundle / name, bundle
    for inp_t, mesh_t in _DATASET_LAYOUTS:
        inp = repo / inp_t.format(mid=mid)
        mesh = repo / mesh_t.format(mid=mid)
        if inp.is_file() and mesh.is_dir():
            return inp, mesh
    return None, None


def _scan_pred_sources(repo: Path) -> list[tuple[str, str, str]]:
    items = []
    results = repo / "results"
    if results.is_dir():
        if any(results.glob("*_pred.json")):
            items.append((str(results), "results/", ""))
        for d in sorted(results.iterdir()):
            if d.is_dir() and any(d.glob("*_pred.json")):
                items.append((str(d), f"results/{d.name}", ""))
    items.append(("CUSTOM", "Custom directory…", "Use the Custom pred dir path"))
    return items


def _scan_models(pred_dir: Path) -> list[tuple[str, str, str]]:
    out = []
    for f in sorted(pred_dir.glob("*_pred.json")):
        mid = f.name[:-len("_pred.json")]
        out.append((mid, mid, ""))
    return out


class GRAPH_OT_refresh_sources(Operator):
    """Scan the repo for prediction sets and models"""
    bl_idname = "graph_func.refresh_sources"
    bl_label = "Scan Predictions"

    def execute(self, context):
        props = _props(context)
        repo = _resolve_repo_root(props)
        sources = _scan_pred_sources(repo)
        gp.set_pred_source_items(sources)
        n_dirs = len(sources) - 1
        src = props.pred_source
        pred_dir = (Path(bpy.path.abspath(props.custom_pred_dir))
                    if src == "CUSTOM" and props.custom_pred_dir
                    else Path(src) if src not in ("", "NONE", "CUSTOM") else None)
        if pred_dir is not None and pred_dir.is_dir():
            models = _scan_models(pred_dir)
            gp.set_model_items(models)
            _log(props, "INFO",
                 f"{n_dirs} prediction set(s); {len(models)} model(s) in "
                 f"{pred_dir.name}", self)
        else:
            gp.set_model_items([])
            _log(props, "INFO", f"{n_dirs} prediction set(s) found "
                 f"(pick one, then Scan again to list models)", self)
        return {"FINISHED"}


class GRAPH_OT_load_model(Operator):
    """Load the selected model: derive input/mesh paths and import"""
    bl_idname = "graph_func.load_model"
    bl_label = "Load Model"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        repo = _resolve_repo_root(props)
        src = props.pred_source
        if src in ("", "NONE"):
            _log(props, "ERR", "no prediction set selected (Scan first)", self)
            return {"CANCELLED"}
        pred_dir = (Path(bpy.path.abspath(props.custom_pred_dir))
                    if src == "CUSTOM" else Path(src))
        mid = props.model_enum
        if mid in ("", "NONE"):
            _log(props, "ERR", "no model selected", self)
            return {"CANCELLED"}
        pred = pred_dir / f"{mid}_pred.json"
        inp, mesh = _derive_paths(repo, mid, pred_dir=pred_dir)
        if not pred.is_file():
            _log(props, "ERR", f"pred not found: {pred}", self)
            return {"CANCELLED"}
        if inp is None:
            _log(props, "ERR",
                 f"no input graph found for '{mid}': no inference manifest "
                 f"or bundle next to the predictions, and no dataset match "
                 f"under {repo}/datasets. Use Manual paths.", self)
            return {"CANCELLED"}
        props.pred_path = str(pred)
        props.input_path = str(inp)
        props.mesh_dir = str(mesh)
        return bpy.ops.graph_func.load_graph()


# ─────────────────────────────────────────────────────── load operator ────

def _populate_joint_rows(props, hinge_records, rail_records, attached_pairs,
                         pred_nodes: dict, n_input: int) -> int:
    """Fill the hinge/rail/handle-group rows from joint records. Shared by
    Load Graph and Attach Opened Result. Returns the number of fired
    handles dropped because their parent already has an anchored handle."""
    with gp.suspend_updates():
        props.hinges.clear()
        for rec in hinge_records:
            item = props.hinges.add()
            item.joint_name = rec["joint"]
            item.door_node = rec["door_node"]
            item.panel_node = rec["panel_node"]
            item.category = props.default_hinge_cat
            try:
                item.variant = props.default_hinge_var
            except (TypeError, ValueError):
                pass
            item.use_auto_scale = True
            item.scale_factor = 1.0
            item.hinge_count = int(props.default_hinge_count)
            item.status = ""

        props.rails.clear()
        for rec in rail_records:
            item = props.rails.add()
            item.joint_name = rec["joint"]
            item.drawer_node = rec["drawer_node"]
            item.panel_node = rec["panel_node"]
            try:
                item.variant = props.default_rail_var
            except (TypeError, ValueError):
                pass
            item.status = ""

        # Handle GROUPS: one row per movable parent (DESIGN.md §I.1) so
        # the multi-handle partition rule installs/replaces atomically.
        # EVERY door/drawer with a motion record gets a row — parents
        # without a fired handle can still be equipped manually.
        door_set = {r["door_node"] for r in hinge_records}
        draw_set = {r["drawer_node"] for r in rail_records}
        groups: dict[str, dict] = {
            p_id: {"new": 0, "anchored": 0}
            for p_id in list(door_set) + list(draw_set)
        }
        for h_id, p_id in attached_pairs:
            g = groups.setdefault(p_id, {"new": 0, "anchored": 0})
            slot = int(pred_nodes.get(h_id, {}).get("slot", -1))
            if 0 <= slot < n_input:
                g["anchored"] += 1
            else:
                g["new"] += 1
        props.handle_groups.clear()
        n_dropped_new = 0
        for p_id, g in groups.items():
            it = props.handle_groups.add()
            it.parent_id = p_id
            it.parent_kind = ("door" if p_id in door_set else
                              "drawer" if p_id in draw_set else "unknown")
            # A parent with an anchored handle keeps it: fired handles
            # on the same parent are dropped (mirrors postprocess).
            if g["anchored"] > 0 and g["new"] > 0:
                n_dropped_new += g["new"]
                g["new"] = 0
            it.n_new = g["new"]
            it.n_anchored = g["anchored"]
            it.count = g["new"] or g["anchored"] or 1
            it.replace_anchored = (g["new"] == 0 and g["anchored"] > 0)
            it.status = ""
    return n_dropped_new


class GRAPH_OT_load_graph(Operator):
    """Load a predicted graph + its input + meshes into the scene"""
    bl_idname = "graph_func.load_graph"
    bl_label = "Load Graph"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        pred_path = Path(bpy.path.abspath(props.pred_path or ""))
        input_path = Path(bpy.path.abspath(props.input_path or ""))
        mesh_dir = Path(bpy.path.abspath(props.mesh_dir or ""))
        for p, what in ((pred_path, "pred"), (input_path, "input")):
            if not p.is_file():
                _log(props, "ERR", f"{what} not found: {p}", self)
                return {"CANCELLED"}
        if not mesh_dir.is_dir():
            _log(props, "ERR", f"mesh_dir not found: {mesh_dir}", self)
            return {"CANCELLED"}

        ih.init_paths(_resolve_repo_root(props))

        try:
            loaded = pg.load_graph_into_scene(
                pred_path, input_path, mesh_dir,
                drop_hallucinated=bool(props.drop_hallucinated),
            )
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(props, "ERR", f"load failed: {exc}", self)
            return {"CANCELLED"}

        state = pg.records_to_scene_state(loaded)
        state["pred_id_to_obj"] = _serialise_obj_map(loaded.pred_id_to_obj)
        _save_state(props, state)

        n_dropped_new = _populate_joint_rows(
            props, loaded.hinge_records, loaded.rail_records,
            loaded.attached_pairs, loaded.pred["nodes"],
            len(loaded.input_ids))

        label = loaded.pred.get("model_id") or input_path.stem
        props.model_label = str(label)
        props.n_free_nodes = len(loaded.free_nodes)
        by_mat: dict[str, int] = {}
        for fn in loaded.free_nodes:
            m = pg.mat_base(fn.get("material", "?"))
            by_mat[m] = by_mat.get(m, 0) + 1
        props.free_summary = ", ".join(f"{v} {k}" for k, v in sorted(by_mat.items()))
        props.load_warnings = len(loaded.warnings)

        if props.show_overlay:
            overlay.rebuild_from_props(props)

        _log(props, "INFO",
             f"loaded {label}: {len(loaded.hinge_records)} hinges, "
             f"{len(loaded.rail_records)} rails, "
             f"{len(props.handle_groups)} handle parent(s), "
             f"{len(loaded.free_nodes)} fired node(s)", self)
        if n_dropped_new:
            _log(props, "INFO",
                 f"dropped {n_dropped_new} fired handle(s) on parents that "
                 f"already have anchored handles")
        for w in loaded.warnings:
            _log(props, "WARN", w)
        if props.auto_apply_on_load:
            # Load = functionalize: chain the automatic pass (modal with
            # progress in the UI, synchronous in background mode).
            bpy.ops.graph_func.auto_functionalize('INVOKE_DEFAULT')
        return {"FINISHED"}


class GRAPH_OT_rebuild_records(Operator):
    """Rebuild joint records from the pred/input paths and current scene
    (recovery after undo or file reload)"""
    bl_idname = "graph_func.rebuild_records"
    bl_label = "Rebuild From Scene"

    def execute(self, context):
        props = _props(context)
        pred_path = Path(bpy.path.abspath(props.pred_path or ""))
        input_path = Path(bpy.path.abspath(props.input_path or ""))
        if not pred_path.is_file() or not input_path.is_file():
            _log(props, "ERR", "pred/input paths not set — cannot rebuild", self)
            return {"CANCELLED"}
        pred = json.loads(pred_path.read_text())
        inp = json.loads(input_path.read_text())
        hinges, rails, body_obbs, panel_idx, static_ids = pg.build_records(pred, inp)
        state = {
            "pred_path": str(pred_path), "input_path": str(input_path),
            "mesh_dir": props.mesh_dir,
            "input_ids": list(inp["nodes"].keys()),
            "hinges": hinges, "rails": rails,
            "body_obbs": [{"center": np.asarray(o.center).tolist(),
                           "size": np.asarray(o.size).tolist()}
                          for o in body_obbs],
            "panel_id_to_idx": panel_idx, "static_ids": static_ids,
            "attached_pairs": [], "free_nodes": [], "warnings": [],
        }
        for e in pred.get("edges", []):
            if e.get("kind") != "attached":
                continue
            s, d = e["src"], e["dst"]
            sm = pg.mat_base(pred["nodes"].get(s, {}).get("material", ""))
            if sm == "handle":
                state["attached_pairs"].append([s, d])
            else:
                state["attached_pairs"].append([d, s])
        state["pred_id_to_obj"] = {
            o.name.split("__mesh")[0]: o.name
            for o in bpy.data.objects if "__mesh" in o.name}
        _save_state(props, state)
        _log(props, "INFO",
             f"records rebuilt: {len(hinges)} hinges, {len(rails)} rails", self)
        return {"FINISHED"}


class GRAPH_OT_attach_result(Operator):
    """Attach the opened inference .blend for editing: derive the pred /
    input / mesh paths from the run's inference_manifest.json, rebuild the
    joint rows, and adopt the already-installed joints without reloading
    or reinstalling anything"""
    bl_idname = "graph_func.attach_result"
    bl_label = "Attach Opened Result"

    def execute(self, context):
        props = _props(context)
        blend = Path(bpy.data.filepath) if bpy.data.filepath else None
        if blend is None:
            _log(props, "ERR", "no .blend open — open a file from an "
                 "inference run's blends/ folder first", self)
            return {"CANCELLED"}
        mid = blend.stem
        run_dir = blend.parent.parent if blend.parent.name == "blends" \
            else blend.parent
        pred_path = run_dir / f"{mid}_pred.json"
        if not pred_path.is_file():
            _log(props, "ERR", f"no prediction next to this blend: "
                 f"{pred_path}", self)
            return {"CANCELLED"}
        input_path = mesh_dir = None
        manifest = run_dir / "inference_manifest.json"
        if manifest.is_file():
            try:
                entry = json.loads(manifest.read_text()).get(mid) or {}
                input_path = entry.get("input")
                mesh_dir = entry.get("mesh_dir")
            except Exception as exc:
                print(f"[attach] manifest parse error: {exc}")
        if not input_path or not Path(input_path).is_file() \
                or not mesh_dir or not Path(mesh_dir).is_dir():
            _log(props, "ERR", "inference_manifest.json missing or its "
                 "input/mesh paths do not exist — set the Manual paths and "
                 "use Rebuild From Scene instead", self)
            return {"CANCELLED"}
        props.pred_path = str(pred_path)
        props.input_path = str(input_path)
        props.mesh_dir = str(mesh_dir)
        ih.init_paths(_resolve_repo_root(props))

        pred = json.loads(pred_path.read_text())
        inp = json.loads(Path(input_path).read_text())
        hinges, rails, body_obbs, panel_idx, static_ids = \
            pg.build_records(pred, inp)
        state = {
            "pred_path": str(pred_path), "input_path": str(input_path),
            "mesh_dir": str(mesh_dir),
            "input_ids": list(inp["nodes"].keys()),
            "hinges": hinges, "rails": rails,
            "body_obbs": [{"center": np.asarray(o.center).tolist(),
                           "size": np.asarray(o.size).tolist()}
                          for o in body_obbs],
            "panel_id_to_idx": panel_idx, "static_ids": static_ids,
            "attached_pairs": [], "free_nodes": [], "warnings": [],
        }
        for e in pred.get("edges", []):
            if e.get("kind") != "attached":
                continue
            s, d = e["src"], e["dst"]
            sm = pg.mat_base(pred["nodes"].get(s, {}).get("material", ""))
            if sm == "handle":
                state["attached_pairs"].append([s, d])
            else:
                state["attached_pairs"].append([d, s])
        state["pred_id_to_obj"] = {
            o.name.split("__mesh")[0]: o.name
            for o in bpy.data.objects if "__mesh" in o.name}
        _save_state(props, state)

        _populate_joint_rows(props, hinges, rails, state["attached_pairs"],
                             pred["nodes"], len(inp["nodes"]))

        # Adopt the batch installs: mark every joint that already has
        # hardware in the scene, and read the hinge class the batch FPC
        # competition picked so the UI reflects reality.
        names = [o.name for o in bpy.data.objects]
        n_adopted = 0
        with gp.suspend_updates():
            for item in props.hinges:
                tag = f"{item.joint_name}_cand_"
                hit = next((n for n in names if tag in n), None)
                if hit is None and any(
                        item.joint_name in n and n.startswith("hinge_master")
                        for n in names):
                    hit = ""
                if hit is None:
                    continue
                if hit:
                    tok = hit.split(tag, 1)[1].split("_", 1)[0]
                    if tok in ("exterior", "interior", "flat"):
                        try:
                            item.category = tok
                        except (TypeError, ValueError):
                            pass
                item.status = "ok (adopted batch install)"
                n_adopted += 1
            for item in props.rails:
                def _root_base(n):
                    b = n.split("__", 1)[0]
                    if "." in b and b.rsplit(".", 1)[-1].isdigit():
                        b = b.rsplit(".", 1)[0]
                    return b
                if any(item.joint_name in n
                       and _root_base(n) in ("rail", "rail_master")
                       for n in names):
                    item.status = "ok (adopted batch install)"
                    n_adopted += 1
        # Tag batch-installed template handles so group replace / re-add
        # stays idempotent (the addon wipes by this tag).
        objs = _deserialise_obj_map(state["pred_id_to_obj"])
        n_tagged = 0
        for it in props.handle_groups:
            p_obj = objs.get(it.parent_id)
            if p_obj is None:
                continue
            for c in p_obj.children_recursive:
                if "__mesh" in c.name or c.get("grafu_handle_parent") is not None:
                    continue
                # Template meshes AND their HANDLE_ROOT empty: wiping by tag
                # must take the root along or re-apply leaves orphan empties.
                if c.type == "MESH" or (c.type == "EMPTY"
                                        and c.name.startswith("HANDLE_ROOT")):
                    c["grafu_handle_parent"] = it.parent_id
                    n_tagged += 1

        props.model_label = mid
        props.n_free_nodes = 0
        props.free_summary = ""
        props.load_warnings = 0
        if props.show_overlay:
            overlay.rebuild_from_props(props)
        _log(props, "INFO",
             f"attached {mid}: {len(hinges)} hinges, {len(rails)} rails, "
             f"{n_adopted} joint(s) adopted, {n_tagged} handle(s) tagged", self)
        return {"FINISHED"}


# ─────────────────────────────────────────────────── hinge/rail installs ───

def _hinge_cfg(item) -> dict:
    return {"category": item.category, "variant": item.variant,
            "use_auto_scale": item.use_auto_scale,
            "scale_factor": item.scale_factor,
            "hinge_count": item.hinge_count,
            "strict_border": item.strict_border,
            "flip_side": item.flip_side,
            "use_decompose": item.use_decompose}


def _restore_hinge_cfg(item, cfg: dict) -> None:
    with gp.suspend_updates():
        item.category = cfg["category"]
        try:
            item.variant = cfg["variant"]
        except (TypeError, ValueError):
            pass
        item.use_auto_scale = cfg["use_auto_scale"]
        item.scale_factor = cfg["scale_factor"]
        item.hinge_count = cfg["hinge_count"]
        item.strict_border = cfg["strict_border"]
        item.flip_side = cfg["flip_side"]
        item.use_decompose = cfg["use_decompose"]


def _run_hinge_install(props, state, rec, item) -> dict:
    scale = None if item.use_auto_scale else float(item.scale_factor)
    return ih.install_one_hinge(
        record=rec,
        body_obbs_picker=state["body_obbs"],
        pred_id_to_obj=state["pred_id_to_obj"],
        category=item.category or props.default_hinge_cat,
        variant_filename=item.variant or props.default_hinge_var,
        hinge_count=int(item.hinge_count or props.default_hinge_count),
        scale_factor=scale,
        strict_border=bool(item.strict_border),
        flip_side=bool(item.flip_side),
        use_panel_decompose=bool(item.use_decompose),
    )


def _install_hinge_row(props, state, item) -> bool:
    rec = _find_record(state, "hinges", item.joint_name)
    if rec is None:
        item.status = "err: record not found"
        return False
    diag = _run_hinge_install(props, state, rec, item)
    item.detail = json.dumps(diag)
    if diag.get("ok"):
        eb = diag.get("effective_border", "?")
        note = f"ok ({diag.get('chosen_type','?')}, border={eb}"
        if diag.get("policy_pen_mm") is not None:
            note += f", pen={diag['policy_pen_mm']:.1f}mm"
        note += ")"
        if diag.get("flip_risk"):
            # A mount on the OPPOSITE edge existed — the geometry pick may
            # have hinged the door on the wrong side. Flip side / re-check.
            note = ("warn: border filter fell back with a flip-risk "
                    f"candidate on the opposite edge "
                    f"({diag.get('chosen_type','?')}) — check hinge side, "
                    f"use Flip side if wrong")
        item.status = note
        item.last_ok = json.dumps(_hinge_cfg(item))
        return True

    err = diag.get("error", "?")
    failed_cat = item.category
    # The failed attempt already removed the previous install (remove-then-
    # place). Never leave the joint bare: revert to the last-known-good
    # settings and re-install them.
    if item.last_ok:
        try:
            cfg = json.loads(item.last_ok)
        except Exception:
            cfg = None
        if cfg and cfg != _hinge_cfg(item):
            _restore_hinge_cfg(item, cfg)
            diag2 = _run_hinge_install(props, state, rec, item)
            if diag2.get("ok"):
                reason = ("no placement candidates"
                          if "no candidates" in err else err.split(": ")[-1][:50])
                item.status = (f"warn: {failed_cat} does not fit this joint "
                               f"({reason}) — kept "
                               f"{diag2.get('chosen_type', cfg['category'])}")
                return False
    item.status = f"err: {err[:80]}"
    return False


def _install_rail_row(props, state, item) -> bool:
    rec = _find_record(state, "rails", item.joint_name)
    if rec is None:
        item.status = "err: record not found"
        return False
    diag = ih.install_one_rail(record=rec,
                               variant_filename=item.variant or props.default_rail_var)
    item.detail = json.dumps(diag)
    if diag.get("ok"):
        item.status = f"ok ({diag.get('style','?')})"
        item.last_ok = json.dumps({"variant": item.variant})
        return True
    err = diag.get("error", "?")
    # Same never-leave-bare rule as hinges: revert to the last-known-good
    # variant when a variant switch fails.
    if item.last_ok:
        try:
            cfg = json.loads(item.last_ok)
        except Exception:
            cfg = None
        if cfg and cfg.get("variant") and cfg["variant"] != item.variant:
            failed_variant = item.variant
            with gp.suspend_updates():
                try:
                    item.variant = cfg["variant"]
                except (TypeError, ValueError):
                    cfg = None
            if cfg:
                diag2 = ih.install_one_rail(record=rec,
                                            variant_filename=cfg["variant"])
                if diag2.get("ok"):
                    item.status = (f"warn: {failed_variant} failed "
                                   f"({err[:50]}) — kept {cfg['variant']}")
                    return False
    item.status = f"err: {err[:80]}"
    return False


class GRAPH_OT_apply_all_hinges(Operator):
    """Install every predicted hinge with its per-row settings"""
    bl_idname = "graph_func.apply_all_hinges"
    bl_label = "Apply All Hinges"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        ih.init_paths(_resolve_repo_root(props))
        n_ok = n_err = 0
        with gp.suspend_updates():
            for item in props.hinges:
                if _install_hinge_row(props, state, item):
                    n_ok += 1
                else:
                    n_err += 1
        overlay.rebuild_from_props(props)
        _log(props, "INFO" if n_err == 0 else "WARN",
             f"hinges: {n_ok} ok, {n_err} err", self)
        return {"FINISHED"}


class GRAPH_OT_reapply_one_hinge(Operator):
    """Re-install the active hinge with its current row settings"""
    bl_idname = "graph_func.reapply_one_hinge"
    bl_label = "Re-apply Hinge"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        if not props.hinges:
            _log(props, "ERR", "no hinges in list", self)
            return {"CANCELLED"}
        idx = max(0, min(props.active_hinge_idx, len(props.hinges) - 1))
        item = props.hinges[idx]
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        ih.init_paths(_resolve_repo_root(props))
        with gp.suspend_updates():
            ok = _install_hinge_row(props, state, item)
        overlay.rebuild_from_props(props)
        if ok:
            _log(props, "INFO", f"{item.joint_name}: {item.status}", self)
            return {"FINISHED"}
        _log(props, "ERR", f"{item.joint_name}: {item.status}", self)
        return {"CANCELLED"}


class GRAPH_OT_cycle_hinge_variant(Operator):
    """Switch the active hinge to the previous/next template variant"""
    bl_idname = "graph_func.cycle_hinge_variant"
    bl_label = "Cycle Hinge Variant"
    bl_options = {"REGISTER", "UNDO"}

    delta: bpy.props.IntProperty(default=1)

    def execute(self, context):
        props = _props(context)
        if not props.hinges:
            return {"CANCELLED"}
        idx = max(0, min(props.active_hinge_idx, len(props.hinges) - 1))
        item = props.hinges[idx]
        if item.category == "auto":
            _log(props, "WARN",
                 "auto mode competes all classes — pick a class to cycle "
                 "its variants", self)
            return {"CANCELLED"}
        items = gp._HINGE_VARIANT_ITEMS_BY_CATEGORY.get(item.category, [])
        names = [it[0] for it in items]
        if not names:
            return {"CANCELLED"}
        try:
            cur = names.index(item.variant)
        except ValueError:
            cur = 0
        with gp.suspend_updates():
            item.variant = names[(cur + int(self.delta)) % len(names)]
        return bpy.ops.graph_func.reapply_one_hinge()


class GRAPH_OT_apply_all_rails(Operator):
    """Install every predicted rail (plus optional support blocks)"""
    bl_idname = "graph_func.apply_all_rails"
    bl_label = "Apply All Rails"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        ih.init_paths(_resolve_repo_root(props))
        n_ok = n_err = 0
        with gp.suspend_updates():
            for item in props.rails:
                if _install_rail_row(props, state, item):
                    n_ok += 1
                else:
                    n_err += 1
        if props.add_support_blocks and state.get("rails"):
            try:
                sup = ih.add_support_blocks(
                    state["rails"], enable_divider=bool(props.enable_divider))
                _log(props, "INFO",
                     f"supports: {sup.get('n_slabs_built',0)} slabs, "
                     f"{sup.get('n_dividers_built',0)} dividers")
            except Exception as exc:
                _log(props, "WARN", f"support blocks failed: {exc}")
        overlay.rebuild_from_props(props)
        _log(props, "INFO" if n_err == 0 else "WARN",
             f"rails: {n_ok} ok, {n_err} err", self)
        return {"FINISHED"}


class GRAPH_OT_reapply_one_rail(Operator):
    """Re-install the active rail with its current row variant"""
    bl_idname = "graph_func.reapply_one_rail"
    bl_label = "Re-apply Rail"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        if not props.rails:
            return {"CANCELLED"}
        idx = max(0, min(props.active_rail_idx, len(props.rails) - 1))
        item = props.rails[idx]
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        ih.init_paths(_resolve_repo_root(props))
        with gp.suspend_updates():
            ok = _install_rail_row(props, state, item)
        overlay.rebuild_from_props(props)
        if ok:
            _log(props, "INFO", f"{item.joint_name}: {item.status}", self)
            return {"FINISHED"}
        _log(props, "ERR", f"{item.joint_name}: {item.status}", self)
        return {"CANCELLED"}


class GRAPH_OT_cycle_rail_variant(Operator):
    """Switch the active rail to the previous/next template variant"""
    bl_idname = "graph_func.cycle_rail_variant"
    bl_label = "Cycle Rail Variant"
    bl_options = {"REGISTER", "UNDO"}

    delta: bpy.props.IntProperty(default=1)

    def execute(self, context):
        props = _props(context)
        if not props.rails:
            return {"CANCELLED"}
        idx = max(0, min(props.active_rail_idx, len(props.rails) - 1))
        item = props.rails[idx]
        names = [it[0] for it in gp._RAIL_VARIANT_ITEMS]
        if not names:
            return {"CANCELLED"}
        try:
            cur = names.index(item.variant)
        except ValueError:
            cur = 0
        with gp.suspend_updates():
            item.variant = names[(cur + int(self.delta)) % len(names)]
        return bpy.ops.graph_func.reapply_one_rail()


# ───────────────────────────────────────────────────── handle groups ───────

def _import_add_handle(props):
    ih.init_paths(_resolve_repo_root(props))
    import importlib
    import add_handle as add_handle_mod      # type: ignore
    importlib.reload(add_handle_mod)
    return add_handle_mod


def _import_add_top(props):
    ih.init_paths(_resolve_repo_root(props))
    import importlib
    import add_top as add_top_mod            # type: ignore
    importlib.reload(add_top_mod)
    return add_top_mod


_HANDLE_PARENT_TAG = "grafu_handle_parent"


def _remove_installed_handles_for_parent(parent_id: str) -> int:
    removed = 0
    for o in list(bpy.data.objects):
        try:
            tag = o.get(_HANDLE_PARENT_TAG) or o.get("__graph_handle_parent")
            if tag == parent_id:
                bpy.data.objects.remove(o, do_unlink=True)
                removed += 1
        except Exception:
            pass
    return removed


def _tag_new_handle_objs(before: set, parent_id: str) -> int:
    tagged = 0
    for o in bpy.data.objects:
        if o.name in before:
            continue
        try:
            o[_HANDLE_PARENT_TAG] = parent_id
            tagged += 1
        except Exception:
            pass
    return tagged


def _motion_for_parent(state: dict, parent_id: str):
    for r in state.get("hinges", []):
        if r["door_node"] == parent_id:
            return (np.asarray(r["hinge_axis"], float),
                    np.asarray(r["joint_origin"], float),
                    "door")
    for r in state.get("rails", []):
        if r["drawer_node"] == parent_id:
            return (np.asarray(r["slide_axis_world"], float),
                    np.asarray(r["slide_origin"], float),
                    "drawer")
    return None


_MESH_BACKUP_TAG = "grafu_orig_mesh"


def _ensure_mesh_backup(obj) -> None:
    """Snapshot the parent's mesh datablock before a destructive edit
    (recessed carving) so the operation stays re-appliable/removable."""
    if obj.get(_MESH_BACKUP_TAG):
        return
    backup = obj.data.copy()
    backup.name = f"{obj.data.name}__grafu_orig"
    backup.use_fake_user = True
    obj[_MESH_BACKUP_TAG] = backup.name


def _restore_mesh_backup(obj) -> bool:
    """Swap the parent's mesh back to the pristine snapshot (keeping the
    snapshot for the next carve). Returns True if a restore happened."""
    name = obj.get(_MESH_BACKUP_TAG)
    if not name:
        return False
    backup = bpy.data.meshes.get(name)
    if backup is None:
        try:
            del obj[_MESH_BACKUP_TAG]
        except Exception:
            pass
        return False
    carved = obj.data
    obj.data = backup.copy()
    obj.data.name = carved.name if carved else obj.name
    if carved is not None and carved.users == 0:
        bpy.data.meshes.remove(carved)
    return True


def _anchored_handle_objs(props, state, parent_id):
    raw = _raw_state(props) or {}
    n_input = len(raw.get("input_ids", []))
    try:
        pred_nodes = json.loads(
            Path(bpy.path.abspath(props.pred_path)).read_text())["nodes"]
    except Exception:
        pred_nodes = {}
    out = []
    for h_id, p_id in state.get("attached_pairs", []):
        if p_id != parent_id:
            continue
        slot = int(pred_nodes.get(h_id, {}).get("slot", -1))
        if 0 <= slot < n_input:
            obj = state["pred_id_to_obj"].get(h_id)
            if obj is not None and obj.name in bpy.data.objects:
                out.append(obj)
    return out


def _install_handle_group(props, state, it) -> bool:
    """Apply the active handle group per its row settings.

    ADDITIVE: install `count` template handles (count>1 → knobs at equal
              partitions (i+1)/(N+1); count==0 → remove). Anchored input
              handle meshes are deleted when `replace_anchored` is on.
    RECESSED: carve a pull groove into the parent panel itself (joint-aware
              placement; single groove — count does not apply).

    Idempotent: previously installed templates are wiped and any prior
    carve is undone (mesh restored from a snapshot) before applying.
    """
    add_handle_mod = _import_add_handle(props)
    parent_obj = state["pred_id_to_obj"].get(it.parent_id)
    if parent_obj is None:
        it.status = "err: parent object missing"
        return False
    motion = _motion_for_parent(state, it.parent_id)
    if motion is None:
        it.status = ("err: no motion record for parent (no hinge/rail edge "
                     "was predicted for it)")
        return False
    axis, origin, kind = motion

    # ── reset to a clean slate ────────────────────────────────────────────
    n_removed = _remove_installed_handles_for_parent(it.parent_id)
    restored = _restore_mesh_backup(parent_obj)

    anchored_objs = _anchored_handle_objs(props, state, it.parent_id)

    def _drop_anchored():
        for obj in anchored_objs:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass

    # ── RECESSED: carve a groove in place ─────────────────────────────────
    if it.mode == "RECESSED":
        from ..utils import handle_placement as hp
        _ensure_mesh_backup(parent_obj)
        prev_active = bpy.context.view_layer.objects.active
        bpy.ops.object.select_all(action="DESELECT")
        parent_obj.select_set(True)
        bpy.context.view_layer.objects.active = parent_obj
        try:
            ok = hp.apply_subtractive_handle_inplace(
                sub_type=it.sub_shape,
                depth_frac=float(it.sub_depth),
                method=it.sub_method,
                joint_info={"type": "hinge" if kind == "door" else "slider",
                            "origin": origin.tolist(),
                            "direction": axis.tolist()},
                target_type=kind,
            )
        except Exception as exc:
            import traceback; traceback.print_exc()
            it.detail = str(exc)
            ok = False
        finally:
            if prev_active is not None and \
                    prev_active.name in bpy.data.objects:
                bpy.context.view_layer.objects.active = prev_active
        if ok:
            if it.replace_anchored:
                _drop_anchored()
            it.status = (f"ok (recessed {it.sub_shape}, "
                         f"depth {it.sub_depth:.0%}, {it.sub_method.lower()})")
            return True
        _restore_mesh_backup(parent_obj)
        it.status = "err: recessed carve failed (see console)"
        return False

    # ── ADDITIVE: template handles ────────────────────────────────────────
    count = int(it.count)
    if count == 0:
        if it.replace_anchored:
            _drop_anchored()
        it.status = (f"ok (removed"
                     f"{f' {n_removed} template' if n_removed else ''}"
                     f"{', mesh restored' if restored else ''})")
        return True

    # Multi-handle rule: equal partitions in the user's chosen style —
    # each handle auto-shrinks to its partition slot (a long bar becomes
    # shorter when several share the edge). Single handles honour the
    # per-group Position preset instead of the partition sequence.
    style = it.style

    n_ok = 0
    before = {o.name for o in bpy.data.objects}
    for i in range(count):
        frac = ((i + 1) / (count + 1) if count > 1
                else float(it.pos_frac))
        try:
            ok = add_handle_mod.add_handle_for_predicted_attachment(
                parent_obj=parent_obj,
                motion_axis=axis,
                motion_origin=origin,
                parent_kind=kind,
                handle_style=style,
                partition_frac=frac,
                n_handles=count,
                edge_margin_frac=float(it.margin_edge),
            )
        except Exception as exc:
            import traceback; traceback.print_exc()
            ok = False
            it.detail = str(exc)
        if ok:
            n_ok += 1
    _tag_new_handle_objs(before, it.parent_id)

    if n_ok and it.replace_anchored:
        _drop_anchored()

    if n_ok == count:
        it.status = (f"ok ({n_ok} {style}{', partitions' if count > 1 else ''}"
                     f"{', replaced anchored' if it.replace_anchored else ''}"
                     f"{f', wiped {n_removed}' if n_removed else ''})")
        return True
    it.status = f"err: {n_ok}/{count} placed"
    return n_ok > 0


class GRAPH_OT_add_group_handles(Operator):
    """Install handles for the active parent group (idempotent: replaces
    this group's previously installed templates)"""
    bl_idname = "graph_func.add_group_handles"
    bl_label = "Add Handles (Group)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        if not props.handle_groups:
            _log(props, "ERR", "no handle groups", self)
            return {"CANCELLED"}
        idx = max(0, min(props.active_hgroup_idx, len(props.handle_groups) - 1))
        it = props.handle_groups[idx]
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        ok = _install_handle_group(props, state, it)
        _log(props, "INFO" if ok else "ERR",
             f"handles on {it.parent_id}: {it.status}", self)
        return {"FINISHED"} if ok else {"CANCELLED"}


class GRAPH_OT_add_all_handles(Operator):
    """Apply every parent group as configured (fully equips the model)"""
    bl_idname = "graph_func.add_all_predicted_handles"
    bl_label = "Apply All Handles"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        n_ok = n_err = 0
        with gp.suspend_updates():
            for it in props.handle_groups:
                if it.mode == "ADDITIVE" and it.count == 0:
                    continue
                if _install_handle_group(props, state, it):
                    n_ok += 1
                else:
                    n_err += 1
        _log(props, "INFO" if n_err == 0 else "WARN",
             f"handle groups: {n_ok} ok, {n_err} err", self)
        return {"FINISHED"}


class GRAPH_OT_remove_group_handles(Operator):
    """Remove this parent's installed templates and undo any carved groove
    (anchored input handles are kept)"""
    bl_idname = "graph_func.remove_group_handles"
    bl_label = "Remove Handles (Group)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        if not props.handle_groups:
            return {"CANCELLED"}
        idx = max(0, min(props.active_hgroup_idx, len(props.handle_groups) - 1))
        it = props.handle_groups[idx]
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        n_removed = _remove_installed_handles_for_parent(it.parent_id)
        parent_obj = state["pred_id_to_obj"].get(it.parent_id)
        restored = (_restore_mesh_backup(parent_obj)
                    if parent_obj is not None else False)
        with gp.suspend_updates():
            it.status = (f"ok (removed {n_removed} template(s)"
                         f"{', groove undone' if restored else ''})")
        _log(props, "INFO", f"{it.parent_id}: {it.status}", self)
        return {"FINISHED"}


class GRAPH_OT_cycle_handle_style(Operator):
    """Switch the active group to the previous/next handle style and
    re-apply"""
    bl_idname = "graph_func.cycle_handle_style"
    bl_label = "Cycle Handle Style"
    bl_options = {"REGISTER", "UNDO"}

    delta: bpy.props.IntProperty(default=1)

    def execute(self, context):
        props = _props(context)
        if not props.handle_groups:
            return {"CANCELLED"}
        idx = max(0, min(props.active_hgroup_idx, len(props.handle_groups) - 1))
        it = props.handle_groups[idx]
        names = [i[0] for i in gp.HANDLE_STYLE_ITEMS]
        try:
            cur = names.index(it.style)
        except ValueError:
            cur = 0
        with gp.suspend_updates():
            it.style = names[(cur + int(self.delta)) % len(names)]
        return bpy.ops.graph_func.add_group_handles()


# ─────────────────────────────────────────────────────────── tops ──────────

def _body_objs_for_top(props, state, add_top_mod):
    pred = json.loads(Path(bpy.path.abspath(props.pred_path)).read_text())
    body_objs = []
    for nid, n in pred["nodes"].items():
        m = pg.mat_base(n.get("material", ""))
        if m in pg.DYNAMIC_MATS or m == "handle":
            continue
        if add_top_mod.is_top_panel_node(n):
            continue
        o = state["pred_id_to_obj"].get(nid)
        if o is not None:
            body_objs.append(o)
    return pred, body_objs


class GRAPH_OT_detect_top(Operator):
    """Check whether the loaded model is missing a top panel (coverage test)"""
    bl_idname = "graph_func.detect_top"
    bl_label = "Detect Missing Top"

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        add_top_mod = _import_add_top(props)
        pred, body_objs = _body_objs_for_top(props, state, add_top_mod)
        n_anchored_tops = 0
        n_input = len(state.get("input_ids", []))
        for nid, n in pred["nodes"].items():
            if add_top_mod.is_top_panel_node(n) and \
                    0 <= int(n.get("slot", -1)) < n_input:
                n_anchored_tops += 1
        fired_top = any(add_top_mod.is_top_panel_node(n)
                        and int(n.get("slot", -1)) >= n_input
                        for n in pred["nodes"].values())
        if not body_objs:
            _log(props, "ERR", "no body meshes to analyse", self)
            return {"CANCELLED"}
        from ..utils import top_processor as tp
        # Join a temp copy of the body meshes for the coverage raster.
        dups = []
        for o in body_objs:
            d = o.copy(); d.data = o.data.copy()
            d.name = "__top_detect_tmp"
            context.scene.collection.objects.link(d)
            dups.append(d)
        try:
            bpy.ops.object.select_all(action="DESELECT")
            for d in dups:
                d.select_set(True)
            context.view_layer.objects.active = dups[0]
            if len(dups) > 1:
                bpy.ops.object.join()
            joined = context.view_layer.objects.active
            res = tp.detect_missing_top(joined, up_axis="+Z")
            missing = bool(res.get("missing"))
            cov = res.get("coverage", -1.0)
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(props, "ERR", f"top detection failed: {exc}", self)
            return {"CANCELLED"}
        finally:
            for d in [o for o in bpy.data.objects
                      if o.name.startswith("__top_detect_tmp")]:
                bpy.data.objects.remove(d, do_unlink=True)
        msg = (f"top coverage {cov:.2f} → "
               f"{'MISSING' if missing else 'present'}"
               f"{'; model fired a top node' if fired_top else ''}"
               f"{'; anchored top exists' if n_anchored_tops else ''}")
        _log(props, "WARN" if missing and not n_anchored_tops else "INFO",
             msg, self)
        return {"FINISHED"}


class GRAPH_OT_add_predicted_top(Operator):
    """Synthesise a top panel above the union of body meshes"""
    bl_idname = "graph_func.add_predicted_top"
    bl_label = "Add Top"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        add_top_mod = _import_add_top(props)
        pred, body_objs = _body_objs_for_top(props, state, add_top_mod)
        if not body_objs:
            _log(props, "ERR", "no body meshes to build top from", self)
            return {"CANCELLED"}
        # Idempotent: remove a previously synthesised top first.
        for o in list(bpy.data.objects):
            if o.name.startswith("predicted_top"):
                bpy.data.objects.remove(o, do_unlink=True)
        try:
            top = add_top_mod.add_top_from_body_meshes(
                body_objs=body_objs,
                up_axis="+Z",
                overhang_pct=float(props.top_overhang),
                thickness_abs=float(props.top_thickness),
                shape_style=props.top_shape_style,
                corner_radius=float(props.top_corner_radius),
                name="predicted_top",
                material=pg.material_for("top panel", alpha=1.0),
            )
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(props, "ERR", f"add_top: {exc}", self)
            return {"CANCELLED"}
        if top is None:
            _log(props, "ERR", "add_top returned None (see console for the "
                 "diagnostic — usually 'generated top too small')", self)
            return {"CANCELLED"}
        _log(props, "INFO", f"added top: {top.name}", self)
        return {"FINISHED"}


# ───────────────────────────────────────────────────────── interior ────────

class GRAPH_OT_detect_interior(Operator):
    """Detect interior compartments from the loaded graph's part boxes"""
    bl_idname = "graph_func.detect_interior"
    bl_label = "Detect Compartments"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        pred = json.loads(Path(bpy.path.abspath(props.pred_path)).read_text())
        from ..utils import interior_graph as ig
        try:
            boxes, mounts = ig.detect_and_visualize(
                state, pred["nodes"],
                props.interior_up_axis, props.interior_front_axis)
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(props, "ERR", f"detect interior: {exc}", self)
            return {"CANCELLED"}
        props.interior_detected_count = len(boxes)
        if not boxes and not mounts:
            _log(props, "WARN", "no compartments found (cavity fully "
                 "occupied or degenerate)", self)
            return {"CANCELLED"}
        _log(props, "INFO", f"detected {len(boxes)} compartment(s), "
             f"{len(mounts)} mounting divider(s)", self)
        return {"FINISHED"}


class GRAPH_OT_instantiate_interior(Operator):
    """One panel per fired shelf/divider node (connectivity-placed,
    idempotent — clears INTERIOR_* first)"""
    bl_idname = "graph_func.instantiate_interior"
    bl_label = "Instantiate From Prediction"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        ih.init_paths(_resolve_repo_root(props))
        import importlib
        import instantiate_interior as inst_mod   # type: ignore
        importlib.reload(inst_mod)
        pred = json.loads(Path(bpy.path.abspath(props.pred_path)).read_text())
        n_input = len(state.get("input_ids", []))
        try:
            res = inst_mod.instantiate_interior_from_pred(
                pred, n_input, seed_name=props.model_label or "scene")
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(props, "ERR", f"instantiate interior: {exc}", self)
            return {"CANCELLED"}
        if res["fired"] == 0:
            _log(props, "INFO", "no fired shelf/divider nodes", self)
            return {"FINISHED"}
        if res["no_compartments"]:
            _log(props, "WARN",
                 f"no free compartments — {res['fired']} fired node(s) "
                 f"could not be instantiated (all-drawer body?)", self)
            return {"CANCELLED"}
        lvl = "INFO" if not res["skipped"] else "WARN"
        _log(props, lvl,
             f"interior: {res['made']} panel(s) for {res['fired']} fired "
             f"node(s)"
             + (f", skipped {len(res['skipped'])}" if res["skipped"] else ""),
             self)
        return {"FINISHED"}


class GRAPH_OT_generate_interior_panels(Operator):
    """Generate equal-partition panels in the selected (or all) compartments"""
    bl_idname = "graph_func.generate_interior_panels"
    bl_label = "Generate Panels"
    bl_options = {"REGISTER", "UNDO"}

    use_all: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        props = _props(context)
        res = iscene.generate_panels_for_compartments(
            orientation=props.interior_orientation,
            num_panels=int(props.interior_num_panels),
            panel_thickness=float(props.interior_panel_thickness),
            front_inset=float(props.interior_front_inset),
            use_all=bool(self.use_all),
        )
        if not res.get("success"):
            _log(props, "ERR", res.get("error") or "panel generation failed",
                 self)
            return {"CANCELLED"}
        _log(props, "INFO",
             f"generated {res['panel_count']} panel(s) in "
             f"{res['compartments_processed']} compartment(s)", self)
        return {"FINISHED"}


class GRAPH_OT_clear_interior_panels(Operator):
    """Remove all generated interior panels"""
    bl_idname = "graph_func.clear_interior_panels"
    bl_label = "Clear Panels"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        iscene.clear_interior_panels()
        _log(_props(context), "INFO", "interior panels cleared", self)
        return {"FINISHED"}


class GRAPH_OT_clear_compartment_boxes(Operator):
    """Remove compartment visualisation boxes"""
    bl_idname = "graph_func.clear_compartment_boxes"
    bl_label = "Clear Detection"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        iscene.clear_compartment_boxes()
        props.interior_detected_count = 0
        _log(props, "INFO", "compartment boxes cleared", self)
        return {"FINISHED"}


class GRAPH_OT_commit_interior(Operator):
    """Commit interior panels (rename to shelf_*/divider_*, drop boxes)"""
    bl_idname = "graph_func.commit_interior"
    bl_label = "Commit Interior"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        res = iscene.commit_interior_panels()
        props.interior_detected_count = 0
        _log(props, "INFO", f"committed {res['panel_count']} panel(s)", self)
        return {"FINISHED"}


class GRAPH_OT_add_support_blocks(Operator):
    """Install drawer support slabs / dividers for the loaded rails"""
    bl_idname = "graph_func.add_support_blocks"
    bl_label = "Add Drawer Support Blocks"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        ih.init_paths(_resolve_repo_root(props))
        try:
            sup = ih.add_support_blocks(
                state["rails"], enable_divider=bool(props.enable_divider))
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(props, "ERR", f"support: {exc}", self)
            return {"CANCELLED"}
        _log(props, "INFO",
             f"support: slabs={sup.get('n_slabs_built',0)} "
             f"dividers={sup.get('n_dividers_built',0)} "
             f"skipped={sup.get('n_skipped',0)}", self)
        return {"FINISHED"}


# ─────────────────────────────────────────────── auto-functionalize ────────

class GRAPH_OT_auto_functionalize(Operator):
    """One-click functionalization: hinges → rails → supports → handles →
    top → interior, with progress and ESC-cancel"""
    bl_idname = "graph_func.auto_functionalize"
    bl_label = "Auto-Functionalize"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _tasks: list = []
    _i = 0
    _n_ok = 0
    _n_err = 0

    def _build_tasks(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        tasks = []
        for item in props.hinges:
            tasks.append((f"hinge {item.joint_name}",
                          lambda it=item: _install_hinge_row(props, state, it)))
        for item in props.rails:
            tasks.append((f"rail {item.joint_name}",
                          lambda it=item: _install_rail_row(props, state, it)))
        if props.add_support_blocks and state.get("rails"):
            def _supports():
                ih.add_support_blocks(state["rails"],
                                      enable_divider=bool(props.enable_divider))
                return True
            tasks.append(("support blocks", _supports))
        for it in props.handle_groups:
            if it.n_new > 0:   # auto pass installs fired handles only
                tasks.append((f"handles {it.parent_id}",
                              lambda g=it: _install_handle_group(props, state, g)))
        raw = _raw_state(props) or {}
        free_mats = {pg.mat_base(f.get("material", ""))
                     for f in raw.get("free_nodes", [])}
        if free_mats & {"top panel", "top_panel", "countertop"}:
            def _top():
                r = bpy.ops.graph_func.add_predicted_top()
                return "FINISHED" in r
            tasks.append(("top panel", _top))
        if free_mats & {"shelf", "divider"}:
            def _interior():
                r = bpy.ops.graph_func.instantiate_interior()
                return "FINISHED" in r
            tasks.append(("interior", _interior))
        return tasks

    def _run_one(self, context) -> bool:
        props = _props(context)
        label, fn = self._tasks[self._i]
        props.auto_progress = (f"{self._i + 1}/{len(self._tasks)}: {label}")
        try:
            ok = bool(fn())
        except Exception as exc:
            import traceback; traceback.print_exc()
            _log(props, "ERR", f"{label}: {exc}")
            ok = False
        self._n_ok += 1 if ok else 0
        self._n_err += 0 if ok else 1
        self._i += 1
        return self._i >= len(self._tasks)

    def _finish(self, context, cancelled=False):
        props = _props(context)
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        try:
            wm.progress_end()
        except Exception:
            pass
        props.auto_running = False
        props.auto_progress = ""
        overlay.rebuild_from_props(props)
        lvl = "WARN" if (self._n_err or cancelled) else "INFO"
        _log(props, lvl,
             f"auto-functionalize {'cancelled' if cancelled else 'done'}: "
             f"{self._n_ok} ok, {self._n_err} err "
             f"({self._i}/{len(self._tasks)} steps)", self)

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        ih.init_paths(_resolve_repo_root(props))
        gp._SUSPEND += 1   # suspended for the whole modal run
        try:
            self._tasks = self._build_tasks(context)
        except Exception:
            gp._SUSPEND -= 1
            raise
        self._i = 0; self._n_ok = 0; self._n_err = 0
        if not self._tasks:
            gp._SUSPEND -= 1
            _log(props, "WARN", "nothing to do", self)
            return {"CANCELLED"}
        wm = context.window_manager
        if bpy.app.background or context.window is None:
            # Headless parity: run synchronously.
            done = False
            while not done:
                done = self._run_one(context)
            self._finish(context)
            gp._SUSPEND -= 1
            return {"FINISHED"}
        props.auto_running = True
        try:
            wm.progress_begin(0, len(self._tasks))
        except Exception:
            pass
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        props = _props(context)
        if event.type == 'ESC':
            self._finish(context, cancelled=True)
            gp._SUSPEND -= 1
            return {"CANCELLED"}
        if event.type == 'TIMER':
            done = self._run_one(context)
            try:
                context.window_manager.progress_update(self._i)
            except Exception:
                pass
            for area in context.screen.areas:
                area.tag_redraw()
            if done:
                self._finish(context)
                gp._SUSPEND -= 1
                return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        # Blender force-terminates modals on file load / window close via
        # cancel(), which never reaches modal(): release the suspend here
        # or live-update callbacks stay disabled for the whole session.
        self._finish(context, cancelled=True)
        gp._SUSPEND -= 1


# ─────────────────────────────────────────── preview / solo / validate ─────

_ANIM_FRAME_END = 100
_ANIM_FRAME_MAX_EXTENT = 50
_SOLO_STASH = "grafu_solo_stash"


class GRAPH_OT_preview_motion(Operator):
    """Play the 0–100 open-close cycle (toggles playback)"""
    bl_idname = "graph_func.preview_motion"
    bl_label = "Preview Motion"

    def execute(self, context):
        scene = context.scene
        scene.frame_start = 0
        scene.frame_end = _ANIM_FRAME_END
        bpy.ops.screen.animation_play()
        return {"FINISHED"}


class GRAPH_OT_jump_max_extent(Operator):
    """Jump to frame 50 — every joint at maximum extent"""
    bl_idname = "graph_func.jump_max_extent"
    bl_label = "Max Extent"

    def execute(self, context):
        context.scene.frame_set(_ANIM_FRAME_MAX_EXTENT)
        return {"FINISHED"}


class GRAPH_OT_solo_joint(Operator):
    """Animate only the active joint (stashes every other joint's action);
    run again to restore"""
    bl_idname = "graph_func.solo_joint"
    bl_label = "Solo Active Joint"
    bl_options = {"REGISTER", "UNDO"}

    row_kind: bpy.props.EnumProperty(items=[("HINGE", "Hinge", ""),
                                            ("RAIL", "Rail", "")],
                                     default="HINGE")

    def execute(self, context):
        props = _props(context)
        # Un-solo pass: restore every stashed action.
        restored = 0
        for o in bpy.data.objects:
            stash = o.get(_SOLO_STASH)
            if stash:
                act = bpy.data.actions.get(stash)
                if act is not None:
                    if o.animation_data is None:
                        o.animation_data_create()
                    o.animation_data.action = act
                try:
                    del o[_SOLO_STASH]
                except Exception:
                    pass
                restored += 1
        if restored:
            _log(props, "INFO", f"solo off — restored {restored} action(s)",
                 self)
            return {"FINISHED"}

        if self.row_kind == "HINGE":
            if not props.hinges:
                return {"CANCELLED"}
            item = props.hinges[max(0, min(props.active_hinge_idx,
                                           len(props.hinges) - 1))]
            keep_key = item.joint_name
        else:
            if not props.rails:
                return {"CANCELLED"}
            item = props.rails[max(0, min(props.active_rail_idx,
                                          len(props.rails) - 1))]
            keep_key = item.joint_name
        n_stashed = 0
        for o in bpy.data.objects:
            ad = o.animation_data
            if ad is None or ad.action is None:
                continue
            keep = (keep_key in o.name
                    or o.get("__rail_joint") == keep_key)
            # Children of the joint's driver also keep animating.
            p = o.parent
            while p is not None and not keep:
                keep = keep_key in p.name or p.get("__rail_joint") == keep_key
                p = p.parent
            if keep:
                continue
            o[_SOLO_STASH] = ad.action.name
            ad.action = None
            n_stashed += 1
        _log(props, "INFO",
             f"solo {keep_key}: muted {n_stashed} other action(s) — run "
             f"again to restore", self)
        return {"FINISHED"}


def _world_bvh(o):
    """World-space BVH for one mesh object.

    BVHTree.FromObject lives in the object's LOCAL space, which is
    frame-invariant for transform-animated parts and unrelated to world
    coordinates — bake matrix_world into a bmesh copy instead (same idiom
    as rail.coborder._build_bvh_from_objects)."""
    import bmesh
    from mathutils.bvhtree import BVHTree
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.transform(o.matrix_world)
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree


class GRAPH_OT_test_connectivity(Operator):
    """Connectivity sweep (M3-style): flags joints whose moving part drifts
    farther than 5 mm from the static body at any point of the open cycle"""
    bl_idname = "graph_func.test_connectivity"
    bl_label = "Connectivity Test"

    _PROX = 0.005      # 5 mm — same threshold the paper's M3 reports
    _MAX_SAMPLES = 120  # vertex samples per dynamic part per frame

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        deps = context.evaluated_depsgraph_get()
        scene = context.scene
        prev_frame = scene.frame_current

        scene.frame_set(0)
        deps.update()
        static_names = {f"{nid}__mesh0" for nid in state.get("static_ids", [])}
        statics = [o for o in bpy.data.objects
                   if o.type == "MESH" and o.name in static_names]
        if not statics:
            _log(props, "ERR", "no static body meshes found", self)
            return {"CANCELLED"}
        # Statics don't animate — build their world-space BVHs once (the
        # queries below use world-space points).
        static_trees = [_world_bvh(o) for o in statics]

        def min_gap(dyn_objs) -> float:
            gap = float("inf")
            for d in dyn_objs:
                mesh = d.data
                n = len(mesh.vertices)
                stride = max(1, n // self._MAX_SAMPLES)
                mw = d.matrix_world
                for vi in range(0, n, stride):
                    w = mw @ mesh.vertices[vi].co
                    for t in static_trees:
                        hit = t.find_nearest(w)
                        if hit is not None and hit[3] < gap:
                            gap = hit[3]
                    if gap <= self._PROX:
                        return gap
            return gap

        rows = [(it, _find_record(state, "hinges", it.joint_name))
                for it in props.hinges]
        rows += [(it, _find_record(state, "rails", it.joint_name))
                 for it in props.rails]
        frames = list(range(0, _ANIM_FRAME_END + 1, 10))
        n_detached = n_checked = 0
        for item, rec in rows:
            if rec is None or not item.status.startswith(("ok", "warn")):
                continue
            dyn_node = rec.get("door_node") or rec.get("drawer_node")
            dyn_objs = [o for o in bpy.data.objects
                        if o.type == "MESH"
                        and o.name.startswith(f"{dyn_node}__mesh")]
            if not dyn_objs:
                continue
            n_checked += 1
            worst_gap, worst_frame = 0.0, None
            for f in frames:
                scene.frame_set(f)
                deps.update()
                g = min_gap(dyn_objs)
                if g > worst_gap:
                    worst_gap, worst_frame = g, f
            if worst_gap > self._PROX:
                item.status = (f"warn: detaches — gap {worst_gap*1000:.1f}mm "
                               f"at frame {worst_frame} (> 5mm)")
                n_detached += 1
        scene.frame_set(prev_frame)
        overlay.rebuild_from_props(props)
        _log(props, "WARN" if n_detached else "INFO",
             f"connectivity: {n_detached}/{n_checked} joint(s) detach "
             f"(threshold 5mm, {len(frames)} frames)", self)
        return {"FINISHED"}


class GRAPH_OT_validate_motion(Operator):
    """Collision sweep: flags joints that gain mesh overlaps during the
    open cycle relative to the rest pose (heuristic triage — the precise
    per-hinge validation runs inside every install)"""
    bl_idname = "graph_func.validate_motion"
    bl_label = "Collision Test"

    def execute(self, context):
        props = _props(context)
        state = _state_with_objs(props)
        if state is None:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        deps = context.evaluated_depsgraph_get()
        scene = context.scene
        prev_frame = scene.frame_current

        static_names = {f"{nid}__mesh0" for nid in state.get("static_ids", [])}

        rows = [( "hinges", it, it.joint_name,
                  state and _find_record(state, "hinges", it.joint_name))
                for it in props.hinges]
        rows += [("rails", it, it.joint_name,
                  _find_record(state, "rails", it.joint_name))
                 for it in props.rails]

        statics = [o for o in bpy.data.objects
                   if o.type == "MESH" and o.name in static_names]
        # Static parts never animate: build their world trees once.
        static_trees = [_world_bvh(s) for s in statics]

        n_warn = 0
        frames = list(range(0, _ANIM_FRAME_END + 1, 10))
        for kind, item, joint, rec in rows:
            if rec is None or not item.status.startswith(("ok", "warn")):
                continue
            dyn_node = rec.get("door_node") or rec.get("drawer_node")
            dyn_objs = [o for o in bpy.data.objects
                        if o.type == "MESH"
                        and o.name.startswith(f"{dyn_node}__mesh")]
            if not dyn_objs or not static_trees:
                continue
            baseline = None
            worst = 0
            for f in frames:
                scene.frame_set(f)
                deps.update()
                n_pairs = 0
                for d in dyn_objs:
                    td = _world_bvh(d)
                    for st in static_trees:
                        n_pairs += len(td.overlap(st))
                if baseline is None:
                    baseline = n_pairs
                else:
                    worst = max(worst, n_pairs - baseline)
            if worst > max(20, (baseline or 0)):
                item.status = f"warn: +{worst} overlap pairs during swing"
                n_warn += 1
        scene.frame_set(prev_frame)
        overlay.rebuild_from_props(props)
        _log(props, "WARN" if n_warn else "INFO",
             f"validate: {n_warn} joint(s) flagged "
             f"({len(rows)} checked, {len(frames)} frames each)", self)
        return {"FINISHED"}


# ────────────────────────────────────────── selection sync / eyedropper ────

class GRAPH_OT_select_row_objects(Operator):
    """Select the scene objects belonging to the active list row"""
    bl_idname = "graph_func.select_row_objects"
    bl_label = "Select In Viewport"

    row_kind: bpy.props.EnumProperty(items=[("HINGE", "Hinge", ""),
                                            ("RAIL", "Rail", ""),
                                            ("HANDLE", "Handle", "")],
                                     default="HINGE")

    def execute(self, context):
        props = _props(context)
        targets = []
        if self.row_kind == "HINGE" and props.hinges:
            it = props.hinges[max(0, min(props.active_hinge_idx,
                                         len(props.hinges) - 1))]
            targets = [o for o in bpy.data.objects
                       if o.name.startswith(f"{it.door_node}__mesh")
                       or it.joint_name in o.name]
        elif self.row_kind == "RAIL" and props.rails:
            it = props.rails[max(0, min(props.active_rail_idx,
                                        len(props.rails) - 1))]
            targets = [o for o in bpy.data.objects
                       if o.name.startswith(f"{it.drawer_node}__mesh")
                       or o.get("__rail_joint") == it.joint_name]
        elif self.row_kind == "HANDLE" and props.handle_groups:
            it = props.handle_groups[max(0, min(props.active_hgroup_idx,
                                                len(props.handle_groups) - 1))]
            targets = [o for o in bpy.data.objects
                       if o.name.startswith(f"{it.parent_id}__mesh")
                       or o.get(_HANDLE_PARENT_TAG) == it.parent_id
                       or o.get("__graph_handle_parent") == it.parent_id]
        if not targets:
            return {"CANCELLED"}
        try:
            bpy.ops.object.select_all(action="DESELECT")
            for o in targets:
                o.select_set(True)
            context.view_layer.objects.active = targets[0]
        except Exception:
            pass
        return {"FINISHED"}


class GRAPH_OT_pick_row_from_object(Operator):
    """Activate the list row that owns the selected object (eyedropper)"""
    bl_idname = "graph_func.pick_row_from_object"
    bl_label = "Pick Row From Selection"

    def execute(self, context):
        props = _props(context)
        obj = context.active_object
        if obj is None:
            _log(props, "WARN", "select a part in the viewport first", self)
            return {"CANCELLED"}
        node = obj.name.split("__mesh")[0] if "__mesh" in obj.name else None
        with gp.suspend_updates():
            for i, it in enumerate(props.hinges):
                if it.door_node == node or it.joint_name in obj.name:
                    props.active_hinge_idx = i
                    _log(props, "INFO", f"hinge row: {it.joint_name}", self)
                    return {"FINISHED"}
            for i, it in enumerate(props.rails):
                if it.drawer_node == node or \
                        obj.get("__rail_joint") == it.joint_name:
                    props.active_rail_idx = i
                    _log(props, "INFO", f"rail row: {it.joint_name}", self)
                    return {"FINISHED"}
            for i, it in enumerate(props.handle_groups):
                if it.parent_id == node or \
                        obj.get(_HANDLE_PARENT_TAG) == it.parent_id:
                    props.active_hgroup_idx = i
                    _log(props, "INFO", f"handle group: {it.parent_id}", self)
                    return {"FINISHED"}
        _log(props, "WARN", f"'{obj.name}' matches no joint row", self)
        return {"CANCELLED"}


# ───────────────────────────────────────────── finalize / export / misc ────

class GRAPH_OT_export_manifest(Operator):
    """Write a JSON manifest of every joint row and its install status"""
    bl_idname = "graph_func.export_manifest"
    bl_label = "Export Manifest"

    def execute(self, context):
        props = _props(context)
        if not props.loaded:
            _log(props, "ERR", "no graph loaded", self)
            return {"CANCELLED"}
        man = {
            "model": props.model_label,
            "pred_path": props.pred_path,
            "hinges": [{"joint": it.joint_name, "category": it.category,
                        "variant": it.variant, "status": it.status,
                        "detail": it.detail} for it in props.hinges],
            "rails": [{"joint": it.joint_name, "variant": it.variant,
                       "status": it.status} for it in props.rails],
            "handles": [{"parent": it.parent_id, "kind": it.parent_kind,
                         "n_new": it.n_new, "n_anchored": it.n_anchored,
                         "style": it.style, "status": it.status}
                        for it in props.handle_groups],
        }
        base = (Path(bpy.data.filepath).with_suffix("")
                if bpy.data.filepath
                else Path(bpy.path.abspath(props.pred_path)).with_suffix(""))
        out = base.parent / f"{base.stem}.manifest.json"
        out.write_text(json.dumps(man, indent=2))
        _log(props, "INFO", f"manifest written: {out}", self)
        return {"FINISHED"}


class GRAPH_OT_commit_finalize(Operator):
    """Commit interior panels, purge helper/temp data, clean orphans"""
    bl_idname = "graph_func.commit_finalize"
    bl_label = "Commit & Clean"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        n_panels = len(iscene.get_interior_panel_objects())
        if n_panels:
            iscene.commit_interior_panels()
            props.interior_detected_count = 0
        n_tmp = 0
        for o in list(bpy.data.objects):
            if o.name.startswith(("__validation_temp__",
                                  "__hinge_validation_temp__",
                                  "__decomp_", "__top_detect_tmp")):
                bpy.data.objects.remove(o, do_unlink=True)
                n_tmp += 1
        for mesh in list(bpy.data.meshes):
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        _log(props, "INFO",
             f"committed {n_panels} interior panel(s), removed {n_tmp} "
             f"temp object(s), purged orphan meshes", self)
        return {"FINISHED"}


class GRAPH_OT_clear_scene(Operator):
    """Wipe everything from the scene so a new graph can be loaded"""
    bl_idname = "graph_func.clear_scene"
    bl_label = "Clear Scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = _props(context)
        pg.clear_scene()
        with gp.suspend_updates():
            props.hinges.clear()
            props.rails.clear()
            props.handle_groups.clear()
        props.loaded = False
        props.state_json = ""
        props.model_label = ""
        props.n_free_nodes = 0
        props.free_summary = ""
        props.interior_detected_count = 0
        overlay.rebuild_from_props(props)
        _log(props, "INFO", "scene cleared", self)
        return {"FINISHED"}


class GRAPH_OT_clear_log(Operator):
    """Clear the status log"""
    bl_idname = "graph_func.clear_log"
    bl_label = "Clear Log"

    def execute(self, context):
        props = _props(context)
        props.log.clear()
        props.status_message = ""
        return {"FINISHED"}


_classes = (
    GRAPH_OT_refresh_sources,
    GRAPH_OT_load_model,
    GRAPH_OT_load_graph,
    GRAPH_OT_rebuild_records,
    GRAPH_OT_attach_result,
    GRAPH_OT_apply_all_hinges,
    GRAPH_OT_reapply_one_hinge,
    GRAPH_OT_cycle_hinge_variant,
    GRAPH_OT_apply_all_rails,
    GRAPH_OT_reapply_one_rail,
    GRAPH_OT_cycle_rail_variant,
    GRAPH_OT_add_group_handles,
    GRAPH_OT_add_all_handles,
    GRAPH_OT_remove_group_handles,
    GRAPH_OT_cycle_handle_style,
    GRAPH_OT_detect_top,
    GRAPH_OT_add_predicted_top,
    GRAPH_OT_detect_interior,
    GRAPH_OT_instantiate_interior,
    GRAPH_OT_generate_interior_panels,
    GRAPH_OT_clear_interior_panels,
    GRAPH_OT_clear_compartment_boxes,
    GRAPH_OT_commit_interior,
    GRAPH_OT_add_support_blocks,
    GRAPH_OT_auto_functionalize,
    GRAPH_OT_preview_motion,
    GRAPH_OT_jump_max_extent,
    GRAPH_OT_solo_joint,
    GRAPH_OT_validate_motion,
    GRAPH_OT_test_connectivity,
    GRAPH_OT_select_row_objects,
    GRAPH_OT_pick_row_from_object,
    GRAPH_OT_export_manifest,
    GRAPH_OT_commit_finalize,
    GRAPH_OT_clear_scene,
    GRAPH_OT_clear_log,
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
