"""Load a predicted graph + its input PNM graph + part meshes into Blender.

Factored out of `install_from_pred_decompose_panel.py` for the addon's
graph-driven workflow:
  1. read pred + input JSONs
  2. clear scene, import each anchored slot's part mesh (.obj/.ply), recolour
     by predicted material (project palette)
  3. build hinge / rail / body OBB records keyed on PREDICTED graph topology
     but with geometry from the INPUT (GT) OBBs (predicted OBBs are bulky)
  4. override each part's OBB with its mesh's world AABB (tighter than the
     labelled OBB on real meshes)

The result (`LoadedGraph`) carries:
  hinge_records   — one dict per door joint (with door_node, panel_node,
                    hinge_axis, joint_origin, panel_idx, OBBs, swing limit)
  rail_records    — one dict per drawer joint (slide_axis_world, ...)
  body_obbs       — `_PickerOBB` for every static body part (panel_idx into
                    this list points the hinge to the right body box)
  pred_id_to_obj  — node id → bpy mesh object (panel meshes, doors, drawers,
                    handles, top-panel placeholders)
  attached_pairs  — list of (handle_id, parent_id) for handle attachments
  pred / input    — the raw JSONs (kept for is_top_panel_node etc.)
  input_ids       — slot index → input PNM node id

This module deliberately stays bpy-free for the records logic so it can be
unit-tested by a non-Blender Python; the import & material code is bpy-only
and gated to be called only from operators.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import bpy
import numpy as np
from mathutils import Vector


# ----------------------------------------------------------------- palette --

# Predicted-material → RGBA. Mirror of MATERIAL_COLOR in
# install_from_pred_decompose_panel.py so visual identity matches that script.
MATERIAL_COLOR = {
    "handle":       (0.85, 0.42, 0.46, 1),  "shelf":        (0.96, 0.78, 0.42, 1),
    "side panel":   (0.35, 0.57, 0.78, 1),  "back panel":   (0.50, 0.68, 0.85, 1),
    "bottom panel": (0.61, 0.72, 0.78, 1),  "top panel":    (0.78, 0.61, 0.36, 1),
    "face frame":   (0.55, 0.49, 0.69, 1),  "leg":          (0.42, 0.35, 0.27, 1),
    "door":         (0.91, 0.60, 0.29, 1),  "drawer":       (0.72, 0.36, 0.36, 1),
    "divider":      (0.49, 0.69, 0.54, 1),  "rail":         (0.23, 0.70, 0.45, 1),
    "hinge":        (0.91, 0.28, 0.33, 1),  "bar":          (0.63, 0.36, 0.61, 1),
    "countertop":   (0.83, 0.63, 0.36, 1),  "misc":         (0.55, 0.55, 0.55, 1),
    "unknown":      (0.75, 0.75, 0.75, 1),
    # Slot-id underscored variants (some pred JSONs use them):
    "top_panel":    (0.78, 0.61, 0.36, 1),
    "side_panel":   (0.35, 0.57, 0.78, 1),
    "back_panel":   (0.50, 0.68, 0.85, 1),
    "bottom_panel": (0.61, 0.72, 0.78, 1),
    "face_frame":   (0.55, 0.49, 0.69, 1),
    "mid_panel":    (0.45, 0.55, 0.72, 1),
}

DYNAMIC_MATS = {"door", "drawer"}


def mat_base(mat: str) -> str:
    """Strip Blender's `.001` duplicate suffix and normalise to lowercase."""
    return re.sub(r"\.\d+$", "", (mat or "")).strip().lower()


def material_for(mat_name: str, alpha: float = 1.0) -> "bpy.types.Material":
    """Get-or-create a Principled BSDF material in the project palette.

    Reuses a single material per (mat, alpha) pair so the .blend stays compact.
    """
    base = mat_base(mat_name)
    rgba = MATERIAL_COLOR.get(base, MATERIAL_COLOR["unknown"])
    if alpha < 1.0:
        rgba = (rgba[0], rgba[1], rgba[2], alpha)
    pname = f"mat_pred_{base.replace(' ', '_')}_{int(alpha * 100)}"
    if pname in bpy.data.materials:
        return bpy.data.materials[pname]
    m = bpy.data.materials.new(pname)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        if alpha < 1.0:
            try:
                bsdf.inputs["Alpha"].default_value = alpha
            except KeyError:
                pass
    if alpha < 1.0:
        m.blend_method = "BLEND"
    return m


# --------------------------------------------------------------- geometry --

# Face encoding (matches build_motion_labels_v3.FACE2IDX):
#   0:min_X 1:max_X 2:min_Y 3:max_Y 4:min_Z 5:max_Z
def face_to_axis_sign(face_idx: int):
    if face_idx is None or face_idx < 0 or face_idx > 5:
        return None, None
    return face_idx // 2, (-1.0 if face_idx % 2 == 0 else +1.0)


def face_to_unit_axis(face_idx: int) -> Optional[np.ndarray]:
    ax, sign = face_to_axis_sign(face_idx)
    if ax is None:
        return None
    v = np.zeros(3)
    v[ax] = sign
    return v


@dataclass
class PickerOBB:
    """Axis-aligned OBB shim. The canonicalised predicted graphs always have
    R=I, so we hard-code that here for speed."""
    center: np.ndarray
    size: np.ndarray  # full extents (2 * half)
    R: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def half(self):
        return self.size * 0.5

    def volume(self) -> float:
        return float(np.prod(self.size))

    def thinnest_axis_idx(self) -> int:
        return int(np.argmin(self.size))


def _door_pin_point(door_center: np.ndarray, door_half: np.ndarray,
                    hinge_border: int) -> np.ndarray:
    """3-D world-space pivot on the door's `hinge_border` face. Pushed inward
    along the thin axis so the pin lies on the door's inside edge."""
    ax, sign = face_to_axis_sign(hinge_border)
    if ax is None:
        return door_center.copy()
    p = door_center.copy()
    p[ax] = door_center[ax] + sign * door_half[ax]
    thin_idx = int(np.argmin(door_half))
    if thin_idx != ax:
        p[thin_idx] = door_center[thin_idx] - door_half[thin_idx]
    return p


# ------------------------------------------------------- mesh import (bpy) --

def clear_scene():
    """Remove every object + collection so a fresh load starts from empty.

    Data-level removal on purpose: operator select+delete skips objects
    hidden via hide_set() (e.g. carve cutters), which would then leak into
    the next loaded model."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def _import_obj_or_ply(path: Path, name: str, material_name: str
                       ) -> Optional["bpy.types.Object"]:
    """Import an .obj or .ply mesh in the project's axis convention and
    recolour it to the predicted material. Returns the joined mesh."""
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    try:
        if ext == ".obj":
            bpy.ops.wm.obj_import(filepath=str(path),
                                  forward_axis='NEGATIVE_Y', up_axis='Z')
        elif ext == ".ply":
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            print(f"[load_graph] unsupported mesh ext: {path}")
            return None
    except Exception as e:
        print(f"[load_graph] mesh import {path.name}: ERR {e}")
        return None
    new = [o for o in bpy.data.objects if o not in before]
    if not new:
        return None
    if len(new) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in new:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = new[0]
    obj.name = name
    obj.data.materials.clear()
    obj.data.materials.append(material_for(material_name))
    return obj


def _make_pred_bbox(name: str, center, half, material_name: str,
                    alpha: float = 0.45) -> "bpy.types.Object":
    """Translucent box mesh from a predicted OBB. Used to visualise free-slot
    adds (model-hallucinated nodes with no input geometry)."""
    e = np.asarray(half, float)
    verts = [
        (-e[0], -e[1], -e[2]), (+e[0], -e[1], -e[2]),
        (+e[0], +e[1], -e[2]), (-e[0], +e[1], -e[2]),
        (-e[0], -e[1], +e[2]), (+e[0], -e[1], +e[2]),
        (+e[0], +e[1], +e[2]), (-e[0], +e[1], +e[2]),
    ]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5),
             (0, 4, 5, 1), (2, 6, 7, 3),
             (1, 5, 6, 2), (0, 3, 7, 4)]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = tuple(np.asarray(center, float))
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material_for(material_name, alpha=alpha))
    return obj


def _world_aabb(obj) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if obj is None or obj.type != "MESH" or obj.data is None:
        return None
    if len(obj.data.vertices) == 0:
        return None
    mw = obj.matrix_world
    lo = [float("inf")] * 3
    hi = [-float("inf")] * 3
    for v in obj.data.vertices:
        w = mw @ v.co
        for k in range(3):
            if w[k] < lo[k]:
                lo[k] = w[k]
            if w[k] > hi[k]:
                hi[k] = w[k]
    return np.asarray(lo), np.asarray(hi)


def _aabb_to_centerhalf(lo, hi) -> tuple[list, list]:
    c = ((lo + hi) * 0.5).tolist()
    h = np.maximum((hi - lo) * 0.5, 1e-4).tolist()
    return c, h


# ----------------------------------------------------------- record builder --

def _gt_obb_from_input(pred_node: dict, input_nodes: dict, input_ids: list):
    """(center, half) of the INPUT (GT) OBB for a prediction node, via the
    slot mapping. Returns (None, None) for free-slot nodes."""
    slot = int(pred_node.get("slot", -1))
    if slot < 0 or slot >= len(input_ids):
        return None, None
    pnm_id = input_ids[slot]
    in_obb = input_nodes.get(pnm_id, {}).get("obb", {})
    c = in_obb.get("center"); h = in_obb.get("half")
    if c is None or h is None:
        return None, None
    return np.asarray(c, float), np.asarray(h, float)


def _aggregate_body_centroid(body_obbs) -> np.ndarray:
    if not body_obbs:
        return np.zeros(3)
    centers = np.stack([np.asarray(b.center, float) for b in body_obbs])
    return centers.mean(axis=0)


def build_records(pred: dict, inp: dict
                  ) -> tuple[list[dict], list[dict], list[PickerOBB],
                             dict[str, int], list[str]]:
    """Produce (hinge_records, rail_records, body_obbs, panel_id_to_idx,
    static_ids) from a graph prediction.

    Geometry uses INPUT (GT) OBBs. Predicted graph supplies topology +
    face-encoded motion attributes (`hinge_axis_signed`, `hinge_border`,
    `rail_axis_signed`).
    """
    nodes = pred["nodes"]
    edges = pred.get("edges", [])
    input_nodes = inp["nodes"]
    input_ids = list(input_nodes.keys())

    # 1. Body OBBs = every static (non-dynamic) anchored node, GT geometry.
    static_ids: list[str] = []
    body_obbs: list[PickerOBB] = []
    panel_id_to_idx: dict[str, int] = {}
    for nid, n in nodes.items():
        mat = mat_base(n.get("material", ""))
        if mat in DYNAMIC_MATS or mat == "handle":
            continue
        c, h = _gt_obb_from_input(n, input_nodes, input_ids)
        if c is None:
            continue
        body_obbs.append(PickerOBB(center=c, size=2 * h))
        panel_id_to_idx[nid] = len(body_obbs) - 1
        static_ids.append(nid)

    body_centroid = _aggregate_body_centroid(body_obbs)

    # 2. Motion records from hinge/rail edges.
    hinge_records: list[dict] = []
    rail_records: list[dict] = []
    for eidx, e in enumerate(edges):
        kind = e.get("kind")
        if kind not in ("hinge", "rail"):
            continue
        s, d = e["src"], e["dst"]
        sm = mat_base(nodes.get(s, {}).get("material", ""))
        dm = mat_base(nodes.get(d, {}).get("material", ""))
        if sm in DYNAMIC_MATS and dm not in DYNAMIC_MATS:
            dyn_id, sta_id = s, d
        elif dm in DYNAMIC_MATS and sm not in DYNAMIC_MATS:
            dyn_id, sta_id = d, s
        else:
            continue

        dyn_center, dyn_half = _gt_obb_from_input(nodes[dyn_id], input_nodes, input_ids)
        sta_center, sta_half = _gt_obb_from_input(nodes[sta_id], input_nodes, input_ids)
        if dyn_center is None or sta_center is None:
            continue

        if kind == "hinge":
            axis_idx = e.get("hinge_axis_signed", -1)
            border_idx = e.get("hinge_border", -1)
            if axis_idx < 0:
                continue
            axis_world = face_to_unit_axis(axis_idx)
            pin_world = _door_pin_point(dyn_center, dyn_half, border_idx)
            panel_idx = panel_id_to_idx.get(sta_id, -1)
            hinge_records.append({
                "joint":        f"hinge_{eidx:02d}_{dyn_id}",
                "door_node":    dyn_id,
                "panel_node":   sta_id,
                "panel_idx":    panel_idx,
                "hinge_axis":   axis_world.tolist(),
                "joint_origin": pin_world.tolist(),
                "limit":        [0.0, float(np.pi / 2)],
                "door_center":  dyn_center.tolist(),
                "door_size":    (2 * dyn_half).tolist(),
                "panel_center": sta_center.tolist(),
                "panel_size":   (2 * sta_half).tolist(),
                "hinge_border": int(border_idx),
                "axis_idx":     int(axis_idx),
            })
        else:  # rail
            axis_idx = e.get("rail_axis_signed", -1)
            if axis_idx < 0:
                continue
            slide_world = face_to_unit_axis(axis_idx)
            # Canonicalize OUTWARD: away from body centroid
            body_to_drawer = dyn_center - body_centroid
            if float(slide_world @ body_to_drawer) < 0.0:
                slide_world = -slide_world
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(world_up, slide_world)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1.0, 0.0, 0.0])
                right = right - slide_world * float(right @ slide_world)
            right = right / max(np.linalg.norm(right), 1e-12)
            width_axis = right
            height_axis = np.cross(slide_world, width_axis)
            height_axis = height_axis / max(np.linalg.norm(height_axis), 1e-12)

            def proj_half(a):
                return float(np.sum(np.abs(a) * dyn_half))
            half_slide = proj_half(slide_world)
            half_width = proj_half(width_axis)
            half_height = proj_half(height_axis)

            rail_records.append({
                "joint":             f"rail_{eidx:02d}_{dyn_id}",
                "drawer_node":       dyn_id,
                "panel_node":        sta_id,
                "drawer_center":     dyn_center.tolist(),
                "drawer_size":       (2 * dyn_half).tolist(),
                "slide_origin":      dyn_center.tolist(),
                "slide_axis_world":  slide_world.tolist(),
                "width_axis_world":  width_axis.tolist(),
                "height_axis_world": height_axis.tolist(),
                "half_slide_m":      half_slide,
                "half_width_m":      half_width,
                "half_height_m":     half_height,
                "limit":             [0.0, 0.2],
            })

    # Dedupe rails: install once per (drawer, slide axis). The graph fix
    # keeps both rail edges; install_rails_for_drawer builds left+right
    # from one record. Keep the first per drawer node.
    seen = set()
    deduped_rails = []
    for r in rail_records:
        if r["drawer_node"] in seen:
            continue
        seen.add(r["drawer_node"])
        deduped_rails.append(r)

    # Dedupe hinges: keep first per (door, axis, panel). Multi-panel
    # multi-hinge cases install both because the panels differ.
    seen = set()
    deduped_hinges = []
    for r in hinge_records:
        key = (r["door_node"], tuple(r["hinge_axis"]), r["panel_node"])
        if key in seen:
            continue
        seen.add(key)
        deduped_hinges.append(r)

    return deduped_hinges, deduped_rails, body_obbs, panel_id_to_idx, static_ids


# ------------------------------------------------------------- top-level API --

# Free-slot (completion) classes the pipeline can materialise. Anything
# else the model fires is a hallucination we can only drop or report.
COMPLETION_MATS = {"handle", "top panel", "top_panel", "countertop",
                   "shelf", "divider"}


@dataclass
class LoadedGraph:
    """Everything subsequent operators need after a graph load."""
    pred: dict
    input: dict
    input_ids: list[str]
    pred_id_to_obj: dict   # node id → bpy mesh object
    hinge_records: list[dict]
    rail_records: list[dict]
    body_obbs: list[PickerOBB]
    panel_id_to_idx: dict[str, int]
    static_ids: list[str]
    attached_pairs: list[tuple[str, str]]   # (handle_id, parent_id)
    pred_path: Path
    input_path: Path
    mesh_dir: Path
    free_nodes: list[dict] = field(default_factory=list)   # fired completion nodes
    warnings: list[str] = field(default_factory=list)


def resolve_mesh_lookup(mesh_dir: Path, input_nodes: dict) -> dict[str, Path]:
    """Map each input node id to its mesh file. Supports two layouts:

      PNM:    <mesh_dir>/parts.json + <mesh_dir>/objs/<file>.obj
      fur_*:  <mesh_dir>/<node_id>.{ply,obj}
    """
    parts_json = mesh_dir / "parts.json"
    objs_dir = mesh_dir / "objs"
    if parts_json.exists() and objs_dir.is_dir():
        parts = json.loads(parts_json.read_text())
        return {p["id"]: objs_dir / Path(p["obj_file"]).name
                for p in parts.get("parts", [])}
    out: dict[str, Path] = {}
    for nid in input_nodes.keys():
        for ext in (".ply", ".obj"):
            cand = mesh_dir / f"{nid}{ext}"
            if cand.exists():
                out[nid] = cand
                break
    return out


def load_graph_into_scene(pred_path: Path, input_path: Path, mesh_dir: Path,
                          drop_hallucinated: bool = True
                          ) -> LoadedGraph:
    """Clear the scene, import every anchored slot's mesh from `mesh_dir`,
    visualise free-slot adds as translucent bboxes, build motion records,
    and tighten every OBB to its mesh's world AABB.

    Caller (an operator) handles bpy context; this function expects to be
    invoked from one.
    """
    pred = json.loads(pred_path.read_text())
    inp = json.loads(input_path.read_text())
    input_ids = list(inp["nodes"].keys())
    mesh_lookup = resolve_mesh_lookup(mesh_dir, inp["nodes"])

    # Note: do NOT call wm.read_factory_settings here. It invalidates the
    # caller-held PointerProperty references (operator's `props` becomes a
    # dangling pointer → segfault on next write). clear_scene() does the
    # job without resetting Blender's global state.
    clear_scene()

    pred_id_to_obj: dict[str, bpy.types.Object] = {}
    free_nodes: list[dict] = []
    warnings: list[str] = []
    n_dropped_hall = 0
    for nid, n in pred["nodes"].items():
        slot = int(n.get("slot", -1))
        mat = n.get("material", "unknown")
        if 0 <= slot < len(input_ids):
            pnm_id = input_ids[slot]
            obj_path = mesh_lookup.get(pnm_id)
            if obj_path is None or not obj_path.exists():
                warnings.append(f"no mesh file for anchored node {nid} "
                                f"(input id {pnm_id})")
                continue
            obj = _import_obj_or_ply(obj_path, name=f"{nid}__mesh0",
                                     material_name=mat)
            if obj is not None:
                pred_id_to_obj[nid] = obj
            else:
                warnings.append(f"mesh import failed for {nid} ({obj_path.name})")
        else:
            # Free-slot add (a completion node not present in the input).
            # No bbox placeholder is drawn — handles / tops / interior are
            # materialised by their operators from the record instead. The
            # node is kept in `free_nodes` so the UI can list what fired.
            if mat_base(mat) in COMPLETION_MATS:
                free_nodes.append({"id": nid, "material": mat, "slot": slot,
                                   "obb": n.get("obb", {})})
            elif drop_hallucinated:
                n_dropped_hall += 1
                warnings.append(f"dropped hallucinated free-slot node {nid} "
                                f"({mat})")
            else:
                free_nodes.append({"id": nid, "material": mat, "slot": slot,
                                   "obb": n.get("obb", {})})

    bpy.context.view_layer.update()
    print(f"[load_graph] imported {len(pred_id_to_obj)} anchored nodes; "
          f"{len(free_nodes)} fired completion node(s); "
          f"dropped {n_dropped_hall} hallucinated")
    for w in warnings:
        print(f"[load_graph] WARN: {w}")

    hinges, rails, body_obbs, panel_id_to_idx, static_ids = build_records(pred, inp)
    print(f"[load_graph] records: hinges={len(hinges)}, rails={len(rails)}, "
          f"body_boxes={len(body_obbs)}")

    # Tighten OBBs to mesh world-AABBs (the labelled OBBs are sometimes
    # bulky; the actual mesh extent is what insert_hinges_for_joint should
    # see). Rebuild panel_idx → list-position from id ordering of body parts.
    new_body_obbs: list[PickerOBB] = []
    new_panel_idx_map: dict[str, int] = {}
    for nid, n in pred["nodes"].items():
        mat = mat_base(n.get("material", ""))
        if mat in DYNAMIC_MATS or mat == "handle":
            continue
        obj = pred_id_to_obj.get(nid)
        ab = _world_aabb(obj) if obj is not None else None
        if ab is None:
            continue
        c, h = _aabb_to_centerhalf(*ab)
        new_body_obbs.append(PickerOBB(center=np.asarray(c), size=2 * np.asarray(h)))
        new_panel_idx_map[nid] = len(new_body_obbs) - 1
    if new_body_obbs:
        body_obbs = new_body_obbs
        panel_id_to_idx = new_panel_idx_map

    for rec in hinges:
        d_obj = pred_id_to_obj.get(rec["door_node"])
        p_obj = pred_id_to_obj.get(rec["panel_node"])
        d_ab = _world_aabb(d_obj) if d_obj is not None else None
        p_ab = _world_aabb(p_obj) if p_obj is not None else None
        if d_ab is not None:
            c, h = _aabb_to_centerhalf(*d_ab)
            rec["door_center"] = c
            rec["door_size"] = (np.asarray(h) * 2).tolist()
            new_pin = _door_pin_point(np.asarray(c), np.asarray(h),
                                       int(rec.get("hinge_border", -1)))
            rec["joint_origin"] = new_pin.tolist()
        if p_ab is not None:
            c, h = _aabb_to_centerhalf(*p_ab)
            rec["panel_center"] = c
            rec["panel_size"] = (np.asarray(h) * 2).tolist()
        rec["panel_idx"] = panel_id_to_idx.get(rec["panel_node"], -1)

    # Collect handle attachments (handle.id, parent.id). Used by add-handle.
    attached_pairs: list[tuple[str, str]] = []
    for e in pred.get("edges", []):
        if e.get("kind") != "attached":
            continue
        s, d = e["src"], e["dst"]
        sm = mat_base(pred["nodes"].get(s, {}).get("material", ""))
        dm = mat_base(pred["nodes"].get(d, {}).get("material", ""))
        if sm == "handle" and dm in DYNAMIC_MATS:
            attached_pairs.append((s, d))
        elif dm == "handle" and sm in DYNAMIC_MATS:
            attached_pairs.append((d, s))

    # Auto-parent every anchored handle mesh to its predicted door/drawer
    # parent. Without this, when hinges/rails are installed later and the
    # door swings, the imported handle stays in place (un-parented). Mirrors
    # the `n_handle_existing` branch of install_from_pred_decompose_panel.
    n_handle_parented = 0
    for h_id, p_id in attached_pairs:
        h_obj = pred_id_to_obj.get(h_id)
        p_obj = pred_id_to_obj.get(p_id)
        if h_obj is None or p_obj is None:
            continue
        # Preserve world pose: parent with keep_transform semantics so the
        # imported handle doesn't snap to the door's origin.
        bpy.ops.object.select_all(action="DESELECT")
        h_obj.select_set(True)
        p_obj.select_set(True)
        bpy.context.view_layer.objects.active = p_obj
        try:
            bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
            n_handle_parented += 1
        except Exception as exc:
            print(f"[load_graph] failed to parent {h_id} → {p_id}: {exc}")
    if n_handle_parented:
        print(f"[load_graph] auto-parented {n_handle_parented} anchored "
              f"handle(s) to their predicted door/drawer parents")

    return LoadedGraph(
        pred=pred, input=inp, input_ids=input_ids,
        pred_id_to_obj=pred_id_to_obj,
        hinge_records=hinges, rail_records=rails,
        body_obbs=body_obbs, panel_id_to_idx=panel_id_to_idx,
        static_ids=static_ids, attached_pairs=attached_pairs,
        pred_path=pred_path, input_path=input_path, mesh_dir=mesh_dir,
        free_nodes=free_nodes, warnings=warnings,
    )


# ----------------------------------------------------- serialisation to scene --

def records_to_scene_state(loaded: LoadedGraph) -> dict:
    """Convert a LoadedGraph into a JSON-serializable dict for stashing on
    the scene (via a StringProperty). Numpy → list everywhere."""
    def _ob(o: PickerOBB):
        return {"center": np.asarray(o.center).tolist(),
                "size":   np.asarray(o.size).tolist()}
    return {
        "pred_path":  str(loaded.pred_path),
        "input_path": str(loaded.input_path),
        "mesh_dir":   str(loaded.mesh_dir),
        "input_ids":  loaded.input_ids,
        "hinges":     loaded.hinge_records,
        "rails":      loaded.rail_records,
        "body_obbs":  [_ob(o) for o in loaded.body_obbs],
        "panel_id_to_idx": loaded.panel_id_to_idx,
        "static_ids": loaded.static_ids,
        "attached_pairs": [list(p) for p in loaded.attached_pairs],
        "free_nodes": loaded.free_nodes,
        "warnings":   loaded.warnings,
    }


def records_from_scene_state(state: dict) -> dict:
    """Inverse of records_to_scene_state — restore numpy/PickerOBB structure
    so install_helpers can consume the dict directly."""
    out = dict(state)
    out["body_obbs"] = [PickerOBB(center=np.asarray(o["center"]),
                                   size=np.asarray(o["size"]))
                        for o in state.get("body_obbs", [])]
    out["attached_pairs"] = [tuple(p) for p in state.get("attached_pairs", [])]
    return out
