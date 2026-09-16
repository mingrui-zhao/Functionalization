"""Run install_one's hinge/rail pipeline driven by GRAPH PREDICTIONS instead
of URDF data.

Inputs (replacing parse_urdf_geometry / classify_all_*):
  --pred=<pred.json>      decoder output (slot{ii}_<material> nodes + edges)
  --input=<input.json>    PNM unfunctional graph (gives slot→PNM-node-id map)
  --mesh_dir=<...>        PNM labeled_normalized/<mid>/  (objs/ + parts.json)
  --hinge=<style>         interior|exterior|flat|auto  (auto = collision-first
                          C>F>P competition via install_policy_fpc;
                          omitting defaults to exterior)
  --rail=<style>          corner|center            (omit if no rails)
  --no-divider            suppress the supporting-divider fallback when a
                          rail's snap plane is too far from any cabinet wall
                          for a slab. Without this flag, a divider is
                          inscribed; with it, that rail just gets no support.
  --drop-hallucinated     drop predicted free-slot nodes (slot outside input
                          range) whose material is NOT top_panel or handle.
                          These would otherwise appear as translucent bbox
                          placeholders in the output .blend. handle/top_panel
                          free-slots are kept because PNM inputs are
                          structurally incomplete for those classes.
  --out=<...>             output .blend path

Adapter logic
-------------
The decoder outputs per-edge `hinge_axis_signed`/`rail_axis_signed`/`hinge_border`
in 0..5 face encoding (min_X, max_X, min_Y, max_Y, min_Z, max_Z) of the
DYNAMIC part's local frame. Predicted OBBs are world-aligned (R = I) by
construction (canonicalize_graphs stage), so the dynamic part's local frame
== world frame here, and we can decode each face index directly into a unit
world axis.

For each predicted hinge edge → emit a hinge "record" matching
classify_all_hinges output:
    door_obb   ← predicted dynamic-side OBB
    panel_obb  ← predicted static-side OBB (the mounting panel chosen by the
                 graph topology — graph edges already encode "which panel")
    hinge_axis ← unit world vec from `hinge_axis_signed`
    joint_origin ← world pin point on the hinge_border face of the door OBB
    panel_idx  ← index into the body-OBB list passed to insert_hinges_for_joint

Drawer record similarly mirrors classify_all_drawers output, with width/
height axes derived from slide_axis_world × world_up.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import bmesh  # type: ignore
import bpy  # type: ignore
import numpy as np
from mathutils import Vector, Matrix


_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE.parent))

from hinge.geometry import HingeType, OBB
from hinge.pipeline import insert_hinges_for_joint
from rail.carve import CARVE_PREFIX as RAIL_CARVE_PREFIX
from rail.pipeline import install_rails_for_drawer
from rail.support_block import process_current_scene as _add_support_blocks
# add_handle / add_top live next to this script and wrap the addon's
# Functionalization Suite logic for predicted-graph attachment.
from add_handle import add_handle_for_predicted_attachment
from add_top import add_top_from_body_meshes, is_top_panel_node
# HSSD's geometric joint classifier. Given (door OBB, panel OBB, axis, body
# center) it labels the joint as CASE_EXTERIOR / CASE_INTERIOR /
# CASE_TUCKIN_PARALLEL — the natural mount topologies. We reuse the picker
# OBB type alias since classify_door_panel takes that flavor (full sizes,
# not half_extents).
from mount_component_picker import (
    OBB as PickerOBB,
    PanelPick,
    classify_door_panel,
    aggregate_body_center,
    CASE_EXTERIOR, CASE_INTERIOR, CASE_TUCKIN_PARALLEL,
)


# Map HSSD's classification cases to the natural set of HingeTypes that
# physically mount on that geometry. One case can admit multiple types
# (TUCKIN_PARALLEL admits both flat and interior — see
# mount_component_picker.suggest_hinge_strategies for the rationale).
_CASE_TO_TYPES: dict[str, list[HingeType]] = {
    CASE_EXTERIOR:        [HingeType.EXTERIOR],
    CASE_TUCKIN_PARALLEL: [HingeType.FLAT, HingeType.INTERIOR],
    CASE_INTERIOR:        [HingeType.INTERIOR],
}


# ── template inventory ───────────────────────────────────────────────────────
TEMPLATE_DIR = _REPO / "annotated_mechanical_parts"
RAIL_TEMPLATES = {
    "corner": TEMPLATE_DIR / "sliding_rail_annotated.blend",
    "center": TEMPLATE_DIR / "sliding_rail_02_annotated.blend",
}
HINGE_TYPES = {
    "interior": HingeType.INTERIOR,
    "exterior": HingeType.EXTERIOR,
    "flat":     HingeType.FLAT,
}

_TEMP_VALIDATION_PREFIX = "__validation_temp__"


def _make_box_mesh_object(obb: OBB, name: str) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    e = obb.half_extents
    verts = [
        (-e[0], -e[1], -e[2]), (+e[0], -e[1], -e[2]),
        (+e[0], +e[1], -e[2]), (-e[0], +e[1], -e[2]),
        (-e[0], -e[1], +e[2]), (+e[0], -e[1], +e[2]),
        (+e[0], +e[1], +e[2]), (-e[0], +e[1], +e[2]),
    ]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5),
             (0, 4, 5, 1), (2, 6, 7, 3),
             (1, 5, 6, 2), (0, 3, 7, 4)]
    mesh.from_pydata(verts, [], faces); mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = tuple(obb.center)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Matrix((
        (float(obb.R[0, 0]), float(obb.R[0, 1]), float(obb.R[0, 2])),
        (float(obb.R[1, 0]), float(obb.R[1, 1]), float(obb.R[1, 2])),
        (float(obb.R[2, 0]), float(obb.R[2, 1]), float(obb.R[2, 2])),
    )).to_quaternion()
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _delete_temp_validation_objects() -> None:
    to_remove = [o for o in bpy.data.objects
                 if o.name.startswith(_TEMP_VALIDATION_PREFIX)]
    for obj in to_remove:
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    bpy.context.view_layer.update()


# ── face encoding (matches build_motion_labels_v3.FACE2IDX) ─────────────────
# 0:min_X 1:max_X 2:min_Y 3:max_Y 4:min_Z 5:max_Z
def _face_to_axis_sign(face_idx: int):
    if face_idx is None or face_idx < 0 or face_idx > 5:
        return None, None
    return face_idx // 2, (-1.0 if face_idx % 2 == 0 else +1.0)


def _face_to_unit_axis(face_idx: int) -> Optional[np.ndarray]:
    ax, sign = _face_to_axis_sign(face_idx)
    if ax is None: return None
    v = np.zeros(3); v[ax] = sign
    return v


def _mat_base(mat: str) -> str:
    return re.sub(r"\.\d+$", "", (mat or "")).strip().lower()


DYNAMIC_MATS = {"door", "drawer"}

# Material → RGBA node palette.
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
}


def _material_for(mat_name: str, alpha: float = 1.0) -> bpy.types.Material:
    """Get-or-create a Principled BSDF material coloured per the project palette.
    Reuses the same Material across calls so the .blend stays compact."""
    base = _mat_base(mat_name)
    rgba = MATERIAL_COLOR.get(base, MATERIAL_COLOR["unknown"])
    if alpha < 1.0:
        rgba = (rgba[0], rgba[1], rgba[2], alpha)
    pname = f"mat_pred_{base.replace(' ', '_')}_{int(alpha*100)}"
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


@dataclass
class GraphGeom:
    """Shape-compatible enough with mount_component_picker.URDFGeometry that
    helpers like _aggregate_body_centroid and the body-OBB list operations
    work without modification."""
    body_boxes: list                # list of obj-with .center .size .R .volume()
    door_records: list[dict]
    drawer_records: list[dict]
    pnm_id_to_obj: dict             # map: pred node id → bpy mesh object


# ── PicketOBB shim — interface compatible with install_one's helpers ─────────
class _PickerOBB:
    """Stand-in for mount_component_picker.OBB with full extents `.size`,
    `.R`, `.center`, plus `.volume()` and `.thinnest_axis_idx()`. The
    canonicalised predicted graphs always have R=I, so we hard-code that
    here."""
    def __init__(self, center, half):
        self.center = np.asarray(center, dtype=float)
        self.size   = 2.0 * np.asarray(half, dtype=float)
        self.R      = np.eye(3)
    @property
    def half(self):
        return self.size / 2.0
    def volume(self):
        return float(np.prod(self.size))
    def thinnest_axis_idx(self):
        return int(np.argmin(self.size))


def _aggregate_body_centroid(body_obbs) -> np.ndarray:
    centers = np.stack([np.asarray(b.center, float) for b in body_obbs])
    return centers.mean(axis=0)


# ── Mesh import ─────────────────────────────────────────────────────────────

def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def _import_obj(path: Path, name: str, material: str = "unknown") -> Optional[bpy.types.Object]:
    """Import an .obj or .ply mesh in the project's axis convention (Z-up,
    -Y forward) and recolour it to the predicted material.

    Supports both PNM's .obj layout (forward=-Y, up=Z) and fur_*'s .ply
    layout from `part_geometries/` (already in world-space; just pass
    through). Returns the joined mesh renamed to `name`.
    """
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    try:
        if ext == ".obj":
            bpy.ops.wm.obj_import(filepath=str(path),
                                   forward_axis='NEGATIVE_Y', up_axis='Z')
        elif ext == ".ply":
            # PLY meshes from part_geometries/ are already in the same world
            # frame as the canonicalised graph OBBs.
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            print(f"  unsupported mesh ext: {path}")
            return None
    except Exception as e:
        print(f"  mesh import {path.name}: ERR {e}")
        return None
    new = [o for o in bpy.data.objects if o not in before]
    if not new: return None
    if len(new) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in new: o.select_set(True)
        bpy.context.view_layer.objects.active = new[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = new[0]
    obj.name = name
    # Replace any imported materials with the project palette colour for the
    # PREDICTED material (so visual identity tracks the model's prediction,
    # not whatever Blender pulled from the .obj file).
    obj.data.materials.clear()
    obj.data.materials.append(_material_for(material))
    return obj


def _make_pred_bbox_mesh(name: str, center, half, material: str,
                          alpha: float = 0.45) -> bpy.types.Object:
    """Create a translucent box mesh from a predicted OBB (used to visualise
    free-slot adds — nodes the model hallucinated that don't have any input
    geometry). Coloured by the predicted material."""
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
    mesh.from_pydata(verts, [], faces); mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = tuple(np.asarray(center, float))
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(_material_for(material, alpha=alpha))
    return obj


def _parent_keep_transform(child: bpy.types.Object, parent: bpy.types.Object):
    """Parent `child` to `parent` while preserving the child's world pose.
    Used to attach handles to their predicted door/drawer parent so they
    follow the dynamic part's motion."""
    bpy.ops.object.select_all(action="DESELECT")
    child.select_set(True)
    parent.select_set(True)
    bpy.context.view_layer.objects.active = parent
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)


# ── Build geometry from prediction ──────────────────────────────────────────

def _door_pin_point(door_center: np.ndarray, door_half: np.ndarray,
                    hinge_border: int) -> np.ndarray:
    """Pick a 3-D world-space pivot point on the predicted door's
    `hinge_border` face. Used as the `pin_hint` to insert_hinges_for_joint.
    Conservative fallback: door center."""
    ax, sign = _face_to_axis_sign(hinge_border)
    if ax is None:
        return door_center.copy()
    p = door_center.copy()
    p[ax] = door_center[ax] + sign * door_half[ax]
    # Push perpendicularly to the thin axis so the pin sits along the door's
    # inside edge (not on its outer face), giving the snapper a sensible
    # mounting point.
    thin_idx = int(np.argmin(door_half))
    if thin_idx != ax:
        p[thin_idx] = door_center[thin_idx] - door_half[thin_idx]
    return p


def _build_records(pred_graph: dict, input_graph: dict):
    """Produce hinge_records, drawer_records, body_boxes from a graph
    prediction. Geometry (centers, half-extents) is read from the INPUT
    GROUND-TRUTH OBBs (more accurate than predicted ones); the prediction
    only supplies graph topology + face-encoded motion attributes
    (hinge_axis_signed, hinge_border, rail_axis_signed).

    Body OBB list mirrors `URDFGeometry.body_boxes`: every static (non-
    door/drawer/handle) anchored node becomes a body box with `panel_idx`
    referencing its position in the list.
    """
    nodes = pred_graph["nodes"]
    edges = pred_graph.get("edges", [])
    input_nodes = input_graph["nodes"]
    input_node_ids = list(input_nodes.keys())
    n_input = len(input_node_ids)

    def _gt_obb(pred_id: str):
        """Return (center, half) of the GROUND-TRUTH (input PNM) OBB
        corresponding to a prediction node, via the slot mapping. Returns
        (None, None) for free-slot adds (no input geometry exists)."""
        nd = nodes.get(pred_id)
        if nd is None: return None, None
        slot = int(nd.get("slot", -1))
        if slot < 0 or slot >= n_input: return None, None
        pnm_id = input_node_ids[slot]
        in_obb = input_nodes.get(pnm_id, {}).get("obb", {})
        c = in_obb.get("center"); h = in_obb.get("half")
        if c is None or h is None: return None, None
        return np.asarray(c, float), np.asarray(h, float)

    # 1. Body OBBs = every static (non-dynamic) anchored node, GT geometry.
    static_ids: list[str] = []
    body_boxes: list[_PickerOBB] = []
    panel_id_to_idx: dict[str, int] = {}
    for nid, n in nodes.items():
        mat = _mat_base(n.get("material", ""))
        if mat in DYNAMIC_MATS or mat == "handle": continue
        c, h = _gt_obb(nid)
        if c is None: continue
        body_boxes.append(_PickerOBB(c, h))
        panel_id_to_idx[nid] = len(body_boxes) - 1
        static_ids.append(nid)

    body_centroid = (_aggregate_body_centroid(body_boxes)
                     if body_boxes else np.zeros(3))

    # 2. For every motion edge, extract a record.
    hinge_records: list[dict] = []
    drawer_records: list[dict] = []
    for eidx, e in enumerate(edges):
        kind = e["kind"]
        if kind not in ("hinge", "rail"): continue
        s, d = e["src"], e["dst"]
        sm = _mat_base(nodes.get(s, {}).get("material", ""))
        dm = _mat_base(nodes.get(d, {}).get("material", ""))
        # Identify dynamic vs static endpoint
        if sm in DYNAMIC_MATS and dm not in DYNAMIC_MATS:
            dyn_id, sta_id = s, d
        elif dm in DYNAMIC_MATS and sm not in DYNAMIC_MATS:
            dyn_id, sta_id = d, s
        else:
            continue   # malformed edge

        # Use GROUND-TRUTH (input PNM) geometry for both endpoints.
        # Predicted OBBs are inaccurate; we only use the prediction for
        # graph topology + face-encoded motion semantics.
        dyn_center, dyn_half = _gt_obb(dyn_id)
        sta_center, sta_half = _gt_obb(sta_id)
        if dyn_center is None or sta_center is None: continue

        if kind == "hinge":
            axis_idx = e.get("hinge_axis_signed", -1)
            border_idx = e.get("hinge_border",      -1)
            if axis_idx < 0:
                continue   # no motion label on this edge — skip rigging
            axis_world = _face_to_unit_axis(axis_idx)
            pin_world  = _door_pin_point(dyn_center, dyn_half, border_idx)
            panel_idx  = panel_id_to_idx.get(sta_id, -1)
            hinge_records.append({
                "joint":        f"hinge_{eidx:02d}_{dyn_id}",
                "child_link":   dyn_id,            # used for door_fragment_prefix
                "panel_idx":    panel_idx,
                "panel_node":   sta_id,
                "door_node":    dyn_id,
                "hinge_axis":   axis_world.tolist(),
                "joint_origin": pin_world.tolist(),
                "limit":        [0.0, float(np.pi/2)],
                "door_center":  dyn_center.tolist(),
                "door_size":    (2 * dyn_half).tolist(),
                "door_R":       np.eye(3).tolist(),
                "panel_center": sta_center.tolist(),
                "panel_size":   (2 * sta_half).tolist(),
                "panel_R":      np.eye(3).tolist(),
                # Stored so install can re-derive the pin after replacing
                # door OBB (e.g. with a mesh-derived AABB whose face
                # positions differ from the labeled OBB).
                "hinge_border": int(border_idx),
                "axis_idx":     int(axis_idx),
            })
        else:  # rail
            axis_idx = e.get("rail_axis_signed", -1)
            if axis_idx < 0: continue
            slide_world = _face_to_unit_axis(axis_idx)
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
            width_axis  = right
            height_axis = np.cross(slide_world, width_axis)
            height_axis = height_axis / max(np.linalg.norm(height_axis), 1e-12)
            # Project drawer half-extents onto canonical axes.
            # With R=I the projection along axis a is sum_i |a_i| * half_i
            def proj_half(a):
                return float(np.sum(np.abs(a) * dyn_half))
            half_slide  = proj_half(slide_world)
            half_width  = proj_half(width_axis)
            half_height = proj_half(height_axis)

            drawer_records.append({
                "joint":             f"rail_{eidx:02d}_{dyn_id}",
                "child_link":        dyn_id,
                "panel_node":        sta_id,
                "drawer_center":     dyn_center.tolist(),
                "drawer_size":       (2 * dyn_half).tolist(),
                "drawer_R":          np.eye(3).tolist(),
                "slide_origin":      dyn_center.tolist(),
                "slide_axis_world":  slide_world.tolist(),
                "width_axis_world":  width_axis.tolist(),
                "height_axis_world": height_axis.tolist(),
                "half_slide_m":      half_slide,
                "half_width_m":      half_width,
                "half_height_m":     half_height,
                "limit":             [0.0, 0.2],   # 20cm of stroke is typical
            })

    # Drawer records: dedupe identical (drawer_node, slide axis) so we don't
    # install the same rail twice. The graph fix preserves both rail edges
    # for completeness, but install_rails_for_drawer already builds left+right
    # from a SINGLE record. Keep the first record per drawer node.
    seen_drawers = set()
    deduped_drawers = []
    for r in drawer_records:
        if r["child_link"] in seen_drawers: continue
        seen_drawers.add(r["child_link"])
        deduped_drawers.append(r)

    # Hinges similarly: install once per (door, hinge axis) pair. Multi-panel
    # multi-hinge cases install both entries because they target different
    # mounting panels and produce independent hinge templates.
    seen_hinges = set()
    deduped_hinges = []
    for r in hinge_records:
        key = (r["door_node"], tuple(r["hinge_axis"]), r["panel_node"])
        if key in seen_hinges: continue
        seen_hinges.add(key)
        deduped_hinges.append(r)

    return GraphGeom(
        body_boxes=body_boxes,
        door_records=deduped_hinges,
        drawer_records=deduped_drawers,
        pnm_id_to_obj={},   # populated after mesh import
    ), panel_id_to_idx, static_ids


# ── Driver ──────────────────────────────────────────────────────────────────

def _user_args():
    return sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []


def _arg(p, d=None):
    for a in _user_args():
        if a.startswith(p): return a.split("=", 1)[1]
    return d


def _flag(name: str) -> bool:
    target = name.lstrip("-")
    return any(a.lstrip("-") == target for a in _user_args())


def main():
    required = ("--pred=", "--input=", "--mesh_dir=", "--out=")
    missing = [p for p in required if _arg(p) is None]
    if missing:
        print("ERROR: missing required argument(s): "
              + " ".join(m + "<path>" for m in missing))
        print("Usage: blender --background --python-exit-code 1 --python "
              "blender_tools/install_from_pred.py -- --pred=<pred.json> "
              "--input=<graph.json> --mesh_dir=<dir> --out=<out.blend> "
              "[--hinge=interior|exterior|flat|auto] [--rail=corner|center] "
              "[--drop-hallucinated] [--no-divider] [--carve] [--force-save]")
        sys.exit(1)
    pred_path  = Path(_arg("--pred="))
    input_path = Path(_arg("--input="))
    mesh_dir   = Path(_arg("--mesh_dir="))
    hinge_arg  = _arg("--hinge=", "exterior")  # default to exterior
    rail_arg   = _arg("--rail=",  "center")    # default to center-mount
    no_divider = _flag("no-divider")
    drop_hall  = _flag("drop-hallucinated")
    hinge_scale_s = _arg("--hinge_scale=", None)
    hinge_scale = float(hinge_scale_s) if hinge_scale_s is not None else None
    # Carving hinge-clearance pockets is opt-in for release: the boolean
    # DIFFERENCE can fail on thin panels whose face lies inside the carve
    # envelope (solver fallback adds volume instead of cutting), causing
    # rest-pose interpenetration. Pass --carve only for 3D-print prep.
    no_carve   = not _flag("carve")
    out_path   = Path(_arg("--out="))

    if not pred_path.exists():
        print(f"ERROR: pred not found: {pred_path}"); sys.exit(1)
    pred = json.loads(pred_path.read_text())
    inp  = json.loads(input_path.read_text())

    # Mesh lookup supports two layouts:
    #   PNM: <mesh_dir>/parts.json + <mesh_dir>/objs/<file>.obj
    #   fur_*: <mesh_dir>/<node_id>.ply  (from part_geometries/)
    parts_json = mesh_dir / "parts.json"
    objs_dir   = mesh_dir / "objs"
    if parts_json.exists() and objs_dir.is_dir():
        parts = json.loads(parts_json.read_text())
        part_obj_files = {p["id"]: objs_dir / Path(p["obj_file"]).name
                          for p in parts.get("parts", [])}
    else:
        # fur_* convention: each node id has a matching <id>.ply file (or
        # .obj) directly in the mesh_dir. Look up by node id.
        part_obj_files = {}
        for nid in inp["nodes"].keys():
            for ext in (".ply", ".obj"):
                cand = mesh_dir / f"{nid}{ext}"
                if cand.exists():
                    part_obj_files[nid] = cand
                    break

    input_ids = list(inp["nodes"].keys())
    geom, panel_id_to_idx, static_ids = _build_records(pred, inp)
    print(f"  body OBBs: {len(geom.body_boxes)}  hinges: {len(geom.door_records)}"
          f"  drawers: {len(geom.drawer_records)}")

    # ── Open empty scene & import meshes ────────────────────────────────────
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _clear_scene()

    pred_id_to_obj: dict[str, bpy.types.Object] = {}
    n_new_bbox = 0
    for nid, n in pred["nodes"].items():
        slot = int(n.get("slot", -1))
        mat  = n.get("material", "unknown")
        if 0 <= slot < len(input_ids):
            # Anchored slot → import the corresponding PNM .obj mesh, named
            # with the door_fragment_prefix convention so insert_hinges
            # finds it via `door_link + "__mesh"` matching. Coloured by the
            # PREDICTED material so visual identity matches the prediction.
            pnm_id = input_ids[slot]
            obj_path = part_obj_files.get(pnm_id)
            if obj_path is None or not obj_path.exists(): continue
            obj = _import_obj(obj_path, name=f"{nid}__mesh0", material=mat)
            if obj is not None:
                pred_id_to_obj[nid] = obj
        else:
            # Free-slot add (model "hallucinated" a node not in the input).
            # No PNM geometry exists; visualise it as a translucent coloured
            # bbox so it's obvious what the model added and where.
            if drop_hall and _mat_base(mat) not in ("top_panel", "top panel", "handle"):
                # Drop this hallucinated node (and any edges referencing it
                # are already filtered by _build_records' _gt_obb=None check).
                continue
            obb = n.get("obb", {})
            c, h = obb.get("center"), obb.get("half")
            if c is None or h is None: continue
            obj = _make_pred_bbox_mesh(name=f"{nid}__predbbox",
                                       center=c, half=h, material=mat)
            pred_id_to_obj[nid] = obj
            n_new_bbox += 1

    bpy.context.view_layer.update()
    print(f"  imported {len(pred_id_to_obj)} meshes ({n_new_bbox} free-slot bboxes)")

    # ── Build {parent_node_id -> (motion_axis, origin, kind)} from records ──
    # Handles get placed via add_handle using the predicted motion axis.
    parent_motion: dict[str, tuple] = {}
    for rec in geom.door_records:
        parent_motion[rec["door_node"]] = (
            np.asarray(rec["hinge_axis"], float),
            np.asarray(rec["joint_origin"], float),
            "door",
        )
    for rec in geom.drawer_records:
        parent_motion[rec["child_link"]] = (
            np.asarray(rec["slide_axis_world"], float),
            np.asarray(rec["slide_origin"],     float),
            "drawer",
        )

    # ── Predicted handle attachments ────────────────────────────────────────
    # Two paths:
    #   (a) handle.slot >= 0 (already in the input): leave its imported OBJ
    #       mesh alone, just parent it to the door/drawer for animation —
    #       same behaviour as before this module existed.
    #   (b) handle.slot <  0 (NEW free-slot prediction): drop the visualisation
    #       bbox and place a template handle via add_handle, on the side
    #       opposite the predicted motion axis (centered with horizontal
    #       margins). Done BEFORE hinge/rail install so the template handle
    #       becomes a child of the door — it follows the hinge animation.
    n_handle_existing = n_handle_template = n_handle_skipped = n_handle_err = 0
    _new_handle_queue: dict[str, list] = {}
    # Parents that already carry an ANCHORED handle never receive a
    # synthesized one (postprocess drops such predictions; this guards
    # older prediction files as well).
    _anchored_handle_parents: set = set()
    for e in pred.get("edges", []):
        if e.get("kind") != "attached":
            continue
        s, d = e["src"], e["dst"]
        sm = _mat_base(pred["nodes"].get(s, {}).get("material", ""))
        dm = _mat_base(pred["nodes"].get(d, {}).get("material", ""))
        if sm == "handle":
            h_id, p_id = s, d
        elif dm == "handle":
            h_id, p_id = d, s
        else:
            continue   # attached edge without a handle endpoint — no guard
        slot = int(pred["nodes"].get(h_id, {}).get("slot", -1))
        if 0 <= slot < len(input_ids):
            _anchored_handle_parents.add(p_id)
    for e in pred.get("edges", []):
        if e.get("kind") != "attached":
            continue
        s, d = e["src"], e["dst"]
        s_node = pred["nodes"].get(s, {})
        d_node = pred["nodes"].get(d, {})
        sm = _mat_base(s_node.get("material", ""))
        dm = _mat_base(d_node.get("material", ""))
        if sm == "handle" and dm in DYNAMIC_MATS:
            handle_id, parent_id, handle_node = s, d, s_node
        elif dm == "handle" and sm in DYNAMIC_MATS:
            handle_id, parent_id, handle_node = d, s, d_node
        else:
            continue

        p_obj = pred_id_to_obj.get(parent_id)
        h_obj = pred_id_to_obj.get(handle_id)
        if p_obj is None:
            n_handle_skipped += 1
            continue

        # A node is NEW (free-slot) when its slot is outside the input
        # range — same condition install_from_pred uses to decide whether
        # to import the input OBJ vs draw a bbox placeholder.
        slot = int(handle_node.get("slot", -1))
        is_new = not (0 <= slot < len(input_ids))

        if not is_new:
            # Existing input handle — keep its mesh, just parent for animation.
            if h_obj is not None:
                _parent_keep_transform(h_obj, p_obj)
                n_handle_existing += 1
            continue

        # is_new == True → free-slot handle. Drop it when the parent
        # already carries an anchored handle; otherwise queue it —
        # installation is grouped per parent so multiple handles on one
        # node are placed at equal-distance partitions.
        if parent_id in _anchored_handle_parents:
            if h_obj is not None and h_obj.name in bpy.data.objects:
                bpy.data.objects.remove(h_obj, do_unlink=True)
                pred_id_to_obj.pop(handle_id, None)
            n_handle_skipped += 1
            continue
        motion = parent_motion.get(parent_id)
        if motion is None:
            if h_obj is not None:
                _parent_keep_transform(h_obj, p_obj)
            n_handle_skipped += 1
            continue
        _new_handle_queue.setdefault(parent_id, []).append(
            (handle_id, h_obj, p_obj, motion))

    for parent_id, items in _new_handle_queue.items():
        n = len(items)
        # Multiple generated handles on one node: equal partitions along
        # the free placement axis, knob templates (bars would overlap).
        style = "handle16_knob" if n > 1 else "handle3_bar"
        for i, (handle_id, h_obj, p_obj, motion) in enumerate(items):
            frac = (i + 1) / (n + 1) if n > 1 else None
            try:
                ok = add_handle_for_predicted_attachment(
                    parent_obj=p_obj,
                    motion_axis=motion[0],
                    motion_origin=motion[1],
                    parent_kind=motion[2],
                    handle_style=style,
                    partition_frac=frac,
                    n_handles=n,
                )
            except Exception as exc:
                print(f"    handle on {parent_id}: ERR {exc}")
                ok = False
                n_handle_err += 1
            if ok:
                n_handle_template += 1
                if h_obj is not None and h_obj.name in bpy.data.objects:
                    bpy.data.objects.remove(h_obj, do_unlink=True)
                    pred_id_to_obj.pop(handle_id, None)
            elif h_obj is not None:
                _parent_keep_transform(h_obj, p_obj)
    print(f"  handles: existing={n_handle_existing} new_templates={n_handle_template} "
          f"skipped={n_handle_skipped} err={n_handle_err}")

    # ── Override every OBB with the actual mesh's world-AABB ──────────────
    # The labeled PNM OBBs are sometimes bulky (e.g. face_frame's depth
    # extent overshooting the actual frame thickness). Replacing them with
    # AABBs computed from the imported meshes themselves gives a tighter
    # geometric input to insert_hinges_for_joint, which selects faces by
    # gap and ranks by leaf footprint — both improved by tight OBBs.
    def _world_aabb_of(obj):
        if obj is None or obj.type != "MESH" or obj.data is None:
            return None
        if len(obj.data.vertices) == 0: return None
        mw = obj.matrix_world
        lo = [ float("inf")] * 3
        hi = [-float("inf")] * 3
        for v in obj.data.vertices:
            w = mw @ v.co
            for k in range(3):
                if w[k] < lo[k]: lo[k] = w[k]
                if w[k] > hi[k]: hi[k] = w[k]
        return np.asarray(lo), np.asarray(hi)

    def _aabb_to_centerhalf(lo, hi):
        c = ((lo + hi) * 0.5).tolist()
        h = np.maximum((hi - lo) * 0.5, 1e-4).tolist()
        return c, h

    # 1. Override hinge records.
    for rec in geom.door_records:
        d_obj = pred_id_to_obj.get(rec["door_node"])
        p_obj = pred_id_to_obj.get(rec["panel_node"])
        d_ab = _world_aabb_of(d_obj) if d_obj is not None else None
        p_ab = _world_aabb_of(p_obj) if p_obj is not None else None
        if d_ab is not None:
            c, h = _aabb_to_centerhalf(*d_ab)
            rec["door_center"] = c
            rec["door_size"]   = (np.asarray(h) * 2).tolist()
            rec["door_R"]      = np.eye(3).tolist()
            # Re-derive pin from the new door OBB and the stored hinge_border
            # face. _door_pin_point: face center + thin-axis push (matches
            # original install_from_pred semantics).
            new_pin = _door_pin_point(np.asarray(c), np.asarray(h),
                                       int(rec.get("hinge_border", -1)))
            rec["joint_origin"] = new_pin.tolist()
        if p_ab is not None:
            c, h = _aabb_to_centerhalf(*p_ab)
            rec["panel_center"] = c
            rec["panel_size"]   = (np.asarray(h) * 2).tolist()
            rec["panel_R"]      = np.eye(3).tolist()

    # 2. Override geom.body_boxes with each body part's mesh AABB.
    new_body_boxes = []
    for nid, n in pred["nodes"].items():
        mat = _mat_base(n.get("material", ""))
        if mat in DYNAMIC_MATS or mat == "handle":
            continue
        obj = pred_id_to_obj.get(nid)
        ab = _world_aabb_of(obj) if obj is not None else None
        if ab is None:
            continue
        c, h = _aabb_to_centerhalf(*ab)
        new_body_boxes.append(_PickerOBB(np.asarray(c), np.asarray(h)))
    geom.body_boxes = new_body_boxes
    # panel_idx in records refers to the OLD body_boxes order; rebuild it
    # by matching panel_node ids to the new body_boxes' centers.
    new_panel_idx_map = {}
    body_node_ids = []
    bb_idx = 0
    for nid, n in pred["nodes"].items():
        mat = _mat_base(n.get("material", ""))
        if mat in DYNAMIC_MATS or mat == "handle":
            continue
        obj = pred_id_to_obj.get(nid)
        ab = _world_aabb_of(obj) if obj is not None else None
        if ab is None:
            continue
        body_node_ids.append(nid)
        new_panel_idx_map[nid] = bb_idx
        bb_idx += 1
    for rec in geom.door_records:
        rec["panel_idx"] = new_panel_idx_map.get(rec["panel_node"], -1)
    print(f"  obb override: {len(new_body_boxes)} body boxes, "
          f"{len(geom.door_records)} hinge records re-derived from mesh AABBs")

    def _local_member_obb(panel_mesh_obj, panel_obb, door_obb, axis_hint, pin_hint):
        """Decompose a wide static partner into its LOCAL mounting member.

        Face frames, full-height dividers, and cabinet-wide top panels are
        single meshes whose AABB spans far beyond the one stile/rail/edge
        the hinge actually mounts on. Enumerating faces on that slab gives
        wrong planes (and, for a face frame, a face center inside the
        opening), which starves flat/exterior of candidates and lets a
        floating interior mount win by default. Slice the partner's verts
        to the door's axial span and to a window around the pin hint on
        every axis where the partner is much wider than the door, and use
        the slice's AABB as the panel OBB. Returns None when no slicing
        applies (partner already local)."""
        if panel_mesh_obj is None or panel_mesh_obj.type != "MESH" \
                or panel_mesh_obj.data is None or len(panel_mesh_obj.data.vertices) == 0:
            return None
        axis = np.asarray(axis_hint, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return None
        axis_i = int(np.argmax(np.abs(axis / n)))
        d_ext = door_obb.half_extents * 2.0
        p_ext = panel_obb.half_extents * 2.0
        slice_axes = [i for i in range(3)
                      if i != axis_i and p_ext[i] > 1.5 * d_ext[i]]
        # The axial band applies whenever ANY slicing happens: a frame's
        # rails span the full width above/below the opening, so without
        # the band they keep phantom in-plane faces inside the window
        # (the stile's reveal face then loses the smallest-gap contest).
        axial_slice = bool(slice_axes) or p_ext[axis_i] > 1.5 * d_ext[axis_i]
        if not slice_axes and not axial_slice:
            return None
        # Slice by GEOMETRY CLIPPING (bmesh bisect), not by vertex
        # filtering or interval intersection. Vertex filtering collapses
        # low-poly slabs (corner-only verts -> a 381mm-deep panel became a
        # 1.1mm sheet -> 0.07x miniature hinge); interval intersection
        # cannot see holes (a window crossing a face-frame OPENING gets a
        # phantom face in mid-air -> the reveal pair loses to a sandwich
        # pair on plane gap). Bisecting a copy of the actual mesh with the
        # window planes and taking the remainder's AABB is exact for both.
        import bmesh
        pin = np.asarray(pin_hint, dtype=float)
        thin = float(np.min(door_obb.half_extents) * 2.0)
        w = max(0.06, 3.0 * thin)
        planes = []   # (point, normal) keeps the half-space normal points AWAY from
        if axial_slice:
            planes.append((door_obb.center[axis_i] - 0.45 * d_ext[axis_i], axis_i, +1.0))
            planes.append((door_obb.center[axis_i] + 0.45 * d_ext[axis_i], axis_i, -1.0))
        for i in slice_axes:
            planes.append((pin[i] - w, i, +1.0))
            planes.append((pin[i] + w, i, -1.0))
        deps = bpy.context.evaluated_depsgraph_get()
        bm = bmesh.new()
        try:
            bm.from_object(panel_mesh_obj, deps)
            bm.transform(panel_mesh_obj.matrix_world)
            for coord, ax, sign in planes:
                pno = [0.0, 0.0, 0.0]; pno[ax] = sign
                res = bmesh.ops.bisect_plane(
                    bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                    plane_co=(coord if ax == 0 else 0.0,
                              coord if ax == 1 else 0.0,
                              coord if ax == 2 else 0.0),
                    plane_no=tuple(pno), clear_outer=False, clear_inner=True)
                if len(bm.verts) == 0:
                    return None
            if len(bm.verts) < 3:
                return None
            K = np.array([[v.co.x, v.co.y, v.co.z] for v in bm.verts], dtype=float)
        finally:
            bm.free()
        lo, hi = K.min(0), K.max(0)
        if np.any(hi - lo < 1e-4):
            return None
        return OBB(center=(lo + hi) * 0.5, half_extents=(hi - lo) * 0.5, R=np.eye(3))

    # ── Install hinges ──────────────────────────────────────────────────────
    body_centroid = _aggregate_body_centroid(geom.body_boxes) if geom.body_boxes else np.zeros(3)
    # Picker-OBB body list (full sizes, not half-extents) for
    # aggregate_body_center used by classify_door_panel.
    picker_body_boxes = [PickerOBB(center=np.asarray(b.center, float),
                                     R=np.eye(3),
                                     size=np.asarray(b.half, float) * 2.0)
                          for b in geom.body_boxes]
    body_center_picker = (aggregate_body_center(picker_body_boxes)
                          if picker_body_boxes else np.zeros(3))

    n_hinge_ok = n_hinge_err = 0
    for rec in geom.door_records:
        joint_name = rec["joint"]
        door_obb_pred = OBB(center=np.asarray(rec["door_center"], float),
                          half_extents=np.asarray(rec["door_size"], float)*0.5,
                          R=np.eye(3))
        panel_obb_pred = OBB(center=np.asarray(rec["panel_center"], float),
                           half_extents=np.asarray(rec["panel_size"], float)*0.5,
                           R=np.eye(3))
        # Body fragments: BOX MESHES of every body OBB except the panel itself.
        body_obbs_pred = [OBB(center=b.center, half_extents=b.half, R=np.eye(3))
                        for i, b in enumerate(geom.body_boxes) if i != rec["panel_idx"]]
        body_objs = [_make_box_mesh_object(o, f"__validation_temp__{joint_name}_{i}")
                     for i, o in enumerate(body_obbs_pred)]
        # Look up the actual imported meshes for the door + panel so the
        # hinge BVH-refinement step can snap leaves onto the real mesh
        # surfaces (handles routed inserts / thin frames inside bulky OBBs).
        door_mesh  = pred_id_to_obj.get(rec["door_node"])
        panel_mesh = pred_id_to_obj.get(rec["panel_node"])

        # ─── Geometric classification → natural hinge types ───────────────
        # Reuse HSSD's classify_door_panel (the same logic that built the
        # original URDF-based hinge records). Returns CASE_EXTERIOR /
        # CASE_INTERIOR / CASE_TUCKIN_PARALLEL based on door↔panel relative
        # geometry. For TUCKIN we fire BOTH flat and interior (per
        # suggest_hinge_strategies — that geometry admits either mount).
        door_picker = PickerOBB(
            center=np.asarray(rec["door_center"], float),
            R=np.asarray(rec["door_R"], float),
            size=np.asarray(rec["door_size"], float),
        )
        panel_picker = PickerOBB(
            center=np.asarray(rec["panel_center"], float),
            R=np.asarray(rec["panel_R"], float),
            size=np.asarray(rec["panel_size"], float),
        )
        panel_pick = PanelPick(idx=rec["panel_idx"], obb=panel_picker,
                                dist_hinge_to_surface=0.0)
        # Shared kwargs for every insert flavour below.
        _base_kwargs = dict(
            door_obb=door_obb_pred, panel_obb=panel_obb_pred,
            body_fragment_objs=body_objs,
            door_fragment_prefix=rec["door_node"] + "__mesh",
            axis_hint=np.asarray(rec["hinge_axis"], float),
            pin_hint=np.asarray(rec["joint_origin"], float),
            body_centroid=body_centroid,
            swing_range=tuple(rec["limit"]),
            swing_samples=12,
            scale_factor=hinge_scale,
            hinge_count=2,
            body_obbs_for_avoidance=body_obbs_pred,
            static_mesh_obj=panel_mesh,
            dynamic_mesh_obj=door_mesh,
        )
        local_panel = _local_member_obb(panel_mesh, panel_obb_pred, door_obb_pred,
                                        _base_kwargs["axis_hint"],
                                        _base_kwargs["pin_hint"])
        if local_panel is not None:
            _base_kwargs["panel_obb"] = local_panel
            print(f"    [member] {joint_name}: static partner localized "
                  f"{np.round(panel_obb_pred.half_extents*2,3).tolist()} -> "
                  f"{np.round(local_panel.half_extents*2,3).tolist()}")

        if hinge_arg == "auto":
            # Collision-first C>F>P competition: snap all classes, keep the
            # winner by (rounded-mm penetration, EXTERIOR>FLAT>INTERIOR).
            from install_policy_fpc import fpc_select_and_install
            try:
                try:
                    winner = fpc_select_and_install(joint_name=joint_name,
                                                    **_base_kwargs)
                except RuntimeError as exc:
                    # Localizing the partner can starve every class on
                    # degenerate geometry (paper-thin doors, odd chunks).
                    # Never lose coverage to the optimization: retry once
                    # with the original full-partner OBB.
                    if local_panel is None or "no hinge class" not in str(exc):
                        raise
                    print(f"      → auto: no candidates with localized "
                          f"partner; retrying with full partner OBB")
                    _base_kwargs["panel_obb"] = panel_obb_pred
                    winner = fpc_select_and_install(joint_name=joint_name,
                                                    **_base_kwargs)
                n_hinge_ok += 1
                print(f"      → auto: {winner['type'].value} "
                      f"(pen={winner['pen']:.2f}mm)")
            except Exception as exc:
                print(f"      → auto: ERR {exc}")
                n_hinge_err += 1
            continue

        # If --hinge=<type> was specified explicitly, install ONLY that
        # type and rely on hinge.enumerate_placements to filter joints
        # where this type has no valid placement (it raises
        # RuntimeError("no candidates of type ...") which we catch below).
        # Otherwise, fall back to classify_door_panel-driven natural-type
        # selection.
        if hinge_arg in HINGE_TYPES:
            natural_types = [HINGE_TYPES[hinge_arg]]
            print(f"    hinge {joint_name}: forced type={hinge_arg} (skip classifier)")
        else:
            cls = classify_door_panel(
                door_obb_body=door_picker,
                panel=panel_pick,
                body_center=body_center_picker,
                hinge_axis_body=np.asarray(rec["hinge_axis"], float),
            )
            natural_types = _CASE_TO_TYPES.get(cls.case, [])
            if not natural_types:
                print(f"    hinge {joint_name}: SKIP (case={cls.case} → no natural type)")
                n_hinge_err += 1
                continue
            print(f"    hinge {joint_name}: case={cls.case}  natural_types={[t.value for t in natural_types]}")

        # Install every natural type. Each call gets its own joint_name
        # suffix so the per-template object names don't collide. The door
        # ends up parented to whichever type install ran last (single
        # driver per door — multihinge convention).
        n_ok_for_joint = 0
        for nt in natural_types:
            suffix = f"_{nt.value}" if len(natural_types) > 1 else ""
            try:
                try:
                    insert_hinges_for_joint(joint_name=joint_name + suffix,
                                            force_hinge_type=nt,
                                            **_base_kwargs)
                except RuntimeError as exc:
                    # Same fallback the auto competition has: the localized
                    # partner can starve a class that the full-partner OBB
                    # admits (e.g. interior on wide dividers).
                    if local_panel is None or "no candidates" not in str(exc):
                        raise
                    print(f"      → {nt.value}: retrying with full partner OBB")
                    retry_kwargs = dict(_base_kwargs, panel_obb=panel_obb_pred)
                    insert_hinges_for_joint(joint_name=joint_name + suffix,
                                            force_hinge_type=nt,
                                            **retry_kwargs)
                n_ok_for_joint += 1
                print(f"      → {nt.value}: OK")
            except Exception as exc:
                print(f"      → {nt.value}: ERR {exc}")
        if n_ok_for_joint > 0:
            n_hinge_ok += 1
        else:
            n_hinge_err += 1

    # ── Install rails ───────────────────────────────────────────────────────
    n_rail_ok = n_rail_err = 0
    for rec in geom.drawer_records:
        prefix = rec["child_link"] + "__mesh"
        drawer_meshes = [o for o in bpy.context.scene.objects
                         if o.type == "MESH" and o.name.startswith(prefix)]
        try:
            res = install_rails_for_drawer(RAIL_TEMPLATES[rail_arg], rec,
                                           drawer_mesh_objs=drawer_meshes)
            n_rail_ok += 1

            def _cap_fmt(d):
                if d.get("capped"):
                    return (f"capped(x{d['thickness_scale']:.2f} "
                            f"protr={d['protrusion_mm']:.1f}mm wall={d['wall']})")
                return d.get("reason", "?") + (
                    f"({d['wall']})" if "wall" in d else "")
            tc = res.diagnostic.get("thickness_cap", {})
            print(f"    rail {rec['joint']}: OK  "
                  f"cap L={_cap_fmt(tc.get('left', {}))} "
                  f"R={_cap_fmt(tc.get('right', {}))}")
        except Exception as exc:
            print(f"    rail {rec['joint']}: ERR {exc}")
            n_rail_err += 1

    # ── Carve clearance pockets ────────────────────────────────────────────
    # Every installed hinge brings a `carve_body_clearance__<joint>_c0__h{i}`
    # mesh (a shrinkwrap-tight envelope of that hinge instance, parented
    # under hinge_master). We Boolean-DIFF it out of BOTH the static-side
    # mesh (panel) and the dynamic-side mesh (door) for that joint, so the
    # cabinet/door geometry gets the hinge-shaped recess needed for 3D
    # printing. Skip silently if the loaded hinge template has no carve mesh
    # (e.g. canonical interior/flat installs that don't ship one).
    def _apply_carve_diff(target_obj, carve_obj, mod_name):
        if target_obj is None or carve_obj is None: return False
        if target_obj.type != "MESH" or carve_obj.type != "MESH": return False
        bpy.ops.object.select_all(action="DESELECT")
        target_obj.select_set(True)
        bpy.context.view_layer.objects.active = target_obj
        if mod_name in target_obj.modifiers:
            target_obj.modifiers.remove(target_obj.modifiers[mod_name])
        for solver in ("EXACT", "FLOAT", "FAST"):
            try:
                mod = target_obj.modifiers.new(name=mod_name, type="BOOLEAN")
                mod.operation = "DIFFERENCE"
                mod.object = carve_obj
                mod.solver = solver
                if solver == "EXACT":
                    # PNM part meshes are often several overlapping closed
                    # shells merged into one mesh; without self-intersection
                    # handling the difference leaves membrane fragments.
                    mod.use_self = True
                bpy.ops.object.modifier_apply(modifier=mod_name)
                return True
            except Exception:
                if mod_name in target_obj.modifiers:
                    target_obj.modifiers.remove(target_obj.modifiers[mod_name])
                continue
        return False

    n_carves_applied = 0; n_carves_skipped = 0
    for rec in ([] if no_carve else geom.door_records):
        door_obj = bpy.data.objects.get(rec["door_node"] + "__mesh0")
        panel_obj = bpy.data.objects.get(rec["panel_node"] + "__mesh0")
        carve_prefix = f"carve_body_clearance__{rec['joint']}"
        carves = [o for o in bpy.data.objects
                  if o.type == "MESH" and o.name.startswith(carve_prefix)]
        if not carves:
            n_carves_skipped += 1
            continue
        for carve in carves:
            if door_obj is not None:
                if _apply_carve_diff(door_obj, carve, f"carve_{carve.name}_to_door"):
                    n_carves_applied += 1
            if panel_obj is not None:
                if _apply_carve_diff(panel_obj, carve, f"carve_{carve.name}_to_panel"):
                    n_carves_applied += 1
    # Rail carve pockets: each installed rail ships a `carve_rail__<joint>__L/R`
    # cutter (rail/carve.py: tight mesh copies over the full slide extent).
    # Subtract it from every static body mesh it overlaps -- the mounting
    # panel gets the nesting pocket, the face frame the slide-through slot.
    def _world_aabb(o):
        pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
        return ([min(p[i] for p in pts) for i in range(3)],
                [max(p[i] for p in pts) for i in range(3)])

    def _aabb_overlap(a, b, tol=1e-4):
        return all(a[1][i] > b[0][i] - tol and b[1][i] > a[0][i] - tol
                   for i in range(3))

    static_body_objs = [
        obj for nid, obj in pred_id_to_obj.items()
        if _mat_base(pred["nodes"].get(nid, {}).get("material", "")) not in DYNAMIC_MATS
        and _mat_base(pred["nodes"].get(nid, {}).get("material", "")) != "handle"
    ]

    def _strip_faces_inside_cutter(target_obj, carve_obj, margin=1e-4):
        """Delete target faces lying strictly inside the (convex) rail
        cutter. PNM meshes sometimes carry zero-thickness sheet faces on a
        panel surface; the boolean cuts the solid but the dangling sheet
        survives and covers the pocket opening. After a DIFFERENCE nothing
        legitimate can remain inside the subtracted volume."""
        Mc = np.array(carve_obj.matrix_world)
        Rc, tc = Mc[:3, :3], Mc[:3, 3]
        # Winding-order normals transform with the determinant sign: the
        # mirrored right-rail cutter has a reflection matrix (det < 0).
        Rn = np.linalg.inv(Rc).T * (1.0 if np.linalg.det(Rc) >= 0 else -1.0)
        planes = []
        for poly in carve_obj.data.polygons:
            n = Rn @ np.array(poly.normal)
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            planes.append((Rc @ np.array(poly.center) + tc, n / ln))
        if not planes:
            return 0
        Mt = np.array(target_obj.matrix_world)
        Rt, tt = Mt[:3, :3], Mt[:3, 3]
        doomed = []
        for poly in target_obj.data.polygons:
            c = Rt @ np.array(poly.center) + tt
            if all(float(np.dot(c - pc, pn)) < -margin for pc, pn in planes):
                doomed.append(poly.index)
        if doomed:
            bm = bmesh.new()
            bm.from_mesh(target_obj.data)
            bm.faces.ensure_lookup_table()
            bmesh.ops.delete(bm, geom=[bm.faces[i] for i in doomed],
                             context="FACES")
            bm.to_mesh(target_obj.data)
            bm.free()
        return len(doomed)

    n_rc = 0
    for rec in ([] if no_carve else geom.drawer_records):
        carves = [o for o in bpy.data.objects if o.type == "MESH"
                  and o.name.startswith(f"{RAIL_CARVE_PREFIX}__{rec['joint']}")]
        for carve in carves:
            cbox = _world_aabb(carve)
            for target in static_body_objs:
                if not _aabb_overlap(cbox, _world_aabb(target)):
                    continue
                n_rc += 1
                nv0 = len(target.data.vertices)
                backup = target.data.copy()
                if _apply_carve_diff(target, carve, f"rcarve_{n_rc}"):
                    nv1 = len(target.data.vertices)
                    if nv1 == 0:
                        # The cutter swallowed the whole panel (thinner than
                        # the rail) -- keep the original mesh instead.
                        old = target.data
                        target.data = backup
                        bpy.data.meshes.remove(old)
                        print(f"    rcarve SKIP (would delete panel): "
                              f"{target.name} by {carve.name}")
                        continue
                    n_carves_applied += 1
                    bpy.data.meshes.remove(backup)
                    if nv1 != nv0:
                        print(f"    rcarve cut: {target.name} "
                              f"({nv0}->{nv1} verts) by {carve.name}")
                    n_strip = _strip_faces_inside_cutter(target, carve)
                    if n_strip:
                        print(f"    rcarve strip: {n_strip} sheet face(s) "
                              f"inside cutter removed from {target.name}")
                else:
                    bpy.data.meshes.remove(backup)
    if geom.door_records or geom.drawer_records:
        print(f"  carve: applied={n_carves_applied}  joints_without_carve={n_carves_skipped}")

    _delete_temp_validation_objects()

    # ── Predicted top panels ────────────────────────────────────────────────
    # Same two-path logic as handles:
    #   (a) top_panel.slot >= 0 (already in input): do nothing — its imported
    #       OBJ is the top.
    #   (b) top_panel.slot <  0 (NEW free-slot prediction): merge all body
    #       meshes (excluding doors/drawers/handles AND any existing OR new
    #       top-panel/countertop nodes), build a top panel above the union
    #       via add_top, drop the predicted bbox.
    n_tops_added = n_tops_existing = 0
    # Body = every predicted node that is NOT a door / drawer / handle, and
    # NOT itself a top-panel-class node (so we don't double-count tops in the
    # silhouette).
    body_objs_for_top = [
        obj for nid, obj in pred_id_to_obj.items()
        if _mat_base(pred["nodes"].get(nid, {}).get("material", "")) not in DYNAMIC_MATS
        and _mat_base(pred["nodes"].get(nid, {}).get("material", "")) != "handle"
        and not is_top_panel_node(pred["nodes"].get(nid, {}))
    ]
    # First pass: count existing (anchored) top-panel-class nodes. If the
    # input already carries a top/countertop, the model's free-slot top
    # prediction is redundant — skip synthesising another one (otherwise we
    # end up with two tops, one labeled and one synthesised on top of it).
    n_anchored_tops = 0
    for nid, n in pred["nodes"].items():
        if not is_top_panel_node(n):
            continue
        slot = int(n.get("slot", -1))
        if 0 <= slot < len(input_ids):
            n_anchored_tops += 1
    for nid, n in pred["nodes"].items():
        if not is_top_panel_node(n):
            continue
        # Same is_new condition as the handle path: a node is NEW when its
        # slot lies outside the input-graph range.
        slot = int(n.get("slot", -1))
        is_new = not (0 <= slot < len(input_ids))
        if not is_new:
            # Existing input top panel — leave it alone.
            n_tops_existing += 1
            continue
        if n_anchored_tops > 0:
            # Input already has a top/countertop. The model "added" another
            # one but it's redundant; drop its visualisation bbox to avoid
            # a duplicated top in the scene.
            old_obj = pred_id_to_obj.pop(nid, None)
            if old_obj is not None and old_obj.name in bpy.data.objects:
                bpy.data.objects.remove(old_obj, do_unlink=True)
            print(f"    top_panel {nid} (NEW): SKIP (anchored top already present)")
            continue
        if not body_objs_for_top:
            print(f"    top_panel {nid} (NEW): SKIP (no body meshes)")
            continue
        try:
            new_top = add_top_from_body_meshes(
                body_objs=body_objs_for_top,
                up_axis="+Z",
                overhang_pct=0.03,
                thickness_abs=0.020,
                shape_style="RECTANGLE",
                name=f"predicted_top_{nid}",
                material=_material_for("top panel", alpha=1.0),
            )
            if new_top is not None:
                n_tops_added += 1
                print(f"    top_panel {nid} (NEW): OK -> '{new_top.name}'")
                # Drop the predicted bbox — the real top is now in the scene.
                old_obj = pred_id_to_obj.pop(nid, None)
                if old_obj is not None and old_obj.name in bpy.data.objects:
                    bpy.data.objects.remove(old_obj, do_unlink=True)
        except Exception as exc:
            print(f"    top_panel {nid} (NEW): ERR {exc}")
        # One synthesized top per cabinet; bail after the first success.
        if n_tops_added > 0:
            break
    print(f"  tops: existing={n_tops_existing} new_added={n_tops_added}")

    # Support blocks (optional but recommended; install_one runs them by default)
    try:
        drawer_links = [r["child_link"] for r in geom.drawer_records]
        sup = _add_support_blocks(drawer_links,
                                  enable_divider=not no_divider)
        print(f"    support: slabs={sup.get('n_slabs_built',0)} "
              f"dividers={sup.get('n_dividers_built',0)} "
              f"skipped={sup.get('n_skipped',0)} "
              f"(divider {'OFF' if no_divider else 'ON'})")
    except Exception as exc:
        print(f"    support: ERR {exc}")

    # If --hinge=<type> was forced AND the model has door records but NO hinge
    # of that type was successfully placed, the requested type doesn't fit
    # this geometry. Skip saving — the wrapper records the absent .blend as
    # "this type is unmountable for this model" rather than producing a
    # door-but-no-hinge artefact. Pass --force-save to override (useful for
    # inspecting the door-without-hinges intermediate result).
    if (hinge_arg in HINGE_TYPES
            and len(geom.door_records) > 0
            and n_hinge_ok == 0
            and not _flag("force-save")):
        print(f"  SKIP save: forced hinge type '{hinge_arg}' did not fit any "
              f"door (0/{len(geom.door_records)} successful placements) "
              f"(pass --force-save to write anyway)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Match the documented closed-open-closed cycle (frames 0..100) so the
    # saved file plays it directly instead of Blender's default 1..250.
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 100
    bpy.context.scene.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
    print(f"  wrote {out_path}  hinges={n_hinge_ok}/{len(geom.door_records)}  "
          f"rails={n_rail_ok}/{len(geom.drawer_records)}")


if __name__ == "__main__":
    main()
