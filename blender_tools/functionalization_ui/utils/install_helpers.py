"""Per-call wrappers around hinge / rail install pipelines.

Why these wrappers exist:
  - hinge.pipeline.insert_hinges_for_joint resolves the template blend by
    looking up a single dict (_TEMPLATE_FOR_TYPE) keyed on HingeType. To
    support multiple variants per type (exterior_01..05, flat_01..02,
    interior_01), we temporarily mutate that dict around each call.
  - Both pipelines live in the repo's `blender_tools/` package dir. We add
    that directory to sys.path on init so the addon can be installed under
    Blender's addons/ folder while the pipeline code stays in the repo.

Public API:
    init_paths(repo_root)             — once, before the helpers are used
    HINGE_VARIANTS                    — {category: [(label, blend_filename)]}
    RAIL_VARIANTS                     — {label: blend_filename}
    install_one_hinge(record, ...)    — install a single hinge joint
                                        (category='auto' runs the C>F>P
                                        collision competition)
    install_one_rail(record, ...)     — install a single rail/drawer joint
    remove_hinge_for_joint(joint)     — wipe a previously installed hinge
    remove_rail_for_joint(joint)      — wipe a previously installed rail
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import bpy
import numpy as np


_REPO_ROOT: Optional[Path] = None


def init_paths(repo_root: Path) -> None:
    """Put the pipeline package dir (hinge / rail / install_policy_fpc)
    onto sys.path and register the handle-template directory.

    Must be called once at addon registration before any install helper is
    invoked. Safe to call multiple times.
    """
    global _REPO_ROOT
    _REPO_ROOT = Path(repo_root).resolve()
    pkg_dir = _REPO_ROOT / "blender_tools"
    if (pkg_dir / "hinge").is_dir() and str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    # The addon may itself be *inside* blender_tools (bundled layout); then
    # repo_root is the repo and the loop above found it. When the addon is
    # installed under Blender's addons/ and repo_root points at the repo,
    # same. Either way, register the template dir for handle placement.
    from . import handle_placement as _hp
    _hp.set_template_dir(_REPO_ROOT / "annotated_mechanical_parts")


def _ensure_paths() -> None:
    if _REPO_ROOT is None:
        raise RuntimeError(
            "install_helpers.init_paths(repo_root) must be called first")


# ─────────────────────────────────────────────────────── template inventory ──
#
# Annotated mechanical parts live in `<repo>/annotated_mechanical_parts/`.
# These names must match the filenames present there exactly — at addon load
# we filter to those that actually exist.

HINGE_VARIANTS: dict[str, list[tuple[str, str]]] = {
    # category → [(short_label, blend_filename)]
    "exterior": [
        ("Exterior 01", "hinge_exterior_01.blend"),
        ("Exterior 02", "hinge_exterior_02.blend"),
        ("Exterior 03", "hinge_exterior_03.blend"),
        ("Exterior 04", "hinge_exterior_04.blend"),
        ("Exterior 05", "hinge_exterior_05.blend"),
    ],
    "flat": [
        ("Flat 01", "hinge_flat_01.blend"),
        ("Flat 02", "hinge_flat_02.blend"),
    ],
    "interior": [
        ("Interior 01", "hinge_interior_01.blend"),
    ],
}

RAIL_VARIANTS: list[tuple[str, str]] = [
    # Center-mount first = the default variant (first enum item wins).
    ("Center (rail 02)", "sliding_rail_02_annotated.blend"),
    ("Corner (rail 01)", "sliding_rail_annotated.blend"),
]


def available_hinge_variants() -> dict[str, list[tuple[str, str]]]:
    """Filter HINGE_VARIANTS to those whose .blend actually exists."""
    _ensure_paths()
    parts = _REPO_ROOT / "annotated_mechanical_parts"
    out: dict[str, list[tuple[str, str]]] = {}
    for cat, items in HINGE_VARIANTS.items():
        keep = [(lbl, fn) for lbl, fn in items if (parts / fn).is_file()]
        if keep:
            out[cat] = keep
    return out


def available_rail_variants() -> list[tuple[str, str]]:
    _ensure_paths()
    parts = _REPO_ROOT / "annotated_mechanical_parts"
    return [(lbl, fn) for lbl, fn in RAIL_VARIANTS if (parts / fn).is_file()]


# ───────────────────────────────────────────────────── hinge template swap ──

@contextmanager
def _override_hinge_template(hinge_type, variant_filename: str):
    """Temporarily override `_TEMPLATE_FOR_TYPE[hinge_type]` so the
    pipeline picks up our chosen variant. Restored on exit."""
    from hinge import pipeline as _hp
    prev = _hp._TEMPLATE_FOR_TYPE.get(hinge_type)
    _hp._TEMPLATE_FOR_TYPE[hinge_type] = variant_filename
    try:
        yield
    finally:
        if prev is None:
            _hp._TEMPLATE_FOR_TYPE.pop(hinge_type, None)
        else:
            _hp._TEMPLATE_FOR_TYPE[hinge_type] = prev


@contextmanager
def _force_dynamic_face_id(face_id, note: Optional[dict] = None):
    """Filter `enumerate_placements` to placements whose door face matches
    `face_id` (0..5, same encoding as hinge_border). Honors the model's
    prediction over hinge's geometry-only "minimum-gap wins" rule.

    Patches BOTH `hinge.geometry.enumerate_placements` AND
    `hinge.pipeline.enumerate_placements` — pipeline.py imports
    `enumerate_placements` by name, creating a local binding the call
    site resolves; patching only `geometry` has no effect on the
    pipeline's actual call.

    If the filter would drop every candidate, the original list is
    returned so the user gets a hinge somewhere rather than a silent
    failure — and `note["border_fallback"]` is set so the caller can
    surface the fallback in the UI (the per-row "Flip side" toggle is the
    manual override).

    face_id=None disables filtering.
    """
    if face_id is None or not (0 <= int(face_id) <= 5):
        yield
        return
    from hinge import geometry as _hg
    from hinge import pipeline as _hp
    target = int(face_id)
    orig_g = _hg.enumerate_placements
    orig_p = _hp.enumerate_placements

    def filtered(*args, **kwargs):
        placements = orig_g(*args, **kwargs)
        keep = [p for p in placements
                if int(getattr(p.dynamic_face, "face_id", -1)) == target]
        if keep:
            return keep
        face_ids = sorted({int(getattr(p.dynamic_face, "face_id", -1))
                           for p in placements})
        # No candidate mounts on the border face itself. That is NORMAL for
        # exterior-class mounts (the leaf sits on the door's broad face,
        # perpendicular to the border edge; the pin still honours the
        # predicted border via pin_hint). It is only a problem when a
        # candidate exists on the OPPOSITE border face — the left↔right
        # flip this filter was built to prevent.
        flip_risk = (target ^ 1) in face_ids
        print(f"[hinge] strict border={target}: 0 placements survived "
              f"(available={face_ids}, flip_risk={flip_risk}). "
              f"Falling back to geometry pick.")
        if note is not None:
            note["border_fallback"] = True
            note["available_faces"] = face_ids
            note["flip_risk"] = flip_risk
        return placements

    _hg.enumerate_placements = filtered
    _hp.enumerate_placements = filtered
    try:
        yield
    finally:
        _hg.enumerate_placements = orig_g
        _hp.enumerate_placements = orig_p


# ───────────────────────────────────────────── panel mesh decomposition ──
#
# Some labelled static parts (face_frame canonically — but the same
# happens for side_panel / top_panel / divider when the annotator fused
# multiple bars) ship as a single mesh with a bulky collective bbox.
# The hinge classifier then sees a panel "surface" that's mostly empty
# space and picks the wrong mount. Decomposing the panel mesh into
# loose parts and selecting the sub-bar nearest the predicted hinge
# axis fixes this. Ported from install_from_pred_decompose_panel.py.

PANEL_DECOMP_MAX_AXIS_DIST = 0.10  # metres — sub-bar must be within this
                                    # of the predicted hinge axis line


def _point_to_line_dist(p, line_pt, line_dir) -> float:
    p = np.asarray(p, float); a = np.asarray(line_pt, float)
    d = np.asarray(line_dir, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return float(np.linalg.norm(p - a))
    d = d / n
    v = p - a
    perp = v - (v @ d) * d
    return float(np.linalg.norm(perp))


def _decompose_panel_into_loose_parts(mesh_obj):
    """Duplicate `mesh_obj`, run remove_doubles + separate-by-loose on
    the copy, return [{center, half, obj}] — one per loose part. The
    objs stay in the scene so callers can pass them as BVH mesh inputs
    or add their non-picked parts to body_obbs_for_avoidance. Call
    `_release_panel_decomp(parts)` after use to clean up.
    """
    from mathutils import Vector
    if mesh_obj is None or mesh_obj.type != "MESH" or mesh_obj.data is None:
        return []
    bpy.ops.object.select_all(action="DESELECT")
    dup = mesh_obj.copy(); dup.data = mesh_obj.data.copy()
    dup.name = f"__decomp_{mesh_obj.name}"
    bpy.context.scene.collection.objects.link(dup)
    bpy.context.view_layer.update()
    dup.select_set(True)
    bpy.context.view_layer.objects.active = dup
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=1e-4)
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception as exc:
        print(f"    [panel-decompose] {mesh_obj.name}: ERR {exc}")
        if dup.name in bpy.data.objects:
            bpy.data.objects.remove(dup, do_unlink=True)
        return []
    raw = [o for o in bpy.data.objects
           if o.name.startswith(f"__decomp_{mesh_obj.name}")]
    parts = []
    for p in raw:
        if p.type != "MESH" or not p.data or len(p.data.vertices) == 0:
            bpy.data.objects.remove(p, do_unlink=True)
            continue
        mw = p.matrix_world
        verts = [mw @ v.co for v in p.data.vertices]
        lo = Vector((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
        hi = Vector((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))
        center = (lo + hi) * 0.5
        half = (hi - lo) * 0.5
        parts.append({
            "center": np.array([center.x, center.y, center.z], float),
            "half":   np.array([max(half.x, 1e-6),
                                max(half.y, 1e-6),
                                max(half.z, 1e-6)], float),
            "obj":    p,
        })
    return parts


def _release_panel_decomp(parts):
    for p in parts:
        o = p.get("obj")
        if o is not None and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)


# ───────────────────────────────────────────────────────── hinge install ────

def _make_box_mesh(obb, name: str) -> "bpy.types.Object":
    """Box mesh object for an OBB. Used for body-fragment collision input
    to hinge's validator.
    """
    from hinge.geometry import OBB
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
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces); mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = tuple(np.asarray(obb.center, float))
    bpy.context.scene.collection.objects.link(obj)
    return obj


_HINGE_TEMP_PREFIX = "__hinge_validation_temp__"
_INSTALLED_HINGE_NAMESPACE = "__hinge_inst__"
_INSTALLED_RAIL_NAMESPACE = "__rail_inst__"


def _clear_temp_objects(prefix: str) -> None:
    for o in [o for o in bpy.data.objects if o.name.startswith(prefix)]:
        try:
            mesh = o.data if o.type == "MESH" else None
            bpy.data.objects.remove(o, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        except Exception:
            pass


def install_one_hinge(record: dict,
                      body_obbs_picker: list,
                      pred_id_to_obj: dict,
                      category: str,
                      variant_filename: str,
                      hinge_count: int = 2,
                      scale_factor: Optional[float] = None,
                      strict_border: bool = True,
                      flip_side: bool = False,
                      use_panel_decompose: bool = True,
                      ) -> dict:
    """Install a single hinge.

    category: 'interior' | 'exterior' | 'flat'
    variant_filename: e.g. 'hinge_exterior_03.blend'  (must exist in
                     annotated_mechanical_parts/)
    scale_factor: None for auto (per-type rules); float for explicit.
    strict_border: if True (default), filter the v2 pipeline's placement
                   candidates to only those whose door face matches the
                   predicted `hinge_border`. Use False to let geometry win.
    flip_side: if True, XOR the hinge_border with 1 (flip along the same
               axis: min↔max face). Lets the user override a wrong
               prediction without re-running the model.
    use_panel_decompose: if True (default), split a fused mounting panel
               (e.g. face_frame) into loose parts and mount on the sub-bar
               nearest the hinge axis. Disable when that substitution
               leaves the placement enumerator with no candidates —
               install_from_pred's behaviour (full panel OBB).

    Returns a small diagnostic dict so the caller can surface placement
    success/failure to the UI.
    """
    _ensure_paths()
    from hinge.geometry import OBB, HingeType
    from hinge.pipeline import insert_hinges_for_joint

    type_map = {
        "interior": HingeType.INTERIOR,
        "exterior": HingeType.EXTERIOR,
        "flat":     HingeType.FLAT,
    }
    auto_mode = (category == "auto")
    hinge_type = None if auto_mode else type_map[category]

    joint_name = record["joint"]
    # Clear any previously-installed hinge objects for this joint AND
    # reset the door mesh's parenting/animation so the next snap reads a
    # clean world pose (otherwise coborder snaps to the stale post-merge
    # door center instead of its left edge).
    remove_hinge_for_joint(joint_name, door_node=record["door_node"])

    door_obb = OBB(center=np.asarray(record["door_center"], float),
                   half_extents=np.asarray(record["door_size"], float) * 0.5,
                   R=np.eye(3))
    panel_obb = OBB(center=np.asarray(record["panel_center"], float),
                    half_extents=np.asarray(record["panel_size"], float) * 0.5,
                    R=np.eye(3))
    panel_idx = int(record["panel_idx"])
    body_obbs_pred = [
        OBB(center=np.asarray(b.center, float),
            half_extents=np.asarray(b.half, float),
            R=np.eye(3))
        for i, b in enumerate(body_obbs_picker) if i != panel_idx
    ]
    body_objs = [_make_box_mesh(o, f"{_HINGE_TEMP_PREFIX}{joint_name}_{i}")
                 for i, o in enumerate(body_obbs_pred)]
    body_centroid = (np.mean([np.asarray(b.center, float)
                              for b in body_obbs_picker], axis=0)
                     if body_obbs_picker else np.zeros(3))

    door_mesh = pred_id_to_obj.get(record["door_node"])
    panel_mesh = pred_id_to_obj.get(record["panel_node"])

    # ── Decompose mounting panel into loose parts ─────────────────────────
    # face_frame (and any other "fused" labelled panel) ships as a single
    # mesh with a bulky collective bbox. We split it into loose parts and
    # use the sub-bar closest to the predicted hinge axis as the panel.
    # The non-picked sub-bars are added to body_obbs_for_avoidance so the
    # hinge template doesn't intersect them.
    panel_decomp_used = False
    decomp_parts: list = []
    decomp_keep_obj = None
    if panel_mesh is not None and use_panel_decompose:
        decomp_parts = _decompose_panel_into_loose_parts(panel_mesh)
        if len(decomp_parts) >= 2:
            axis_pt = np.asarray(record["joint_origin"], float)
            axis_dir = np.asarray(record["hinge_axis"], float)
            dists = [_point_to_line_dist(p["center"], axis_pt, axis_dir)
                     for p in decomp_parts]
            best = int(np.argmin(dists))
            if dists[best] > PANEL_DECOMP_MAX_AXIS_DIST:
                print(f"    panel-decompose ({record['panel_node']}): "
                      f"{len(decomp_parts)} parts, best d_axis="
                      f"{dists[best]:.4f}m > {PANEL_DECOMP_MAX_AXIS_DIST}m"
                      f" → skip override")
                _release_panel_decomp(decomp_parts)
                decomp_parts = []
            else:
                # DEBUG: dump every sub-bar so we can compare against the
                # reference pipeline's choice when the addon picks wrong.
                for i, p in enumerate(decomp_parts):
                    c = p["center"]; h = p["half"]
                    print(f"      bar {i}: c=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) "
                          f"h=({h[0]:.3f},{h[1]:.3f},{h[2]:.3f}) "
                          f"d_axis={dists[i]:.4f}{'  ← PICKED' if i == best else ''}")
                best_part = decomp_parts[best]
                c_sub = best_part["center"]; h_sub = best_part["half"]
                panel_obb = OBB(center=c_sub, half_extents=h_sub, R=np.eye(3))
                decomp_keep_obj = best_part["obj"]
                decomp_keep_obj.name = f"__panel_sub_{joint_name}"
                # Add the non-picked sub-bars to avoidance.
                for i, p in enumerate(decomp_parts):
                    if i == best:
                        continue
                    sub_obb = OBB(center=p["center"],
                                  half_extents=p["half"], R=np.eye(3))
                    body_obbs_pred.append(sub_obb)
                    body_objs.append(
                        _make_box_mesh(sub_obb,
                                       f"{_HINGE_TEMP_PREFIX}{joint_name}_pdecomp_{i}"))
                print(f"    panel-decompose ({record['panel_node']}): "
                      f"{len(decomp_parts)} parts → bar {best} "
                      f"d_axis={dists[best]:.4f}m (others d_max={max(dists):.4f})")
                panel_decomp_used = True
        else:
            _release_panel_decomp(decomp_parts)
            decomp_parts = []
    panel_mesh_for_bvh = decomp_keep_obj if panel_decomp_used else panel_mesh

    # Resolve effective hinge_border (predicted + optional flip + strict filter).
    predicted_border = int(record.get("hinge_border", -1))
    effective_border = predicted_border
    if flip_side and 0 <= predicted_border <= 5:
        effective_border ^= 1   # min↔max along same axis (0↔1, 2↔3, 4↔5)
    force_face = effective_border if (strict_border and 0 <= effective_border <= 5) else None

    # When the user flips the side, also recompute the pin_hint so the
    # Rule-B "not opposite to pin" filter doesn't accidentally reject the
    # new face. Same _door_pin_point geometry as the loader uses.
    pin_world = np.asarray(record["joint_origin"], float)
    if flip_side and 0 <= effective_border <= 5:
        from . import predicted_graph as _pg
        door_center = np.asarray(record["door_center"], float)
        door_half = np.asarray(record["door_size"], float) * 0.5
        pin_world = _pg._door_pin_point(door_center, door_half, effective_border)

    diag: dict = {"joint": joint_name, "category": category,
                  "variant": variant_filename, "ok": False,
                  "predicted_border": predicted_border,
                  "effective_border": effective_border,
                  "strict_border": strict_border}
    border_note: dict = {}
    common_kwargs = dict(
        door_obb=door_obb, panel_obb=panel_obb,
        body_fragment_objs=body_objs,
        door_fragment_prefix=record["door_node"] + "__mesh",
        axis_hint=np.asarray(record["hinge_axis"], float),
        pin_hint=pin_world,
        body_centroid=body_centroid,
        swing_range=tuple(record.get("limit", [0.0, np.pi / 2])),
        swing_samples=12,
        scale_factor=scale_factor,
        hinge_count=hinge_count,
        body_obbs_for_avoidance=body_obbs_pred,
        static_mesh_obj=panel_mesh_for_bvh,
        dynamic_mesh_obj=door_mesh,
    )
    try:
        if auto_mode:
            # Collision-first C>F>P competition (DESIGN.md §I.2): snap all
            # classes, keep the winner by (rounded-mm penetration,
            # EXTERIOR>FLAT>INTERIOR), delete losers.
            from install_policy_fpc import fpc_select_and_install
            with _force_dynamic_face_id(force_face, border_note):
                winner = fpc_select_and_install(
                    joint_name=_INSTALLED_HINGE_NAMESPACE + joint_name,
                    **common_kwargs)
            diag["ok"] = True
            diag["chosen_type"] = winner["type"].value
            diag["policy_pen_mm"] = winner["pen"]
            diag["policy_attempts"] = winner["attempts"]
            diag["panel_decomp_used"] = panel_decomp_used
        else:
            with _override_hinge_template(hinge_type, variant_filename), \
                 _force_dynamic_face_id(force_face, border_note):
                result = insert_hinges_for_joint(
                    joint_name=_INSTALLED_HINGE_NAMESPACE + joint_name,
                    force_hinge_type=hinge_type,
                    **common_kwargs)
            diag["ok"] = True
            diag["chosen_type"] = result.chosen_type.value
            diag["panel_decomp_used"] = panel_decomp_used
        diag["border_fallback"] = bool(border_note.get("border_fallback"))
        diag["flip_risk"] = bool(border_note.get("flip_risk"))
        if border_note.get("available_faces") is not None:
            diag["available_faces"] = border_note["available_faces"]
    except Exception as exc:
        diag["error"] = str(exc)
        print(f"[install_one_hinge] {joint_name}: ERR {exc}")
    finally:
        _clear_temp_objects(_HINGE_TEMP_PREFIX)
        # Release decomposition sub-meshes (we kept the picked sub-bar as
        # static_mesh_obj for BVH; remove it now so it doesn't pollute
        # the saved .blend).
        if decomp_parts:
            _release_panel_decomp(decomp_parts)
    return diag


def _reset_door_mesh_state(door_node: str) -> None:
    """Undo the parenting + matrix mutations applied by a previous hinge
    install on this door. Without this, a re-apply sees a door whose
    matrix_basis (set when hinge reparented it to the hinge driver)
    is stale relative to the world coordinates the records assume — and
    coborder ends up snapping to the door's stale center instead of its
    canonical left edge.

    Specifically:
      * unparent the door's mesh fragments (preserves world pose)
      * clear matrix_parent_inverse so matrix_world == matrix_basis once
        the parent is None
      * clear any animation_data hinge added (the door swing fcurves)
    """
    prefix = f"{door_node}__mesh"
    for o in list(bpy.data.objects):
        if o.type != "MESH" or not o.name.startswith(prefix):
            continue
        # Unparent while preserving the world matrix so vertices stay put.
        if o.parent is not None:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_parent_inverse.identity()
            o.matrix_world = mw
        else:
            o.matrix_parent_inverse.identity()
        # Drop swing fcurves so they don't keep pulling the door through
        # the next install's snap.
        if o.animation_data is not None:
            o.animation_data_clear()


def remove_hinge_for_joint(joint_name: str, door_node: str | None = None) -> int:
    """Delete every object installed for this joint, then reset the door
    mesh so re-applying with a different variant snaps cleanly.

    `door_node` (the predicted node id whose mesh was parented during the
    first install) is optional: if omitted, only the installed mechanism
    objects get removed. Pass it from install_one_hinge so the door is
    properly un-stuck before the next snap.
    """
    needle = _INSTALLED_HINGE_NAMESPACE + joint_name
    removed = 0
    for o in [o for o in bpy.data.objects
              if needle in o.name or _is_batch_joint_object(o, joint_name)]:
        try:
            bpy.data.objects.remove(o, do_unlink=True)
            removed += 1
        except Exception:
            pass
    if door_node:
        _reset_door_mesh_state(door_node)
    return removed


def _is_batch_joint_object(o, joint_name: str) -> bool:
    """Objects installed for this joint by the BATCH pipeline (attached
    inference blends): every batch hinge/rail/cutter object carries the
    joint name, while part meshes never do (slot naming) and are further
    protected by the __mesh marker."""
    return (joint_name in o.name and "__mesh" not in o.name
            and o.get("generated_by") != "support_block")


# ────────────────────────────────────────────────────────── rail install ────

def install_one_rail(record: dict,
                     variant_filename: str,
                     ) -> dict:
    """Install left+right rail pair for one drawer joint."""
    _ensure_paths()
    from rail.pipeline import install_rails_for_drawer

    joint_name = record["joint"]
    remove_rail_for_joint(joint_name)

    template_path = _REPO_ROOT / "annotated_mechanical_parts" / variant_filename

    # Drawer mesh fragments in the open scene, keyed by name prefix the
    # graph loader used: '<drawer_node>__mesh*'.
    prefix = record["drawer_node"] + "__mesh"
    drawer_meshes = [o for o in bpy.context.scene.objects
                     if o.type == "MESH" and o.name.startswith(prefix)]

    diag = {"joint": joint_name, "variant": variant_filename, "ok": False}
    try:
        # We tag installed rail objects so we can later remove them. Easiest
        # way: snapshot the object set, call install, mark new objects.
        before = {o.name for o in bpy.data.objects}
        result = install_rails_for_drawer(
            template_path, record,
            drawer_mesh_objs=drawer_meshes,
        )
        after_objs = [o for o in bpy.data.objects if o.name not in before]
        for o in after_objs:
            # Don't rename drawer meshes that were re-parented; only tag NEW
            # objects.
            if not o.name.startswith(_INSTALLED_RAIL_NAMESPACE):
                o["__rail_joint"] = joint_name
        diag["ok"] = True
        diag["style"] = result.style
        diag["n_new_objs"] = len(after_objs)
    except Exception as exc:
        diag["error"] = str(exc)
        print(f"[install_one_rail] {joint_name}: ERR {exc}")
    return diag


def remove_rail_for_joint(joint_name: str) -> int:
    """Remove rail objects previously tagged with `__rail_joint ==
    joint_name`, plus any batch-installed objects for the joint (attached
    inference blends carry no tags)."""
    removed = 0
    for o in list(bpy.data.objects):
        if (o.get("__rail_joint") == joint_name
                or _is_batch_joint_object(o, joint_name)):
            try:
                bpy.data.objects.remove(o, do_unlink=True)
                removed += 1
            except Exception:
                pass
    return removed


# ──────────────────────────────────────────── support blocks for drawers ────

def add_support_blocks(rail_records: list[dict],
                       enable_divider: bool = True) -> dict:
    """Optional: install support slabs under drawer rails. Mirrors the
    `_add_support_blocks` call in install_from_pred_decompose_panel."""
    _ensure_paths()
    from rail.support_block import process_current_scene
    drawer_links = [r["drawer_node"] for r in rail_records]
    return process_current_scene(drawer_links, enable_divider=enable_divider)
