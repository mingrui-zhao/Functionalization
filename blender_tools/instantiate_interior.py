"""Instantiate fired shelf/divider free slots on an installed scene.

Inserts EXACTLY one panel per fired shelf/divider node (N fired nodes ->
N panels) — never per-compartment blanket fills. Compartment choice per
fired node:
  1. connectivity — the compartment with the largest overlap between its
     (MARGIN-inflated) bounds and the scene AABBs of the fired node's
     predicted-edge neighbours;
  2. random fallback (seeded on blend name + node id) when no neighbour
     overlaps any compartment.
Multiple same-type panels landing in one compartment are spread at equal
partitions along the panel axis. Existing INTERIOR_* panels are cleared
first, so the run is idempotent.

Reusable API (used by the functionalization_ui addon):
    instantiate_interior_from_pred(pred, n_input, seed_name=...) -> dict

CLI (headless batch):
    blender --background <blend> --python instantiate_interior.py -- \
        --pred=<pred.json> --input=<input.json> [--save]
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import bpy  # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore

_HERE = Path(__file__).resolve()
_ADDON_UTILS = _HERE.parent / "functionalization_ui" / "utils"
for p in (str(_HERE.parent), str(_ADDON_UTILS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from interior_graph import compute_compartments, norm_mat  # noqa: E402

THICK = 0.018
FRONT_INSET = 0.01
MARGIN = 0.03  # compartment inflation when scoring neighbour overlap
FIT_GATE = 4 * THICK
PALETTE = {"shelf": (0.365, 0.639, 0.627, 1.0),
           "divider": (0.788, 0.627, 0.235, 1.0)}


def _node_aabb(nid: str):
    mn = np.full(3, np.inf); mx = np.full(3, -np.inf)
    found = False
    for o in bpy.data.objects:
        if o.type != "MESH" or not o.name.startswith(nid + "__mesh"):
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mn = np.minimum(mn, [w.x, w.y, w.z])
            mx = np.maximum(mx, [w.x, w.y, w.z])
            found = True
    return (mn, mx) if found else None


def _clear_previous() -> int:
    old = [o for o in bpy.data.objects if o.name.startswith("INTERIOR_")]
    for o in old:
        me = o.data
        bpy.data.objects.remove(o)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    if old:
        print(f"[interior] cleared {len(old)} old panels")
    return len(old)


def _make_panel(name: str, mn, mx, mat_name: str):
    mesh = bpy.data.meshes.new(name)
    cx = [(mn[i] + mx[i]) * 0.5 for i in range(3)]
    hx = [(mx[i] - mn[i]) * 0.5 for i in range(3)]
    verts = [(cx[0] + sx * hx[0], cx[1] + sy * hx[1], cx[2] + sz * hx[2])
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    m = bpy.data.materials.get(f"interior_{mat_name}")
    if m is None:
        m = bpy.data.materials.new(f"interior_{mat_name}")
        m.diffuse_color = PALETTE[mat_name]
    obj.data.materials.append(m)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def instantiate_interior_from_pred(pred: dict, n_input: int,
                                   seed_name: str | None = None) -> dict:
    """One panel per fired shelf/divider node, on the currently open scene.

    Returns {"fired", "cleared", "made", "skipped", "no_compartments"}.
    Idempotent: clears INTERIOR_* objects first.
    """
    result = {"fired": 0, "cleared": 0, "made": 0,
              "skipped": [], "no_compartments": False}

    fired = []  # (nid, mat) per fired shelf/divider node
    for nid in pred["nodes"]:
        m = re.match(r"slot(\d+)_(.+)", nid)
        if m and int(m.group(1)) >= n_input:
            mat = norm_mat(m.group(2))
            if mat in ("shelf", "divider"):
                fired.append((nid, mat))
    result["fired"] = len(fired)

    result["cleared"] = _clear_previous()

    if not fired:
        print("[interior] no fired shelf/divider — nothing to do")
        return result

    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()

    parts, drawer_boxes, dyn_centers = [], [], []
    static_mn = np.full(3, np.inf); static_mx = np.full(3, -np.inf)
    for nid, n in pred["nodes"].items():
        mat = norm_mat(re.sub(r"^slot\d+_", "", nid))
        ab = _node_aabb(nid)
        if ab is None:
            continue
        if mat == "drawer":
            drawer_boxes.append({"mn": ab[0], "mx": ab[1]})
            dyn_centers.append((ab[0] + ab[1]) * 0.5)
        elif mat == "door":
            dyn_centers.append((ab[0] + ab[1]) * 0.5)
        elif mat != "handle":
            parts.append({"mat": mat, "mn": ab[0], "mx": ab[1]})
            static_mn = np.minimum(static_mn, ab[0])
            static_mx = np.maximum(static_mx, ab[1])

    # front axis = horizontal axis with the largest dynamic-part offset
    # from the static body centre (fallback -Y)
    up_ax = 2
    body_c = (static_mn + static_mx) * 0.5
    front_ax, front_sign = 1, -1.0
    if dyn_centers:
        off = np.mean(dyn_centers, axis=0) - body_c
        front_ax = int(np.argmax(np.abs(off[:2])))
        front_sign = 1.0 if off[front_ax] > 0 else -1.0
    side_ax = [a for a in range(3) if a not in (up_ax, front_ax)][0]

    comps, _mounts = compute_compartments(parts, drawer_boxes,
                                          up_ax, front_ax, front_sign)
    print(f"[interior] fired={[(n, m) for n, m in fired]} "
          f"compartments={len(comps)} front_ax={front_ax} sign={front_sign:+.0f}")
    if not comps:
        # Loud, per DESIGN.md: all-drawer bodies legitimately have no free
        # compartment — the caller must surface this, not silently skip.
        print("[interior] no free compartments — cannot instantiate "
              f"({len(fired)} fired node(s) dropped)")
        result["no_compartments"] = True
        result["skipped"] = [nid for nid, _ in fired]
        return result

    # ── per-node compartment assignment ─────────────────────────────────
    adj = defaultdict(set)
    for e in pred.get("edges", []):
        s, t = e.get("src"), e.get("dst")
        if s and t:
            adj[s].add(t)
            adj[t].add(s)

    def overlap_score(comp, boxes):
        mn = comp[0] - MARGIN
        mx = comp[1] + MARGIN
        s = 0.0
        for bmn, bmx in boxes:
            o = np.minimum(mx, bmx) - np.maximum(mn, bmn)
            if np.all(o > 0):
                s += float(np.prod(o))
        return s

    def panel_axis(mat):
        return up_ax if mat == "shelf" else side_ax

    def fits(comp, mat):
        ax = panel_axis(mat)
        return (comp[1][ax] - comp[0][ax]) > FIT_GATE

    seed = seed_name or Path(bpy.data.filepath).stem or "unsaved"
    rng = random.Random(seed)
    groups = defaultdict(list)  # (comp_idx, mat) -> [nid]
    for nid, mat in fired:
        cands = [ci for ci in range(len(comps)) if fits(comps[ci], mat)]
        if not cands:
            print(f"[interior] {nid}: no compartment fits a {mat} — skipped")
            result["skipped"].append(nid)
            continue
        nbr_boxes = [b for b in (_node_aabb(v) for v in adj[nid])
                     if b is not None]
        scores = {ci: overlap_score(comps[ci], nbr_boxes) for ci in cands}
        best = max(cands, key=lambda ci: scores[ci])
        if scores[best] > 1e-9:
            print(f"[interior] {nid} -> comp {best} "
                  f"(connectivity, score={scores[best]:.5f})")
        else:
            best = rng.choice(cands)
            print(f"[interior] {nid} -> comp {best} (random fallback)")
        groups[(best, mat)].append(nid)

    # ── insert panels: equal partitions per (compartment, type) group ────
    n_made = 0
    for (ci, mat), nids in sorted(groups.items()):
        mn = comps[ci][0].copy(); mx = comps[ci][1].copy()
        if front_sign > 0:
            mx[front_ax] -= FRONT_INSET
        else:
            mn[front_ax] += FRONT_INSET
        ax = panel_axis(mat)
        k = len(nids)
        for j in range(k):
            pos = mn[ax] + (mx[ax] - mn[ax]) * (j + 1) / (k + 1)
            pmn, pmx = mn.copy(), mx.copy()
            pmn[ax], pmx[ax] = pos - THICK / 2, pos + THICK / 2
            _make_panel(f"INTERIOR_{mat}_{ci:02d}_{j:02d}", pmn, pmx, mat)
            n_made += 1

    result["made"] = n_made
    print(f"[interior] inserted {n_made} panels for {len(fired)} fired nodes")
    return result


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = dict(a.split("=", 1) for a in argv if "=" in a)
    pred = json.load(open(args["--pred"]))
    n_in = len(json.load(open(args["--input"]))["nodes"])
    res = instantiate_interior_from_pred(pred, n_in)
    if "--save" in argv and (res["made"] > 0 or res["cleared"] > 0):
        bpy.ops.wm.save_mainfile()
        print(f"[interior] saved {bpy.data.filepath}")


if __name__ == "__main__":
    main()
