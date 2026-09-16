"""Graph-aware interior compartment detection.

Raycast-free: the graph workflow imports one mesh per part node, so the
cavity can be computed directly from the labelled part AABBs (no fused
"static frame" mesh to probe):

  side panels   → lateral cavity bounds (inner faces)
  back panel    → back bound (inner face)
  bottom panel  → bottom bound (top face)
  top panel     → top bound (bottom face); if the model is TOP-LESS the
                  cavity is capped at the LOWEST side/back wall top — never
                  extended past real geometry
  face frame    → front bound (back face of the frame)
  shelf-like    → horizontal occupiers (split the cavity into vertical bands)
  divider-like  → vertical splitters (split a band side-by-side)
  drawers       → occupied vertical spans (from the rail records)

Shelf/divider/mid-panel parts are classified by their thin axis, not just
material: thin along up = shelf, thin along side = divider.

`compute_compartments` is pure (numpy only) so it can be unit-tested outside
Blender; `detect_and_visualize` does the bpy work and emits the standard
compartment-box visualisation used by generate/clear/commit.
"""
from __future__ import annotations

import re

import numpy as np


DEFAULT_WALL_THICKNESS = 0.018   # fallback when a bounding panel is missing
MIN_COMP_HEIGHT = 0.03           # ignore bands shorter than 3cm
MIN_COMP_WIDTH = 0.05            # ignore side-split slots narrower than 5cm
PAD = 0.002                      # padding between compartment and geometry
EPS_MOUNT = 0.04                 # max gap for a drawer face to count as
                                 # "mounted" on adjacent vertical structure

SIDE_MATS = {"side panel"}
BACK_MATS = {"back panel"}
BOTTOM_MATS = {"bottom panel"}
TOP_MATS = {"top panel", "countertop"}
FRAME_MATS = {"face frame"}
SHELFY_MATS = {"shelf", "divider", "mid panel"}
IGNORE_MATS = {"leg", "misc", "bar", "rail", "hinge", "unknown"}


def norm_mat(mat: str) -> str:
    """'side_panel' / 'side panel.001' → 'side panel'.
    Same normalisation as predicted_graph.mat_base, plus underscore folding
    (duplicated here so this module stays importable without bpy)."""
    base = re.sub(r"\.\d+$", "", (mat or "")).strip().lower()
    return base.replace("_", " ")


def _merge_intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, hi in sorted((min(s), max(s)) for s in spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _free_gaps(lo: float, hi: float, occupied: list[tuple[float, float]],
               min_size: float) -> list[tuple[float, float]]:
    """Complement of `occupied` within [lo, hi], keeping gaps >= min_size."""
    gaps = []
    cur = lo
    for o_lo, o_hi in _merge_intervals(occupied):
        if o_hi <= lo or o_lo >= hi:
            continue
        if o_lo - cur >= min_size:
            gaps.append((cur, o_lo))
        cur = max(cur, o_hi)
    if hi - cur >= min_size:
        gaps.append((cur, hi))
    return gaps


def compute_compartments(parts: list[dict],
                         drawer_boxes: list[dict],
                         up_ax: int, front_ax: int, front_sign: float,
                         wall_thickness: float = DEFAULT_WALL_THICKNESS,
                         ) -> tuple[list[tuple[np.ndarray, np.ndarray]],
                                    list[tuple[np.ndarray, np.ndarray]]]:
    """Cavity + free-space compartments from labelled part AABBs.

    parts: [{"mat": material (normalised here), "mn": np(3), "mx": np(3)}]
           (static parts only — no doors/drawers/handles)
    drawer_boxes: [{"mn": np(3), "mx": np(3)}] occupied drawer volumes

    Returns (compartments, mount_dividers), both [(mn, mx)] world AABBs:
      compartments   — free-space partition of the cavity in the
                       (side × up) cross-section (each occupier blocks
                       only its own lateral extent)
      mount_dividers — divider panels to INSERT flush against drawer
                       faces that have no vertical structure within
                       EPS_MOUNT to mount the rails on
    """
    side_ax = [a for a in range(3) if a not in (up_ax, front_ax)][0]

    parts = [{**p, "mat": norm_mat(p["mat"])} for p in parts]
    struct = [p for p in parts if p["mat"] not in IGNORE_MATS]
    if not struct:
        return [], []
    body_mn = np.min([p["mn"] for p in struct], axis=0)
    body_mx = np.max([p["mx"] for p in struct], axis=0)
    body_center = (body_mn + body_mx) * 0.5

    def sel(mats):
        return [p for p in struct if p["mat"] in mats]

    def cen(p, ax):
        return (p["mn"][ax] + p["mx"][ax]) * 0.5

    # ── lateral bounds from side panels ─────────────────────────────────
    # Panels near the lateral centre are interior columns, not outer walls
    # (using them as walls inverts the bounds on multi-column cabinets);
    # they act as vertical splitters instead.
    lo_s = body_mn[side_ax] + wall_thickness
    hi_s = body_mx[side_ax] - wall_thickness
    margin = 0.15 * (body_mx[side_ax] - body_mn[side_ax])
    left = [p for p in sel(SIDE_MATS)
            if cen(p, side_ax) < body_center[side_ax] - margin]
    right = [p for p in sel(SIDE_MATS)
             if cen(p, side_ax) > body_center[side_ax] + margin]
    wall_ids = {id(p) for p in left} | {id(p) for p in right}
    mid_side = [p for p in sel(SIDE_MATS) if id(p) not in wall_ids]
    if left:
        lo_s = max(p["mx"][side_ax] for p in left)
    if right:
        hi_s = min(p["mn"][side_ax] for p in right)
    if lo_s >= hi_s - MIN_COMP_WIDTH:
        print(f"[interior_graph] side bounds inverted/collapsed "
              f"({lo_s:.3f},{hi_s:.3f}) → body fallback")
        lo_s = body_mn[side_ax] + wall_thickness
        hi_s = body_mx[side_ax] - wall_thickness

    # ── depth bounds: back panel inner face + face-frame back face ──────
    # Only panels in the back half count as back walls (models sometimes
    # carry a second 'back panel' label on a front stretcher); frames only
    # in the front half.
    if front_sign > 0:
        backs = [p for p in sel(BACK_MATS)
                 if cen(p, front_ax) < body_center[front_ax]]
        frames = [p for p in sel(FRAME_MATS)
                  if cen(p, front_ax) >= body_center[front_ax]]
        lo_f = (max(p["mx"][front_ax] for p in backs) if backs
                else body_mn[front_ax] + wall_thickness)          # back
        hi_f = body_mx[front_ax] - 0.005                          # front
        if frames:
            hi_f = min(hi_f, min(p["mn"][front_ax] for p in frames))
    else:
        backs = [p for p in sel(BACK_MATS)
                 if cen(p, front_ax) > body_center[front_ax]]
        frames = [p for p in sel(FRAME_MATS)
                  if cen(p, front_ax) <= body_center[front_ax]]
        lo_f = body_mn[front_ax] + 0.005                          # front
        hi_f = (min(p["mn"][front_ax] for p in backs) if backs
                else body_mx[front_ax] - wall_thickness)          # back
        if frames:
            lo_f = max(lo_f, max(p["mx"][front_ax] for p in frames))
    if hi_f - lo_f < 0.05:
        print(f"[interior_graph] depth bounds collapsed "
              f"({lo_f:.3f},{hi_f:.3f}) → body fallback")
        if front_sign > 0:
            lo_f = body_mn[front_ax] + wall_thickness
            hi_f = body_mx[front_ax] - 0.005
        else:
            lo_f = body_mn[front_ax] + 0.005
            hi_f = body_mx[front_ax] - wall_thickness

    # ── vertical bounds ──────────────────────────────────────────────────
    lo_u = body_mn[up_ax] + wall_thickness
    bottoms = [p for p in sel(BOTTOM_MATS)
               if (p["mn"][up_ax] + p["mx"][up_ax]) * 0.5 < body_center[up_ax]]
    if bottoms:
        lo_u = max(p["mx"][up_ax] for p in bottoms)

    tops = [p for p in sel(TOP_MATS)
            if (p["mn"][up_ax] + p["mx"][up_ax]) * 0.5 >= body_center[up_ax]]
    if tops:
        hi_u = min(p["mn"][up_ax] for p in tops)
    else:
        # TOP-LESS model: cap the cavity at the lowest surrounding wall top
        # so shelves never overshoot the real geometry.
        walls = left + right + backs
        if walls:
            hi_u = min(p["mx"][up_ax] for p in walls)
        else:
            hi_u = body_mx[up_ax] - wall_thickness
    hi_u = min(hi_u, body_mx[up_ax])

    if not (lo_s < hi_s and lo_f < hi_f and lo_u + MIN_COMP_HEIGHT < hi_u):
        print(f"[interior_graph] degenerate cavity "
              f"side=({lo_s:.3f},{hi_s:.3f}) depth=({lo_f:.3f},{hi_f:.3f}) "
              f"up=({lo_u:.3f},{hi_u:.3f})")
        return [], []
    print(f"[interior_graph] cavity side=({lo_s:.3f},{hi_s:.3f}) "
          f"depth=({lo_f:.3f},{hi_f:.3f}) up=({lo_u:.3f},{hi_u:.3f})"
          f"{'' if tops else '  (top-less: capped at wall top)'}")

    def _in_cavity(p) -> bool:
        c = (p["mn"] + p["mx"]) * 0.5
        return (lo_s - PAD < c[side_ax] < hi_s + PAD
                and lo_f - PAD < c[front_ax] < hi_f + PAD)

    # ── occupier rectangles in the (side × up) cross-section ────────────
    # Each occupier blocks only its own lateral extent, so e.g. a corner
    # drawer doesn't poison its whole horizontal band.
    def _rect(p):
        s = (max(p["mn"][side_ax], lo_s), min(p["mx"][side_ax], hi_s))
        u = (max(p["mn"][up_ax], lo_u), min(p["mx"][up_ax], hi_u))
        if s[1] - s[0] <= 1e-6 or u[1] - u[0] <= 1e-6:
            return None
        return {"s": s, "u": u}

    occ: list[dict] = []
    vertical_structs: list[dict] = []   # rects that can mount drawer rails
    for p in sel(SHELFY_MATS) + mid_side:
        if not _in_cavity(p):
            continue
        thin = int(np.argmin(p["mx"] - p["mn"]))
        if thin not in (up_ax, side_ax):
            continue    # thin along depth: doesn't partition the cavity
        r = _rect(p)
        if r is None:
            continue
        occ.append(r)
        if thin == side_ax:
            vertical_structs.append(r)

    drawer_rects: list[dict] = []
    for d in drawer_boxes:
        r = _rect(d)
        if r is not None:
            occ.append(r)
            drawer_rects.append(r)

    # ── mounting dividers ────────────────────────────────────────────────
    # A drawer face with no vertical structure within EPS_MOUNT gets a
    # divider inserted flush against it (the rail mounting panel). The new
    # divider joins the occupiers, so it also becomes a column boundary
    # and can serve the next drawer's check.
    def _overlap(a, b):
        return min(a[1], b[1]) - max(a[0], b[0])

    def _has_mount(face_x, u_span):
        if abs(face_x - lo_s) < EPS_MOUNT or abs(face_x - hi_s) < EPS_MOUNT:
            return True
        need = 0.5 * (u_span[1] - u_span[0])
        for r in vertical_structs:
            if (r["s"][0] - EPS_MOUNT <= face_x <= r["s"][1] + EPS_MOUNT
                    and _overlap(r["u"], u_span) >= need):
                return True
        return False

    mounts: list[dict] = []
    for dr in drawer_rects:
        for face_x, direction in ((dr["s"][0], -1), (dr["s"][1], +1)):
            if _has_mount(face_x, dr["u"]):
                continue
            s_int = ((face_x - wall_thickness, face_x) if direction < 0
                     else (face_x, face_x + wall_thickness))
            s_int = (max(s_int[0], lo_s), min(s_int[1], hi_s))
            if s_int[1] - s_int[0] < 0.5 * wall_thickness:
                continue    # no room between drawer and wall
            # Extend vertically through the free space beside the drawer:
            # from the nearest occupier below to the nearest one above.
            u_lo, u_hi = lo_u, hi_u
            for o in occ:
                if _overlap(o["s"], s_int) <= 1e-4:
                    continue
                if o["u"][1] <= dr["u"][0] + 1e-6:
                    u_lo = max(u_lo, o["u"][1])
                elif o["u"][0] >= dr["u"][1] - 1e-6:
                    u_hi = min(u_hi, o["u"][0])
            r = {"s": s_int, "u": (u_lo, u_hi)}
            mounts.append(r)
            occ.append(r)
            vertical_structs.append(r)
            print(f"[interior_graph] mount divider side="
                  f"({s_int[0]:.3f},{s_int[1]:.3f}) up=({u_lo:.3f},{u_hi:.3f}) "
                  f"for drawer face at {face_x:.3f}")

    # ── column-first free-space partition ────────────────────────────────
    # Cut columns at every occupier edge, take the free vertical gaps per
    # column, then merge laterally-adjacent cells with identical vertical
    # intervals back into maximal compartments.
    cuts = {lo_s, hi_s}
    for o in occ:
        for x in o["s"]:
            if lo_s + 1e-6 < x < hi_s - 1e-6:
                cuts.add(x)
    xs = sorted(cuts)

    cells: list[dict] = []
    for x0, x1 in zip(xs, xs[1:]):
        if x1 - x0 <= 1e-6:
            continue
        blockers = [o["u"] for o in occ if _overlap(o["s"], (x0, x1)) > 1e-6]
        for u0, u1 in _free_gaps(lo_u, hi_u, blockers, MIN_COMP_HEIGHT):
            cells.append({"s": (x0, x1), "u": (u0, u1)})

    cells.sort(key=lambda c: (round(c["u"][0], 6), round(c["u"][1], 6),
                              c["s"][0]))
    merged: list[dict] = []
    for c in cells:
        m = merged[-1] if merged else None
        if (m is not None
                and abs(m["u"][0] - c["u"][0]) < 1e-6
                and abs(m["u"][1] - c["u"][1]) < 1e-6
                and abs(m["s"][1] - c["s"][0]) < 1e-6):
            m["s"] = (m["s"][0], c["s"][1])
        else:
            merged.append(dict(c))

    comps: list[tuple[np.ndarray, np.ndarray]] = []
    for c in merged:
        if c["s"][1] - c["s"][0] < MIN_COMP_WIDTH:
            continue
        mn = np.zeros(3); mx = np.zeros(3)
        mn[side_ax], mx[side_ax] = c["s"][0] + PAD, c["s"][1] - PAD
        mn[front_ax], mx[front_ax] = lo_f + PAD, hi_f - PAD
        mn[up_ax], mx[up_ax] = c["u"][0] + PAD, c["u"][1] - PAD
        comps.append((mn, mx))

    mount_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for r in mounts:
        mn = np.zeros(3); mx = np.zeros(3)
        mn[side_ax], mx[side_ax] = r["s"]
        mn[front_ax], mx[front_ax] = lo_f, hi_f
        mn[up_ax], mx[up_ax] = r["u"]
        mount_boxes.append((mn, mx))

    return comps, mount_boxes


# ─────────────────────────────────────────────────────────── bpy glue ──────

def detect_and_visualize(state: dict, pred_nodes: dict,
                         up_axis_str: str, front_axis_str: str
                         ) -> tuple[list, list]:
    """Gather part AABBs from the loaded scene, compute the free-space
    partition, insert mounting dividers, and create the compartment
    boxes. Returns (compartment_anchors, mount_divider_objs)."""
    import bpy
    from mathutils import Vector
    from . import interior_scene as di
    from . import predicted_graph as pg

    up_ax, _ = di._parse_axis_to_index(up_axis_str)
    front_ax, front_sign = di._parse_axis_to_index(front_axis_str)
    if up_ax == front_ax:
        raise ValueError(f"up and front axes must differ "
                         f"({up_axis_str} vs {front_axis_str})")
    side_ax = [a for a in range(3) if a not in (up_ax, front_ax)][0]

    id_to_obj = state.get("pred_id_to_obj", {})

    parts: list[dict] = []
    for nid, n in pred_nodes.items():
        mat = norm_mat(n.get("material", ""))
        if mat in pg.DYNAMIC_MATS or mat == "handle":
            continue
        obj = id_to_obj.get(nid)
        ab = pg._world_aabb(obj) if obj is not None else None
        if ab is None:
            continue
        parts.append({"mat": mat, "mn": ab[0], "mx": ab[1]})

    # Drawer occupied boxes from the (pose-independent) rail records; fall
    # back to the scene AABB for drawers that got no rail edge.
    drawer_boxes: list[dict] = []
    railed = set()
    for r in state.get("rails", []):
        c = np.asarray(r["drawer_center"], float)
        h = np.asarray(r["drawer_size"], float) * 0.5
        drawer_boxes.append({"mn": c - h, "mx": c + h})
        railed.add(r["drawer_node"])
    for nid, n in pred_nodes.items():
        if norm_mat(n.get("material", "")) != "drawer" or nid in railed:
            continue
        ab = pg._world_aabb(id_to_obj.get(nid))
        if ab is not None:
            drawer_boxes.append({"mn": ab[0], "mx": ab[1]})

    comps, mounts = compute_compartments(parts, drawer_boxes,
                                         up_ax, front_ax, front_sign)

    di.clear_compartment_boxes()
    # Remove mount dividers from a previous detect so re-runs don't stack.
    for o in list(bpy.data.objects):
        if o.name.startswith(f"{di.INTERIOR_PANEL_PREFIX}MOUNT"):
            bpy.data.objects.remove(o, do_unlink=True)
    col = di._ensure_interior_collection()

    # Insert mounting dividers as real panels (they are required structure,
    # not suggestions). orientation='vertical' so downstream recolour /
    # commit treats them like dividers; compartment_id=-1 keeps them out of
    # the per-compartment regeneration clears.
    mount_objs = []
    for i, (mn, mx) in enumerate(mounts):
        name = f"{di.INTERIOR_PANEL_PREFIX}MOUNT_{i:02d}"
        p = di._create_panel_box(name, Vector(mn.tolist()), Vector(mx.tolist()))
        for c in list(p.users_collection):
            c.objects.unlink(p)
        col.objects.link(p)
        p["compartment_id"] = -1
        p["panel_index"] = i
        p["orientation"] = "vertical"
        mount_objs.append(p)

    boxes = []
    for i, (mn, mx) in enumerate(comps):
        name = f"{di.INTERIOR_COMPARTMENT_PREFIX}{i:02d}"
        box = di._create_compartment_box(name, Vector(mn.tolist()),
                                         Vector(mx.tolist()), color_index=i)
        for c in list(box.users_collection):
            c.objects.unlink(box)
        col.objects.link(box)
        box["compartment_id"] = i
        box["side_ax"] = side_ax
        box["up_ax"] = up_ax
        box["front_ax"] = front_ax
        boxes.append(box)
        print(f"[interior_graph] compartment {i}: "
              f"mn=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) "
              f"mx=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})")
    return boxes, mount_objs
