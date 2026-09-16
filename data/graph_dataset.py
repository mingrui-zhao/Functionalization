"""
graph_dataset.py
================
PyTorch Geometric Dataset for the FurFun graph-to-graph translation task.

Node vocabulary (material category, 17 classes):
  handle, shelf, side panel, misc, leg, door, drawer,
  back panel, bottom panel, top panel, face frame,
  divider, rail, hinge, bar, countertop, unknown(16)

Both the input (unfunctional) graph and the target (functional) graph use
the same material vocabulary.  The model takes the input material labels as
encoder features and must predict the correct output material labels for
added/modified nodes, plus reconstruct edge types.

Per-node attributes:
  .x            (N, 9)  float32  [centroid(3), bbox(6)]
  .pos          (N, 3)  float32  centroid (equivariant positional feature)
  .material     (N,)    long     material category index
  .point_clouds list of (n_pc_points, 3) tensors
  .edge_index   (2, E)  long
  .edge_attr    (E,)    long     edge-type index

EDGE_TYPES = {contact:0, hinge:1, rail:2, attached:3, null:4}
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

# ── Vocabularies ─────────────────────────────────────────────────────────────

# Material category — encoder input AND decoder prediction target (semantic part label)
MATERIAL_VOCAB: Dict[str, int] = {
    "handle":       0,
    "shelf":        1,
    "side panel":   2,
    "misc":         3,
    "leg":          4,
    "door":         5,
    "drawer":       6,
    "back panel":   7,
    "bottom panel": 8,
    "top panel":    9,
    "face frame":   10,
    "divider":      11,
    "rail":         12,
    "hinge":        13,
    "bar":          14,
    "countertop":   15,
}
MATERIAL_UNKNOWN   = len(MATERIAL_VOCAB)      # 16 — for nodes without a label
NUM_MATERIAL_CLASSES = len(MATERIAL_VOCAB) + 1  # 17 (includes unknown)

IDX2MATERIAL: Dict[int, str] = {v: k for k, v in MATERIAL_VOCAB.items()}
IDX2MATERIAL[MATERIAL_UNKNOWN] = "unknown"

# ── Public material vocabulary (release interface) ──────────────────────────
# 14 part materials. `hinge` and `rail` are EDGE kinds (joint types), not node
# materials — they remain in the internal embedding table above only for
# checkpoint index compatibility, are rejected on input, and can never be
# predicted (masked in the decoder). Unlabeled parts use "unknown".
PUBLIC_MATERIALS = [
    "handle", "shelf", "side panel", "misc", "leg", "door", "drawer",
    "back panel", "bottom panel", "top panel", "face frame", "divider",
    "bar", "countertop",
]
_INTERNAL_ONLY = {"rail", "hinge"}


def public_material_to_idx(name: str) -> int:
    """Map a public material name to its internal embedding index.
    Edge-kind names (hinge/rail) and unrecognized labels map to unknown."""
    base = (name or "").replace("_", " ").strip().lower()
    if base in _INTERNAL_ONLY:
        return MATERIAL_UNKNOWN
    return MATERIAL_VOCAB.get(base, MATERIAL_UNKNOWN)

# Edge types
EDGE_TYPE_VOCAB: Dict[str, int] = {"contact": 0, "hinge": 1, "rail": 2, "attached": 3}
NULL_EDGE_CLASS  = 4
NUM_EDGE_CLASSES = 5   # contact / hinge / rail / attached / null

# Default location of pre-sampled point clouds
_DEFAULT_GEOM_ROOT = Path(__file__).resolve().parents[1] / "part_geometries"


# ── OBB helpers ───────────────────────────────────────────────────────────────

def obb_to_bbox(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Return AABB [min_xyz | max_xyz] (6-dim) from an axis-aligned box."""
    return np.concatenate([center - half, center + half], axis=0)   # (6,)


def _quat_to_matrix(quat) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def canonicalize_obb(center, half, quat):
    """Bake quaternion into a world-axis-aligned box.

    Returns (new_center, new_half) for the tight world-AABB hull of the
    oriented box. Exact for 90°-multiple rotations; a loose hull otherwise.
    Callers can then treat the box as axis-aligned (quat = identity).
    """
    c = np.asarray(center, dtype=np.float32)
    h = np.asarray(half,   dtype=np.float32)
    R = _quat_to_matrix(quat)
    world_half = np.abs(R) @ h
    return c, world_half.astype(np.float32)


def _sample_obb_surface(center: np.ndarray, half: np.ndarray,
                        n_points: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Uniform surface sampling on an axis-aligned box — fallback only."""
    h = np.maximum(half, 1e-4)
    areas = np.array([h[1]*h[2], h[0]*h[2], h[0]*h[1]], dtype=np.float32)
    areas = np.repeat(areas, 2)
    probs = areas / areas.sum()
    face_ids = rng.choice(6, size=n_points, p=probs)
    pts = []
    for fid in face_ids:
        axis = fid // 2
        sign = 1 if fid % 2 == 0 else -1
        u = rng.uniform(-1, 1, size=3).astype(np.float32)
        u[axis] = sign
        pts.append(center + h * u)
    return np.stack(pts, axis=0)   # (n_points, 3)


# ── Point-cloud loading ───────────────────────────────────────────────────────

def load_point_cloud(model_id: str,
                     node_id: str,
                     n_points: int,
                     geom_root: Path,
                     rng: np.random.Generator,
                     obb_center: np.ndarray,
                     obb_half: np.ndarray) -> np.ndarray:
    """Return (n_points, 3) float32 point cloud for one node.

    Lookup order
    ------------
    1. ``geom_root/<model_id>/<node_id>.npy``  — pre-sampled real geometry
       (exported from Blender, then sampled with sample_pointclouds.py).
       If the stored cloud has more than n_points, random-subsample.
       If fewer, sample with replacement.
    2. OBB surface fallback — used for nodes without a geometry file.
    """
    npy_path = geom_root / model_id / f"{node_id}.npy"
    if npy_path.exists():
        stored = np.load(str(npy_path))         # (M, 3) float32
        M = len(stored)
        if M >= n_points:
            idx = rng.choice(M, size=n_points, replace=False)
        else:
            idx = rng.choice(M, size=n_points, replace=True)
        return stored[idx].astype(np.float32)
    else:
        return _sample_obb_surface(obb_center, obb_half, n_points, rng)


# ── Single-graph loader ───────────────────────────────────────────────────────

def load_graph_json(json_path: str,
                    n_pc_points: int = 512,
                    geom_root: Optional[Path] = None,
                    rng: Optional[np.random.Generator] = None) -> Data:
    """Parse a graph JSON file into a PyG Data object.

    Parameters
    ----------
    json_path   : path to the .json graph file.
    n_pc_points : number of points per node point cloud.
    geom_root   : root of pre-sampled geometry directory.
                  Defaults to <project_root>/part_geometries.
    rng         : numpy random generator (created if None).

    Returns
    -------
    data : Data with attributes
        .material     (N,)    long    material category index
        .pos          (N, 3)  float32 centroid
        .bbox         (N, 6)  float32
        .point_clouds list of (n_pc_points, 3) float32 tensors
        .x            (N, 9)  float32 [centroid(3), bbox(6)]
        .edge_index   (2, E)  long
        .edge_attr    (E,)    long
        .num_nodes    int
        .model_id     str
        .node_names   list[str]  part names (e.g. 'back_panel', 'door.001')
        .pc_source    list[str]  'real' or 'obb' per node
    """
    if rng is None:
        rng = np.random.default_rng()
    if geom_root is None:
        geom_root = _DEFAULT_GEOM_ROOT

    with open(json_path) as f:
        raw = json.load(f)

    model_id  = raw.get("model_id", Path(json_path).stem)
    nodes_raw = raw["nodes"]
    edges_raw = raw.get("edges", [])

    node_ids = list(nodes_raw.keys())
    id2idx   = {nid: i for i, nid in enumerate(node_ids)}
    N        = len(node_ids)

    materials    = []
    centroids    = []
    bboxes       = []
    point_clouds = []
    pc_sources   = []
    node_names   = []
    handle_borders = []      # per-node (only meaningful on door nodes), -1 otherwise

    for nid in node_ids:
        nd     = nodes_raw[nid]
        mat_name = nd.get("material", "")
        # Blender's duplicate-material suffix (e.g. "side panel.001") maps to base vocab.
        mat_base = re.sub(r'\.\d+$', '', mat_name) if mat_name else ""
        materials.append(public_material_to_idx(mat_base))
        node_names.append(nid)

        obb    = nd["obb"]
        center = np.array(obb["center"], dtype=np.float32)
        half   = np.array(obb["half"],   dtype=np.float32)
        quat   = obb.get("quat", [1.0, 0.0, 0.0, 0.0])
        center, half = canonicalize_obb(center, half, quat)

        centroids.append(center)
        bboxes.append(obb_to_bbox(center, half))

        pc = load_point_cloud(model_id, nid, n_pc_points,
                               geom_root, rng, center, half)
        point_clouds.append(torch.tensor(pc, dtype=torch.float32))

        npy_exists = (geom_root / model_id / f"{nid}.npy").exists()
        pc_sources.append("real" if npy_exists else "obb")

        # v3 handle-border label (lives on door nodes; -1 elsewhere)
        hb = nd.get("handle_border", -1)
        try:
            handle_borders.append(int(hb))
        except (TypeError, ValueError):
            handle_borders.append(-1)

    material_t  = torch.tensor(materials,  dtype=torch.long)         # (N,)
    pos_t       = torch.tensor(np.stack(centroids), dtype=torch.float32)  # (N,3)
    bbox_t      = torch.tensor(np.stack(bboxes),    dtype=torch.float32)  # (N,6)
    x           = torch.cat([pos_t, bbox_t], dim=1)                       # (N,9)

    # Edges (undirected: store both directions)
    # Motion label encoding (v2 — relative face pair):
    #   hinge_face_src/dst: 0..5 (min_X/max_X/min_Y/max_Y/min_Z/max_Z), -1=unknown
    #   hinge_direction:    0=positive, 1=negative, -1=unknown
    #   hinge_axis:         0=X, 1=Y, 2=Z, -1=unknown
    #   rail_axis:          0=X, 1=Y, 2=Z, -1=unknown
    # When we duplicate to both directions, the face labels swap.

    src_list, dst_list, etype_list = [], [], []
    hf_src_list, hf_dst_list, h_ax_list, h_dir_list, ra_list = [], [], [], [], []
    # v3 per-edge labels (see build_motion_labels_v3.py): integers in
    # {0..5} encoding ±X/±Y/±Z faces in the DYNAMIC part's local frame,
    # or -1 if unlabelled. `dyn_is_src` marks which endpoint is the
    # dynamic part (needed so edge-direction duplication doesn't swap
    # the semantics — labels stay attached to the dynamic node).
    hb_list, has_list, ras3_list, dyn_is_src_list = [], [], [], []
    b4_list, as4_list = [], []
    # Which material each endpoint has (for loss-time masking)
    for e in edges_raw:
        s = id2idx.get(e["src"])
        d = id2idx.get(e["dst"])
        if s is None or d is None:
            continue
        etype = EDGE_TYPE_VOCAB.get(e.get("kind", "contact"), 0)

        def _int_or(v, default=-1):
            if v is None:
                return default
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        hf_s = _int_or(e.get("hinge_face_src"))
        hf_d = _int_or(e.get("hinge_face_dst"))
        h_ax = _int_or(e.get("hinge_axis"))
        hd   = _int_or(e.get("hinge_direction"))
        r_ax = _int_or(e.get("rail_axis"))

        # v3 per-dynamic-part labels
        hb   = _int_or(e.get("hinge_border"))
        has_ = _int_or(e.get("hinge_axis_signed"))
        ras3 = _int_or(e.get("rail_axis_signed"))
        # v3_pose_inv labels (same edge JSON; relabel script writes both)
        b4_  = _int_or(e.get("hinge_border_4"))
        as_4 = _int_or(e.get("hinge_axis_sign"))

        # Which endpoint is the dynamic one (by material)?
        def _dyn_src(edge):
            sm = MATERIAL_VOCAB.get(
                re.sub(r"\.\d+$", "", nodes_raw[edge["src"]].get("material","")),
                MATERIAL_UNKNOWN)
            dm = MATERIAL_VOCAB.get(
                re.sub(r"\.\d+$", "", nodes_raw[edge["dst"]].get("material","")),
                MATERIAL_UNKNOWN)
            # door=5, drawer=6
            s_dyn = sm in (5, 6)
            d_dyn = dm in (5, 6)
            if s_dyn and not d_dyn: return 1
            if d_dyn and not s_dyn: return 0
            return -1
        dyn_src = _dyn_src(e)

        src_list  += [s, d]
        dst_list  += [d, s]
        etype_list += [etype, etype]
        # Face labels swap with direction (v2 legacy)
        hf_src_list += [hf_s, hf_d]
        hf_dst_list += [hf_d, hf_s]
        h_ax_list   += [h_ax, h_ax]
        h_dir_list  += [hd,   hd]
        ra_list     += [r_ax, r_ax]
        # v3 labels are tied to the dynamic node — same value in both
        # directions; dyn_is_src flips.
        hb_list   += [hb,   hb]
        has_list  += [has_, has_]
        ras3_list += [ras3, ras3]
        b4_list   += [b4_,  b4_]
        as4_list  += [as_4, as_4]
        dyn_is_src_list += [dyn_src, 1 - dyn_src if dyn_src in (0, 1) else -1]

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(etype_list, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros(0, dtype=torch.long)

    def _long(xs):
        return torch.tensor(xs, dtype=torch.long) if xs else torch.zeros(0, dtype=torch.long)

    hinge_face_src_t = _long(hf_src_list)
    hinge_face_dst_t = _long(hf_dst_list)
    hinge_axis_t     = _long(h_ax_list)
    hinge_dir_t      = _long(h_dir_list)
    rail_ax_t        = _long(ra_list)
    hinge_border_t        = _long(hb_list)
    hinge_axis_signed_t   = _long(has_list)
    rail_axis_signed_t    = _long(ras3_list)
    hinge_border_4_t      = _long(b4_list)
    hinge_axis_sign_t     = _long(as4_list)
    dyn_is_src_t          = _long(dyn_is_src_list)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos_t,
        bbox=bbox_t,
        material=material_t,
        num_nodes=N,
    )
    data.point_clouds  = point_clouds
    data.pc_source     = pc_sources
    data.node_names    = node_names
    data.hinge_face_src = hinge_face_src_t
    data.hinge_face_dst = hinge_face_dst_t
    data.hinge_axis     = hinge_axis_t
    data.hinge_dir      = hinge_dir_t
    data.rail_axis      = rail_ax_t
    # v3 per-dynamic-part labels (same value on both edge directions)
    data.hinge_border        = hinge_border_t
    data.hinge_axis_signed   = hinge_axis_signed_t
    data.rail_axis_signed    = rail_axis_signed_t
    # v3_pose_inv labels (same edge JSON; -1 if not relabelled).
    data.hinge_border_4      = hinge_border_4_t
    data.hinge_axis_sign     = hinge_axis_sign_t
    data.dyn_is_src          = dyn_is_src_t
    # Per-node handle-border label (door nodes only; -1 elsewhere)
    data.handle_border       = torch.tensor(handle_borders, dtype=torch.long)
    data.model_id       = model_id
    return data


# ── Masking augmentation ──────────────────────────────────────────────────────

# Legacy edge-collapse strategies — heavily down-weighted in v3. The real
# `unfunc_root` data ALREADY supplies contact-only graphs that mirror what
# these strategies produce, so they are largely redundant. Kept (at low
# weight) only for the variants that aren't covered by unfunc data:
#   - isolate_door:    drops door-incident edges (not in unfunc)
#   - fully_anonymized: wipes all materials to UNKNOWN (not in unfunc)
_LEGACY_STRATEGIES = {
    "isolate_door":      0.05,
    "fully_anonymized":  0.05,
}

# Category-drop strategies — supervise free-slot existence by removing
# specific structural categories the model needs to recover. Bumped from
# 0.05–0.08 to ~0.15 each so each category sees ~6× more positive
# free-slot supervision per epoch (data-driven completion signal).
_CATEGORY_DROP_STRATEGIES = {
    "drop_handles":      0.15,
    "drop_top":          0.15,
    "drop_bottom":       0.10,
    "drop_doors":        0.08,
    "drop_drawers":      0.08,
    "drop_completion":   0.10,    # drop 2-3 categories simultaneously
    "random":            0.05,    # rarely useful, kept at low weight
}
# Optional strategies (e.g. "drop_shelf"/"drop_divider" for the broadened
# reconstruction-coverage recipe) are enabled per-run via the yaml key
# `category_drop_overrides` — see train.py.

# Edge-preserving strategies — surviving edges keep their FUNCTIONAL type
# (hinge/rail/attached/contact), so the encoder edge-type embedding rows for
# hinge/rail/attached actually receive gradient. They also act as a masked-
# graph reconstruction pretext: the model sees a partial functional graph
# and must complete it.
_EDGE_PRESERVING_STRATEGIES = {
    "mask_random_edges_keep_types": 0.10,
    "drop_nodes_keep_edge_types":   0.05,
    "mask_materials_keep_edges":    0.04,
    # Drops 1–4 random nodes (excluding doors/drawers) and KEEPS all
    # functional edge types. Trains the model on partial-functional inputs
    # — needed for cascaded inference where prior steps labelled some edges.
    "fully_functional_drop":        0.20,
}

# Combined weight table used at sampling time.
_DROP_STRATEGY_WEIGHTS = {**_LEGACY_STRATEGIES, **_CATEGORY_DROP_STRATEGIES,
                          **_EDGE_PRESERVING_STRATEGIES}
_DROP_STRATEGIES = list(_DROP_STRATEGY_WEIGHTS.keys())

# Curriculum: when the dataset is told its training "progress" via
# `set_curriculum(t)` (t=0..1), drop rates anneal from "aggressive" to
# "modest". Forces the existence head to develop strong "fire" priors
# before the data settles into mostly-complete graphs.
_CURRICULUM_PROGRESS = [0.0]   # mutable scalar: 0=start, 1=end of training


# Per-sample cap on number of nodes dropped. None = no cap (use rate × N as is).
# When set (e.g. 4), every dropping strategy clips drop_n at this value so
# the model can pair each missing node with one free slot. Set this to
# match `max_free_slots` so the existence head always has a feasible target.
_MAX_DROP_NODES: Optional[int] = None

def set_max_drop_nodes(n: Optional[int]) -> None:
    global _MAX_DROP_NODES
    _MAX_DROP_NODES = int(n) if n is not None else None


def _cap_drop_n(drop_n: int) -> int:
    return min(drop_n, _MAX_DROP_NODES) if _MAX_DROP_NODES else drop_n


# Multiplier applied to the edge-preserving strategies' sampling weights at
# strategy-selection time. Boosts the rate of "partial functional graph"
# samples — useful for cascaded inference where the model needs to handle
# inputs that already have some hinge/rail/attached edges labelled.
_EDGE_PRESERVING_BOOST: float = 1.0

def set_edge_preserving_boost(b: float) -> None:
    global _EDGE_PRESERVING_BOOST
    _EDGE_PRESERVING_BOOST = float(b)


def set_curriculum_progress(t: float) -> None:
    """Set training progress for curriculum drop rates. Called from train.py
    each epoch with t = epoch / total_epochs."""
    _CURRICULUM_PROGRESS[0] = float(max(0.0, min(1.0, t)))


# Module-level drop bands. Configurable via set_drop_bands() so a single
# training run can hold drop rates at a constant level (e.g. (0.30, 0.50))
# instead of annealing them down. Defaults reproduce the original curriculum.
_DROP_BANDS = {"early": (0.30, 0.50), "late": (0.10, 0.20)}


def set_drop_bands(early: Tuple[float, float], late: Tuple[float, float]) -> None:
    """Override the default (early, late) drop-rate bands. Set early == late
    to disable annealing and hold drop rates constant throughout training."""
    _DROP_BANDS["early"] = (float(early[0]), float(early[1]))
    _DROP_BANDS["late"]  = (float(late[0]),  float(late[1]))


def _curriculum_drop_rate(early: Optional[Tuple[float, float]] = None,
                           late: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """Return a (lo, hi) range for random drop rate that lerps between
    the early-training band and the late-training band based on progress.
    When early/late are None, uses the module-level _DROP_BANDS (settable
    via set_drop_bands)."""
    t = _CURRICULUM_PROGRESS[0]
    el, eh = early if early is not None else _DROP_BANDS["early"]
    ll, lh = late  if late  is not None else _DROP_BANDS["late"]
    return (el + (ll - el) * t, eh + (lh - eh) * t)


def _graph_is_connected(raw_graph: dict) -> bool:
    """BFS connectivity check on a graph JSON (any-kind edges)."""
    nodes = raw_graph.get("nodes", {})
    if not nodes:
        return False
    from collections import defaultdict, deque
    adj: Dict[str, set] = defaultdict(set)
    for nid in nodes: adj[nid] = set()
    for e in raw_graph.get("edges", []):
        adj[e["src"]].add(e["dst"]); adj[e["dst"]].add(e["src"])
    start = next(iter(nodes))
    seen = {start}; q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen: seen.add(v); q.append(v)
    return len(seen) == len(nodes)


# ── Mirror/flip augmentation ────────────────────────────────────────────────
#
# A world mirror across axis m transforms the motion labels in TWO distinct
# ways, and conflating them mislabels every hinge:
#
#   * POSITIONAL faces (hinge_border, handle_border) and TRANSLATION
#     directions (rail_axis_signed): the ±m face pair swaps, all other
#     faces unchanged.
#   * ROTATION axes (hinge_axis_signed) are pseudo-vectors: the mirrored
#     motion is R∘Rot(a,θ)∘R⁻¹ = Rot(−Ra, θ), so the pair ALIGNED with m
#     stays put and the two PERPENDICULAR pairs swap sign. (Real-data
#     check: a left/right door pair mirrored across X carries borders 0/1
#     with axes 4/5 — border pair swaps AND the z-axis pair swaps.)
#
# Z-flip stays disabled (would invert "up").
_FACE_SWAP_X = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5}
_FACE_SWAP_Y = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5}
_AXIAL_SWAP_X = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4}
_AXIAL_SWAP_Y = {0: 1, 1: 0, 2: 2, 3: 3, 4: 5, 5: 4}


def _apply_mirror(data: Data, axis: int) -> Data:
    """Mirror a Data object across world axis `axis` ∈ {0=X, 1=Y}.
    Negates that coord on pos/bbox/point_clouds and remaps every motion
    label representation: 6-way faces, 6-way rotation axes, and the
    v3_pose_inv thin-frame labels (hinge_border_4 / hinge_axis_sign)."""
    swap = _FACE_SWAP_X if axis == 0 else _FACE_SWAP_Y
    axial = _AXIAL_SWAP_X if axis == 0 else _AXIAL_SWAP_Y
    # 1. Flip pos / bbox along `axis`
    data.pos = data.pos.clone()
    data.pos[:, axis] *= -1.0
    # bbox = [min_x, min_y, min_z, max_x, max_y, max_z]; flipping `axis`
    # swaps min<->max on that axis with sign-negation.
    data.bbox = data.bbox.clone()
    lo = -data.bbox[:, 3 + axis]   # new min = -(old max)
    hi = -data.bbox[:, axis]       # new max = -(old min)
    data.bbox[:, axis] = lo
    data.bbox[:, 3 + axis] = hi
    # 2. Flip point clouds
    new_pcs = []
    for pc in data.point_clouds:
        pc = pc.clone()
        pc[:, axis] = -pc[:, axis]
        new_pcs.append(pc)
    data.point_clouds = new_pcs
    # 3. Remap face-encoded motion labels.
    def _remap(t, table):
        if t is None or t.numel() == 0: return t
        out = t.clone()
        for src, dst in table.items():
            if src != dst:
                out = torch.where(t == src, torch.tensor(dst, dtype=t.dtype), out)
        return out
    for attr in ("hinge_border", "rail_axis_signed", "handle_border",
                 "hinge_face_src", "hinge_face_dst"):
        if hasattr(data, attr):
            setattr(data, attr, _remap(getattr(data, attr), swap))
    if hasattr(data, "hinge_axis_signed"):
        data.hinge_axis_signed = _remap(data.hinge_axis_signed, axial)

    # 4. v3_pose_inv labels live in the dynamic part's thin-axis frame:
    #    non_thin = sorted non-thin axes of the dynamic node,
    #    hinge_border_4 = 2*slot + side   (border face on non_thin[slot]),
    #    hinge_axis_sign = sign of the rotation axis along non_thin[1-slot].
    #    Mirroring keeps extents (thin frame unchanged); the border side
    #    flips iff its axis IS the mirror axis, while the rotation-axis
    #    sign flips iff its axis is NOT the mirror axis (pseudo-vector).
    if (hasattr(data, "hinge_border_4") and data.hinge_border_4.numel() > 0
            and hasattr(data, "dyn_is_src")):
        b4 = data.hinge_border_4.clone()
        a2 = data.hinge_axis_sign.clone()
        ext = data.bbox[:, 3:] - data.bbox[:, :3]        # (N, 3) extents
        ei = data.edge_index
        for k in range(b4.numel()):
            if int(b4[k]) < 0:
                continue
            dis = int(data.dyn_is_src[k])
            if dis < 0:
                continue
            dyn = int(ei[0, k] if dis == 1 else ei[1, k])
            thin = int(torch.argmin(ext[dyn]))
            non_thin = [a for a in (0, 1, 2) if a != thin]
            slot = int(b4[k]) // 2
            if non_thin[slot] == axis:
                b4[k] = int(b4[k]) ^ 1
            if int(a2[k]) >= 0 and non_thin[1 - slot] != axis:
                a2[k] = int(a2[k]) ^ 1
        data.hinge_border_4 = b4
        data.hinge_axis_sign = a2
    # 5. Refresh `x` (centroid + bbox concatenation)
    data.x = torch.cat([data.pos, data.bbox], dim=1)
    return data

_STRATEGY_MATERIAL_MAP = {
    "drop_handles": 0,
    "drop_top":     9,
    "drop_bottom":  8,
    "drop_doors":   5,
    "drop_drawers": 6,
    "drop_shelf":   1,
    "drop_divider": 11,
}


def corrupt_functional_graph(func_data: Data,
                              node_mask_rate:     float = 0.2,
                              edge_mask_rate:     float = 0.0,
                              centroid_jitter:    float = 0.0,
                              material_mask_rate: float = 0.15,
                              rng: Optional[random.Random] = None) -> Data:
    """Generate a corrupted (unfunctional) view of a functional graph.

    Corruptions (input side only, target unchanged):
      1. Structured node dropping — randomly pick a strategy:
         - "random":        drop 10-30% of nodes uniformly.
         - "drop_handles":  drop all nodes with material=handle.
         - "drop_top":      drop all top panels.
         - "drop_bottom":   drop all bottom panels.
         - "drop_doors":    drop all doors.
         - "drop_drawers":  drop all drawers.
         If the chosen category is absent in this graph, fall back to random.
      2. Edge type collapse: surviving edges → all set to 'contact' (type 0).
         No edge dropping.
      3. No centroid jitter (geometry preserved exactly).
      4. Material masking: randomly set ~15% of surviving materials to UNKNOWN.
    """
    if rng is None:
        rng = random.Random()

    N = func_data.num_nodes
    mat = func_data.material

    # 1. Choose a strategy. Edge-preserving strategies optionally boosted
    # via _EDGE_PRESERVING_BOOST (set via set_edge_preserving_boost()).
    strategies = list(_DROP_STRATEGY_WEIGHTS.keys())
    weights = []
    for s in strategies:
        w = _DROP_STRATEGY_WEIGHTS[s]
        if s in _EDGE_PRESERVING_STRATEGIES:
            w *= _EDGE_PRESERVING_BOOST
        weights.append(w)
    strategy = rng.choices(strategies, weights=weights, k=1)[0]
    keep_mask = torch.ones(N, dtype=torch.bool)
    isolate_door_mode = (strategy == "isolate_door")
    fully_anonymized  = (strategy == "fully_anonymized")
    keep_edge_types   = strategy in _EDGE_PRESERVING_STRATEGIES   # NEW

    # ── Node-keep mask per strategy ──────────────────────────────────────────
    # Curriculum: drop rates anneal from "aggressive" (early) to "modest"
    # (late). Forces the existence head to see lots of "must fire" examples
    # before settling into mostly-complete graphs.
    rand_lo, rand_hi = _curriculum_drop_rate()  # uses _DROP_BANDS
    if strategy == "random":
        # Random drop EXCLUDING doors/drawers — those are the high-value
        # functional anchors; their categorisation is rare in PNM data and
        # we want the model to always retain them in the input view.
        DOOR, DRAWER = 5, 6
        rate = rng.uniform(rand_lo, rand_hi)
        drop_n = _cap_drop_n(int(N * rate))
        if drop_n > 0 and N - drop_n >= 1:
            allowed = ((mat != DOOR) & (mat != DRAWER)).nonzero(as_tuple=True)[0]
            if allowed.numel() > 0:
                drop_n = min(drop_n, allowed.numel())
                pick = allowed[torch.randperm(allowed.numel())[:drop_n]]
                keep_mask[pick] = False
    elif strategy == "fully_functional_drop":
        # Keep all functional edge types (set later) + drop 1..4 random nodes
        # excluding doors/drawers. Trains the model on partial-functional
        # graphs where some edges are already labelled — useful for cascaded
        # inference where a previous step has already committed to some edges.
        DOOR, DRAWER = 5, 6
        cap = _MAX_DROP_NODES if _MAX_DROP_NODES else 4
        drop_n = rng.randint(1, max(1, cap))
        allowed = ((mat != DOOR) & (mat != DRAWER)).nonzero(as_tuple=True)[0]
        if allowed.numel() > 0 and N - drop_n >= 1:
            drop_n = min(drop_n, allowed.numel())
            pick = allowed[torch.randperm(allowed.numel())[:drop_n]]
            keep_mask[pick] = False
    elif strategy == "drop_completion":
        # Multi-category drop: pick 2-3 of the most-commonly-missing categories
        # and drop them ALL. Mirrors PNM-style "incomplete graph" reality and
        # forces multiple free slots to fire on the same sample.
        cats = ["top panel", "shelf", "divider", "bottom panel", "handle"]
        # convert to material indices
        targets = [MATERIAL_VOCAB.get(c) for c in cats]
        targets = [t for t in targets if t is not None]
        n_drop_cats = rng.randint(2, 3)
        rng.shuffle(targets)
        chosen = targets[:n_drop_cats]
        for tm in chosen:
            cm = mat == tm
            if cm.any():
                keep_mask = keep_mask & ~cm
        # Cap the total drop count so it doesn't exceed _MAX_DROP_NODES.
        if _MAX_DROP_NODES is not None:
            dropped = (~keep_mask).nonzero(as_tuple=True)[0]
            if dropped.numel() > _MAX_DROP_NODES:
                # Randomly keep the excess
                excess = dropped.numel() - _MAX_DROP_NODES
                restore = dropped[torch.randperm(dropped.numel())[:excess]]
                keep_mask[restore] = True
        # Safety: never drop EVERY node.
        if int(keep_mask.sum()) < 1:
            keep_mask = torch.ones(N, dtype=torch.bool)
    elif strategy in _STRATEGY_MATERIAL_MAP:
        target_mat = _STRATEGY_MATERIAL_MAP[strategy]
        cat_mask = mat == target_mat
        cat_count = int(cat_mask.sum().item())
        # Cap: if dropping all of this category exceeds the cap, drop only
        # _MAX_DROP_NODES of them (chosen randomly).
        if cat_mask.any() and (N - cat_count) >= 1:
            if _MAX_DROP_NODES is not None and cat_count > _MAX_DROP_NODES:
                cat_idx = cat_mask.nonzero(as_tuple=True)[0]
                pick = cat_idx[torch.randperm(cat_count)[:_MAX_DROP_NODES]]
                keep_mask[pick] = False
            else:
                keep_mask[cat_mask] = False
        else:
            # Category absent → fall back to a random drop (curriculum-rated).
            rate = rng.uniform(rand_lo, rand_hi)
            drop_n = _cap_drop_n(int(N * rate))
            if drop_n > 0 and N - drop_n >= 1:
                keep_mask[torch.randperm(N)[:drop_n]] = False
    elif strategy == "drop_nodes_keep_edge_types":
        # Same node-drop as 'random' but edges of survivors keep their types.
        rate = rng.uniform(rand_lo, rand_hi)
        drop_n = _cap_drop_n(int(N * rate))
        if drop_n > 0 and N - drop_n >= 1:
            keep_mask[torch.randperm(N)[:drop_n]] = False
    # mask_random_edges_keep_types / mask_materials_keep_edges / isolate_door /
    # fully_anonymized → all nodes kept; edge logic below.

    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    old2new  = {old.item(): new for new, old in enumerate(keep_idx)}

    new_x            = func_data.x[keep_mask].clone()
    new_pos          = func_data.pos[keep_mask].clone()
    new_bbox         = func_data.bbox[keep_mask].clone()
    new_material     = func_data.material[keep_mask].clone()
    hb_src = getattr(func_data, "handle_border", None)
    new_handle_border = hb_src[keep_mask].clone() if hb_src is not None else None

    # ── Material masking ────────────────────────────────────────────────────
    if fully_anonymized:
        new_material = torch.full_like(new_material, MATERIAL_UNKNOWN)
    elif strategy == "mask_materials_keep_edges":
        # Heavier masking — model must lean on graph topology + edges.
        rate = rng.uniform(0.30, 0.60)
        mask_mat = torch.rand(new_material.shape[0]) < rate
        new_material = new_material.clone()
        new_material[mask_mat] = MATERIAL_UNKNOWN
    elif material_mask_rate > 0:
        mask_mat = torch.rand(new_material.shape[0]) < material_mask_rate
        new_material = new_material.clone()
        new_material[mask_mat] = MATERIAL_UNKNOWN

    new_pc     = [func_data.point_clouds[i] for i in keep_idx.tolist()]
    new_source = [func_data.pc_source[i]    for i in keep_idx.tolist()]
    new_names  = [func_data.node_names[i]   for i in keep_idx.tolist()]

    # ── Edge filtering & typing ─────────────────────────────────────────────
    ei  = func_data.edge_index
    ea  = func_data.edge_attr   # original types: contact/hinge/rail/attached
    valid = keep_mask[ei[0]] & keep_mask[ei[1]]

    # isolate_door: drop every edge that touches a door node.
    if isolate_door_mode:
        door_idx = (func_data.material == 5).nonzero(as_tuple=True)[0]
        if door_idx.numel() > 0:
            door_set = set(door_idx.tolist())
            touches_door = torch.tensor(
                [s.item() in door_set or d.item() in door_set
                 for s, d in zip(ei[0], ei[1])],
                dtype=torch.bool,
            )
            valid = valid & ~touches_door

    # mask_random_edges_keep_types: randomly drop 25-50% of surviving edges.
    if strategy == "mask_random_edges_keep_types" and valid.any():
        keep_rate = 1.0 - rng.uniform(0.25, 0.50)
        edge_keep = (torch.rand(valid.shape[0]) < keep_rate)
        valid = valid & edge_keep

    ei_v = ei[:, valid]
    ea_v = ea[valid]

    if ei_v.shape[1] > 0:
        new_src = torch.tensor([old2new[s.item()] for s in ei_v[0]], dtype=torch.long)
        new_dst = torch.tensor([old2new[d.item()] for d in ei_v[1]], dtype=torch.long)
        new_ei  = torch.stack([new_src, new_dst], dim=0)
        if keep_edge_types:
            new_ea = ea_v.clone()                     # original types preserved
        else:
            new_ea = torch.zeros(new_ei.shape[1], dtype=torch.long)   # collapse → contact
    else:
        new_ei = torch.zeros((2, 0), dtype=torch.long)
        new_ea = torch.zeros(0, dtype=torch.long)

    corrupt = Data(
        x=new_x,
        edge_index=new_ei,
        edge_attr=new_ea,
        pos=new_pos,
        bbox=new_bbox,
        material=new_material,
        num_nodes=keep_mask.sum().item(),
    )
    corrupt.point_clouds = new_pc
    corrupt.pc_source    = new_source
    corrupt.node_names   = new_names
    corrupt.model_id     = getattr(func_data, "model_id", "unknown") + "_corrupt"
    # Motion labels: not meaningful for corrupted inputs (all edges → contact),
    # but must exist for PyG batching compatibility
    n_edges = corrupt.edge_index.shape[1] if corrupt.edge_index.numel() > 0 else 0
    corrupt.hinge_face_src = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.hinge_face_dst = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.hinge_axis     = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.hinge_dir      = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.rail_axis      = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.hinge_border      = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.hinge_axis_signed = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.hinge_border_4    = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.hinge_axis_sign   = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.rail_axis_signed  = torch.full((n_edges,), -1, dtype=torch.long)
    corrupt.dyn_is_src        = torch.full((n_edges,), -1, dtype=torch.long)
    if new_handle_border is not None:
        corrupt.handle_border = new_handle_border
    else:
        corrupt.handle_border = torch.full(
            (int(keep_mask.sum().item()),), -1, dtype=torch.long)
    return corrupt


# ── Dataset ───────────────────────────────────────────────────────────────────

class FurFunGraphDataset(Dataset):
    """Dataset of (input_graph, target_graph) pairs for graph-to-graph translation.

    Sources
    -------
    1. Real paired samples: graph_func/ (target) + graph_unfunc/ (input).
    2. Augmented samples: randomly corrupted functional graphs used as inputs.

    Point clouds per node are loaded from ``geom_root/<model_id>/<node>.npy``
    (pre-exported from Blender + surface-sampled).  Nodes without a geometry
    file fall back to OBB surface sampling.

    Parameters
    ----------
    func_root      : path to graph_func/ directory.
    unfunc_root    : path to graph_unfunc/ directory (optional).
    geom_root      : path to part_geometries/ directory (optional).
    n_pc_points    : points to return per node (sub/super-sampled from stored cloud).
    node_mask_rate : node drop rate for augmented corruption.
    edge_mask_rate : edge drop rate for augmented corruption.
    augment_ratio  : ratio of augmented pairs to add (1.0 = 1× extra).
    seed           : reproducibility seed.
    """

    def __init__(self,
                 func_root:      str,
                 unfunc_root:    Optional[str] = None,
                 geom_root:      Optional[str] = None,
                 n_pc_points:    int   = 512,
                 node_mask_rate: float = 0.2,
                 edge_mask_rate: float = 0.3,
                 augment_ratio:  float = 1.0,
                 identity_pair_rate: float = 0.0,
                 cache_in_ram:   bool = False,
                 seed:           int   = 42,
                 scaled_geom_root: Optional[str] = None,
                 scale_variants:   Optional[List[Dict]]  = None,
                 p_scale:          float = 0.0,
                 mirror_p:         float = 0.5,
                 mirror_axes:      tuple = (0, 1),
                 require_connected: bool = True):
        super().__init__()
        self.func_root      = Path(func_root)
        self.unfunc_root    = Path(unfunc_root) if unfunc_root else None
        self.geom_root      = Path(geom_root) if geom_root else _DEFAULT_GEOM_ROOT
        self.n_pc_points    = n_pc_points
        self.node_mask_rate = node_mask_rate
        self.edge_mask_rate = edge_mask_rate
        self.augment_ratio  = augment_ratio
        # (b-recipe) fraction of AUGMENTED draws served as complete-input
        # pairs (unfunc input, no node drops) — explicit "fire nothing"
        # supervision for the free-slot exist heads.
        self.identity_pair_rate = float(identity_pair_rate)
        # RAM memoization of parsed graphs (+ sampled point clouds). The
        # loader re-parses identical JSON + reloads .npy every epoch
        # otherwise; caching gives ~3x epoch speedup at ~100MB RAM. Cached
        # graphs are .clone()d before any mutating transform.
        self._graph_cache = {} if cache_in_ram else None
        self.seed           = seed
        # mirror_p: per-sample probability of applying a random axis flip
        # (X or Y mirror, never Z — keeps "up" semantics intact). Doubles
        # effective data + helps left/right symmetric handedness.
        self.mirror_p       = float(mirror_p)
        self.mirror_axes    = tuple(int(a) for a in mirror_axes) or (0,)
        # require_connected: skip graphs with multiple connected components
        # at index-build time. Catches the 3 known fur_* cases (fur_072 mesh-
        # screen casualty + fur_114, fur_116 pre-existing data flaws).
        self.require_connected = bool(require_connected)
        self._rng_np        = np.random.default_rng(seed)
        self._rng_py        = random.Random(seed)

        # Optional per-sample scale augmentation. When `scale_variants` is
        # provided and `p_scale > 0`, training samples are randomly drawn from
        # a pre-generated scaled geometry directory; `pos`/`bbox` are
        # multiplied by the variant's (sx, sy, sz).
        self.scaled_geom_root = Path(scaled_geom_root) if scaled_geom_root else None
        self.scale_variants   = scale_variants or []
        self.p_scale          = float(p_scale)

        self._pairs:                List[Tuple[str, Optional[str]]] = []
        self._augmented_func_paths: List[str] = []
        self._build_index()

    def _build_index(self):
        func_jsons: Dict[str, str] = {}
        n_skipped_empty = 0
        n_skipped_disconnected = 0
        for sample_dir in sorted(self.func_root.iterdir()):
            if not sample_dir.is_dir():
                continue
            jsons = list(sample_dir.glob("*.json"))
            if not jsons:
                continue
            # Skip empty graphs (e.g. glasses_*: status=ok but nodes={}).
            try:
                with open(jsons[0]) as f:
                    raw = json.load(f)
            except Exception:
                continue
            if not raw.get("nodes"):
                n_skipped_empty += 1
                continue
            # Skip disconnected graphs — a "functional" furniture graph should
            # have all parts in one connected component. Train signal from
            # disconnected graphs is unreliable (free-floating components
            # confuse the contact-edge head).
            if self.require_connected and not _graph_is_connected(raw):
                n_skipped_disconnected += 1
                continue
            func_jsons[sample_dir.name] = str(jsons[0])
        if n_skipped_empty or n_skipped_disconnected:
            print(f"[FurFunGraphDataset] skipped {n_skipped_empty} empty, "
                  f"{n_skipped_disconnected} disconnected graphs at index-build")

        for model_id, fpath in func_jsons.items():
            unfunc_path = None
            if self.unfunc_root is not None:
                cand = self.unfunc_root / model_id
                if cand.is_dir():
                    jsons = list(cand.glob("*.json"))
                    if jsons:
                        unfunc_path = str(jsons[0])
            self._pairs.append((fpath, unfunc_path))

        n_aug = int(len(self._pairs) * self.augment_ratio)
        func_paths = [p for p, _ in self._pairs]
        self._augmented_func_paths = [
            func_paths[i % len(func_paths)] for i in range(n_aug)
        ]

    # -- stats helper --
    def pc_coverage(self) -> Dict[str, int]:
        """Return counts of nodes with real vs OBB point clouds in the func set."""
        real = obb = 0
        for fpath, _ in self._pairs:
            g = load_graph_json(fpath, n_pc_points=4, geom_root=self.geom_root,
                                rng=np.random.default_rng(0))
            real += sum(1 for s in g.pc_source if s == "real")
            obb  += sum(1 for s in g.pc_source if s == "obb")
        return {"real": real, "obb": obb}

    def len(self) -> int:
        return len(self._pairs) + len(self._augmented_func_paths)

    def _maybe_scale(self, data: Data):
        """Randomly pick a scale variant and apply it to pos/bbox/point_clouds.
        Returns (scale_tuple, geom_root) so the caller can load PCs from the
        matching pre-generated directory."""
        if (self.p_scale <= 0
                or not self.scale_variants
                or self.scaled_geom_root is None):
            return (1.0, 1.0, 1.0), self.geom_root
        if self._rng_py.random() >= self.p_scale:
            return (1.0, 1.0, 1.0), self.geom_root
        v = self._rng_py.choice(self.scale_variants)
        sx, sy, sz = v["sx"], v["sy"], v["sz"]
        return (sx, sy, sz), self.scaled_geom_root / v["name"]

    def _apply_scale_to_graph(self, data: Data, scale_tuple):
        sx, sy, sz = scale_tuple
        if sx == 1.0 and sy == 1.0 and sz == 1.0:
            return data
        s = torch.tensor([sx, sy, sz], dtype=torch.float32)
        data.pos  = data.pos  * s
        # bbox = [min_xyz, max_xyz], both scaled
        data.bbox = data.bbox * s.repeat(2)
        data.x    = torch.cat([data.pos, data.bbox], dim=1)
        # Point clouds — sampled from the scaled PLY already (so the centroids
        # they represent are already in scaled world space).  But we also want
        # the scale-variant's PC to be in the per-node relative frame; since
        # they're stored in world space, they're consistent with pos/bbox.
        return data

    def _load_cached(self, path, geom_root):
        if self._graph_cache is None:
            return load_graph_json(path, self.n_pc_points, geom_root, self._rng_np)
        key = (str(path), str(geom_root))
        g = self._graph_cache.get(key)
        if g is None:
            g = load_graph_json(path, self.n_pc_points, geom_root, self._rng_np)
            self._graph_cache[key] = g
        return g.clone()

    def get(self, idx: int) -> Tuple[Data, Data]:
        """Return (input_graph, target_graph)."""
        scale_tuple, geom_root = self._maybe_scale(None)

        if idx < len(self._pairs):
            func_path, unfunc_path = self._pairs[idx]
            target = self._load_cached(func_path, geom_root)
            target = self._apply_scale_to_graph(target, scale_tuple)
            if unfunc_path is not None:
                inp = self._load_cached(unfunc_path, geom_root)
                inp = self._apply_scale_to_graph(inp, scale_tuple)
            else:
                inp = corrupt_functional_graph(
                    target, self.node_mask_rate, self.edge_mask_rate, rng=self._rng_py)
        else:
            aug_idx   = idx - len(self._pairs)
            func_path = self._augmented_func_paths[aug_idx]
            target    = self._load_cached(func_path, geom_root)
            target    = self._apply_scale_to_graph(target, scale_tuple)
            if (self.identity_pair_rate > 0
                    and self._rng_py.random() < self.identity_pair_rate):
                # complete-input pair: same nodes, contact-only edges — the
                # correct completion action is to add NOTHING.
                unfunc_path = self._pairs[aug_idx % len(self._pairs)][1]
                if unfunc_path is not None:
                    inp = self._load_cached(unfunc_path, geom_root)
                    inp = self._apply_scale_to_graph(inp, scale_tuple)
                else:
                    inp = corrupt_functional_graph(
                        target, 0.0, self.edge_mask_rate, rng=self._rng_py)
            else:
                inp = corrupt_functional_graph(
                    target, self.node_mask_rate, self.edge_mask_rate, rng=self._rng_py)

        # Mirror/flip augmentation: with prob mirror_p, flip both input and
        # target along the SAME axis. Doubles effective data + helps
        # left-right symmetric handedness (face indices remap accordingly).
        if self.mirror_p > 0 and self._rng_py.random() < self.mirror_p:
            axis = self._rng_py.choice(list(self.mirror_axes))   # never Z (preserves up)
            inp    = _apply_mirror(inp,    axis)
            target = _apply_mirror(target, axis)

        # Anchored-slot identity map: for each input node, the index of the
        # SAME part in the target node list (matched by part id, verified
        # geometrically). The pinned Hungarian matcher uses it so anchored
        # slot k can only ever match its own part — without it, mirror
        # augmentation lets the matcher swap symmetric siblings and the
        # gradients teach identity-permuted layouts. -1 = never pinned.
        n_in = int(inp.pos.shape[0])
        idxs = torch.full((n_in,), -1, dtype=torch.long)
        t_names = list(getattr(target, "node_names", []) or [])
        i_names = list(getattr(inp, "node_names", []) or [])
        lookup = {nid: j for j, nid in enumerate(t_names)}
        for k in range(min(n_in, len(i_names))):
            j = lookup.get(i_names[k], -1)
            if j >= 0 and float((inp.pos[k] - target.pos[j]).norm()) < 0.05:
                idxs[k] = j
        inp.anchor_gt_idx = idxs

        return inp, target

    def __getitem__(self, idx: int):
        return self.get(idx)

    def __len__(self) -> int:
        return self.len()

    def model_id_of_index(self, idx: int) -> str:
        """Return the model_id that dataset index `idx` belongs to."""
        if idx < len(self._pairs):
            return Path(self._pairs[idx][0]).parent.name
        return Path(self._augmented_func_paths[idx - len(self._pairs)]).parent.name


def split_by_model_id(
    dataset: "FurFunGraphDataset",
    test_frac: float = 0.1,
    val_frac:  float = 0.1,
    seed:      int   = 42,
):
    """Return (train, val, test) Subsets that never share a model_id.

    Groups all dataset indices by model_id, shuffles the unique model_ids
    deterministically, then assigns whole groups to each split.
    """
    from torch.utils.data import Subset

    groups: Dict[str, List[int]] = {}
    for i in range(len(dataset)):
        mid = dataset.model_id_of_index(i)
        groups.setdefault(mid, []).append(i)

    mids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(mids)

    n = len(mids)
    n_test = max(1, int(round(n * test_frac)))
    n_val  = max(1, int(round(n * val_frac)))
    n_train = n - n_val - n_test
    assert n_train > 0, f"Too few models ({n}) for split ({n_train} train)"

    test_mids  = mids[:n_test]
    val_mids   = mids[n_test:n_test + n_val]
    train_mids = mids[n_test + n_val:]

    def flatten(ms):
        out = []
        for m in ms:
            out.extend(groups[m])
        return out

    return (Subset(dataset, flatten(train_mids)),
            Subset(dataset, flatten(val_mids)),
            Subset(dataset, flatten(test_mids)),
            {"train_mids": train_mids, "val_mids": val_mids, "test_mids": test_mids})


# ── Collation helper ──────────────────────────────────────────────────────────

def collate_graph_pairs(batch: List[Tuple[Data, Data]]):
    """Custom collate for (input, target) pairs.

    Returns two PyG Batch objects.  Point clouds are stored as a flat list
    alongside a *pc_node_batch* index mapping each node to its graph index.
    """
    from torch_geometric.data import Batch

    inputs, targets = zip(*batch)

    def batch_with_pc(graphs):
        # Pull out list attributes that PyG can't batch automatically
        pc_lists   = [g.point_clouds for g in graphs]
        src_lists  = [g.pc_source    for g in graphs]
        name_lists = [g.node_names   for g in graphs]
        mid_list   = [g.model_id     for g in graphs]
        for g in graphs:
            del g.point_clouds
            del g.pc_source
            del g.node_names
            del g.model_id

        batched = Batch.from_data_list(list(graphs))
        batched.point_clouds  = [pc for pcs in pc_lists   for pc in pcs]
        batched.pc_source     = [s  for ss  in src_lists  for s  in ss]
        batched.node_names    = [n  for ns  in name_lists for n  in ns]
        batched.model_id      = mid_list
        batched.pc_node_batch = batched.batch    # node → graph index

        for g, pcs, srcs, names, mid in zip(graphs, pc_lists, src_lists, name_lists, mid_list):
            g.point_clouds = pcs
            g.pc_source    = srcs
            g.node_names   = names
            g.model_id     = mid

        return batched

    return batch_with_pc(list(inputs)), batch_with_pc(list(targets))
