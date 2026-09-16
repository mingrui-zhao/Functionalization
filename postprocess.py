"""Geometric fixes for predicted functional graphs (release post-processing).

Applied to raw prediction JSONs:
  1. handle dedup    — NEW handles on a parent that already has an anchored
                       (input) handle are dropped; among the remaining NEW
                       (free-slot) handles attached to the same
                       door/drawer, keep only the best-placed one (nearest to
                       its parent's OBB center); drop the rest.
  2. orphan removal  — drop NEW handles whose attached parent does not exist
                       in the final graph or is not a door/drawer, and NEW
                       non-handle nodes with no edge to any kept node.

New-node material whitelisting (top panel / shelf / divider / handle) is
enforced structurally inside the decoder and needs no post-processing.

Usage:
    python postprocess.py --pred_dir <dir> [--out <dir>]      # batch
or  from postprocess import postprocess_pred                   # single dict
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

COMPLETION_MATS = {"handle", "shelf", "top panel", "divider"}

# ── Functional-edge type system ──────────────────────────────────────────────
# hinge:    door    ↔ static support (never handle, never two movables)
# rail:     drawer  ↔ static support
# attached: handle  ↔ door | drawer
# contact:  unconstrained
_MOVABLE = {"door", "drawer"}
_STATIC_SUPPORT = {"side panel", "back panel", "bottom panel", "top panel",
                   "face frame", "divider", "shelf", "misc", "bar",
                   "countertop", "leg", "unknown"}


def _edge_kind_legal(kind: str, mat_a: str, mat_b: str) -> bool:
    if kind == "hinge":
        return ((mat_a == "door" and mat_b in _STATIC_SUPPORT) or
                (mat_b == "door" and mat_a in _STATIC_SUPPORT))
    if kind == "rail":
        return ((mat_a == "drawer" and mat_b in _STATIC_SUPPORT) or
                (mat_b == "drawer" and mat_a in _STATIC_SUPPORT))
    if kind == "attached":
        return ((mat_a == "handle" and mat_b in _MOVABLE) or
                (mat_b == "handle" and mat_a in _MOVABLE))
    return True   # contact etc.


def _nodes_list(pred):
    n = pred["nodes"]
    return n if isinstance(n, list) else list(n.values())


def postprocess_pred(pred: dict, n_input: int) -> dict:
    """Return a new pred dict with dedup + orphan fixes applied.
    `n_input` = number of anchored slots (= nodes in the input graph)."""
    nodes = _nodes_list(pred)
    ids = [n["id"] for n in nodes]
    new_ids = set(ids[n_input:])
    mat = {n["id"]: (n.get("material") or "").replace("_", " ") for n in nodes}
    center = {n["id"]: np.asarray(n["obb"]["center"], float) for n in nodes}
    # 0. drop type-system-illegal functional edges (e.g. hinge on a handle,
    #    rail on a shelf) — see _edge_kind_legal
    edges = [e for e in pred["edges"]
             if _edge_kind_legal(e.get("kind", "contact"),
                                 mat.get(e["src"], ""), mat.get(e["dst"], ""))]

    def parent_of(handle_id):
        for e in edges:
            if e.get("kind") != "attached":
                continue
            if e["src"] == handle_id:
                return e["dst"]
            if e["dst"] == handle_id:
                return e["src"]
        return None

    drop = set()
    # 1a. a movable part that already carries an ANCHORED (input) handle
    #     needs no synthesized one: drop every NEW handle attached to it.
    anchored_handle_parents = set()
    for n in nodes:
        nid = n["id"]
        if mat.get(nid) == "handle" and nid not in new_ids:
            p = parent_of(nid)
            if p is not None:
                anchored_handle_parents.add(p)
    # 1b. group NEW handles by parent, keep best-placed per parent
    groups: dict = {}
    for nid in new_ids:
        if mat.get(nid) != "handle":
            continue
        p = parent_of(nid)
        if p is None or p in drop or p not in mat or mat[p] not in ("door", "drawer"):
            drop.add(nid)          # orphan: no valid door/drawer parent
            continue
        if p in anchored_handle_parents:
            drop.add(nid)          # parent already has a real handle
            continue
        groups.setdefault(p, []).append(nid)
    for p, hs in groups.items():
        if len(hs) <= 1:
            continue
        hs.sort(key=lambda h: float(np.linalg.norm(center[h] - center[p])))
        drop.update(hs[1:])
    # 2. NEW non-handle nodes must be REACHABLE from the input model — an
    #    edge count alone would keep islands of new nodes wired only to
    #    each other, which are still orphans.
    anchored_ids = set(ids[:n_input])
    adj: dict = {}
    for e in edges:
        adj.setdefault(e["src"], set()).add(e["dst"])
        adj.setdefault(e["dst"], set()).add(e["src"])
    reach = set(anchored_ids)
    frontier = list(anchored_ids)
    while frontier:
        cur = frontier.pop()
        for nxt in adj.get(cur, ()):
            if nxt not in reach and nxt not in drop:
                reach.add(nxt)
                frontier.append(nxt)
    for nid in new_ids:
        if nid in drop or mat.get(nid) == "handle":
            continue
        if nid not in reach:
            drop.add(nid)

    kept_nodes = [n for n in nodes if n["id"] not in drop]
    kept_ids = {n["id"] for n in kept_nodes}
    kept_edges = [e for e in edges if e["src"] in kept_ids and e["dst"] in kept_ids]
    out = dict(pred)
    # preserve the input container type (slot order matters for anchoring)
    if isinstance(pred["nodes"], dict):
        out["nodes"] = {n["id"]: n for n in kept_nodes}
    else:
        out["nodes"] = kept_nodes
    out["edges"] = kept_edges
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True,
                    help="Directory of <mid>_pred.json files")
    ap.add_argument("--input_dir", required=True,
                    help="Directory of matching <mid>_input.json (or <mid>/<mid>.json) "
                         "graphs, used for the anchored-slot count")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <pred_dir>_postproc)")
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out) if args.out else pred_dir.parent / (pred_dir.name + "_postproc")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_changed = 0
    for f in sorted(pred_dir.glob("*_pred.json")):
        mid = f.stem.replace("_pred", "")
        gin = None
        for cand in (input_dir / f"{mid}_input.json", input_dir / mid / f"{mid}.json"):
            if cand.exists():
                gin = json.loads(cand.read_text())
                break
        if gin is None:
            print(f"  [{mid}] no input graph found — skipped")
            continue
        pred = json.loads(f.read_text())
        fixed = postprocess_pred(pred, len(gin["nodes"]))
        if len(_nodes_list(fixed)) != len(_nodes_list(pred)):
            n_changed += 1
        (out_dir / f.name).write_text(json.dumps(fixed))
    print(f"post-processed → {out_dir}  (modified {n_changed} graphs)")


if __name__ == "__main__":
    main()
