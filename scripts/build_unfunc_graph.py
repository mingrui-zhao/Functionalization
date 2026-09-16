"""Build an unfunctional (contact-only) input graph from per-part meshes.

Input:  a model directory containing per-part meshes named
        <material>_<idx>.obj  (e.g. side_panel_0.obj, door_0.obj, shelf_1.obj),
        either directly or under an objs/ subfolder. Underscored material
        names map to the public vocabulary ("side_panel" → "side panel").
Output: <out>/<mid>/<mid>.json with the release graph schema:
        nodes: {id: {id, type:"part", material, obb{center,half,quat}}}
        edges: [{src, dst, kind:"contact", connector:null}]

Contact rule: axis-aligned bounding boxes within CONTACT_TOL (5 mm in
normalized units) on every axis — the same tolerance as the shipped test
sets. Meshes are expected in a normalized frame (unit-ish scale, Z-up).

Usage:
    python build_unfunc_graph.py --mesh_dir <model_dir> --out <graphs_dir>
    python build_unfunc_graph.py --mesh_root <root_of_models> --out <graphs_dir>
"""
from __future__ import annotations
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.graph_dataset import PUBLIC_MATERIALS  # noqa: E402

CONTACT_TOL = 0.005


def load_obj_bounds(path: Path):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                v = np.asarray([float(x) for x in line.split()[1:4]])
                lo = np.minimum(lo, v)
                hi = np.maximum(hi, v)
    if not np.isfinite(lo).all():
        return None
    return lo, hi


def material_of(stem: str):
    m = re.match(r"(.+?)_(\d+)$", stem)
    base = (m.group(1) if m else stem).replace("_", " ").lower()
    return base if base in PUBLIC_MATERIALS else "misc"


def build_one(mesh_dir: Path, out_root: Path):
    mid = mesh_dir.name
    obj_dir = mesh_dir / "objs" if (mesh_dir / "objs").is_dir() else mesh_dir
    objs = sorted(obj_dir.glob("*.obj"))
    if not objs:
        print(f"  [{mid}] no .obj meshes — skipped")
        return False
    nodes = {}
    for p in objs:
        b = load_obj_bounds(p)
        if b is None:
            continue
        lo, hi = b
        nodes[p.stem] = {
            "id": p.stem, "type": "part", "material": material_of(p.stem),
            "obb": {"center": [round(float(x), 6) for x in (lo + hi) / 2],
                    "half":   [round(float(x), 6) for x in (hi - lo) / 2],
                    "quat":   [1.0, 0.0, 0.0, 0.0]},
        }
    edges = []
    for a, b in itertools.combinations(nodes, 2):
        ca = np.asarray(nodes[a]["obb"]["center"]); ha = np.asarray(nodes[a]["obb"]["half"])
        cb = np.asarray(nodes[b]["obb"]["center"]); hb = np.asarray(nodes[b]["obb"]["half"])
        if float((np.abs(ca - cb) - (ha + hb)).max()) <= CONTACT_TOL:
            edges.append({"src": a, "dst": b, "kind": "contact", "connector": None})
    out_dir = out_root / mid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{mid}.json").write_text(json.dumps(
        {"model_id": mid, "nodes": nodes, "edges": edges}))
    print(f"  [{mid}] {len(nodes)} nodes, {len(edges)} contact edges")
    return True


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mesh_dir", help="single model directory")
    g.add_argument("--mesh_root", help="root containing one directory per model")
    ap.add_argument("--out", required=True, help="output graphs directory")
    args = ap.parse_args()
    out_root = Path(args.out)
    if args.mesh_dir:
        build_one(Path(args.mesh_dir), out_root)
    else:
        n = sum(build_one(d, out_root)
                for d in sorted(Path(args.mesh_root).iterdir()) if d.is_dir())
        print(f"built {n} graphs → {out_root}")


if __name__ == "__main__":
    main()
