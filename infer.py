"""GraFu inference: unfunctional graph → predicted functional graph.

Release semantics (enforced by construction, no flags needed):
  * public material vocabulary = 14 part materials (+ "unknown" fallback).
    `hinge` / `rail` are joint EDGE kinds, never node materials — such input
    labels map to unknown, and the decoder cannot predict them.
  * new (free-slot) nodes can only be completion parts: top panel, shelf,
    divider, handle (masked structurally in the decoder in eval mode).
  * geometric fixes run by default: per-door/drawer new-handle dedup and
    orphan removal (see postprocess.py). Disable with --no-postprocess.

Usage:
    python infer.py --dataset pnm                       # 345 clean PNM graphs
    python infer.py --dataset hssd                      # 50 HSSD test graphs
    python infer.py --input_dir <dir> [--geom_root <dir>]   # custom graphs
Input layouts supported: <dir>/<mid>/<mid>.json or <dir>/<mid>_input.json.
Outputs <out>/<mid>_pred.json.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.graph_dataset import load_graph_json, PUBLIC_MATERIALS, _INTERNAL_ONLY
from infer_common import (load_model, pack_single, pred_to_json,
                          summarize_graph)
from postprocess import postprocess_pred

DATASETS = {
    "pnm":  dict(input_dir=ROOT / "datasets/pnm_clean/graphs_unfunctional",
                 geom_root=ROOT / "datasets/pnm_clean/part_geometries"),
    "hssd": dict(input_dir=ROOT / "datasets/hssd_test/graph_unfunc",
                 geom_root=ROOT / "datasets/hssd_test/part_geometries"),
}


def discover_graphs(input_dir: Path):
    """Yield (mid, json_path) for both supported layouts."""
    out = []
    for d in sorted(input_dir.iterdir()):
        if d.is_dir():
            j = d / f"{d.name}.json"
            if j.exists():
                out.append((d.name, j))
        elif d.suffix == ".json" and d.stem.endswith("_input"):
            out.append((d.stem[:-len("_input")], d))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="GraFu inference: predict a functional graph per model, "
                    "then (by default) realize each prediction as an "
                    "animated .blend via Blender.")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/grafu_full.pt"),
                    help="Model checkpoint (.pt).")
    ap.add_argument("--dataset", choices=sorted(DATASETS),
                    default=None,
                    help="Built-in test set preset (pnm | hssd).")
    ap.add_argument("--input_dir", default=None,
                    help="Custom unfunctional-graph dir (overrides --dataset).")
    ap.add_argument("--geom_root", default=None,
                    help="Per-part mesh root for point-cloud features; parts "
                         "without meshes fall back to OBB-surface sampling.")
    ap.add_argument("--out", default=str(ROOT / "results"),
                    help="Output directory for predictions + manifest.")
    ap.add_argument("--n", type=int, default=0,
                    help="Only run the first N models (0 = all).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Existence threshold for new (free-slot) nodes.")
    ap.add_argument("--max_new_nodes", type=int, default=None,
                    help="Cap new-node firings per graph (default: unlimited).")
    ap.add_argument("--dyn_type_mutex", action="store_true",
                    help="Force each door/drawer slot to exactly one dynamic "
                         "type based on its strongest hinge-vs-rail edge.")
    ap.add_argument("--no-postprocess", dest="postprocess", action="store_false",
                    help="Skip the geometric postprocess (orphan removal, "
                         "joint-type-system edge checks).")
    ap.add_argument("--no-blenderize", dest="blenderize", action="store_false",
                    help="Skip realizing each prediction as an animated .blend.")
    ap.add_argument("--blender", default=None,
                    help="Blender binary (default: $BLENDER, then PATH).")
    ap.add_argument("--hinge", default="auto",
                    choices=["auto", "interior", "exterior", "flat"],
                    help="Hinge policy for blenderization (auto = per-joint "
                         "swing-collision selection).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for point-cloud sampling.")
    args = ap.parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()
        geom_root = Path(args.geom_root).resolve() if args.geom_root else None
    elif args.dataset:
        input_dir = DATASETS[args.dataset]["input_dir"]
        geom_root = DATASETS[args.dataset]["geom_root"]
    else:
        ap.error("pass --dataset pnm|hssd or --input_dir <dir>")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.checkpoint, device)
    model.eval()   # activates the structural material constraints

    graphs = discover_graphs(input_dir)
    if args.n:
        graphs = graphs[: args.n]
    print(f"{len(graphs)} graphs from {input_dir}")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    # Manifest mapping mid -> its input graph + mesh dir. The Blender
    # add-on reads this to resolve load paths with zero configuration.
    manifest_path = out_root / "inference_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    n_failed = 0
    for mid, json_path in graphs:
        # One bad model must not abort the run: report it, keep going, and
        # keep the manifest current for the models that did complete.
        try:
            g = json.loads(json_path.read_text())
            bad = sorted({(n.get("material") or "") for n in
                          (g["nodes"].values() if isinstance(g["nodes"], dict) else g["nodes"])
                          if (n.get("material") or "").replace("_", " ") in _INTERNAL_ONLY})
            if bad:
                print(f"  [{mid}] note: input labels {bad} are edge kinds, not "
                      f"materials — treated as 'unknown'")
            rng_np = np.random.default_rng(args.seed)
            data = load_graph_json(str(json_path), n_pc_points=cfg["n_pc_points"],
                                   geom_root=geom_root, rng=rng_np)
            data.model_id = mid
            with torch.no_grad():
                out = model(pack_single(data, device))
            pred = pred_to_json(out, 0, mid, args.threshold,
                                max_new_nodes=args.max_new_nodes,
                                dyn_type_mutex=args.dyn_type_mutex)
            if args.postprocess:
                pred = postprocess_pred(pred, n_input=data.num_nodes)
        except Exception as e:
            n_failed += 1
            print(f"  {mid}: FAILED ({type(e).__name__}: {e}) — skipping")
            continue
        (out_root / f"{mid}_pred.json").write_text(json.dumps(pred, indent=2))
        mesh_dir = (geom_root / mid) if geom_root else None
        manifest[mid] = {
            "input": str(json_path),
            "mesh_dir": str(mesh_dir) if mesh_dir and mesh_dir.is_dir() else None,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        nn, mats, ek, motion = summarize_graph(pred)
        print(f"  {mid}: {data.num_nodes} → {nn} nodes  {mats}  edges {ek}")

    if n_failed:
        print(f"  {n_failed} model(s) failed — see messages above")

    # ── Geometry realization: one animated .blend per prediction ──────────
    if args.blenderize:
        blender = (args.blender or os.environ.get("BLENDER")
                   or shutil.which("blender"))
        if blender is not None and shutil.which(blender) is None:
            # An explicit --blender/$BLENDER that doesn't resolve must not
            # crash the run after all predictions succeeded.
            print(f"\nblenderize: Blender binary not found at {blender!r} "
                  f"— skipping realization.")
            blender = None
        installer = ROOT / "blender_tools" / "install_from_pred.py"
        if blender is None:
            print("\nblenderize: no Blender binary found (set --blender or "
                  "$BLENDER, or pass --no-blenderize) — skipping.")
        else:
            blend_dir = out_root / "blends"
            blend_dir.mkdir(exist_ok=True)
            n_ok = n_err = n_skip = 0
            for mid, json_path in graphs:
                pred_path = out_root / f"{mid}_pred.json"
                if mid not in manifest or not pred_path.is_file():
                    continue   # prediction failed above
                mesh_dir = (geom_root / mid) if geom_root else None
                if mesh_dir is None or not mesh_dir.is_dir():
                    print(f"  blenderize {mid}: SKIP (no mesh dir)")
                    n_skip += 1
                    continue
                out_blend = blend_dir / f"{mid}.blend"
                cmd = [blender, "--background", "--factory-startup",
                       "--python-exit-code", "1",
                       "--python", str(installer), "--",
                       f"--pred={pred_path}",
                       f"--input={json_path}",
                       f"--mesh_dir={mesh_dir}",
                       f"--hinge={args.hinge}",
                       "--drop-hallucinated",
                       f"--out={out_blend}"]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=900)
                    ok = r.returncode == 0 and out_blend.is_file()
                    tail = ((r.stderr or "").strip().splitlines()[-3:]
                            or (r.stdout or "").strip().splitlines()[-3:])
                except subprocess.TimeoutExpired:
                    ok = False
                    tail = ["(timed out after 900s)"]
                if ok:
                    manifest[mid]["blend"] = str(out_blend)
                    n_ok += 1
                    print(f"  blenderize {mid}: OK → {out_blend.name}")
                else:
                    n_err += 1
                    print(f"  blenderize {mid}: ERR" +
                          ("".join(f"\n    {l}" for l in tail)))
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"blenderize: {n_ok} ok, {n_err} err, {n_skip} skipped "
                  f"→ {blend_dir}")

    print(f"\ndone → {out_root}  (+ inference_manifest.json)")
    print(f"public materials ({len(PUBLIC_MATERIALS)}): {', '.join(PUBLIC_MATERIALS)}")


if __name__ == "__main__":
    main()
