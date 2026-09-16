
"""
train.py
========
Training loop for the graph-to-graph translation model.

Usage
-----
  python train.py [--config config.yaml]

All hyperparameters can be tuned via the CONFIG dict at the top of this file
(or overridden by a YAML config if provided).

Training strategy
-----------------
- 80 / 10 / 10 split on the FurFun dataset + augmented samples.
- Adam optimiser with cosine-annealing LR schedule.
- Gradient clipping to prevent instability from the Hungarian matching.
- Checkpoints saved every N epochs (best val loss retained).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.multiprocessing as _mp
# Use 'spawn' so DataLoader workers don't inherit CUDA state from fork() —
# fork+CUDA causes intermittent segfaults / CUDA asserts in autograd backward.
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch as PyGBatch

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.graph_dataset import (
    FurFunGraphDataset,
    collate_graph_pairs,
    NULL_EDGE_CLASS,
    split_by_model_id,
)
from models.backbone import GraphTranslationModel
from models.losses import GraphTranslationLoss


# ── Default config ────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "func_root":        str(ROOT / "datasets" / "furfun" / "graphs_functional"),
    "unfunc_root":      str(ROOT / "datasets" / "furfun" / "graphs_unfunctional"),
    "geom_root":        str(ROOT / "datasets" / "furfun" / "part_geometries"),
    "n_pc_points":      512,
    "node_mask_rate":   0.2,       # used as fallback; actual rate varies per strategy
    "edge_mask_rate":   0.0,       # no edge dropping — type collapse only
    "augment_ratio":    5.0,       # 5× structured node-drop augmentations
    "seed":             42,

    # Model — graph-primary defaults: material dominates, geometry small
    "num_material_types":  17,  # encoder input + decoder output: 16 material categories + unknown
    "num_edge_types":      5,
    "type_emb_dim":     128,    # was 32 — graph channel now ~57% of node feat
    "pc_dim":           64,     # was 128
    "geom_dim":         32,     # was 64
    "hidden_dim":       256,
    "edge_dim":         128,    # was 64 — give edge channel more capacity
    "enc_layers":       4,
    "dec_layers":       4,
    "num_heads":        8,
    "max_nodes":        64,         # bumped from 32 to cover all 122 fur_* graphs
    "max_free_slots":   16,         # bumped from 8 — paired with multi-category drops
    "dropout":          0.1,
    # Geometry dropout: at training time, with this prob the entire geometry
    # stream (PointNet++ + geom MLP) is zeroed for the forward pass, forcing
    # the encoder to develop a graph-only fallback path. 0 = legacy.
    "geometry_dropout_p": 0.3,
    # Graph-conditional slot queries: free slots get a residual derived from
    # the graph mean-pool, so they can decide whether to fire based on what
    # the input graph as a whole looks like.
    "use_graph_cond_slots": True,
    # Mirror/flip augmentation probability per (input, target) pair.
    "mirror_p":         0.5,

    # Loss
    "w_node":           1.0,
    "w_edge":           1.0,
    "w_parent":         0.5,     # softmax-parent loss weight
    "w_hinge_target":   0.5,     # softmax hinge-target loss weight
    "w_motion":         0.3,     # motion attribute (pos/dir/axis) loss weight
    "null_weight":      1.0,     # was 0.2 — stop down-weighting null
    # Per-class edge weights [contact, hinge, rail, attached, null]. None →
    # uses null_weight only. Set explicitly to up-weight under-represented
    # functional classes (inverse-frequency style).
    "edge_class_weights": None,
    "cost_class":       1.0,
    "cost_pos":         2.0,

    # Training
    "epochs":           450,
    "batch_size":       4,
    "lr":               3e-4,
    "weight_decay":     1e-4,
    "grad_clip":        1.0,
    "val_split":        0.1,
    "test_split":       0.1,
    "save_every":       10,
    "checkpoint_dir":   str(ROOT / "checkpoints"),
    "device":           "cuda" if torch.cuda.is_available() else "cpu",

    # Ablation toggle — False ablates motion heads entirely (decoder doesn't
    # build them and they receive no gradient).
    "use_motion_heads": True,
    # "v2" = legacy 5-head face-pair encoding.
    # "v3" = 3-head per-dynamic-part encoding (Kabsch-derived, signed).
    "motion_format":    "v3_pose_inv",   # matches the shipped configs/datasets

    # Path to a frozen JSON split (auto-generated on first run if missing).
    "split_file":       None,

    # Optional checkpoint to warm-start from (pretrain → finetune).
    "pretrain_ckpt":    None,

    # Dump N (input, target) training sample visualizations at epoch 0.
    "num_vis_samples":  6,

    # ── Progress monitoring ──────────────────────────────────────────────
    # WandB logging (set wandb=true in config to enable).
    "wandb":            False,
    "wandb_project":    "furfun_motion",
    "wandb_entity":     None,        # optional team/user override
    # Periodic train+val sample dump: every N epochs, run inference on a
    # fixed handful of train + val mids and save (input, target, pred)
    # JSONs into <checkpoint_dir>/progress_samples/epXXXX/. Set 0 to
    # disable.
    "sample_log_every_n_epochs": 25,
    "num_progress_train_samples": 4,
    "num_progress_val_samples":   4,
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_target_list(target_batch: PyGBatch, device: str,
                      inp_batch: PyGBatch = None) -> List[Dict]:
    """Unpack a batched target graph into a list of per-graph dicts."""
    B = int(target_batch.batch.max().item()) + 1
    targets = []
    for b in range(B):
        mask = target_batch.batch == b
        ei   = target_batch.edge_index
        ea   = target_batch.edge_attr

        # Get node count up to graph b to remap edge indices
        offset = mask.nonzero(as_tuple=True)[0][0].item()
        n_b    = mask.sum().item()

        # Filter edges belonging to graph b and re-index to [0, n_b)
        if ei.shape[1] > 0:
            # Edges belonging to graph b: both endpoints in [offset, offset+n_b)
            lo, hi = offset, offset + n_b
            e_mask = (ei[0] >= lo) & (ei[0] < hi) & (ei[1] >= lo) & (ei[1] < hi)
            ei_b   = ei[:, e_mask] - offset
            ea_b   = ea[e_mask]
        else:
            ei_b = torch.zeros((2, 0), dtype=torch.long)
            ea_b = torch.zeros(0, dtype=torch.long)

        tgt_dict = {
            "material":   target_batch.material[mask].to(device),
            "centroid":   target_batch.pos[mask].to(device),
            "bbox":       target_batch.bbox[mask].to(device),
            "edge_index": ei_b.to(device),
            "edge_attr":  ea_b.to(device),
        }
        # Per-node v3 label (door nodes → handle_border, others → -1)
        if hasattr(target_batch, "handle_border") and target_batch.handle_border is not None:
            tgt_dict["handle_border"] = target_batch.handle_border[mask].to(device)
        # Pass through motion labels if available (v2 + v3 + v3_pose_inv)
        for attr in ("hinge_face_src", "hinge_face_dst", "hinge_axis",
                     "hinge_dir", "rail_axis",
                     "hinge_border", "hinge_axis_signed", "rail_axis_signed",
                     "hinge_border_4", "hinge_axis_sign",
                     "dyn_is_src"):
            if hasattr(target_batch, attr):
                vals = getattr(target_batch, attr)
                if vals is not None and vals.numel() > 0:
                    tgt_dict[attr] = vals.cpu()[e_mask.cpu()].to(device) if ei.shape[1] > 0 else torch.zeros(0, dtype=torch.long, device=device)
        # Anchored-slot pinning map for the Hungarian matcher (slot k of
        # this sample -> its own GT node index; -1 = unpinned).
        if inp_batch is not None and hasattr(inp_batch, "anchor_gt_idx"):
            im = inp_batch.batch == b
            tgt_dict["pinned_slot_gt"] = inp_batch.anchor_gt_idx[im].to(device)
        targets.append(tgt_dict)
    return targets


def move_batch(batch: PyGBatch, device: str) -> PyGBatch:
    """Move a collated batch to device (tensor attributes only)."""
    for attr in ["x", "edge_index", "edge_attr", "pos", "bbox",
                 "material", "batch", "pc_node_batch",
                 "handle_border", "anchor_gt_idx"]:
        if hasattr(batch, attr) and isinstance(getattr(batch, attr), torch.Tensor):
            setattr(batch, attr, getattr(batch, attr).to(device))
    batch.point_clouds = [pc.to(device) for pc in batch.point_clouds]
    return batch


# ── Training and validation steps ────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimiser, cfg) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    n_batches = 0

    for inp_batch, tgt_batch in loader:
        inp_batch = move_batch(inp_batch, cfg["device"])
        tgt_batch = move_batch(tgt_batch, cfg["device"])
        targets   = build_target_list(tgt_batch, cfg["device"], inp_batch)

        optimiser.zero_grad()
        out    = model(inp_batch)
        losses = criterion(out, targets)

        losses["loss_total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimiser.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, criterion, cfg) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    n_batches = 0

    for inp_batch, tgt_batch in loader:
        inp_batch = move_batch(inp_batch, cfg["device"])
        tgt_batch = move_batch(tgt_batch, cfg["device"])
        targets   = build_target_list(tgt_batch, cfg["device"], inp_batch)

        out    = model(inp_batch)
        losses = criterion(out, targets)

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def _resolve_split(dataset, cfg):
    """Return (train_ds, val_ds, test_ds, split_info).

    When `cfg['split_file']` is set:
      * If the file exists → load the frozen {train,val,test}_mids and build Subsets.
      * Else → generate a fresh split via split_by_model_id, then dump to the file.
    When unset, just use split_by_model_id with the configured fractions.

    Models referenced by the frozen split that are absent from the current
    dataset (e.g. synth pretrain using a real-data split) are silently dropped.
    """
    import json as _json
    from torch.utils.data import Subset

    split_file = cfg.get("split_file")
    if split_file and os.path.isfile(split_file):
        with open(split_file) as f:
            info = _json.load(f)
        groups = {}
        for i in range(len(dataset)):
            groups.setdefault(dataset.model_id_of_index(i), []).append(i)

        def _flatten(mids):
            out = []
            for m in mids:
                out.extend(groups.get(m, []))
            return out

        train_ds = Subset(dataset, _flatten(info["train_mids"]))
        val_ds   = Subset(dataset, _flatten(info["val_mids"]))
        test_ds  = Subset(dataset, _flatten(info["test_mids"]))
        return train_ds, val_ds, test_ds, info

    train_ds, val_ds, test_ds, info = split_by_model_id(
        dataset,
        test_frac=cfg["test_split"],
        val_frac=cfg["val_split"],
        seed=cfg["seed"],
    )
    if split_file:
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        with open(split_file, "w") as f:
            _json.dump(info, f, indent=2)
        print(f"Wrote frozen split → {split_file}")
    return train_ds, val_ds, test_ds, info


def _save_training_samples(dataset, info, cfg):
    """Dump a few (input, target) graph JSONs from the training split so we can
    eyeball what the model is actually learning on."""
    import json as _json

    n = int(cfg.get("num_vis_samples", 0))
    if n <= 0:
        return
    out_dir = Path(cfg["checkpoint_dir"]) / "train_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    for i in range(len(dataset)):
        groups.setdefault(dataset.model_id_of_index(i), []).append(i)

    # Pick one sample per training model id, up to N
    train_mids = [m for m in info["train_mids"] if m in groups][:n]
    for mid in train_mids:
        idx = groups[mid][0]
        inp, tgt = dataset[idx]

        def _dump(prefix, g):
            d = {
                "model_id":    mid,
                "material":    g.material.tolist(),
                "pos":         g.pos.tolist(),
                "bbox":        g.bbox.tolist(),
                "edge_index":  g.edge_index.tolist(),
                "edge_attr":   g.edge_attr.tolist(),
                "node_names":  getattr(g, "node_names", None),
            }
            with open(out_dir / f"{mid}_{prefix}.json", "w") as f:
                _json.dump(d, f, indent=2)

        _dump("input", inp)
        _dump("target", tgt)
    print(f"Saved {len(train_mids)} training sample pairs → {out_dir}")


def _to_serializable_graph(g, mid: str, source: str, idx2material):
    """Convert a PyG Data into a graph-JSON dict (same format as the
    dataset's graph JSONs)."""
    EDGE_NAMES = {0:"contact",1:"hinge",2:"rail",3:"attached"}
    nodes = {}
    names = getattr(g, "node_names", None) or [f"slot{i:02d}" for i in range(g.num_nodes)]
    mats  = g.material.tolist()
    pos   = g.pos.tolist()
    bbox  = g.bbox.tolist()
    for i, nm in enumerate(names):
        half = [(bbox[i][3]-bbox[i][0])/2, (bbox[i][4]-bbox[i][1])/2, (bbox[i][5]-bbox[i][2])/2]
        mat_name = idx2material.get(int(mats[i]), "unknown")
        nodes[nm] = {
            "id":       nm, "material": mat_name,
            "type":     "dynamic" if mat_name in ("door","drawer","handle") else "static",
            "obb":      {"center": pos[i], "half": half, "quat": [1,0,0,0]},
        }
    edges = []
    if g.edge_index.numel() > 0:
        seen = set()
        ei = g.edge_index.tolist(); ea = g.edge_attr.tolist()
        for s, d, t in zip(ei[0], ei[1], ea):
            a, b = sorted([s, d])
            if (a, b, t) in seen: continue
            seen.add((a, b, t))
            edges.append({"src": names[s], "dst": names[d],
                          "kind": EDGE_NAMES.get(t, "contact")})
    return {"model_id": mid, "source": source, "nodes": nodes, "edges": edges}


@torch.no_grad()
def _save_progress_samples(model, dataset, info, cfg, epoch: int, wandb_run=None):
    """Run the model on a fixed handful of train + val mids and dump
    (input, target, pred) JSONs — plus optionally an HTML 4-col compare
    and per-mid wandb HTML artefact — so progress is visible across epochs.
    """
    import json as _json
    n_train = int(cfg.get("num_progress_train_samples", 0))
    n_val   = int(cfg.get("num_progress_val_samples", 0))
    if n_train <= 0 and n_val <= 0:
        return
    from data.graph_dataset import IDX2MATERIAL, collate_graph_pairs
    sys.path.insert(0, str(ROOT))
    from infer_common import pred_to_json

    device = cfg["device"]
    out_root = Path(cfg["checkpoint_dir"]) / "progress_samples" / f"ep{epoch:04d}"
    out_root.mkdir(parents=True, exist_ok=True)

    groups = {}
    for i in range(len(dataset)):
        groups.setdefault(dataset.model_id_of_index(i), []).append(i)

    # Use a fixed seed for the dataset RNG so the SAME corruption is
    # produced each epoch — the prediction evolution is the only thing
    # changing across snapshots.
    train_mids = [m for m in info["train_mids"] if m in groups][:n_train]
    val_mids   = [m for m in info["val_mids"]   if m in groups][:n_val]

    was_training = model.training
    model.eval()

    # The per-mid reseed below must not disturb the live training RNGs:
    # snapshot and restore them so a progress dump never changes the
    # training trajectory. (And seed from crc32, not hash() — builtin
    # hash() is salted per process, so it isn't stable across runs.)
    import zlib
    rng_py_state = dataset._rng_py.getstate()
    rng_np_saved = dataset._rng_np

    snapshots = []
    for split, mids in [("train", train_mids), ("val", val_mids)]:
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for mid in mids:
            idx = groups[mid][0]
            # Reset dataset RNG for deterministic corruption per (mid, epoch)
            mid_seed = 7 + zlib.crc32(mid.encode()) % 1000
            dataset._rng_py.seed(mid_seed)
            dataset._rng_np = np.random.default_rng(mid_seed)
            inp, tgt = dataset[idx]
            # Save input/target JSONs
            (split_dir / f"{mid}_input.json").write_text(_json.dumps(
                _to_serializable_graph(inp, mid, "corrupted_input", IDX2MATERIAL),
                indent=2))
            (split_dir / f"{mid}_target.json").write_text(_json.dumps(
                _to_serializable_graph(tgt, mid, "target", IDX2MATERIAL),
                indent=2))
            # Run inference
            inp_batch, _ = collate_graph_pairs([(inp, tgt)])
            inp_batch = move_batch(inp_batch, device)
            out = model(inp_batch)
            pred = pred_to_json(out, 0, mid, 0.5)
            (split_dir / f"{mid}_pred.json").write_text(_json.dumps(pred, indent=2))
            snapshots.append((split, mid, str(split_dir / f"{mid}_pred.json")))

    dataset._rng_py.setstate(rng_py_state)
    dataset._rng_np = rng_np_saved
    if was_training: model.train()

    print(f"  [progress] saved {len(snapshots)} (input,target,pred) triples "
          f"→ {out_root}")


def main(cfg: Dict):
    set_seed(cfg["seed"])
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    scale_variants = None
    if cfg.get("scale_variants_file"):
        import json as _json
        with open(cfg["scale_variants_file"]) as f:
            scale_variants = [v for v in _json.load(f) if v["kind"] != "baseline"]

    dataset = FurFunGraphDataset(
        func_root=cfg["func_root"],
        unfunc_root=cfg["unfunc_root"],
        geom_root=cfg["geom_root"],
        n_pc_points=cfg["n_pc_points"],
        node_mask_rate=cfg["node_mask_rate"],
        edge_mask_rate=cfg["edge_mask_rate"],
        augment_ratio=cfg["augment_ratio"],
        identity_pair_rate=float(cfg.get("identity_pair_rate", 0.0)),
        cache_in_ram=bool(cfg.get("cache_in_ram", False)),
        seed=cfg["seed"],
        scaled_geom_root=cfg.get("scaled_geom_root"),
        scale_variants=scale_variants,
        p_scale=float(cfg.get("p_scale", 0.0)),
        mirror_p=float(cfg.get("mirror_p", 0.0)),
        mirror_axes=tuple(cfg.get("mirror_axes", [0, 1])),
        require_connected=bool(cfg.get("require_connected", True)),
    )
    print(f"Dataset size: {len(dataset)} pairs")

    train_ds, val_ds, test_ds, split_info = _resolve_split(dataset, cfg)
    print(f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"(by model_id: {len(split_info['train_mids'])}/"
          f"{len(split_info['val_mids'])}/{len(split_info['test_mids'])})")

    _save_training_samples(dataset, split_info, cfg)

    # DataLoader: with num_workers>0 + persistent_workers we keep parallel
    # CPU prep alive across epochs so the GPU isn't starved between batches.
    # pin_memory speeds the CPU→GPU copy.
    nw = int(cfg.get("num_workers", 0))
    persistent = nw > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_graph_pairs, num_workers=nw,
        pin_memory=(nw > 0), persistent_workers=persistent)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_graph_pairs, num_workers=nw,
        pin_memory=(nw > 0), persistent_workers=persistent)

    # ── Model ─────────────────────────────────────────────────────────────
    model = GraphTranslationModel(
        num_material_types=cfg["num_material_types"],
        num_edge_types=cfg["num_edge_types"],
        type_emb_dim=cfg["type_emb_dim"],
        pc_dim=cfg["pc_dim"],
        geom_dim=cfg["geom_dim"],
        hidden_dim=cfg["hidden_dim"],
        edge_dim=cfg["edge_dim"],
        enc_layers=cfg["enc_layers"],
        dec_layers=cfg["dec_layers"],
        num_heads=cfg["num_heads"],
        max_nodes=cfg["max_nodes"],
        max_free_slots=cfg["max_free_slots"],
        dropout=cfg["dropout"],
        geometry_dropout_p=float(cfg.get("geometry_dropout_p", 0.0)),
        use_motion_heads=cfg["use_motion_heads"],
        motion_format=cfg.get("motion_format", "v2"),
        use_graph_cond_slots=bool(cfg.get("use_graph_cond_slots", True)),
        use_signed_pair_features=bool(cfg.get("use_signed_pair_features", False)),
    ).to(cfg["device"])
    # Release inference constraints must not leak into train-time validation:
    for _m in model.modules():
        _m.structural_masks = False

    # Optional warm-start from a pretrain checkpoint
    if cfg.get("pretrain_ckpt"):
        ckpt = torch.load(cfg["pretrain_ckpt"], map_location=cfg["device"])
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Warm-started from {cfg['pretrain_ckpt']} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}  use_motion_heads={cfg['use_motion_heads']}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.get("lr_T_max", cfg["epochs"]))

    # When motion heads are ablated, zero-weight the motion loss so it never
    # appears in the total (the decoder output also drops motion logits).
    w_motion = cfg["w_motion"] if cfg["use_motion_heads"] else 0.0
    criterion = GraphTranslationLoss(
        w_node=cfg["w_node"],
        w_edge=cfg["w_edge"],
        w_parent=cfg["w_parent"],
        w_hinge_target=cfg["w_hinge_target"],
        w_motion=w_motion,
        w_anchor=cfg.get("w_anchor", 0.0),
        w_anchor_bbox=cfg.get("w_anchor_bbox", 0.0),
        w_free_exist=cfg.get("w_free_exist", 0.0),
        free_exist_pw_floor=cfg.get("free_exist_pw_floor", 5.0),
        w_count=cfg.get("w_count", 0.0),
        w_type=cfg.get("w_type", 1.0),
        w_centroid=cfg.get("w_centroid", 2.0),
        w_bbox=cfg.get("w_bbox", 1.0),
        cost_class=cfg["cost_class"],
        cost_pos=cfg["cost_pos"],
        null_weight=cfg["null_weight"],
        edge_class_weights=cfg.get("edge_class_weights"),
        motion_format=cfg.get("motion_format", "v2"),
        w_handle=cfg.get("w_handle", 0.3),
        w_handle_opp=cfg.get("w_handle_opp", 0.15),
    ).to(cfg["device"])

    best_val = float("inf")

    # Optional resume from an interrupted run's periodic checkpoint
    start_epoch = 1
    if cfg.get("resume_ckpt"):
        rck = torch.load(cfg["resume_ckpt"], map_location=cfg["device"])
        model.load_state_dict(rck["model_state"])
        optimiser.load_state_dict(rck["optim_state"])
        start_epoch = int(rck["epoch"]) + 1
        # Prefer the checkpoint's own record; the resumed epoch's val_loss
        # is the fallback for older checkpoints. Leaving best_val at inf
        # would let the first post-resume epoch clobber model_best.pt.
        best_val = float(cfg.get("resume_best_val",
                                 rck.get("best_val",
                                         rck.get("val_loss", best_val))))
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Resumed from {cfg['resume_ckpt']} (epoch {rck['epoch']}, "
              f"continuing at {start_epoch}, best_val={best_val:.3f})")

    # Curriculum: notify the dataset of training progress each epoch so the
    # node-drop rates anneal from aggressive early to modest late.
    from data.graph_dataset import (set_curriculum_progress, set_drop_bands,
                                       set_max_drop_nodes, set_edge_preserving_boost)
    if cfg.get("drop_band_early") is not None or cfg.get("drop_band_late") is not None:
        set_drop_bands(cfg.get("drop_band_early", (0.30, 0.50)),
                        cfg.get("drop_band_late",  (0.10, 0.20)))
        print(f"  drop bands overridden: early={cfg.get('drop_band_early')} late={cfg.get('drop_band_late')}")
    if cfg.get("max_drop_nodes") is not None:
        set_max_drop_nodes(int(cfg["max_drop_nodes"]))
        print(f"  max_drop_nodes set to {cfg['max_drop_nodes']}")
    if cfg.get("edge_preserving_boost") is not None:
        set_edge_preserving_boost(float(cfg["edge_preserving_boost"]))
        print(f"  edge_preserving_boost set to {cfg['edge_preserving_boost']}")

    # ── WandB init (optional) ─────────────────────────────────────────────
    wandb_run = None
    if cfg.get("wandb"):
        try:
            import wandb
            run_name = Path(cfg["checkpoint_dir"]).name
            wandb_run = wandb.init(
                project=cfg.get("wandb_project", "furfun_motion"),
                entity=cfg.get("wandb_entity"),
                name=run_name,
                config=cfg,
                dir=cfg["checkpoint_dir"],
                reinit=True,
            )
            wandb.watch(model, log=None, log_freq=200)
            print(f"WandB run: {wandb_run.url if hasattr(wandb_run, 'url') else 'local'}")
        except Exception as e:
            print(f"WandB init failed (continuing without): {e}")
            wandb_run = None

    sample_every = int(cfg.get("sample_log_every_n_epochs", 0))

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        set_curriculum_progress((epoch - 1) / max(cfg["epochs"] - 1, 1))
        train_losses = train_epoch(model, train_loader, criterion, optimiser, cfg)
        val_losses   = val_epoch(model, val_loader, criterion, cfg)
        scheduler.step()

        lr_now = optimiser.param_groups[0]["lr"]
        base = (f"Epoch {epoch:03d}/{cfg['epochs']:03d}  "
                f"lr={lr_now:.2e}  "
                f"train={train_losses['loss_total']:.3f}  "
                f"val={val_losses['loss_total']:.3f}  "
                f"[mat={val_losses.get('loss_material',0):.2f} "
                f"pos={val_losses.get('loss_centroid',0):.2f} "
                f"edge={val_losses.get('loss_edge',0):.2f}")
        if cfg.get("motion_format", "v2") == "v3":
            motion_str = (f" hb={val_losses.get('loss_hinge_border',0):.2f} "
                          f"has={val_losses.get('loss_hinge_axis_signed',0):.2f} "
                          f"ras={val_losses.get('loss_rail_axis_signed',0):.2f} "
                          f"hd={val_losses.get('loss_handle',0):.2f} "
                          f"op={val_losses.get('loss_opp',0):.2f}]")
        else:
            motion_str = (f" fsrc={val_losses.get('loss_hinge_face_src',0):.2f} "
                          f"fdst={val_losses.get('loss_hinge_face_dst',0):.2f} "
                          f"hax={val_losses.get('loss_hinge_axis',0):.2f} "
                          f"hdir={val_losses.get('loss_hinge_dir',0):.2f} "
                          f"rax={val_losses.get('loss_rail_axis',0):.2f}]")
        print(base + motion_str)

        is_best = val_losses["loss_total"] < best_val
        if is_best:
            best_val = val_losses["loss_total"]

        # ── WandB scalar logging (loss curves) ──────────────────────────
        if wandb_run is not None:
            try:
                log_dict = {
                    "lr":            lr_now,
                    "epoch":         epoch,
                    "best_val":      best_val,
                }
                for k, v in train_losses.items():
                    log_dict[f"train/{k}"] = float(v)
                for k, v in val_losses.items():
                    log_dict[f"val/{k}"]   = float(v)
                wandb_run.log(log_dict, step=epoch)
            except Exception as e:
                print(f"  [wandb] log failed: {e}")

        # ── Periodic train + val sample dump (with HTML viz) ───────────
        if sample_every > 0 and (epoch % sample_every == 0 or epoch == 1):
            try:
                _save_progress_samples(model, dataset, split_info, cfg,
                                        epoch, wandb_run=wandb_run)
            except Exception as e:
                print(f"  [progress] dump failed: {e}")

        if epoch % cfg["save_every"] == 0 or is_best:
            ckpt = {
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "val_loss":    val_losses["loss_total"],
                "best_val":    best_val,
                "cfg":         cfg,
            }
            # An epoch can be both a new best AND a periodic snapshot —
            # write both files rather than collapsing to "best" only.
            paths = []
            if is_best:
                paths.append(Path(cfg["checkpoint_dir"]) / "model_best.pt")
            if epoch % cfg["save_every"] == 0:
                paths.append(Path(cfg["checkpoint_dir"]) / f"model_ep{epoch:04d}.pt")
            for path in paths:
                torch.save(ckpt, path)
            if is_best:
                print(f"  → New best checkpoint saved: "
                      f"{Path(cfg['checkpoint_dir']) / 'model_best.pt'}")

    if wandb_run is not None:
        try: wandb_run.finish()
        except Exception: pass
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (optional).")
    args = parser.parse_args()

    cfg = CONFIG.copy()

    if args.config:
        import yaml
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        cfg.update(overrides)

    # Resolve dataset/checkpoint paths relative to the repository root so
    # configs stay portable.
    for k in ("func_root", "unfunc_root", "geom_root", "split_file",
              "checkpoint_dir", "resume_ckpt", "pretrain_ckpt",
              "scale_variants_file"):
        v = cfg.get(k)
        if v and not os.path.isabs(str(v)):
            cfg[k] = str(ROOT / v)

    # Optional augmentation-recipe overrides (e.g. the broadened
    # reconstruction-coverage variant): update the sampling weight table.
    overrides = cfg.get("category_drop_overrides")
    if overrides:
        from data import graph_dataset as _gd
        for k, v in overrides.items():
            _gd._DROP_STRATEGY_WEIGHTS[k] = float(v)
        print(f"[aug] category_drop_overrides applied: {overrides}")

    main(cfg)
