"""
backbone.py
===========
Top-level model for graph-to-graph translation.

Assembles:
  NodeEncoder  (node_encoder.py)
  EquivariantGraphEncoder  (graph_encoder.py)
  GraphDecoder (graph_decoder.py)

Forward pass
------------
Input : a PyG Batch of unfunctional assembly graphs.
Output: decoder predictions dict (see GraphDecoder.forward docstring).

Usage
-----
  model = GraphTranslationModel(...)
  out   = model(inp_batch)
  # out keys: exist_logits, type_logits, centroid_pred, bbox_pred, edge_logits

Sanity check
------------
Run this file directly to verify the model builds and forward-passes
without errors on a random batch of 2 graphs.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch

from .node_encoder import NodeEncoder
from .graph_encoder import EquivariantGraphEncoder
from .graph_decoder import GraphDecoder


class GraphTranslationModel(nn.Module):
    """End-to-end model for mechanical assembly graph-to-graph translation.

    Parameters
    ----------
    num_material_types   : int — material category vocab (encoder input + decoder output) = 17.
    num_edge_types       : int — edge type vocab (contact/hinge/rail/attached/null = 5).
    type_emb_dim         : int — kinematic type embedding dim.
    pc_dim               : int — PointNet++ output dim.
    geom_dim             : int — geometry MLP output dim.
    hidden_dim           : int — shared hidden dim for encoder/decoder.
    enc_layers           : int — number of EGT encoder layers.
    dec_layers           : int — number of slot decoder layers.
    num_heads            : int — attention heads.
    max_nodes            : int — maximum output node slots.
    dropout              : float — dropout rate.
    """

    def __init__(self,
                 num_material_types:  int   = 17,
                 num_edge_types:      int   = 5,
                 type_emb_dim:        int   = 128,
                 pc_dim:              int   = 64,
                 geom_dim:            int   = 32,
                 hidden_dim:          int   = 256,
                 edge_dim:            int   = 128,
                 enc_layers:          int   = 4,
                 dec_layers:          int   = 4,
                 num_heads:           int   = 8,
                 max_nodes:           int   = 64,            # was 32
                 max_free_slots:      int   = 16,            # was 8
                 dropout:             float = 0.1,
                 geometry_dropout_p:  float = 0.0,
                 use_motion_heads:    bool  = True,
                 motion_format:       str   = "v2",
                 use_graph_cond_slots: bool = True,
                 use_signed_pair_features: bool = False):
        super().__init__()

        self.node_enc = NodeEncoder(
            num_material_types=num_material_types,
            type_emb_dim=type_emb_dim,
            pc_dim=pc_dim,
            geom_dim=geom_dim,
            geometry_dropout_p=geometry_dropout_p,
            dropout=dropout,
        )

        self.graph_enc = EquivariantGraphEncoder(
            in_dim=self.node_enc.out_dim,
            hidden_dim=hidden_dim,
            num_layers=enc_layers,
            num_heads=num_heads,
            num_edge_types=num_edge_types,
            edge_dim=edge_dim,
            dropout=dropout,
            use_signed_pair_features=use_signed_pair_features,
        )

        self.decoder = GraphDecoder(
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
            max_free_slots=max_free_slots,
            num_dec_layers=dec_layers,
            num_heads=num_heads,
            num_material_types=num_material_types,
            num_edge_types=num_edge_types,
            dropout=dropout,
            use_motion_heads=use_motion_heads,
            motion_format=motion_format,
            use_graph_cond_slots=use_graph_cond_slots,
            use_signed_pair_features=use_signed_pair_features,
        )

        self.max_nodes = max_nodes

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        batch : PyG Batch of unfunctional graphs, produced by collate_graph_pairs.
                Expected attributes:
                  .node_type    (N_total,) long
                  .pos          (N_total, 3) float
                  .bbox         (N_total, 6) float
                  .point_clouds list of (n_pts, 3) tensors
                  .pc_node_batch (N_total,) long
                  .edge_index   (2, E_total)
                  .edge_attr    (E_total,) long
                  .batch        (N_total,)  PyG graph index

        Returns
        -------
        dict — see GraphDecoder.forward docstring.
        """
        device = batch.pos.device

        # ── 1. Node encoding ─────────────────────────────────────────────
        pc_batch = batch.pc_node_batch.to(device)
        h = self.node_enc(
            material=batch.material.to(device),
            pos=batch.pos.to(device),
            bbox=batch.bbox.to(device),
            point_clouds=[pc.to(device) for pc in batch.point_clouds],
            pc_node_batch=pc_batch,
        )   # (N_total, node_enc.out_dim)

        # ── 2. Graph encoding ─────────────────────────────────────────────
        enc_batch = batch.batch.to(device)
        h_enc = self.graph_enc(
            h=h,
            pos=batch.pos.to(device),
            edge_index=batch.edge_index.to(device),
            edge_attr=batch.edge_attr.to(device),
            batch=enc_batch,
            bbox=batch.bbox.to(device),     # for AABB-gap edge feature
        )   # (N_total, hidden_dim)

        # ── 3. Decode ─────────────────────────────────────────────────────
        # Use original input centroids (not encoder-drifted) as geometric prior
        batch_size = int(enc_batch.max().item()) + 1
        out = self.decoder(
            h_enc=h_enc,
            enc_batch=enc_batch,
            batch_size=batch_size,
            enc_pos=batch.pos.to(device),
            enc_bbox=batch.bbox.to(device),
        )

        return out

    def predict(self,
                batch: Batch,
                exist_threshold: float = 0.5) -> list:
        """Run inference and return decoded graphs as list of dicts.

        Parameters
        ----------
        batch            : PyG Batch (same format as forward()).
        exist_threshold  : probability threshold for node existence.

        Returns
        -------
        list of dicts, one per graph in batch:
            'node_types'  : (n,) long
            'centroids'   : (n, 3) float
            'bboxes'      : (n, 6) float
            'edge_index'  : (2, e) long   (among predicted real nodes)
            'edge_types'  : (e,)   long
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(batch)

        B = out["exist_logits"].shape[0]
        results = []
        for b in range(B):
            exist_prob = out["exist_logits"][b, :, 0].sigmoid()    # (M,)
            # Anchored slots (= input nodes) always survive regardless of threshold
            anchor = out["anchor_mask"][b] if "anchor_mask" in out else \
                     torch.zeros(exist_prob.shape[0], dtype=torch.bool, device=exist_prob.device)
            real_mask  = (exist_prob > exist_threshold) | anchor
            real_idx   = real_mask.nonzero(as_tuple=True)[0]

            materials  = out["material_logits"][b, real_idx].argmax(-1)
            centroids  = out["centroid_pred"][b, real_idx]
            bboxes     = out["bbox_pred"][b, real_idx]

            n = real_idx.shape[0]
            edge_list_s, edge_list_d, edge_list_t = [], [], []
            if n > 0:
                edge_sub = out["edge_logits"][b][real_idx][:, real_idx]  # (n,n,C)
                edge_types_mat = edge_sub.argmax(-1)                      # (n,n)
                from data.graph_dataset import NULL_EDGE_CLASS
                ATTACHED = 3
                HANDLE_MAT, DOOR_MAT, DRAWER_MAT = 0, 5, 6
                # Soft-parent override: for each handle, pick exactly one parent
                if "parent_scores" in out:
                    local_mats = materials.cpu().tolist()
                    parent_scores_full = out["parent_scores"][b][real_idx][:, real_idx].cpu()
                    noparent_logit = float(out["parent_noparent_logit"].item())
                    for lh, mat in enumerate(local_mats):
                        if mat != HANDLE_MAT: continue
                        cand = [lj for lj, mj in enumerate(local_mats)
                                if lj != lh and mj in (DOOR_MAT, DRAWER_MAT)]
                        if not cand: continue
                        scores = parent_scores_full[lh, cand].tolist() + [noparent_logit]
                        best = int(max(range(len(scores)), key=lambda k: scores[k]))
                        chosen = cand[best] if best < len(cand) else -1
                        for c in cand:
                            if c == chosen:
                                edge_types_mat[lh, c] = ATTACHED
                                edge_types_mat[c, lh] = ATTACHED
                            else:
                                if edge_types_mat[lh, c].item() == ATTACHED:
                                    edge_types_mat[lh, c] = NULL_EDGE_CLASS
                                if edge_types_mat[c, lh].item() == ATTACHED:
                                    edge_types_mat[c, lh] = NULL_EDGE_CLASS
                for i in range(n):
                    for j in range(i + 1, n):
                        et = edge_types_mat[i, j].item()
                        if et != NULL_EDGE_CLASS:
                            edge_list_s += [i, j]
                            edge_list_d += [j, i]
                            edge_list_t += [et, et]

            if edge_list_s:
                ei = torch.tensor([edge_list_s, edge_list_d], dtype=torch.long)
                ea = torch.tensor(edge_list_t, dtype=torch.long)
            else:
                ei = torch.zeros((2, 0), dtype=torch.long)
                ea = torch.zeros(0, dtype=torch.long)

            results.append({
                "materials":  materials.cpu(),
                "centroids":  centroids.cpu(),
                "bboxes":     bboxes.cpu(),
                "edge_index": ei,
                "edge_types": ea,
            })
        return results


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import random
    import numpy as np
    from torch_geometric.data import Batch
    from data.graph_dataset import collate_graph_pairs, NULL_EDGE_CLASS

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running sanity check on {DEVICE}")

    # ── Build model ───────────────────────────────────────────────────────
    model = GraphTranslationModel(
        hidden_dim=128,
        edge_dim=32,
        enc_layers=2,
        dec_layers=2,
        max_nodes=16,
        max_free_slots=4,
        pc_dim=64,
        geom_dim=32,
        type_emb_dim=16,
        num_heads=4,
        dropout=0.0,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Create fake batch of 2 graphs ─────────────────────────────────────
    def make_fake_graph(n_nodes=5, n_edges=6, n_pts=32):
        """Build a minimal fake Data object matching the dataset format."""
        from torch_geometric.data import Data
        import random as rnd

        material = torch.randint(0, 17, (n_nodes,))          # material category
        pos      = torch.randn(n_nodes, 3)
        bbox     = torch.randn(n_nodes, 6)
        x        = torch.cat([pos, bbox], dim=1)

        if n_edges > 0:
            src = torch.randint(0, n_nodes, (n_edges,))
            dst = torch.randint(0, n_nodes, (n_edges,))
            ei  = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
            ea  = torch.randint(0, 4, (ei.shape[1],))
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros(0, dtype=torch.long)

        d = Data(x=x, edge_index=ei, edge_attr=ea,
                 pos=pos, bbox=bbox, material=material, num_nodes=n_nodes)
        d.point_clouds = [torch.randn(n_pts, 3) for _ in range(n_nodes)]
        d.node_names   = [f"node_{i}" for i in range(n_nodes)]
        d.model_id = "fake"
        return d

    graphs = [make_fake_graph(5, 4), make_fake_graph(7, 8)]

    # Manual "collate"
    from torch_geometric.data import Batch as PyGBatch
    pc_lists   = [g.point_clouds for g in graphs]
    name_lists = [g.node_names   for g in graphs]
    for g in graphs:
        del g.point_clouds
        del g.node_names

    batch = PyGBatch.from_data_list(graphs)
    batch.point_clouds  = [pc for pcs in pc_lists   for pc in pcs]
    batch.node_names    = [n  for ns  in name_lists for n  in ns]
    batch.pc_node_batch = batch.batch
    for g, pcs, names in zip(graphs, pc_lists, name_lists):
        g.point_clouds = pcs
        g.node_names   = names
    batch = batch.to(DEVICE)

    # ── Forward pass ──────────────────────────────────────────────────────
    model.train()
    out = model(batch)
    print("Forward pass outputs:")
    for k, v in out.items():
        print(f"  {k:20s} : {tuple(v.shape)}")

    # ── Dummy loss ────────────────────────────────────────────────────────
    from models.losses import GraphTranslationLoss

    criterion = GraphTranslationLoss().to(DEVICE)

    targets = []
    for b in range(2):
        g = graphs[b]
        targets.append({
            "material":   g.material.to(DEVICE),
            "centroid":   g.pos.to(DEVICE),
            "bbox":       g.bbox.to(DEVICE),
            "edge_index": g.edge_index.to(DEVICE),
            "edge_attr":  g.edge_attr.to(DEVICE),
        })

    losses = criterion(out, targets)
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k:20s} : {v.item():.4f}")

    losses["loss_total"].backward()
    print("\nBackward pass: OK")
    print("Sanity check PASSED.")
