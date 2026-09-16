"""
graph_encoder.py
================
Full Graph Transformer encoder for the assembly graph.

Design
------
- Fully-connected within-graph attention: every node pair attends to every other.
  Missing edges (not in input graph) are initialised as NULL type.
- Edge features are first-class: initialised from edge-type embedding + pairwise
  distance, then updated each layer as a function of endpoint node features
  (Dwivedi & Bresson 2020 Graph Transformer).
- Edge features contribute to attention scores (additive bias) and to value
  aggregation (additive term on V[src]).
- No equivariant coordinate update — furniture parts are static.

Each layer
----------
  1. Node multi-head attention:
       score_ij  = Q_i · K_j / √d  +  edge_to_attn(e_ij)
       value_ij  = V[src]  +  edge_to_val(e_ij)
       h_i      += scatter_softmax_weighted(value_ij)
  2. Edge update (pre-norm residual):
       e_ij      = LayerNorm(e_ij + MLP(h_i || h_j || e_ij))
  3. Node FFN (pre-norm residual).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import softmax as pyg_softmax

from .node_encoder import mlp


# ── Graph Transformer layer ──────────────────────────────────────────────────

class GTLayer(nn.Module):
    """Single Graph Transformer layer with learnable edge feature updates.

    Parameters
    ----------
    hidden_dim : int  — node feature dimension.
    num_heads  : int  — number of attention heads.
    edge_dim   : int  — edge feature dimension.
    dropout    : float.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads:  int   = 8,
                 edge_dim:   int   = 64,
                 dropout:    float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Node attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge → attention score bias (one scalar per head)
        self.edge_to_attn = nn.Linear(edge_dim, num_heads, bias=False)
        # Edge → value contribution (same dim as node value)
        self.edge_to_val  = nn.Linear(edge_dim, hidden_dim, bias=False)

        # Edge feature update: MLP(h_src || h_dst || e_ij) → e_ij
        # use_bn=False: edge count varies per forward pass
        self.edge_update = mlp(
            [hidden_dim * 2 + edge_dim, edge_dim * 2, edge_dim],
            act=nn.GELU, dropout=dropout, use_bn=False)
        self.edge_norm = nn.LayerNorm(edge_dim)

        # Node FFN
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self,
                h:          Tensor,
                edge_index: Tensor,
                edge_feat:  Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        h          : (N, D)        node features
        edge_index : (2, E)        [src, dst] — fully connected within each graph
        edge_feat  : (E, edge_dim) edge features

        Returns
        -------
        h_new        : (N, D)
        edge_feat_new: (E, edge_dim)
        """
        N = h.shape[0]
        src, dst = edge_index

        h_norm = self.norm1(h)

        # ── Multi-head attention ──────────────────────────────────────────
        Q = self.q_proj(h_norm).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(h_norm).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(h_norm).view(N, self.num_heads, self.head_dim)

        # Dot-product score + edge bias
        qk         = (Q[dst] * K[src]).sum(-1) * self.scale  # (E, H)
        attn_score = qk + self.edge_to_attn(edge_feat)        # (E, H)
        attn_w     = pyg_softmax(attn_score, dst, num_nodes=N) # (E, H)

        # Value: V[src] + edge contribution, weighted by attention
        V_src    = V[src]                                              # (E, H, hd)
        edge_val = self.edge_to_val(edge_feat).view(-1, self.num_heads,
                                                    self.head_dim)    # (E, H, hd)
        weighted = attn_w.unsqueeze(-1) * (V_src + edge_val)          # (E, H, hd)

        agg = torch.zeros(N, self.num_heads, self.head_dim,
                          device=h.device, dtype=h.dtype)
        agg.scatter_add_(0,
                         dst.view(-1, 1, 1).expand_as(weighted),
                         weighted)
        agg  = agg.view(N, -1)
        h    = h + self.drop(self.o_proj(agg))

        # ── Edge feature update ───────────────────────────────────────────
        edge_in   = torch.cat([h_norm[src], h_norm[dst], edge_feat], dim=-1)
        edge_feat = self.edge_norm(edge_feat + self.edge_update(edge_in))

        # ── Node FFN ──────────────────────────────────────────────────────
        h = h + self.drop(self.ff(self.norm2(h)))

        return h, edge_feat


# ── Full encoder ─────────────────────────────────────────────────────────────

class EquivariantGraphEncoder(nn.Module):
    """Stack of GTLayers forming the graph encoder.

    Builds fully-connected within-graph edges before the first layer
    (missing pairs get NULL edge type).  Edge features are initialised
    from the edge-type embedding and pairwise Euclidean distance, then
    refined by each GTLayer.

    Parameters
    ----------
    in_dim        : int  — NodeEncoder output dimension.
    hidden_dim    : int  — node hidden dimension (default 256).
    num_layers    : int  — number of GTLayer stacks (default 4).
    num_heads     : int  — attention heads (default 8).
    num_edge_types: int  — edge type vocabulary size including NULL (default 5).
    edge_dim      : int  — edge feature dimension (default 64).
    dropout       : float.
    """

    def __init__(self,
                 in_dim:         int,
                 hidden_dim:     int   = 256,
                 num_layers:     int   = 4,
                 num_heads:      int   = 8,
                 num_edge_types: int   = 5,
                 edge_dim:       int   = 64,
                 dropout:        float = 0.1,
                 use_signed_pair_features: bool = False):
        super().__init__()
        self.null_edge_type = num_edge_types - 1   # NULL_EDGE_CLASS = 4
        self.edge_dim       = edge_dim
        self.hidden_dim     = hidden_dim
        # Ablation flag: when False, signed unit direction (3) and per-axis
        # surface gap (3) are zeroed before entering the dist_mlp. Capacity
        # stays the same — dist_mlp input dim is fixed at 8 — but the
        # orientation-aware channels carry no signal.
        self.use_signed_pair_features = use_signed_pair_features

        # Node input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Edge feature initialisation: type embedding + geometric MLP → edge_dim.
        # The geometric MLP encodes BOTH centroid distance AND OBB-surface gap
        # (= 0 if the OBBs overlap, else min surface-to-surface distance).
        # Centroid distance alone is misleading for thin/elongated parts:
        # a side panel's centroid can be tens of cm from a touching panel's
        # centroid because of part aspect ratios. The AABB-gap term gives
        # the encoder orientation-aware proximity. Both fed jointly so the
        # model can use whichever signal is better per-pair.
        # Geometric input dim:
        #   1 — centroid distance ‖r‖
        #   1 — AABB surface gap ‖per_axis_gap‖
        #   3 — signed unit direction r̂ (for "A is below/right/etc. of B")
        #   3 — per-axis surface gap (positive components only; tells which
        #       axes the OBBs separate along)
        # = 8 dims into the geometric MLP.
        half = edge_dim // 2
        self.edge_type_emb  = nn.Embedding(num_edge_types, half)
        self.dist_mlp       = mlp([8, half, half], act=nn.GELU,
                                  dropout=0.0, use_bn=False)
        self.edge_init_proj = nn.Linear(half * 2, edge_dim)

        # GT layers
        self.layers = nn.ModuleList([
            GTLayer(hidden_dim, num_heads, edge_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    # ── Full connectivity helper ──────────────────────────────────────────────

    def _full_edge_index(self,
                         batch_vec:  Tensor,
                         edge_index: Tensor,
                         edge_attr:  Tensor) -> Tuple[Tensor, Tensor]:
        """Return fully-connected within-graph edges.

        Existing edges keep their type; missing pairs get NULL edge type.
        Self-loops are excluded.
        """
        device  = batch_vec.device
        counts  = torch.bincount(batch_vec)
        existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        all_src  = list(edge_index[0].tolist())
        all_dst  = list(edge_index[1].tolist())
        all_attr = list(edge_attr.tolist())

        offset = 0
        for n_b in counts.tolist():
            for i in range(n_b):
                for j in range(n_b):
                    if i == j:
                        continue
                    gi, gj = offset + i, offset + j
                    if (gi, gj) not in existing:
                        all_src.append(gi)
                        all_dst.append(gj)
                        all_attr.append(self.null_edge_type)
            offset += n_b

        full_ei = torch.tensor([all_src, all_dst], dtype=torch.long, device=device)
        full_ea = torch.tensor(all_attr,           dtype=torch.long, device=device)
        return full_ei, full_ea

    def forward(self,
                h:          Tensor,
                pos:        Tensor,
                edge_index: Tensor,
                edge_attr:  Tensor,
                batch:      Tensor,
                bbox:       Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        h          : (N, in_dim)   node features from NodeEncoder
        pos        : (N, 3)        centroid coordinates
        edge_index : (2, E)        sparse edge connectivity from input graph
        edge_attr  : (E,)          edge type indices
        batch      : (N,)          graph index per node
        bbox       : (N, 6) | None [min_xyz | max_xyz] for AABB-gap edge
                                    feature. If None, gap defaults to 0.

        Returns
        -------
        h_enc : (N, hidden_dim)    context-enriched node features
        """
        # Build fully-connected edges
        full_ei, full_ea = self._full_edge_index(batch, edge_index, edge_attr)
        src, dst = full_ei

        # Initialise edge features from type embedding + geometric MLP
        # over (dist, gap, signed direction, per-axis gap).
        r_ij = pos[dst] - pos[src]                                     # (E, 3) signed
        dist = r_ij.norm(dim=-1, keepdim=True).clamp(min=1e-6)        # (E, 1)
        unit_dir = r_ij / dist                                         # (E, 3) signed unit direction
        if bbox is not None:
            half_i = (bbox[:, 3:] - bbox[:, :3]) * 0.5                 # (N, 3)
            sum_half = half_i[src] + half_i[dst]                       # (E, 3)
            abs_diff = r_ij.abs()                                      # (E, 3)
            per_axis_gap = (abs_diff - sum_half).clamp(min=0.0)        # (E, 3)
            gap = per_axis_gap.norm(dim=-1, keepdim=True)              # (E, 1)
        else:
            gap = torch.zeros_like(dist)
            per_axis_gap = torch.zeros_like(unit_dir)
        if not self.use_signed_pair_features:
            unit_dir     = torch.zeros_like(unit_dir)
            per_axis_gap = torch.zeros_like(per_axis_gap)
        type_feat = self.edge_type_emb(full_ea)                        # (E, half)
        dist_feat = self.dist_mlp(
            torch.cat([dist, gap, unit_dir, per_axis_gap], dim=-1))   # (E, half)
        edge_feat = self.edge_init_proj(
            torch.cat([type_feat, dist_feat], dim=-1))                 # (E, edge_dim)

        h = self.input_proj(h)
        for layer in self.layers:
            h, edge_feat = layer(h, full_ei, edge_feat)

        return self.out_norm(h)
