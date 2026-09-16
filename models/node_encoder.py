"""
node_encoder.py
===============
Encodes per-node features into a single fixed-dimensional embedding vector.

Processing pipeline per node
-----------------------------
  1. PointNet++ (lightweight, set-abstraction) on the (N_i, 3) point cloud.
  2. Learned embedding for node_type (categorical).
  3. Linear projection of bbox (6-dim) and centroid (3-dim).
  4. All three streams are concatenated → NodeEncoder.out_dim.

PointNet++ details
------------------
Two Set-Abstraction (SA) layers with ball-query grouping, followed by a
global max-pool to produce a fixed-size descriptor.  The implementation uses
torch_geometric.nn primitives (PointNetConv + fps / radius) so no external
point-cloud library is needed.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool


# ── Shared helpers ───────────────────────────────────────────────────────────

def mlp(dims: List[int], act: type = nn.GELU, dropout: float = 0.1,
        use_bn: bool = True) -> nn.Sequential:
    """Build a multi-layer perceptron with optional BatchNorm, GELU and dropout.

    Parameters
    ----------
    dims   : list of layer widths, e.g. [in, hidden, out].
    act    : activation class (default GELU).
    dropout: dropout probability on intermediate layers.
    use_bn : if True, insert BatchNorm1d after each intermediate linear layer.
             Set False when the MLP will be applied to 3-D tensors (B, *, C).
    """
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:          # no norm/act on final layer
            if use_bn:
                # Ablation: BatchNorm1d → LayerNorm so part-level statistics
                # aren't averaged across the batch.
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ── Set-Abstraction layer (PointNet++) ───────────────────────────────────────

class SALayer(nn.Module):
    """Single Set-Abstraction (SA) layer for PointNet++.

    Uses farthest-point sampling to select centroids, then groups
    local neighbourhoods with a radius ball-query.

    Parameters
    ----------
    ratio : float
        FPS sub-sampling ratio (0 < ratio ≤ 1).
    radius : float
        Radius of the ball query neighbourhood.
    max_neighbors : int
        Maximum number of points per neighbourhood.
    in_channels : int
        Input per-point feature dimension (0 for raw xyz only).
    out_channels : List[int]
        Channel widths for the PointNetConv MLP.
    """

    def __init__(self,
                 ratio: float,
                 ball_radius: float,
                 max_neighbors: int,
                 in_channels: int,
                 out_channels: List[int]):
        super().__init__()
        self.ratio         = ratio
        self.ball_radius   = ball_radius
        self.max_neighbors = max_neighbors

        # PointNetConv expects local_nn: (in_channels + 3) → out_channels
        in_ch  = in_channels + 3                  # +3 for relative xyz
        dims   = [in_ch] + out_channels
        local_nn = mlp(dims, act=nn.GELU, dropout=0.0)
        self.conv = PointNetConv(local_nn=local_nn, global_nn=None, add_self_loops=False)
        self.out_channels = out_channels[-1]

    def forward(self, x: Optional[Tensor], pos: Tensor, batch: Tensor):
        """
        Parameters
        ----------
        x    : (M, C) | None  per-point features (None for first layer).
        pos  : (M, 3)         per-point xyz.
        batch: (M,)           batch index per point.

        Returns
        -------
        x_new   : (M', out_channels)
        pos_new : (M', 3)
        batch_new: (M',)
        """
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        row, col = radius(pos, pos[idx], self.ball_radius,
                          batch, batch[idx],
                          max_num_neighbors=self.max_neighbors)
        edge_index = torch.stack([col, row], dim=0)   # col→source, row→target centroid
        x_new = self.conv(x, (pos, pos[idx]), edge_index)
        return x_new, pos[idx], batch[idx]


# ── Lightweight PointNet++ encoder ───────────────────────────────────────────

class PointNetPPEncoder(nn.Module):
    """Encode a variable-size point cloud into a fixed-dim latent vector.

    Two SA layers followed by global max-pooling.

    Parameters
    ----------
    out_dim : int
        Output embedding dimension.
    dropout : float
        Dropout probability in the final MLP.
    """

    def __init__(self, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # SA layer 1: raw xyz → 64-dim, 50 % sub-sample
        self.sa1 = SALayer(
            ratio=0.5, ball_radius=0.2, max_neighbors=32,
            in_channels=0, out_channels=[32, 64])
        # SA layer 2: 64-dim → 128-dim, 50 % sub-sample of SA1 output
        self.sa2 = SALayer(
            ratio=0.5, ball_radius=0.4, max_neighbors=32,
            in_channels=64, out_channels=[64, 128])
        # Final projection
        self.head = mlp([128, 128, out_dim], act=nn.GELU, dropout=dropout)
        self.out_dim = out_dim

    def forward(self,
                point_clouds: List[Tensor],
                pc_node_batch: Tensor) -> Tensor:
        """
        Parameters
        ----------
        point_clouds : list of (Ni, 3) tensors — one per node across the batch.
        pc_node_batch : (total_nodes,) indicating which *graph* each node belongs to.
            (same as PyG's `batch` vector on the flattened graph batch)

        Returns
        -------
        embeddings : (total_nodes, out_dim)
        """
        # Flatten all point clouds into one big point cloud with a per-point node-id
        pts_list   = []
        node_batch = []                  # per-point: which node index (global)
        for node_idx, pc in enumerate(point_clouds):
            pts_list.append(pc)
            node_batch.extend([node_idx] * pc.shape[0])

        pos   = torch.cat(pts_list, dim=0).to(pc_node_batch.device)   # (P, 3)
        batch = torch.tensor(node_batch, dtype=torch.long, device=pos.device)

        # SA layers
        x1, pos1, batch1 = self.sa1(None, pos, batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)

        # Global max-pool → one vector per node
        total_nodes = len(point_clouds)
        out = global_max_pool(x2, batch2, size=total_nodes)          # (N, 128)

        # Handle nodes whose SA output might be missing (degenerate PC)
        # global_max_pool fills with 0 for empty nodes — that's fine.
        return self.head(out)                                          # (N, out_dim)


# ── Full Node Encoder ────────────────────────────────────────────────────────

class NodeEncoder(nn.Module):
    """Encode per-node attributes into a single embedding vector.

    Streams:
    1. Learned embedding for material (semantic, categorical) → type_emb_dim   [GRAPH]
    2. PointNet++ on point cloud → pc_dim                                       [GEOM]
    3. MLP on [centroid(3) | bbox(6)] → geom_dim                                [GEOM]

    Graph-primary defaults rebalance dims so the GRAPH channel (material) is the
    dominant component (~57% of out_dim by default), inverting the previous
    geometry-dominant 14/86 split.

    `geometry_dropout_p`: at training time, with this probability the entire
    geometry stream (pc_enc + geom_mlp outputs) is replaced with zero so the
    encoder has to solve from material + edges alone. Inference still uses
    geometry. Default 0 keeps legacy behaviour.

    Parameters
    ----------
    num_material_types : int  — material category vocab (16 categories + unknown = 17).
    type_emb_dim       : int  — embedding dim for the material embedding.
    pc_dim             : int  — PointNet++ output dim.
    geom_dim           : int  — geometry MLP output dim.
    geometry_dropout_p : float — prob. of zeroing the geometry stream during training.
    dropout            : float.
    """

    def __init__(self,
                 num_material_types: int   = 17,
                 type_emb_dim:       int   = 128,
                 pc_dim:             int   = 64,
                 geom_dim:           int   = 32,
                 geometry_dropout_p: float = 0.0,
                 dropout:            float = 0.1):
        super().__init__()
        self.material_emb = nn.Embedding(num_material_types, type_emb_dim)
        self.pc_enc       = PointNetPPEncoder(out_dim=pc_dim, dropout=dropout)
        self.geom_mlp     = mlp([9, 64, geom_dim], act=nn.GELU, dropout=dropout)
        self.geometry_dropout_p = float(geometry_dropout_p)

        self.out_dim = type_emb_dim + pc_dim + geom_dim
        self.pc_dim  = pc_dim
        self.geom_dim = geom_dim

    def forward(self,
                material: Tensor,
                pos: Tensor,
                bbox: Tensor,
                point_clouds: List[Tensor],
                pc_node_batch: Tensor) -> Tensor:
        """
        Parameters
        ----------
        material      : (N,)   long  — material category index
        pos           : (N, 3) float — centroid
        bbox          : (N, 6) float — [min_xyz | max_xyz]
        point_clouds  : list of (Ni, 3) — per-node point clouds
        pc_node_batch : (N,)   long  — which graph each node belongs to

        Returns
        -------
        h : (N, out_dim)
        """
        h_mat  = self.material_emb(material)                     # (N, type_emb_dim)
        h_pc   = self.pc_enc(point_clouds, pc_node_batch)        # (N, pc_dim)
        geom   = torch.cat([pos, bbox], dim=-1)                  # (N, 9)
        h_geom = self.geom_mlp(geom)                             # (N, geom_dim)

        # Geometry dropout: at training time, randomly zero the entire geom
        # stream per BATCH so the encoder must learn a graph-only fallback.
        # Per-batch (not per-node) so all nodes in a graph drop together.
        if self.training and self.geometry_dropout_p > 0:
            if torch.rand((), device=h_pc.device).item() < self.geometry_dropout_p:
                h_pc   = torch.zeros_like(h_pc)
                h_geom = torch.zeros_like(h_geom)

        return torch.cat([h_mat, h_pc, h_geom], dim=-1)         # (N, out_dim)
