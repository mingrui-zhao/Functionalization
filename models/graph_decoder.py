"""
graph_decoder.py
================
DETR-style parallel slot decoder for graph generation.

Architecture
------------
  1. **Slot initialisation**: max_nodes learned query vectors (object queries).
  2. **Cross-attention layers**: slots attend to encoder node memories, plus
     self-attention between slots (enables slot interaction for edge reasoning).
  3. **Node existence head**: binary MLP per slot → real / virtual.
  4. **Node attribute head**: predict node_type (classification) and
     updated centroid/bbox (regression) per slot.
  5. **Edge attribute head**: pairwise MLP on concatenated slot pairs →
     edge type ∈ {contact, hinge, rail, attached, null}.

All MLPs use GELU activations and dropout as required by the spec.

Key design choices
------------------
- Full parallel decoding (no autoregression).
- Slots do NOT attend to their own position in the sequence —
  permutation-equivariant w.r.t. slot ordering (DETR-like).
- Edge head is O(max_nodes²) but max_nodes=64 is affordable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .node_encoder import mlp


# ── Cross-attention decoder layer ────────────────────────────────────────────

class SlotDecoderLayer(nn.Module):
    """One transformer decoder layer for the slot decoder.

    Order: slot self-attention → cross-attention with encoder memory → FFN.
    Each sub-layer uses pre-norm and dropout residual connections.

    Parameters
    ----------
    hidden_dim : int  — dimension of slot and memory vectors.
    num_heads  : int  — attention heads.
    dropout    : float.
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads:  int = 8,
                 dropout:    float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(hidden_dim, num_heads,
                                                 dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                                 dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self,
                slots: Tensor,
                memory: Tensor,
                memory_key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        slots    : (B, max_nodes, D)   slot query vectors.
        memory   : (B, N_enc, D)       encoder output (padded).
        memory_key_padding_mask : (B, N_enc) bool — True where padding.

        Returns
        -------
        slots_out : (B, max_nodes, D)
        """
        # Slot self-attention
        x = self.norm1(slots)
        x, _ = self.self_attn(x, x, x)
        slots = slots + self.drop(x)

        # Cross-attention: slots attend to encoder memory
        x = self.norm2(slots)
        x, _ = self.cross_attn(x, memory, memory,
                                key_padding_mask=memory_key_padding_mask)
        slots = slots + self.drop(x)

        # FFN
        slots = slots + self.drop(self.ff(self.norm3(slots)))
        return slots


# ── Graph Decoder ────────────────────────────────────────────────────────────

class GraphDecoder(nn.Module):
    """Parallel slot-based graph decoder.

    Takes the per-node encoder memory and decodes into a set of
    output node slots, then predicts edges between all slot pairs.

    Parameters
    ----------
    hidden_dim        : int   — feature dimension (must match encoder output).
    max_nodes         : int   — maximum number of output nodes (slot budget).
    num_dec_layers    : int   — number of SlotDecoderLayer stacks.
    num_heads         : int   — attention heads.
    num_material_types: int   — output material vocabulary size (16 categories + unknown = 17).
    num_edge_types    : int   — output edge type vocabulary (contact/hinge/rail/attached/null = 5).
    dropout           : float.
    """

    def __init__(self,
                 hidden_dim:         int = 256,
                 max_nodes:          int = 64,            # was 32 — see fur_033
                 max_free_slots:     int = 16,            # was 8
                 num_dec_layers:     int = 4,
                 num_heads:          int = 8,
                 num_material_types: int = 17,
                 num_edge_types:     int = 5,
                 dropout:            float = 0.1,
                 use_motion_heads:   bool = True,
                 motion_format:      str  = "v2",
                 use_graph_cond_slots: bool = True,
                 use_signed_pair_features: bool = False):
        super().__init__()
        self.max_nodes        = max_nodes
        self.max_free_slots   = max_free_slots
        self.hidden_dim       = hidden_dim
        self.num_material_types = num_material_types
        self.num_edge_types   = num_edge_types
        self.use_motion_heads = use_motion_heads
        self.motion_format    = motion_format    # "v2" (legacy) or "v3"
        self.use_graph_cond_slots = use_graph_cond_slots
        # When False, the signed-direction (3) and per-axis OBB-gap (3) terms
        # in the per-pair geometry features are zeroed out (capacity stays
        # the same — head MLP input dim is fixed at 2D+10 — but the new
        # signal channels carry no information). Lets us A/B the geometric
        # extension without retraining a different-shaped model.
        self.use_signed_pair_features = use_signed_pair_features

        # Learned object queries — one per free slot.
        # NOTE: max_free_slots bumped from 8 → 16 so the model has room to
        # add multiple missing categories per sample (especially when paired
        # with the new "drop_completion" augmentation which removes 2-3
        # categories at once).
        self.slot_queries = nn.Parameter(torch.randn(max_free_slots, hidden_dim))

        # Graph-level conditioning of free-slot queries: free slots receive
        # a residual derived from a pooled summary of the input graph, so
        # they can decide whether/what to add based on what the graph looks
        # like as a whole. Without this, free slots are just position-coded
        # constants and have no signal that "this graph is missing a top
        # panel". Residual is added after layer-norm to keep magnitudes safe.
        if use_graph_cond_slots:
            self.graph_cond_proj = mlp([hidden_dim, hidden_dim, hidden_dim],
                                        act=nn.GELU, dropout=dropout, use_bn=False)
            self.graph_cond_norm = nn.LayerNorm(hidden_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            SlotDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_dec_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

        # Decoder heads operate on 3-D slot tensors (B, M, D) so BatchNorm
        # cannot be used here — use use_bn=False throughout.

        # ── Node existence head ──────────────────────────────────────────
        self.exist_head = mlp([hidden_dim, hidden_dim // 2, 1],
                              act=nn.GELU, dropout=dropout, use_bn=False)

        # ── Node attribute heads ─────────────────────────────────────────
        # Predicts material category (semantic label), not kinematic type
        self.material_head = mlp([hidden_dim, hidden_dim // 2, num_material_types],
                                 act=nn.GELU, dropout=dropout, use_bn=False)
        self.centroid_head  = mlp([hidden_dim, hidden_dim // 2, 3],
                                   act=nn.GELU, dropout=dropout, use_bn=False)
        # Predict half-size (always positive via softplus); bbox derived as
        # [centroid - half_size, centroid + half_size] — geometrically consistent.
        self.half_size_head = mlp([hidden_dim, hidden_dim // 2, 3],
                                   act=nn.GELU, dropout=dropout, use_bn=False)

        # ── Edge attribute head ──────────────────────────────────────────
        # Input: [si | sj | |ci-cj|(3) | dist(1)]
        self.edge_head = mlp([hidden_dim * 2 + 10, hidden_dim, hidden_dim // 2, num_edge_types],
                             act=nn.GELU, dropout=dropout, use_bn=False)

        # ── Soft-parent head (handle → door/drawer attached) ─────────────
        # For each (handle, candidate) pair, a scalar compatibility score.
        # At inference, softmax over (candidate slots ∪ "no-parent") selects
        # exactly one parent. Input matches edge-head geometry terms for
        # consistent symmetric treatment: [si | sj | |ci-cj|(3) | dist(1)]
        self.parent_score_head = mlp([hidden_dim * 2 + 10, hidden_dim // 2, 1],
                                     act=nn.GELU, dropout=dropout, use_bn=False)
        self.parent_noparent_logit = nn.Parameter(torch.zeros(1))

        # ── Hinge-target head (door → exactly one static via hinge) ──────
        self.hinge_target_head = mlp([hidden_dim * 2 + 10, hidden_dim // 2, 1],
                                     act=nn.GELU, dropout=dropout, use_bn=False)
        self.hinge_nohinge_logit = nn.Parameter(torch.zeros(1))

        # ── Motion attribute heads (per edge pair) ───────────────────────
        if use_motion_heads and motion_format == "v2":
            # Legacy v2: 5 heads
            self.hinge_face_src_head = mlp([hidden_dim * 2 + 10, hidden_dim // 2, 6],
                                            act=nn.GELU, dropout=dropout, use_bn=False)
            self.hinge_face_dst_head = mlp([hidden_dim * 2 + 10, hidden_dim // 2, 6],
                                            act=nn.GELU, dropout=dropout, use_bn=False)
            self.hinge_dir_head = mlp([hidden_dim * 2 + 10, hidden_dim // 2, 2],
                                       act=nn.GELU, dropout=dropout, use_bn=False)
            self.hinge_axis_head = mlp([hidden_dim * 2 + 10, hidden_dim // 2, 3],
                                        act=nn.GELU, dropout=dropout, use_bn=False)
            self.rail_axis_head = mlp([hidden_dim * 2 + 10, hidden_dim // 2, 3],
                                       act=nn.GELU, dropout=dropout, use_bn=False)
        elif use_motion_heads and motion_format == "v3":
            # v3: 3 per-edge heads + 1 per-node head, all 6-way (±X/±Y/±Z in
            # the door/drawer local frame). Kabsch-derived, deduped.
            self.hinge_border_head        = mlp(
                [hidden_dim * 2 + 10, hidden_dim // 2, 6],
                act=nn.GELU, dropout=dropout, use_bn=False)
            self.hinge_axis_signed_head   = mlp(
                [hidden_dim * 2 + 10, hidden_dim // 2, 6],
                act=nn.GELU, dropout=dropout, use_bn=False)
            self.rail_axis_signed_head    = mlp(
                [hidden_dim * 2 + 10, hidden_dim // 2, 6],
                act=nn.GELU, dropout=dropout, use_bn=False)
            # Per-node head: where on the door does the handle sit? 6-way
            # face label in door-local frame (subset of 4 non-thin faces in
            # practice). Operates on single slot features.
            self.handle_border_head       = mlp(
                [hidden_dim, hidden_dim // 2, 6],
                act=nn.GELU, dropout=dropout, use_bn=False)
        elif use_motion_heads and motion_format == "v3_pose_inv":
            # Pose-invariant: hinge edge described in the door's own
            # thin-aware frame, decoupled from the door's world-frame pose.
            #   hinge_border_4   : 4-way (panel-face edges, non-thin)
            #   hinge_axis_sign  : 2-way (sign of axis along the OTHER
            #                       non-thin axis; the axis itself is
            #                       determined by border_4 + the door OBB)
            #   rail_axis_signed : kept 6-way (drawer slide axis)
            #   handle_border_4  : 4-way (handle's panel-face edge)
            self.hinge_border_4_head      = mlp(
                [hidden_dim * 2 + 10, hidden_dim // 2, 4],
                act=nn.GELU, dropout=dropout, use_bn=False)
            self.hinge_axis_sign_head     = mlp(
                [hidden_dim * 2 + 10, hidden_dim // 2, 2],
                act=nn.GELU, dropout=dropout, use_bn=False)
            self.rail_axis_signed_head    = mlp(
                [hidden_dim * 2 + 10, hidden_dim // 2, 6],
                act=nn.GELU, dropout=dropout, use_bn=False)
            self.handle_border_4_head     = mlp(
                [hidden_dim, hidden_dim // 2, 4],
                act=nn.GELU, dropout=dropout, use_bn=False)

    def _build_memory_batch(self,
                            h_enc: Tensor,
                            enc_batch: Tensor,
                            batch_size: int) -> Tuple[Tensor, Tensor]:
        """Pack variable-length encoder outputs into a padded batch tensor.

        Parameters
        ----------
        h_enc     : (N_total, D)  concatenated encoder node features.
        enc_batch : (N_total,)    batch index per node.
        batch_size: int

        Returns
        -------
        memory : (B, N_max, D)
        mask   : (B, N_max) bool  True where padding
        """
        device = h_enc.device
        counts = torch.bincount(enc_batch, minlength=batch_size)
        N_max  = int(counts.max().item())
        memory = torch.zeros(batch_size, N_max, self.hidden_dim, device=device)
        mask   = torch.ones(batch_size, N_max, dtype=torch.bool, device=device)

        for b in range(batch_size):
            nodes_b = h_enc[enc_batch == b]          # (n_b, D)
            n_b = nodes_b.shape[0]
            memory[b, :n_b] = nodes_b
            mask[b, :n_b] = False                    # False = valid token

        return memory, mask

    def forward(self,
                h_enc: Tensor,
                enc_batch: Tensor,
                batch_size: int,
                enc_pos: Tensor | None = None,
                enc_bbox: Tensor | None = None) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        h_enc      : (N_total, D)   encoder output node features.
        enc_batch  : (N_total,)     PyG batch vector.
        batch_size : int            number of graphs in this batch.
        enc_pos    : (N_total, 3)   encoder output positions (used as centroid
                                    prior for anchored slots).

        Returns
        -------
        dict with keys:
          'slot_feats'      : (B, max_nodes, D)
          'exist_logits'    : (B, max_nodes, 1)
          'material_logits' : (B, max_nodes, num_material_types)
          'centroid_pred'   : (B, max_nodes, 3)
          'bbox_pred'       : (B, max_nodes, 6)
          'edge_logits'     : (B, max_nodes, max_nodes, num_edge_types)
          'anchor_mask'     : (B, max_nodes) bool
        """
        device = h_enc.device
        B = batch_size

        # Pack encoder output into padded memory
        memory, mem_mask = self._build_memory_batch(h_enc, enc_batch, B)

        # ── Slot initialisation ──────────────────────────────────────────
        # Slots 0..n_b-1      : anchored to input node encodings (always exist).
        # Slots n_b..n_b+K-1  : learned free queries, K = max_free_slots.
        # Slots beyond n_b+K  : zero-padded (supervised to be non-existent).
        counts = torch.bincount(enc_batch, minlength=B)
        slots_list = []
        anchor_counts: list[int] = []
        anchor_pos = torch.zeros(B, self.max_nodes, 3, device=device)
        anchor_half = torch.zeros(B, self.max_nodes, 3, device=device)

        # Pre-compute per-batch graph summary (mean-pool over encoder nodes)
        # for graph-conditional slot queries.
        graph_summary = None
        if self.use_graph_cond_slots:
            graph_summary = torch.zeros(B, self.hidden_dim, device=device)
            for b in range(B):
                nb = h_enc[enc_batch == b]
                if nb.shape[0] > 0:
                    graph_summary[b] = nb.mean(dim=0)
            # Project + LN so the residual sits in slot-feature space.
            graph_summary = self.graph_cond_norm(self.graph_cond_proj(graph_summary))

        for b in range(B):
            n_b    = min(int(counts[b].item()), self.max_nodes)
            n_free = min(self.max_free_slots, self.max_nodes - n_b)
            n_pad  = self.max_nodes - n_b - n_free

            anchored = h_enc[enc_batch == b][:n_b]                       # (n_b, D)
            free     = self.slot_queries[:n_free]                         # (n_free, D)
            # Add graph-level conditioning to free-slot queries: each free
            # slot starts as (its position-coded query) + (graph summary).
            # Anchor slots already see the full encoder output, so they
            # don't need this residual.
            if self.use_graph_cond_slots and n_free > 0:
                free = free + graph_summary[b].unsqueeze(0)              # (n_free, D)
            pad      = torch.zeros(n_pad, self.hidden_dim, device=device) # (n_pad, D)
            slots_list.append(torch.cat([anchored, free, pad], dim=0))   # (M, D)
            anchor_counts.append(n_b)
            if enc_pos is not None:
                anchor_pos[b, :n_b] = enc_pos[enc_batch == b][:n_b]
            if enc_bbox is not None:
                # bbox is [min_xyz, max_xyz] (6) → half_size = (max - min) / 2
                bb = enc_bbox[enc_batch == b][:n_b]
                anchor_half[b, :n_b] = (bb[:, 3:] - bb[:, :3]) * 0.5

        slots = torch.stack(slots_list, dim=0)                            # (B, M, D)

        # Build anchor mask (used by loss to force existence = 1)
        anchor_mask = torch.zeros(B, self.max_nodes, dtype=torch.bool, device=device)
        for b, n_b in enumerate(anchor_counts):
            anchor_mask[b, :n_b] = True

        # Decode
        for layer in self.layers:
            slots = layer(slots, memory, mem_mask)
        slots = self.out_norm(slots)                                       # (B, M, D)

        # ── Node heads ───────────────────────────────────────────────────
        exist_logits    = self.exist_head(slots)                           # (B, M, 1)
        material_logits = self.material_head(slots)                        # (B, M, C_mat)
        # ── Structural inference constraints (release semantics) ─────────
        # Applied in eval mode only (training is unconstrained, matching the
        # published training recipe; constraints belong at inference):
        #  1. hinge (13) and rail (12) are EDGE kinds, never node materials —
        #     those material classes are disabled for every slot.
        #  2. New (free-slot) nodes can only be completion categories by
        #     construction: handle (0), shelf (1), top panel (9), divider (11).
        # `structural_masks` (default True) applies release inference
        # constraints. train.py sets it False for the WHOLE run so that
        # validation losses stay faithful to the unconstrained training
        # objective (otherwise val samples with dropped doors/drawers would
        # be scored against masked logits, distorting best-val selection).
        if getattr(self, "structural_masks", True) and not self.training:
            neg = -30.0
            material_logits = material_logits.clone()
            material_logits[..., 12] = neg   # rail
            material_logits[..., 13] = neg   # hinge
            allowed = material_logits.new_zeros(material_logits.shape[-1])
            allowed[[0, 1, 9, 11]] = 1.0
            free_mask = ~anchor_mask                                       # (B, M)
            material_logits = torch.where(
                free_mask.unsqueeze(-1) & (allowed < 0.5),
                torch.full_like(material_logits, neg),
                material_logits)
        # Centroid: residual Δpos added to anchor position (zero for free/pad slots)
        delta_centroid  = self.centroid_head(slots)                        # (B, M, 3)
        centroid_pred   = anchor_pos + delta_centroid                      # (B, M, 3)
        # Half-size: always positive via softplus → bbox = [c-s, c+s]
        half_size       = F.softplus(self.half_size_head(slots))           # (B, M, 3)
        bbox_pred       = torch.cat([centroid_pred - half_size,
                                     centroid_pred + half_size], dim=-1)   # (B, M, 6)

        # ── Edge head (pairwise) with rich geometric features ─────────────
        # Per-pair geometry features (10 dims, was 4):
        #    abs_diff    (3) — symmetric magnitude of centroid difference
        #    dist        (1) — centroid Euclidean distance
        #    signed_diff (3) — c_i - c_j (orientation: which side is i on?)
        #    per_axis_gap(3) — OBB surface gap per axis (where do they touch?)
        # The signed direction breaks the i↔j symmetry the abs_diff loses,
        # which the parent_score_head needs to distinguish "handle on left
        # of door A" from "handle on right of door B" when distances tie.
        M  = self.max_nodes
        si = slots.unsqueeze(2).expand(-1, -1, M, -1)                     # (B,M,M,D)
        sj = slots.unsqueeze(1).expand(-1, M, -1, -1)                     # (B,M,M,D)
        ci = centroid_pred.unsqueeze(2).expand(-1, -1, M, -1)             # (B,M,M,3)
        cj = centroid_pred.unsqueeze(1).expand(-1, M, -1, -1)             # (B,M,M,3)
        signed_diff = ci - cj                                              # (B,M,M,3)
        abs_diff    = signed_diff.abs()                                    # (B,M,M,3)
        dist        = abs_diff.norm(dim=-1, keepdim=True)                  # (B,M,M,1)
        # Per-axis OBB-hull gap: max(0, |c_i-c_j| - (half_i + half_j)).
        # Predicted OBBs are canonical (R = I), so per-axis bbox = pos ± half_size.
        half_i = half_size.unsqueeze(2).expand(-1, -1, M, -1)              # (B,M,M,3)
        half_j = half_size.unsqueeze(1).expand(-1, M, -1, -1)              # (B,M,M,3)
        per_axis_gap = (abs_diff - (half_i + half_j)).clamp(min=0.0)       # (B,M,M,3)
        if not self.use_signed_pair_features:
            # Ablation: zero out the new orientation-aware channels so the
            # head MLPs only see (abs_diff, dist) — same input shape, but
            # signed_diff and per_axis_gap carry no information.
            signed_diff  = torch.zeros_like(signed_diff)
            per_axis_gap = torch.zeros_like(per_axis_gap)
        pairs    = torch.cat([si, sj, abs_diff, dist,
                              signed_diff, per_axis_gap], dim=-1)          # (B,M,M,2D+10)
        edge_logits = self.edge_head(pairs)                                # (B,M,M,C_edge)

        # Soft-parent compatibility score per slot pair (B, M, M)
        parent_scores = self.parent_score_head(pairs).squeeze(-1)

        # Hinge-target scores (door → static, analogous to parent_scores)
        hinge_target_scores = self.hinge_target_head(pairs).squeeze(-1)  # (B,M,M)

        out = {
            "slot_feats":       slots,
            "exist_logits":     exist_logits,
            "material_logits":  material_logits,
            "centroid_pred":    centroid_pred,
            "half_size":        half_size,
            "bbox_pred":        bbox_pred,
            "edge_logits":      edge_logits,
            "anchor_mask":      anchor_mask,
            "anchor_pos":       anchor_pos,        # for anchor-consistency loss
            "anchor_half":      anchor_half,       # for anchor-bbox consistency loss
            "parent_scores":    parent_scores,
            "parent_noparent_logit": self.parent_noparent_logit,
            "hinge_target_scores": hinge_target_scores,
            "hinge_nohinge_logit": self.hinge_nohinge_logit,
        }

        # Motion heads (applied to same pair features as edge_head).
        if self.use_motion_heads and self.motion_format == "v2":
            out["hinge_face_src_logits"] = self.hinge_face_src_head(pairs)
            out["hinge_face_dst_logits"] = self.hinge_face_dst_head(pairs)
            out["hinge_dir_logits"]      = self.hinge_dir_head(pairs)
            out["hinge_axis_logits"]     = self.hinge_axis_head(pairs)
            out["rail_axis_logits"]      = self.rail_axis_head(pairs)
        elif self.use_motion_heads and self.motion_format == "v3":
            out["hinge_border_logits"]      = self.hinge_border_head(pairs)     # (B,M,M,6)
            out["hinge_axis_signed_logits"] = self.hinge_axis_signed_head(pairs) # (B,M,M,6)
            out["rail_axis_signed_logits"]  = self.rail_axis_signed_head(pairs)  # (B,M,M,6)
            # Per-slot head (not per-pair) — runs over slot_feats directly.
            out["handle_border_logits"]     = self.handle_border_head(slots)    # (B,M,6)
        elif self.use_motion_heads and self.motion_format == "v3_pose_inv":
            out["hinge_border_4_logits"]    = self.hinge_border_4_head(pairs)   # (B,M,M,4)
            out["hinge_axis_sign_logits"]   = self.hinge_axis_sign_head(pairs)  # (B,M,M,2)
            out["rail_axis_signed_logits"]  = self.rail_axis_signed_head(pairs) # (B,M,M,6)
            out["handle_border_4_logits"]   = self.handle_border_4_head(slots)  # (B,M,4)

        return out
