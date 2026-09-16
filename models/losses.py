"""
losses.py
=========
Training objectives for graph-to-graph translation.

Components
----------
1. **HungarianMatcher**: bipartite matching between predicted slots and GT nodes
   using scipy.optimize.linear_sum_assignment.  Match cost is a weighted
   combination of node-type cross-entropy and centroid L2 distance.

2. **NodeSetLoss**: after matching, compute
   - Cross-entropy on node type (matched slots only).
   - L2 on centroid prediction.
   - L2 on bbox prediction.
   - Binary cross-entropy on node existence (all slots: real vs. virtual).

3. **EdgeLoss**: cross-entropy over all slot pairs for edge type
   (including null class).  Handles class imbalance by up-weighting
   non-null edges.

4. **GraphTranslationLoss**: assembles all three losses with configurable weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

NULL_EDGE_CLASS = 4     # index for "no edge" in edge_type vocab


# ── Hungarian Matcher ────────────────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """Match predicted slots to ground-truth nodes using optimal bipartite matching.

    Cost = λ_cls * CE(type_pred, type_gt)  +  λ_pos * ||centroid_pred - centroid_gt||

    Parameters
    ----------
    cost_class : float  — weight on classification cost term.
    cost_pos   : float  — weight on centroid distance cost term.
    """

    def __init__(self, cost_class: float = 1.0, cost_pos: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_pos   = cost_pos

    @torch.no_grad()
    def forward(self,
                material_logits: Tensor,
                centroid_pred:   Tensor,
                gt_materials:    List[Tensor],
                gt_centroids:    List[Tensor],
                pinned:          Optional[List[Optional[Tensor]]] = None
                ) -> List[Tuple[Tensor, Tensor]]:
        """
        Parameters
        ----------
        material_logits : (B, M, C)  material category logits per slot.
        centroid_pred   : (B, M, 3)  centroid predictions per slot.
        gt_materials    : list of B tensors, each (n_i,) long — GT material indices.
        gt_centroids    : list of B tensors, each (n_i, 3) float.

        Returns
        -------
        indices : list of B tuples (pred_idx, gt_idx) — both LongTensors of length min(M, n_i).
        """
        B, M, C = material_logits.shape
        gt_types    = gt_materials
        type_logits = material_logits
        indices = []

        for b in range(B):
            n_gt = gt_types[b].shape[0]
            if n_gt == 0:
                indices.append((
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                ))
                continue

            # Classification cost: soft-max probabilities vs. one-hot GT
            prob = type_logits[b].softmax(-1)              # (M, C)
            gt_oh = F.one_hot(gt_types[b], C).float()      # (n_gt, C)
            cost_cls = -(prob @ gt_oh.T)                   # (M, n_gt)

            # Position cost: pairwise L2
            dp = centroid_pred[b].unsqueeze(1) - gt_centroids[b].unsqueeze(0)  # (M,n,3)
            cost_pos = dp.norm(dim=-1)                     # (M, n_gt)

            cost = (self.cost_class * cost_cls
                    + self.cost_pos  * cost_pos).cpu().numpy()

            # Pinned assignment: anchored slot k is force-matched to its own
            # GT node (pinned[b][k]); Hungarian runs only over the remaining
            # free slots x unmatched GT nodes. Prevents symmetric-sibling
            # identity swaps under mirror augmentation.
            pin = pinned[b] if pinned is not None else None
            if pin is not None and pin.numel() > 0:
                pin = pin.detach().cpu()
                rows, cols = [], []
                used_r, used_c = set(), set()
                for k in range(min(int(pin.numel()), M)):
                    j = int(pin[k])
                    if 0 <= j < n_gt and k not in used_r and j not in used_c:
                        rows.append(k); cols.append(j)
                        used_r.add(k); used_c.add(j)
                free_r = [r for r in range(M) if r not in used_r]
                free_c = [c for c in range(n_gt) if c not in used_c]
                if free_c and free_r:
                    import numpy as _np
                    sub = cost[_np.ix_(free_r, free_c)]
                    pr, gc = linear_sum_assignment(sub)
                    rows += [free_r[r] for r in pr]
                    cols += [free_c[c] for c in gc]
                pred_idx = torch.as_tensor(rows, dtype=torch.long)
                gt_idx   = torch.as_tensor(cols, dtype=torch.long)
                indices.append((pred_idx, gt_idx))
                continue

            pred_idx, gt_idx = linear_sum_assignment(cost)
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(gt_idx,   dtype=torch.long),
            ))

        return indices


# ── Node set loss ────────────────────────────────────────────────────────────

class NodeSetLoss(nn.Module):
    """Loss over matched (slot, GT node) pairs.

    Parameters
    ----------
    w_type     : float — weight on node type CE.
    w_centroid : float — weight on centroid L2.
    w_bbox     : float — weight on bbox L2.
    w_exist    : float — weight on existence BCE.
    """

    def __init__(self,
                 w_type:     float = 1.0,
                 w_centroid: float = 2.0,
                 w_bbox:     float = 1.0,
                 w_exist:    float = 1.0,
                 w_anchor:   float = 0.0,
                 w_anchor_bbox: float = 0.0,
                 w_free_exist: float = 0.0,
                 free_exist_pw_floor: float = 5.0,
                 w_count:    float = 0.0):
        super().__init__()
        self.w_type     = w_type
        self.w_centroid = w_centroid
        self.w_bbox     = w_bbox
        self.w_exist    = w_exist
        # Anchor-consistency: penalise ||delta_centroid||² on every anchored
        # slot regardless of Hungarian matching. Catches the OOD side-panel
        # drift where the matcher swaps a slot with a wrong GT node and the
        # original slot's centroid then drifts unchecked.
        self.w_anchor   = w_anchor
        # Anchor-bbox consistency: ||half_size - anchor_half||² on every
        # anchored slot. Symmetric to w_anchor but on bbox half-extents.
        # Forces anchored slots' predicted half_size to match the input
        # node's half_size (which the model receives via encoder bbox).
        self.w_anchor_bbox = w_anchor_bbox
        # Free-slot existence reweighting (completeness signal): when > 0, an
        # auxiliary BCE term is computed ONLY on free slots, with pos_weight
        # floored at `free_exist_pw_floor` so free-slot positives (= GT nodes
        # the input was missing) get an undiluted gradient. Without this, the
        # main BCE pools all 64 slots and the abundant always-positive anchor
        # slots dominate the per-batch pos_weight (which falls to ~1-4),
        # causing free slots to collapse to "always 0" → model is reluctant
        # to add new nodes. Default 0 = legacy behaviour.
        self.w_free_exist        = w_free_exist
        self.free_exist_pw_floor = free_exist_pw_floor
        # Total-count regularizer: a soft, single-scalar-per-graph signal that
        # sum(sigmoid(exist_logits)) should equal the GT node count. Pushes
        # the model to halt firing once the predicted total matches the
        # target — counter-balances the over-firing tendency of the free-slot
        # BCE reweighting. Per-graph and category-agnostic; aligned with
        # "completeness emerges from data" not "per-category cardinality rules".
        self.w_count             = w_count

    def forward(self,
                pred:    Dict[str, Tensor],
                targets: List[Dict],
                indices: List[Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        pred : dict from GraphDecoder.forward(), containing:
               'exist_logits'    (B, M, 1)
               'material_logits' (B, M, C_mat)
               'centroid_pred'   (B, M, 3)
               'bbox_pred'       (B, M, 6)
        targets : list of B dicts, each with keys:
               'material'   (n_i,) long  — material category index
               'centroid'   (n_i, 3) float
               'bbox'       (n_i, 6) float
        indices : output of HungarianMatcher.

        Returns
        -------
        losses : dict with keys 'loss_material', 'loss_centroid', 'loss_bbox', 'loss_exist'.
        """
        B, M = pred["exist_logits"].shape[:2]
        device = pred["exist_logits"].device

        loss_material = torch.tensor(0.0, device=device)
        loss_centroid = torch.tensor(0.0, device=device)
        loss_bbox     = torch.tensor(0.0, device=device)

        n_matched = 0
        exist_targets = torch.zeros(B, M, device=device)

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            pred_idx = pred_idx.to(device)
            gt_idx   = gt_idx.to(device)
            n        = pred_idx.numel()
            n_matched += n

            exist_targets[b, pred_idx] = 1.0

            # Material category classification
            loss_material = loss_material + F.cross_entropy(
                pred["material_logits"][b][pred_idx],
                targets[b]["material"][gt_idx])

            # Centroid
            loss_centroid = loss_centroid + F.mse_loss(
                pred["centroid_pred"][b][pred_idx],
                targets[b]["centroid"][gt_idx])

            # Bbox via half-size: GT half_size = (max - min) / 2
            gt_bbox      = targets[b]["bbox"][gt_idx]              # (n, 6)
            gt_half_size = (gt_bbox[:, 3:] - gt_bbox[:, :3]) / 2  # (n, 3)
            loss_bbox = loss_bbox + F.mse_loss(
                pred["half_size"][b][pred_idx], gt_half_size)

        loss_material = loss_material / B
        loss_centroid = loss_centroid / B
        loss_bbox     = loss_bbox     / B

        # Anchored slots (= input nodes) must always exist — override targets
        if "anchor_mask" in pred:
            exist_targets = torch.maximum(exist_targets, pred["anchor_mask"].float())

        # Existence BCE (all slots)
        exist_logits = pred["exist_logits"].squeeze(-1)    # (B, M)
        loss_exist = F.binary_cross_entropy_with_logits(
            exist_logits, exist_targets,
            pos_weight=torch.tensor(M / max(exist_targets.sum().item(), 1),
                                    device=device))

        # Auxiliary free-slot BCE — completeness signal. Computed only on
        # slots that are NOT anchored, with pos_weight balanced against the
        # free-slot population (not the all-slot population). This stops the
        # abundant always-positive anchor slots from diluting the gradient
        # on free-slot positives, which are the model's only signal that
        # missing nodes should be added back.
        loss_free_exist = torch.tensor(0.0, device=device)
        if self.w_free_exist > 0 and "anchor_mask" in pred:
            free_mask = ~pred["anchor_mask"].bool()           # (B, M)
            if free_mask.any():
                free_logits  = exist_logits[free_mask]
                free_targets = exist_targets[free_mask]
                free_pos = free_targets.sum().item()
                free_neg = free_targets.numel() - free_pos
                free_pw  = max(free_neg / max(free_pos, 1.0),
                                float(self.free_exist_pw_floor))
                loss_free_exist = F.binary_cross_entropy_with_logits(
                    free_logits, free_targets,
                    pos_weight=torch.tensor(free_pw, device=device))

        # Total-count regularizer: per-graph predicted-fire-count vs GT count.
        # Single scalar per graph, no per-category rules. Tells the model
        # "fire approximately N slots, where N is the GT node count" without
        # dictating which categories. Pairs with w_free_exist to give the
        # model BOTH a "do fire missing nodes" signal AND a "but don't fire
        # extras beyond what's needed" signal.
        loss_count = torch.tensor(0.0, device=device)
        if self.w_count > 0 and "anchor_mask" in pred:
            # Apply count regularizer ONLY on free slots. Anchored slots are
            # always-positive by construction, so a global count loss could
            # be satisfied by dimming anchors instead of firing free slots
            # (v2 did this and collapsed to 0-node predictions). We compare
            # the SUM of free-slot existence probabilities against the
            # number of GT nodes NOT covered by the input.
            anchor_mask = pred["anchor_mask"].bool()             # (B, M)
            n_anchored  = anchor_mask.float().sum(dim=-1)         # (B,)
            exist_probs = torch.sigmoid(exist_logits)             # (B, M)
            free_pred_sum = (exist_probs * (~anchor_mask).float()).sum(dim=-1)  # (B,)
            target_total = torch.tensor(
                [t["material"].shape[0] for t in targets],
                device=device, dtype=torch.float32)               # (B,)
            free_target = (target_total - n_anchored).clamp(min=0.0)  # (B,)
            # sqrt(abs) of count error per sample, then mean. Sub-linear so a
            # single outlier graph (e.g. fur_228 with 67 nodes vs max_free_slots=4)
            # doesn't dominate the batch via the squared term.
            diff = (free_pred_sum - free_target).abs()
            loss_count = (diff + 1e-6).sqrt().mean()

        # Anchor-consistency: ||centroid_pred - anchor_pos||² on anchored slots.
        # Requires `anchor_pos` and `anchor_mask` from the decoder forward.
        loss_anchor = torch.tensor(0.0, device=device)
        if (self.w_anchor > 0
                and "anchor_pos" in pred and "anchor_mask" in pred):
            delta = pred["centroid_pred"] - pred["anchor_pos"]   # (B, M, 3)
            mask = pred["anchor_mask"].bool()                     # (B, M)
            if mask.any():
                loss_anchor = (delta[mask] ** 2).sum(-1).mean()

        # Anchor-bbox consistency: ||half_size - anchor_half||² on anchored slots.
        # Symmetric to anchor_pos. Stops the half_size_head from drifting on
        # anchored slots (which previously had no skip back to input bbox).
        loss_anchor_bbox = torch.tensor(0.0, device=device)
        if (self.w_anchor_bbox > 0
                and "anchor_half" in pred and "anchor_mask" in pred):
            delta_h = pred["half_size"] - pred["anchor_half"]      # (B, M, 3)
            mask = pred["anchor_mask"].bool()
            if mask.any():
                loss_anchor_bbox = (delta_h[mask] ** 2).sum(-1).mean()

        return {
            "loss_material":    self.w_type        * loss_material,
            "loss_centroid":    self.w_centroid    * loss_centroid,
            "loss_bbox":        self.w_bbox        * loss_bbox,
            "loss_exist":       self.w_exist       * loss_exist,
            "loss_anchor":      self.w_anchor      * loss_anchor,
            "loss_anchor_bbox": self.w_anchor_bbox * loss_anchor_bbox,
            "loss_free_exist":  self.w_free_exist  * loss_free_exist,
            "loss_count":       self.w_count       * loss_count,
        }


# ── Edge loss ─────────────────────────────────────────────────────────────────

class EdgeLoss(nn.Module):
    """Cross-entropy over all pairwise slot predictions.

    GT edge matrix is built from the matching indices:
    - matched slot pairs that have an edge in GT → GT edge type
    - all other pairs → null class

    Parameters
    ----------
    num_edge_types : int   — total edge classes including null (default 5).
    null_weight    : float — weight on the null class. Legacy used 0.1-0.2 to
                             *down-weight* null, which encouraged predicting
                             null and produced floating-subgraph outputs. The
                             default here is 1.0 (no down-weighting); pair with
                             `class_weights` for inverse-frequency balancing.
    class_weights  : list[float] | None  — explicit length-num_edge_types
                             override for [contact, hinge, rail, attached, null].
                             When provided, replaces the null_weight default.
    """

    def __init__(self,
                 num_edge_types: int = 5,
                 null_weight:    float = 1.0,
                 label_smoothing: float = 0.0,
                 class_weights:   Optional[List[float]] = None):
        super().__init__()
        self.num_edge_types  = num_edge_types
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            assert len(class_weights) == num_edge_types, \
                f"class_weights must have length {num_edge_types}"
            cw = torch.tensor(class_weights, dtype=torch.float32)
        else:
            cw = torch.cat([
                torch.ones(num_edge_types - 1),
                torch.tensor([null_weight]),
            ])
        self.register_buffer("class_weights", cw)

    def forward(self,
                edge_logits: Tensor,
                targets:     List[Dict],
                indices:     List[Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        edge_logits : (B, M, M, C_edge)
        targets : list of B dicts with keys:
                  'edge_index' (2, E_i) long  — GT edge pairs (original node indices)
                  'edge_attr'  (E_i,)   long  — GT edge types
                  'node_type'  (n_i,)   long  (used for node count)
        indices : HungarianMatcher output (pred_idx, gt_idx) per graph.

        Returns
        -------
        losses : dict with key 'loss_edge'.
        """
        B, M, _, C = edge_logits.shape
        device = edge_logits.device
        loss_edge = torch.tensor(0.0, device=device)

        for b, (pred_idx, gt_idx) in enumerate(indices):
            # Build GT edge matrix in slot space (M x M), default null
            gt_edge_mat = torch.full((M, M), NULL_EDGE_CLASS,
                                     dtype=torch.long, device=device)

            if pred_idx.numel() > 0 and targets[b]["edge_index"].shape[1] > 0:
                pred_idx = pred_idx.to(device)
                gt_idx   = gt_idx.to(device)
                # Map GT node index → matched slot index
                n_gt = targets[b]["material"].shape[0]
                gt2slot = torch.full((n_gt,), -1, dtype=torch.long, device=device)
                gt2slot[gt_idx] = pred_idx

                ei  = targets[b]["edge_index"].to(device)   # (2, E)
                ea  = targets[b]["edge_attr"].to(device)    # (E,)

                for e_idx in range(ei.shape[1]):
                    gs = ei[0, e_idx].item()
                    gd = ei[1, e_idx].item()
                    ps = gt2slot[gs].item() if gs < n_gt else -1
                    pd = gt2slot[gd].item() if gd < n_gt else -1
                    if ps >= 0 and pd >= 0:
                        etype = ea[e_idx].item()
                        gt_edge_mat[ps, pd] = etype
                        gt_edge_mat[pd, ps] = etype     # undirected

            # Cross-entropy with class weights
            logits_b = edge_logits[b].view(M * M, C)       # (M²,  C)
            gt_flat  = gt_edge_mat.view(M * M)              # (M²,)
            loss_edge = loss_edge + F.cross_entropy(
                logits_b, gt_flat,
                weight=self.class_weights.to(device),
                label_smoothing=self.label_smoothing)

        return {"loss_edge": loss_edge / B}


# ── Handle-parent (softmax over parents) loss ────────────────────────────────

HANDLE_MAT, DOOR_MAT, DRAWER_MAT = 0, 5, 6
ATTACHED_EDGE = 3


class ParentLoss(nn.Module):
    """Softmax over candidate parents for each handle.

    For every handle slot the model scores each other slot + a "no-parent"
    option. Target is the slot index of the handle's unique parent
    (door/drawer with an ATTACHED edge in GT), or the "no-parent" class
    if the handle is an orphan.

    Uses the Hungarian matcher's slot ↔ GT-node alignment to figure out
    which slot is the parent.
    """

    def forward(self,
                pred: Dict[str, Tensor],
                targets: List[Dict],
                indices: List[Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
        scores = pred["parent_scores"]          # (B, M, M)
        noparent_logit = pred["parent_noparent_logit"]  # (1,)
        material_logits = pred["material_logits"]       # (B, M, C_mat)
        B, M, _ = scores.shape
        device = scores.device

        # Build GT adjacency per batch: for each GT node, its ATTACHED parent
        # among door/drawer nodes.
        total_loss = torch.tensor(0.0, device=device)
        n_terms = 0

        # Predicted material index per slot (argmax) — used to identify
        # candidate door/drawer slots.
        pred_mats = material_logits.argmax(-1)          # (B, M)

        for b, (pred_idx, gt_idx) in enumerate(indices):
            tgt = targets[b]
            n_gt = int(tgt["material"].shape[0])
            if n_gt == 0 or pred_idx.numel() == 0:
                continue
            pred_idx = pred_idx.to(device); gt_idx = gt_idx.to(device)
            mat_gt = tgt["material"].to(device)          # (n_gt,)

            # Build gt2slot mapping
            gt2slot = torch.full((n_gt,), -1, dtype=torch.long, device=device)
            gt2slot[gt_idx] = pred_idx

            # Identify GT handle nodes and their (unique) parent GT idx
            ei = tgt["edge_index"].to(device)
            ea = tgt["edge_attr"].to(device)
            handle_gt_idx = (mat_gt == HANDLE_MAT).nonzero(as_tuple=True)[0]

            for h_gt in handle_gt_idx.tolist():
                h_slot = int(gt2slot[h_gt].item())
                if h_slot < 0: continue   # unmatched handle — skip
                # Find parent in GT
                parent_slot = -1  # "no-parent" default
                if ei.shape[1] > 0:
                    for e_idx in range(ei.shape[1]):
                        s, d = int(ei[0, e_idx].item()), int(ei[1, e_idx].item())
                        et = int(ea[e_idx].item())
                        if et != ATTACHED_EDGE: continue
                        if s == h_gt:
                            other = d
                        elif d == h_gt:
                            other = s
                        else:
                            continue
                        if other < 0 or other >= n_gt: continue
                        if mat_gt[other].item() in (DOOR_MAT, DRAWER_MAT):
                            parent_slot = int(gt2slot[other].item())
                            if parent_slot >= 0:
                                break

                # Build candidate set = slots currently predicted as door/drawer
                cand_mask = (pred_mats[b] == DOOR_MAT) | (pred_mats[b] == DRAWER_MAT)
                # Also include the GT parent slot if it was matched but the
                # predicted material happens to disagree (rare)
                if parent_slot >= 0:
                    cand_mask[parent_slot] = True

                cand_idx = cand_mask.nonzero(as_tuple=True)[0]
                if cand_idx.numel() == 0:
                    # Fallback: model has no candidate. Supervise "no parent".
                    logits = noparent_logit.expand(1)
                    target = torch.tensor([0], dtype=torch.long, device=device)
                else:
                    cand_scores = scores[b, h_slot, cand_idx]       # (K,)
                    logits = torch.cat([cand_scores,
                                        noparent_logit.expand(1)])  # (K+1,)
                    if parent_slot >= 0:
                        target_idx = (cand_idx == parent_slot).nonzero(as_tuple=True)[0]
                        if target_idx.numel() > 0:
                            target = torch.tensor([int(target_idx.item())],
                                                  dtype=torch.long, device=device)
                        else:
                            target = torch.tensor([len(cand_idx)],
                                                  dtype=torch.long, device=device)
                    else:
                        target = torch.tensor([len(cand_idx)],
                                              dtype=torch.long, device=device)
                total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), target)
                n_terms += 1

        if n_terms == 0:
            return {"loss_parent": torch.tensor(0.0, device=device)}
        return {"loss_parent": total_loss / n_terms}


# ── Hinge-target (door → one static via hinge) loss ──────────────────────────

class HingeTargetLoss(nn.Module):
    """Softmax over candidate static nodes per door for the single hinge connection.
    Mirrors ParentLoss but for doors and hinge edges."""

    def forward(self, pred, targets, indices):
        scores = pred.get("hinge_target_scores")
        if scores is None:
            return {"loss_hinge_target": torch.tensor(0.0)}
        nohinge = pred["hinge_nohinge_logit"]
        mat_logits = pred["material_logits"]
        B, M, _ = scores.shape
        device = scores.device
        pred_mats = mat_logits.argmax(-1)
        total_loss = torch.tensor(0.0, device=device)
        n_terms = 0
        static_mats = {1,2,3,4,7,8,9,10,11,12,14,15}  # non-dynamic materials

        for b, (pred_idx, gt_idx) in enumerate(indices):
            tgt = targets[b]
            n_gt = int(tgt["material"].shape[0])
            if n_gt == 0 or pred_idx.numel() == 0: continue
            pred_idx_d = pred_idx.to(device); gt_idx_d = gt_idx.to(device)
            mat_gt = tgt["material"].to(device)
            gt2slot = torch.full((n_gt,), -1, dtype=torch.long, device=device)
            gt2slot[gt_idx_d] = pred_idx_d
            ei = tgt["edge_index"].to(device)
            ea = tgt["edge_attr"].to(device)

            door_gt_idx = (mat_gt == DOOR_MAT).nonzero(as_tuple=True)[0]
            for d_gt in door_gt_idx.tolist():
                d_slot = int(gt2slot[d_gt].item())
                if d_slot < 0: continue
                # Find GT hinge partner
                partner_slot = -1
                if ei.shape[1] > 0:
                    for e_idx in range(ei.shape[1]):
                        s, d = int(ei[0,e_idx].item()), int(ei[1,e_idx].item())
                        et = int(ea[e_idx].item())
                        if et != 1: continue  # 1 = hinge
                        other = d if s == d_gt else (s if d == d_gt else -1)
                        if other < 0 or other >= n_gt: continue
                        if int(mat_gt[other].item()) in static_mats:
                            ps = int(gt2slot[other].item())
                            if ps >= 0:
                                partner_slot = ps
                                break

                cand_mask = torch.zeros(M, dtype=torch.bool, device=device)
                for ci in range(M):
                    if int(pred_mats[b, ci].item()) in static_mats:
                        cand_mask[ci] = True
                if partner_slot >= 0:
                    cand_mask[partner_slot] = True
                cand_idx = cand_mask.nonzero(as_tuple=True)[0]
                if cand_idx.numel() == 0: continue

                cand_scores = scores[b, d_slot, cand_idx]
                logits = torch.cat([cand_scores, nohinge.expand(1)])
                if partner_slot >= 0:
                    ti = (cand_idx == partner_slot).nonzero(as_tuple=True)[0]
                    target = torch.tensor([int(ti.item()) if ti.numel() > 0 else len(cand_idx)],
                                          dtype=torch.long, device=device)
                else:
                    target = torch.tensor([len(cand_idx)], dtype=torch.long, device=device)
                total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), target)
                n_terms += 1

        if n_terms == 0:
            return {"loss_hinge_target": torch.tensor(0.0, device=device)}
        return {"loss_hinge_target": total_loss / n_terms}


# ── Motion attribute losses ──────────────────────────────────────────────────

class MotionLoss(nn.Module):
    """v2 (legacy): per-edge losses for the relative face-pair encoding.

    Heads: hinge_face_src, hinge_face_dst, hinge_axis, hinge_dir, rail_axis.
    Each only contributes a term when the GT label is ≥ 0 (unknowns skipped)."""

    def forward(self, pred, targets, indices):
        device = pred["material_logits"].device
        B, M, _, _ = pred["edge_logits"].shape
        loss_fs = torch.tensor(0.0, device=device)
        loss_fd = torch.tensor(0.0, device=device)
        loss_ha = torch.tensor(0.0, device=device)
        loss_hd = torch.tensor(0.0, device=device)
        loss_ra = torch.tensor(0.0, device=device)
        n_fs = n_fd = n_ha = n_hd = n_ra = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            tgt = targets[b]
            n_gt = int(tgt["material"].shape[0])
            if n_gt == 0 or pred_idx.numel() == 0:
                continue
            pred_idx_d = pred_idx.to(device); gt_idx_d = gt_idx.to(device)
            gt2slot = torch.full((n_gt,), -1, dtype=torch.long, device=device)
            gt2slot[gt_idx_d] = pred_idx_d

            ei = tgt["edge_index"].to(device)
            ea = tgt["edge_attr"].to(device)
            fs_gt = tgt.get("hinge_face_src")
            fd_gt = tgt.get("hinge_face_dst")
            ha_gt = tgt.get("hinge_axis")
            hd_gt = tgt.get("hinge_dir")
            ra_gt = tgt.get("rail_axis")

            if ei.shape[1] == 0:
                continue

            def _at(t, idx):
                return int(t[idx].item()) if (t is not None and idx < t.shape[0]) else -1

            for e_idx in range(ei.shape[1]):
                gs, gd = int(ei[0, e_idx].item()), int(ei[1, e_idx].item())
                ps = int(gt2slot[gs].item()) if gs < n_gt else -1
                pd = int(gt2slot[gd].item()) if gd < n_gt else -1
                if ps < 0 or pd < 0:
                    continue

                etype = int(ea[e_idx].item())

                if etype == 1:  # hinge
                    fs_l = _at(fs_gt, e_idx)
                    fd_l = _at(fd_gt, e_idx)
                    ha_l = _at(ha_gt, e_idx)
                    hd_l = _at(hd_gt, e_idx)
                    if fs_l >= 0:
                        loss_fs = loss_fs + F.cross_entropy(
                            pred["hinge_face_src_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([fs_l], device=device))
                        n_fs += 1
                    if fd_l >= 0:
                        loss_fd = loss_fd + F.cross_entropy(
                            pred["hinge_face_dst_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([fd_l], device=device))
                        n_fd += 1
                    if ha_l >= 0:
                        loss_ha = loss_ha + F.cross_entropy(
                            pred["hinge_axis_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([ha_l], device=device))
                        n_ha += 1
                    if hd_l >= 0:
                        loss_hd = loss_hd + F.cross_entropy(
                            pred["hinge_dir_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([hd_l], device=device))
                        n_hd += 1

                elif etype == 2:  # rail
                    fs_l = _at(fs_gt, e_idx)
                    fd_l = _at(fd_gt, e_idx)
                    ra_l = _at(ra_gt, e_idx)
                    if fs_l >= 0:
                        loss_fs = loss_fs + F.cross_entropy(
                            pred["hinge_face_src_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([fs_l], device=device))
                        n_fs += 1
                    if fd_l >= 0:
                        loss_fd = loss_fd + F.cross_entropy(
                            pred["hinge_face_dst_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([fd_l], device=device))
                        n_fd += 1
                    if ra_l >= 0:
                        loss_ra = loss_ra + F.cross_entropy(
                            pred["rail_axis_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([ra_l], device=device))
                        n_ra += 1

        return {
            "loss_hinge_face_src": loss_fs / max(n_fs, 1),
            "loss_hinge_face_dst": loss_fd / max(n_fd, 1),
            "loss_hinge_axis":     loss_ha / max(n_ha, 1),
            "loss_hinge_dir":      loss_hd / max(n_hd, 1),
            "loss_rail_axis":      loss_ra / max(n_ra, 1),
        }


class MotionLossV3(nn.Module):
    """v3 motion loss — 3 per-dynamic-part heads, local to the door/drawer.

    Heads:
      hinge_border_logits        (B, M, M, 6)  — which face of the door-local OBB is the hinge edge
      hinge_axis_signed_logits   (B, M, M, 6)  — signed rotation axis (±X/±Y/±Z)
      rail_axis_signed_logits    (B, M, M, 6)  — signed slide axis for drawers

    Supervision runs only on edges whose GT label ≥ 0.  Which endpoint is
    dynamic is stored in `targets[b]['dyn_is_src']`: 1 → src, 0 → dst, −1 → skip.
    Since labels are tied to the dynamic node, the (i, j) and (j, i) directions
    of the same edge both carry the same label, so we don't need to swap.
    """

    def forward(self, pred, targets, indices):
        device = pred["material_logits"].device
        B, M, _, _ = pred["edge_logits"].shape
        loss_hb = torch.tensor(0.0, device=device)
        loss_ha = torch.tensor(0.0, device=device)
        loss_ra = torch.tensor(0.0, device=device)
        n_hb = n_ha = n_ra = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            tgt = targets[b]
            n_gt = int(tgt["material"].shape[0])
            if n_gt == 0 or pred_idx.numel() == 0:
                continue
            pred_idx_d = pred_idx.to(device); gt_idx_d = gt_idx.to(device)
            gt2slot = torch.full((n_gt,), -1, dtype=torch.long, device=device)
            gt2slot[gt_idx_d] = pred_idx_d

            ei = tgt["edge_index"].to(device)
            ea = tgt["edge_attr"].to(device)
            hb_gt = tgt.get("hinge_border")
            ha_gt = tgt.get("hinge_axis_signed")
            ra_gt = tgt.get("rail_axis_signed")

            if ei.shape[1] == 0:
                continue

            def _at(t, idx):
                return int(t[idx].item()) if (t is not None and idx < t.shape[0]) else -1

            for e_idx in range(ei.shape[1]):
                gs, gd = int(ei[0, e_idx].item()), int(ei[1, e_idx].item())
                ps = int(gt2slot[gs].item()) if gs < n_gt else -1
                pd = int(gt2slot[gd].item()) if gd < n_gt else -1
                if ps < 0 or pd < 0:
                    continue

                etype = int(ea[e_idx].item())

                if etype == 1:  # hinge
                    hb_l = _at(hb_gt, e_idx)
                    ha_l = _at(ha_gt, e_idx)
                    if hb_l >= 0:
                        loss_hb = loss_hb + F.cross_entropy(
                            pred["hinge_border_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([hb_l], device=device))
                        n_hb += 1
                    if ha_l >= 0:
                        loss_ha = loss_ha + F.cross_entropy(
                            pred["hinge_axis_signed_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([ha_l], device=device))
                        n_ha += 1
                elif etype == 2:  # rail
                    ra_l = _at(ra_gt, e_idx)
                    if ra_l >= 0:
                        loss_ra = loss_ra + F.cross_entropy(
                            pred["rail_axis_signed_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([ra_l], device=device))
                        n_ra += 1

        return {
            "loss_hinge_border":      loss_hb / max(n_hb, 1),
            "loss_hinge_axis_signed": loss_ha / max(n_ha, 1),
            "loss_rail_axis_signed":  loss_ra / max(n_ra, 1),
        }


class MotionLossV3PoseInv(nn.Module):
    """Pose-invariant motion loss for the v3_pose_inv head set.

    Heads supervised:
      hinge_border_4_logits      (B, M, M, 4)  — one of 4 panel-face edges
                                                   in the door's thin-aware frame
      hinge_axis_sign_logits     (B, M, M, 2)  — sign of the rotation axis
      rail_axis_signed_logits    (B, M, M, 6)  — kept 6-way for drawer slides

    Targets per-edge (set by graph_dataset when motion_format=v3_pose_inv):
      hinge_border_4   ∈ {-1, 0, 1, 2, 3}
      hinge_axis_sign  ∈ {-1, 0, 1}
      rail_axis_signed ∈ {-1, 0..5}     (unchanged from v3)

    Edges with -1 GT are skipped, identical to MotionLossV3 semantics.
    """

    def forward(self, pred, targets, indices):
        device = pred["material_logits"].device
        B, M, _, _ = pred["edge_logits"].shape
        loss_b4 = torch.tensor(0.0, device=device)
        loss_as = torch.tensor(0.0, device=device)
        loss_ra = torch.tensor(0.0, device=device)
        n_b4 = n_as = n_ra = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            tgt = targets[b]
            n_gt = int(tgt["material"].shape[0])
            if n_gt == 0 or pred_idx.numel() == 0:
                continue
            pred_idx_d = pred_idx.to(device); gt_idx_d = gt_idx.to(device)
            gt2slot = torch.full((n_gt,), -1, dtype=torch.long, device=device)
            gt2slot[gt_idx_d] = pred_idx_d

            ei = tgt["edge_index"].to(device)
            ea = tgt["edge_attr"].to(device)
            b4_gt = tgt.get("hinge_border_4")
            as_gt = tgt.get("hinge_axis_sign")
            ra_gt = tgt.get("rail_axis_signed")
            if ei.shape[1] == 0:
                continue

            def _at(t, idx):
                return int(t[idx].item()) if (t is not None and idx < t.shape[0]) else -1

            for e_idx in range(ei.shape[1]):
                gs, gd = int(ei[0, e_idx].item()), int(ei[1, e_idx].item())
                ps = int(gt2slot[gs].item()) if gs < n_gt else -1
                pd = int(gt2slot[gd].item()) if gd < n_gt else -1
                if ps < 0 or pd < 0:
                    continue
                etype = int(ea[e_idx].item())
                if etype == 1:        # hinge
                    b4_l = _at(b4_gt, e_idx)
                    as_l = _at(as_gt, e_idx)
                    if b4_l >= 0:
                        loss_b4 = loss_b4 + F.cross_entropy(
                            pred["hinge_border_4_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([b4_l], device=device))
                        n_b4 += 1
                    if as_l >= 0:
                        loss_as = loss_as + F.cross_entropy(
                            pred["hinge_axis_sign_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([as_l], device=device))
                        n_as += 1
                elif etype == 2:      # rail
                    ra_l = _at(ra_gt, e_idx)
                    if ra_l >= 0:
                        loss_ra = loss_ra + F.cross_entropy(
                            pred["rail_axis_signed_logits"][b, ps, pd].unsqueeze(0),
                            torch.tensor([ra_l], device=device))
                        n_ra += 1

        return {
            "loss_hinge_border_4":   loss_b4 / max(n_b4, 1),
            "loss_hinge_axis_sign":  loss_as / max(n_as, 1),
            "loss_rail_axis_signed": loss_ra / max(n_ra, 1),
        }


# ── Handle-border (per-door) loss with hinge-opposition auxiliary ───────────

_OPPOSITE_FACE = torch.tensor([1, 0, 3, 2, 5, 4], dtype=torch.long)


class HandleBorderLoss(nn.Module):
    """Two per-matched-door CE losses on `handle_border_logits`:

      * L_handle  : direct supervision against the GT handle_border.
      * L_opp     : hinge-opposition consistency — the handle prediction
                    should disagree 180° from the GT hinge_border. Only
                    active when both labels are present on the same door.

    Both terms share the same head (no architecture duplication), with L_opp
    weighted lower so the direct signal dominates and the consistency acts
    as a soft prior.
    """

    def __init__(self, w_opp: float = 0.5):
        super().__init__()
        self.w_opp = w_opp

    def forward(self, pred, targets, indices):
        logits = pred.get("handle_border_logits")
        if logits is None:
            device = pred["material_logits"].device
            return {"loss_handle": torch.tensor(0.0, device=device),
                    "loss_opp":    torch.tensor(0.0, device=device)}
        device = logits.device
        B, M, _ = logits.shape

        total_h = torch.tensor(0.0, device=device)
        total_o = torch.tensor(0.0, device=device)
        n_h = n_o = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            tgt = targets[b]
            n_gt = int(tgt["material"].shape[0])
            if n_gt == 0 or pred_idx.numel() == 0:
                continue
            pred_idx_d = pred_idx.to(device); gt_idx_d = gt_idx.to(device)

            # Per-GT-node handle_border label
            hb_gt_all = tgt.get("handle_border")
            if hb_gt_all is None:
                continue
            hb_gt_all = hb_gt_all.to(device)

            # Optional: also pull hinge_border per-edge for the opposition aux.
            # We resolve it per-door node: find an incident hinge edge with a
            # valid hinge_border label.
            ei = tgt["edge_index"].to(device)
            ea = tgt["edge_attr"].to(device)
            hi_edge_gt = tgt.get("hinge_border")
            hi_edge_gt = hi_edge_gt.to(device) if hi_edge_gt is not None else None

            # Build a GT-door-node → hinge_border map (from any incident edge)
            door_to_hinge: Dict[int, int] = {}
            if hi_edge_gt is not None and ei.shape[1] > 0:
                for e_idx in range(ei.shape[1]):
                    v = int(hi_edge_gt[e_idx].item())
                    if v < 0:
                        continue
                    gs, gd = int(ei[0, e_idx].item()), int(ei[1, e_idx].item())
                    # Doors have material code 5
                    if gs < n_gt and int(tgt["material"][gs].item()) == 5:
                        door_to_hinge.setdefault(gs, v)
                    if gd < n_gt and int(tgt["material"][gd].item()) == 5:
                        door_to_hinge.setdefault(gd, v)

            # Iterate over matched GT → slot pairs
            for k in range(pred_idx_d.numel()):
                ps = int(pred_idx_d[k].item())
                gs = int(gt_idx_d[k].item())
                # Only doors carry handle_border; others have -1
                if int(tgt["material"][gs].item()) != 5:
                    continue
                hb_gt = int(hb_gt_all[gs].item())
                if hb_gt >= 0:
                    total_h = total_h + F.cross_entropy(
                        logits[b, ps].unsqueeze(0),
                        torch.tensor([hb_gt], device=device))
                    n_h += 1
                # Opposition aux: target = OPPOSITE[hinge_border]
                hi = door_to_hinge.get(gs, -1)
                if hi >= 0:
                    opp_target = int(_OPPOSITE_FACE[hi].item())
                    total_o = total_o + F.cross_entropy(
                        logits[b, ps].unsqueeze(0),
                        torch.tensor([opp_target], device=device))
                    n_o += 1

        return {
            "loss_handle": total_h / max(n_h, 1),
            "loss_opp":    total_o / max(n_o, 1),
        }


# ── Combined loss ─────────────────────────────────────────────────────────────

class GraphTranslationLoss(nn.Module):
    """Combines node set loss and edge loss.

    Parameters
    ----------
    w_node : float — overall weight on node losses.
    w_edge : float — overall weight on edge loss.
    cost_class, cost_pos : Hungarian matcher weights.
    null_weight : edge null class down-weight.
    """

    def __init__(self,
                 w_node:      float = 1.0,
                 w_edge:      float = 1.0,
                 w_parent:    float = 0.5,
                 w_hinge_target: float = 0.5,
                 w_motion:    float = 0.3,
                 w_anchor:    float = 0.0,
                 w_anchor_bbox: float = 0.0,
                 w_free_exist: float = 0.0,
                 free_exist_pw_floor: float = 5.0,
                 w_count:     float = 0.0,
                 w_type:      float = 1.0,
                 w_centroid:  float = 2.0,
                 w_bbox:      float = 1.0,
                 cost_class:  float = 1.0,
                 cost_pos:    float = 2.0,
                 null_weight: float = 1.0,
                 edge_class_weights: Optional[List[float]] = None,
                 edge_label_smoothing: float = 0.0,
                 motion_format: str = "v2",
                 w_handle:       float = 0.3,
                 w_handle_opp:   float = 0.15):
        super().__init__()
        self.matcher   = HungarianMatcher(cost_class, cost_pos)
        self.node_loss = NodeSetLoss(w_type=w_type,
                                      w_centroid=w_centroid,
                                      w_bbox=w_bbox,
                                      w_anchor=w_anchor,
                                      w_anchor_bbox=w_anchor_bbox,
                                      w_free_exist=w_free_exist,
                                      free_exist_pw_floor=free_exist_pw_floor,
                                      w_count=w_count)
        self.edge_loss = EdgeLoss(null_weight=null_weight,
                                  class_weights=edge_class_weights,
                                  label_smoothing=edge_label_smoothing)
        self.parent_loss = ParentLoss()
        self.hinge_target_loss = HingeTargetLoss()
        self.motion_format = motion_format
        if motion_format == "v2":
            self.motion_loss = MotionLoss()
        elif motion_format == "v3_pose_inv":
            self.motion_loss = MotionLossV3PoseInv()
        else:
            self.motion_loss = MotionLossV3()
        self.handle_loss    = HandleBorderLoss()
        self.w_node    = w_node
        self.w_edge    = w_edge
        self.w_parent  = w_parent
        self.w_hinge_target = w_hinge_target
        self.w_motion  = w_motion
        self.w_anchor  = w_anchor
        self.w_handle     = w_handle
        self.w_handle_opp = w_handle_opp

    def forward(self,
                pred:    Dict[str, Tensor],
                targets: List[Dict]) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        pred    : decoder output dict.
        targets : list of B target dicts with keys:
                  'material'   (n_i,)   long  — material category index
                  'centroid'   (n_i, 3) float
                  'bbox'       (n_i, 6) float
                  'edge_index' (2, E_i) long
                  'edge_attr'  (E_i,)   long

        Returns
        -------
        losses : dict with all individual losses and 'loss_total'.
        """
        gt_materials = [t["material"] for t in targets]
        gt_centroids = [t["centroid"] for t in targets]

        indices = self.matcher(
            pred["material_logits"],
            pred["centroid_pred"],
            gt_materials,
            gt_centroids,
            pinned=[t.get("pinned_slot_gt") for t in targets],
        )

        node_losses   = self.node_loss(pred, targets, indices)
        edge_losses   = self.edge_loss(pred["edge_logits"], targets, indices)
        parent_losses = (self.parent_loss(pred, targets, indices)
                         if "parent_scores" in pred else {"loss_parent": torch.tensor(0.0, device=pred["material_logits"].device)})
        hinge_t_losses = (self.hinge_target_loss(pred, targets, indices)
                          if "hinge_target_scores" in pred else {"loss_hinge_target": torch.tensor(0.0, device=pred["material_logits"].device)})
        device = pred["material_logits"].device
        zero = lambda: torch.tensor(0.0, device=device)

        if self.motion_format == "v2":
            if "hinge_face_src_logits" in pred:
                motion_losses = self.motion_loss(pred, targets, indices)
            else:
                motion_losses = {
                    "loss_hinge_face_src": zero(), "loss_hinge_face_dst": zero(),
                    "loss_hinge_axis":     zero(), "loss_hinge_dir":      zero(),
                    "loss_rail_axis":      zero(),
                }
        elif self.motion_format == "v3_pose_inv":
            if "hinge_border_4_logits" in pred:
                motion_losses = self.motion_loss(pred, targets, indices)
            else:
                motion_losses = {
                    "loss_hinge_border_4":   zero(),
                    "loss_hinge_axis_sign":  zero(),
                    "loss_rail_axis_signed": zero(),
                }
        else:  # v3
            if "hinge_border_logits" in pred:
                motion_losses = self.motion_loss(pred, targets, indices)
            else:
                motion_losses = {
                    "loss_hinge_border":      zero(),
                    "loss_hinge_axis_signed": zero(),
                    "loss_rail_axis_signed":  zero(),
                }

        # Handle-border loss (v3 only — head isn't built in v2)
        # v3_pose_inv has its own handle_border_4 head; with w_handle=0
        # in the verified experiments we just zero this out.
        if "handle_border_logits" in pred:
            handle_losses = self.handle_loss(pred, targets, indices)
        else:
            handle_losses = {"loss_handle": zero(), "loss_opp": zero()}

        losses = {**node_losses, **edge_losses, **parent_losses, **hinge_t_losses,
                  **motion_losses, **handle_losses}
        # Isolate NodeSetLoss terms (material/centroid/bbox/exist/anchor) from edge/parent/hinge-target/motion
        _zero = torch.tensor(0.0, device=losses["loss_material"].device)
        node_total = (losses["loss_material"] + losses["loss_centroid"]
                      + losses["loss_bbox"] + losses["loss_exist"]
                      + losses.get("loss_anchor", _zero)
                      + losses.get("loss_anchor_bbox", _zero)
                      + losses.get("loss_free_exist", _zero)
                      + losses.get("loss_count", _zero))
        if self.motion_format == "v2":
            motion_total = (losses["loss_hinge_face_src"] + losses["loss_hinge_face_dst"]
                            + losses["loss_hinge_axis"] + losses["loss_hinge_dir"]
                            + losses["loss_rail_axis"])
        elif self.motion_format == "v3_pose_inv":
            motion_total = (losses["loss_hinge_border_4"]
                            + losses["loss_hinge_axis_sign"]
                            + losses["loss_rail_axis_signed"])
        else:
            motion_total = (losses["loss_hinge_border"]
                            + losses["loss_hinge_axis_signed"]
                            + losses["loss_rail_axis_signed"])
        total = (self.w_node * node_total
                 + self.w_edge * losses["loss_edge"]
                 + self.w_parent * losses["loss_parent"]
                 + self.w_hinge_target * losses["loss_hinge_target"]
                 + self.w_motion * motion_total
                 + self.w_handle     * losses["loss_handle"]
                 + self.w_handle_opp * losses["loss_opp"])
        losses["loss_total"] = total
        return losses
