"""Shared inference helpers used by infer.py and train.py.

Provides checkpoint loading (load_model), single-graph batching
(pack_single), decoder-output serialization (pred_to_json), and prediction
summaries (summarize_graph). Run inference through infer.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.graph_dataset import (load_graph_json, collate_graph_pairs,
                                 NULL_EDGE_CLASS, EDGE_TYPE_VOCAB, IDX2MATERIAL)
from models.backbone import GraphTranslationModel
from train import CONFIG, move_batch

IDX2EDGE = {v: k for k, v in EDGE_TYPE_VOCAB.items()}
IDX2EDGE[NULL_EDGE_CLASS] = "null"


def pred_to_json(pred_out: Dict,
                 b: int,
                 model_id: str,
                 exist_threshold: float,
                 max_new_nodes: Optional[int] = None,
                 dyn_type_mutex: bool = False) -> Dict:
    """Serialize the model's predicted set for graph *b*.

    Pure DETR-style output: each slot surviving the existence threshold
    (sigmoid(exist_logits) > threshold) becomes a node named
    ``slot{i:02d}_{predicted_material}``. Material, centroid, and half-size
    come directly from the model heads. No anchor override, no input-name
    injection — this is exactly what the model output, visible as-is.

    `max_new_nodes` (cascaded inference): if set, keep at most this many
    NEW (= non-anchored) free slots, picking the highest-confidence ones.
    Anchored slots are always kept regardless. Set to 1 to enforce the
    "predict-one-new-node-per-call" iteration loop.

    `dyn_type_mutex`: if True, enforce a cross-type mutual exclusion on each
    dynamic slot — a slot whose argmax material is door OR drawer is forced
    to be exactly one of the two based on whichever edge type (hinge vs rail)
    has the higher max-confidence connection to it. The losing-type edges
    incident to that slot are suppressed and the slot's material is rewritten
    to match (door↔hinge, drawer↔rail).
    """
    exist_prob = pred_out["exist_logits"][b, :, 0].sigmoid().cpu()
    real_mask  = exist_prob > exist_threshold
    # Anchored slots mirror input nodes and always exist — same rule as the
    # model's own decode (backbone: real_mask = thresholded | anchor) and a
    # precondition for postprocess's positional anchored/new split.
    if "anchor_mask" in pred_out:
        real_mask = real_mask | pred_out["anchor_mask"][b].cpu()
    real_idx   = real_mask.nonzero(as_tuple=True)[0].tolist()

    # Cascaded-inference cap on free-slot firings.
    if max_new_nodes is not None and "anchor_mask" in pred_out:
        anchor_mask_b = pred_out["anchor_mask"][b].cpu().tolist()
        free = [i for i in real_idx if not anchor_mask_b[i]]
        anch = [i for i in real_idx if anchor_mask_b[i]]
        if len(free) > max_new_nodes:
            free.sort(key=lambda i: -float(exist_prob[i]))
            free = free[:max_new_nodes]
        real_idx = sorted(anch + free)

    # ── Edge constants + caches (used by both the mutex pass and dedup) ────
    HINGE_EDGE_CLASS = 1
    RAIL_EDGE_CLASS  = 2
    DOOR_MAT_IDX     = 5
    DRAWER_MAT_IDX   = 6
    edge_argmax = pred_out["edge_logits"][b].argmax(-1).cpu()
    hinge_logits_pair = pred_out["edge_logits"][b, :, :, HINGE_EDGE_CLASS].cpu()
    rail_logits_pair  = pred_out["edge_logits"][b, :, :, RAIL_EDGE_CLASS].cpu()
    pred_mats_all = pred_out["material_logits"][b].argmax(-1).cpu().tolist()
    suppressed_pairs: set = set()   # set of (min_slot, max_slot, edge_class)

    # ── Cross-type dynamic mutex ───────────────────────────────────────────
    # For every slot whose argmax material is door OR drawer, compare the
    # max hinge-edge logit vs. max rail-edge logit across all of its
    # incident edges. The higher one wins: the slot is rewritten to that
    # dynamic class (door↔hinge, drawer↔rail) and every losing-type edge
    # incident to it is added to suppressed_pairs (nullified at emission).
    # This must run BEFORE _dedup_for_dynamic so the same-type dedup sees
    # the updated material.
    if dyn_type_mutex:
        for d in real_idx:
            if pred_mats_all[d] not in (DOOR_MAT_IDX, DRAWER_MAT_IDX):
                continue
            best_hinge = best_rail = -float("inf")
            for j in real_idx:
                if j == d: continue
                i_, j_ = (min(d, j), max(d, j))
                h = 0.5 * (hinge_logits_pair[i_, j_].item()
                           + hinge_logits_pair[j_, i_].item())
                r = 0.5 * (rail_logits_pair[i_, j_].item()
                           + rail_logits_pair[j_, i_].item())
                if h > best_hinge: best_hinge = h
                if r > best_rail: best_rail = r
            if best_hinge >= best_rail:
                target_mat, lose_class = DOOR_MAT_IDX, RAIL_EDGE_CLASS
            else:
                target_mat, lose_class = DRAWER_MAT_IDX, HINGE_EDGE_CLASS
            pred_mats_all[d] = target_mat
            for j in real_idx:
                if j == d: continue
                i_, j_ = (min(d, j), max(d, j))
                if int(edge_argmax[i_, j_].item()) == lose_class:
                    suppressed_pairs.add((i_, j_, lose_class))

    nodes = {}
    idx2name = {}
    for slot_i in real_idx:
        mat_idx  = pred_mats_all[slot_i]   # may have been overridden by mutex
        mat_name = IDX2MATERIAL.get(mat_idx, "unknown")
        centroid = pred_out["centroid_pred"][b, slot_i].cpu().tolist()
        if "half_size" in pred_out:
            half = pred_out["half_size"][b, slot_i].cpu().tolist()
        else:
            bbox = pred_out["bbox_pred"][b, slot_i].cpu().tolist()
            half = [(bbox[3]-bbox[0])/2, (bbox[4]-bbox[1])/2, (bbox[5]-bbox[2])/2]
        half = [max(float(h), 0.005) for h in half]

        node_name = f"slot{slot_i:02d}_{mat_name.replace(' ', '_')}"
        idx2name[slot_i] = node_name
        nodes[node_name] = {
            "id":       node_name,
            "slot":     slot_i,
            "type":     "dynamic" if mat_name in ("door", "drawer", "handle") else "static",
            "material": mat_name,
            "obb":      {"center": centroid, "half": half, "quat": [1, 0, 0, 0]},
        }

    # ── Per-dynamic-part edge dedup ────────────────────────────────────────
    # Each door slot may have multiple hinge edges and each drawer may have
    # multiple rail edges via the per-pair edge head — but real cabinets
    # have ONE hinge-mount panel per door and ONE rail per drawer. We keep
    # only the highest-confidence partner edge (mean of both-direction
    # logits) per dynamic part and suppress the rest by overriding their
    # edge class to NULL further down.

    def _dedup_for_dynamic(dyn_mat_code: int, edge_class: int,
                            logits_pair: torch.Tensor,
                            keep_top_k: int) -> None:
        """For every slot whose predicted material == dyn_mat_code, keep
        only the top-`keep_top_k` highest-confidence outgoing `edge_class`
        edges; mark the rest for nullification.

        Door/hinge: keep_top_k=1 (one mount panel per door).
        Drawer/rail: keep_top_k=2 (drawers often slide between two side
        panels via two parallel rails, so retain up to two highest-conf
        rail edges).
        """
        for d in real_idx:
            if pred_mats_all[d] != dyn_mat_code:
                continue
            partners = []
            for j in real_idx:
                if j == d: continue
                i_, j_ = (min(d, j), max(d, j))
                if int(edge_argmax[i_, j_].item()) == edge_class:
                    score = (logits_pair[i_, j_].item()
                             + logits_pair[j_, i_].item()) * 0.5
                    partners.append((j, score, (i_, j_)))
            if len(partners) > keep_top_k:
                partners.sort(key=lambda x: -x[1])
                for _, _, pair_key in partners[keep_top_k:]:
                    suppressed_pairs.add((*pair_key, edge_class))

    _dedup_for_dynamic(DOOR_MAT_IDX,   HINGE_EDGE_CLASS, hinge_logits_pair, keep_top_k=1)
    _dedup_for_dynamic(DRAWER_MAT_IDX, RAIL_EDGE_CLASS,  rail_logits_pair,  keep_top_k=2)

    # ── Soft-parent override for handles ───────────────────────────────────
    # For each handle slot, argmax over (door/drawer candidates ∪ no-parent)
    # via parent_scores. If a specific parent wins, that pair becomes the
    # unique ATTACHED edge; all other handle↔door/drawer attached edges are
    # forced to NULL.
    handle_to_parent: Dict[int, int] = {}
    if "parent_scores" in pred_out:
        parent_scores = pred_out["parent_scores"][b].cpu()    # (M, M)
        noparent_logit = float(pred_out["parent_noparent_logit"].item())
        # Determine each slot's predicted material (argmax)
        pred_mats = pred_out["material_logits"][b].argmax(-1).cpu().tolist()
        HANDLE_MAT, DOOR_MAT, DRAWER_MAT = 0, 5, 6
        ATTACHED = 3
        # Only consider predicted handles that survived existence threshold
        real_set = set(real_idx)
        for h in real_idx:
            if pred_mats[h] != HANDLE_MAT:
                continue
            cand = [j for j in real_idx
                    if j != h and pred_mats[j] in (DOOR_MAT, DRAWER_MAT)]
            if not cand:
                continue
            scores = parent_scores[h, cand].tolist() + [noparent_logit]
            best = int(max(range(len(scores)), key=lambda k: scores[k]))
            if best < len(cand):
                handle_to_parent[h] = cand[best]
            else:
                handle_to_parent[h] = -1    # "no parent"

    edges = []
    seen: set = set()
    for i in real_idx:
        for j in real_idx:
            if i >= j:
                continue
            et = int(edge_argmax[i, j].item())
            # Suppress duplicate hinge/rail edges per dynamic part (post-process)
            if (i, j, et) in suppressed_pairs:
                et = NULL_EDGE_CLASS
            # Apply soft-parent override for handle-(door/drawer) attached edges
            if "parent_scores" in pred_out:
                for h, p in ((i, j), (j, i)):
                    if h in handle_to_parent:
                        # h is a handle; if et would be ATTACHED but p is not its chosen parent, nullify
                        chosen = handle_to_parent[h]
                        if et == 3 and p != chosen:    # 3 == ATTACHED
                            et = NULL_EDGE_CLASS
                        # If h has a chosen parent and this pair IS the chosen one, force ATTACHED
                        elif chosen >= 0 and p == chosen:
                            et = 3
            if et == NULL_EDGE_CLASS:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            edge_out = {
                "src":  idx2name[i],
                "dst":  idx2name[j],
                "kind": IDX2EDGE.get(et, "contact"),
            }
            # Attach motion attributes for hinge / rail edges
            # v2 heads
            if et == 1 and "hinge_face_src_logits" in pred_out:   # hinge
                edge_out["hinge_face_src"] = int(pred_out["hinge_face_src_logits"][b, i, j].argmax(-1).item())
                edge_out["hinge_face_dst"] = int(pred_out["hinge_face_dst_logits"][b, i, j].argmax(-1).item())
                edge_out["hinge_axis"]     = int(pred_out["hinge_axis_logits"][b, i, j].argmax(-1).item())
                edge_out["hinge_direction"] = int(pred_out["hinge_dir_logits"][b, i, j].argmax(-1).item())
            elif et == 2 and "rail_axis_logits" in pred_out:      # rail
                edge_out["hinge_face_src"] = int(pred_out["hinge_face_src_logits"][b, i, j].argmax(-1).item())
                edge_out["hinge_face_dst"] = int(pred_out["hinge_face_dst_logits"][b, i, j].argmax(-1).item())
                edge_out["rail_axis"]      = int(pred_out["rail_axis_logits"][b, i, j].argmax(-1).item())
            # v3 heads — per-edge signed labels in the dynamic part's local frame
            if et == 1 and "hinge_border_logits" in pred_out:
                edge_out["hinge_border"]      = int(pred_out["hinge_border_logits"][b, i, j].argmax(-1).item())
                edge_out["hinge_axis_signed"] = int(pred_out["hinge_axis_signed_logits"][b, i, j].argmax(-1).item())
            elif et == 2 and "rail_axis_signed_logits" in pred_out:
                edge_out["rail_axis_signed"] = int(pred_out["rail_axis_signed_logits"][b, i, j].argmax(-1).item())
            # v3_pose_inv heads — decode 4-way border + 2-way sign back to the
            # canonical 6-way (hinge_border / hinge_axis_signed) using the
            # predicted dynamic part's thin axis. install_from_pred and the
            # blenderize pipeline both consume the 6-way fields, so we emit
            # those plus the raw 4-way fields for transparency.
            if et == 1 and "hinge_border_4_logits" in pred_out:
                b4   = int(pred_out["hinge_border_4_logits"][b, i, j].argmax(-1).item())
                asgn = int(pred_out["hinge_axis_sign_logits"][b, i, j].argmax(-1).item())
                edge_out["hinge_border_4"]  = b4
                edge_out["hinge_axis_sign"] = asgn
                # find dynamic endpoint (door/drawer)
                mat_i = pred_mats_all[i] if i < len(pred_mats_all) else -1
                mat_j = pred_mats_all[j] if j < len(pred_mats_all) else -1
                dyn = i if mat_i in (DOOR_MAT_IDX, DRAWER_MAT_IDX) else j
                # Read the predicted door OBB's half to find thin axis
                if "half_size" in pred_out:
                    h = pred_out["half_size"][b, dyn].cpu().tolist()
                else:
                    bb = pred_out["bbox_pred"][b, dyn].cpu().tolist()
                    h = [(bb[3]-bb[0])/2, (bb[4]-bb[1])/2, (bb[5]-bb[2])/2]
                thin = min(range(3), key=lambda i: abs(h[i]))
                non_thin = [a for a in (0, 1, 2) if a != thin]
                # border: 4 -> world face index
                slot       = b4 // 2                        # 0 or 1 → first / second non-thin axis
                sign_b     = -1 if (b4 % 2 == 0) else +1
                border_axis = non_thin[slot]
                world_border = border_axis * 2 + (0 if sign_b < 0 else 1)
                # axis: along the OTHER non-thin axis, with predicted sign
                axis_axis = non_thin[1 - slot]
                world_axis = axis_axis * 2 + (1 if asgn == 1 else 0)
                edge_out["hinge_border"]      = world_border
                edge_out["hinge_axis_signed"] = world_axis
            edges.append(edge_out)

    return {"model_id": model_id, "nodes": nodes, "edges": edges}



def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = {**CONFIG, **ck.get('cfg', {})}
    model = GraphTranslationModel(
        num_material_types=cfg.get('num_material_types', 17),
        num_edge_types=cfg['num_edge_types'],
        type_emb_dim=cfg['type_emb_dim'], pc_dim=cfg['pc_dim'],
        geom_dim=cfg['geom_dim'], hidden_dim=cfg['hidden_dim'],
        edge_dim=cfg.get('edge_dim', 64), enc_layers=cfg['enc_layers'],
        dec_layers=cfg['dec_layers'], num_heads=cfg['num_heads'],
        max_nodes=cfg['max_nodes'], max_free_slots=cfg.get('max_free_slots', 8),
        dropout=cfg['dropout'],
        use_motion_heads=cfg.get('use_motion_heads', True),
        motion_format=cfg.get('motion_format', 'v2'),
        use_graph_cond_slots=bool(cfg.get('use_graph_cond_slots', True)),
        use_signed_pair_features=bool(cfg.get('use_signed_pair_features', False))).to(device)
    missing, unexpected = model.load_state_dict(ck['model_state'], strict=False)
    if missing or unexpected:
        # strict=False tolerates architecture drift, but silence here would
        # let an incompatible checkpoint "load" and predict garbage.
        print(f"[load_model] state_dict mismatch: {len(missing)} missing, "
              f"{len(unexpected)} unexpected key(s)"
              + (f"; missing e.g. {missing[:2]}" if missing else "")
              + (f"; unexpected e.g. {unexpected[:2]}" if unexpected else ""))
    model.eval()
    return model, cfg


def pack_single(data, device):
    """Wrap a single PyG Data object into a batch expected by the model."""
    pcs   = data.point_clouds
    srcs  = data.pc_source
    names = data.node_names
    mid   = data.model_id
    del data.point_clouds, data.pc_source, data.node_names, data.model_id
    batched = Batch.from_data_list([data])
    batched.point_clouds  = pcs
    batched.pc_source     = srcs
    batched.node_names    = names
    batched.model_id      = [mid]
    batched.pc_node_batch = batched.batch
    return move_batch(batched, device)


def summarize_graph(pred: Dict):
    n_nodes = len(pred['nodes'])
    by_mat = {}
    nodes = (pred['nodes'].values() if isinstance(pred['nodes'], dict)
             else pred['nodes'])
    for n in nodes:
        m = n['material']; by_mat[m] = by_mat.get(m, 0) + 1
    edges_by_kind = {}
    motion_edges = []
    for e in pred['edges']:
        edges_by_kind[e['kind']] = edges_by_kind.get(e['kind'], 0) + 1
        if e['kind'] in ('hinge', 'rail'):
            motion_edges.append(e)
    return n_nodes, by_mat, edges_by_kind, motion_edges
