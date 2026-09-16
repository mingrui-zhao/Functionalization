"""Handle placement engine (additive + subtractive).

The single
implementation both the interactive addon (via blender_tools/add_handle.py)
and the headless batch installer use.

Public API:
    apply_handle_to_selected_object(...)   — additive template handle on the
                                             active object (door/drawer face)
    apply_subtractive_handle_inplace(...)  — carved pull / groove
    set_template_dir(path)                 — where handle*.blend templates live

Placement rules (see DESIGN.md §I.1):
  - doors: side OPPOSITE the hinge axis, twist align_long_to_motion
  - drawers: centered on the front face, twist align_long_to_right
  - partition_frac: 0-1 position along the free axis for multi-handle
    parents (equal partitions (i+1)/(N+1), knob templates)
  - scale: protrusion-driven target with per-style mm clamps, in-plane
    minor/major caps, and axis-wise caps (MAX_AXIS_FRAC = 0.9)
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

# Template directory: set once by the addon / batch driver; falls back to
# the repo-root annotated_mechanical_parts next to blender_tools/.
_TEMPLATE_DIR: Optional[Path] = None


def set_template_dir(path) -> None:
    global _TEMPLATE_DIR
    _TEMPLATE_DIR = Path(path) if path else None


def _template_dirs() -> list:
    dirs = []
    if _TEMPLATE_DIR is not None:
        dirs.append(_TEMPLATE_DIR)
    # addon layout: <repo>/blender_tools/functionalization_ui/utils/ → <repo>/
    dirs.append(Path(__file__).resolve().parents[3] / "annotated_mechanical_parts")
    return dirs

def _get_joint_info_for_object(obj):
    """
    Get joint info (motion_origin, motion_dir) for an object from scene data.
    
    The addon stores joint info when loading models. This function retrieves it
    from custom properties or scene data.
    
    Returns:
        tuple: (motion_origin: Vector, motion_dir: Vector) or (None, None)
    """
    from mathutils import Vector
    
    # Custom properties are the only supported joint-info side channel.
    if obj.get('motion_origin') and obj.get('motion_dir'):
        origin = Vector(obj['motion_origin'])
        direction = Vector(obj['motion_dir'])
        return origin, direction
    return None, None


def _detect_open_front_face(target_obj, front, up, right):
    """Detect which of +front / -front is the truly open (unobstructed) face.

    Casts rays from several probe points across the door plane in both
    directions, ignoring hits on the target itself. The side with greater
    clearance to other scene geometry is the true front.

    Returns the corrected front direction (same as input, or -input).
    """
    from mathutils import Vector

    verts = [target_obj.matrix_world @ v.co for v in target_obj.data.vertices]
    if not verts:
        return front

    rs = [right.dot(v) for v in verts]
    us = [up.dot(v) for v in verts]
    fs = [front.dot(v) for v in verts]
    r_min, r_max = min(rs), max(rs)
    u_min, u_max = min(us), max(us)
    f_min, f_max = min(fs), max(fs)

    span_r = r_max - r_min
    span_u = u_max - u_min
    span_f = f_max - f_min
    span = max(span_r, span_u, span_f)
    if span <= 1e-9:
        return front

    # Skip auto-detect if no other meshes in the scene
    other_meshes = [o for o in bpy.data.objects
                    if o.type == 'MESH' and o != target_obj and not o.hide_viewport]
    if not other_meshes:
        return front

    eps = max(span * 0.005, 1e-5)
    max_dist = span * 20.0

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Probe grid across the door face (3x3 interior points)
    probe_fracs = (0.2, 0.5, 0.8)
    probes = []
    for fr in probe_fracs:
        for fu in probe_fracs:
            pr = r_min + fr * span_r
            pu = u_min + fu * span_u
            probes.append((pr, pu))

    def side_clearance(ray_dir, start_f):
        """Average clearance when casting rays outward from the given face."""
        total = 0.0
        count = 0
        for (pr, pu) in probes:
            origin = right * pr + up * pu + front * start_f
            cur_origin = origin.copy()
            remaining = max_dist
            hit_dist = max_dist
            # Step past any self-intersections with target_obj
            for _ in range(8):
                hit, loc, _, _, hit_obj, _ = bpy.context.scene.ray_cast(
                    depsgraph, cur_origin, ray_dir, distance=remaining
                )
                if not hit:
                    break
                if hit_obj == target_obj:
                    step = (loc - cur_origin).length + eps
                    cur_origin = cur_origin + ray_dir * step
                    remaining -= step
                    if remaining <= 0:
                        break
                    continue
                hit_dist = (loc - origin).length
                break
            total += hit_dist
            count += 1
        return total / max(1, count)

    front_clear = side_clearance(front, f_max + eps)
    back_clear = side_clearance(-front, f_min - eps)

    print(f"[front-detect] +front avg clearance: {front_clear:.4f}")
    print(f"[front-detect] -front avg clearance: {back_clear:.4f}")

    # Need asymmetry to flip. A 30% margin avoids noise-driven flips.
    if back_clear > front_clear * 1.3:
        return -front
    return front


def _post_snap_to_surface(root, other_meshes, target_obj, front, f_max,
                          clearance=0.0005, max_shift=0.05):
    """Mesh-based contact snap along `front`, run AFTER the snap-plane
    center lands on the center-raycast hit point.

    The center ray guarantees one surface point, but the handle's
    mounting feet can span recessed panels or chamfers away from that
    point — and on a ray miss the fallback is the AABB front plane —
    leaving the feet floating off the actual mesh (or, over protruding
    ornament, sunk into it). Sample the door/drawer surface under every
    base vertex of the handle with a per-object BVH (immune to occluding
    siblings, unlike scene ray_cast) and translate the whole handle
    along `front` so the closest base vertex rests `clearance` off the
    real surface.

    Returns the applied signed shift in meters (positive = moved toward
    the door to close a gap, 0.0 = no-op)."""
    from mathutils.bvhtree import BVHTree
    base_verts = []
    for o in other_meshes:
        if o.type != "MESH" or o.data is None:
            continue
        mw = o.matrix_world
        base_verts.extend(mw @ v.co for v in o.data.vertices)
    if not base_verts:
        return 0.0
    fs = [front.dot(v) for v in base_verts]
    b_min, b_max = min(fs), max(fs)
    # Mounting-side band: the feet/stem tips, not the whole handle.
    band = b_min + max(0.002, 0.10 * max(b_max - b_min, 1e-6))
    feet = [v for v, f in zip(base_verts, fs) if f <= band]
    if len(feet) > 120:
        feet = feet[:: max(1, len(feet) // 120)]

    deps = bpy.context.evaluated_depsgraph_get()
    try:
        tree = BVHTree.FromObject(target_obj, deps)
    except Exception:
        return 0.0
    # BVHTree.FromObject lives in the object's LOCAL space (imported PNM
    # meshes carry flip matrices), so rays travel through the inverse
    # transform and hits come back through the forward one.
    M = target_obj.matrix_world
    Minv = M.inverted()
    dir_local = (Minv.to_3x3() @ (-front)).normalized()
    start_f = f_max + 0.02
    h_best = None
    for v in feet:
        origin = v + front * (start_f - front.dot(v))
        loc = tree.ray_cast(Minv @ origin, dir_local)[0]
        if loc is None:
            continue          # foot overhangs the door edge: no surface below
        h = front.dot(M @ loc)
        h_best = h if h_best is None else max(h_best, h)
    if h_best is None:
        return 0.0
    shift = (b_min - h_best) - clearance
    if abs(shift) < 1e-5:
        return 0.0
    shift = max(-max_shift, min(max_shift, shift))
    root.location -= front * shift
    bpy.context.view_layer.update()
    return shift


def apply_handle_to_selected_object(
    handle_style='BAR',
    target_type='door',
    place_right_frac=0.5,
    place_up_frac=0.5,
    front_axis='-Y',
    up_axis='+Z',
    auto_rescale=True,
    base_path=None,
    joint_info=None,
    twist_mode='align_long_to_motion',
    edge_margin_frac=0.10,
    snap_mode='raycast',
    partition_frac=None,
    n_handles=1
):
    """
    Apply a handle to the currently selected object in Blender.
    
    Works directly with scene objects instead of importing/exporting OBJ
    files.
    
    Args:
        handle_style: Style of handle ('BAR', 'KNOB', 'CUP', 'RING', 'EDGE')
        partition_frac: Optional 0-1 position along the free placement axis
            (doors: along the handle-side edge; drawers: along the width).
            Used to distribute multiple handles on the same parent at
            equal-distance partitions.
        target_type: 'door' or 'drawer'
        place_right_frac: Horizontal position (0-1), overridden if joint_info provided
        place_up_frac: Vertical position (0-1), overridden if joint_info provided
        front_axis: Front direction (e.g., '-Y')
        up_axis: Up direction (e.g., '+Z')
        auto_rescale: Auto-scale handle to fit target
        base_path: Optional base path override
        joint_info: Optional dict with 'origin' and 'direction' for handle side detection
        twist_mode: Handle twist mode ('align_long_to_motion', 'align_long_to_up', 'keep')
        edge_margin_frac: Margin from edge for handle placement (0-0.5)
        snap_mode: 'raycast' or 'bbox' for handle placement
        
    Returns:
        bool: True if successful
    """
    # Get selected object
    target_obj = bpy.context.active_object
    if not target_obj or target_obj.type != 'MESH':
        print("Error: No mesh object selected")
        return False

    print(f"Applying handle to: {target_obj.name}")

    # Get handle blend file path
    # Support both:
    # - Direct file names: 'handle2', 'handle3', etc.
    # - Style names: 'BAR', 'KNOB', etc.
    fallback_dirs = list(_template_dirs())
    if base_path:
        fallback_dirs.append(Path(base_path) / "annotated_mechanical_parts")
    
    # Map abstract style names to representative filenames after the
    # handle{N}_{style}.blend rename. RING/EDGE have no template in the
    # current asset set; fall back to BAR.
    style_to_file = {
        'BAR':  'handle3_bar.blend',
        'KNOB': 'handle5_knob.blend',
        'CUP':  'handle19_cup.blend',
        'RING': 'handle3_bar.blend',  # alias — no RING template available
        'EDGE': 'handle3_bar.blend',  # alias — no EDGE template available
    }

    # Per-style protrusion (= depth along snap-plane normal) target as a
    # ratio of door thickness, plus absolute mm clamps. Real cabinet
    # hardware: bar pulls ~30mm protrusion on ~18mm doors (~1.7×),
    # knobs ~25mm, cup pulls ~15mm.
    STYLE_RATIO   = {'BAR': 2.0, 'RING': 1.8, 'KNOB': 1.2, 'CUP': 0.8, 'EDGE': 0.0}
    STYLE_MIN_MM  = {'BAR':  18, 'RING':  20, 'KNOB':  18, 'CUP': 12, 'EDGE': 0}
    STYLE_MAX_MM  = {'BAR':  50, 'RING':  45, 'KNOB':  35, 'CUP': 28, 'EDGE': 0}

    # Determine filename. New convention: every template lives at
    # `handle{N}_{style}.blend`. We accept three input forms:
    #   1. 'handle3_bar' / 'handle19_cup' (the dropdown values)
    #   2. 'handle3' / 'handle19' (older naming variant) — search by glob
    #   3. 'BAR' / 'KNOB' / 'CUP' / 'RING' / 'EDGE' (abstract styles)
    if handle_style.lower().startswith('handle'):
        # Form 1: already has the style suffix
        if "_" in handle_style:
            filename = f"{handle_style}.blend"
        else:
            # Form 2: older bare 'handleN' - resolve by searching directories
            filename = None
            for fallback_dir in fallback_dirs:
                if not fallback_dir.exists():
                    continue
                matches = list(fallback_dir.glob(f"{handle_style}_*.blend"))
                if matches:
                    filename = matches[0].name
                    break
            if filename is None:
                # Last-resort: un-suffixed name variant
                filename = f"{handle_style}.blend"
    else:
        filename = style_to_file.get(handle_style.upper(), 'handle3_bar.blend')
    
    # Find the handle file
    handle_blend = None
    for fallback_dir in fallback_dirs:
        candidate = fallback_dir / filename
        if candidate.exists():
            handle_blend = candidate
            break
    
    if not handle_blend or not handle_blend.exists():
        print(f"Handle blend file not found for style {handle_style}")
        print(f"  Tried filename: {filename}")
        print(f"  Searched dirs: {fallback_dirs}")
        return False

    print(f"Using handle: {handle_blend}")

    # Resolve abstract style key (BAR/KNOB/CUP/RING/EDGE) for the
    # protrusion-driven scaling step further down. Filename suffix takes
    # precedence; abstract style name from input is the fallback.
    _style_key = None
    _stem_parts = handle_blend.stem.rsplit("_", 1)
    if len(_stem_parts) == 2 and _stem_parts[1].upper() in STYLE_RATIO:
        _style_key = _stem_parts[1].upper()
    elif handle_style.upper() in STYLE_RATIO:
        _style_key = handle_style.upper()
    else:
        _style_key = 'BAR'   # safe default
    print(f"[scale] style_key resolved to: {_style_key}")
    
    # Get joint info for handle side detection
    motion_origin = None
    motion_dir = None
    
    if joint_info:
        # Joint info provided directly from operator
        print(f"\n=== Joint Info Received ===")
        print(f"  joint_info keys: {list(joint_info.keys())}")
        motion_origin = joint_info.get('origin')
        motion_dir = joint_info.get('direction')
        print(f"  Raw origin: {motion_origin} (type: {type(motion_origin).__name__})")
        print(f"  Raw direction: {motion_dir} (type: {type(motion_dir).__name__})")
        
        if motion_origin is not None:
            motion_origin = Vector(motion_origin) if not isinstance(motion_origin, Vector) else motion_origin
        if motion_dir is not None:
            motion_dir = Vector(motion_dir) if not isinstance(motion_dir, Vector) else motion_dir
        print(f"  Converted origin: {motion_origin}")
        print(f"  Converted direction: {motion_dir}")
    else:
        print(f"\n=== No Joint Info Provided ===")
        print(f"  Handle will be placed at center (no hinge side detection)")
        # Try to get joint info from scene/object as fallback
        motion_origin, motion_dir = _get_joint_info_for_object(target_obj)
        if motion_origin and motion_dir:
            print(f"  Found joint info via fallback: origin={motion_origin}, dir={motion_dir}")
    
    try:
        # Parse axis directions (same as handle_utils.compute_object_basis)
        def parse_axis(s):
            s = s.strip().upper()
            sign = -1.0 if s.startswith('-') else 1.0
            axis = s[-1]
            vec = Vector((0, 0, 0))
            if axis == 'X':
                vec.x = sign
            elif axis == 'Y':
                vec.y = sign
            elif axis == 'Z':
                vec.z = sign
            return vec.normalized()

        front = parse_axis(front_axis)
        up = parse_axis(up_axis)
        right = up.cross(front).normalized()

        # Re-orthonormalize
        front = right.cross(up).normalized()

        # Auto-detect the actually-open front face by raycasting both sides
        # into scene geometry. If the user's front axis points into blocking
        # geometry (e.g. cabinet body behind the door), flip it.
        detected_front = _detect_open_front_face(target_obj, front, up, right)
        if detected_front.dot(front) < 0.0:
            print(f"[front-detect] Flipping front axis: user-given side is blocked by other geometry")
            front = detected_front
            right = up.cross(front).normalized()
            front = right.cross(up).normalized()

        # Get target object world vertices for handle side detection
        verts = [target_obj.matrix_world @ v.co for v in target_obj.data.vertices]
        if not verts:
            print("Error: No vertices in target object")
            return False
        
        # Calculate bounds in Right/Up frame
        rs = [right.dot(v) for v in verts]
        us = [up.dot(v) for v in verts]
        fs = [front.dot(v) for v in verts]
        
        r_min, r_max = min(rs), max(rs)
        u_min, u_max = min(us), max(us)
        f_min, f_max = min(fs), max(fs)
        
        target_width = r_max - r_min
        target_height = u_max - u_min
        target_depth = f_max - f_min
        
        print(f"Target bounds: W={target_width:.3f}, H={target_height:.3f}, D={target_depth:.3f}")

        # Which in-plane axis multi-handle partitions run along ('W'|'H').
        # Used to shrink the handle so n_handles fit side by side.
        partition_axis = None

        # Drawers place at (right_frac, up_frac) directly — partition along width
        if target_type != 'door':
            partition_axis = 'W'
            if partition_frac is not None:
                place_right_frac = max(0.05, min(0.95, float(partition_frac)))
                print(f"[drawer] partition_frac -> place_right_frac={place_right_frac:.3f}")

        # Detect handle side from motion axis
        handle_side = None
        hinge_side = None
        
        if motion_origin is not None and motion_dir is not None and target_type == 'door':
            # Use the same algorithm as detect_handle_side_from_motion_axis
            boundary_eps_frac = 0.02
            dist_quantile = 0.1
            
            r_span = r_max - r_min
            u_span = u_max - u_min
            eps_r = max(1e-9, boundary_eps_frac * r_span)
            eps_u = max(1e-9, boundary_eps_frac * u_span)
            
            # Motion line in RU coordinates
            r0 = right.dot(motion_origin)
            u0 = up.dot(motion_origin)
            dr = right.dot(motion_dir.normalized())
            du = up.dot(motion_dir.normalized())
            
            # Build boundary vertex sets
            left_idx = [i for i, r in enumerate(rs) if r <= r_min + eps_r]
            right_idx = [i for i, r in enumerate(rs) if r >= r_max - eps_r]
            bottom_idx = [i for i, u in enumerate(us) if u <= u_min + eps_u]
            top_idx = [i for i, u in enumerate(us) if u >= u_max - eps_u]
            
            # Relax if needed
            if not left_idx or not right_idx or not bottom_idx or not top_idx:
                eps_r = max(eps_r, 0.05 * r_span)
                eps_u = max(eps_u, 0.05 * u_span)
                left_idx = [i for i, r in enumerate(rs) if r <= r_min + eps_r]
                right_idx = [i for i, r in enumerate(rs) if r >= r_max - eps_r]
                bottom_idx = [i for i, u in enumerate(us) if u <= u_min + eps_u]
                top_idx = [i for i, u in enumerate(us) if u >= u_max - eps_u]
            
            side_to_idx = {
                "left": left_idx, "right": right_idx,
                "bottom": bottom_idx, "top": top_idx,
            }
            
            # Distance from point to line in 2D
            def point_line_dist_2d(a0, a1, p0, p1, d0, d1):
                denom = math.hypot(d0, d1)
                if denom < 1e-12:
                    return math.hypot(a0 - p0, a1 - p1)
                return abs(d1 * (a0 - p0) - d0 * (a1 - p1)) / denom
            
            # Compute scores for each side (lower = closer to hinge axis)
            scores = {}
            for side, idxs in side_to_idx.items():
                if not idxs:
                    scores[side] = float('inf')
                    continue
                dists = [point_line_dist_2d(rs[i], us[i], r0, u0, dr, du) for i in idxs]
                dists.sort()
                q_idx = int(round(dist_quantile * (len(dists) - 1)))
                scores[side] = dists[q_idx]
            
            # Hinge side is closest to motion axis, handle is opposite.
            # Restrict candidates by axis orientation first: a vertical
            # hinge axis can only pin on the left/right edges (the top and
            # bottom edges share a corner vertex with the axis line, so
            # their min-distances tie with the true side to within float
            # noise and can win arbitrarily).
            axis_du = abs(du); axis_dr = abs(dr)
            if axis_du > 1.5 * axis_dr:
                cand_sides = ["left", "right"]
            elif axis_dr > 1.5 * axis_du:
                cand_sides = ["top", "bottom"]
            else:
                cand_sides = list(scores.keys())
            hinge_side = min(cand_sides, key=lambda k: scores[k])
            opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
            handle_side = opposite[hinge_side]
            
            print(f"[door] hinge_side={hinge_side} | handle_side={handle_side}")
            print(f"[door] side_scores={scores}")
            
            # Calculate placement fractions (same as auto_place_fracs_from_handle_side)
            m = max(0.0, min(0.49, edge_margin_frac))
            c = (0.5 if partition_frac is None
                 else max(0.05, min(0.95, float(partition_frac))))
            
            if handle_side == "left":
                place_right_frac, place_up_frac = m, c
                partition_axis = 'H'
            elif handle_side == "right":
                place_right_frac, place_up_frac = 1.0 - m, c
                partition_axis = 'H'
            elif handle_side == "bottom":
                place_right_frac, place_up_frac = c, m
                partition_axis = 'W'
            elif handle_side == "top":
                place_right_frac, place_up_frac = c, 1.0 - m
                partition_axis = 'W'
        
        if not handle_side:
            print(f"Using manual placement: right_frac={place_right_frac}, up_frac={place_up_frac}")
        
        # Append handle objects from the template blend.
        snap_plane_regex = r"(snap|snapplane|snap_plane|attach|plane)"

        appended = []
        with bpy.data.libraries.load(str(handle_blend), link=False) as (data_from, data_to):
            data_to.objects = data_from.objects[:]
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.scene.collection.objects.link(obj)
                appended.append(obj)
        
        if not appended:
            print("Error: No objects loaded from handle blend file")
            return False
        
        handle_objs = [o for o in appended if o is not None and o.type in {"MESH", "EMPTY"}]
        # Some templates ship with hide_render baked on the handle mesh
        # (e.g. handle16_knob) — appended copies inherit it and vanish
        # from renders. Force-visible; snap planes are hidden separately.
        for o in handle_objs:
            o.hide_render = False
            o.hide_viewport = False
        handle_meshes = [o for o in handle_objs if o.type == "MESH" and o.data is not None]
        
        if not handle_meshes:
            print("Error: No mesh objects found in handle asset")
            return False
        
        # Find snap plane
        snap_plane = None
        for obj in handle_meshes:
            if re.search(snap_plane_regex, obj.name, re.IGNORECASE):
                snap_plane = obj
                break
        
        if not snap_plane and handle_meshes:
            snap_plane = handle_meshes[0]
        
        other_meshes = [o for o in handle_meshes if o != snap_plane]
        
        print(f"Snap plane: {snap_plane.name if snap_plane else 'None'}")
        print(f"Handle meshes: {[o.name for o in handle_meshes]}")
        print(f"Other meshes (for scaling): {[o.name for o in other_meshes]}")
        
        # If other_meshes is empty but we have handle meshes, use all meshes for scaling
        # (excluding ONLY the snap_plane for scaling calculations)
        scale_meshes = other_meshes if other_meshes else handle_meshes
        
        # Infer protrusion direction from snap plane (same as handle_utils.infer_handle_protrusion_dir_world)
        snap_verts = [snap_plane.matrix_world @ v.co for v in snap_plane.data.vertices]
        snap_center = sum(snap_verts, Vector((0, 0, 0))) / len(snap_verts)
        
        # Get snap plane normal
        snap_normal = Vector((0, 0, 0))
        for poly in snap_plane.data.polygons:
            snap_normal += snap_plane.matrix_world.to_3x3() @ poly.normal
        snap_normal = snap_normal.normalized() if snap_normal.length > 0.001 else Vector((0, 0, -1))
        
        # Protrusion direction = opposite of the snap-plane normal.
        # The annotated snap-plane is curated to be axis-aligned with its
        # normal pointing AWAY from the handle body (toward the door at
        # attach time). Using the normal directly avoids the small (10-14°)
        # tilt that the centroid-offset heuristic introduces when the
        # handle body's mass isn't perfectly symmetric about the snap plane.
        protrusion = -snap_normal
        # Sanity check against centroid-offset; warn loudly if they disagree
        # by a lot (almost always = bad annotation, e.g. flipped normal).
        if other_meshes:
            handle_center = Vector((0, 0, 0))
            count = 0
            for obj in other_meshes:
                for v in obj.data.vertices:
                    handle_center += obj.matrix_world @ v.co
                    count += 1
            if count > 0:
                handle_center /= count
                centroid_offset = (handle_center - snap_center)
                if centroid_offset.length > 1e-9:
                    centroid_offset.normalize()
                    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, protrusion.dot(centroid_offset)))))
                    if angle_deg > 60.0:
                        print(f"[snap] WARN: snap-plane normal disagrees with body-centroid offset by "
                              f"{angle_deg:.1f}° -- annotation may have a flipped normal. Using normal anyway.")

        print(f"[snap] snap_plane={snap_plane.name} protrusion_dir={tuple(protrusion)}")
        
        # Create root empty at snap plane centroid
        root = bpy.data.objects.new("HANDLE_ROOT", None)
        root.empty_display_type = "PLAIN_AXES"
        root.location = snap_center
        bpy.context.scene.collection.objects.link(root)
        
        # Parent all handle objects to root
        for o in handle_objs:
            if o and o.name in bpy.data.objects:
                o.parent = root
                o.matrix_parent_inverse = root.matrix_world.inverted()
        
        # Rotate handle: protrusion -> door front
        root.rotation_mode = "QUATERNION"
        q_align = protrusion.rotation_difference(front)
        root.rotation_quaternion = q_align
        
        # Optional twist: align handle long axis to desired direction in door plane.
        # Works without motion info: align_long_to_up for doors, align_long_to_right for drawers.
        if twist_mode != "keep" and other_meshes:
            # Calculate anisotropy ratio (handle elongation)
            bpy.context.view_layer.update()
            all_verts = []
            for obj in other_meshes:
                all_verts.extend([obj.matrix_world @ v.co for v in obj.data.vertices])

            if all_verts:
                h_rs = [right.dot(v) for v in all_verts]
                h_us = [up.dot(v) for v in all_verts]
                h_span_r = max(h_rs) - min(h_rs)
                h_span_u = max(h_us) - min(h_us)

                ratio = max(h_span_r, h_span_u) / max(1e-12, min(h_span_r, h_span_u))
                thresh = 1.2

                print(f"[twist] in_plane_anisotropy_ratio={ratio:.3f} (thresh={thresh})")

                if ratio >= thresh:
                    # Principal direction (long axis)
                    principal = up if h_span_u > h_span_r else right

                    # Desired direction
                    if twist_mode == "align_long_to_motion" and motion_dir is not None:
                        # Project motion_dir onto door plane
                        d_proj = motion_dir - front * motion_dir.dot(front)
                        d_proj = d_proj.normalized() if d_proj.length > 1e-12 else up
                        desired = d_proj
                    elif twist_mode == "align_long_to_right":
                        desired = right
                    else:
                        # Default (and fallback when motion info is unavailable): vertical
                        desired = up

                    # Calculate rotation angle
                    def signed_angle(v1, v2, axis):
                        v1n = v1.normalized()
                        v2n = v2.normalized()
                        c = v1n.dot(v2n)
                        s = v1n.cross(v2n).dot(axis)
                        return math.atan2(s, c)

                    phi = signed_angle(principal, desired, front)
                    root.rotation_quaternion = Quaternion(front, phi) @ root.rotation_quaternion
                else:
                    print("[twist] skipped (symmetric/knob-like handle)")
        
        # Auto-rescale handle to fit door
        if auto_rescale and scale_meshes:
            bpy.context.view_layer.update()
            
            # Door dimensions
            door_w = max(1e-12, target_width)
            door_h = max(1e-12, target_height)
            door_major = max(door_w, door_h)
            door_minor = min(door_w, door_h)
            
            # Handle dimensions in door plane
            all_verts = []
            for obj in scale_meshes:
                all_verts.extend([obj.matrix_world @ v.co for v in obj.data.vertices])
            
            h_rs = [right.dot(v) for v in all_verts]
            h_us = [up.dot(v) for v in all_verts]
            h_fs = [front.dot(v) for v in all_verts]
            handle_w = max(1e-12, max(h_rs) - min(h_rs))
            handle_h = max(1e-12, max(h_us) - min(h_us))
            handle_protrusion = max(1e-12, max(h_fs) - min(h_fs))   # depth along door front
            handle_major = max(handle_w, handle_h)
            handle_minor = min(handle_w, handle_h)

            door_thickness = max(1e-12, target_depth)

            print(f"[scale] door_w={door_w:.4f} door_h={door_h:.4f} door_major={door_major:.4f} door_minor={door_minor:.4f} door_thick={door_thickness:.4f}")
            print(f"[scale] handle_w={handle_w:.4f} handle_h={handle_h:.4f} handle_major={handle_major:.4f} handle_minor={handle_minor:.4f} handle_prot={handle_protrusion:.4f}")
            
            # === Protrusion-driven scaling (primary) ===
            # The handle's protrusion (depth out of the door) is what
            # determines grippability. Target a per-style ratio of door
            # thickness, clamped to absolute mm range based on real-world
            # cabinet hardware sizes. This is the primary scale driver.
            ratio = STYLE_RATIO.get(_style_key, 2.0)
            min_prot_mm = STYLE_MIN_MM.get(_style_key, 18) / 1000.0
            max_prot_mm = STYLE_MAX_MM.get(_style_key, 50) / 1000.0
            target_protrusion = max(min_prot_mm,
                                    min(max_prot_mm, ratio * door_thickness))
            scale_factor = target_protrusion / handle_protrusion

            # In-plane safeguards (max-only): prevent the protrusion-driven
            # scale from making the handle absurdly wide on the door.
            # Hard caps are absolute (real-hardware) AND relative.
            ABS_MAX_MINOR = 0.10   # 100 mm — beyond which handle looks comical
            ABS_MAX_MAJOR = 0.30   # 300 mm — wider than most furniture handles
            max_minor = min(0.85 * door_minor, ABS_MAX_MINOR)
            max_major = min(0.35 * door_major, ABS_MAX_MAJOR)
            if handle_minor * scale_factor > max_minor:
                new_scale = max_minor / handle_minor
                print(f"[scale] in-plane minor cap: handle_minor*{scale_factor:.3f}={handle_minor*scale_factor*1000:.1f}mm "
                      f"> {max_minor*1000:.1f}mm -> scale dropped to {new_scale:.4f}")
                scale_factor = new_scale
            if handle_major * scale_factor > max_major:
                new_scale = max_major / handle_major
                print(f"[scale] in-plane major cap: handle_major*{scale_factor:.3f}={handle_major*scale_factor*1000:.1f}mm "
                      f"> {max_major*1000:.1f}mm -> scale dropped to {new_scale:.4f}")
                scale_factor = new_scale
            # Axis-wise hard caps: the handle must never exceed the door's
            # own extent along either in-plane axis. The major-vs-major cap
            # above misses the case where the handle's long axis lies along
            # the door's MINOR axis (thin-but-long doors), which let bars
            # grow longer than the door itself.
            MAX_AXIS_FRAC = 0.9
            if handle_w * scale_factor > MAX_AXIS_FRAC * door_w:
                new_scale = MAX_AXIS_FRAC * door_w / handle_w
                print(f"[scale] axis cap W: handle_w*{scale_factor:.3f}={handle_w*scale_factor*1000:.1f}mm "
                      f"> {MAX_AXIS_FRAC*door_w*1000:.1f}mm -> scale dropped to {new_scale:.4f}")
                scale_factor = new_scale
            if handle_h * scale_factor > MAX_AXIS_FRAC * door_h:
                new_scale = MAX_AXIS_FRAC * door_h / handle_h
                print(f"[scale] axis cap H: handle_h*{scale_factor:.3f}={handle_h*scale_factor*1000:.1f}mm "
                      f"> {MAX_AXIS_FRAC*door_h*1000:.1f}mm -> scale dropped to {new_scale:.4f}")
                scale_factor = new_scale
            # Multi-handle slot cap: with N handles at equal partitions
            # along `partition_axis`, each one must fit its own slot —
            # a long bar shrinks so N of them sit side by side without
            # overlapping. MULTI_SLOT_FRAC leaves a gap between slots.
            MULTI_SLOT_FRAC = 0.8
            if n_handles > 1 and partition_axis is not None:
                if partition_axis == 'W':
                    slot = door_w / n_handles
                    ext = handle_w
                    tag = 'W'
                else:
                    slot = door_h / n_handles
                    ext = handle_h
                    tag = 'H'
                if ext * scale_factor > MULTI_SLOT_FRAC * slot:
                    new_scale = MULTI_SLOT_FRAC * slot / ext
                    print(f"[scale] multi-slot cap {tag}: {n_handles} handles, "
                          f"slot={slot*1000:.1f}mm, ext*{scale_factor:.3f}="
                          f"{ext*scale_factor*1000:.1f}mm -> scale dropped "
                          f"to {new_scale:.4f}")
                    scale_factor = new_scale

            print(f"[scale] protrusion target={target_protrusion*1000:.1f}mm "
                  f"(ratio={ratio} × {door_thickness*1000:.1f}mm, clamp "
                  f"[{STYLE_MIN_MM.get(_style_key,18)},{STYLE_MAX_MM.get(_style_key,50)}]mm). "
                  f"native={handle_protrusion*1000:.1f}mm -> scale={scale_factor:.4f}")
            
            # Clamp to reasonable range. The lower bound is intentionally
            # generous (1e-5) so kilometer-scale templates (e.g., handles
            # authored at mm-values-as-meters) can be brought down to ~1cm.
            scale_factor = max(1e-5, min(100.0, scale_factor))
            
            print(f"[scale] applying isotropic scale factor: {scale_factor:.4f}")
            print(f"[scale] root BEFORE scale: {tuple(root.scale)}")
            root.scale = (scale_factor, scale_factor, scale_factor)
            print(f"[scale] root AFTER scale: {tuple(root.scale)}")
            
            # CRITICAL: Update view layer after scaling so transforms are applied
            bpy.context.view_layer.update()
            print(f"[scale] root AFTER view_layer update: {tuple(root.scale)}")
        
        # Calculate target position using RAYCAST (not bounding box)
        # This handles doors with protruding ornaments correctly
        ray_margin = 0.02
        
        # Raycast origin: slightly in front of the bounding box
        target_r = r_min + place_right_frac * (r_max - r_min)
        target_u = u_min + place_up_frac * (u_max - u_min)
        
        ray_origin = right * target_r + up * target_u + front * (f_max + ray_margin)
        ray_direction = -front
        
        # Perform raycast to find actual surface
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ray_distance = (f_max - f_min) * 3.0 + ray_margin * 2.0 + 1.0
        
        hit, loc, norm, face_idx, hit_obj, _ = bpy.context.scene.ray_cast(
            depsgraph, ray_origin, ray_direction, distance=ray_distance
        )
        
        if hit and hit_obj == target_obj:
            target_pos = loc
            print(f"[place] raycast HIT at {tuple(target_pos)}")
        else:
            # Fallback to bounding box if raycast misses
            target_pos = right * target_r + up * target_u + front * f_max
            print(f"[place] raycast MISS - fallback to bbox front")
        
        print(f"[place] target_loc={tuple(target_pos)} place_right_frac={place_right_frac} place_up_frac={place_up_frac}")
        
        # Move root so snap plane lands on target
        bpy.context.view_layer.update()
        snap_verts_new = [snap_plane.matrix_world @ v.co for v in snap_plane.data.vertices]
        snap_center_new = sum(snap_verts_new, Vector((0, 0, 0))) / len(snap_verts_new)
        
        offset = target_pos - snap_center_new
        root.location += offset
        bpy.context.view_layer.update()

        # Mesh-based contact snap: the center ray only guarantees ONE
        # surface point; settle the feet onto the actual mesh.
        shift = _post_snap_to_surface(root, other_meshes, target_obj,
                                      front, f_max)
        if abs(shift) > 1e-6:
            print(f"[place] mesh-contact snap: {shift*1000:+.1f}mm along front")

        # Hide snap plane (optional, can be removed later)
        snap_plane.hide_viewport = True
        snap_plane.hide_render = True

        # Update view
        bpy.context.view_layer.update()
        
        # PARENT the handle root to the door so it moves with animation
        root.parent = target_obj
        root.matrix_parent_inverse = target_obj.matrix_world.inverted()
        
        print(f"✓ Handle attached to {target_obj.name} (parented for animation)")
        return True
        
    except Exception as e:
        print(f"Error applying handle: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# SUBTRACTIVE HANDLE (IN-PLACE CARVING)
# ============================================================================

def apply_subtractive_handle_inplace(
    sub_type='flat',
    engrave_location='auto',
    margin_frac=0.08,
    depth_frac=0.5,
    width_frac=0.8,
    length_frac=1.0,
    front_axis='-Y',
    up_axis='+Z',
    joint_info=None,
    target_type='door',
    door_mode='opposite',
    drawer_mode='top',
    front_position_frac=0.10,
    front_lip_depth_frac=0.3,
    cyl_segments=32,
    overshoot_frac=0.01,
    face_band_frac=0.02,
    method='DEFORM',
    subdivisions=2,
    falloff=0.3
):
    """
    Apply a subtractive (carved/engraved) handle to the selected object in-place.
    
    Supports two methods:
    - BOOLEAN: Sharp cut using CSG - may cause mesh artifacts
    - DEFORM: Soft vertex displacement - preserves mesh topology
    
    Smart placement based on joint type:
    - DOORS: Never on hinge side. Opposite=center groove, Top/Bottom=far from hinge
    - DRAWERS: Top/Bottom=central full-length, Front=horizontal near top
    
    Args:
        sub_type: 'flat' (rectangular pocket) or 'u' (U-groove with rounded bottom)
        engrave_location: 'auto' for smart placement, or manual: 'top', 'bottom', etc.
        margin_frac: Margin from edges as fraction of span (0.08 = 8%)
        depth_frac: Depth as fraction of FACE THICKNESS (0.5 = 50%)
        width_frac: Width as fraction of face thickness (0.8 = 80%)
        length_frac: Length as fraction of available edge (1.0 = full)
        front_axis: Front direction (e.g., '-Y')
        up_axis: Up direction (e.g., '+Z')
        joint_info: Dict with 'origin', 'direction', 'type' for smart placement
        target_type: 'door' or 'drawer' (auto-detected from joint_info if available)
        door_mode: For doors - 'opposite', 'top', 'bottom', 'top_bottom', 'front'
        drawer_mode: For drawers - 'top', 'bottom', 'both', 'front'
        front_position_frac: Position from edge for front groove (0.1 = 10%)
        front_lip_depth_frac: Lip protrusion for front groove grip
        cyl_segments: Number of segments for U-groove cylinder
        overshoot_frac: Extra penetration (BOOLEAN only)
        face_band_frac: Band width for thickness face detection
        method: 'BOOLEAN' or 'DEFORM'
        subdivisions: Subdivision levels before deformation (DEFORM only)
        falloff: Edge smoothness (0=sharp, 1=smooth) (DEFORM only)
        
    Returns:
        bool: True if successful
    """
    import math
    
    # Get selected object
    target_obj = bpy.context.active_object
    if not target_obj or target_obj.type != 'MESH':
        print("Error: No mesh object selected")
        return False
    
    print(f"\n=== Subtractive Handle (In-Place) ===")
    print(f"  Object: {target_obj.name}")
    print(f"  Sub type: {sub_type}")
    print(f"  Engrave location: {engrave_location}")
    print(f"  Margin: {margin_frac:.2%}, Depth: {depth_frac:.2%}")
    
    # Parse axis directions
    def parse_axis(s):
        s = s.strip().upper()
        sign = -1.0 if s.startswith('-') else 1.0
        axis = s[-1]
        vec = Vector((0, 0, 0))
        if axis == 'X':
            vec.x = sign
        elif axis == 'Y':
            vec.y = sign
        elif axis == 'Z':
            vec.z = sign
        return vec.normalized()
    
    front = parse_axis(front_axis)
    up = parse_axis(up_axis)
    right = up.cross(front).normalized()
    
    # Re-orthonormalize
    front = right.cross(up).normalized()
    
    # Get object world vertices
    verts = [target_obj.matrix_world @ v.co for v in target_obj.data.vertices]
    if not verts:
        print("Error: No vertices in target object")
        return False
    
    # Calculate bounds in Right/Up/Front frame
    rs = [right.dot(v) for v in verts]
    us = [up.dot(v) for v in verts]
    fs = [front.dot(v) for v in verts]
    
    r_min, r_max = min(rs), max(rs)
    u_min, u_max = min(us), max(us)
    f_min, f_max = min(fs), max(fs)
    
    panel_width = r_max - r_min
    panel_height = u_max - u_min
    panel_thickness = f_max - f_min
    
    print(f"  Panel bounds: W={panel_width:.4f}, H={panel_height:.4f}, D={panel_thickness:.4f}")
    
    # Detect front panel by finding faces that are on the FRONT surface
    # These are faces with normals pointing in the front direction
    front_panel_info = _detect_front_panel(target_obj, front, up, right, 
                                           r_min, r_max, u_min, u_max, f_min, f_max,
                                           face_band_frac)
    
    # Determine engrave location from joint info (same logic as additive)
    if engrave_location == 'auto' and joint_info is not None:
        motion_origin = joint_info.get('origin')
        motion_dir = joint_info.get('direction')
        
        if motion_origin is not None and motion_dir is not None:
            motion_origin = Vector(motion_origin) if not isinstance(motion_origin, Vector) else motion_origin
            motion_dir = Vector(motion_dir) if not isinstance(motion_dir, Vector) else motion_dir
            
            # Same hinge-side detection logic as additive handle
            boundary_eps_frac = 0.02
            dist_quantile = 0.1
            
            r_span = r_max - r_min
            u_span = u_max - u_min
            eps_r = max(1e-9, boundary_eps_frac * r_span)
            eps_u = max(1e-9, boundary_eps_frac * u_span)
            
            # Motion line in RU coordinates
            r0 = right.dot(motion_origin)
            u0 = up.dot(motion_origin)
            dr = right.dot(motion_dir.normalized())
            du = up.dot(motion_dir.normalized())
            
            # Build boundary vertex sets
            left_idx = [i for i, r in enumerate(rs) if r <= r_min + eps_r]
            right_idx = [i for i, r in enumerate(rs) if r >= r_max - eps_r]
            bottom_idx = [i for i, u in enumerate(us) if u <= u_min + eps_u]
            top_idx = [i for i, u in enumerate(us) if u >= u_max - eps_u]
            
            # Relax if needed
            if not left_idx or not right_idx or not bottom_idx or not top_idx:
                eps_r = max(eps_r, 0.05 * r_span)
                eps_u = max(eps_u, 0.05 * u_span)
                left_idx = [i for i, r in enumerate(rs) if r <= r_min + eps_r]
                right_idx = [i for i, r in enumerate(rs) if r >= r_max - eps_r]
                bottom_idx = [i for i, u in enumerate(us) if u <= u_min + eps_u]
                top_idx = [i for i, u in enumerate(us) if u >= u_max - eps_u]
            
            side_to_idx = {
                "left": left_idx, "right": right_idx,
                "bottom": bottom_idx, "top": top_idx,
            }
            
            # Distance from point to line in 2D
            def point_line_dist_2d(a0, a1, p0, p1, d0, d1):
                denom = math.hypot(d0, d1)
                if denom < 1e-12:
                    return math.hypot(a0 - p0, a1 - p1)
                return abs(d1 * (a0 - p0) - d0 * (a1 - p1)) / denom
            
            # Compute scores for each side (lower = closer to hinge axis)
            scores = {}
            for side, idxs in side_to_idx.items():
                if not idxs:
                    scores[side] = float('inf')
                    continue
                dists = [point_line_dist_2d(rs[i], us[i], r0, u0, dr, du) for i in idxs]
                dists.sort()
                q_idx = int(round(dist_quantile * (len(dists) - 1)))
                scores[side] = dists[q_idx]
            
            # Hinge side is closest to motion axis
            hinge_side = min(scores.keys(), key=lambda k: scores[k])
            opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
            handle_side = opposite[hinge_side]
            
            print(f"  Auto-detected: hinge_side={hinge_side}, handle_side={handle_side}")
    
    # Compute overshoot in world units
    overshoot = overshoot_frac * min(panel_width, panel_height)
    
    # =================================================================
    # SMART GROOVE PLACEMENT
    # =================================================================
    grooves_to_cut = []
    groove_configs = {}  # Store per-groove configuration (position offset, length)
    
    if engrave_location == 'auto':
        # Smart placement based on target type and mode
        if target_type == 'door':
            print(f"  Door mode: {door_mode}, hinge_side: {hinge_side if 'hinge_side' in dir() else 'unknown'}")
            
            if door_mode == 'opposite':
                # Central groove on opposite side from hinge
                if 'handle_side' in dir():
                    grooves_to_cut = [handle_side]
                    groove_configs[handle_side] = {'position': 'center', 'length_frac': length_frac}
                else:
                    grooves_to_cut = ['right']  # Default fallback
                    groove_configs['right'] = {'position': 'center', 'length_frac': length_frac}
                    
            elif door_mode == 'top':
                # Groove on top edge, positioned far from hinge
                grooves_to_cut = ['top']
                if 'hinge_side' in dir() and hinge_side in ('left', 'right'):
                    # Position in far 1/4 from hinge
                    pos = 'far_right' if hinge_side == 'left' else 'far_left'
                    groove_configs['top'] = {'position': pos, 'length_frac': min(length_frac, 0.4)}
                else:
                    groove_configs['top'] = {'position': 'center', 'length_frac': length_frac}
                    
            elif door_mode == 'bottom':
                # Groove on bottom edge, positioned far from hinge
                grooves_to_cut = ['bottom']
                if 'hinge_side' in dir() and hinge_side in ('left', 'right'):
                    pos = 'far_right' if hinge_side == 'left' else 'far_left'
                    groove_configs['bottom'] = {'position': pos, 'length_frac': min(length_frac, 0.4)}
                else:
                    groove_configs['bottom'] = {'position': 'center', 'length_frac': length_frac}
                    
            elif door_mode == 'top_bottom':
                # Both top and bottom, far from hinge
                grooves_to_cut = ['top', 'bottom']
                if 'hinge_side' in dir() and hinge_side in ('left', 'right'):
                    pos = 'far_right' if hinge_side == 'left' else 'far_left'
                    groove_configs['top'] = {'position': pos, 'length_frac': min(length_frac, 0.4)}
                    groove_configs['bottom'] = {'position': pos, 'length_frac': min(length_frac, 0.4)}
                else:
                    groove_configs['top'] = {'position': 'center', 'length_frac': length_frac}
                    groove_configs['bottom'] = {'position': 'center', 'length_frac': length_frac}
                    
            elif door_mode == 'front':
                # Front surface groove parallel to hinge axis
                grooves_to_cut = ['front']
                groove_configs['front'] = {
                    'position': 'far_from_hinge',
                    'length_frac': length_frac,
                    'hinge_side': hinge_side if 'hinge_side' in dir() else 'left'
                }
        else:
            # DRAWER
            print(f"  Drawer mode: {drawer_mode}")
            
            if drawer_mode == 'top':
                grooves_to_cut = ['top']
                groove_configs['top'] = {'position': 'center', 'length_frac': length_frac}
                
            elif drawer_mode == 'bottom':
                grooves_to_cut = ['bottom']
                groove_configs['bottom'] = {'position': 'center', 'length_frac': length_frac}
                
            elif drawer_mode == 'both':
                grooves_to_cut = ['top', 'bottom']
                groove_configs['top'] = {'position': 'center', 'length_frac': length_frac}
                groove_configs['bottom'] = {'position': 'center', 'length_frac': length_frac}
                
            elif drawer_mode == 'front':
                # Front surface groove horizontal near top
                grooves_to_cut = ['front']
                groove_configs['front'] = {'position': 'near_top', 'length_frac': length_frac}
    else:
        # Manual location mode
        if engrave_location == 'side':
            grooves_to_cut = ['left', 'right']
            groove_configs['left'] = {'position': 'center', 'length_frac': length_frac}
            groove_configs['right'] = {'position': 'center', 'length_frac': length_frac}
        elif engrave_location == 'front':
            grooves_to_cut = ['front']
            groove_configs['front'] = {'position': 'center', 'length_frac': length_frac}
        else:
            grooves_to_cut = [engrave_location]
            groove_configs[engrave_location] = {'position': 'center', 'length_frac': length_frac}
    
    print(f"  Grooves to cut: {grooves_to_cut}")
    
    try:
        # Ensure object mode
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        for loc in grooves_to_cut:
            # Get groove configuration
            config = groove_configs.get(loc, {'position': 'center', 'length_frac': length_frac})
            pos_mode = config.get('position', 'center')
            loc_length_frac = config.get('length_frac', length_frac)
            
            print(f"  [{loc}] Config: position={pos_mode}, length_frac={loc_length_frac}")
            
            # Handle FRONT groove separately
            if loc == 'front':
                # Front groove on the graspable surface
                _apply_front_groove(
                    target_obj=target_obj,
                    front=front, up=up, right=right,
                    r_min=r_min, r_max=r_max,
                    u_min=u_min, u_max=u_max,
                    f_min=f_min, f_max=f_max,
                    panel_width=panel_width, panel_height=panel_height,
                    config=config,
                    target_type=target_type,
                    margin_frac=margin_frac,
                    depth_frac=depth_frac,
                    width_frac=width_frac,
                    length_frac=loc_length_frac,
                    front_position_frac=front_position_frac,
                    front_lip_depth_frac=front_lip_depth_frac,
                    sub_type=sub_type,
                    method=method,
                    subdivisions=subdivisions,
                    falloff=falloff,
                    overshoot=overshoot,
                    cyl_segments=cyl_segments
                )
                continue
            
            # Get face thickness AND center for this edge
            face_thickness, thickness_center = _get_face_thickness(
                target_obj, front, up, right,
                r_min, r_max, u_min, u_max, f_min, f_max,
                loc, face_band_frac
            )
            
            print(f"  [{loc}] Face thickness: {face_thickness:.4f}, center: {thickness_center:.4f}")
            
            # =============================================================
            # SOCKET-STYLE DIMENSIONS (based on face thickness)
            # =============================================================
            margin_T = margin_frac * face_thickness
            max_width = face_thickness - 2 * margin_T
            groove_width = max_width * width_frac
            radius = groove_width / 2
            
            # Depth: Fraction of face thickness
            depth = face_thickness * depth_frac
            depth = min(depth, groove_width * 1.5)  # Safety cap
            
            print(f"  [{loc}] Groove width: {groove_width:.4f}, depth: {depth:.4f}")
            
            if loc in ('top', 'bottom'):
                # Horizontal groove (runs along RIGHT axis)
                margin_R = margin_frac * panel_width
                available_len = (r_max - r_min) - 2 * margin_R
                long_len = max(0.001, available_len * loc_length_frac)
                
                # Determine position along the edge
                if pos_mode == 'center':
                    r_center = (r_min + r_max) / 2
                elif pos_mode == 'far_left':
                    # Position in left 1/4, centered within that region
                    region_start = r_min + margin_R
                    region_end = r_min + margin_R + available_len * 0.25
                    r_center = (region_start + region_end) / 2
                elif pos_mode == 'far_right':
                    # Position in right 1/4, centered within that region
                    region_start = r_max - margin_R - available_len * 0.25
                    region_end = r_max - margin_R
                    r_center = (region_start + region_end) / 2
                else:
                    r_center = (r_min + r_max) / 2
                
                # Face anchor position
                if loc == 'top':
                    u_face = u_max
                    pen_dir = -up
                else:
                    u_face = u_min
                    pen_dir = up
                
                face_anchor = right * r_center + up * u_face + front * thickness_center
                long_axis = right
                thickness_axis = front
                
            else:  # left or right
                # Vertical groove (runs along UP axis)
                margin_U = margin_frac * panel_height
                available_len = (u_max - u_min) - 2 * margin_U
                long_len = max(0.001, available_len * loc_length_frac)
                
                # Center position (vertical grooves typically centered)
                u_center = (u_min + u_max) / 2
                
                # Face anchor position
                if loc == 'left':
                    r_face = r_min
                    pen_dir = right
                else:
                    r_face = r_max
                    pen_dir = -right
                
                face_anchor = right * r_face + up * u_center + front * thickness_center
                long_axis = up
                thickness_axis = front
            
            print(f"  [{loc}] Groove: len={long_len:.4f}, width={groove_width:.4f}, depth={depth:.4f}")
            print(f"  [{loc}] Method: {method}")
            
            if method == 'DEFORM':
                # ============================================================
                # SOFT DEFORM METHOD - vertex displacement
                # ============================================================
                _apply_soft_groove_deform(
                    target_obj=target_obj,
                    face_anchor=face_anchor,
                    long_axis=long_axis,
                    pen_dir=pen_dir,
                    long_len=long_len,
                    width=groove_width,
                    depth=depth,
                    sub_type=sub_type,
                    subdivisions=subdivisions,
                    falloff=falloff,
                    loc=loc
                )
                print(f"  [{loc}] ✓ Groove deformed (soft)")
            else:
                # ============================================================
                # BOOLEAN METHOD - CSG cutting
                # ============================================================
                # Create cutter based on sub_type
                if sub_type == 'u':
                    cutter = _create_u_groove_cutter(
                        name=f"ENGRAVE_{loc.upper()}",
                        center=face_anchor,
                        long_axis=long_axis,
                        thickness_axis=thickness_axis,
                        pen_dir=pen_dir,
                        long_len=long_len,
                        radius=radius,
                        depth=depth,
                        overshoot=overshoot,
                        cyl_segments=cyl_segments
                    )
                else:
                    # Flat pocket: use groove_width
                    cutter = _create_flat_pocket_cutter(
                        name=f"ENGRAVE_{loc.upper()}",
                        center=face_anchor,
                        long_axis=long_axis,
                        pen_dir=pen_dir,
                        long_len=long_len,
                        width=groove_width,
                        depth=depth,
                        overshoot=overshoot
                    )
                
                if not cutter:
                    print(f"Error: Failed to create cutter for {loc}")
                    continue
                
                # Apply boolean difference
                print(f"  [{loc}] Applying boolean difference...")
                
                # Make target active and add boolean modifier
                bpy.ops.object.select_all(action='DESELECT')
                target_obj.select_set(True)
                bpy.context.view_layer.objects.active = target_obj
                
                mod = target_obj.modifiers.new(name=f"ENGRAVE_{loc.upper()}", type='BOOLEAN')
                mod.operation = 'DIFFERENCE'
                mod.object = cutter
                
                # Set solver (try EXACT first, fallback to FLOAT)
                if hasattr(mod, 'solver'):
                    try:
                        mod.solver = 'EXACT'
                    except:
                        try:
                            mod.solver = 'FLOAT'
                        except:
                            pass
                
                # Robustness settings
                if hasattr(mod, 'use_hole_tolerant'):
                    mod.use_hole_tolerant = True
                if hasattr(mod, 'double_threshold'):
                    mod.double_threshold = 1e-6
                
                # Apply the modifier
                bpy.ops.object.modifier_apply(modifier=mod.name)
                
                # Remove cutter
                bpy.data.objects.remove(cutter, do_unlink=True)
                
                print(f"  [{loc}] ✓ Groove carved (boolean)")
        
        print(f"✓ Subtractive handle applied to {target_obj.name}")
        return True
        
    except Exception as e:
        print(f"Error applying subtractive handle: {e}")
        import traceback
        traceback.print_exc()
        return False


def _detect_front_panel(target_obj, front, up, right, r_min, r_max, u_min, u_max, f_min, f_max, band_frac):
    """
    Detect the front panel of a door/drawer.
    
    The front panel is the graspable surface - typically the largest face 
    facing the front direction.
    
    Returns:
        dict: Information about the front panel
    """
    mw = target_obj.matrix_world
    me = target_obj.data
    
    front_faces = []
    
    # Find faces that face the front direction (normal aligned with front)
    for poly in me.polygons:
        normal_world = (mw.to_3x3() @ poly.normal).normalized()
        alignment = normal_world.dot(front)
        
        if alignment > 0.7:  # Face is mostly pointing front
            # Get face center
            center = mw @ poly.center
            front_faces.append({
                'poly': poly,
                'center': center,
                'normal': normal_world,
                'alignment': alignment,
                'area': poly.area
            })
    
    # The front panel is typically the largest front-facing area
    if front_faces:
        # Sort by area (largest first)
        front_faces.sort(key=lambda x: x['area'], reverse=True)
        return {
            'center': front_faces[0]['center'],
            'normal': front_faces[0]['normal'],
            'total_area': sum(f['area'] for f in front_faces)
        }
    
    # Fallback: use bounding box front face
    return {
        'center': right * (r_min + r_max) / 2 + up * (u_min + u_max) / 2 + front * f_max,
        'normal': front,
        'total_area': (r_max - r_min) * (u_max - u_min)
    }


def _get_face_thickness(target_obj, front, up, right, r_min, r_max, u_min, u_max, f_min, f_max, 
                        face_kind, band_frac):
    """
    Get the thickness AND center of a specific edge face (top, bottom, left, right).
    
    This measures the FRONT extent of faces on the specified edge,
    which gives the actual graspable thickness and its center position.
    
    Args:
        target_obj: The Blender mesh object
        front, up, right: Axis vectors
        r_min, r_max, u_min, u_max, f_min, f_max: Bounding box limits
        face_kind: 'top', 'bottom', 'left', or 'right'
        band_frac: Fraction of span for edge detection band
        
    Returns:
        tuple: (thickness, thickness_center) - thickness value and center along front axis
    """
    mw = target_obj.matrix_world
    me = target_obj.data
    
    panel_width = r_max - r_min
    panel_height = u_max - u_min
    panel_depth = f_max - f_min
    
    # Determine which axis defines this face and the band
    if face_kind in ('top', 'bottom'):
        face_axis = up
        if face_kind == 'top':
            face_extreme = u_max
        else:
            face_extreme = u_min
        band = band_frac * panel_height
    else:  # left, right
        face_axis = right
        if face_kind == 'left':
            face_extreme = r_min
        else:
            face_extreme = r_max
        band = band_frac * panel_width
    
    # Collect points from faces near the extreme position
    thickness_points = []
    
    for poly in me.polygons:
        vs_world = [mw @ me.vertices[i].co for i in poly.vertices]
        
        # Check if face is near the extreme (within band)
        dists = [abs(face_axis.dot(v) - face_extreme) for v in vs_world]
        
        # Accept face if most vertices are within band
        n_in_band = sum(1 for d in dists if d <= band)
        if n_in_band >= max(2, len(dists) // 2):
            # Get front-extent of this face
            f_vals = [front.dot(v) for v in vs_world]
            thickness_points.extend(f_vals)
    
    if thickness_points:
        # Use percentile to be robust to outliers
        thickness_points.sort()
        n = len(thickness_points)
        p05_idx = int(0.05 * n) if n > 20 else 0
        p95_idx = int(0.95 * n) if n > 20 else n - 1
        t_min = thickness_points[p05_idx]
        t_max = thickness_points[p95_idx]
        
        thickness = max(0.001, t_max - t_min)
        thickness_center = 0.5 * (t_min + t_max)  # Center of the edge face
        return thickness, thickness_center
    
    # Fallback: use full panel depth and center
    return panel_depth, (f_min + f_max) / 2


def _apply_front_groove(target_obj, front, up, right,
                        r_min, r_max, u_min, u_max, f_min, f_max,
                        panel_width, panel_height,
                        config, target_type,
                        margin_frac, depth_frac, width_frac, length_frac,
                        front_position_frac, front_lip_depth_frac,
                        sub_type, method, subdivisions, falloff,
                        overshoot, cyl_segments):
    """
    Apply a groove on the front (graspable) surface.
    
    For DOORS: Vertical groove parallel to hinge axis, positioned far from hinge
    For DRAWERS: Horizontal groove near the top edge
    """
    print(f"  [front] Applying front surface groove (type={target_type})")
    
    # Front surface position
    f_surface = f_max
    
    # Groove dimensions (use panel thickness for front grooves)
    panel_thickness = f_max - f_min
    groove_depth = panel_thickness * depth_frac * 0.5  # Shallower for front
    groove_width = panel_thickness * width_frac * 0.3  # Narrower for front (finger groove)
    
    # Safety limits
    groove_depth = min(groove_depth, panel_thickness * 0.3)  # Max 30% of thickness
    groove_width = min(groove_width, panel_thickness * 0.5)  # Max 50% of thickness
    
    if target_type == 'door':
        # DOOR: Vertical groove parallel to hinge axis
        # Positioned at front_position_frac from the handle side (far from hinge)
        hinge_side = config.get('hinge_side', 'left')
        
        # Groove runs vertically (along UP axis)
        long_axis = up
        
        # Position from edge (on the handle side, which is opposite to hinge)
        if hinge_side == 'left':
            # Handle side is right, so groove is near right edge
            r_pos = r_max - panel_width * front_position_frac
        else:
            # Handle side is left, so groove is near left edge
            r_pos = r_min + panel_width * front_position_frac
        
        # Groove length (vertical span)
        margin_U = margin_frac * panel_height
        available_len = panel_height - 2 * margin_U
        long_len = available_len * length_frac
        
        # Groove center
        u_center = (u_min + u_max) / 2
        
        # Penetration direction (into the front surface)
        pen_dir = -front
        
        # Face anchor
        face_anchor = right * r_pos + up * u_center + front * f_surface
        
        print(f"  [front] Door groove: hinge={hinge_side}, pos={r_pos:.4f}, len={long_len:.4f}")
        
    else:
        # DRAWER: Horizontal groove near the top
        # Positioned at front_position_frac from top edge
        
        # Groove runs horizontally (along RIGHT axis)
        long_axis = right
        
        # Position near top
        u_pos = u_max - panel_height * front_position_frac
        
        # Groove length (horizontal span)
        margin_R = margin_frac * panel_width
        available_len = panel_width - 2 * margin_R
        long_len = available_len * length_frac
        
        # Groove center
        r_center = (r_min + r_max) / 2
        
        # Penetration direction (into the front surface)
        pen_dir = -front
        
        # Face anchor
        face_anchor = right * r_center + up * u_pos + front * f_surface
        
        print(f"  [front] Drawer groove: pos={u_pos:.4f}, len={long_len:.4f}")
    
    print(f"  [front] Groove: width={groove_width:.4f}, depth={groove_depth:.4f}, len={long_len:.4f}")
    
    # Apply the groove using the appropriate method
    if method == 'DEFORM':
        _apply_soft_groove_deform(
            target_obj=target_obj,
            face_anchor=face_anchor,
            long_axis=long_axis,
            pen_dir=pen_dir,
            long_len=long_len,
            width=groove_width,
            depth=groove_depth,
            sub_type=sub_type,
            subdivisions=subdivisions,
            falloff=falloff,
            loc='front'
        )
    else:
        # Boolean method
        thickness_axis = long_axis.cross(pen_dir).normalized()
        
        if sub_type == 'u':
            cutter = _create_u_groove_cutter(
                name="ENGRAVE_FRONT",
                center=face_anchor,
                long_axis=long_axis,
                thickness_axis=thickness_axis,
                pen_dir=pen_dir,
                long_len=long_len,
                radius=groove_width / 2,
                depth=groove_depth,
                overshoot=overshoot,
                cyl_segments=cyl_segments
            )
        else:
            cutter = _create_flat_pocket_cutter(
                name="ENGRAVE_FRONT",
                center=face_anchor,
                long_axis=long_axis,
                pen_dir=pen_dir,
                long_len=long_len,
                width=groove_width,
                depth=groove_depth,
                overshoot=overshoot
            )
        
        if cutter:
            bpy.ops.object.select_all(action='DESELECT')
            target_obj.select_set(True)
            bpy.context.view_layer.objects.active = target_obj
            
            mod = target_obj.modifiers.new(name="ENGRAVE_FRONT", type='BOOLEAN')
            mod.operation = 'DIFFERENCE'
            mod.object = cutter
            
            if hasattr(mod, 'solver'):
                try:
                    mod.solver = 'EXACT'
                except:
                    try:
                        mod.solver = 'FLOAT'
                    except:
                        pass
            
            bpy.ops.object.modifier_apply(modifier=mod.name)
            bpy.data.objects.remove(cutter, do_unlink=True)
    
    print(f"  [front] ✓ Front groove applied")


def _apply_soft_groove_deform(target_obj, face_anchor, long_axis, pen_dir,
                               long_len, width, depth, sub_type='flat',
                               subdivisions=2, falloff=0.3, loc='top'):
    """
    Apply a soft groove deformation by displacing vertices.
    
    This method preserves mesh topology and avoids boolean artifacts.
    Vertices inside the groove region are pushed inward proportionally.
    
    Args:
        target_obj: The Blender mesh object to deform
        face_anchor: Center of groove opening (Vector)
        long_axis: Direction along groove length
        pen_dir: Penetration direction (into the surface)
        long_len: Length of groove
        width: Width of groove (diameter)
        depth: Maximum depth of groove
        sub_type: 'flat' or 'u' (affects bottom shape)
        subdivisions: Number of subdivision levels before deformation
        falloff: Edge smoothness (0=sharp, 1=very smooth)
        loc: Location identifier for logging
    """
    import math
    
    # Ensure object mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Select and activate target
    bpy.ops.object.select_all(action='DESELECT')
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    
    # Apply subdivisions if requested (for smoother deformation)
    if subdivisions > 0:
        print(f"  [{loc}] Applying {subdivisions} subdivision(s)...")
        mod = target_obj.modifiers.new(name="GROOVE_SUBDIV", type='SUBSURF')
        mod.levels = subdivisions
        mod.render_levels = subdivisions
        mod.subdivision_type = 'SIMPLE'  # Use simple for predictable results
        bpy.ops.object.modifier_apply(modifier=mod.name)
    
    # Get mesh data
    me = target_obj.data
    mw = target_obj.matrix_world
    mw_inv = mw.inverted()
    
    # Compute groove parameters
    half_len = long_len / 2
    half_width = width / 2
    
    # Normalize axes
    long_axis = long_axis.normalized()
    pen_dir = pen_dir.normalized()
    cross_axis = long_axis.cross(pen_dir).normalized()
    
    # Calculate falloff margin (smooth edge region)
    falloff_margin = falloff * half_width
    
    print(f"  [{loc}] Deforming vertices (depth={depth:.4f}, falloff={falloff:.2f})...")
    
    # Enter edit mode to work with bmesh for efficient vertex access
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(me)
    
    vertices_modified = 0
    
    for v in bm.verts:
        # Get vertex position in world space
        v_world = mw @ v.co
        
        # Vector from face anchor to vertex
        offset = v_world - face_anchor
        
        # Project onto groove coordinate system
        along_long = long_axis.dot(offset)  # Position along groove length
        along_cross = cross_axis.dot(offset)  # Position across groove width
        along_pen = pen_dir.dot(offset)  # Position along penetration direction
        
        # Check if vertex is within groove region (long axis bounds)
        if abs(along_long) > half_len:
            continue
        
        # Check if vertex is within groove region (cross axis bounds)
        if abs(along_cross) > half_width:
            continue
        
        # Check if vertex is on the surface (not deep inside the object)
        # Only affect vertices near the face (within 2x depth of surface)
        if along_pen > depth * 0.5 or along_pen < -depth * 2:
            continue
        
        # Calculate displacement amount
        # Based on distance from groove edge (with falloff)
        
        # Distance from long axis edge (0 at center, 1 at edge)
        long_edge_dist = abs(along_long) / half_len if half_len > 0 else 0
        
        # Distance from cross axis edge (0 at center, 1 at edge)
        cross_edge_dist = abs(along_cross) / half_width if half_width > 0 else 0
        
        # Compute falloff factor for smooth edges
        # Sharp in center, smooth transition at edges
        long_factor = 1.0
        cross_factor = 1.0
        
        if falloff > 0:
            # Smooth falloff from edge
            falloff_start = 1.0 - falloff
            if long_edge_dist > falloff_start:
                long_factor = 1.0 - (long_edge_dist - falloff_start) / falloff
                long_factor = max(0, min(1, long_factor))
            if cross_edge_dist > falloff_start:
                cross_factor = 1.0 - (cross_edge_dist - falloff_start) / falloff
                cross_factor = max(0, min(1, cross_factor))
        
        # Combined factor
        edge_factor = long_factor * cross_factor
        
        # Depth profile based on sub_type
        if sub_type == 'u':
            # U-groove: rounded bottom (use cosine profile)
            # Deeper in center, curves up at sides
            cross_profile = math.cos(cross_edge_dist * math.pi / 2)
        else:
            # Flat: constant depth across width
            cross_profile = 1.0
        
        # Calculate final displacement
        # Maximum displacement at surface (along_pen = 0), decreasing deeper
        surface_factor = max(0, 1.0 - along_pen / (depth * 0.5)) if along_pen > 0 else 1.0
        
        displacement = depth * edge_factor * cross_profile * surface_factor
        
        if displacement > 0.0001:
            # Apply displacement in penetration direction
            new_world = v_world + pen_dir * displacement
            # Convert back to local space
            v.co = mw_inv @ new_world
            vertices_modified += 1
    
    # Update mesh
    bmesh.update_edit_mesh(me)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Note: We no longer apply a global SMOOTH modifier as it affects
    # areas outside the groove. The falloff parameter now only controls
    # the edge transition within the groove itself, not post-processing.
    # The deformation already creates smooth edges via the falloff calculation.
    
    print(f"  [{loc}] Modified {vertices_modified} vertices")


def _create_flat_pocket_cutter(name, center, long_axis, pen_dir, long_len,
                               width, depth, overshoot=0.0):
    """
    Create a flat rectangular pocket cutter.

    Args:
        name: Object name
        center: Position at the groove OPENING on the surface (Vector)
        long_axis: Direction along the groove length
        pen_dir: Penetration direction (into the surface)
        long_len: Length of the groove
        width: Width of the groove
        depth: Depth of the pocket (measured inward from `center`)
        overshoot: Extra length past the surface for a clean cut

    Returns:
        Blender object or None
    """
    # Create a cube and transform it
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    cutter = bpy.context.active_object
    cutter.name = name

    # Scale to match dimensions
    cutter.scale = (long_len, width, depth + overshoot)
    
    # Build rotation matrix to orient the cube
    long_axis = long_axis.normalized()
    pen_dir = pen_dir.normalized()
    cross = long_axis.cross(pen_dir).normalized()
    
    # Rotation matrix: X=long, Y=cross, Z=pen
    rot_mat = Matrix((
        (long_axis.x, cross.x, pen_dir.x),
        (long_axis.y, cross.y, pen_dir.y),
        (long_axis.z, cross.z, pen_dir.z)
    )).transposed()
    
    cutter.rotation_euler = rot_mat.to_euler()
    # Offset inward so the pocket spans [center - overshoot, center + depth]
    # along pen_dir (same convention as the U-groove cutter). Centering on
    # the surface would carve only half the requested depth.
    cutter.location = center + pen_dir * (depth / 2 - overshoot / 2)

    # Apply transforms
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    return cutter


def _create_u_groove_cutter(name, center, long_axis, thickness_axis, pen_dir, 
                            long_len, radius, depth, overshoot, cyl_segments=32):
    """
    Create a U-groove cutter (box + cylinder for rounded bottom).
    
    Args:
        name: Object name
        center: Center position at the groove opening (Vector)
        long_axis: Direction along the groove length
        thickness_axis: Direction across the groove width
        pen_dir: Penetration direction (into the surface)
        long_len: Length of the groove
        radius: Radius of the rounded bottom (also half the groove width)
        depth: Total depth of the groove
        overshoot: Extra penetration to ensure clean cut
        cyl_segments: Number of segments for cylinder
        
    Returns:
        Blender object or None
    """
    parts = []
    
    # Normalize vectors
    long_axis = long_axis.normalized()
    thickness_axis = thickness_axis.normalized()
    pen_dir = pen_dir.normalized()
    
    # Box part (walls) - from surface to (depth - radius)
    box_depth = max(depth - radius, 0.0)
    if box_depth > 1e-9:
        box_center = center + pen_dir * (box_depth / 2 - overshoot / 2)
        
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        box = bpy.context.active_object
        box.name = f"{name}_BOX"
        box.scale = (long_len, 2 * radius, box_depth + overshoot)
        
        # Orient the box
        rot_mat = Matrix((
            (long_axis.x, thickness_axis.x, pen_dir.x),
            (long_axis.y, thickness_axis.y, pen_dir.y),
            (long_axis.z, thickness_axis.z, pen_dir.z)
        )).transposed()
        
        box.rotation_euler = rot_mat.to_euler()
        box.location = box_center
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        parts.append(box)
    
    # Cylinder (rounded bottom)
    cyl_center = center + pen_dir * (depth - radius)
    
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=long_len, vertices=cyl_segments)
    cyl = bpy.context.active_object
    cyl.name = f"{name}_CYL"
    
    # Orient cylinder so its axis is along long_axis
    # Default cylinder axis is Z, we want it along long_axis
    z_to_long = Vector((0, 0, 1)).rotation_difference(long_axis)
    cyl.rotation_quaternion = z_to_long
    cyl.rotation_mode = 'XYZ'
    cyl.rotation_euler = z_to_long.to_euler()
    cyl.location = cyl_center
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    parts.append(cyl)
    
    # Join parts into single object
    if len(parts) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for p in parts:
            p.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        bpy.ops.object.join()
        cutter = bpy.context.active_object
        cutter.name = name
        return cutter
    elif parts:
        parts[0].name = name
        return parts[0]
    
    return None


