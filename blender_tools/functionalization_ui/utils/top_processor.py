# ============================================================================
# Top Processor - Cabinet Missing Top Detection and Addition
# ============================================================================
# Utility module for detecting missing tops on cabinet-like meshes and adding
# top panels when needed. Based on rasterized top-down coverage detection
# and slicing-based shape generation.
#
# Adapted from: top_detect_fix/fix_missing_top_raster_detect_slice_shape.py
# ============================================================================

import bpy
import bmesh
from mathutils import Vector, geometry
from collections import deque


# ----------------------------
# Axis + small helpers
# ----------------------------

def parse_signed_axis(s: str):
    """Parse axis string like '+Z', '-Y' etc. into index, sign, and vector."""
    s = s.strip().upper()
    sign = +1
    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("-"):
        sign = -1
        s = s[1:]
    mapping = {"X": 0, "Y": 1, "Z": 2}
    if s not in mapping:
        raise ValueError("up_axis must be one of +X,+Y,+Z,-X,-Y,-Z")
    idx = mapping[s]
    up = Vector((0.0, 0.0, 0.0))
    up[idx] = float(sign)
    return idx, sign, up


def plane_axes_from_up(up_idx: int):
    """Get the two horizontal plane axes given the up axis index."""
    axes = [0, 1, 2]
    axes.remove(up_idx)
    return axes[0], axes[1]


def bbox_minmax_local(obj, ax):
    """Get min and max values along an axis from object's bounding box."""
    vals = [v[ax] for v in obj.bound_box]
    return float(min(vals)), float(max(vals))


def bbox_extents_local(obj):
    """Get bounding box extents (width, height, depth) in local space."""
    xs = [v[0] for v in obj.bound_box]
    ys = [v[1] for v in obj.bound_box]
    zs = [v[2] for v in obj.bound_box]
    return (max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))


def bbox_plane_area_local(obj, ax1, ax2):
    """Get the bounding box area in the plane defined by two axes."""
    ext = bbox_extents_local(obj)
    return max(1e-12, float(ext[ax1]) * float(ext[ax2]))


def obj_height_along_axis_local(obj, up_idx):
    """Get object height along the specified up axis."""
    mn, mx = bbox_minmax_local(obj, up_idx)
    return max(1e-12, mx - mn)


def percentile(sorted_vals, q):
    """Calculate percentile value from sorted list."""
    if not sorted_vals:
        return 0.0
    q = min(1.0, max(0.0, float(q)))
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(n - 1, lo + 1)
    t = pos - lo
    return float(sorted_vals[lo] * (1.0 - t) + sorted_vals[hi] * t)


def polygon_signed_area_2d(pts):
    """Calculate signed area of a 2D polygon."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % n]
        a += x1*y2 - x2*y1
    return 0.5 * a


def polygon_area_centroid_2d(poly):
    """
    Area centroid (Cx, Cy) of a simple polygon (not self-intersecting).
    poly: list of (x,y) WITHOUT repeating the first point at the end.
    Returns (cx, cy, signed_area).
    Falls back to vertex-mean if area is tiny.
    """
    n = len(poly)
    if n < 3:
        return (0.0, 0.0, 0.0)

    A2 = 0.0  # 2*area (signed)
    Cx6 = 0.0 # 6*A*Cx (signed)
    Cy6 = 0.0 # 6*A*Cy (signed)

    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A2 += cross
        Cx6 += (x0 + x1) * cross
        Cy6 += (y0 + y1) * cross

    # Signed area
    A = 0.5 * A2

    if abs(A2) < 1e-18:
        # Degenerate: fallback to vertex mean
        cx = sum(p[0] for p in poly) / max(1, n)
        cy = sum(p[1] for p in poly) / max(1, n)
        return (cx, cy, A)

    cx = Cx6 / (3.0 * A2)
    cy = Cy6 / (3.0 * A2)
    return (cx, cy, A)


def aabb_area_2d(points):
    """Calculate axis-aligned bounding box area of 2D points."""
    if not points:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(0.0, (max(xs)-min(xs))*(max(ys)-min(ys)))


# ----------------------------
# Raster detection (top-down coverage)
# ----------------------------

def new_mask(w, h, val=0):
    """Create a new 2D mask array."""
    return [bytearray([val])*w for _ in range(h)]


def tri_area2(a, b, c):
    """Calculate twice the signed area of triangle abc."""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def point_in_tri(p, a, b, c):
    """Check if point p is inside triangle abc."""
    s1 = tri_area2(p, a, b)
    s2 = tri_area2(p, b, c)
    s3 = tri_area2(p, c, a)
    has_neg = (s1 < 0) or (s2 < 0) or (s3 < 0)
    has_pos = (s1 > 0) or (s2 > 0) or (s3 > 0)
    return not (has_neg and has_pos)


def barycentric(p, a, b, c):
    """Calculate barycentric coordinates of point p in triangle abc."""
    denom = tri_area2(a, b, c)
    if abs(denom) < 1e-20:
        return None
    w0 = tri_area2(p, b, c) / denom
    w1 = tri_area2(p, c, a) / denom
    w2 = 1.0 - w0 - w1
    return (w0, w1, w2)


def build_topdown_buffers(obj, up_idx, up_vec, grid_res, padding_ratio, up_dot_thresh):
    """
    Build top-down coverage buffers.
    
    occ[y][x] = 1 if any triangle projects into the cell
    z_upmax[y][x] = max up-coordinate of *up-facing* triangles at that pixel
    """
    ax1, ax2 = plane_axes_from_up(up_idx)

    xmn, xmx = bbox_minmax_local(obj, ax1)
    ymn, ymx = bbox_minmax_local(obj, ax2)
    span = max(xmx-xmn, ymx-ymn, 1e-9)
    pad = float(padding_ratio) * span
    xmn -= pad; xmx += pad
    ymn -= pad; ymx += pad

    R = max(64, int(grid_res))
    cell = max((xmx-xmn)/R, (ymx-ymn)/R, 1e-9)
    W = max(64, int((xmx-xmn)/cell) + 1)
    H = max(64, int((ymx-ymn)/cell) + 1)

    occ = new_mask(W, H, 0)
    z_upmax = [[-1e30]*W for _ in range(H)]

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()

    bm_tmp = bm.copy()
    try:
        bmesh.ops.triangulate(bm_tmp, faces=bm_tmp.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    except TypeError:
        bmesh.ops.triangulate(bm_tmp, faces=bm_tmp.faces[:])

    for f in bm_tmp.faces:
        if len(f.verts) != 3:
            continue
        v0, v1, v2 = (f.verts[0].co, f.verts[1].co, f.verts[2].co)

        a = (float(v0[ax1]), float(v0[ax2]))
        b = (float(v1[ax1]), float(v1[ax2]))
        c = (float(v2[ax1]), float(v2[ax2]))
        if abs(tri_area2(a, b, c)) < 1e-18:
            continue

        is_up = (f.normal.dot(up_vec) >= float(up_dot_thresh))

        xmin = min(a[0], b[0], c[0]); xmax = max(a[0], b[0], c[0])
        ymin = min(a[1], b[1], c[1]); ymax = max(a[1], b[1], c[1])

        ix0 = max(0, min(W-1, int((xmin - xmn)/cell) - 1))
        ix1 = max(0, min(W-1, int((xmax - xmn)/cell) + 1))
        iy0 = max(0, min(H-1, int((ymin - ymn)/cell) - 1))
        iy1 = max(0, min(H-1, int((ymax - ymn)/cell) + 1))

        z0 = float(v0[up_idx]); z1 = float(v1[up_idx]); z2 = float(v2[up_idx])

        for iy in range(iy0, iy1+1):
            y = ymn + (iy + 0.5)*cell
            row_occ = occ[iy]
            row_z = z_upmax[iy]
            for ix in range(ix0, ix1+1):
                x = xmn + (ix + 0.5)*cell
                if not point_in_tri((x, y), a, b, c):
                    continue
                row_occ[ix] = 1
                if is_up:
                    bc = barycentric((x, y), a, b, c)
                    if bc is None:
                        continue
                    w0, w1, w2 = bc
                    z = w0*z0 + w1*z1 + w2*z2
                    if z > row_z[ix]:
                        row_z[ix] = z

    bm_tmp.free()
    bm.free()

    return occ, z_upmax, (xmn, xmx, ymn, ymx, cell, W, H, ax1, ax2)


def detect_missing_top(obj, up_axis="+Z", top_quantile=0.995, grid_res=512, 
                       grid_padding_ratio=0.02, up_dot_thresh=0.6, 
                       top_eps_ratio=0.01, coverage_thresh=0.6):
    """
    Detect if an object is missing a top surface using rasterized top-down coverage.
    
    Args:
        obj: Blender mesh object to analyze
        up_axis: Direction that is "up" (e.g., "+Z", "-Y")
        top_quantile: Quantile along up-axis used as top_coord
        grid_res: Top-down raster resolution
        grid_padding_ratio: Padding around bbox as fraction of span
        up_dot_thresh: Threshold for considering a triangle as up-facing
        top_eps_ratio: Thickness of "near-top band" as ratio of object height
        coverage_thresh: If coverage < this, top is considered missing
        
    Returns:
        tuple: (is_missing, detection_info_dict)
    """
    up_idx, up_sign, up_vec = parse_signed_axis(up_axis)
    
    # Pick top_coord from vertex quantile
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    coords = sorted(float(v.co[up_idx]) for v in bm.verts)
    bm.free()
    top_coord = percentile(coords, float(top_quantile)) if up_sign > 0 else percentile(coords, 1.0-float(top_quantile))

    height = obj_height_along_axis_local(obj, up_idx)
    top_eps = float(top_eps_ratio) * float(height)

    occ, z_upmax, grid = build_topdown_buffers(
        obj=obj, up_idx=up_idx, up_vec=up_vec,
        grid_res=grid_res, padding_ratio=grid_padding_ratio,
        up_dot_thresh=up_dot_thresh
    )
    _xmn, _xmx, _ymn, _ymx, cell, W, H, _ax1, _ax2 = grid

    footprint_cnt = 0
    top_cnt = 0

    def is_near_top(z):
        if z <= -1e20:
            return False
        return (z >= top_coord - top_eps) if up_sign > 0 else (z <= top_coord + top_eps)

    for y in range(H):
        occ_row = occ[y]
        z_row = z_upmax[y]
        for x in range(W):
            if occ_row[x]:
                footprint_cnt += 1
                if is_near_top(z_row[x]):
                    top_cnt += 1

    cell_area = cell*cell
    footprint_area = footprint_cnt * cell_area
    top_cell_area = top_cnt * cell_area
    coverage = top_cell_area / max(1e-12, footprint_area)
    missing = coverage < float(coverage_thresh)

    info = {
        "top_coord": top_coord,
        "top_eps": top_eps,
        "grid": {"W": W, "H": H, "cell": cell},
        "footprint_area": footprint_area,
        "top_cell_area": top_cell_area,
        "coverage": coverage,
        "coverage_thresh": coverage_thresh,
        "missing_by_coverage": missing,
    }
    return missing, info


# ----------------------------
# Slicing shape extraction (triangle-plane -> segments -> welded CCs)
# ----------------------------

def slice_segments_from_bmesh(bm, up_idx, slice_coord, tol):
    """Extract 2D slice segments from mesh at given height."""
    ax1, ax2 = plane_axes_from_up(up_idx)

    bm_tmp = bm.copy()
    try:
        bmesh.ops.triangulate(bm_tmp, faces=bm_tmp.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    except TypeError:
        bmesh.ops.triangulate(bm_tmp, faces=bm_tmp.faces[:])

    segs = []
    for f in bm_tmp.faces:
        if len(f.verts) != 3:
            continue
        p = [v.co.copy() for v in f.verts]
        d = [float(pi[up_idx] - slice_coord) for pi in p]

        # skip coplanar triangles
        if all(abs(di) <= tol for di in d):
            continue

        inter = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            d0, d1 = d[i], d[j]
            p0, p1 = p[i], p[j]

            if abs(d0) <= tol:
                inter.append(p0)
            if abs(d1) <= tol:
                inter.append(p1)

            if d0 * d1 < -(tol*tol):
                t = d0 / (d0 - d1)
                inter.append(p0.lerp(p1, t))

        if len(inter) < 2:
            continue

        # 2D dedup by tol grid
        uniq = {}
        inv = 1.0 / max(1e-30, tol)
        for pt in inter:
            k = (int(round(float(pt[ax1])*inv)), int(round(float(pt[ax2])*inv)))
            uniq[k] = pt
        inter = list(uniq.values())
        if len(inter) < 2:
            continue

        # choose farthest pair
        best_d2 = None
        best_pair = None
        for a in range(len(inter)):
            for b in range(a+1, len(inter)):
                dx = float(inter[a][ax1] - inter[b][ax1])
                dy = float(inter[a][ax2] - inter[b][ax2])
                d2 = dx*dx + dy*dy
                if best_d2 is None or d2 > best_d2:
                    best_d2 = d2
                    best_pair = (inter[a], inter[b])

        if best_pair is None or best_d2 is None or best_d2 < (tol*tol):
            continue

        a, b = best_pair
        segs.append(((float(a[ax1]), float(a[ax2])), (float(b[ax1]), float(b[ax2]))))

    bm_tmp.free()
    return segs


def weld_points_and_edges(segments_2d, weld_tol):
    """Weld segment endpoints and build edge graph."""
    if not segments_2d:
        return [], [], []

    pts = []
    key_to_idx = {}

    def key(x, y):
        inv = 1.0 / max(1e-30, float(weld_tol))
        return (int(round(x*inv)), int(round(y*inv)))

    def add(x, y):
        k = key(x, y)
        if k in key_to_idx:
            return key_to_idx[k]
        idx = len(pts)
        key_to_idx[k] = idx
        pts.append((x, y))
        return idx

    edges_set = set()
    seg_edges = []
    for (x0, y0), (x1, y1) in segments_2d:
        i0 = add(x0, y0)
        i1 = add(x1, y1)
        if i0 == i1:
            continue
        a, b = (i0, i1) if i0 < i1 else (i1, i0)
        edges_set.add((a, b))
        seg_edges.append((a, b))
    return pts, list(edges_set), seg_edges


def connected_components(num_pts, edges):
    """Find connected components in edge graph."""
    adj = [[] for _ in range(num_pts)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    seen = [False]*num_pts
    comps = []
    for i in range(num_pts):
        if seen[i] or not adj[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def segments_for_component(seg_edges, pts, comp_set):
    """Get segments belonging to a component."""
    segs = []
    for a, b in seg_edges:
        if a in comp_set and b in comp_set:
            segs.append((pts[a], pts[b]))
    return segs


# ----------------------------
# Footprint construction (from points)
# ----------------------------

def compute_min_area_rect(poly):
    """Compute minimum area bounding rectangle of polygon."""
    if len(poly) < 3:
        return []
    best = None
    best_rect = None
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % len(poly)]
        dx, dy = x2-x1, y2-y1
        L = (dx*dx+dy*dy)**0.5
        if L < 1e-12:
            continue
        ux, uy = dx/L, dy/L
        vx, vy = -uy, ux
        us = [px*ux+py*uy for px, py in poly]
        vs = [px*vx+py*vy for px, py in poly]
        umin, umax = min(us), max(us)
        vmin, vmax = min(vs), max(vs)
        area = (umax-umin)*(vmax-vmin)
        if best is None or area < best:
            best = area
            corners_uv = [(umin, vmin), (umax, vmin), (umax, vmax), (umin, vmax)]
            corners = []
            for u, v in corners_uv:
                x = u*ux+v*vx
                y = u*uy+v*vy
                corners.append((x, y))
            best_rect = corners
    if not best_rect:
        return []
    if polygon_signed_area_2d(best_rect) < 0:
        best_rect = list(reversed(best_rect))
    return best_rect


def footprint_from_points(points, mode):
    """
    Build footprint polygon from slice points.
    
    Args:
        points: List of 2D points
        mode: "min_area_rect", "convex_hull", or "aabb"
    """
    if len(points) < 3:
        return []
    if mode == "convex_hull":
        idx = geometry.convex_hull_2d(points)
        poly = [points[i] for i in idx]
        if polygon_signed_area_2d(poly) < 0:
            poly = list(reversed(poly))
        return poly
    if mode == "min_area_rect":
        idx = geometry.convex_hull_2d(points)
        hull = [points[i] for i in idx]
        return compute_min_area_rect(hull)
    if mode == "aabb":
        xs = [x for x, _ in points]; ys = [y for _, y in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    raise ValueError(f"Unknown footprint_mode: {mode}")


# ----------------------------
# Shape transformation functions
# ----------------------------

import math

def get_rect_local_frame(poly_2d):
    """
    Extract local coordinate frame from a (potentially rotated) rectangle polygon.
    
    For a 4-vertex rectangle, returns:
    - center (cx, cy)
    - width and height (w, h) - width is the longer dimension
    - unit vectors (ux, uy) for local X axis and (vx, vy) for local Y axis
    
    For non-4-vertex polygons, falls back to axis-aligned bounding box.
    """
    n = len(poly_2d)
    
    if n == 4:
        # Assume it's a rectangle - compute local frame from edges
        p0, p1, p2, p3 = poly_2d[0], poly_2d[1], poly_2d[2], poly_2d[3]
        
        # Center
        cx = sum(p[0] for p in poly_2d) / 4
        cy = sum(p[1] for p in poly_2d) / 4
        
        # Edge vectors
        e1x, e1y = p1[0] - p0[0], p1[1] - p0[1]
        e2x, e2y = p2[0] - p1[0], p2[1] - p1[1]
        
        # Edge lengths
        len1 = math.sqrt(e1x*e1x + e1y*e1y)
        len2 = math.sqrt(e2x*e2x + e2y*e2y)
        
        if len1 < 1e-9 or len2 < 1e-9:
            # Degenerate, fall back to AABB
            return get_rect_local_frame_aabb(poly_2d)
        
        # Normalize
        e1x, e1y = e1x/len1, e1y/len1
        e2x, e2y = e2x/len2, e2y/len2
        
        # width = longer edge, height = shorter edge
        if len1 >= len2:
            w, h = len1, len2
            ux, uy = e1x, e1y
            vx, vy = e2x, e2y
        else:
            w, h = len2, len1
            ux, uy = e2x, e2y
            vx, vy = e1x, e1y
        
        return cx, cy, w, h, ux, uy, vx, vy
    else:
        return get_rect_local_frame_aabb(poly_2d)


def get_rect_local_frame_aabb(poly_2d):
    """Get local frame from axis-aligned bounding box."""
    xs = [p[0] for p in poly_2d]
    ys = [p[1] for p in poly_2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    
    # Axis-aligned: local X = world X, local Y = world Y
    return cx, cy, w, h, 1.0, 0.0, 0.0, 1.0


def local_to_world(lx, ly, cx, cy, ux, uy, vx, vy):
    """Transform local coordinates to world coordinates."""
    wx = cx + lx * ux + ly * vx
    wy = cy + lx * uy + ly * vy
    return wx, wy


def transform_to_rounded_rect(poly_2d, corner_radius_frac=0.15, segments=8):
    """
    Transform rectangular polygon to rounded rectangle that CIRCUMSCRIBES the original.
    The rounded corners bulge outward to ensure full coverage of the original rectangle.
    Works with rotated rectangles by operating in local coordinates.
    """
    cx, cy, w, h, ux, uy, vx, vy = get_rect_local_frame(poly_2d)
    
    # Half dimensions
    hw, hh = w / 2, h / 2
    
    # Corner radius based on smaller dimension
    min_dim = min(w, h)
    r = min_dim * corner_radius_frac
    
    if r <= 0.001:
        return poly_2d
    
    points = []
    
    # For circumscribing: corners bulge OUTWARD
    # Corner arc centers are at the original corners, arcs extend outward
    # This ensures the rounded shape fully contains the rectangle
    corners_local = [
        (-hw, -hh, math.pi, 1.5 * math.pi),      # Bottom-left corner
        (hw, -hh, 1.5 * math.pi, 2 * math.pi),   # Bottom-right corner
        (hw, hh, 0, 0.5 * math.pi),              # Top-right corner
        (-hw, hh, 0.5 * math.pi, math.pi),       # Top-left corner
    ]
    
    for lcx, lcy, start_angle, end_angle in corners_local:
        for i in range(segments + 1):
            t = i / segments
            angle = start_angle + t * (end_angle - start_angle)
            # Arc extends outward from the corner
            lx = lcx + r * math.cos(angle)
            ly = lcy + r * math.sin(angle)
            wx, wy = local_to_world(lx, ly, cx, cy, ux, uy, vx, vy)
            points.append((wx, wy))
    
    return points


def transform_to_oval(poly_2d, segments=32):
    """
    Transform rectangular polygon to oval/ellipse that CIRCUMSCRIBES the rectangle.
    The ellipse passes through all 4 corners, fully containing the rectangle.
    Works with rotated rectangles.
    """
    cx, cy, w, h, ux, uy, vx, vy = get_rect_local_frame(poly_2d)
    
    # For an ellipse to pass through corners (±w/2, ±h/2):
    # With same aspect ratio (a/b = w/h), scale factor is sqrt(2)
    # Semi-axes of circumscribed ellipse
    scale = math.sqrt(2)
    rx, ry = (w / 2) * scale, (h / 2) * scale
    
    points = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        # Local coordinates on ellipse
        lx = rx * math.cos(angle)
        ly = ry * math.sin(angle)
        # Transform to world
        wx, wy = local_to_world(lx, ly, cx, cy, ux, uy, vx, vy)
        points.append((wx, wy))
    
    return points


def transform_to_capsule(poly_2d, segments=16):
    """
    Transform rectangular polygon to capsule/stadium shape that MINIMALLY circumscribes the rectangle.
    The capsule is the smallest stadium shape that fully contains the rectangle.
    Works with rotated rectangles.
    
    For a rectangle w×h (w >= h):
    - Semicircle radius = h/2 (matches the short dimension)
    - Straight section length = w - h
    - Top/bottom of capsule align with top/bottom of rectangle
    - Semicircles extend beyond left/right edges by (h/2) each
    """
    cx, cy, w, h, ux, uy, vx, vy = get_rect_local_frame(poly_2d)
    
    hw, hh = w / 2, h / 2
    
    # Minimal circumscribing capsule:
    # - Radius equals half the shorter dimension
    # - Straight section spans the difference
    r = hh  # Semicircle radius = half height
    straight_half = hw  # Semicircle centers at ±hw
    
    # This creates a capsule where:
    # - Top edge at y = +hh (matches rectangle)
    # - Bottom edge at y = -hh (matches rectangle)  
    # - Left end extends to x = -(hw + r) = -(hw + hh)
    # - Right end extends to x = +(hw + r) = +(hw + hh)
    # The rectangle is fully contained with minimal extra area
    
    points = []
    
    # Left semicircle (centered at -hw, 0)
    for i in range(segments + 1):
        angle = math.pi / 2 + math.pi * i / segments
        lx = -straight_half + r * math.cos(angle)
        ly = r * math.sin(angle)
        wx, wy = local_to_world(lx, ly, cx, cy, ux, uy, vx, vy)
        points.append((wx, wy))
    
    # Right semicircle (centered at +hw, 0)
    for i in range(segments + 1):
        angle = -math.pi / 2 + math.pi * i / segments
        lx = straight_half + r * math.cos(angle)
        ly = r * math.sin(angle)
        wx, wy = local_to_world(lx, ly, cx, cy, ux, uy, vx, vy)
        points.append((wx, wy))
    
    return points


def transform_to_chamfered(poly_2d, chamfer_frac=0.15):
    """
    Transform rectangular polygon to chamfered rectangle that CIRCUMSCRIBES the original.
    The chamfered edges extend outward at 45° from corners.
    Works with rotated rectangles.
    """
    cx, cy, w, h, ux, uy, vx, vy = get_rect_local_frame(poly_2d)
    
    hw, hh = w / 2, h / 2
    
    # Chamfer size based on smaller dimension
    min_dim = min(w, h)
    c = min_dim * chamfer_frac
    
    if c <= 0.001:
        return poly_2d
    
    # For circumscribing: the chamfered edges extend OUTWARD
    # 8 points in local coordinates (counter-clockwise from bottom-left)
    # Each corner has a 45° cut that extends outward
    local_points = [
        (-hw - c, -hh),      # Left of bottom-left corner (extended)
        (-hw, -hh - c),      # Below bottom-left corner (extended)
        (hw, -hh - c),       # Below bottom-right corner (extended)
        (hw + c, -hh),       # Right of bottom-right corner (extended)
        (hw + c, hh),        # Right of top-right corner (extended)
        (hw, hh + c),        # Above top-right corner (extended)
        (-hw, hh + c),       # Above top-left corner (extended)
        (-hw - c, hh),       # Left of top-left corner (extended)
    ]
    
    # Transform to world coordinates
    points = []
    for lx, ly in local_points:
        wx, wy = local_to_world(lx, ly, cx, cy, ux, uy, vx, vy)
        points.append((wx, wy))
    
    return points


def apply_shape_style(poly_2d, shape_style, corner_radius=0.15, corner_segments=8):
    """
    Apply shape style transformation to polygon.
    
    All non-rectangle shapes CIRCUMSCRIBE the original footprint rectangle,
    ensuring the top fully covers the cabinet without any gaps or holes.
    Works with both axis-aligned and rotated rectangles.
    
    Args:
        poly_2d: Original polygon points (typically rectangular)
        shape_style: 'RECTANGLE', 'ROUNDED', 'OVAL', 'CAPSULE', 'CHAMFERED'
        corner_radius: Radius fraction for rounded/chamfered corners
        corner_segments: Number of segments for curved parts
    
    Returns:
        Transformed polygon points (circumscribing the original bounds)
    """
    if shape_style == 'RECTANGLE':
        return poly_2d
    elif shape_style == 'ROUNDED':
        return transform_to_rounded_rect(poly_2d, corner_radius, corner_segments)
    elif shape_style == 'OVAL':
        return transform_to_oval(poly_2d, corner_segments * 4)
    elif shape_style == 'CAPSULE':
        return transform_to_capsule(poly_2d, corner_segments * 2)
    elif shape_style == 'CHAMFERED':
        return transform_to_chamfered(poly_2d, corner_radius)
    else:
        return poly_2d


# ----------------------------
# Add top from footprint polygon
# ----------------------------

def weld_mesh(bm, weld_dist):
    """Weld mesh vertices within distance."""
    d = float(weld_dist)
    if d <= 0:
        return
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=d)


def make_top_from_footprint(bm, up_idx, up_vec, top_coord, poly_2d, overhang_pct, thickness, extrude_dir,
                            shape_style='RECTANGLE', corner_radius=0.15, corner_segments=8):
    """
    Create top geometry from footprint polygon with optional shape transformation.
    
    Args:
        bm: BMesh to add geometry to
        up_idx: Index of up axis (0=X, 1=Y, 2=Z)
        up_vec: Up vector
        top_coord: Z coordinate for top surface
        poly_2d: 2D polygon points (footprint)
        overhang_pct: Overhang fraction per side
        thickness: Top thickness
        extrude_dir: 'outward' or 'inward'
        shape_style: 'RECTANGLE', 'ROUNDED', 'OVAL', 'CAPSULE', 'CHAMFERED'
        corner_radius: Radius fraction for rounded corners
        corner_segments: Number of segments for curves
    """
    if len(poly_2d) < 3:
        return False
    ax1, ax2 = plane_axes_from_up(up_idx)
    
    # Apply overhang first (scale the base polygon)
    if float(overhang_pct) != 0.0:
        s = 1.0 + 2.0*float(overhang_pct)
        cx, cy, _ = polygon_area_centroid_2d(poly_2d)
        poly_2d = [(cx + (x - cx) * s, cy + (y - cy) * s) for x, y in poly_2d]
    
    # Apply shape transformation
    shaped_poly = apply_shape_style(poly_2d, shape_style, corner_radius, corner_segments)
    
    if len(shaped_poly) < 3:
        return False

    new_verts = []
    for x, y in shaped_poly:
        co = Vector((0, 0, 0))
        co[ax1], co[ax2], co[up_idx] = float(x), float(y), float(top_coord)
        new_verts.append(bm.verts.new(co))
    bm.verts.ensure_lookup_table()

    try:
        face = bm.faces.new(new_verts)
    except ValueError:
        return False

    bm.normal_update()
    if face.normal.dot(up_vec) < 0.0:
        face.normal_flip()

    ext = bmesh.ops.extrude_face_region(bm, geom=[face])
    ext_verts = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]
    vec = (+up_vec if extrude_dir == "outward" else -up_vec) * float(thickness)
    bmesh.ops.translate(bm, verts=ext_verts, vec=vec)
    return True


def add_top_to_object(obj, up_axis="+Z", top_coord=None, footprint_area=None,
                      slice_delta_ratio=0.001, slice_tol_ratio=0.001,
                      slice_weld_ratio=0.002, footprint_cc_select="aabb",
                      footprint_cc_keep_rel=0.2, footprint_cc_keep_abs_frac=0.0,
                      footprint_mode="min_area_rect", overhang_pct=0.03,
                      thickness_abs=None, thickness_ratio=0.02, weld_dist=0.0,
                      extrude_dir="outward", min_generated_top_area_frac=0.55,
                      shape_style='RECTANGLE', corner_radius=0.15, corner_segments=8,
                      create_separate=True):
    """
    Add a top panel to the given object using slicing-based shape extraction.
    
    Args:
        obj: Blender mesh object
        up_axis: Direction that is "up" (e.g., "+Z", "-Y")
        top_coord: Height coordinate for the top (if None, will be calculated)
        footprint_area: Expected footprint area (for size validation)
        slice_delta_ratio: How far below top_coord to slice
        slice_tol_ratio: Triangle-plane intersection tolerance
        slice_weld_ratio: Endpoint weld tolerance
        footprint_cc_select: How to select CCs ("aabb" or "fallback")
        footprint_cc_keep_rel: Keep CC if area >= keep_rel * max_area
        footprint_cc_keep_abs_frac: Keep CC if area frac >= this
        footprint_mode: "min_area_rect", "convex_hull", or "aabb"
        overhang_pct: Overhang fraction per side
        thickness_abs: Absolute thickness (overrides ratio if set)
        thickness_ratio: Top thickness as ratio of height
        weld_dist: Merge-by-distance before processing
        extrude_dir: "outward" or "inward"
        min_generated_top_area_frac: Reject if top area < this * footprint_area
        
    Returns:
        tuple: (success, info_dict)
    """
    up_idx, up_sign, up_vec = parse_signed_axis(up_axis)
    
    me = obj.data
    height = obj_height_along_axis_local(obj, up_idx)
    
    # Calculate top_coord if not provided
    if top_coord is None:
        bm_temp = bmesh.new()
        bm_temp.from_mesh(me)
        coords = sorted(float(v.co[up_idx]) for v in bm_temp.verts)
        bm_temp.free()
        top_coord = percentile(coords, 0.995) if up_sign > 0 else percentile(coords, 0.005)
    
    # Calculate footprint_area if not provided
    if footprint_area is None:
        ax1, ax2 = plane_axes_from_up(up_idx)
        footprint_area = bbox_plane_area_local(obj, ax1, ax2)

    # Slice plane (below top)
    slice_delta = float(slice_delta_ratio) * float(height)
    slice_coord = float(top_coord) - float(up_sign) * slice_delta

    slice_tol = max(1e-9, float(slice_tol_ratio) * float(height))
    slice_weld_tol = max(slice_tol, float(slice_weld_ratio) * float(height))

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.normal_update()
    weld_mesh(bm, weld_dist=weld_dist)

    segs_all = slice_segments_from_bmesh(bm, up_idx=up_idx, slice_coord=slice_coord, tol=slice_tol)
    pts_weld, edges, seg_edges = weld_points_and_edges(segs_all, weld_tol=slice_weld_tol)
    comps = connected_components(len(pts_weld), edges)

    if not pts_weld or not comps:
        bm.free()
        return False, {"reason": "no_slice_components", "slice_coord": slice_coord, "segments": len(segs_all)}

    # CC selection (default: AABB extent based)
    ax1, ax2 = plane_axes_from_up(up_idx)
    bbox_area = bbox_plane_area_local(obj, ax1, ax2)

    cc_aabb_areas = []
    cc_segments = []
    for comp in comps:
        pts_cc = [pts_weld[i] for i in comp]
        cc_aabb_areas.append(aabb_area_2d(pts_cc))
        cc_segments.append(segments_for_component(seg_edges, pts_weld, set(comp)))

    max_aabb = max(cc_aabb_areas) if cc_aabb_areas else 0.0

    chosen_points = []
    keep_stats = []
    kept = 0
    for ci, comp in enumerate(comps):
        pts_cc = [pts_weld[i] for i in comp]
        aabb = cc_aabb_areas[ci]
        frac_aabb = aabb / max(1e-12, bbox_area)

        if footprint_cc_select == "aabb":
            keep = (frac_aabb >= float(footprint_cc_keep_abs_frac)) or (max_aabb > 0.0 and aabb >= float(footprint_cc_keep_rel) * max_aabb)
        else:
            # fallback: keep anything with non-trivial extent
            keep = frac_aabb >= max(1e-6, float(footprint_cc_keep_abs_frac))

        keep_stats.append({"ci": ci, "aabb_area_frac": frac_aabb, "kept": keep, "points": len(pts_cc), "segments": len(cc_segments[ci])})
        if keep:
            chosen_points.extend(pts_cc)
            kept += 1

    if len(chosen_points) < 3:
        chosen_points = list(pts_weld)

    footprint_poly = footprint_from_points(chosen_points, mode=footprint_mode)
    if len(footprint_poly) < 3:
        footprint_poly = footprint_from_points(chosen_points, mode="aabb")

    # Ornamentation guard: reject tiny "tops"
    top_poly_area = abs(polygon_signed_area_2d(footprint_poly))
    area_frac = top_poly_area / max(1e-12, float(footprint_area))

    if area_frac < float(min_generated_top_area_frac):
        bm.free()
        return False, {
            "reason": "generated_top_too_small_vs_footprint",
            "top_poly_area": top_poly_area,
            "footprint_area": float(footprint_area),
            "area_frac": area_frac,
            "min_required_frac": float(min_generated_top_area_frac),
            "slice_coord": slice_coord,
            "segments": len(segs_all),
            "cc": len(comps),
        }

    # Thickness
    if thickness_abs is not None:
        thickness = float(thickness_abs)
    else:
        thickness = float(thickness_ratio) * float(height)
    thickness = max(1e-6, thickness)

    # We don't need the bmesh for the base object if creating separate
    bm.free()
    
    if create_separate:
        # Create top as a separate object
        ok, top_obj = create_top_as_separate_object(
            base_obj=obj,
            up_axis=up_axis,
            top_coord=float(top_coord),
            poly_2d=footprint_poly,
            overhang_pct=float(overhang_pct),
            thickness=thickness,
            extrude_dir=extrude_dir,
            shape_style=shape_style,
            corner_radius=corner_radius,
            corner_segments=corner_segments,
        )
    else:
        # Create top merged into the base object (old behavior)
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.normal_update()
        
        ok = make_top_from_footprint(
            bm=bm,
            up_idx=up_idx,
            up_vec=up_vec,
            top_coord=float(top_coord),
            poly_2d=footprint_poly,
            overhang_pct=float(overhang_pct),
            thickness=thickness,
            extrude_dir=extrude_dir,
            shape_style=shape_style,
            corner_radius=corner_radius,
            corner_segments=corner_segments,
        )

        if ok:
            # Triangulate for compatibility
            try:
                bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
            except TypeError:
                bmesh.ops.triangulate(bm, faces=bm.faces[:])

            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            me.update()

        bm.free()

    return ok, {
        "slice_coord": slice_coord,
        "slice_tol": slice_tol,
        "slice_weld_tol": slice_weld_tol,
        "segments": len(segs_all),
        "cc": len(comps),
        "kept_cc": kept,
        "footprint_mode": footprint_mode,
        "footprint_cc_select": footprint_cc_select,
        "thickness": thickness,
        "keep_stats": keep_stats,
        "created_separate": create_separate,
    }


def remove_existing_top_objects(base_obj_name):
    """
    Remove previously added top objects (created as separate objects).
    
    Looks for objects named "{base_obj_name}_Top" or "{base_obj_name}_Top.001" etc.
    
    Args:
        base_obj_name: Name of the base object
        
    Returns:
        int: Number of objects removed
    """
    import bpy
    
    removed_count = 0
    objects_to_remove = []
    
    # Find all top objects for this base
    for obj in bpy.data.objects:
        # Match pattern: "BaseName_Top" or "BaseName_Top.001" etc.
        if obj.name.startswith(f"{base_obj_name}_Top"):
            objects_to_remove.append(obj)
        # Also check for generic "AddedTop_" prefix
        elif obj.name.startswith("AddedTop_"):
            objects_to_remove.append(obj)
    
    # Remove the objects
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
        removed_count += 1
    
    return removed_count


def create_top_as_separate_object(base_obj, up_axis, top_coord, poly_2d, overhang_pct, 
                                   thickness, extrude_dir, shape_style, corner_radius, 
                                   corner_segments):
    """
    Create top as a separate object (not merged into the base mesh).
    
    This allows easy removal and replacement of tops.
    
    Args:
        base_obj: The base object (cabinet) to create top for
        up_axis: Up axis string (e.g., "+Z")
        top_coord: Height coordinate for top placement
        poly_2d: 2D footprint polygon
        overhang_pct: Overhang fraction
        thickness: Top thickness
        extrude_dir: "outward" or "inward"
        shape_style: Shape style string
        corner_radius: Corner radius fraction
        corner_segments: Number of segments for curves
        
    Returns:
        tuple: (success, top_object or None)
    """
    import bpy
    
    up_idx, up_sign, up_vec = parse_signed_axis(up_axis)
    ax1, ax2 = plane_axes_from_up(up_idx)
    
    if len(poly_2d) < 3:
        return False, None
    
    # Apply overhang first (scale the base polygon)
    if float(overhang_pct) != 0.0:
        s = 1.0 + 2.0 * float(overhang_pct)
        cx, cy, _ = polygon_area_centroid_2d(poly_2d)
        poly_2d = [(cx + (x - cx) * s, cy + (y - cy) * s) for x, y in poly_2d]
    
    # Apply shape transformation
    shaped_poly = apply_shape_style(poly_2d, shape_style, corner_radius, corner_segments)
    
    if len(shaped_poly) < 3:
        return False, None
    
    # Create new mesh and object
    mesh = bpy.data.meshes.new(f"{base_obj.name}_Top_Mesh")
    top_obj = bpy.data.objects.new(f"{base_obj.name}_Top", mesh)
    
    # Link to scene
    bpy.context.collection.objects.link(top_obj)
    
    # Create BMesh for the top
    bm = bmesh.new()
    
    # Create vertices
    new_verts = []
    for x, y in shaped_poly:
        co = Vector((0, 0, 0))
        co[ax1], co[ax2], co[up_idx] = float(x), float(y), float(top_coord)
        new_verts.append(bm.verts.new(co))
    bm.verts.ensure_lookup_table()
    
    # Create face
    try:
        face = bm.faces.new(new_verts)
    except ValueError:
        bm.free()
        bpy.data.objects.remove(top_obj, do_unlink=True)
        return False, None
    
    bm.normal_update()
    if face.normal.dot(up_vec) < 0.0:
        face.normal_flip()
    
    # Extrude for thickness
    ext = bmesh.ops.extrude_face_region(bm, geom=[face])
    ext_verts = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]
    vec = (+up_vec if extrude_dir == "outward" else -up_vec) * float(thickness)
    bmesh.ops.translate(bm, verts=ext_verts, vec=vec)
    
    # Triangulate for compatibility
    try:
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    except TypeError:
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
    
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    
    # Write to mesh
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()
    
    # Copy material from base object if available
    if base_obj.data.materials:
        top_obj.data.materials.append(base_obj.data.materials[0])
    
    # Parent to base object (optional, for organization)
    # top_obj.parent = base_obj
    
    return True, top_obj


def detect_and_add_top(obj, up_axis="+Z", force_add=False, 
                       # Detection parameters
                       top_quantile=0.995, grid_res=512, grid_padding_ratio=0.02,
                       up_dot_thresh=0.6, top_eps_ratio=0.01, coverage_thresh=0.6,
                       # Top creation parameters
                       overhang_pct=0.03, thickness_abs=None, thickness_ratio=0.02,
                       footprint_mode="min_area_rect",
                       # Advanced parameters (match original script defaults)
                       min_generated_top_area_frac=0.3, slice_delta_ratio=0.001,
                       # Shape parameters
                       shape_style='RECTANGLE', corner_radius=0.15, corner_segments=8,
                       # Separate object creation
                       create_separate=True):
    """
    High-level function to detect if top is missing and add one if needed.
    
    Args:
        obj: Blender mesh object
        up_axis: Direction that is "up" (e.g., "+Z", "-Y")
        force_add: If True, add top even if not detected as missing
        min_generated_top_area_frac: Min top area as fraction of footprint (lower = allow smaller)
        slice_delta_ratio: How far below top to slice for shape detection
        (remaining args passed to detection and creation functions)
        
    Returns:
        dict: Result containing detection and creation info
    """
    result = {
        "object_name": obj.name,
        "up_axis": up_axis,
        "detection_performed": True,
        "missing_detected": False,
        "top_added": False,
        "detection_info": None,
        "creation_info": None,
        "error": None,
    }
    
    try:
        # First, detect if top is missing
        missing, det_info = detect_missing_top(
            obj=obj,
            up_axis=up_axis,
            top_quantile=top_quantile,
            grid_res=grid_res,
            grid_padding_ratio=grid_padding_ratio,
            up_dot_thresh=up_dot_thresh,
            top_eps_ratio=top_eps_ratio,
            coverage_thresh=coverage_thresh,
        )
        
        result["missing_detected"] = missing
        result["detection_info"] = det_info
        
        # Add top if missing or forced
        if missing or force_add:
            top_coord = det_info.get("top_coord")
            footprint_area = det_info.get("footprint_area")
            
            success, create_info = add_top_to_object(
                obj=obj,
                up_axis=up_axis,
                top_coord=top_coord,
                footprint_area=footprint_area,
                overhang_pct=overhang_pct,
                thickness_abs=thickness_abs,
                thickness_ratio=thickness_ratio,
                footprint_mode=footprint_mode,
                min_generated_top_area_frac=min_generated_top_area_frac,
                slice_delta_ratio=slice_delta_ratio,
                shape_style=shape_style,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
                create_separate=create_separate,
            )
            
            result["top_added"] = success
            result["creation_info"] = create_info
            
            if not success:
                result["error"] = create_info.get("reason", "Unknown error during top creation")
        
    except Exception as e:
        result["error"] = str(e)
        
    return result

