"""In-scene interior visualization + panel generation.

Interior panel instantiation for the graph workflow (the parts of the
interior pipeline
the graph workflow uses). Consumers:
  - utils/interior_graph.py  (detect_and_visualize draws compartment boxes)
  - operators/graph_operators.py  (generate / clear / commit panels)

Object conventions (stable so existing scenes
keep working):
  COMPARTMENT_NN         anchor empty + wire box + corner handles, tagged
                         is_compartment_box / _wire / _corner, with
                         compartment_min/max, compartment_id, side_ax/up_ax/front_ax
  INTERIOR_PANEL_CXX_PYY generated panels, tagged compartment_id/panel_index/orientation
  commit renames panels to shelf_* / divider_* and drops the boxes.
"""
from __future__ import annotations

import bpy
from mathutils import Vector

INTERIOR_COMPARTMENT_PREFIX = "COMPARTMENT_"
INTERIOR_PANEL_PREFIX = "INTERIOR_PANEL_"
INTERIOR_COLLECTION_NAME = "Interior_Components"

def _ensure_interior_collection():
    """Ensure the interior components collection exists."""
    if INTERIOR_COLLECTION_NAME not in bpy.data.collections:
        col = bpy.data.collections.new(INTERIOR_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(col)
    return bpy.data.collections[INTERIOR_COLLECTION_NAME]


def _parse_axis_to_index(axis_str):
    """Convert axis string like 'PosZ', '+Z', 'NegY' to (axis_index, sign)."""
    axis_str = axis_str.strip().upper()
    
    sign = 1
    if axis_str.startswith('NEG') or axis_str.startswith('-'):
        sign = -1
        axis_str = axis_str.replace('NEG', '').replace('-', '')
    elif axis_str.startswith('POS') or axis_str.startswith('+'):
        sign = 1
        axis_str = axis_str.replace('POS', '').replace('+', '')
    
    axis_str = axis_str.strip()
    if 'X' in axis_str:
        return 0, sign
    elif 'Y' in axis_str:
        return 1, sign
    elif 'Z' in axis_str:
        return 2, sign
    
    return 2, 1  # Default to Z


def _get_remaining_axis(ax1, ax2):
    """Get the remaining axis index given two axes."""
    for i in range(3):
        if i != ax1 and i != ax2:
            return i
    return 0


def _object_world_bounds(obj):
    """Get world-space bounding box of an object."""
    if not obj or obj.type != 'MESH':
        return None, None
    
    # Get world-space corners
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    
    return mn, mx


def _create_compartment_box(name, mn, mx, color_index=0):
    """
    Create a wireframe box representing a compartment with a center anchor for easy selection.
    
    Uses WIRE display type for see-through visualization that works in Workbench.
    Adds a small icosphere at the center as a clickable anchor point.
    """
    center = (mn + mx) * 0.5
    size = mx - mn
    
    # Distinct colors for compartments
    colors = [
        (0.2, 0.6, 1.0, 1.0),   # Blue
        (0.2, 1.0, 0.4, 1.0),   # Green
        (1.0, 0.6, 0.2, 1.0),   # Orange
        (1.0, 0.2, 0.6, 1.0),   # Pink
        (0.6, 0.2, 1.0, 1.0),   # Purple
        (1.0, 1.0, 0.2, 1.0),   # Yellow
        (0.2, 1.0, 1.0, 1.0),   # Cyan
        (1.0, 0.4, 0.4, 1.0),   # Red
    ]
    box_color = colors[color_index % len(colors)]
    
    # --- Create wireframe box ---
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    box_obj = bpy.context.active_object
    box_obj.name = f"{name}_wire"
    box_obj.scale = (size.x, size.y, size.z)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # Wireframe display
    box_obj.display_type = 'WIRE'
    box_obj.show_wire = True
    box_obj.show_all_edges = True
    box_obj.color = box_color
    box_obj['is_compartment_wire'] = True
    
    # --- Create center anchor (small icosphere for easy clicking) ---
    anchor_size = min(size.x, size.y, size.z) * 0.15  # 15% of smallest dimension
    anchor_size = max(anchor_size, 0.02)  # Minimum size
    
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2,
        radius=anchor_size,
        location=center
    )
    anchor_obj = bpy.context.active_object
    anchor_obj.name = name  # Main name for the anchor (this is what users select)
    anchor_obj.color = box_color
    anchor_obj.display_type = 'SOLID'  # Solid so it's visible and clickable
    anchor_obj.show_wire = True
    
    # --- Create corner markers for additional selection points ---
    # Small cubes at corners
    corner_size = min(size.x, size.y, size.z) * 0.08
    corner_size = max(corner_size, 0.015)
    
    corners = [
        Vector((mn.x, mn.y, mn.z)),
        Vector((mx.x, mn.y, mn.z)),
        Vector((mn.x, mx.y, mn.z)),
        Vector((mx.x, mx.y, mn.z)),
        Vector((mn.x, mn.y, mx.z)),
        Vector((mx.x, mn.y, mx.z)),
        Vector((mn.x, mx.y, mx.z)),
        Vector((mx.x, mx.y, mx.z)),
    ]
    
    corner_objs = []
    for ci, corner in enumerate(corners):
        bpy.ops.mesh.primitive_cube_add(size=corner_size, location=corner)
        corner_obj = bpy.context.active_object
        corner_obj.name = f"{name}_corner{ci}"
        corner_obj.color = box_color
        corner_obj.display_type = 'SOLID'
        corner_obj['is_compartment_corner'] = True
        corner_objs.append(corner_obj)
    
    # --- Parent everything to the anchor ---
    box_obj.parent = anchor_obj
    box_obj.matrix_parent_inverse = anchor_obj.matrix_world.inverted()
    
    for corner_obj in corner_objs:
        corner_obj.parent = anchor_obj
        corner_obj.matrix_parent_inverse = anchor_obj.matrix_world.inverted()
    
    # Store bounds as custom properties on anchor (the main object)
    anchor_obj['compartment_min'] = list(mn)
    anchor_obj['compartment_max'] = list(mx)
    anchor_obj['is_compartment_box'] = True
    anchor_obj['color_index'] = color_index
    
    return anchor_obj


def _create_panel_box(name, mn, mx, color=(0.8, 0.65, 0.45, 1.0)):
    """Create a solid panel box (shelf or divider)."""
    center = (mn + mx) * 0.5
    size = mx - mn
    
    # Create cube
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (size.x, size.y, size.z)
    
    # Apply scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # Create wood-like material
    mat_name = f"MAT_{name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color[:3] + (1.0,)
        bsdf.inputs['Roughness'].default_value = 0.7
    
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    
    return obj


def compute_equal_panels_in_aabb(
    mn, mx,
    orientation,
    num_panels,
    panel_thickness,
    side_ax,
    up_ax,
    front_inset=0.0
):
    """
    Compute N equally spaced panels inside an AABB.
    
    - orientation='horizontal' => shelves (thickness along up axis)
    - orientation='vertical'   => dividers (thickness along side axis)
    
    Returns list of (panel_min, panel_max) tuples.
    """
    N = int(num_panels)
    t = float(panel_thickness)
    if N <= 0 or t <= 0.0:
        return []
    
    mn = list(mn)
    mx = list(mx)
    
    ori = (orientation or "").strip().lower()
    if ori.startswith("h"):
        axis = up_ax  # Split along up axis for horizontal shelves
    elif ori.startswith("v"):
        axis = side_ax  # Split along side axis for vertical dividers
    else:
        axis = up_ax  # Default to horizontal
    
    span = float(mx[axis] - mn[axis])
    free = span - N * t
    if free <= 1e-9:
        return []
    
    gap = free / (N + 1)
    
    panels = []
    for i in range(N):
        panel_mn = mn.copy()
        panel_mx = mx.copy()
        
        # Position along the split axis
        pos = mn[axis] + gap * (i + 1) + t * i
        panel_mn[axis] = pos
        panel_mx[axis] = pos + t
        
        # Apply front inset if specified
        if front_inset > 0:
            # Determine front axis (the one not side or up)
            front_ax = _get_remaining_axis(side_ax, up_ax)
            # Inset from the max side (assuming front is at max)
            panel_mx[front_ax] = panel_mx[front_ax] - front_inset
        
        panels.append((Vector(panel_mn), Vector(panel_mx)))
    
    return panels



def generate_panels_for_compartments(
    compartment_ids=None,
    orientation='horizontal',
    num_panels=2,
    panel_thickness=0.018,
    front_inset=0.01,
    use_all=False,
    clear_all=False
):
    """
    Generate interior panels (shelves/dividers) for specified compartments.
    
    ACCUMULATIVE: Panels for different compartments accumulate. Only panels for
    the currently processed compartments are replaced (so you can regenerate
    with different settings). Use clear_all=True to start fresh.
    
    Args:
        compartment_ids: List of compartment IDs to fill, or None for selected/all
        orientation: 'horizontal' for shelves, 'vertical' for dividers
        num_panels: Number of panels per compartment
        panel_thickness: Thickness of each panel
        front_inset: Distance to inset panels from front
        use_all: If True, generate panels for ALL compartments (ignore selection)
        clear_all: If True, clear ALL existing panels first (fresh start)
        
    Returns:
        dict with 'success', 'panel_count', 'panel_objects'
    """
    print(f"\n=== Generating Interior Panels (Accumulative) ===")
    print(f"  Orientation: {orientation}")
    print(f"  Num panels per compartment: {num_panels}")
    print(f"  Thickness: {panel_thickness}")
    print(f"  Front inset: {front_inset}")
    print(f"  Use all: {use_all}")
    
    result = {
        'success': False,
        'panel_count': 0,
        'panel_objects': [],
        'compartments_processed': 0,
        'error': None
    }
    
    # Find compartment objects
    all_compartment_objs = [obj for obj in bpy.data.objects 
                            if obj.get('is_compartment_box', False)]
    
    # Also check by prefix for backwards compatibility
    if not all_compartment_objs:
        all_compartment_objs = [obj for obj in bpy.data.objects 
                                if obj.name.startswith(INTERIOR_COMPARTMENT_PREFIX)]
    
    if use_all:
        # Use all compartments
        compartment_objs = all_compartment_objs
    elif compartment_ids is not None:
        # Use specified IDs
        compartment_objs = []
        for obj in all_compartment_objs:
            comp_id = obj.get('compartment_id', -1)
            if comp_id in compartment_ids or 'all' in str(compartment_ids).lower():
                compartment_objs.append(obj)
    else:
        # Use selected compartments (check both anchor and children)
        compartment_objs = []
        for obj in all_compartment_objs:
            # Check if this compartment or any of its children is selected
            is_selected = obj.select_get()
            if not is_selected:
                # Check children (wire box, corners)
                for child in obj.children:
                    if child.select_get():
                        is_selected = True
                        break
            if is_selected:
                compartment_objs.append(obj)
        
        if not compartment_objs:
            print("  No compartments selected - please select compartment anchor or corners")
            result['error'] = "No compartments selected"
            return result
    
    if not compartment_objs:
        result['error'] = "No compartments available"
        print(f"  Error: {result['error']}")
        return result
    
    # Get IDs of compartments being processed
    processing_ids = set()
    for obj in compartment_objs:
        comp_id = obj.get('compartment_id', -1)
        if comp_id >= 0:
            processing_ids.add(comp_id)
    
    print(f"  Processing {len(compartment_objs)} compartment(s): IDs {sorted(processing_ids)}")
    
    # Clear panels - either ALL or just for processed compartments
    if clear_all:
        print("  Clearing ALL existing panels (fresh start)")
        clear_interior_panels()
    else:
        # Only clear panels for compartments being regenerated (accumulative mode)
        _clear_panels_for_compartments(processing_ids)
    
    col = _ensure_interior_collection()
    
    for comp_obj in compartment_objs:
        comp_id = comp_obj.get('compartment_id', 0)
        
        # Get compartment bounds from custom properties or object bounds
        if 'compartment_min' in comp_obj and 'compartment_max' in comp_obj:
            mn = Vector(comp_obj['compartment_min'])
            mx = Vector(comp_obj['compartment_max'])
        else:
            mn, mx = _object_world_bounds(comp_obj)
        
        side_ax = comp_obj.get('side_ax', 0)
        up_ax = comp_obj.get('up_ax', 2)
        
        print(f"  Compartment {comp_id}: {mn} to {mx}")
        
        # Compute panel positions
        panels = compute_equal_panels_in_aabb(
            mn, mx,
            orientation=orientation,
            num_panels=num_panels,
            panel_thickness=panel_thickness,
            side_ax=side_ax,
            up_ax=up_ax,
            front_inset=front_inset
        )
        
        if not panels:
            print(f"    No valid panel positions (compartment may be too small)")
            continue
        
        # Create panel objects
        for i, (panel_mn, panel_mx) in enumerate(panels):
            panel_name = f"{INTERIOR_PANEL_PREFIX}C{comp_id:02d}_P{i:02d}"
            
            panel_obj = _create_panel_box(panel_name, panel_mn, panel_mx)
            
            # Move to collection
            for c in list(panel_obj.users_collection):
                c.objects.unlink(panel_obj)
            col.objects.link(panel_obj)
            
            # Store metadata
            panel_obj['compartment_id'] = comp_id
            panel_obj['panel_index'] = i
            panel_obj['orientation'] = orientation
            
            result['panel_objects'].append(panel_obj)
            print(f"    Created {panel_name}")
        
        result['panel_count'] += len(panels)
        result['compartments_processed'] += 1
    
    result['success'] = True
    print(f"  Generated {result['panel_count']} panel(s) in {result['compartments_processed']} compartment(s)")
    
    return result


def clear_compartment_boxes():
    """Remove all compartment visualization boxes from the scene."""
    # Find all compartment anchors (main objects)
    anchors = []
    for obj in bpy.data.objects:
        if obj.get('is_compartment_box', False):
            anchors.append(obj)
        elif obj.name.startswith(INTERIOR_COMPARTMENT_PREFIX) and not obj.parent:
            # Fallback for objects without the custom property
            anchors.append(obj)
    
    # Collect all children first, then delete everything
    to_remove = []
    for anchor in anchors:
        # Add children (wire, corners)
        for child in anchor.children:
            to_remove.append(child)
        to_remove.append(anchor)
    
    # Also remove any orphaned wire/corner objects
    for obj in bpy.data.objects:
        if obj.get('is_compartment_wire', False) or obj.get('is_compartment_corner', False):
            if obj not in to_remove:
                to_remove.append(obj)
    
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    print(f"  Cleared {len(anchors)} compartment(s) ({len(to_remove)} objects total)")


def clear_interior_panels():
    """Remove all generated interior panels from the scene."""
    to_remove = []
    for obj in bpy.data.objects:
        if obj.name.startswith(INTERIOR_PANEL_PREFIX):
            to_remove.append(obj)
    
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    if to_remove:
        print(f"  Cleared {len(to_remove)} interior panel(s)")


def _clear_panels_for_compartments(compartment_ids):
    """
    Remove panels only for specific compartments (for accumulative regeneration).
    
    This allows regenerating panels for one compartment without losing panels
    in other compartments.
    """
    if not compartment_ids:
        return
    
    to_remove = []
    for obj in bpy.data.objects:
        if obj.name.startswith(INTERIOR_PANEL_PREFIX):
            panel_comp_id = obj.get('compartment_id', -1)
            if panel_comp_id in compartment_ids:
                to_remove.append(obj)
    
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    if to_remove:
        print(f"  Cleared {len(to_remove)} panel(s) for compartment(s) {sorted(compartment_ids)}")


def get_compartment_objects():
    """Get all compartment box objects in the scene."""
    return [obj for obj in bpy.data.objects if obj.name.startswith(INTERIOR_COMPARTMENT_PREFIX)]


def get_interior_panel_objects():
    """Get all interior panel objects in the scene."""
    return [obj for obj in bpy.data.objects if obj.name.startswith(INTERIOR_PANEL_PREFIX)]


def commit_interior_panels():
    """
    Commit interior panels - make them permanent part of the scene.
    
    Renames panels to remove the prefix and removes compartment boxes.
    """
    print("\n=== Committing Interior Panels ===")
    
    # Rename panels
    panels = get_interior_panel_objects()
    for panel in panels:
        # Remove prefix, keep the rest
        new_name = panel.name.replace(INTERIOR_PANEL_PREFIX, "shelf_" if panel.get('orientation') == 'horizontal' else "divider_")
        panel.name = new_name
        
        # Remove metadata
        if 'compartment_id' in panel:
            del panel['compartment_id']
        if 'panel_index' in panel:
            del panel['panel_index']
        if 'orientation' in panel:
            del panel['orientation']
    
    # Clear compartment boxes
    clear_compartment_boxes()
    
    print(f"  Committed {len(panels)} panel(s)")
    
    return {'success': True, 'panel_count': len(panels)}
