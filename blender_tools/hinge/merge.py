"""Mesh-fragment merge for movable parts (doors, drawers).

In the decomposed .blend each link's GLB is split into per-submesh objects
named `<link>__mesh00`, `<link>__mesh00.001`, `<link>__mesh00.002`, etc.
For static body fragments we leave them split. For movable parts we want a
single merged mesh -- one rigid body to parent under hinge_dynamic_end and
one BVH for validate.py to test.

Static body fragments must NOT be merged; they're tested as individual
fragments by the policy ranker and validator.
"""

from __future__ import annotations

import bpy  # type: ignore


def merge_fragments_to_single_mesh(
    prefix: str,
    scene: bpy.types.Scene | None = None,
) -> bpy.types.Object | None:
    """Join every MESH object whose name starts with `prefix` into one.

    Returns the merged object (the one selected as active for join), or
    None if no matching meshes were found.

    World transforms are preserved. Caller is responsible for re-parenting
    the merged result if needed.
    """
    if scene is None:
        scene = bpy.context.scene
    matching = [
        obj for obj in scene.objects
        if obj.type == "MESH" and obj.name.startswith(prefix)
    ]
    if not matching:
        return None
    if len(matching) == 1:
        return matching[0]

    # Use Blender's native join op. Active = the survivor.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in matching:
        obj.select_set(True)
    active = matching[0]
    bpy.context.view_layer.objects.active = active
    bpy.ops.object.join()
    bpy.context.view_layer.update()
    return active
