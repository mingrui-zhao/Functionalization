"""Measure the AABB of every annotated and mechanism mesh in each hinge
template. Output JSON consumed by policy.py's MECHANISM_FOOTPRINT.

Run inside Blender:
    blender --background --python blender_tools/hinge/measure_template_footprints.py

Output: blender_tools/hinge/template_footprints.json

Coordinate frame:
    All AABBs are reported in the "pin frame" -- the frame where the
    rotation_axis empty's world position is at the origin and its world +Z
    direction is +Z in this frame. Downstream policy code wants its
    mechanism geometry in this frame so it can be reused at any axis_origin
    by translation only.

Pin frame construction:
    pin_world  = axis_obj.matrix_world.translation
    z_world    = axis_obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
    pick any orthonormal x_world, y_world spanning plane perp to z_world
    R_world_to_pin = stack rows [x_world, y_world, z_world]
    p_pin = R_world_to_pin @ (p_world - pin_world)

Negative-scale gotcha:
    hinge_interior_01.blend has hinge_master at scale (-0.115, +0.115, +0.115).
    matrix_world.to_3x3() correctly captures the orientation including the
    sign flip; what's wrong is matrix_world.to_quaternion() which routes the
    sign into the rotation. We use the matrix's column for axis direction,
    not a decomposed quaternion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo-level imports
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE.parents[1]))   # blender_tools/ on sys.path

import bpy  # type: ignore
from mathutils import Matrix, Vector  # type: ignore

from hinge.templates import TemplateRefs, load_template


TEMPLATE_DIR = _REPO / "annotated_mechanical_parts"
OUTPUT_PATH = _HERE.parent / "template_footprints.json"

TEMPLATES = {
    "hinge_interior_01": TEMPLATE_DIR / "hinge_interior_01.blend",
    "hinge_exterior_01": TEMPLATE_DIR / "hinge_exterior_01.blend",
    "hinge_flat_01":     TEMPLATE_DIR / "hinge_flat_01.blend",
}


# ============================================================ helpers ==

def _clear_scene():
    """Delete all objects from the current scene + purge orphan data."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.objects, bpy.data.actions,
                       bpy.data.materials, bpy.data.armatures):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _world_to_pin_frame(axis_obj: bpy.types.Object) -> tuple[Vector, Matrix]:
    """Return (pin_world, R_world_to_pin) for the rotation_axis empty.

    R_world_to_pin is a 3x3 mathutils.Matrix mapping world-frame vectors
    to pin-frame vectors. Pin-frame +Z aligns with axis local +Z (the
    hinge axis direction), pin-frame origin sits at axis world position.
    """
    pin_world = axis_obj.matrix_world.translation.copy()
    rot3 = axis_obj.matrix_world.to_3x3()
    # axis local +Z in world coords. This is the hinge axis direction.
    z_world = (rot3 @ Vector((0.0, 0.0, 1.0))).normalized()
    # Build any orthonormal x perpendicular to z_world.
    helper = Vector((1.0, 0.0, 0.0)) if abs(z_world.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    x_world = (helper - helper.dot(z_world) * z_world).normalized()
    y_world = z_world.cross(x_world).normalized()
    # World-to-pin is the inverse of pin-to-world. pin-to-world has columns
    # (x_world, y_world, z_world), and since they're orthonormal the inverse
    # is the transpose -- i.e. rows (x_world, y_world, z_world).
    R_world_to_pin = Matrix((x_world, y_world, z_world))
    return pin_world, R_world_to_pin


def _mesh_aabb_in_pin_frame(
    obj: bpy.types.Object,
    pin_world: Vector,
    R_world_to_pin: Matrix,
) -> tuple[list[float], list[float]]:
    """Return ([min_x, min_y, min_z], [max_x, max_y, max_z]) of obj's mesh
    vertices, transformed into the pin frame."""
    if obj.type != "MESH" or obj.data is None or len(obj.data.vertices) == 0:
        # Empty / lightless object -- fall back to its origin point.
        p_pin = R_world_to_pin @ (obj.matrix_world.translation - pin_world)
        return list(p_pin), list(p_pin)
    mw = obj.matrix_world
    pin_xs = []
    for v in obj.data.vertices:
        v_world = mw @ v.co
        v_pin = R_world_to_pin @ (v_world - pin_world)
        pin_xs.append(v_pin)
    arr = [list(p) for p in pin_xs]
    mins = [min(p[i] for p in arr) for i in range(3)]
    maxs = [max(p[i] for p in arr) for i in range(3)]
    return mins, maxs


def _record_for_object(
    obj: bpy.types.Object,
    pin_world: Vector,
    R_world_to_pin: Matrix,
) -> dict:
    mins, maxs = _mesh_aabb_in_pin_frame(obj, pin_world, R_world_to_pin)
    center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    size = [maxs[i] - mins[i] for i in range(3)]
    # Object origin (matrix_world.to_translation()) in pin frame. Differs
    # from AABB center when the mesh is modeled offset from its object's
    # local origin -- e.g. hinge_flat_01's snap planes have mesh origins at
    # world (0,0,0) but mesh vertices offset by ~14mm. snap.py reads object
    # origin via matrix_world.to_translation(); policy.py needs the same
    # value to compute coplanarity residuals consistent with snap residuals.
    origin_world = obj.matrix_world.translation
    origin_pin = R_world_to_pin @ (origin_world - pin_world)
    return {
        "name": obj.name,
        "type": obj.type,
        "aabb_min_pin_frame": mins,
        "aabb_max_pin_frame": maxs,
        "center_pin_frame": center,
        "object_origin_pin_frame": list(origin_pin),
        "size": size,
    }


def measure_template(template_name: str, path: Path) -> dict:
    """Load template into a fresh empty scene, return measurements dict."""
    _clear_scene()
    refs: TemplateRefs = load_template(path, suffix=template_name)

    pin_world, R_world_to_pin = _world_to_pin_frame(refs.axis)

    # axis world direction (in world coords) for sanity reference.
    rot3 = refs.axis.matrix_world.to_3x3()
    axis_world_dir = list((rot3 @ Vector((0.0, 0.0, 1.0))).normalized())

    # snap_static world normal: snap plane's local +Z direction in world.
    snap_rot = refs.snap_static.matrix_world.to_3x3()
    snap_static_world_normal = list(
        (snap_rot @ Vector((0.0, 0.0, 1.0))).normalized()
    )
    snap_dyn_rot = refs.snap_dynamic.matrix_world.to_3x3()
    snap_dynamic_world_normal = list(
        (snap_dyn_rot @ Vector((0.0, 0.0, 1.0))).normalized()
    )

    return {
        "template_path": str(path.relative_to(_REPO)),
        "axis_world_position": list(pin_world),
        "axis_world_direction": axis_world_dir,
        "snap_static_world_normal": snap_static_world_normal,
        "snap_dynamic_world_normal": snap_dynamic_world_normal,
        "static_leaf": _record_for_object(refs.static_leaf, pin_world, R_world_to_pin),
        "dynamic_leaf": _record_for_object(refs.dynamic_leaf, pin_world, R_world_to_pin),
        "snap_static": _record_for_object(refs.snap_static, pin_world, R_world_to_pin),
        "snap_dynamic": _record_for_object(refs.snap_dynamic, pin_world, R_world_to_pin),
        "mechanism_parts": [
            _record_for_object(p, pin_world, R_world_to_pin)
            for p in refs.mechanism_parts
        ],
        "mechanism_aggregate_aabb": _aggregate_mechanism_aabb(
            refs.mechanism_parts, pin_world, R_world_to_pin
        ),
    }


def _aggregate_mechanism_aabb(
    parts: list[bpy.types.Object],
    pin_world: Vector,
    R_world_to_pin: Matrix,
) -> dict:
    """Union of all mechanism parts' AABBs in the pin frame."""
    if not parts:
        return {"aabb_min_pin_frame": [0.0, 0.0, 0.0], "aabb_max_pin_frame": [0.0, 0.0, 0.0]}
    all_mins = []
    all_maxs = []
    for p in parts:
        mins, maxs = _mesh_aabb_in_pin_frame(p, pin_world, R_world_to_pin)
        all_mins.append(mins)
        all_maxs.append(maxs)
    mins = [min(m[i] for m in all_mins) for i in range(3)]
    maxs = [max(m[i] for m in all_maxs) for i in range(3)]
    size = [maxs[i] - mins[i] for i in range(3)]
    center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    return {
        "aabb_min_pin_frame": mins,
        "aabb_max_pin_frame": maxs,
        "size": size,
        "center_pin_frame": center,
    }


# ============================================================ entry ==

def main() -> None:
    output: dict = {}
    for name, path in TEMPLATES.items():
        if not path.exists():
            print(f"  SKIP: {path} (not found)")
            continue
        print(f"  measuring {name}...")
        output[name] = measure_template(name, path)

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nwrote {OUTPUT_PATH}")
    print(f"  {len(output)} templates measured")


if __name__ == "__main__":
    main()
