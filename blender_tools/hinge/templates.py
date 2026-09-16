"""Hinge template loading and annotation canonicalization.

Why this module exists -- the canonicalization rationale:

    Template files ship with per-file quirks. Measuring all three shows
    hinge_exterior_01 carries scale (-0.115, +0.115, +0.115) on its root,
    while hinge_interior_01's root has scale (0.024, 0.048, 0.017).
    Hard-coding assumptions about which template has which quirk would
    make the others silently decompose wrong, surfacing only as weird
    snap residuals.

    Canonicalization at load -- one lookup table mapping name variants to
    a canonical TemplateRefs struct -- means downstream code never
    references templates by name. A new template variant with a fresh
    naming convention or scale quirk is added here, once, and snap.py /
    measure_template_footprints.py / policy.py keep working unchanged.

Two name flavors exist across the templates:

    canonical name      flavor A (interior, exterior)     flavor B (flat)
    ---------------     -----------------------------     ----------------
    root                hinge_master                      hinge
    axis                rotation_axis                     rotation_axis
    static_leaf         hinge_static_end                  hinge_static_end
    dynamic_leaf        hinge_dynamic_end                 hinge_dynamic_end
    snap_static         snap_plane_static_parts           snap_plane_static_end
    snap_dynamic        snap_plane_dynamic_parts          snap_plane_dynamic_end

This module resolves either flavor and returns a TemplateRefs struct with
canonical attribute names. Downstream code (snap.py, footprint script)
references TemplateRefs only -- no raw bpy lookups by literal string.

Negative-scale gotcha:
    hinge_exterior_01's hinge_master ships with
    scale=(-0.115, +0.115, +0.115). matrix_world.to_quaternion() routes the
    negative determinant into the rotation, returning a wrong quaternion;
    matrix_world.to_scale() returns (-0.115, -0.115, -0.115) (also wrong).
    The matrix itself is correct -- matrix_world @ vec_local works.
    Callers that need the orientation as a quaternion (currently only
    snap.py on hinge_master) are responsible for converting rotation_mode
    to QUATERNION themselves AND syncing rotation_quaternion from the
    object's prior rotation channel before reading. load_template no
    longer touches rotation_mode anywhere -- doing so unconditionally
    silently disconnects rotation_euler fcurves on animated objects
    (Blender does not auto-convert keys when rotation_mode changes).

Stale-matrix_world gotcha:
    bpy.data.libraries.load() does NOT trigger a depsgraph evaluation.
    Until depsgraph runs, child objects' matrix_world is the cached pre-link
    identity transform, ignoring the parent's scale. For
    hinge_exterior_01 specifically, this makes leaf AABBs come out at
    template-local scale (24mm-40mm read as 24-40 *meters*).
    load_template() calls bpy.context.view_layer.update() after appending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import bpy  # type: ignore


# Canonical name -> tuple of acceptable on-disk names (flavor A, flavor B, ...).
_NAME_LOOKUP: dict[str, tuple[str, ...]] = {
    "root":         ("hinge_master", "hinge"),
    "axis":         ("rotation_axis",),
    "static_leaf":  ("hinge_static_end",),
    "dynamic_leaf": ("hinge_dynamic_end",),
    "snap_static":  ("snap_plane_static_parts", "snap_plane_static_end"),
    "snap_dynamic": ("snap_plane_dynamic_parts", "snap_plane_dynamic_end"),
}

# Mesh objects that should NOT be classified as "mechanism parts" even
# though they live under root. Carve helpers and anything matching the
# canonical-annotation set are excluded.
_MECHANISM_EXCLUDE_PREFIXES = ("carve_", "__hidden_")
_MECHANISM_EXCLUDE_NAMES = {
    name for variants in _NAME_LOOKUP.values() for name in variants
}


@dataclass
class TemplateRefs:
    """Canonical refs to one loaded hinge template."""
    root: bpy.types.Object
    axis: bpy.types.Object
    static_leaf: bpy.types.Object
    dynamic_leaf: bpy.types.Object
    snap_static: bpy.types.Object
    snap_dynamic: bpy.types.Object
    mechanism_parts: list[bpy.types.Object] = field(default_factory=list)
    template_path: Path = Path()


def load_template(path: Path, suffix: str) -> TemplateRefs:
    """Append every object from `path` into the active scene, suffix names
    with `__{suffix}` to avoid collisions, and return canonicalized refs.

    Sets rotation_mode = 'QUATERNION' on every annotated object so callers
    can read rotation_quaternion directly without worrying about the
    negative-scale gotcha on hinge_interior_01.

    Raises:
        FileNotFoundError: template file missing.
        KeyError: a required canonical annotation is absent from the template.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"template not found: {path}")

    # ---- append all objects ----
    with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    imported = [obj for obj in data_to.objects if obj is not None]
    scene_collection = bpy.context.scene.collection
    for obj in imported:
        if obj.name not in scene_collection.objects:
            scene_collection.objects.link(obj)

    # Force depsgraph evaluation. Without this, matrix_world for newly-linked
    # objects can be the cached pre-link identity transform; in particular,
    # hinge_exterior_01's hinge_master scale=(-0.115, +0.115, +0.115) doesn't
    # propagate to children's matrix_world until depsgraph runs, causing
    # mesh_world AABBs to come out 1000x too large.
    bpy.context.view_layer.update()

    # ---- index by ORIGINAL name (before suffixing) ----
    by_orig: dict[str, bpy.types.Object] = {}
    for obj in imported:
        # Blender may have appended ".001" if the name was already in scene.
        # Strip any trailing ".NNN" to recover the template's original name.
        orig = obj.name
        if "." in orig and orig.rsplit(".", 1)[-1].isdigit():
            orig = orig.rsplit(".", 1)[0]
        by_orig[orig] = obj

    # ---- resolve canonical roles ----
    resolved: dict[str, bpy.types.Object] = {}
    for canonical, variants in _NAME_LOOKUP.items():
        for variant in variants:
            if variant in by_orig:
                resolved[canonical] = by_orig[variant]
                break
        else:
            raise KeyError(
                f"template {path.name} missing canonical annotation '{canonical}' "
                f"(looked for any of: {variants})"
            )

    # ---- find mechanism parts (mesh descendants of root, minus annotations) ----
    root = resolved["root"]
    mechanism_parts: list[bpy.types.Object] = []
    for obj in imported:
        if obj is root:
            continue
        if obj.type != "MESH":
            continue
        if any(obj.name.startswith(p) for p in _MECHANISM_EXCLUDE_PREFIXES):
            continue
        # Strip suffix if any to compare against canonical annotation names.
        base_name = obj.name
        if "." in base_name and base_name.rsplit(".", 1)[-1].isdigit():
            base_name = base_name.rsplit(".", 1)[0]
        if base_name in _MECHANISM_EXCLUDE_NAMES:
            continue
        if not _is_descendant_of(obj, root):
            continue
        mechanism_parts.append(obj)

    # NOTE: previously set rotation_mode='QUATERNION' on every annotated
    # object here. That silently broke hinge_dynamic_end's animation -- it
    # ships with rotation_euler fcurves and Blender does not auto-convert
    # keys when rotation_mode changes. snap.py now handles its own
    # rotation_mode conversion on hinge_master only.

    # ---- suffix every imported name to avoid scene-wide collisions ----
    if suffix:
        sfx = f"__{suffix}"
        for obj in imported:
            obj.name = f"{obj.name}{sfx}"

    return TemplateRefs(
        root=resolved["root"],
        axis=resolved["axis"],
        static_leaf=resolved["static_leaf"],
        dynamic_leaf=resolved["dynamic_leaf"],
        snap_static=resolved["snap_static"],
        snap_dynamic=resolved["snap_dynamic"],
        mechanism_parts=mechanism_parts,
        template_path=path,
    )


def _is_descendant_of(obj: bpy.types.Object, root: bpy.types.Object) -> bool:
    cur = obj.parent
    while cur is not None:
        if cur is root:
            return True
        cur = cur.parent
    return False
