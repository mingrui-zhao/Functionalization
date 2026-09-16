"""Rail template loading + canonicalization. Mirrors hinge/templates.py.

Sliding rail template anatomy:
    rail_master                       root empty (parent of everything)
    rail_static_end                   mesh -- attaches to cabinet wall
    rail_dynamic_end                  mesh -- attaches to drawer side, slides
    sliding_axis                      empty SINGLE_ARROW, +Z = slide direction
    snap_plane_static_end             mesh rectangle, lies on cabinet wall
    snap_plane_rail_dynamic_end       mesh rectangle, lies on drawer side
    snap_plane_drawer_support         mesh rectangle, lies on drawer bottom
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bpy  # type: ignore


# Required for ALL rail templates.
_NAME_LOOKUP_REQUIRED: dict[str, tuple[str, ...]] = {
    "root":               ("rail_master", "rail"),
    "axis":               ("sliding_axis",),
    "static_end":         ("rail_static_end",),
    "dynamic_end":        ("rail_dynamic_end",),
    "snap_static":        ("snap_plane_static_end",),
    "snap_dynamic":       ("snap_plane_rail_dynamic_end", "snap_plane_dynamic_end"),
}
# Optional: only template_01 (corner-mount) has a 3rd snap plane for the drawer bottom.
_NAME_LOOKUP_OPTIONAL: dict[str, tuple[str, ...]] = {
    "snap_support":       ("snap_plane_drawer_support",),
}


@dataclass
class RailTemplateRefs:
    root:          bpy.types.Object
    axis:          bpy.types.Object
    static_end:    bpy.types.Object
    dynamic_end:   bpy.types.Object
    snap_static:   bpy.types.Object
    snap_dynamic:  bpy.types.Object
    snap_support:  bpy.types.Object | None = None  # only template_01 has it
    style:         str = "corner"                  # "corner" | "center"
    template_path: Path = Path()


def load_rail_template(path: Path, suffix: str) -> RailTemplateRefs:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"rail template not found: {path}")

    with bpy.data.libraries.load(str(path), link=False) as (df, dt):
        dt.objects = list(df.objects)
    imported = [o for o in dt.objects if o is not None]
    coll = bpy.context.scene.collection
    for o in imported:
        if o.name not in coll.objects:
            coll.objects.link(o)
    bpy.context.view_layer.update()

    by_orig: dict[str, bpy.types.Object] = {}
    for o in imported:
        n = o.name
        if "." in n and n.rsplit(".", 1)[-1].isdigit():
            n = n.rsplit(".", 1)[0]
        by_orig[n] = o

    resolved: dict[str, bpy.types.Object] = {}
    for canonical, variants in _NAME_LOOKUP_REQUIRED.items():
        for v in variants:
            if v in by_orig:
                resolved[canonical] = by_orig[v]
                break
        else:
            raise KeyError(
                f"rail template {path.name} missing required '{canonical}' "
                f"(looked for any of: {variants})"
            )
    # Optional: snap_support (only present in corner-mount template).
    snap_support_obj = None
    for v in _NAME_LOOKUP_OPTIONAL["snap_support"]:
        if v in by_orig:
            snap_support_obj = by_orig[v]
            break
    style = "corner" if snap_support_obj is not None else "center"

    if suffix:
        sfx = f"__{suffix}"
        for o in imported:
            o.name = f"{o.name}{sfx}"

    return RailTemplateRefs(
        root=resolved["root"], axis=resolved["axis"],
        static_end=resolved["static_end"], dynamic_end=resolved["dynamic_end"],
        snap_static=resolved["snap_static"],
        snap_dynamic=resolved["snap_dynamic"],
        snap_support=snap_support_obj,
        style=style,
        template_path=path,
    )
