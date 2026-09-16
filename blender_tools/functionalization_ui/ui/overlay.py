"""Viewport overlay: predicted motion axes + hinge borders.

Draws, for every loaded joint row:
  hinges — the predicted rotation axis as a line through the pin point
           (extended over the door's axial extent) plus a rectangle
           outlining the predicted hinge_border face of the door OBB;
  rails  — the slide direction as an arrow from the drawer centre.

Colors follow row status: pending orange, ok green, warn yellow, err red.
The geometry is REBUILT only when operators call `rebuild_from_props()`
(load / apply / status change) — the draw callback just replays cached GPU
batches, so leaving the overlay on has no per-frame cost beyond the draw.
"""
from __future__ import annotations

import json
from pathlib import Path

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_HANDLER = None
_BATCHES: list = []          # [(batch, rgba)]
_SHADER = None

_COL_PENDING = (1.00, 0.55, 0.10, 1.0)
_COL_OK      = (0.20, 0.90, 0.30, 1.0)
_COL_WARN    = (1.00, 0.85, 0.10, 1.0)
_COL_ERR     = (1.00, 0.15, 0.15, 1.0)


def _color_for(status: str):
    if status.startswith("ok"):
        return _COL_OK
    if status.startswith("warn"):
        return _COL_WARN
    if status.startswith("err"):
        return _COL_ERR
    return _COL_PENDING


def _face_rect(center, half, face_idx):
    """Corner loop (8 line-segment endpoints) of an AABB face."""
    ax, sign = face_idx // 2, (-1.0 if face_idx % 2 == 0 else 1.0)
    others = [a for a in range(3) if a != ax]
    a, b = others
    corners = []
    for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = list(center)
        p[ax] = center[ax] + sign * half[ax]
        p[a] = center[a] + sa * half[a]
        p[b] = center[b] + sb * half[b]
        corners.append(tuple(p))
    segs = []
    for i in range(4):
        segs.append(corners[i])
        segs.append(corners[(i + 1) % 4])
    return segs


def _axis_line(origin, axis, half_len):
    o = list(origin)
    a = list(axis)
    p0 = tuple(o[k] - a[k] * half_len for k in range(3))
    p1 = tuple(o[k] + a[k] * half_len for k in range(3))
    return [p0, p1]


def _arrow(origin, axis, length):
    o = list(origin)
    a = list(axis)
    tip = [o[k] + a[k] * length for k in range(3)]
    segs = [tuple(o), tuple(tip)]
    # simple cross-flare head
    import math
    up = (0.0, 0.0, 1.0) if abs(a[2]) < 0.9 else (1.0, 0.0, 0.0)
    side = (a[1] * up[2] - a[2] * up[1],
            a[2] * up[0] - a[0] * up[2],
            a[0] * up[1] - a[1] * up[0])
    n = math.sqrt(sum(s * s for s in side)) or 1.0
    side = [s / n for s in side]
    head = 0.12 * length
    for sgn in (-1.0, 1.0):
        segs.append(tuple(tip))
        segs.append(tuple(tip[k] - a[k] * head + sgn * side[k] * head * 0.6
                          for k in range(3)))
    return segs


def rebuild_from_props(props) -> None:
    """Recompute the cached batches from the sidecar state + row statuses.
    No-op while the overlay is disabled (or in background mode) — the draw
    handler is the only consumer of the batches."""
    global _BATCHES, _SHADER
    _BATCHES = []
    if _HANDLER is None or bpy.app.background:
        return
    if not props.loaded or not props.state_json:
        return
    p = Path(props.state_json)
    if not p.is_file():
        return
    try:
        state = json.loads(p.read_text())
    except Exception:
        return
    if _SHADER is None:
        _SHADER = gpu.shader.from_builtin('UNIFORM_COLOR')

    status_by_joint = {it.joint_name: it.status for it in props.hinges}
    status_by_joint.update({it.joint_name: it.status for it in props.rails})

    grouped: dict[tuple, list] = {}
    for rec in state.get("hinges", []):
        col = _color_for(status_by_joint.get(rec.get("joint", ""), ""))
        origin = rec.get("joint_origin")
        axis = rec.get("hinge_axis")
        dc = rec.get("door_center"); ds = rec.get("door_size")
        if origin is None or axis is None or dc is None or ds is None:
            continue
        half = [s * 0.5 for s in ds]
        ax_i = max(range(3), key=lambda k: abs(axis[k]))
        half_len = half[ax_i] * 1.15 + 0.02
        segs = _axis_line(origin, axis, half_len)
        border = int(rec.get("hinge_border", -1))
        if 0 <= border <= 5:
            segs += _face_rect(dc, half, border)
        grouped.setdefault(col, []).extend(segs)

    for rec in state.get("rails", []):
        col = _color_for(status_by_joint.get(rec.get("joint", ""), ""))
        origin = rec.get("slide_origin")
        axis = rec.get("slide_axis_world")
        length = float(rec.get("half_slide_m", 0.15)) * 1.5 + 0.02
        if origin is None or axis is None:
            continue
        grouped.setdefault(col, []).extend(_arrow(origin, axis, length))

    for col, coords in grouped.items():
        if not coords:
            continue
        batch = batch_for_shader(_SHADER, 'LINES', {"pos": coords})
        _BATCHES.append((batch, col))


def _draw():
    if not _BATCHES or _SHADER is None:
        return
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    gpu.state.depth_test_set('NONE')
    for batch, col in _BATCHES:
        _SHADER.uniform_float("color", col)
        batch.draw(_SHADER)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def enable() -> None:
    global _HANDLER
    if _HANDLER is None:
        _HANDLER = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_VIEW')
    _tag_redraw()


def disable() -> None:
    global _HANDLER, _BATCHES
    if _HANDLER is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_HANDLER, 'WINDOW')
        except Exception:
            pass
        _HANDLER = None
    _BATCHES = []
    _tag_redraw()


def _tag_redraw() -> None:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
