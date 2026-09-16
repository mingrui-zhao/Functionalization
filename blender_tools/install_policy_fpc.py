"""Collision-first C>F>P hinge selection.

For one hinge joint: snap ALL applicable hinge classes (flat, interior,
exterior), validate each with the swing-collision sweep that
insert_hinges_for_joint already runs, keep the winner ranked by
  1. collision (max door/mech penetration, banded to 1 mm)
  2. pin fidelity: perpendicular distance from the realized pin axis to
     the prediction-derived pin_hint, banded to 2 cm. A candidate whose
     pin drifts to the wrong door edge (e.g. a degenerate flat mount at
     the door's centerline that never collides because it barely moves)
     must not beat a correctly bordered mount of a lower-priority class.
  3. hinge-type priority C(exterior) > F(flat) > P(interior)
and delete the losers' objects.

Used by:
  - install_from_pred.py       --hinge=auto
  - functionalization_ui       per-joint "Auto (collision-tested)" type

CLI compatibility with the old wrapper script:
  blender --background --python install_policy_fpc.py -- <install args>
runs install_from_pred with --hinge=auto forced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy  # type: ignore

_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from hinge.geometry import HingeType
from hinge.multihinge import parent_door_to_driver
from hinge.pipeline import insert_hinges_for_joint

PRIORITY = {HingeType.EXTERIOR: 0, HingeType.FLAT: 1, HingeType.INTERIOR: 2}


def _delete_objs(names: set) -> None:
    for n in names:
        o = bpy.data.objects.get(n)
        if o is not None:
            mesh = o.data if o.type == "MESH" else None
            bpy.data.objects.remove(o, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    bpy.context.view_layer.update()


def fpc_select_and_install(joint_name: str,
                           candidate_types=(HingeType.FLAT,
                                            HingeType.INTERIOR,
                                            HingeType.EXTERIOR),
                           template_for_type: dict | None = None,
                           **insert_kwargs) -> dict:
    """Run the F>P>C competition for one joint and keep only the winner.

    insert_kwargs are forwarded to insert_hinges_for_joint verbatim (must
    NOT contain joint_name / force_hinge_type). `template_for_type`
    optionally maps HingeType → template filename so callers with variant
    galleries (the addon) can pin per-type variants; entries are applied
    via hinge.pipeline._TEMPLATE_FOR_TYPE around each candidate.

    Returns {"type": HingeType, "jr": JointResult, "pen": float,
             "attempts": [{type, pen, status, error?} ...]}.
    Raises RuntimeError when no hinge class produced a candidate.
    """
    from hinge import pipeline as _hp
    import numpy as np

    door_prefix = insert_kwargs["door_fragment_prefix"]
    pin_hint = insert_kwargs.get("pin_hint")
    axis_hint = insert_kwargs.get("axis_hint")
    if pin_hint is not None:
        pin_hint = np.asarray(pin_hint, dtype=float)
    if axis_hint is not None:
        axis_hint = np.asarray(axis_hint, dtype=float)
        n = float(np.linalg.norm(axis_hint))
        axis_hint = axis_hint / n if n > 1e-9 else None

    def _pin_hint_dist(obj_names: set) -> float:
        """Perpendicular distance (m) from the candidate's pin axis to the
        predicted pin point. Along-axis offset is ignored: multihinge
        spreads instances along the axis by design."""
        if pin_hint is None:
            return 0.0
        best = None
        for n in obj_names:
            if not n.startswith("rotation_axis__"):
                continue
            o = bpy.data.objects.get(n)
            if o is None:
                continue
            p = np.asarray(o.matrix_world.translation, dtype=float) - pin_hint
            if axis_hint is not None:
                p = p - float(p @ axis_hint) * axis_hint
            d = float(np.linalg.norm(p))
            best = d if best is None else min(best, d)
        return best if best is not None else 0.0

    attempts = []
    report = []
    for htype in candidate_types:
        before = {o.name for o in bpy.data.objects}
        prev_tpl = None
        if template_for_type and htype in template_for_type:
            prev_tpl = _hp._TEMPLATE_FOR_TYPE.get(htype)
            _hp._TEMPLATE_FOR_TYPE[htype] = template_for_type[htype]
        try:
            jr = insert_hinges_for_joint(
                joint_name=f"{joint_name}_cand_{htype.value}",
                force_hinge_type=htype,
                **insert_kwargs)
        except Exception as exc:
            after = {o.name for o in bpy.data.objects}
            # clean partial objects, but never the merged door
            _delete_objs({n for n in after - before
                          if not n.startswith(door_prefix)})
            print(f"    [policy] {joint_name} {htype.value}: no candidate ({exc})")
            report.append({"type": htype.value, "error": str(exc)})
            continue
        finally:
            if template_for_type and htype in template_for_type:
                if prev_tpl is None:
                    _hp._TEMPLATE_FOR_TYPE.pop(htype, None)
                else:
                    _hp._TEMPLATE_FOR_TYPE[htype] = prev_tpl
        after = {o.name for o in bpy.data.objects}
        new_objs = {n for n in after - before if not n.startswith(door_prefix)}
        pen = max(jr.validation.max_door_penetration_mm or 0.0,
                  jr.validation.max_mechanism_penetration_mm or 0.0)
        pin_d = _pin_hint_dist(new_objs)
        attempts.append({"type": htype, "jr": jr, "objs": new_objs,
                         "pen": pen, "band": round(pen),
                         "pin_d": pin_d, "pin_band": round(pin_d / 0.02)})
        report.append({"type": htype.value, "pen": pen,
                       "status": jr.validation.status})
        print(f"    [policy] {joint_name} {htype.value}: pen={pen:.2f}mm "
              f"pin_d={pin_d*1000:.0f}mm status={jr.validation.status}")

    if not attempts:
        raise RuntimeError(
            f"policy: no hinge class produced a candidate for {joint_name}")

    attempts.sort(key=lambda a: (a["band"], a["pin_band"],
                                 PRIORITY[a["type"]], a["pen"]))
    winner = attempts[0]

    # The door is currently parented to whichever candidate installed last.
    # Unparent it (preserving world pose) BEFORE deleting losers so the
    # door never jumps when its parent driver disappears.
    door = next((o for o in bpy.data.objects
                 if o.type == "MESH" and o.name.startswith(door_prefix)), None)
    if door is not None and door.parent is not None:
        mw = door.matrix_world.copy()
        door.parent = None
        door.matrix_parent_inverse.identity()
        door.matrix_world = mw
        if door.animation_data is not None:
            door.animation_data_clear()

    for a in attempts[1:]:
        _delete_objs(a["objs"])

    if door is not None:
        parent_door_to_driver(door, winner["jr"].multi)
    print(f"    [policy] {joint_name} → {winner['type'].value} "
          f"(pen={winner['pen']:.2f}mm)")
    return {"type": winner["type"], "jr": winner["jr"],
            "pen": winner["pen"], "attempts": report}


def main():
    """CLI wrapper: forward to install_from_pred with --hinge=auto."""
    if "--" in sys.argv:
        extra = sys.argv[sys.argv.index("--") + 1:]
        extra = [a for a in extra if not a.startswith("--hinge=")]
        sys.argv = sys.argv[:sys.argv.index("--") + 1] + extra + ["--hinge=auto"]
    else:
        sys.argv = sys.argv + ["--", "--hinge=auto"]
    import install_from_pred
    install_from_pred.main()


if __name__ == "__main__":
    main()
