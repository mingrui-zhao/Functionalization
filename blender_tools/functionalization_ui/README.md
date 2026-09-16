# GraFu Functionalization Suite (Blender add-on)

Interactive geometry realization for GraFu predictions. Loading a model
functionalizes it automatically (hinges, rails, supports, predicted handles,
top panel, interior); every installed joint then remains editable in the
viewport. The interactive and headless batch workflows share the same
installation implementation.

## Requirements

- Blender 4.2+ (5.0 tested)
- The GraFu repo, with `annotated_mechanical_parts/` template assets

## Install

From the repo checkout, either link or copy the package into Blender's
add-ons folder:

```
ln -s <repo>/blender_tools/functionalization_ui \
      ~/.config/blender/<ver>/scripts/addons/functionalization_ui
```

or

```
rsync -a --exclude='__pycache__' <repo>/blender_tools/functionalization_ui/ \
      ~/.config/blender/<ver>/scripts/addons/functionalization_ui/
echo "<repo>" > ~/.config/blender/<ver>/scripts/addons/functionalization_ui/repo_location.txt
```

Enable "GraFu Functionalization Suite" in Preferences > Add-ons. The
symlink layout finds the repo by itself; the copied layout reads
`repo_location.txt`.

## Use

Run inference first (`python infer.py ... --out results/<run>`). The run
directory is self-describing: it holds the predictions and an
`inference_manifest.json` pointing at each model's input graph and
meshes.

In the 3D viewport sidebar (`N`), open the **Functionalize** tab:

1. **Model**: scan, pick the run, pick a model, press **Load Model**. The model
   imports and functionalizes in one pass (toggle off "Auto-functionalize
   on load" to install manually). Manual paths remain available for data
   without a manifest. Alternatively, open one of the run's generated
   Blender scenes (`<run>/blends/<mid>.blend`) directly and press **Attach
   Opened Result**: the add-on reads the graphs through the manifest and
   adopts the installed joints for editing, without rebuilding the scene.
2. **Review**: ok/warn/err triage across all joints, a viewport overlay
   of predicted motion axes, motion preview (frames 0 to 100, max extent
   at 50), an eyedropper that jumps from a selected part to its joint
   row, and collision / connectivity checks that flag joints worth a
   look.
3. **Hinges**: per joint, keep the collision-tested automatic class or
   force one, switch template variants, adjust scale and count. Changes
   re-apply in place; a change that does not fit the geometry reverts to
   the last working configuration instead of leaving the joint bare.
4. **Rails**: the same per-joint controls for drawer rails; toggles fix
   mis-predicted mounting edges.
5. **Handles**: one entry per door or drawer, whether or not a handle
   was predicted. Choose among 18 templates or a recessed carved groove,
   set count (several handles distribute at equal partitions and shrink
   to fit) and placement offsets. **Apply** installs or replaces the
   handle; **Remove** deletes it; a carved groove is undone from a mesh
   snapshot.
6. **Tops**: detect whether a top is missing, then synthesize one; shape
   (rectangle, rounded, chamfered, oval, capsule), thickness, overhang,
   and corner radius are tunable live.
7. **Interior**: instantiate panels from the prediction (one per predicted
   shelf/divider node) or design manually via detected compartments.
8. **Finalize**: re-run the checks, commit and clean helper data, export
   a JSON manifest of every joint and its settings.

Everything logs to the Status Log panel and Blender's Info editor.

## Notes

- Frames 0 to 100 are a closed-open-closed cycle; frame 50 is maximum
  extent.
- Installed objects carry tags (`grafu_handle_parent`, `__rail_joint`,
  hinge names containing `__hinge_inst__`), which is what makes
  re-apply and removal reliable; avoid renaming them.

## Troubleshooting

- "repo root not auto-detected": write `repo_location.txt` (see
  Install) or set Repo root under Manual paths.
- Empty variant dropdowns: `annotated_mechanical_parts/` is missing;
  the inventory lists only templates present on disk.
- Lists stale after undo or file reload: Finalize > Rebuild From Scene.
