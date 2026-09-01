# geometry/

Shared 3D math, I/O, and the `O3DSceneObject` dataclass. Base layer — imports nothing local.

## Non-Obvious Constraints

**If you want to import from sensor/physics/registration inside here, the dependency graph is wrong.** Refactor the caller instead.

**PLY comment format is a load-bearing protocol.** Key names (`geocenter_x`, `gt_qw`, etc.) are parsed by downstream loaders. Do not rename or reformat them without updating all readers.

**`O3DSceneObject.id` is unset at construction.** It is assigned as a side effect of `scene_render()` adding geometry to the O3D raycast scene. Any code reading `.id` before a `scene_render()` call will get `None`.

**`pcd_geocenter()` returns a 4×4 transform, not a point.** It is the *inverse* frame transform — the matrix you APPLY to the cloud to put it in the frame — so the translation sits in **column 3** (`[:3, 3]`) and row 3 is always `[0, 0, 0, 1]`, as for any homogeneous transform.

This file used to claim the translation was in row 3. It never was, and `SaveStage` was written against that claim: `geocenter_x/y/z` in every PLY ever exported read `geocenter[3, 0..2]` and was therefore `0.0` regardless of the frame. If you change the packing in `_pack`, fix this line in the same commit.

**`app.geocenter` is the transform that has been APPLIED**, not one that is pending. It starts at identity (geometry is in the import frame) and `DownsampleStage.recenter_mesh_pcd` composes into it, so the viewer's world frame is always the exported frame and `geo_center.json` stays a truthful identity. Its inverse is the model frame expressed in the pre-recentre frame, which is what the `geocenter_*` PLY comments carry.

**The model frame is deterministic but not robust.** `pcd_geocenter` is pure — same cloud in, bitwise-identical matrix out — and the whole app path is deterministic (no RNG in `raycast_stage`, `analyse_ambiguity` runs at `seed=0`). But change the voxel size, the view count or the ambiguity config and a *different* ambiguity axis can win: measured on 25333MB000, three sample densities gave folds C4/C1/C2 with the third on an axis 45° away, moving the frame 60.6° and 28.3 mm. Anything pairing a reference cloud with scenes must therefore check they came from the same export — see `reference_frames_agree()`.
