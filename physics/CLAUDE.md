# physics/

MuJoCo rigid-body bin simulation: place N parts, settle, extract ground-truth poses.

## Non-Obvious Constraints

**Final part count may be less than `n_parts`.** Collision-aware placement skips parts that cannot be placed without overlap after ~50 attempts. Callers must not assume `len(scene_state) == n_parts`.

**`bin` key is always present in `mujoco_scene_to_o3d()` output.** `RenderStage` depends on this to separate bin points from part points via `geom_id`. Do not make it conditional. Partition/tray fixtures are merged into `bin_mesh`, so they share the bin geom id and are treated as background too.

**`extract_scene_state()` and `export_scene_state()` report DIFFERENT frames, on purpose.** The class requires a mesh centred on its own origin (every rotation, spawn height, stable pose and tray pocket is computed about `(0,0,0)`), so raw body poses describe the *centred* body. `extract_scene_state` returns exactly those -- correct for the live GUI preview, which pairs them with `mj_scene.part_mesh`, the centred mesh. `export_scene_state` composes `body_offset` back in so `scene_state.npz`'s `T_gt` means "model frame -> world", matching `sample_<i>.ply`'s `gt_*` header. Callers that centre their mesh **must** pass `body_offset`, or the scene-level GT silently reverts to the body frame while every other export stays in the model frame. `body_offset` is also written into the npz permanently -- it is the only record of where the body origin sat, so it is what lets a reader recover raw body poses from the file alone. Covered by `stages/tests/test_gt_frame_agreement.py`.

**`simulate()` is safe to call from a background thread.** MuJoCo has no main-thread requirement. The `render=True` passive viewer is the exception -- it is blocking and must never be used in headless or batch runs.

**Convex mesh input comes from `DecomposeStage`, which is synchronous.** It runs VHACD in a `ProcessPoolExecutor` and blocks, so `app.convex_meshes` is guaranteed populated when its worker returns. There is no thread to join.
