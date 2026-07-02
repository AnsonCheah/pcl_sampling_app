# physics/

MuJoCo rigid-body bin simulation: place N parts, settle, extract ground-truth poses.

## Non-Obvious Constraints

**Final part count may be less than `n_parts`.** Collision-aware placement skips parts that cannot be placed without overlap after ~50 attempts. Callers must not assume `len(scene_state) == n_parts`.

**`bin` key is always present in `mujoco_scene_to_o3d()` output.** `RenderStage` depends on this to separate bin points from part points via `geom_id`. Do not make it conditional. Partition/tray fixtures are merged into `bin_mesh`, so they share the bin geom id and are treated as background too.

**`simulate()` is safe to call from a background thread.** MuJoCo has no main-thread requirement. The `render=True` passive viewer is the exception — it is blocking and must never be used in headless or batch runs.

**Convex mesh input comes from `ImportMeshStage.decompose_mesh()`.** That decomposition runs in its own daemon thread and may not be complete when `SceneStage` starts. `SceneStage` must wait on `decompose_thread.join()` before calling `MujocoBinScene`.
