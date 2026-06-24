# stages/

6-stage pipeline orchestration: GUI panels, background workers, scene visualization. Calls into sensor/, physics/, geometry/ — implements none of their domain logic.

## Non-Obvious Constraints

**Never touch Open3D GUI objects from `worker()`.** O3D's GUI is not thread-safe. All scene mutations from worker threads must go through `self.app.main_thread(lambda: ...)`. Violating this produces silent corruption or crashes, not exceptions.

**Stage communication is exclusively through `app` attributes.** No direct stage-to-stage calls. The full attribute handoff table is in the root `CLAUDE.md`.

**`build_panel()` returns `None` in headless mode.** Every stage checks `self.app.headless` at the top of `build_panel()` and `_refresh_ui()`. Headless callers drive workers directly via `stage._run_worker()`.

**Synthetic generation is split into `SceneStage` → `RenderStage`.** `SceneStage` (`scene_stage.py`) builds the physical bin scene (MuJoCo arrangement/settle, partitions/tray) → `app.o3d_scene` + `app.mj_scene`; `RenderStage` (`render_stage.py`) simulates the sensor (raycast → noise → segmentation → export). They communicate only through `app` attributes. If sensor math appears in `scene_stage.py`, or MuJoCo logic in `render_stage.py`, it's in the wrong place.

**`decompose_thread` race condition.** `ImportMeshStage.decompose_mesh()` runs in a daemon thread that starts during `worker()`. `SceneStage` requires the decomposition to be complete before calling `MujocoBinScene`. The join must happen before sim init.
