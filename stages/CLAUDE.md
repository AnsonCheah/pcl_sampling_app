# stages/

6-stage pipeline orchestration: GUI panels, background workers, scene visualization. Calls into sensor/, physics/, geometry/ — implements none of their domain logic.

## Non-Obvious Constraints

**Never touch Open3D GUI objects from `worker()`.** O3D's GUI is not thread-safe. All scene mutations from worker threads must go through `self.app.main_thread(lambda: ...)`. Violating this produces silent corruption or crashes, not exceptions.

**Stage communication is exclusively through `app` attributes.** No direct stage-to-stage calls. The full attribute handoff table is in the root `CLAUDE.md`.

**`build_panel()` returns `None` in headless mode.** Every stage checks `self.app.headless` at the top of `build_panel()` and `_refresh_ui()`. Headless callers drive workers directly via `stage._run_worker()`.

**`SyntheticStage` orchestrates but does not implement.** All sensor physics and physics simulation happen in `sensor/` and `physics/`. If sensor math or MuJoCo logic starts appearing in `synthetic_stage.py`, it's in the wrong place.

**`decompose_thread` race condition.** `ImportMeshStage.decompose_mesh()` runs in a daemon thread that starts during `worker()`. `SyntheticStage` requires the decomposition to be complete before calling `MujocoBinScene`. The join must happen before sim init.
