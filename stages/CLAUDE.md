# stages/

6-stage pipeline orchestration: GUI panels, background workers, scene visualization. Calls into sensor/, physics/, geometry/ — implements none of their domain logic.

## Non-Obvious Constraints

**Never touch Open3D GUI objects from `worker()`.** O3D's GUI is not thread-safe. All scene mutations from worker threads must go through `self.app.main_thread(lambda: ...)`. Violating this produces silent corruption or crashes, not exceptions.

**Stage communication is exclusively through `app` attributes.** No direct stage-to-stage calls. The full attribute table is in `stages/README.md`, generated from the `downstream` dicts.

**State clearing is declarative, not hand-written.** Each stage declares the `app.*` attributes it owns in a `downstream` class dict (attr → default factory). `app.clear_state_from(stage)` resets that stage's own state plus every later stage's, by strict `Stage` enum order — used by both `app._restart()` and each worker's start-of-run wipe. Do NOT add ad-hoc per-attribute clearing; add the attribute to the owning stage's `downstream` dict instead. Clearing is strict-order, so it may over-clear sibling branches (e.g. resetting `RAYCAST` also clears `convex_meshes`); that is intentional and always safe. GUI/scratch cleanup goes in a stage's `on_clear()` hook; `CropStage` is the only stage that overrides `reset()` (it restores `cropped_pcd` instead of nulling it).

**`build_panel()` returns `None` in headless mode.** Every stage checks `self.app.headless` at the top of `build_panel()` and `_refresh_ui()`. Headless callers drive workers directly via `stage._run_worker()`.

**Synthetic generation is split into `SceneStage` → `RenderStage`.** `SceneStage` (`scene_stage.py`) builds the physical bin scene (MuJoCo arrangement/settle, partitions/tray) → `app.o3d_scene` + `app.mj_scene`; `RenderStage` (`render_stage.py`) simulates the sensor (raycast → noise → segmentation → export). They communicate only through `app` attributes. If sensor math appears in `scene_stage.py`, or MuJoCo logic in `render_stage.py`, it's in the wrong place.

**`DecomposeStage` is synchronous — there is no thread to join.** It runs VHACD in a `ProcessPoolExecutor` and blocks on the result, so `app.convex_meshes` is guaranteed populated when the worker returns. (It used to be a daemon thread callers had to `join()` before `SceneStage`; that is gone.) The child process exists because `vhacdx.compute_vhacd` holds the GIL for its whole runtime and would otherwise freeze the GUI.

**`app.main_thread(fn)` silently DROPS `fn` when headless.** It is a no-op, not a direct call. Any state change routed through it therefore does not happen in a headless run — `_express_sampling_worker` sets the stage that way, so `app.stage` never becomes `SAVE`, `SaveStage.worker()` matches neither of its branches, and the reference bundle is not written. No error is raised. Headless drivers must call `app.set_stage(...)` **directly**; see `bench/generate_scenes.py`.

**Runtime `print()` must stay ASCII.** A Windows console piping stdout uses cp1252, so an em dash or arrow in a print statement raises `UnicodeEncodeError` and kills the run — which is exactly what a logged batch job does. Docstrings and comments are unaffected; only text that reaches stdout matters.
