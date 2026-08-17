# stages

GUI panels and pipeline orchestration. Calls into `sensor/`, `physics/`, and `geometry/` — implements none of their domain logic.

## Pipeline

Stages run in sequence. All share `app` as their only communication channel.

| Stage enum | Class | Heavy work |
|------------|-------|------------|
| `IMPORT_MESH` | `ImportMeshStage` | STL loading + convex decomposition (daemon thread) |
| `RAYCAST` | `RaycastStage` | Fibonacci-sphere multi-view raycasting |
| `CROP` | `CropStage` | Box-select UI, mask-based point removal |
| `DOWNSAMPLE` | `DownsampleStage` | Uniform or adaptive voxel downsampling |
| `SAVE` | `SaveStage` | PLY export with geocenter + GT pose comments |
| `DECOMPOSE` | `DecomposeStage` | VHACD convex decomposition for sim collision |
| `SCENE` | `SceneStage` | MuJoCo bin arrangement/settle (random, or structured ± partition/tray) |
| `RENDER` | `RenderStage` | Structured-light noise chain + instance segmentation + export |

## `BaseStage`

All stages extend `BaseStage`:

```python
class BaseStage(ABC):
    def __init__(self, app):
        self.app = app            # central state object
        self.widgets = []         # UI controls owned by this stage
        self.panel = ...          # built by build_panel()
        self.worker_thread = None

    downstream = {...}            # app.* attrs this stage owns -> default factory
    @abstractmethod
    def build_panel(self): ...    # return None in headless mode
    @abstractmethod
    def _refresh_ui(self): ...    # called on scene change
    @abstractmethod
    def worker(self): ...         # heavy computation (runs in background thread)
    def reset(self): ...          # default: clear this stage + downstream, then refresh
    def on_clear(self): ...       # hook: GUI/scratch cleanup when this stage's data is cleared
```

Running a worker: `stage._run_worker()` spawns `worker()` in a background thread.

## Automatic downstream clearing

Each stage declares the `app.*` attributes it owns in a `downstream` class dict mapping
attr name → a zero-arg default factory. `BaseStage.clear_downstream()` resets them (and runs
`on_clear()`); `app.clear_state_from(stage, inclusive=True)` resets a stage's own state plus
every later stage's, by strict `Stage` enum order. This is the single source of truth:
`app._restart()` and each worker's start-of-run wipe both go through it, so adding a stage or
an attribute means editing one `downstream` dict and nothing else. `reset()` is the generic
default in `BaseStage`; only `CropStage` overrides it (it restores `cropped_pcd` from
`raw_pcd` instead of nulling it). GUI-only cleanup (e.g. RenderStage's comboboxes) lives in
each stage's `on_clear()`.

## `app` state attributes

Key attributes written and read between stages:

| Attribute | Written by | Read by |
|-----------|------------|---------|
| `app.target_mesh` | `ImportMeshStage` | `RaycastStage`, `SceneStage` |
| `app.convex_meshes` | `DecomposeStage` | `SceneStage` |
| `app.raw_pcd` | `RaycastStage` | `CropStage` |
| `app.down_pcd` | `DownsampleStage` | `SaveStage`, `RenderStage` |
| `app.o3d_scene` | `SceneStage` | `RenderStage` |
| `app.mj_scene` | `SceneStage` | `RenderStage`, `app._reframe` |
| `app.synthetic_scenes` | `RenderStage` | `SaveStage` |

Stages never call each other's methods directly. All handoffs go through `app`.

## Why `geocenter` is owned by IMPORT_MESH

`DownsampleStage.recenter_mesh_pcd` is what *writes* `app.geocenter`, but `IMPORT_MESH`
declares it in its `downstream` dict, so it is cleared only when a new mesh is loaded.

The recentre transforms `target_mesh`, `raw_pcd` and `cropped_pcd` in place, and
`clear_state_from(DOWNSAMPLE)` cannot undo that. If DOWNSAMPLE owned the record, re-running
Downsample after a recentre would reset `geocenter` to identity while the geometry stayed
moved — and the `geocenter_*` PLY comments, which are the exported provenance of the model
frame, would silently be wrong. Loading a new mesh resets geometry and record together,
which is the only point at which they are consistent.

## Headless mode

Every stage checks `self.app.headless` at the top of `build_panel()` and `_refresh_ui()` and returns early. `worker()` runs identically headless or GUI.

```python
app = MeshSamplingApp(headless=True, mesh_path="part.STL")
app.stages[Stage.IMPORT_MESH]._run_worker()
# Default: auto-size count to ~60% volumetric fill (set generate_mode="count" to override).
app.stages[Stage.SCENE].fill_rate = 0.6
app.stages[Stage.SCENE]._run_worker()    # build + settle physical scene -> app.o3d_scene
app.stages[Stage.RENDER]._run_worker()   # sensor sim + segmentation -> app.synthetic_targets
```

For a structured scene with fixtures, set on `SceneStage` before the worker: `arrangement="structured"`,
`structure_type` (`"none"|"partition"|"tray"`), `structure_height_pct` (50–100), `clearance_mode`
(`"snug"|"medium"|"loose"`), and `stable_pose_R` (one scene per stable pose).

## Threading rules

- **Never touch Open3D GUI from `worker()`**: O3D GUI is not thread-safe. Route all GUI mutations through `self.app.main_thread(lambda: ...)`.
- **`SceneStage` must join `decompose_thread`** before constructing `MujocoBinScene`. The convex decomposition from `DecomposeStage`/`ImportMeshStage` runs in a daemon thread and may still be running when `SceneStage` starts.

## Constraints

- **Stages orchestrate, never implement**: domain logic (sensor physics, MuJoCo setup, geometry math) belongs in `sensor/`, `physics/`, `geometry/`. If domain logic appears in a stage file, it is in the wrong place.
- **`SceneStage` and `RenderStage` are the integration points**: `SceneStage` calls `physics/`, `RenderStage` calls `sensor/`. The noise call sequence in `RenderStage` is intentional — do not reorder.
