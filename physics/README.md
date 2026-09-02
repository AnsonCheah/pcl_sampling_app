# physics

MuJoCo-based rigid-body simulation of parts settling in a bin. Produces collision-free part placements with ground-truth poses.

## Contents

| File | Description |
|------|-------------|
| `mujoco_bin_scene.py` | `MujocoBinScene` class -- scene construction, gravity settling, pose extraction |

## `MujocoBinScene`

```python
class MujocoBinScene:
    def __init__(
        self,
        part_mesh,                          # trimesh.Trimesh -- the CAD part
        part_convex_meshes,                 # list[trimesh.Trimesh] -- convex hull decomposition
        n_parts: int = 1,
        bin_dim: tuple = (0.76, 0.585, 0.25, 0.005),  # (W, D, H, wall_thickness) metres
        settle_time: float = 10.0,          # simulation seconds to run
        render: bool = True,                # show passive MuJoCo viewer
        arrangement: str = "random",        # "random" or "stable_pose"
        stable_pose_R = None               # pre-computed stable rotation matrices
    )
```

### Workflow

```python
scene = MujocoBinScene(part_mesh, convex_meshes, n_parts=8)
scene_state = scene.simulate()                      # run gravity settling
o3d_objects = scene.mujoco_scene_to_o3d(scene_state)  # -> {geom_id: O3DSceneObject}
```

### `scene_state` dict

After `simulate()`, `scene_state` contains one entry per simulated body:

```python
{
    "bin": {"mesh": ..., "T": ...},          # always present
    0:     {"mesh": ..., "T": ...},          # part instance 0
    1:     {"mesh": ..., "T": ...},          # part instance 1
    ...
}
```

`T` is a 4x4 world-frame transform. The `bin` key is always present -- `RenderStage` uses it to separate bin points from part points during labelling. Partition/tray fixtures are merged into `bin_mesh`, so they share the bin id and segment as background.

## Constraints

- **Final part count may be less than `n_parts`**: Collision-aware placement skips parts that cannot be placed without overlap after ~50 insertion attempts. Never assume `len(scene_state) - 1 == n_parts`.
- **`render=True` is blocking**: The passive MuJoCo viewer must not be used in headless mode. Pass `render=False` when running without a display.
- **`simulate()` is thread-safe**: MuJoCo has no main-thread requirement. `SceneStage` runs it in a background worker. Structured `none` is a no-op (static poses); `partition`/`tray` settle the parts under gravity.
- **Convex meshes come from `DecomposeStage`**, which is synchronous (VHACD in a `ProcessPoolExecutor`), so they are ready by the time `SceneStage` runs. No thread to join.
