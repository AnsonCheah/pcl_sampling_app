# physics

MuJoCo-based rigid-body simulation of parts settling in a bin. Produces collision-free part placements with ground-truth poses.

## Contents

| File | Description |
|------|-------------|
| `mujoco_bin_scene.py` | `MujocoBinScene` class -- scene construction, gravity settling, pose extraction; `solve_bin_dim()` dynamic bin sizing |
| `profile_structured_scene.py` | Profile build + settle across random / partition / tray to find where the time goes |

## MuJoCo version floor

`mujoco >= 3.8` is enforced at import (`MIN_MUJOCO_VERSION`); `environment.yaml` pins 3.12.0.
Multi-point mesh contacts (multiccd) became default-on in 3.8.0 and the opt-in
`mjENBL_MULTICCD` was removed from the enable set, so this module sets nothing. On an older
MuJoCo that same silence means multiccd is **off** -- flat-faced parts get single-point contacts
and the pile settles wrong, with no error anywhere. Hence the hard failure rather than a
fallback. Sleeping islands (`mjENBL_SLEEP`, 3.4+) are a separate flag and stay **off**; see
`physics/tests/test_sleep_equivalence.py` for the measurements and the gate to re-run.

## Dynamic bin sizing

`solve_bin_dim(part_mesh, fill_rate)` derives the bin from the part instead of the part count
from a fixed bin, and returns `(bin_dim, n_parts, layers_at_fill)`. The bin is solved first and
is authoritative: every lower bound is expressed as a bin dimension, the largest wins, and the
count is read off the final bin with the unchanged fill formula.

`packing` (`obb_packing_factor`) is a **volume** fraction and is spent in exactly two places:
`h_eff = layer_h / packing` (vertical bridging) and the count formula. It must **not** enter the
footprint term -- `a_parts = MIN_PARTS_PER_LAYER * obb_vol / layer_h`, no packing. Dividing there
as well made the realised parts-per-layer `MIN_PARTS_PER_LAYER / packing`, which is
shape-dependent and worst for low-packing parts: a 50x40x5 plate was handed ~128 footprints per
layer instead of 25 and its bin barely shrank. Locked by
`physics/tests/test_dynamic_bin.py::test_parts_per_layer_is_shape_independent`.

`MAX_RELEASE_BATCHES` caps release waves. It fires on the **unshrunken** bin too, where it is a
change to settling dynamics rather than a rescue (500 parts: 50 waves of 10 -> 8 of 63, 25 s of
staged release down to 4 s). The `max_release_batches` constructor argument and
`bench/validate_bin_equivalence.py --batch-cap` exist so that change can be measured.

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
