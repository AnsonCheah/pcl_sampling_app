# geometry

Base layer for 3D math, file I/O, and the shared `O3DSceneObject` dataclass. No local imports — all other packages depend on this one.

## Contents

| File | Description |
|------|-------------|
| `geom_utils.py` | `O3DSceneObject` dataclass, Open3D helpers, trimesh↔Open3D conversion |
| `file_utils.py` | PLY export with comment metadata, file-dialog helpers |
| `math_utils.py` | Curvature analytics (`plot_curvature_cdf`, `find_cdf_knee`) |
| `debug_utils.py` | Ray visualisation (`visualize_rays`, `visualize_projector_rays`) |

## `O3DSceneObject`

Universal geometry carrier passed between all packages:

```python
@dataclass
class O3DSceneObject:
    geom: o3d.geometry.Geometry3D
    ref_geom: Optional[o3d.geometry.Geometry3D] = None
    material: Optional[rendering.MaterialRecord] = None
    id: Optional[int] = None       # assigned by scene_render() as a side effect
    T_gt: Optional[np.ndarray] = None
    xyz0: Optional[np.ndarray] = None
    xyz1: Optional[np.ndarray] = None
    overlap: Optional[float] = None
```

**`id` is `None` until `sensor.scene_render()` assigns it.** Do not read `.id` before calling `scene_render`.

## PLY comment keys

`file_utils.py` writes ground-truth pose data as PLY header comments. These keys are parsed by downstream loaders — do not rename them:

```
geocenter_x, geocenter_y, geocenter_z
geocenter_qx, geocenter_qy, geocenter_qz, geocenter_qw
gt_x, gt_y, gt_z
gt_qx, gt_qy, gt_qz, gt_w
```

Note the ground-truth scalar term is `gt_w`, **not** `gt_qw` — that is what `render_stage.py` writes and what `MM_Optimizer/optimizer_utils.read_gt_pose_from_ply` requires.

The `geocenter_*` keys carry the model frame expressed in the pre-recentre frame, i.e. where
the geocenter origin sat before `recenter_mesh_pcd` moved everything. They read as identity on
a bundle that was never recentred, which is honest: nothing was applied.

## Key helpers

```python
# Open3D display wrapper (blocking)
o3d_display(geometries: list) -> o3d.visualization.Visualizer

# Uniform random rotation
random_rotation_matrix() -> np.ndarray  # 4×4

# Camera extrinsic from position + look-at
camera_view_matrix(eye, center, up) -> np.ndarray  # 4×4

# Curvature-based knee detection for adaptive downsampling
find_cdf_knee(curvatures: np.ndarray) -> float
```

## Constraints

- **One-way imports**: this package must never import `sensor`, `physics`, `registration`, or `stages`.
- **`pcd_geocenter()` returns a 4×4 transform, not a point** — the matrix you apply to the cloud to put it in the frame. Translation is in **column 3** (`[:3, 3]`), row 3 is `[0, 0, 0, 1]`. (This note previously claimed row 3; it was wrong, and `SaveStage` read row 3 for years, writing `geocenter_x/y/z = 0.0` into every exported PLY.)
- **`reference_frames_agree(a, b)`** — returns `None` when two reference clouds are the same export in the same model frame, else a reason. The model frame is reproducible only while the mesh *and* the sampling settings are unchanged, so any code pairing a reference cloud with generated scenes should check.
