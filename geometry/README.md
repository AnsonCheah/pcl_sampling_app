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
gt_qx, gt_qy, gt_qz, gt_qw
```

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
- **`pcd_geocenter()` returns a 4×4 transform with translation in row 3** (not column 3 as is conventional). Callers must account for this.
