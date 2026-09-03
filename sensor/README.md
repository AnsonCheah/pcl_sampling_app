# sensor

Structured-light depth sensor simulation. Takes a set of meshes and a camera pose, produces a point cloud with physically-motivated noise.

## Contents

| File | Description |
|------|-------------|
| `scene_render.py` | Pure canonical raycast -> `render dict`; all noise functions |
| `segment_instances.py` | 2D instance segmentation error simulation |

## Core concept

The noise pipeline is applied in a fixed order in `RenderStage.worker()`:

```
scene_render()                  # canonical raycast (pure, no noise)
    v
compute_dropout_mask()          # angle/roughness/albedo-based point removal
    v
add_image_space_effects()       # smoothing, fringe correlation
    v
add_edge_artifacts()            # depth discontinuity blooming
    v
add_multipath_outliers()        # multipath interference points
add_pepper_noise()              # random outliers
    v
add_scan_line_banding()         # structured-light fringe banding
    v
add_sensor_noise()              # per-point depth noise (angle, distance dependent)
    v
add_surface_noise()             # surface micro-roughness perturbation
```

**Call order is load-bearing.** The sequence is physically motivated -- later stages depend on the output of earlier ones.

## `render dict`

`scene_render()` returns this dictionary. All noise functions take it as input and modify arrays in-place (they do not return a new dict):

| Key | Shape | Description |
|-----|-------|-------------|
| `points` | `(N, 3)` | World-space hit positions |
| `normals` | `(N, 3)` | Surface normals |
| `geom_ids` | `(N,)` | Geometry instance IDs |
| `pixel_idx` | `(N,)` | Original raster pixel indices |
| `cos_cam` | `(N,)` | Cosine of incidence angle to camera |
| `cos_proj` | `(N,)` | Cosine of incidence angle to projector |
| `snr_proxy` | `(N,)` | SNR estimate |
| `depth_img` | `(H, W)` | Depth map |
| `res` | `(H, W)` | Resolution tuple |
| `sensor_origin` | `(3,)` | Camera position in world space |

## Instance segmentation simulation

`segment_instances.py` simulates errors a real 2D segmentation network would make:

- **Erosion / dilation** -- mask boundary shifts
- **Instance confusion** -- nearby instances merged or split
- **Occlusion-edge gaps** -- missing points at depth discontinuities
- **Soft-mask threshold noise** -- stochastic boundary pixels

Operations happen in 2D image space, then back-projected to 3D. This matches how real segmentation errors arise.

## API

```python
from sensor.scene_render import scene_render

render = scene_render(
    meshes,        # list[O3DSceneObject]
    T_cam,         # 4x4 camera extrinsic
    look_at,       # (3,) target point
    fov,           # horizontal field of view (degrees)
    W, H           # image resolution
)
```

## Constraints

- **`scene_render()` is pure**: its only side effect is assigning `O3DSceneObject.id`. It must remain stateless -- it is called by both `RaycastStage` and `RenderStage`.
- **Noise functions are independently callable**: Phase 2 will fit each function's parameters against real-data residuals one at a time. Do not chain them internally.
- **Parameters are physical quantities**: `roughness`, `albedo`, `sigma_fringe_corr` must correspond to measurable material/sensor properties. Non-physical tuning knobs belong in the Phase 2 learned residual.
