# pcl_sampling_app

Synthetic point cloud data generation for industrial bin-picking pose estimation.

## What it does

Two primary outputs:

1. **Reference point clouds** — clean, downsampled PLYs derived from a CAD mesh. Used as templates for pose estimation algorithms.
2. **Synthetic scene point clouds** — simulated structured-light scans of parts dropped in a bin, with realistic sensor noise and ground-truth pose annotations. Used to auto-tune registration parameters and generate training data for learned models.

The pipeline is designed to scale to thousands of part geometries without any per-part manual configuration.

## Architecture

The codebase is split into domain packages with a strict one-way dependency graph:

```
geometry/          ← base layer (no local imports)
    ↑
sensor/            ← geometry only
physics/           ← geometry only
registration/      ← no local imports (standalone)
    ↑
stages/            ← geometry, sensor, physics
    ↑
app_v2.py          ← stages only
```

Domain logic lives in its domain package. `stages/` orchestrates but does not implement physics or sensor math.

## Getting started

### Install

```bash
conda env create -f environment.yaml
conda activate pcd-sampling
```

Python interpreter: `C:\Users\Hmgics\AppData\Local\anaconda3\envs\pcd-sampling\python.exe`

### GUI

```bash
python app_v2.py
```

### Headless (scripted)

```python
from app_v2 import MeshSamplingApp
from enums import Stage

app = MeshSamplingApp(headless=True, mesh_path="part.STL")
app.stages[Stage.IMPORT_MESH]._run_worker()

app.stages[Stage.DOWNSAMPLE].use_adaptive = True
app._express_sampling_worker()

app.stages[Stage.SYNTHETIC].num_targets = 6
app.stages[Stage.SYNTHETIC]._run_worker()
```

See [headless_app_example.py](headless_app_example.py) for a runnable example.

## Pipeline stages

| # | Stage | Output |
|---|-------|--------|
| 1 | **Import Mesh** | Loaded STL + convex decomposition |
| 2 | **Raycast** | Multi-view canonical point cloud |
| 3 | **Crop** | Outlier-free point cloud |
| 4 | **Downsample** | Uniform or adaptive voxel-downsampled cloud |
| 5 | **Save** | PLY with ground-truth pose metadata in comments |
| 6 | **Synthetic** | MuJoCo bin sim + structured-light noise → labelled scene PLY |

## Key contracts

**`O3DSceneObject`** (`geometry/geom_utils.py`) — universal geometry carrier. The `.id` field is `None` until `sensor.scene_render()` assigns it.

**`render dict`** — output of `sensor.scene_render()`, consumed by all noise functions. Keys: `points, normals, geom_ids, pixel_idx, cos_cam, cos_proj, snr_proxy, depth_img, res, sensor_origin`.

**PLY comment keys** — load-bearing; downstream loaders parse them by name. Do not rename: `geocenter_x/y/z`, `geocenter_qx/qy/qz/qw`, `gt_x/y/z`, `gt_qx/qy/qz/w`.

**`app` state** — the only communication channel between stages. Stages never call each other directly.

## Phase status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Parametric baseline: sensor model, MuJoCo sim, segmentation sim, NPZ export | ✓ Complete |
| 2 | PD-Flow + Noise Flow | ○ Not started |
| 3 | CVAE + Flow | ○ Not started |

## Package overview

| Package | Description |
|---------|-------------|
| [geometry/](geometry/) | 3D math, file I/O, `O3DSceneObject` dataclass |
| [sensor/](sensor/) | Structured-light depth sensor simulation and noise chain |
| [physics/](physics/) | MuJoCo rigid-body bin simulation |
| [registration/](registration/) | Geometry-derived pose estimation parameter auto-tuning |
| [stages/](stages/) | GUI panels and pipeline orchestration |

## Dependencies

Key libraries (see [environment.yaml](environment.yaml) for full pinned versions):

- `open3d` — geometry, raycasting, GUI
- `mujoco` — rigid-body physics simulation
- `trimesh` + `vhacdx` — mesh loading and convex decomposition
- `optuna` — parameter optimisation
- `numpy`, `scipy`, `scikit-learn` — numerical utilities
