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
app.py             ← stages only
```

Domain logic lives in its domain package. `stages/` orchestrates but does not implement physics or sensor math.

## Getting started

### Install

```bash
conda env create -f environment.yaml
conda activate autotune
```

Every command below then runs as plain `python`, on any platform. If you invoke the
interpreter directly instead of activating, point an environment variable at it rather than
hardcoding the path — the location is install- and OS-specific:

```bash
export PCL_PY="$LOCALAPPDATA/anaconda3/envs/autotune/python.exe"   # Git Bash on Windows
export PCL_PY="$HOME/miniconda3/envs/autotune/bin/python"          # Linux / macOS
```

### GUI

```bash
python app.py
```

### Headless (interactive)

```bash
python app.py --headless                 # prompts for mesh + options
python app.py --headless --mesh part.STL  # pre-seed the mesh, skip the prompt
```

### Headless (scripted)

```python
from app import MeshSamplingApp
from enums import Stage

app = MeshSamplingApp(headless=True, mesh_path="part.STL")
app.stages[Stage.IMPORT_MESH]._run_worker()

app.stages[Stage.DOWNSAMPLE].use_adaptive = True
app._express_sampling_worker()

# Default: auto-size part count to ~60% volumetric fill of the bin.
app.stages[Stage.SCENE].fill_rate = 0.6
# To force an exact count instead:
#   app.stages[Stage.SCENE].generate_mode = "count"
#   app.stages[Stage.SCENE].num_targets = 6
app.stages[Stage.SCENE]._run_worker()     # build + settle physical scene
app.stages[Stage.RENDER]._run_worker()    # sensor sim + segmentation -> synthetic targets
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
