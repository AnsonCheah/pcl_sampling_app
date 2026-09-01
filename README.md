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
geometry/                       base layer (no local imports)
    ↑
sensor/  physics/               geometry only
    ↑
stages/                         geometry, sensor, physics, MM_Optimizer
    ↑
app.py                          stages only

MM_Optimizer/                   geometry; drives MechVision via the mm_adapter package
registration/ppf/ + _shared/    no local imports at all — copyable out as a pair
registration/ppf_saliency/      geometry + registration.ppf + _shared
bench/                          top tier; may import everything
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

## Pipeline stages

The nine stages of `enums.Stage`, in pipeline order:

| # | Stage | Output |
|---|-------|--------|
| 1 | **Import Mesh** | Loaded STL, unit-normalised and repaired |
| 2 | **Raycast** | Multi-view canonical point cloud |
| 3 | **Crop** | Outlier-free point cloud (GUI only) |
| 4 | **Downsample** | Uniform or adaptive voxel-downsampled cloud + ambiguity profile |
| 5 | **Save** | PLY with geocenter and ground-truth pose metadata in comments |
| 6 | **Decompose** | Convex decomposition (VHACD) for physics collision |
| 7 | **Scene** | MuJoCo bin arrangement, settled under gravity |
| 8 | **Render** | Structured-light sensor sim + segmentation → labelled scene PLY/NPZ |
| 9 | **Tuning** | Optuna search over MechVision parameters against the synthetic GT |

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
| [geometry/](geometry/) | 3D math, file I/O, ambiguity analysis, `O3DSceneObject` |
| [sensor/](sensor/) | Structured-light depth sensor simulation and noise chain |
| [physics/](physics/) | MuJoCo rigid-body bin simulation (piles, partitions, trays) |
| [registration/](registration/) | From-scratch PPF pose matcher, plus a weighted-voting variant |
| [MM_Optimizer/](MM_Optimizer/) | Optuna tuner for MechVision pose-estimation parameters |
| [stages/](stages/) | GUI panels and pipeline orchestration |
| [bench/](bench/) | Batch scene generation, dataset fetch, ambiguity validation |

## Dependencies

Key libraries (see [environment.yaml](environment.yaml) for full pinned versions):

- `open3d` — geometry, raycasting, GUI
- `mujoco` — rigid-body physics simulation
- `trimesh` + `vhacdx` — mesh loading and convex decomposition
- `optuna` — MechVision parameter tuning (`gp` sampler needs `torch`)
- `numpy`, `scipy`, `scikit-learn` — numerical utilities
