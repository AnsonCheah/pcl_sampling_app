# PCL Sampling App

## Mission
Synthetic point cloud data generation for industrial bin-picking:
1. Auto-tune pose estimation parameters for thousands of parts — no manual physical testing
2. Generate training data for learned point cloud registration models

**Scale constraint**: Must work identically for part #5000 as part #1. Any solution requiring per-part manual configuration is rejected. "Easier to configure" is not the same as "automated".

## Architecture Principle
Parametric model first, learned model targets the residual. The CVAE+Flow is not a replacement for the parametric noise model — it learns what the parametric model cannot explain. Building learned-first produces redundant representations and masks systematic biases.

## Conda Env Locations
Use pcd-sampling for all scripts in the workspace.

base                   C:\Users\Hmgics\AppData\Local\anaconda3
pcd-sampling           C:\Users\Hmgics\AppData\Local\anaconda3\envs\pcd-sampling

use "C:\Users\Hmgics\AppData\Local\anaconda3\envs\pcd-sampling\python.exe" for running scripts

## Package Dependency Graph
```
geometry/          ← no local imports (base layer)
    ↑
sensor/            ← geometry only
physics/           ← geometry only
registration/      ← no local imports (standalone)
    ↑
stages/            ← geometry, sensor, physics (not registration yet)
    ↑
app.py             ← stages only
```


## Cross-Domain Contracts

**`O3DSceneObject`** (defined in `geometry/geom_utils.py`) — the universal geometry carrier.
`id` field is unset until `sensor.scene_render()` assigns it as a side effect of adding geometry to the raycast scene. Do not read `id` before calling `scene_render`.

**`render dict`** — output of `sensor.scene_render()`, input to all noise functions.
Keys: `points, normals, geom_ids, pixel_idx, cos_cam, cos_proj, snr_proxy, depth_img, res, sensor_origin`.
Noise functions return modified arrays; they do not return a new render dict.

**`app` state** — the only communication channel between stages. Stages never call each other directly. See `stages/CLAUDE.md` for the full attribute table.

**PLY comment keys** — load-bearing, used by downstream loaders. Do not rename:
`geocenter_x/y/z`, `geocenter_qx/qy/qz/qw`, `gt_x/y/z`, `gt_qx/qy/qz/w`

## Phase Status
- Phase 1 (Parametric baseline): ✓ Complete — sensor model, MuJoCo sim, segmentation sim, NPZ export
- Phase 2 (PD-Flow + Noise Flow): ○ Not started
- Phase 3 (CVAE+Flow): ○ Not started

**Reference implementations**: SAPIEN active stereo sensor, PD-Flow, PointFlow, Noise Flow, ScoreDenoise, DREDS


## User Preference
1. When diagnosing issues or bugs, DO NOT GUESS, spawn agents to check resources online to verify hypothesis
2. When planning for implementations, ask as much clarifying questions as possible BEFORE creating a plan, and BEFORE modifying plan from user feedbacks
3. As much as possible, add test cases for new implementations for closing the build-test loop
4. Never commit or push */CLAUDE.md into `main` branch, keep them in the `develop` branch, and any feature should branch from `develop` or subbranch of it to utilize CLAUDE.md