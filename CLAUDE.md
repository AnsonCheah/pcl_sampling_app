# PCL Sampling App

## Mission
Synthetic point cloud data generation for industrial bin-picking:
1. Auto-tune pose estimation parameters for thousands of parts -- no manual physical testing
2. Generate training data for learned point cloud registration models

**Scale constraint**: Must work identically for part #5000 as part #1. Any solution requiring per-part manual configuration is rejected. "Easier to configure" is not the same as "automated".

## Architecture Principle
Parametric model first, learned model targets the residual. The CVAE+Flow is not a replacement for the parametric noise model -- it learns what the parametric model cannot explain. Building learned-first produces redundant representations and masks systematic biases.

## Conda Env
**One env: `autotune`**, defined by `environment.yaml`. (`pcd-sampling` is obsolete -- if you
find it named anywhere, that reference is stale.)

**`mujoco >= 3.8` is a hard floor, enforced at `physics/mujoco_bin_scene.py` import time**
(pinned to 3.12.0). Below it, multi-point mesh contacts are opt-in and the module's deliberate
silence would disable them -- degraded pile physics with no error. Rebuild the env rather than
working around the check.

**Never hardcode an interpreter path in code, docs or docstrings.** The install location is
user- and OS-specific, and Windows and Linux do not even agree on the shape
(`<env>\python.exe` vs `<env>/bin/python`), so a literal path cannot be correct in both. Use
the `PCL_PY` environment variable, set once per machine:

```bash
export PCL_PY="$LOCALAPPDATA/anaconda3/envs/autotune/python.exe"   # Git Bash on Windows
export PCL_PY="$HOME/miniconda3/envs/autotune/bin/python"          # Linux / macOS
```
```powershell
$env:PCL_PY = "$env:LOCALAPPDATA\anaconda3\envs\autotune\python.exe"
```

Then `"$PCL_PY" -m pytest ...`. Note `conda` is **not on PATH** in Git Bash or PowerShell here,
so `conda run` / `conda activate` are not available as a substitute.

**Optional dependency, deliberately not installed.** `bop_toolkit_lib` is commented out in
`environment.yaml` and must be installed `--no-deps` (its pyproject pins `numpy<2.0.0`, which
would drag open3d and scipy down with it). Its imports in
`registration/ppf_saliency/bench/metrics.py` are **function-local**, so the module imports
fine without it and only the calls (`evaluate_pose`, `symmetry_transforms_from_bop`) raise --
which is what stops the `ppf_saliency` ablation and one test in `registration/tests`.
`registration/ppf` implements the same metrics directly and is unaffected; that is the point
of its dependency pin.

## Package Dependency Graph
```
registration/ppf/ + registration/_shared/
                   <- NO local imports at all, not even `registration.*`. The vanilla PPF
                     matcher, its benchmark harness (ppf/bench/), and the backend/frames
                     code it shares with ppf_saliency. Copyable into another project as a
                     PAIR; relative imports are mandatory inside both, and three tests in
                     registration/tests/test_ppf.py enforce it.

geometry/          <- no local imports (base layer)
    ^
sensor/            <- geometry only
physics/           <- geometry only
registration/ppf_saliency/
                   <- geometry + registration.ppf.bench.dataset + registration._shared. The
                     weighted-voting variant and its ablation harness. NOT standalone and
                     not part of the copyable unit: its arms are defined by the ambiguity
                     heat map.
    ^
stages/            <- geometry, sensor, physics, and MM_Optimizer (TuningStage only)
    ^
app.py             <- stages only

MM_Optimizer/      <- geometry; drives MechVision through the mm_adapter pip package.
                     Imported by stages/tuning_stage.py and bench/generate_scenes.py.

bench/             <- top tier; may import everything. Scene GENERATION, dataset fetch, and
                     MODEL validation (ambiguity vs BOP, dynamic bin vs max bin) only --
                     matcher benchmarking lives in the packages.
```

`DecomposeStage` runs VHACD in a `ProcessPoolExecutor` and blocks on the result -- a normal
synchronous stage, with no thread to join.


## Cross-Domain Contracts

**`O3DSceneObject`** (defined in `geometry/geom_utils.py`) -- the universal geometry carrier.
`id` field is unset until `sensor.scene_render()` assigns it as a side effect of adding geometry to the raycast scene. Do not read `id` before calling `scene_render`.

**`render dict`** -- output of `sensor.scene_render()`, input to all noise functions. The
field table lives in that module's docstring and nowhere else; do not copy it. Noise
functions return modified arrays, not a new render dict, and their **call order is
load-bearing** -- see `sensor/CLAUDE.md`.

**`app` state** -- the only communication channel between stages; stages never call each
other directly. Each stage declares the attributes it owns in a `downstream` class dict,
which is also what seeds them before any panel is built. Full table: `stages/README.md`.

**PLY comment keys** -- load-bearing, used by downstream loaders. Do not rename:
`geocenter_x/y/z`, `geocenter_qx/qy/qz/qw`, `gt_x/y/z`, `gt_qx/qy/qz/w`

## Phase Status
- Phase 1 (Parametric baseline): [x] Complete -- sensor model, MuJoCo sim, segmentation sim, NPZ export
- Phase 2 (PD-Flow + Noise Flow): [ ] Not started
- Phase 3 (CVAE+Flow): [ ] Not started

**Reference implementations**: SAPIEN active stereo sensor, PD-Flow, PointFlow, Noise Flow, ScoreDenoise, DREDS


## User Preference
1. When diagnosing issues or bugs, DO NOT GUESS, spawn agents to check resources online to verify hypothesis
2. When planning for implementations, ask as much clarifying questions as possible BEFORE creating a plan, and BEFORE modifying plan from user feedbacks
3. As much as possible, add test cases for new implementations for closing the build-test loop
4. Never commit or push */CLAUDE.md into `main` branch, keep them in the `develop` branch, and any feature should branch from `develop` or subbranch of it to utilize CLAUDE.md