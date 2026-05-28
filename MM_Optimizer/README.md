# MM_Optimizer

Black-box optimizer for MechVision pose estimation parameters. Iterates: set parameters → trigger vision run → compare poses to synthetic ground truth → score → repeat.

The optimizer has no visibility into MechVision internals. It only observes pose outputs and timing.

## Files

| File | Description |
|------|-------------|
| `optimizer.py` | 6-phase hierarchical coordinate descent optimizer |
| `optuna_optimizer.py` | Alternative: fully joint NSGA-II multi-objective optimizer |
| `search_config.py` | All phase tables, thresholds, and search bounds (pure data) |
| `mesh_analysis.py` | Phase 0: geometry analysis → warm-start parameters |
| `eval_cache.py` | Transposition table (SHA-256 keyed, disk-persistent) |
| `optimizer_utils.py` | PLY loading, scene composition, GT pose extraction |
| `tests/` | Unit and integration tests |

## Data flow

```
Reference PLY
    ↓ mesh_analysis.py → WarmStart (diameter, distQ, angleQ, symmetry class)
    
Synthetic scenes (output/synthetic_target/<part>/scene_NNNNN/sample_*.ply)
    ↓ optimizer_utils.py → scene_groups, merged scene.ply, GT poses

optimizer.py (6 phases)
    ↓ coarse_params + fine_params
    → mm_adapter: set_params() + run_vision()  [gRPC to 127.0.0.1:5307]
    ← fine_poses, fine_confidences, timing
    ↓ match_poses_to_gt()
    → EvalResult: score, coverage, mean_time
    ↓ eval_cache
    
Final: best_config_<part>.json
```

## Hierarchical optimizer (`optimizer.py`)

Six phases of progressive refinement, each building on the previous result.

### Phase sequence

| Phase | What it searches | Strategy |
|-------|-----------------|----------|
| 0 | Geometry analysis | Derive `WarmStart` from reference PLY |
| 1 | Registration regime | Grid over 4 mode combinations (Surface/Edge × coarse/fine) |
| 2a | Coarse sampling | Joint grid: `refStep` × `distQuantification` |
| 2b | Remaining coarse params | Coordinate descent, 8 params in pipeline order |
| 3 | Fine matching params | Coordinate descent, 6 params, position-only scoring |
| 4 | Symmetry confirmation | Test N-fold axes if angular errors show periodicity |
| 5 | Joint coarse refinement | 3×3×3 grid with fine locked, top-K tiebreaking |
| 6 | Interval narrowing | Dense continuous sweep around Phase 5 winner |

### Scoring

```
raw_score  = (mean_time / 5.0) + (1 − coverage)
quality    = 1 − raw_score / 2.0   ∈ [0, 1]
```

Lower `raw_score` is better. Phase 5 uses soft tiebreaking on mean position error of passing instances.

Angular error threshold progresses through phases:
- Phases 1–3: position-only (angular gate = 360°)
- Phases 4–6: tight (angular gate = 5°)

### Two-pass multi-fidelity (Strategy 2)

Expensive candidates are pre-screened cheaply:

1. **Pass 1**: evaluate all candidates on `M_SMALL=5` scenes
2. **Pass 2**: full `M_FULL=30` evaluation on top `K_SURVIVORS=3`; early-exit at `TARGET_COVERAGE=0.90`

Disable with `--no_two_pass`.

### Phase gates (Strategy 3)

| Gate | Threshold | Action |
|------|-----------|--------|
| After Phase 1 | coverage < 0.50 | Stop — parameters cannot be tuned |
| After Phase 2a | coverage < 0.30 | Skip Phase 2b |
| After Phase 2 | coverage < 0.65 | Skip Phase 4 (symmetry correction) |
| After Phase 3 | coverage < 0.75 | Skip Phase 6 (interval narrowing) |

### CLI

```bash
python optimizer.py --part 25333MB000 \
    [--dry_run]           \  # mock MechVision calls for testing
    [--scenes_dir PATH]   \  # override synthetic scenes directory
    [--m_full N]          \  # override full-pass scene count
    [--no_cache]          \  # bypass transposition table
    [--no_two_pass]       \  # skip cheap pre-screening
    [--seed 42]           \
    [--export_best]          # write best_config_<part>.json
```

## Optuna optimizer (`optuna_optimizer.py`)

Alternative to hierarchical CD: a single joint Optuna study over all 16–18 coarse + fine parameters simultaneously.

- **Multi-objective**: maximise coverage, minimise mean_time (Pareto front)
- **Sampler**: NSGA-II (default) or TPE — set with `--sampler`
- **Phases 0, 1, 4** are delegated to a wrapped `Optimizer` instance; the joint study replaces Phases 2–3–5–6
- **Crash-resume**: study state persisted to SQLite (`--storage`)

### Constraints enforced during search

- `referredStep ≤ refStep` — hard feasibility; violations return sentinel (cov=0, time=5s)
- Time guard: prune trial if running time exceeds 3× current best
- Coverage floor: prune after 3 scenes if coverage < 0.10

### CLI

```bash
python optuna_optimizer.py --part 25333MB000 \
    [--n_trials_joint 150]       \
    [--n_rounds 2]               \
    [--sampler nsgaii | tpe]     \
    [--storage PATH]             \  # SQLite base path for crash-resume
    [--no_adaptive_thresh]       \
    [--pos_thresh_k 0.01]           # adaptive threshold scale factor
```

## `mesh_analysis.py` — WarmStart

Reads the reference PLY and derives geometry-based starting parameters before any MechVision calls.

```python
from MM_Optimizer.mesh_analysis import analyze_mesh

warm_start = analyze_mesh(ref_pcd, n_instances=1)
```

Key fields:

| Field | Derived from | Used for |
|-------|-------------|---------|
| `diameter_m` | Max spatial extent | Scales all geometry-relative params |
| `distQuantification` | — | Optimal PPF bin width (default 1.0) |
| `angleQuantification` | — | Hough accumulator resolution (default 60) |
| `minVoxelLength_mm` | 0.5% of diameter | Pose verification voxel grid |
| `maxVoxelLength_mm` | 2.0% of diameter | Pose verification voxel grid |
| `prefer_edge` | Normal concentration + flatness | Regime hint (edge vs surface) |
| `symmetry_class` | Eigenvalue + Chamfer analysis | One of: `ASYMMETRIC`, `C2`, `C3`, `C4`, `C6`, `SO2`, `SO3` |

Symmetry classification pipeline: PCA eigenvalue ratios → SO3/SO2 candidate detection → N-fold test via Chamfer distance at 180° rotations. Chamfer threshold = max(2% of diameter, 3 mm).

## `eval_cache.py` — Transposition table

Avoids re-evaluating parameter configurations that were already scored in a prior phase.

```python
key = EvalCache.make_key(config_dict, scene_paths)  # 16-char SHA-256 prefix
result = cache.get(key)   # None on miss
cache.put(key, result)
cache.save()              # persist to JSON (crash-safe)
```

Cache is only valid for live runs. Phase-specific angular thresholds must be included in the config dict so that Phase 2 (loose) and Phase 5 (tight) results are keyed separately.

## `optimizer_utils.py`

| Function | Purpose |
|----------|---------|
| `list_synthetic_scenes(part_dir)` | Returns `List[List[str]]` — one group per `scene_NNNNN/` directory |
| `compose_scene(ply_list, work_dir)` | Merges N sample PLYs into one `scene.ply`; extracts GT poses from PLY headers |
| `read_synthetic(path)` | Called by MechVision `Pre_Segmentation` step; returns `[x,y,z,nx,ny,nz,0]` arrays |
| `read_gt_pose_from_ply(ply_path)` | Parses `gt_x/y/z/qw/qx/qy/qz` from PLY comment lines |

## `search_config.py`

Central configuration — edit here to adjust search space without touching optimizer logic.

Key knobs:

| Constant | Default | Effect |
|----------|---------|--------|
| `POS_THRESH_TIGHT` | 2 mm | Position error gate for a detection to count as correct |
| `ANG_THRESH_TIGHT` | 5° | Angular error gate (Phases 4–6) |
| `M_SMALL` / `M_FULL` | 5 / 30 | Two-pass scene counts |
| `K_SURVIVORS` | 3 | Pass-1 → pass-2 survivors |
| `SCORE_TIME_NORM` | 5.0 s | Time normalisation denominator in scoring formula |
| `OPTUNA_N_TRIALS_JOINT` | 150 | Joint Optuna study trial budget |

## Constraints

- MechVision Hub must be running at `127.0.0.1:5307` before any evaluation.
- Ground truth comes exclusively from PLY comment headers written by `stages/SaveStage` — do not rename those keys.
- Phase gate thresholds (`0.50`, `0.65`, `0.75`) are empirically tuned; changing them requires re-validation against held-out scenes.
- The scoring formula (`time/5.0 + (1−coverage)`) is used consistently across all phases and both optimizers — changing it mid-run invalidates cached results.
