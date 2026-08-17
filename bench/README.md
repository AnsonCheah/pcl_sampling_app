# bench

Repo-tier benchmarking: things that need the **full pipeline** (MuJoCo, the sensor simulator,
the segmentation stage) or that validate a repo module against external ground truth.

**Matcher benchmarking does not live here any more.** It moved next to the code it measures:

| Was | Now |
|---|---|
| `bench/dataset.py` | `registration/ppf/bench/dataset.py` |
| `bench/metrics.py` | `registration/ppf_saliency/bench/metrics.py` (a standalone reimplementation lives in `registration/ppf/bench/metrics.py`) |
| `bench/ablation.py`, `arms.py`, `visualize_ppf.py` | `registration/ppf_saliency/bench/` |
| `registration/tests/test_ppf_saliency_bench.py` | `registration/tests/test_ppf_saliency_bench.py` |

A package whose accuracy claims can only be reproduced from a directory two levels above it
is not really shippable, and `registration/ppf/` is meant to be liftable into another project.
What stayed here is what genuinely cannot move.

## Files

| File | Purpose |
|---|---|
| `fetch_dataset.py` | Download benchmark meshes into `mesh_raw/` (T-LESS: 30 objects + symmetry GT, CC BY 4.0) |
| `generate_scenes.py` | Batch-drive the app pipeline over a mesh directory to produce scenes with GT poses |
| `validate_ambiguity.py` | Check `geometry/ambiguity.py` against BOP's published symmetry annotations |

Reserved for an **Optuna tuning benchmark** (deferred): scoring MechVision parameter searches
from `MM_Optimizer/`. That is what this directory is for — driving a black box that is not
part of any package here.

## Workflow

```bash
python bench/fetch_dataset.py --dataset tless             # 30 industrial meshes + symmetry GT
python bench/validate_ambiguity.py --verbose              # ambiguity module vs BOP
python bench/generate_scenes.py --meshes mesh_raw/tless --scenes 4

# then, from the packages:
python -m registration.ppf.bench.run --all \
    --scenes-root output/synthetic_target \
    --models-info mesh_raw/tless/models_info.json
python -m registration.ppf_saliency.bench.ablation --all --out ablation.json
```

`validate_ambiguity.py` runs straight after the fetch: it analyses the BOP meshes directly and
needs no scenes. It is minutes per part, so use `--parts obj_000005 ...` to spot-check.

Scene generation is the slow step (~3 min per part per scene, most of it VHACD convex
decomposition, which is one-time per part) and is **resumable** — a part that already has
enough scenes is skipped, and one that fails is logged and stepped over. `--no-ambiguity`
saves 1-3 min per part, but it is **not** merely a speed switch: it changes the exported model
frame from ambiguity-aligned to PCA, so scenes produced with and without it are not
interchangeable. It no longer starves any downstream arm — `ablation.py` recomputes the
analysis itself, from each part's exported STL.

Raw output goes to `output/synthetic_target/`, which is gitignored.

## Reading any of these numbers

- **Compare only within a block.** Parts differ enormously in difficulty.
- **Two arms differ only if their CIs do not overlap.** At 26 instances the half-width on a
  ~0.5 recall is +/-19 points, wider than most arm differences — hence the 100-instance cap
  per part and the 30-part sweep.
- **`BOP mssd<0.2D` is the headline**, not `@2mm/5deg`. The latter is MechVision's coarse
  *plus fine* target; PPF alone is a coarse stage and scoring it there compresses every arm
  toward zero.
- **`margin`** is accumulator health (`1 - runner_up/peak`). Low recall with a *high* margin
  means confidently wrong — the signature of a model that lost coverage and is aliasing.
- **`tag`** is the share of detections that are failures a view-dependent ambiguity axis
  predicted. It is set only on instances missing the LOOSE gate, so it is bounded by the miss
  rate, not by 1.

## Gotchas

**`normal_radius` is required, not defaulted, in `load_scene`.** It moves recall by ~20
points (2*tau vs 4*spacing measured 0.77 vs 0.62), so a quiet default would become the most
influential untracked parameter in the benchmark. Pass `2 * cfg.tau`.

**Use `sample_i.npz["T_gt"]`, never `scene_state.npz["T_gt"]`.** They disagree by a
mesh-recentring shift; only the per-sample one shares a frame with `reference_cloud.ply`.

**`sample_<i>` does not necessarily correspond to `part_<i>`.** Samples are numbered by a
counter over instances passing the 2D filter.

**`bop_toolkit` must be installed with `--no-deps`** — its pyproject pins `numpy<2.0.0` and a
normal install downgrades numpy, open3d and scipy together. `registration/ppf/bench/metrics.py`
deliberately does not use it at all; `registration/ppf_saliency/bench/metrics.py` still does.

**`fetch_dataset.py` needs `requests` + `certifi`.** `requests` is imported at module scope,
so it is a hard dependency of that script; `environment.yaml` pins it explicitly.
