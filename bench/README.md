# bench

Benchmark harness for the PPF matcher: fetch meshes, generate scenes, score poses, run the
reference-cloud ablation. Top tier of the dependency graph — may import every other package.

## Workflow

```bash
python bench/fetch_dataset.py --dataset tless          # 30 industrial meshes + symmetry GT
python bench/generate_scenes.py --meshes mesh_raw/tless --scenes 4
python bench/ablation.py --all --max-instances 100 --out ablation.json
python bench/validate_ambiguity.py --verbose           # ambiguity module vs BOP
python bench/visualize_ppf.py                          # overlay poses on a scene
```

Scene generation is the slow step (~2 h for 30 parts x 4 scenes, ~8.8 GB) and is resumable —
a part that already has enough scenes is skipped, and one that fails is logged and stepped
over. Raw ablation JSON goes to `output/bench/`, which is gitignored.

## Files

| File | Purpose |
|---|---|
| `fetch_dataset.py` | Download benchmark meshes into `mesh_raw/` |
| `dataset.py` | Load `output/synthetic_target/<part>/scene_*/` |
| `generate_scenes.py` | Batch-drive the app pipeline over a mesh directory |
| `metrics.py` | Symmetry-aware pose error (MSSD / ADI / ADD via `bop_toolkit`) |
| `arms.py` | The reference-cloud / vote-weight variants under test |
| `ablation.py` | Run every arm over every part, stratified by symmetry class |
| `validate_ambiguity.py` | Check `geometry/ambiguity.py` against BOP annotations |
| `visualize_ppf.py` | Side-by-side pose overlay, one panel per arm |

## Results (2026-08, 30 T-LESS parts, 2789 instances/arm, CI +/-1.9 points)

| arm | BOP recall (MSSD<0.2D) | @2mm/5deg | MSSD p50 | s/inst |
|---|---|---|---|---|
| **A_uniform** | **0.66** | **0.35** | **5.65 mm** | 0.182 |
| F_ppf_weight | 0.64 | 0.29 | 7.06 mm | 0.240 |
| E_heat_weight | 0.56 | 0.20 | 13.29 mm | 0.240 |
| D_curv_heat | 0.40 | 0.09 | 32.5 mm | 0.102 |
| B_heat_prune | 0.32 | 0.07 | 40.4 mm | 0.089 |
| H_edge | 0.26 | 0.05 | 49.1 mm | 0.053 |
| C_curvature | 0.16 | 0.01 | 56.5 mm | 0.067 |

**Uniform coverage wins. No weighting or pruning scheme beat it.** Pruning arms are 2-3x
faster — the trade K-PPF reports — but cost 26-50 recall points here rather than the sub-1
point they claim. This is Birdal & Ilic's even-spacing argument (IROS 2017) confirmed at
n=2789, and it reproduces the repo's own Jan-Feb 2026 curvature-cloud finding with a much
sharper edge.

Per symmetry class, continuous parts are the weak spot everywhere: 0.44 against 0.81 for
discrete and 0.73 for asymmetric.

## Reading the ablation table

- **Compare only within a block.** Parts differ enormously in difficulty.
- **Two arms differ only if their CIs do not overlap.** At 26 instances the half-width on a
  ~0.5 recall is +/-19 points, wider than any arm difference — hence the 100-instance cap
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
normal install downgrades numpy, open3d and scipy together.
