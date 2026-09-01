# MM_Optimizer

Auto-tunes MechVision's 3D coarse + fine matching parameters for a part, scoring candidate
configurations against the synthetic ground truth this repo generates. Optuna is the only
tuner; the hand-written coordinate-descent optimizer it replaced has been removed.

Constraints an agent must not break are in [CLAUDE.md](CLAUDE.md).

## Files

| File | Role |
|---|---|
| `tuner.py` | `Tuner` — multi-objective Optuna study over all coarse + fine params, plus the CLI |
| `mv_evaluator.py` | `MVEvaluator` — the evaluation harness `Tuner` extends: scene sampling, param-dict construction, MechVision execution, GT matching, scoring, export |
| `search_config.py` | Pure data: search space, thresholds, scoring constants, sampler list |
| `mesh_analysis.py` | Geometry analysis → `WarmStart` seed parameters and symmetry classification |
| `model_sync.py` | Deploys a part's reference bundle into the MechVision model library |
| `eval_cache.py` | SHA-256 transposition table over (config, scenes, thresholds) |
| `optimizer_utils.py` | Scene listing, PLY GT parsing, **and the Python callbacks MechVision loads by absolute path** |
| `visualize_match.py` | Standalone diagnostic: re-runs a tuned config and overlays matched vs GT poses |

## Data flow

```
output/reference_pcd/<part>/        output/synthetic_target/<part>/scene_*/
        |                                          |
        | model_sync.sync_regime_model             | list_synthetic_scenes
        v                                          v
CAD_Match/resource/3d_matching/<part>/<part>.ply   scene dirs + GT from PLY headers
        |                                          |
        +---------------> MVEvaluator <------------+
                               ^
                               | extends
                            Tuner  ->  Optuna study  ->  results/<PART>_<SAMPLER>.db
                                                          results/<SAMPLER>_best_config_<part>.json
```

## How a run proceeds

1. **Warm start** — `mesh_analysis.analyze_mesh` derives seed parameters from part geometry.
2. **Regime gate** — `phase1_regime_gate` evaluates the surface and edge regimes sequentially,
   installing each cloud through `model_sync`, and keeps those above
   `REGIME_COVERAGE_GATE`. The winner is locked for the study.
3. **Joint study** — one multi-objective Optuna study over all 16–18 coarse and fine
   parameters. Objectives are `(coverage, mean_time)`; the winner is taken from the Pareto
   front, max coverage first and min time as tiebreaker.
4. **Symmetry sweep** — `phase4_symmetry`, run only if coverage clears
   `SYMMETRY_COVERAGE_GATE`.

Rounds extend the same study, so the sampler keeps its model across them.

## Samplers

`--sampler nsgaii | tpe | gp`, default **`gp`** (one definition, in `search_config.py`). All
three are multi-objective. They differ in the trial budget they need and in how they handle
this study's conditional and shifting search space — `Tuner._create_study` documents the
trade-off against Optuna's published guidance, and it is why `gp` is the default at this
project's ~150–250 trial budget.

## Scoring

```
raw_score     = mean_time / SCORE_TIME_NORM + (1 - coverage) * SCORE_COV_NORM
score_quality = 1 - raw_score / SCORE_WORST_CASE          in [0, 1]
```

Lower `raw_score` is better. The regime gate scores position-only (`ang_thresh = 360°`)
because a rotationally symmetric part returns a valid but flipped pose; everything after it
uses the tight 5° gate.

## Pruning

Deliberately minimal, and never competitive:

- a coverage floor (`COV_PRUNE_FLOOR`) after 3+ scenes, and
- a **fixed** absolute per-trial time cap (`TIME_ABS_CAP`), a safety valve for pathological
  configs only.

There is no `best_mean_time × ratio` guard: it ratchets down after one fast low-quality trial
and then prunes most of the search, biasing away from the slow-but-accurate region that
matters here.

## CLI

```bash
python MM_Optimizer/tuner.py --part 25333MB000 \
    [--sampler gp] [--n_trials 150] [--n_rounds 2] \
    [--scenes_dir PATH] [--m_full N] [--no_cache] [--seed 42] \
    [--storage results/] [--dry_run]
```

`--dry_run` builds the parameter dicts and runs the study without calling MechVision.

The GUI wraps the same `Tuner` in `stages/tuning_stage.py`, which adds a live 3D overlay and
an Optuna-dashboard subprocess.

## `mesh_analysis.py` — WarmStart

Derives seed values (`refStep`, `distQuantification`, `angleQuantification`,
`maxNumOfPointPairsPerFeature`, voxel bounds) plus a symmetry classification from the
reference cloud, so the study starts from a plausible region rather than the middle of the
space. `load_reference_pcd` reads the app's own bundle — not the deployed MechVision library,
which holds only whichever regime was synced last.

## `eval_cache.py` — Transposition table

JSON-backed, keyed on a SHA-256 of (coarse params, fine params, scene signature, thresholds).
The scene signature folds each `sample_*.ply`'s `(name, size, mtime_ns)`, so regenerating
scenes into the same directory invalidates the entry rather than serving a stale score.
Disable with `ENABLE_CACHE = False` in `mv_evaluator.py`.

## `search_config.py`

Pure data, no imports from the adapter or the tuner: search-space bounds and choices, scoring
constants, the regime table, symmetry-classification thresholds and the sampler list. Edit
here to change what the study explores.
