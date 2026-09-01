# MM_Optimizer/

Black-box tuner: set MechVision parameters → trigger a vision run → score the returned poses
against synthetic GT → repeat. The tuner owns the loop; MechVision does not know it is being
optimised. See `MM_Optimizer/README.md` for the module map and how to run it.

## Non-Obvious Constraints

**`.vis` files are MechVision executables.** Do not edit them programmatically. They are also
opaque to grep, so a Python callback referenced only from a `.vis` step looks unused —
`optimizer_utils.py`'s `read_single_ply`, `pre_/post_coarse`, `pre_/post_fine` and
`reshape_coarse_pose_list` have no repo callers and must not be deleted on that basis.

**`optimizer_utils.py` is loaded by MechVision by absolute path** (`OPTIMIZER_UTILS_PATH`), as
a standalone file. It must import only stdlib and third-party packages — never anything from
this repository — which is why it keeps its own copies of the `sample_*.ply` scan. Do not
rename it.

**One library entry per part, holding one cloud.** Coarse and fine always share a cloud type
(`search_config.REGIMES`), so the regime is expressed by *which* cloud sits in
`CAD_Match/resource/3d_matching/<part>/<part>.ply`. `model_sync.sync_regime_model` rewrites
that folder per regime, so regimes must be evaluated **sequentially** — a parallel gate would
corrupt the library. Nothing is placed there by hand; the source bundle is
`output/reference_pcd/<part>/<part>_<type>/`.

**`geo_center.json` must be the identity pose `[0, 0, 0, 1, 0, 0, 0]`.** A wrong quaternion
there makes MechVision report every pose rotated off the synthetic GT — it presents as a
uniform `ang_error ≈ 179.7°` at the tight threshold, which looks exactly like unhandled
rotational symmetry. Check the geocenter before investigating symmetry. `SaveStage` writes
these values from `inv(app.geocenter)`.

**The eval cache key must include the accuracy thresholds.** `evaluate_config` runs at both the
loose regime-gate threshold (`ang_thresh = 360°`, position-only) and the tight one (5°). Omit
`pos_thresh`/`ang_thresh` from the key and a loose `coverage = 1.0` is served to a tight query,
hiding orientation errors entirely. `EvalCache._scene_signature` additionally folds each
`sample_*.ply`'s `(name, size, mtime_ns)` into the key, so regenerating scenes into the same
directory invalidates it — a path-only key returned stale results and hid a real world-Z offset.

**Only keys in `mv_evaluator._COARSE_TYPES` / `_FINE_TYPES` reach MechVision.** Anything else is
either Optuna search-space bookkeeping (listed in `_NON_MV_KEYS`) or a typo, and a typo now
warns instead of vanishing silently.

**Do not re-run MechMind's adapter generator for the fine registration parameters.** It fails
when generating an adapter for more than 30 numbers.

## Environment

- MechVision **1.8.3**, installed at `C:\Mech-Mind\Mech-Vision & Mech-Viz-1.8.3\`.
  Console logs: `…\Mech-Vision\logs\YYYY-MM-DD.log` — tail them, do not read whole files.
- `mm_adapter` is an installed pip package (see `environment.yaml`), not a directory in this
  repo. A stale local `mm_adapter/` folder on `sys.path` would shadow it.
- Step documentation: https://docs.mech-mind.net/en/suite-software-manual/1.8.3/vision-steps/steps.html
  — in particular `3d-coarse-matching-v2`, `3d-fine-matching-lite`, `calc-results-by-python`
  and `easy-create-string-list`.

## Objective, in priority order

1. Maximise pose precision against the synthetic ground truth.
2. Minimise pose error against the GT poses carried in the synthetic PLY headers.
3. Minimise cycle time — only as a tiebreaker once precision meets the tolerance.

GT poses come from the PLY files the sampling app writes. The tuner reads those files; it never
calls into the sampling app.
