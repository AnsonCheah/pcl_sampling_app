---
name: MM_Optimizer Subagent
description: Constraints and contracts for the MechMind pose optimizer domain
type: project
---

# MM_Optimizer

## Role
Black-box optimizer: set MechVision parameters → trigger vision → compare returned poses against synthetic GT → score → repeat.
The optimizer owns the main loop. MechVision is a dumb pose estimator — it does not know it is being optimized.
Currently using MechVision 1.8.2, always read documentation on: https://docs.mech-mind.net/en/suite-software-manual/1.8.2/vision-steps/steps.html

Key references:
https://docs.mech-mind.net/en/suite-software-manual/1.8.2/vision-steps/3d-coarse-matching-v2.html
https://docs.mech-mind.net/en/suite-software-manual/1.8.2/vision-steps/3d-fine-matching-lite.html
https://docs.mech-mind.net/en/suite-software-manual/1.8.2/vision-steps/calc-results-by-python.html
https://docs.mech-mind.net/en/suite-software-manual/1.8.2/vision-steps/easy-create-string-list.html

## Immutable Constraints
- `communication/mm_adapter/` folder hierarchy cannot change (MechVision requires it)
- `.vis` files are MechVision executables — do not edit programmatically
- `Auto_CADMatch_Tuner/` and `CAD_MATCH_OPTIMISATION/` are deprecated — ignore completely
- MM_Optimiser console log location: `C:\Program Files\Mech-Mind\Mech-Vision & Mech-Viz-1.8.2\Mech-Vision\logs\YYYY-MM-DD.log` Note: tail the recent outputs instead of reading everything to be efficient with context.

## Adapter Architecture
Two commands on `mm_adapter.py`:
- set parameters: accepts `{module_name: {param_key: (value, type, unit)}}` — already multi-module-aware
- trigger vision run: returns a flat list of poses (no dict structure from MechVision)
MechVision returns poses as a flat list. Labels are disabled for now (`need_label: false` in config) but can be re-enabled.
Do NOT re-run MechMind's adapter generator for fine registration parameters — it fails on generating adapter of more than 30 numbers
MechVision logs folder: C:\Mech-Mind\Mech-Vision & Mech-Viz-1.8.3\Mech-Vision\logs

## Diagnostic Constraints

**`geo_center.json` must be the identity pose `[0, 0, 0, 1, 0, 0, 0]`.** A wrong quaternion there
makes MechVision report every pose rotated off the synthetic GT — it presents as a uniform
`ang_error ≈ 179.7°` at the tight threshold, which looks exactly like unhandled rotational
symmetry. Check the geocenter before investigating symmetry. `SaveStage` writes these values from
`inv(app.geocenter)`.

**The eval cache key must include the accuracy thresholds.** `evaluate_config` runs at both the
loose regime-gate threshold (`ang_thresh = 360°`, position-only) and the tight one (5°). Omit
`pos_thresh`/`ang_thresh` from the key and a loose `coverage = 1.0` is served to a tight query,
hiding orientation errors entirely. `EvalCache._scene_signature` additionally folds each
`sample_*.ply`'s `(name, size, mtime_ns)` into the key, so regenerating scenes into the same
directory invalidates it — a path-only key returned stale results and hid a real world-Z offset.

## Optimization Objective (priority order)
1. Maximize pose precision vs synthetic ground truth (primary)
2. Minimize pose error against GT poses from synthetic PLY (from the sampling app's pipeline)
3. Minimize cycle time — only as a tiebreaker when precision meets the tolerance threshold

## Ground Truth Interface
GT poses come from the synthetic PLY files produced by the sampling app's pipeline. The optimizer reads these PLY files; it does not call into the sampling app directly 
