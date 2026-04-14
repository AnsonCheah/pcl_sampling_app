# MM_Optimizer — Project State Log

Concise changelog of implementation decisions and state transitions.

---

## 2026-04-10 — Initial implementation

**What was built:**
- `search_config.py` — all phase candidate tables, scoring constants, two-pass settings
- `eval_cache.py` — Strategy 1 (transposition table): SHA-256 keyed result cache, disk-persistent
- `mesh_analysis.py` — Phase 0 geometry analysis: diameter, surface area, regime hint, symmetry hint, warm-start params
- `optimizer_utils.py` — extended with `list_synthetic_plys()`, `compose_scene()` (merges N sample PLYs via open3d, writes to temp dir)
- `optimizer.py` — full 6-phase hierarchical coordinator + Strategy 2 (two-pass multi-fidelity) + Strategy 3 (phase-level gates)
- `tests/test_cache.py` — unit test: cache hit/miss, persistence, stats
- `tests/test_mesh_analysis.py` — smoke test: warm-start derivation on 25333MB000 reference cloud
- `tests/test_evaluate.py` — live MechVision: single evaluate_config() call, verifies result dict shape
- `tests/test_phase1.py` — live MechVision: Phase 1 regime gate on 25333MB000 scene_00000 samples

**Key design decisions:**
- Strategy 1+2+3 are detachable: `ENABLE_CACHE`, `ENABLE_TWO_PASS` flags at top of `optimizer.py`; phase gates can be individually commented out
- N_INSTANCES=1 for initial development — single-instance scenes using individual sample_N.ply files
- Scene composition: merges N sample PLYs into a single `scene.ply` in a temp subdirectory; temp dir is re-used across evaluations within one optimizer run and cleaned up at exit
- Warm start for 25333MB000: refStep≈5, distQuantification≈5.0, angleQuantification=60 (derived from ~50mm diameter)
- `coarse_matching_scores` key (not `coarse_scores`) fixed in run_vision return dict

**MechVision path confirmed:**
- Model dir: `MM_Optimizer/CAD_Match/resource/3d_matching/25333MB000_surface/`
- Scene path: passed as directory containing `scene.ply`
- Project ID resolved by name `CAD_Match`

---

## 2026-04-10 — Phase 1 debugging and fixes

**Bugs found and fixed:**

1. **distQuantification = 4.6 (wrong)** — `mesh_analysis.py` was treating `distQuantification` as
   an absolute distance (D/RELATIVE_STEP ≈ 4.6 for 92mm part). MechVision's `distQuantification`
   is a UNITLESS FACTOR: DistanceInterval = distQuantification × SamplingInterval.  
   Default = 1.0 (optimal bin width ≈ one sampling interval). Fixed to `ws.distQuantification = 1.0`.  
   Also updated `PHASE2A_DIST_RATIOS` to include [0.15, 0.25] to rediscover distQ≈1.0 in grid search.

2. **read_single_ply missing from Pre_Segmentation params** — `_make_params_dict` was not setting
   `scriptFilePath` / `funcName` on the Pre_Segmentation step. Added explicit injection.
   Also: `compose_scene()` fixed to return `(scene_ply_path, gt_poses)` not `(work_dir, gt_poses)`.

3. **Phase 1 coverage = 0 despite correct detections** — Fine match correctly finds the part
   position (< 1mm), but returns an orientation ≈ 180° from GT for parts with rotational symmetry.
   Root cause: 25333MB000 has 2-fold symmetry; MechVision's ICP fine match converges to an
   equivalent orientation. The 10° angular threshold rejected ALL detections.  
   Fix: added `ANG_THRESH_REGIME_GATE = 360.0` in `search_config.py`. Phase 1 regime gate now
   uses position-only scoring (any orientation accepted). Orientation accuracy is Phase 3's job.

**Phase 1 results on 25333MB000 (M_FULL=30, n_instances=1):**
- Regime A (Surface/Surface): cov=1.00 ← best
- Regime B (Edge/Edge):       cov=0.60 (edge matching fails for 2/5 scenes, no tangent vectors)
- Regime C (Edge/Surface):    cov=0.60
- Regime D (Surface/Edge):    cov=0.60
- All 4 pass PHASE1_COVERAGE_GATE=0.50

**Tests passing:** test_cache.py (5/5), test_mesh_analysis.py (3/3),
                   test_evaluate.py (2/2), test_phase1.py (dry + live PASS)

---

## 2026-04-13 — Phase 2 passing + distQ grid fix

**Bugs fixed:**

1. **PHASE2A_DIST_RATIOS formula generated distQ > 5.0** — original grid was `dist = ref * ratio`,
   so with `ref=8` (scale=2.0×warm) and `ratio=1.3`, dist=10.4. MechVision rejects `distQuantification > ~5`.
   Fix: replaced `PHASE2A_DIST_RATIOS` with `PHASE2A_DISTQ_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]`
   (direct distQ values, independent of refStep). Grid is now 5×6=30 configs, all valid.

2. **Phase 2/3 coverage = 0 due to tight angular threshold** — same root cause as Phase 1 fix.
   Phase 2 optimises coarse localisation, not orientation. Orientation accuracy is Phase 4's job.
   Fix: Phase 2 and Phase 3 now use `ANG_THRESH_REGIME_GATE=360.0` throughout (position-only).
   Added `ang_thresh` parameter to `_phase2a_joint_grid`, `_sweep_param`, `_sweep_param_direct`.

**Phase 2 results on 25333MB000 (M_FULL=5, no two-pass):**
- Phase 2a best: refStep=6, distQ=3.0, cov=1.00
- Phase 2b: `referredStep=2` drops to cov=0.80 (correctly penalised); all others = 1.00
- Converged in Round 3, 800 total MechVision calls (M_FULL=5 × ~30 configs × ~5 sweeps + baselines)
- Tiebreaker (mean_time) effectively ranks configs within cov=1.00 set

**Threshold schedule:**
- Phase 1 gate: `ANG_THRESH_REGIME_GATE=360°` (position-only)
- Phase 2 CD: `ANG_THRESH_REGIME_GATE=360°` (position-only)
- Phase 3 CD: `ANG_THRESH_REGIME_GATE=360°` (position-only)
- Phase 4 symmetry: switches to `ANG_THRESH_TIGHT=5°` (after symmetry params added)
- Phase 5/6: `ANG_THRESH_TIGHT=5°`

**Tests passing:** test_phase2.py (dry + phase2a + full PASS)

---

## 2026-04-13 — Phase 3 passing + scene aggregation + full dry run

**Bugs fixed / improvements:**

1. **Phase 3 angular threshold** — Same position-only fix applied (ANG_THRESH_REGIME_GATE=360°).
   `_sweep_param`, `_sweep_param_direct`, `_phase2a_joint_grid` now accept `ang_thresh` kwarg.
   Phase 2 and Phase 3 explicitly pass `SC.ANG_THRESH_REGIME_GATE`. Phases 5/6 use tight thresholds.

2. **Scene pool used only scene_00000 (69 PLYs)** — `list_synthetic_plys` updated to accept either
   a single scene dir OR a part-level directory, aggregating across all `scene_NNNNN/` subdirs.
   Part `25333MB000` now provides 330 PLYs (5 scene dirs × ~66 samples each).
   Tests and CLI updated to point at part-level directory.

**Phase 3 results on 25333MB000 (M_FULL=5):**
- `deviationCorrectionCapacity=1.0/2.0`: drops to cov=0.00 → stays at 0.0=Small
- `confidenceThreshold=0.2+`: drops detections → stays at 0.0
- `operationApproach` look-ahead: median_pos_err=0.56mm → candidates=[0.0, 1.0] (HighSpeed, Standard)
- Final fine params: operationApproach=1.0 (Standard), deviationCorrectionCapacity=0.0, confThresh=0.0

**Full pipeline (Phases 1–6) dry run: PASS**
- All 6 phases execute without error
- Phase 5: 54-candidate joint grid (6 voteRatios × 3 outputNums × 3 referredSteps)
- Phase 6: interval narrowing over refStep, maxVoteRatio, confidenceThreshold

**Live full run result (M_FULL=5, 330 PLYs, cache ON, two_pass OFF):**
- coverage=1.0, mean_time=0.434s, evals≈970, wall_time≈16min
- Best coarse: refStep=8, distQ=3.0, angleQ=30, maxPairs=1250, maxVoteRatio=0.8, useDistNMS=false
- Best fine: operationApproach=1.0, deviationCorrectionCapacity=0.0, confThresh=0.0
- **BUT:** ang_errors≈179.7° in final result — coverage=1.0 was a false positive from cache bug

---

## 2026-04-13 — Three bugs found in Phase 4–6 path; all fixed

**Bugs found and fixed after first full live run:**

1. **`_sample_scenes` always returned plys[0..4]** — `random.shuffle(indices)` advanced RNG but
   `chosen` used `g * group_size % len(plys)` (ignored shuffled indices). All phases evaluated
   the SAME 5 PLYs every time; 330-PLY pool was never actually sampled.
   Fix: use shuffled indices for selection.

2. **Cache key excluded thresholds** — `evaluate_config` keyed cache on `(config, scenes)` but
   not on `(pos_thresh, ang_thresh)`. Phase 2 evaluated configs with `ang_thresh=360°` (position-only)
   and stored `cov=1.0` results. Phase 5 later queried the same cache key (tight ang_thresh=5°)
   and got cache HIT → returned the loose-threshold result with `cov=1.0` even though orientations
   were ~180° off. This masked the Phase 4 adoption bug entirely.
   Fix: include `_pos` and `_ang` in cache key dict.

3. **Phase 4 adoption logic used incomparable scores** — Phase 3 score was computed with
   `ANG_THRESH_REGIME_GATE=360°` (position-only). Phase 4 evaluates with `ANG_THRESH_TIGHT=5°`.
   For 25333MB000, Phase 4's symmetry-enabled config had `mean_time ≈ 0.09s` vs Phase 3's
   `mean_time ≈ 0.08s` → `Phase4.score > Phase3.score` → Phase 4 never adopted.
   Phase 5 then used fine params without `angleStep`, giving cov=0.0 with tight thresholds.
   Bug was invisible because cache bug (Bug 2) served Phase 2 results to Phase 5.
   Fix: adopt Phase 4 result if `coverage > 0.0` (symmetry search found any pose at all).

4. **Phase 5 double evaluation** — After `evaluate_phase_sweep(54 configs)`, Phase 5 re-evaluated
   all 54 configs a SECOND time. With the `_sample_scenes` bug (always same scenes) + cache (same
   key), second pass was 100% cache hits, so cost was zero. But after fixing `_sample_scenes`,
   each call returns different scenes → second pass would do 270 new MechVision calls.
   Fix: replaced double evaluation with a single direct loop over all 54 configs on one shared
   scene set, collecting all results for Phase 6 top-K selection.

**First live run explained:**
- Bug 2 (cache key) masked Bug 3 (Phase 4): Phase 5/6 saw cov=1.0 by re-serving Phase 2/3 cache
- Final result: coverage=1.0 but ang_errors≈179.7° — correct position, wrong orientation
- Config had NO `angleStep` — would return orientation-flipped poses in production

**Next required action:** Re-run live test with all 4 bugs fixed (fresh cache recommended).

---

## 2026-04-13 — Root cause identified: geocenter orientation bug; first clean pass

**Root cause of all 180° angular errors:**

The `geo_center.json` files for both `25333MB000_surface` and `25333MB000_edge` had been
imported into the MechVision project with an incorrect quaternion (180° flip in X axis,
ZYX Euler convention). This caused MechVision to report all detected poses with an orientation
180° off from the synthetic GT ground truth stored in the PLY files.

Consequence: every run prior to this fix showed `ang_errors ≈ 179.7°` at the Phase 5 tight
threshold (5°). This was falsely attributed to part rotational symmetry, triggering two full
debugging sessions on Phase 4 symmetry handling — all unnecessary.

**What was actually wrong with previous optimizer fixes:**
- The Phase 4 symmetry path, `_confirm_symmetry`, and all `angleStep` work were developed to
  solve a phantom problem. With the geocenter corrected, Phase 4 correctly fires "No symmetry
  confirmed — skipping Phase 4" for 25333MB000.
- The 4 code bugs found (sampling, cache key, Phase 4 adoption, Phase 5 double eval) are all
  genuine correctness bugs and remain fixed. They were independent of the geocenter issue.
- The Phase 4 `operationApproach >= 1.0` fix (added for HighSpeed incompatibility with
  `angleStep`) remains as defensive code for genuinely symmetric parts.

**Fix applied:** Both `geo_center.json` files corrected to identity pose
`[0, 0, 0, 1, 0, 0, 0]` (qw=1, qx=qy=qz=0). Cache cleared (all 488 stale entries discarded).

**First clean full run result (M_FULL=5, 330 PLYs, fresh cache, corrected geocenter):**
- Phase 1: cov=1.00, best regime = A (Surface coarse + Surface fine)
- Phase 2: cov=1.00, time=0.406s
- Phase 3: cov=1.00, time=0.391s
- Phase 4: skipped (no symmetry confirmed — correct)
- Phase 5: best cov=1.00
- Phase 6: cov=1.00, time=0.385s, score=0.385
- **PASS**

**Final best config:**
```json
{
  "coverage": 1.0,
  "mean_time": 0.385s,
  "coarse": {"refStep": 5, "distQuantification": 3.75, "angleQuantification": 30,
             "maxNumOfPointPairsPerFeature": 5000, "maxVoteRatio": 0.8,
             "referredStep": 2, "useDistanceNMS": true},
  "fine": {"operationApproach": 0.0, "deviationCorrectionCapacity": 0.0,
           "onlyConsiderVisibleSurfaceOfModel": true, "considerErrorofNormalAngles": true,
           "scoreLevel": 0.0, "confidenceThreshold": 0.0}
}
```

No `angleStep` — correct for a non-symmetric part.

---