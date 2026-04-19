# Plan: MM_Optimizer — Heuristic-Seeded Hierarchical Coordinate Descent

## Context

MechVision "3D Coarse Matching V2" (`Coarse_Match_Synthetics`) + "3D Fine Matching Lite"
(`Fine_Match_Synthetics`) expose ~25 tunable parameters. Goal: automated optimizer that finds
the best configuration for any given part — no per-part manual effort.

**Objective**: Maximize scene coverage (fraction of N-instance scenes where all GT poses are
recovered within tolerance). Cycle time is a tiebreaker only after coverage meets the target.

---

## Current Codebase State

### Already done
- `mm_adapter/mm_adapter.py`: `EasyCreateStringList`, `CoarseMatchingV2`, `FineMatchingLite`
  dataclasses exist. **But `run_vision()` still returns a flat pose list — needs extension
  to structured dict.**
- `registration/ppf_helpers.py`: `compute_model_diameter()` and `estimate_surface_area()` ready.
- `MM_Optimizer/optimizer_utils.py`: `read_gt_pose_from_ply()` exists. Needs `list_synthetic_plys()`,
  `compose_scene()`, and `read_synthetic(path)` accepting explicit path.

### Not yet created
- `MM_Optimizer/optimizer.py`
- `MM_Optimizer/search_config.py`
- `MM_Optimizer/mesh_analysis.py`

---

## Why Heuristic Initialization + Coordinate Descent

- Heuristic init: O(1) per part, derived from geometry, interpretable
- Warm-started coordinate descent: far fewer evaluations than cold start
- Reuses `registration/ppf_helpers.py` (`derive_ppf_params`, `PartProfile`) patterns

---

## Scene Composition & Evaluation Notation

- N = part instances per scene (per MechVision run)
- M = number of distinct scenes evaluated per config (diversity)
- H = number of hypotheses per run of coarse matching, equivalent to outputNum parameter in CoarseMatchingV2
- Hierarchy: `output/synthetic_target/{part}/scene_{M:05d}/sample_{N}.ply`
- One MechVision run: merged N-instance cloud in → N poses out

**Pose matching strategy:**

MechVision's output order is **not guaranteed to be preserved** — this is untested.
Index-based matching is therefore unsafe. Instead: **threshold-gated nearest-neighbor**.

For each GT pose, find the nearest returned pose (by position). Only assign if within
`POS_THRESH_MATCH` — this rejects garbage poses and avoids force-matching. No .vis changes
needed beyond what's already planned.

`confidenceThreshold=0.0` is still fixed during evaluation phases 1–5 (so MechVision returns
all N hypotheses), but matching correctness no longer depends on order.

```python
# alpha_rot: derived from tight thresholds so pos and rot have equal weight at the boundary.
# alpha_rot = pos_thresh_tight / ang_thresh_tight = 0.002 / 5 = 0.0004 m/deg
# This is a geometry-derived constant — no per-part tuning needed.
ALPHA_ROT = 0.0004

# Generous gate: accept any returned pose within 3× tight threshold.
# Prevents false-matching a garbage pose far from GT while still catching valid results.
POS_THRESH_MATCH = 0.006   # 3 × 0.002 m

def match_poses_to_gt(returned_poses, gt_poses):
    """Threshold-gated NN. No order assumption, handles confidenceThreshold filtering."""
    results = [(float('inf'), float('inf'))] * len(gt_poses)
    used = set()
    for gt_idx, gt in enumerate(gt_poses):
        candidates = [
            (np.linalg.norm(np.array(p[:3]) - gt[:3]), i, p)
            for i, p in enumerate(returned_poses)
            if i not in used
        ]
        if candidates:
            dist, idx, pred = min(candidates)
            if dist < POS_THRESH_MATCH:
                used.add(idx)
                results[gt_idx] = (dist, rotation_error_deg(pred[3:], gt[3:]))
    return results
```

> **Alternative (future)**: Fine Matching Lite's matched scene cloud output can be used to
> assign GTs by spatial overlap — this is more principled but requires the `.vis` project to
> serialize matched clouds over gRPC (additional wiring, non-trivial data size). Implement
> this if the threshold-gated NN produces false assignments in practice.

---

## Pre-Optimizer Setup — Reference Model Preparation

Runs once per part via `optimizer.setup_part(part_name)` before the optimization loop starts.

### Directory structure (actual, from `CAD_Match/resource/3d_matching/`)

```
MM_Optimizer/CAD_Match/resource/3d_matching/
├── {part}_surface/
│   ├── {part}_surface.ply            ← surface point cloud model
│   ├── geo_center.json               ← [[x, y, z, qw, qx, qy, qz]]
│   ├── poses.poses                   ← geocenter in MechVision native format
│   ├── pick_points.json              ← empty [] (not used by optimizer)
│   └── pick_points_labels.json       ← empty [] (not used by optimizer)
└── {part}_edge/
    ├── {part}_edge.ply               ← edge point cloud model
    ├── geo_center.json
    ├── poses.poses
    ├── pick_points.json
    └── pick_points_labels.json
```

### File formats

**`geo_center.json`** — 1-element JSON array, each element is a 7-value pose `[x, y, z, qw, qx, qy, qz]`.
For origin-centred models (our reference PLYs), this is `[[0, 0, 0, 1, 0, 0, 0, 0]]`.

**`poses.poses`** — JSON array of pose records. Only the geocenter entry is needed:
```json
[{
    "label": "0",
    "name": "{part}_{model_type}_geocenter",
    "pose": [0, 0, 0, 1.0, 0, 0, 0],
    "pose_type": 2
}]
```
`pose_type=2` is what MechVision uses in practice for geocenter entries.
`pick_points.json` / `pick_points_labels.json` are written as empty arrays `[]`.

### Source

Surface PLY: `output/synthetic_target/{part}/scene_00000/reference_cloud.ply`
Edge PLY: generated externally (MechVision edge model builder or pcd-sampling pipeline).
`setup_part()` skips edge setup if the edge PLY source is not found.

### How model paths map to adapter params

When building `params_dict`, the model type suffix determines `modelSelection` and paths:
```python
model_type = "surface"  # or "edge"
model_dir  = f"{MM_OPT_DIR}/CAD_Match/resource/3d_matching/{part}_{model_type}"
coarse = CoarseMatchingV2(
    modelSelection   = (f"{part}_{model_type}", "string", ""),
    modelFileName    = (f"{model_dir}/{part}_{model_type}.ply", "string", ""),
    geoCenterFileName= (f"{model_dir}/geo_center.json", "string", ""),
    ...
)
```
Mixed-mode example (edge coarse + surface fine): `coarse.modelSelection = "25333MB000_edge"`,
`fine.modelSelection = "25333MB000_surface"`.

---

## MechVision .vis Pipeline (CAD_Match_vision_flow.png)

```
┌───────────────────────────────┐
│  Easy Create String List      │  Step name: "Scene_Path"
│  (Scene_Path)                 │  param: strings = abs path to merged PLY
└──────────────┬────────────────┘
               │ stringList
               ▼
┌───────────────────────────────┐
│  Calc Results by Python       │  Pre-processing: loads cloud from path,
│  (Pre_Segmentation)           │  estimates normals, returns cloud+normals
└──────────────┬────────────────┘
               │ CloudXYZ.RGBN (scene cloud with normals)
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────────────────────────────────────┐
│  Calc by    │  │  3D Coarse Matching V2                        │
│  Python     │  │  (Coarse_Match_Synthetics)                    │
│ (Pre_Coarse)│  │                                               │
│             │  │  Inputs:  scene cloud, model params           │
│ → timestamp │  │  Outputs: NumberList-0 = coarse poses (N×H×7)│
└──────┬──────┘  │           NumberList-1 = coarse match scores  │
       │         └──────────────────┬───────────────────────────┘
       │ tik                        │ coarse_poses, coarse_scores
       ▼                            ▼
┌───────────────────────────────────────────┐
│  Calc Results by Python (Post_Coarse)     │
│                                           │
│  Computes: coarse_time_s = now - tik      │
│  Passes through: coarse_poses,            │
│                  coarse_matching_scores   │
└──────────────┬────────────────────────────┘
               │ coarse_poses (fed as initial guess to fine)
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────────────────────────────────────┐
│  Calc by    │  │  3D Fine Matching Lite                        │
│  Python     │  │  (Fine_Match_Synthetics)                      │
│  (Pre_Fine) │  │                                               │
│             │  │  Inputs:  coarse poses, scene cloud, model    │
│ → timestamp │  │  Outputs: NumberList-0 = fine poses (N×7)    │
└──────┬──────┘  │           NumberList-1 = fine confidence (N) │
       │         │           CloudXYZ.RGBN = visualization cloud │
       │ tik     └──────────────────┬───────────────────────────┘
       │                            │ fine_poses, fine_confidences
       ▼                            ▼
┌───────────────────────────────────────────┐
│  Calc Results by Python (Post_Fine)       │
│                                           │
│  Computes: fine_time_s = now - tik        │
│  Passes through: fine_poses,              │
│                  fine_confidences         │
└──────────────┬────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────┐
│  Procedure Out (0)                        │
│                                           │
│  fine_poses            [N × [x,y,z,qw,qx,qy,qz]]  │
│  fine_confidences      [N × float]                 │
│  coarse_poses          [N × H × [x,y,z,qw,qx,qy,qz]]│
│  coarse_matching_scores[N × H × int]               │
│  coarse_time_s         [float]  (1-element list)   │
│  fine_time_s           [float]  (1-element list)   │
└───────────────────────────────────────────┘
               │
               ▼
        run_vision() return dict
```

**Key observations from the flow:**
- The scene cloud is prepared (normals estimated) by `Pre_Segmentation` — **not** by MechVision's coarse module itself. The optimizer feeds a raw PLY path; `Pre_Segmentation` owns normal estimation.
- `Pre_Fine` timer step exists separately from `Post_Coarse` — fine timing is isolated cleanly.
- Coarse poses feed **directly** into Fine Matching as initial guess candidates — the optimizer cannot inject a different coarse hypothesis; it must control `outputNum` to determine how many candidates Fine sees.
- The visualization cloud from Fine is discarded by the optimizer (not in Procedure Out).

---

## MechVision Adapter Extension (mm_adapter.py)

`run_vision()` must return a structured dict. The `.vis` Python timer/output scripts are
**confirmed working** (tested 2026-04-09). Actual return structure from MechVision:

```json
{
    "fine_poses":            [[x, y, z, qw, qx, qy, qz]],   // list of N 7-element poses
    "fine_confidences":      [0.27],                          // list of N floats
    "coarse_poses":          [[[x, y, z, qw, qx, qy, qz]]], // [N instances × H hypotheses × 7]
    "coarse_matching_scores":[[240]],                         // [N instances × H vote scores]
    "coarse_time_s":         [0.0209],                        // single float wrapped in list
    "fine_time_s":           [0.0306],                        // single float wrapped in list
    "requestId":             "0",
    "stepInfo":              {},
    "vision_name":           "CAD_Match",
    "z_offset_compatibility_mode": true
}
```

**Note**: `coarse_matching_scores` (not `coarse_scores`) is the actual key name.
All timing fields are wrapped in a 1-element list — unwrap with `[0]`.
`run_vision()` implementation: parse keys above from `result`, unwrap timing lists, fall back to wall clock if keys absent.

---

## Scoring

```python
POS_THRESH_LOOSE, ANG_THRESH_LOOSE = 0.005, 10.0   # Phase 1 regime gate
POS_THRESH_TIGHT, ANG_THRESH_TIGHT = 0.002, 5.0    # Phase 2+
TARGET_COVERAGE = 0.90

def compute_score(run_results, pos_thresh, ang_thresh):
    per_run_ok = [
        all(p < pos_thresh and a < ang_thresh for p, a in run)
        for run in run_results
    ]
    coverage  = sum(per_run_ok) / len(per_run_ok)
    mean_time = np.mean([r["coarse_time_s"] + r["fine_time_s"] for r in run_results])
    score = (1.0 - coverage) * 1e6 + mean_time
    return score, coverage, mean_time
```

---

## Parameter Coupling & Constraints

Explicit constraints the optimizer must enforce. Violating these produces oscillating
coordinate descent or structurally incorrect Hough spaces.

### Coarse Matching — PPF + Hough Voting

#### HARD COUPLING: refStep ↔ distQuantification
**Mechanism**: PPF builds a Hough accumulator indexed by (distance, angle) feature pairs.
`refStep` controls the model point spacing; `distQuantification` controls the width of
distance bins. If bin width >> feature spacing, many distinct features map to the same bin
(votes smear, no peak). If bin width << feature spacing, bins are mostly empty (no votes
accumulate). OpenCV PPF documentation explicitly states: `distQuantification ≈ refStep`.

**Rule**: Always derive `distQuantification = dist_ratio × refStep`.
Lock `dist_ratio` in Phase 2a. Never move them independently after that.

```python
# CORRECT: joint motion
distQuantification = dist_ratio * refStep

# WRONG: independent coordinate descent
# Phase 2 round 1: refStep=5, distQuantification=3  ← optimizer sets this
# Phase 2 round 2: refStep=7, distQuantification=3  ← breaks the accumulator
```

#### LOOSE COUPLING: angleQuantification ↔ refStep
**Mechanism**: `angleQuantification` (= number of angle bins, bin width = 360°/N) should be
coarse enough to tolerate normal estimation error. Normal quality is not controlled by
`refStep`, but model sparsity (driven by `refStep`) does affect how many normals are averaged.
Denser model (smaller `refStep`) → more stable normals → can use higher `angleQuantification`.

**Rule**: Treat as independent after quantization ratio is locked. `angleQuantification` can
move freely in Phase 2b; no enforcement needed.

#### DOWNSTREAM-ONLY: maxVoteRatio
**Mechanism**: Thresholds the Hough accumulator output — it does NOT affect how votes are
cast. Changing it never invalidates the quantization configuration.

**Rule**: Must be optimized **after** refStep/distQuantification/angleQuantification are
settled. Doing it earlier wastes budget: the threshold may shift once the Hough space changes.

#### INDEPENDENT: maxNumOfPointPairsPerFeature, referredStep, useDistanceNMS, outputNum
These parameters operate on separate subsystems (scene-side memory, scene sampling, NMS
post-processing, output truncation). No coupling with quantization or each other.

**Rule**: Can be optimized in any order after quantization is settled.

---

### Fine Matching — FilterReg (Probabilistic ICP)

#### STRONG COUPLING: operationApproach ↔ deviationCorrectionCapacity
**Mechanism**: `operationApproach` controls the number of ICP iterations and convergence
tolerance. `deviationCorrectionCapacity` controls the GMM outlier model σ (how far a
scene point can be from the model before being down-weighted).

Combined effect:
- `HighSpeed + Small`: Assumes coarse pose is within ±5mm. Few iterations, tight inlier radius.
  Fails if coarse was imprecise.
- `HighAccuracy + Large`: Tolerates coarse pose error up to ±50mm. Many iterations, broad
  inlier radius. Works for difficult initializations but slow.
- `HighSpeed + Large`: Broad inlier radius but too few iterations to converge → worse than both.
- `HighAccuracy + Small`: Many iterations on a tight inlier set → can over-fit to noise.

**Rule**: Optimize `operationApproach` first (Priority 1). When a new `operationApproach` is
selected, `deviationCorrectionCapacity` must be re-evaluated (Priority 2). They are not
fully independent — coordinate descent is acceptable here because the interaction is monotonic
(more iterations benefit from tighter tolerance once converged).

#### REQUIRED SEQUENTIAL ORDER: scoreLevel → confidenceThreshold
**Mechanism**: `scoreLevel` defines the scoring function strictness (how confidence is
computed from inlier fraction + RMSE + normal consistency). `confidenceThreshold` is a
scalar cutoff on the resulting confidence value. Sweeping `confidenceThreshold` before
`scoreLevel` is fixed is meaningless — the same threshold value yields different recall
depending on `scoreLevel`.

**Rule**: Always fix `scoreLevel` first, then sweep `confidenceThreshold`. Never swap this
order. Never sweep them simultaneously.

#### INDEPENDENT: onlyConsiderVisibleSurface, considerErrorofNormalAngles
Boolean flags that modify the scoring model but do not couple with each other or with
convergence parameters.

**Rule**: Can be optimized in any order after `operationApproach` + `deviationCorrectionCapacity`.

---

### Cross-Stage Coupling (Coarse → Fine)

`operationApproach` optimal value depends on coarse stage accuracy: a well-tuned coarse
stage (tight refStep/distQuantification) produces better initial hypotheses → finer
matching can use `HighSpeed`. A loose coarse stage needs `HighAccuracy`.

**Rule**: Phase 3 runs after Phase 2 is converged. Phase 5 (joint refinement) re-sweeps
`maxVoteRatio` and `outputNum` with fine params locked to catch any residual interaction.

---

## Phase 0 — Mesh Analysis & Heuristic Warm Start

One-time per part. Wraps `registration/ppf_helpers.py`.

```python
D  = compute_model_diameter(ref_pcd)       # meters
SA = estimate_surface_area(ref_pcd, D)     # from ppf_helpers

# PPF theory: refStep and distQuantification are dimensionless quantities in MechVision.
# They encode feature spacing relative to model geometry. OpenCV PPF docs state
# distQuantification ≈ refStep. Use the same base formula for both to enforce this coupling.
RELATIVE_STEP = 0.02   # 2% of diameter per step — gives refStep≈5 for D=10cm (matches default)

warm = dict(
    refStep              = max(1, int(D / RELATIVE_STEP)),   # ~5 for D=10cm
    distQuantification   = max(0.5, D / RELATIVE_STEP),      # same scale as refStep (MUST stay coupled)
    angleQuantification  = 60,                                # 6° bins; adequate for clean normals
    maxNumOfPointPairsPerFeature = 5000 if D < 0.1 else 10000,
    maxVoxelLength_mm    = D * 0.02 * 1000,
    minVoxelLength_mm    = D * 0.005 * 1000,
    outputNum            = N_INSTANCES,
)
```

Regime hint (Surface vs Edge):
```python
normal_concentration = np.mean(np.abs(normals @ dominant_normal) > 0.85)
bb = np.sort(ref_pcd.get_axis_aligned_bounding_box().get_extent())
flatness_ratio = bb[2] / bb[0]
prefer_edge = (normal_concentration > 0.50) or (flatness_ratio > 5.0)
```

Symmetry pre-detection:
```python
eigenvalues = np.sort(np.linalg.eigh(np.cov(pts.T))[0])
sym_z = (eigenvalues[1] / eigenvalues[2]) < 0.15
sym_x = (eigenvalues[0] / eigenvalues[1]) > 0.85
```

`maxScenePointNum` fixed at 5,000,000 — never swept.
`autoCalculateExpectedPointsNum` fixed True — never swept.

---

## Phase 1 — Regime Gate (2–4 evals × M scenes)

| # | coarse mode | fine mode | Requires edge model |
|---|------------|-----------|---------------------|
| A | 0.0 Surface | 0.0 Surface | No |
| B | 1.0 Edge | 1.0 Edge | Yes |
| C | 1.0 Edge | 0.0 Surface | Yes |
| D | 0.0 Surface | 1.0 Edge | Yes |

Gate: coverage ≥ 70% at loose tolerance (5mm, 10°). Failing combos discarded.
Order determined by Phase 0 geometry hint.

---

## Phase 2 — Coarse Coordinate Descent (warm-started, tight threshold)

Max 3 rounds, repeat until no improvement.

**Critical: `refStep` and `distQuantification` are PPF-coupled** — they must NOT be optimized
independently. If refStep changes, optimal distQuantification shifts proportionally. Optimizing
one after the other will oscillate. Solution: treat them as a 2D joint grid search.

### Phase 2a — Joint Quantization Grid (replaces independent Priority 1 & 3)

2D grid over (refStep scale, dist_ratio = distQuantification/refStep):
- `refStep` candidates: warm × {0.5, 0.75, 1.0, 1.5, 2.0} (5 values)
- `dist_ratio` candidates: {0.7, 1.0, 1.3, 1.7} (4 values — keeps distQuantification ≈ refStep per PPF theory)
- Total: 5 × 4 = 20 evals. Not coordinate descent — full 2D grid once per round.

### Phase 2b — Remaining Parameters (coordinate descent)

After best (refStep, dist_ratio) is locked from Phase 2a:

| Priority | Parameter | Candidates | Notes |
|----------|-----------|------------|-------|
| 1 | `angleQuantification` | {30, 45, 60, 90} | After quantization locked |
| 2 | `maxVoteRatio` | {0.3, 0.5, 0.6, 0.7, 0.8, 0.9} | **HIGH empirical sensitivity** — downstream Hough threshold |
| 3 | `maxNumOfPointPairsPerFeature` | warm × {0.25, 0.5, 1.0, 2.0, 4.0} | Independent memory budget |
| 4 | `referredStep` | {1, 2, 3} | Scene sampling, independent |
| 5 | `useDistanceNMS` | {True, False} | Boolean, near-zero cost |
| 6 | `outputNum` | {1, 2, 3} | Simple truncation |
| Edge | `filterCandidatePoseByAxis` | {True, False} | |
| Edge | `angleThreshold` | {45, 90, 135} | Only if above=True |

`voxelLengthGenetationStrategy` stays 0.0=Auto — not swept.

---

## Phase 3 — Fine Coordinate Descent

Fix best coarse from Phase 2. Sweep sequentially.

> **Re-tune caveat**: `operationApproach` optimal value depends on how well coarse alignment
> converged. After Phase 2 coarse parameters are finalized, Phase 3 is correct as stated.
> Phase 5 (joint refinement) catches any residual coarse-fine interaction.

| Priority | Parameter | Candidates | Notes |
|----------|-----------|------------|-------|
| 1 | `operationApproach` | {0.0, 1.0, 2.0, 3.0} | ICP iteration budget; highest algorithmic impact |
| 2 | `deviationCorrectionCapacity` | {0.0, 1.0, 2.0} | Outlier tolerance in GMM; coupled with above |
| 3 | `onlyConsiderVisibleSurfaceOfModel` | {False, True} | Semi-independent flag |
| 4 | `considerErrorofNormalAngles` | {False, True} | Independent flag |
| 5 | `scoreLevel` | {0.0, 1.0, 2.0, 3.0} with `confidenceThreshold=0.0` | Categorical; all 4 tested |
| 6 | `confidenceThreshold` | {0.0, 0.1, 0.2, 0.3, 0.4, 0.6} | Only after scoreLevel locked |
| — | `candidateTopNum` | fixed at 1 | |

scoreLevel and confidenceThreshold **must be swept sequentially**, never simultaneously.

---

## Phase 4 — Symmetry Confirmation (Conditional)

```python
def confirm_symmetry(ang_errors, geo_hint_order):
    for n in [2, 3, 4]:
        target = 360.0 / n
        frac_ambiguous = sum(abs(e - target) < 20 for e in ang_errors) / len(ang_errors)
        if frac_ambiguous > 0.25 and (geo_hint_order == n or geo_hint_order is None):
            return n, _infer_axis(ang_errors)
    return None, None
```

If confirmed, sweep: `rotationStrategy` {0.0, 1.0, 2.0}, `angleStep` {360/n, 360/(2n)}.
`minAngle`/`maxAngle` derived, not swept. ~10 evals total.

---

## Phase 5 — Joint Refinement (~20 evals)

Re-run targeted coarse re-sweep with fine params locked. **Only the coarse params most
affected by fine matching quality are re-tested** — refStep/distQuantification and
angleQuantification are stable regardless of fine quality:

| Parameter | Why re-check | Candidates |
|-----------|-------------|------------|
| `maxVoteRatio` | Better fine matching can tolerate more coarse hypotheses (lower threshold) | {0.3, 0.5, 0.6, 0.7, 0.8, 0.9} |
| `outputNum` | With precise fine matching, fewer coarse guesses may suffice → reduces cycle time | {1, 2, 3} |
| `referredStep` | Minor re-check since scene coverage now more uniformly scored | {1, 2, 3} |

~12 evals. Run once, no repeat rounds.

### Tie-breaking in Phase 5

When fine matching is good, many coarse configs achieve the same coverage. The primary
score function handles this by cycle time:

```python
score = (1.0 - coverage) * 1e6 + mean_time
```

But `mean_time` has measurement noise (~5% variance from OS scheduling). For configs whose
scores differ by less than `NOISE_MARGIN = 0.1 * best_mean_time`, treat them as tied
and add to a `top_k=3` list rather than discarding.

Secondary tiebreaker (applied only within `top_k`): **mean position error** of matched
poses that passed the threshold — lower is better (more margin from the threshold boundary).

```python
def soft_score(run_results, pos_thresh, ang_thresh):
    passing = [
        (p, a) for run in run_results for p, a in run
        if p < pos_thresh and a < ang_thresh
    ]
    return np.mean([p for p, _ in passing]) if passing else float('inf')
```

All `top_k` configs feed into Phase 6 interval narrowing. Phase 6 picks the final best.

---

## Phase 6 — Continuous Interval Narrowing (~25 evals)

Dense search around optima for continuous params only. Discrete params (registrationMode,
operationApproach, deviationCorrectionCapacity, etc.) are converged — skip.

**refStep + distQuantification [JOINT]**: The dist_ratio (`distQuantification / refStep`) is
locked at its Phase 2a optimal value. Only `refStep` moves; `distQuantification` follows:
- Candidates: refStep ∈ {best-2, best-1, best, best+1, best+2}, distQuantification = best_ratio × new_refStep
- 5 evals (not independent; they move together)

**angleQuantification**: Already a small categorical set {30, 45, 60, 90} — fully swept in Phase 2b. Skip Phase 6 for this.

**maxVoteRatio**: 5 values uniformly spaced in [best - 0.1, best + 0.1] (clipped to [0, 1]).

**confidenceThreshold**: 5 values uniformly spaced in [best - 0.1, best + 0.1] (clipped to [0, 1]).

**maxNumOfPointPairsPerFeature**: Try best × {0.75, 1.25}.

**angleStep** (if symmetry active): Try ½ and ¼ of best.

Total: ~15–20 evals.

---

## Optimizer Flowchart

```
optimizer.py --part <name>
        │
        ▼
┌───────────────┐
│    Phase 0    │  Mesh analysis: diameter, SA, regime hint,
│  (1× setup)   │  symmetry hint, warm-start params
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    Phase 1    │  Test combos A/B/C/D (2–4 evals × M scenes)
│  Regime Gate  │  confidenceThreshold=0, threshold-gated NN matching
└───────┬───────┘
        │  ≥1 combo passes coverage≥70% @ loose (5mm, 10°)
        ▼
┌───────────────┐
│    Phase 2    │  2a: Joint grid (refStep×distQuantification, 20 evals)
│   Coarse CD   │  2b: Coord descent remaining params
│  ~40 evals    │  Warm-started, max 3 rounds, tight threshold
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    Phase 3    │  Coordinate descent over fine params
│    Fine CD    │  scoreLevel first (confThresh=0), then confThresh
│  ~20 evals    │  threshold-gated NN matching throughout
└───────┬───────┘
        │
        ▼
┌───────────────┐  If Phase 0 geometry hint + Phase 3 error pattern agree:
│    Phase 4    │  sweep rotationStrategy + angleStep
│  Symmetry     │  ~10 evals (or 0 if not triggered)
│  (conditional)│
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    Phase 5    │  Re-sweep maxVoteRatio, outputNum, referredStep
│Joint Refinement│ ~20 evals; top_k=3 tied configs carry forward
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    Phase 6    │  Dense ±30% search around optima, continuous params only
│  Interval     │  ~25 evals
│  Narrowing    │
└───────┬───────┘
        │
        ▼
  best_config_{part}.yaml   (human-readable, production-deployable)
```

---

## Files to Create / Modify

| Action | File | What |
|--------|------|------|
| **Create** | `MM_Optimizer/optimizer.py` | Main loop, coordinate descent, scoring, logging, CLI |
| **Create** | `MM_Optimizer/search_config.py` | Phase definitions, candidate tables, warm-start formulas |
| **Create** | `MM_Optimizer/mesh_analysis.py` | Regime hint, symmetry detection — wraps `registration/ppf_helpers.py` |
| **Modify** | `MM_Optimizer/optimizer_utils.py` | Add `list_synthetic_plys()`, `compose_scene()`, fix `read_synthetic(path)` |
| **Modify** | `mm_adapter/mm_adapter.py` | Extend `run_vision()` → structured dict with timing fields |

---

## Reuse

- `mm_adapter/mm_adapter.py`: `MechVisionClient`, `CoarseMatchingV2`, `FineMatchingLite`, `EasyCreateStringList` (step name `"Scene_Path"`, field `strings`)
- `MM_Optimizer/optimizer_utils.py`: `read_gt_pose_from_ply()`
- `registration/ppf_helpers.py`: `compute_model_diameter()`, `estimate_surface_area()`

---

## Per-run params_dict structure

```python
scene  = EasyCreateStringList(strings=(merged_ply_path, "string", ""))
coarse = CoarseMatchingV2(name="Coarse_Match_Synthetics", ..., **current_coarse_params)
fine   = FineMatchingLite(name="Fine_Match_Synthetics", ..., **current_fine_params)
params_dict = {
    scene.name:  scene.to_step_params(),
    coarse.name: coarse.to_step_params(),
    fine.name:   fine.to_step_params(),
}
```

---

## Verification

1. **Adapter return dict** — confirmed working (tested 2026-04-09). The `.vis` Python timer
   and output scripts return the expected structured dict including `coarse_time_s`, `fine_time_s`,
   `coarse_poses`, `coarse_matching_scores`. No further verification needed before implementing.

2. **Output order consistency test** (run before optimizer, one-time per part):
   - Use a 2-instance scene where both instances have well-separated GT poses (e.g., ΔX > 0.2m).
   - Run the same scene 10× with fixed params.
   - For each run, record which returned pose is closer to GT_0 vs GT_1 (by position).
   - **If the mapping is consistent across all 10 runs**: MechVision has a deterministic ordering
     rule (likely descending confidence/vote score). Index-based matching is safe — document
     which instance maps to which index and set `USE_INDEX_MATCHING = True`.
   - **If the mapping varies across runs**: Ordering is non-deterministic. Threshold-gated NN
     is mandatory. Set `USE_INDEX_MATCHING = False` (current default).
   - This test only needs to run once; the result is fixed per-part and can be cached.

3. `python MM_Optimizer/optimizer.py --dry_run --part 25333MB000` — Phase 0 heuristics + 4 Phase 1 param dicts, no MechVision call

3. Phase 1 live: Surface and Edge regimes attempt; results in `results/ph1_*.json`; gate eliminates failing regimes

4. Phase 2 live: per-param improvement logged each round; convergence expected within 2 rounds

5. Symmetry test: known-symmetric part; Phase 4 log → `confirm_symmetry() → order=2, axis=Z`

6. Final export: `--export_best` writes `best_config_{part_name}.yaml` — human-readable, production-deployable
