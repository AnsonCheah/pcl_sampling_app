"""
search_config.py
----------------
All phase candidate tables, scoring constants, and two-pass budget settings.

This is pure data — no imports from mm_adapter or optimizer logic.
Edit this file to change the search space without touching optimizer.py.
"""

# ---------------------------------------------------------------------------
# Scoring thresholds
# ---------------------------------------------------------------------------

POS_THRESH_LOOSE = 0.005    # 5 mm  — Phase 1 regime gate
ANG_THRESH_LOOSE = 10.0     # 10°   — Phase 1 regime gate

# Phase 1 uses position-only detection: parts with rotational symmetry return a
# geometrically valid but orientation-flipped pose. The regime gate checks whether
# coarse+fine can LOCALIZE the part — orientation accuracy is Phase 3's job.
ANG_THRESH_REGIME_GATE = 360.0   # accept any orientation in Phase 1 regime gate

POS_THRESH_TIGHT = 0.002    # 2 mm  — Phase 2+
ANG_THRESH_TIGHT = 5.0      # 5°    — Phase 2+

POS_THRESH_MATCH = 0.006    # 3 × tight — threshold-gated NN matching gate

TARGET_COVERAGE  = 0.90     # stop early if this is reached

# ---------------------------------------------------------------------------
# Two-pass multi-fidelity settings  (Strategy 2)
# Set M_SMALL=M_FULL or ENABLE_TWO_PASS=False in optimizer.py to disable.
# ---------------------------------------------------------------------------

M_SMALL      = 5    # scenes for cheap Pass 1 screening
M_FULL       = 30   # scenes for full Pass 2 evaluation
K_SURVIVORS  = 3    # top-K configs from Pass 1 that proceed to Pass 2

# Minimum candidates to bother with two-pass (skip if N <= this)
TWO_PASS_MIN_CANDIDATES = 3

# ---------------------------------------------------------------------------
# Phase 1 — Regime combos (coarse_mode, fine_mode)
# 0.0=Surface, 1.0=Edge
# ---------------------------------------------------------------------------

PHASE1_REGIMES = [
    {"id": "A", "coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False},
    {"id": "B", "coarse_mode": 1.0, "fine_mode": 1.0, "needs_edge": True},
    {"id": "C", "coarse_mode": 1.0, "fine_mode": 0.0, "needs_edge": True},
    {"id": "D", "coarse_mode": 0.0, "fine_mode": 1.0, "needs_edge": True},
]

PHASE1_COVERAGE_GATE = 0.50    # min coverage @ loose threshold to pass Phase 1

# ---------------------------------------------------------------------------
# Phase 2a — Joint quantization grid (refStep × distQuantification)
# refStep sweeps the full MechMind range 20→1 (coarser/faster first); no geometry scaling.
# distQuantification is MechVision's UNITLESS FACTOR (default 1.0), independent of refStep.
# Valid range confirmed by MechVision: roughly [0.1, 5.0].
# ---------------------------------------------------------------------------

PHASE2A_REFSTEP_VALUES = list(range(1, 21))   # 1→20, finer/slower first; high-coverage trials complete before TPE activates
# Direct distQ FACTOR values to try — independent of refStep.
# Centred on 1.0 (MechVision optimal), exploring ½×–3× range.
PHASE2A_DISTQ_VALUES   = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# ---------------------------------------------------------------------------
# Phase 2b — Remaining coarse params (pipeline-ordered for coordinate descent)
# Each entry: (param_name, candidates_list, is_edge_only)
#
# Order follows the MechVision coarse matching pipeline:
#   scene sub-sampling (referredStep)
#   → Hough accumulator structure (angleQ, pairs, distQ already locked in 2a)
#   → Hough threshold (maxVoteRatio — must be after voting params stable)
#   → NMS / axis filtering (useDistanceNMS, edge-only axis filter)
#   → pose verification voxel grid (voxelLengthRange)
#   → final output count (outputNum — most downstream, must be last)
# ---------------------------------------------------------------------------

PHASE2B_PARAMS = {
    # Order is intentional: follows MechVision coarse matching pipeline
    "referredStep":                 {"candidates": [3, 2, 1],                   "edge_only": False},  # scene sub-sampling — coarser/faster first
    "angleQuantification":          {"candidates": [180, 120, 90, 60],          "edge_only": False},  # Hough angle bins — coarser/faster first
    "maxNumOfPointPairsPerFeature": {"candidates": None,                        "edge_only": False},  # voting density, warm-relative
    "maxVoteRatio":                 {"candidates": [0.5, 0.6, 0.7, 0.8, 0.9],   "edge_only": False},  # Hough threshold — after voting stable
    "useDistanceNMS":               {"candidates": [True, False],               "edge_only": False},  # NMS candidate filter
    "filterCandidatePoseByAxis":    {"candidates": [True, False],               "edge_only": True},   # edge only — axis filter
    "angleThreshold":               {"candidates": [45, 90, 135],               "edge_only": True},   # edge only — conditional on above
    "voxelLengthRange":             {"candidates": None,                        "edge_only": False},  # pose verification, warm-relative pairs
    "outputNum":                    {"candidates": [1, 2, 3],                   "edge_only": False},  # final output count — most downstream
}

# Multipliers for maxNumOfPointPairsPerFeature relative to warm-start value
PHASE2B_PAIRS_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]

# Multipliers for voxelLengthRange (min, max) relative to geometry-derived warm values.
# Preserves the 1:4 min:max ratio at every scale so min < max is guaranteed.
PHASE2B_VOXEL_SCALES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

# Max rounds for Phase 2 coordinate descent
PHASE2_MAX_ROUNDS = 3

# ---------------------------------------------------------------------------
# Phase 3 — Fine matching params (coordinate descent, in priority order)
# ---------------------------------------------------------------------------

PHASE3_PARAMS = {
    # Order is intentional: priority order for coordinate descent
    "operationApproach":                  [0.0, 1.0, 2.0, 3.0],            # Priority 1
    "deviationCorrectionCapacity":        [0.0, 1.0, 2.0],                 # Priority 2
    "onlyConsiderVisibleSurfaceOfModel":  [False, True],
    "considerErrorofNormalAngles":        [False, True],
    "scoreLevel":                         [0.0, 1.0, 2.0, 3.0],            # sweep with confThresh=0
    "confidenceThreshold":                [0.0, 0.1, 0.2, 0.3, 0.4, 0.6], # after scoreLevel locked
}

# ---------------------------------------------------------------------------
# Phase 3 look-ahead: coarse median position error → operationApproach candidates
# Avoids sweeping all 4 values; uses Phase 2 results.
# ---------------------------------------------------------------------------

def phase3_approach_candidates(median_coarse_pos_err_m):
    """Return operationApproach candidates based on coarse pose quality."""
    if median_coarse_pos_err_m < 0.003:
        return [0.0, 1.0]            # HighSpeed, Standard
    elif median_coarse_pos_err_m < 0.010:
        return [1.0, 2.0]            # Standard, HighAccuracy
    else:
        return [2.0, 3.0]            # HighAccuracy, ExtraHighAccuracy


# ---------------------------------------------------------------------------
# Phase 4 — Symmetry confirmation
# ---------------------------------------------------------------------------

PHASE4_ROTATION_STRATEGIES = [0.0, 1.0, 2.0]   # X, Y, Z axis

# For confirmed n-fold symmetry: angleStep = 360/n and 360/(2n)
def phase4_angle_steps(sym_order):
    return [360.0 / sym_order, 360.0 / (2 * sym_order)]

# Phase 4 symmetry detection thresholds
SYM_AMBIGUOUS_FRAC_THRESHOLD = 0.25   # fraction of ang errors near target to confirm
SYM_ANGLE_TOL_DEG            = 20.0   # ±tolerance around 360/n target

# ---------------------------------------------------------------------------
# Symmetry classification (mesh_analysis._classify_symmetry)
# ---------------------------------------------------------------------------

# Chamfer distance threshold as a fraction of model diameter.
# 0.5% is tight for clean CAD mesh (no sensor noise) and loose enough for
# minor mesh sampling artefacts. Tunable — log chamfer values and adjust if
# false positives arise in practice.
# 2% of diameter gives a noise floor ~3-5× above sampling noise (~1-3mm for 5000 points)
# while staying well below the asymmetric Chamfer signal (~10-50% of diameter).
SYM_CHAMFER_THRESH_FRAC = 0.02    # 2% of diameter

# Absolute minimum threshold (mm→m) so small parts (<150mm) aren't starved of margin.
SYM_CHAMFER_THRESH_MIN_M = 0.003  # 3mm minimum

# N-fold orders to test, coarse-to-fine (highest order found first).
# Only divisors of 360 make sense for industrial parts.
SYM_NFOLD_CANDIDATES = [6, 4, 3, 2]

# Inertia eigenvalue ratio thresholds for SO2/SO3 candidate proposal.
# SO3: all three eigenvalues within 5% of each other.
# SO2: two eigenvalues within 5% of each other (one degenerate axis).
# Slightly lenient (1.15) to handle sampling noise on sphere/disc point clouds.
# An elongated cylinder has max/min eigenvalue ratio ~5x, well above 1.15.
SYM_SO3_EIGEN_RATIO = 1.15
SYM_SO2_EIGEN_RATIO = 1.10

# ---------------------------------------------------------------------------
# Phase 5 — Joint refinement (coarse re-sweep with fine locked)
# ---------------------------------------------------------------------------

PHASE5_PARAMS = {
    "maxVoteRatio":  [0.5, 0.6, 0.7, 0.8, 0.9],
    "outputNum":     [1, 2, 3],
    "referredStep":  [1, 2, 3],
}

# Tie-breaking: configs within this fraction of best mean_time are "tied"
PHASE5_NOISE_MARGIN = 0.10   # 10% of best_mean_time
PHASE5_TOP_K        = 3      # top-k tied configs carry forward to Phase 6

# ---------------------------------------------------------------------------
# Phase 6 — Continuous interval narrowing
# ---------------------------------------------------------------------------

PHASE6_INTERVAL_STEPS = 5    # number of uniformly-spaced values in ± window
PHASE6_REFSTEP_DELTA  = 2    # ±2 integer steps around best refStep
PHASE6_VOTE_WINDOW    = 0.10 # ±0.10 around best maxVoteRatio
PHASE6_CONF_WINDOW    = 0.10 # ±0.10 around best confidenceThreshold
PHASE6_PAIRS_SCALES   = [0.75, 1.25]  # ×best for maxNumOfPointPairsPerFeature

# ---------------------------------------------------------------------------
# Phase gates  (Strategy 3)
# After each phase, check coverage against these bounds.
# If below threshold, skip subsequent phases.
# ---------------------------------------------------------------------------

PHASE_GATES = {
    # phase_key: (coverage_threshold, action_if_below)
    "after_phase1":  (0.50, "stop"),           # can't tune at all
    "after_phase2a": (0.30, "warn_skip_2b"),   # PPF quantization broken
    "after_phase2":  (0.65, "skip_phase4"),    # symmetry correction not worth it
    "after_phase3":  (0.75, "skip_phase6"),    # interval narrowing not justified
}

# ---------------------------------------------------------------------------
# Normalized scoring  (used by optimizer.py evaluate_config and optuna_optimizer.py)
# raw_score = mean_time / SCORE_TIME_NORM + (1 - coverage) * SCORE_COV_NORM
# Both terms are in [0, 1].  score_quality = 1 - raw_score / SCORE_WORST_CASE ∈ [0, 1].
# ---------------------------------------------------------------------------

SCORE_TIME_NORM  = 5.0   # reference cycle time (s); normalises time term to [0, 1]
SCORE_COV_NORM   = 1.0   # weight on (1-coverage); 1.0 keeps both terms dimensionless
SCORE_WORST_CASE = 2.0   # = 1.0 + SCORE_COV_NORM; denominator for quality

# ---------------------------------------------------------------------------
# Optuna settings  (used by optuna_optimizer.py)
# ---------------------------------------------------------------------------

# Fully joint TPE study (16D: all coarse + fine params in one study)
OPTUNA_N_TRIALS_JOINT        = 150  # round 0: 30 grid warm-starts + 120 TPE-guided
OPTUNA_N_TRIALS_JOINT_REFINE = 50   # round 1+: extend same study (TPE keeps density model)
OPTUNA_N_ROUNDS              = 2    # 1 = single pass, no refinement round
OPTUNA_N_STARTUP_JOINT       = 20   # startup trials before TPE uses joint kernel (16D needs ≥20)
OPTUNA_SCORE_IMPROVE_MIN     = 0.01 # stop rounds early if improvement < this (normalised score)

# Time guard: prune if running mean_time > best × RATIO.
# OPTUNA_TIME_INITIAL_CAP seeds _best_mean_time so the very first trial is guarded.
OPTUNA_TIME_RATIO       = 3.0
OPTUNA_TIME_INITIAL_CAP = 5.0  # seconds; replaces float("inf") at study start

# Coverage prune floor: prune if running_cov < this after 3+ scenes (step ≥ 2, 0-indexed).
OPTUNA_COV_PRUNE_FLOOR = 0.10

# NSGA-II sampler settings (replaces TPESampler)
OPTUNA_NSGA_POPULATION_SIZE = 100   # 18D needs larger population than default 50

# Optuna integer/float bounds — derived from existing phase tables so
# suggest_params functions never hard-code numbers.
REFSTEP_BOUNDS           = (1, 20)   # MechMind hard limit: integer 1–20 for both refStep and referredStep
OPTUNA_DISTQ_BOUNDS      = (min(PHASE2A_DISTQ_VALUES), max(PHASE2A_DISTQ_VALUES))
OPTUNA_VOTERATIO_BOUNDS  = (min(PHASE2B_PARAMS["maxVoteRatio"]["candidates"]),
                             max(PHASE2B_PARAMS["maxVoteRatio"]["candidates"]))
OPTUNA_OUTPUTNUM_BOUNDS  = (min(PHASE2B_PARAMS["outputNum"]["candidates"]),
                             max(PHASE2B_PARAMS["outputNum"]["candidates"]))
OPTUNA_CONFTHRESH_BOUNDS = (min(PHASE3_PARAMS["confidenceThreshold"]),
                             max(PHASE3_PARAMS["confidenceThreshold"]))

# Optuna categorical / integer bounds — all derived from PHASE tables (no hardcoding).
OPTUNA_ANGLQ_CHOICES     = PHASE2B_PARAMS["angleQuantification"]["candidates"]
OPTUNA_ANGLETHRESH_BOUNDS = (min(PHASE2B_PARAMS["angleThreshold"]["candidates"]),
                              max(PHASE2B_PARAMS["angleThreshold"]["candidates"]))
OPTUNA_OPAPP_BOUNDS      = (int(min(PHASE3_PARAMS["operationApproach"])),
                             int(max(PHASE3_PARAMS["operationApproach"])))
OPTUNA_DEVCAP_BOUNDS     = (int(min(PHASE3_PARAMS["deviationCorrectionCapacity"])),
                             int(max(PHASE3_PARAMS["deviationCorrectionCapacity"])))
OPTUNA_OPAPP_CHOICES     = PHASE3_PARAMS["operationApproach"]
OPTUNA_DEVCAP_CHOICES    = PHASE3_PARAMS["deviationCorrectionCapacity"]
OPTUNA_SCORELV_CHOICES   = PHASE3_PARAMS["scoreLevel"]

# 360.0 = disabled (MechVision convention); included so NSGA-II can choose "no rotation"
OPTUNA_ANGLE_STEP_SYM_CHOICES = [30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 360.0]
