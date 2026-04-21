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
# refStep candidates are multipliers on the warm-start value.
# distQuantification is MechVision's UNITLESS FACTOR (default 1.0).
# Explored independently of refStep — values are direct distQ candidates.
# Valid range confirmed by MechVision: roughly [0.1, 5.0].
# ---------------------------------------------------------------------------

PHASE2A_REFSTEP_SCALES = [2.0, 1.5, 1.0, 0.75, 0.5]   # coarser/faster first
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

PHASE2B_PARAMS = [
    ("referredStep",                  [3, 2, 1],                       False),  # scene sub-sampling — coarser/faster first
    ("angleQuantification",           [90, 60, 45, 30],                False),  # Hough angle bins — coarser/faster first
    ("maxNumOfPointPairsPerFeature",  None,                            False),  # voting density, warm-relative
    ("maxVoteRatio",                  [0.3, 0.5, 0.6, 0.7, 0.8, 0.9], False),  # Hough threshold — after voting stable
    ("useDistanceNMS",                [True, False],                   False),  # NMS candidate filter
    ("filterCandidatePoseByAxis",     [True, False],                   True),   # edge only — axis filter
    ("angleThreshold",                [45, 90, 135],                   True),   # edge only — conditional on above
    ("voxelLengthRange",              None,                            False),  # pose verification, warm-relative pairs
    ("outputNum",                     [1, 2, 3],                       False),  # final output count — most downstream
]

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

PHASE3_PARAMS = [
    ("operationApproach",                 [0.0, 1.0, 2.0, 3.0]),   # Priority 1
    ("deviationCorrectionCapacity",       [0.0, 1.0, 2.0]),        # Priority 2
    ("onlyConsiderVisibleSurfaceOfModel", [False, True]),
    ("considerErrorofNormalAngles",       [False, True]),
    ("scoreLevel",                        [0.0, 1.0, 2.0, 3.0]),   # sweep with confThresh=0
    ("confidenceThreshold",               [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]),  # after scoreLevel locked
]

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
# Phase 5 — Joint refinement (coarse re-sweep with fine locked)
# ---------------------------------------------------------------------------

PHASE5_PARAMS = [
    ("maxVoteRatio",  [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ("outputNum",     [1, 2, 3]),
    ("referredStep",  [1, 2, 3]),
]

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
# Optuna settings  (used by optuna_optimizer.py)
# ---------------------------------------------------------------------------

# Legacy single-study budget (kept for backward compat with old CLI --n_trials)
OPTUNA_N_TRIALS  = 100
OPTUNA_N_STARTUP = 10

# Staged architecture trial budgets
OPTUNA_N_TRIALS_1A = 35   # Stage 1a: refStep × distQ grid (30) + TPE (5)
OPTUNA_N_TRIALS_1B = 50   # Stage 1b: remaining coarse params
OPTUNA_N_TRIALS_2  = 50   # Stage 2:  fine params

# Time guard: prune if running mean_time > best × RATIO (Stage 1a only).
# Warm-start gives ~4s/scene → cap ~12s. refStep=5 (40-74s) pruned at scene 2.
OPTUNA_TIME_RATIO  = 3.0

# Optuna integer/float bounds — derived from existing phase tables so
# suggest_params functions never hard-code numbers.
OPTUNA_REFSTEP_BOUNDS    = (1, 20)
OPTUNA_DISTQ_BOUNDS      = (min(PHASE2A_DISTQ_VALUES), max(PHASE2A_DISTQ_VALUES))
OPTUNA_VOTERATIO_BOUNDS  = (min(next(c for n, c, _ in PHASE2B_PARAMS if n == "maxVoteRatio")),
                             max(next(c for n, c, _ in PHASE2B_PARAMS if n == "maxVoteRatio")))
OPTUNA_REFERRED_BOUNDS   = (min(next(c for n, c, _ in PHASE2B_PARAMS if n == "referredStep")),
                             max(next(c for n, c, _ in PHASE2B_PARAMS if n == "referredStep")))
OPTUNA_OUTPUTNUM_BOUNDS  = (min(next(c for n, c, _ in PHASE2B_PARAMS if n == "outputNum")),
                             max(next(c for n, c, _ in PHASE2B_PARAMS if n == "outputNum")))
OPTUNA_CONFTHRESH_BOUNDS = (min(next(c for n, c in PHASE3_PARAMS if n == "confidenceThreshold")),
                             max(next(c for n, c in PHASE3_PARAMS if n == "confidenceThreshold")))

# Optuna categorical choices — lists already in fast→slow order from PHASE2B
OPTUNA_ANGLQ_CHOICES   = next(c for n, c, _ in PHASE2B_PARAMS if n == "angleQuantification")
OPTUNA_OPAPP_CHOICES   = next(c for n, c in PHASE3_PARAMS if n == "operationApproach")
OPTUNA_DEVCAP_CHOICES  = next(c for n, c in PHASE3_PARAMS if n == "deviationCorrectionCapacity")
OPTUNA_SCORELV_CHOICES = next(c for n, c in PHASE3_PARAMS if n == "scoreLevel")
