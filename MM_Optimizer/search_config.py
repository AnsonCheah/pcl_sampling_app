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

PHASE2A_REFSTEP_SCALES = [0.5, 0.75, 1.0, 1.5, 2.0]
# Direct distQ FACTOR values to try — independent of refStep.
# Centred on 1.0 (MechVision optimal), exploring ½×–3× range.
PHASE2A_DISTQ_VALUES   = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# ---------------------------------------------------------------------------
# Phase 2b — Remaining coarse params (coordinate descent, in priority order)
# Each entry: (param_name, candidates_list, is_edge_only)
# ---------------------------------------------------------------------------

PHASE2B_PARAMS = [
    ("angleQuantification",          [30, 45, 60, 90],               False),
    ("maxVoteRatio",                  [0.3, 0.5, 0.6, 0.7, 0.8, 0.9], False),
    ("maxNumOfPointPairsPerFeature",  None,                            False),  # warm-relative, computed at runtime
    ("referredStep",                  [1, 2, 3],                       False),
    ("useDistanceNMS",                [True, False],                   False),
    ("outputNum",                     [1, 2, 3],                       False),
    ("filterCandidatePoseByAxis",     [True, False],                   True),   # edge only
    ("angleThreshold",                [45, 90, 135],                   True),   # edge only, only if above=True
]

# Multipliers for maxNumOfPointPairsPerFeature relative to warm-start value
PHASE2B_PAIRS_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]

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
