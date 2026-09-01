"""Search space, scoring constants and gates for the MechVision tuner.

Pure data — no imports from mm_adapter or tuner logic. Edit here to change what
`tuner.suggest_params` explores; the same space and warm-start seed are shared by all
three samplers (NSGA-II / TPE / GP).
"""

# ---------------------------------------------------------------------------
# Scoring thresholds
# ---------------------------------------------------------------------------

POS_THRESH_LOOSE = 0.005    # 5 mm  — regime gate
POS_THRESH_TIGHT = 0.002    # 2 mm
ANG_THRESH_TIGHT = 5.0      # 5°
POS_THRESH_MATCH = 0.006    # 3 × tight — threshold-gated NN matching gate

# The regime gate is position-only: a rotationally symmetric part returns a geometrically
# valid but orientation-flipped pose, and whether coarse+fine can LOCALIZE the part is a
# separate question from orientation accuracy.
ANG_THRESH_REGIME_GATE = 360.0

M_FULL = 30                 # scenes per full evaluation

# ---------------------------------------------------------------------------
# Regimes — (coarse_mode, fine_mode); 0.0=Surface, 1.0=Edge
# ---------------------------------------------------------------------------

# Coarse and fine always share a cloud type: `3d_matching/<part>/` holds exactly one
# `<part>.ply`, swapped per regime by `model_sync.py`. Mixed combinations would need two
# models loaded at once, which that layout cannot express.
REGIMES = [
    {"id": "A", "coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False},   # surface
    {"id": "B", "coarse_mode": 1.0, "fine_mode": 1.0, "needs_edge": True},    # edge
]

REGIME_COVERAGE_GATE   = 0.50  # min coverage @ loose threshold for a regime to pass
SYMMETRY_COVERAGE_GATE = 0.65  # below this, symmetry correction is not worth running


def approach_candidates(median_coarse_pos_err_m):
    """operationApproach values worth trying, given coarse pose quality (metres)."""
    if median_coarse_pos_err_m < 0.003:
        return [0.0, 1.0]            # HighSpeed, Standard
    if median_coarse_pos_err_m < 0.010:
        return [1.0, 2.0]            # Standard, HighAccuracy
    return [2.0, 3.0]                # HighAccuracy, ExtraHighAccuracy


# ---------------------------------------------------------------------------
# Symmetry confirmation
# ---------------------------------------------------------------------------

ROTATION_STRATEGIES = [0.0, 1.0, 2.0]   # X, Y, Z axis


def angle_steps(sym_order):
    """angleStep candidates for a confirmed n-fold symmetry: 360/n and 360/(2n)."""
    return [360.0 / sym_order, 360.0 / (2 * sym_order)]


SYM_AMBIGUOUS_FRAC_THRESHOLD = 0.25   # fraction of ang errors near target to confirm
SYM_ANGLE_TOL_DEG            = 20.0   # ±tolerance around the 360/n target

# ---------------------------------------------------------------------------
# Symmetry classification (mesh_analysis._classify_symmetry)
# ---------------------------------------------------------------------------

# Chamfer threshold as a fraction of model diameter. 2% sits ~3-5× above sampling noise
# (~1-3 mm for 5000 points) and well below the asymmetric signal (~10-50% of diameter).
SYM_CHAMFER_THRESH_FRAC  = 0.02
SYM_CHAMFER_THRESH_MIN_M = 0.003  # absolute floor so parts <150 mm keep some margin

SYM_NFOLD_CANDIDATES = [6, 4, 3, 2]   # only divisors of 360 matter for industrial parts

# Inertia eigenvalue ratios for SO2/SO3 proposal. Lenient enough for sampling noise on
# sphere/disc clouds; an elongated cylinder sits ~5×, well clear.
SYM_SO3_EIGEN_RATIO = 1.15
SYM_SO2_EIGEN_RATIO = 1.10

# ---------------------------------------------------------------------------
# Normalized scoring
#   raw_score = mean_time / SCORE_TIME_NORM + (1 - coverage) * SCORE_COV_NORM
#   score_quality = 1 - raw_score / SCORE_WORST_CASE  ∈ [0, 1]
# ---------------------------------------------------------------------------

SCORE_TIME_NORM  = 5.0   # reference cycle time (s)
SCORE_COV_NORM   = 1.0   # weight on (1-coverage)
SCORE_WORST_CASE = 2.0   # = 1.0 + SCORE_COV_NORM

# ---------------------------------------------------------------------------
# Study budget
# ---------------------------------------------------------------------------

# Declared here, not in tuner.py, because the GUI needs the list to build its combo box
# and must not pay for tuner.py's optuna / mm_adapter imports to get it.
SAMPLER_CHOICES = ("nsgaii", "tpe", "gp")
SAMPLER_DEFAULT = "gp"

N_TRIALS          = 150  # round 0: 30 grid warm-starts + 120 model-guided
N_TRIALS_REFINE   = 50   # round 1+: extend the same study, sampler keeps its model
N_ROUNDS          = 2    # 1 = single pass, no refinement round
N_STARTUP         = 20   # random trials before TPE/GP use their model (18D needs >=20)
SCORE_IMPROVE_MIN = 0.01 # stop rounds early below this normalised-score gain

NSGA_POPULATION_SIZE = 100   # 18D needs more than the default 50

# Absolute per-trial time cap (s): a pure safety valve for pathological configs, not
# competitive pruning. Deliberately generous (~5× a good config) so slow-but-accurate
# trials are never pruned on time.
TIME_ABS_CAP     = 20.0
TIME_INITIAL_CAP = 5.0   # sentinel time (with coverage=0) for infeasible referredStep
COV_PRUNE_FLOOR  = 0.10  # prune if running coverage is below this after 3+ scenes

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

REFSTEP_BOUNDS     = (1, 20)        # MechMind hard limit for refStep & referredStep
DISTQ_BOUNDS       = (0.5, 3.0)     # distQuantification factor (½×–3× around 1.0)
VOTERATIO_BOUNDS   = (0.5, 0.9)     # maxVoteRatio (Hough threshold)
OUTPUTNUM_BOUNDS   = (1, 3)         # coarse outputNum
ANGLETHRESH_BOUNDS = (45, 135)      # edge-only axis angleThreshold
OPAPP_BOUNDS       = (0, 3)         # operationApproach
DEVCAP_BOUNDS      = (0, 2)         # deviationCorrectionCapacity

ANGLQ_CHOICES = [180, 120, 90, 60]  # angleQuantification (Hough angle bins)
PAIRS_SCALES  = [0.25, 0.5, 1.0, 2.0, 4.0]  # ×warm-start maxNumOfPointPairsPerFeature
