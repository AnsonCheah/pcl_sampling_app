"""Search space, scoring constants and gates for the MechVision tuner.

Pure data -- no imports from mm_adapter or tuner logic. Edit here to change what
`tuner.suggest_params` explores; the same space and warm-start seed are shared by all
three samplers (NSGA-II / TPE / GP).
"""

# ---------------------------------------------------------------------------
# Scoring thresholds
# ---------------------------------------------------------------------------

POS_THRESH_LOOSE = 0.005    # 5 mm  -- regime gate
POS_THRESH_TIGHT = 0.002    # 2 mm
ANG_THRESH_TIGHT = 5.0      # 5deg
POS_THRESH_MATCH = 0.006    # 3 x tight -- threshold-gated NN matching gate

# Position-only scoring, now reserved for continuous-symmetry parts (`ambiguity_fold == 0`),
# whose roll about the axis is physically unrecoverable -- scoring it would report ~0 coverage
# whatever any parameter did.
#
# This used to be the threshold for the regime gate AND the whole study, on the reasoning that
# a rotationally symmetric part returns a geometrically valid but orientation-flipped pose. The
# symmetry search (`rotationStrategy` + `angleStep`) is what fixes those flips, and it can only
# be tuned against an objective that can see them: scored position-only, a finer angleStep is
# pure cost with zero upside and every sampler drives it to 360.
ANG_THRESH_POSITION_ONLY = 360.0

M_FULL = 30                 # scenes per full evaluation

# ---------------------------------------------------------------------------
# Regimes -- (coarse_mode, fine_mode); 0.0=Surface, 1.0=Edge
# ---------------------------------------------------------------------------

# Coarse and fine always share a cloud type: `3d_matching/<part>/` holds exactly one
# `<part>.ply`, swapped per regime by `model_sync.py`. Mixed combinations would need two
# models loaded at once, which that layout cannot express.
REGIMES = [
    {"id": "A", "coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False},   # surface
    {"id": "B", "coarse_mode": 1.0, "fine_mode": 1.0, "needs_edge": True},    # edge
]

# Min coverage for a regime to pass. The gate scores orientation like the study, but runs
# BEFORE angleStep is tuned, so an N-fold part is expected to return flipped poses here.
#
# Bound, not a guess: when fine matching picks uniformly among N equivalent basins, expected
# tight coverage is ~1/N of the position-only coverage. The largest fold worth handling is 6
# (SYM_NFOLD_CANDIDATES), so a gate above 1/6 = 0.167 would reject exactly the parts the
# symmetry search exists to rescue. 0.15 sits just under that.
#
# Measured on 25333MB000 with default params: position-only 0.945, tight 0.844 (angular error
# median 1.2 deg, p75 1.9 deg, max 89.9 deg -- the 90 deg tail is the 4-fold flip). A working
# regime clears this gate by a wide margin; the low value only protects the ambiguous case.
REGIME_COVERAGE_GATE = 0.15


def approach_candidates(median_coarse_pos_err_m):
    """operationApproach values worth trying, given coarse pose quality (metres)."""
    if median_coarse_pos_err_m < 0.003:
        return [0.0, 1.0]            # HighSpeed, Standard
    if median_coarse_pos_err_m < 0.010:
        return [1.0, 2.0]            # Standard, HighAccuracy
    return [2.0, 3.0]                # HighAccuracy, ExtraHighAccuracy


# ---------------------------------------------------------------------------
# Symmetry search -- MechVision seeds fine matching from several initial orientations
# about one geocenter-frame axis, stepping angleStep from minAngle to maxAngle.
# ---------------------------------------------------------------------------

# rotationStrategy encoding is 0=X, 1=Y, 2=Z. `pcd_geocenter(pcd, axis=dominant)` makes the
# ambiguity axis frame Z, so Z is the answer by construction whenever the exported bundle is
# ambiguity-aligned. mm_adapter defaults this field to 1.0 (Y), so it must be set explicitly.
ROTATION_STRATEGY_Z = 2.0

# MechVision's documented "no symmetry" value.
ANGLE_STEP_NONE = 360

# Floor on the step, hence a ceiling on cost: seeds ~= 360/step + 1, so 5 degrees is already
# ~73 registrations per candidate pose.
ANGLE_STEP_FLOOR = 5

# Divisors of 360 at or above the floor, so every step tiles the circle exactly -- a step that
# leaves a remainder arc spaces the seeds unevenly and leaves one gap wider than the rest.
#
# Ascending, and suggested by INDEX rather than value: the sampler then sees an ordered
# quantity (90 is nearer 72 than 5) instead of an unordered categorical.
ANGLE_STEP_LADDER = [5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30,
                     36, 40, 45, 60, 72, 90, 120, 180, 360]

# ---------------------------------------------------------------------------
# Symmetry classification (mesh_analysis._classify_symmetry)
# ---------------------------------------------------------------------------

# Chamfer threshold as a fraction of model diameter. 2% sits ~3-5x above sampling noise
# (~1-3 mm for 5000 points) and well below the asymmetric signal (~10-50% of diameter).
SYM_CHAMFER_THRESH_FRAC  = 0.02
SYM_CHAMFER_THRESH_MIN_M = 0.003  # absolute floor so parts <150 mm keep some margin

SYM_NFOLD_CANDIDATES = [6, 4, 3, 2]   # only divisors of 360 matter for industrial parts

# Inertia eigenvalue ratios for SO2/SO3 proposal. Lenient enough for sampling noise on
# sphere/disc clouds; an elongated cylinder sits ~5x, well clear.
SYM_SO3_EIGEN_RATIO = 1.15
SYM_SO2_EIGEN_RATIO = 1.10

# ---------------------------------------------------------------------------
# Normalized scoring
#   raw_score = mean_time / SCORE_TIME_NORM + (1 - coverage) * SCORE_COV_NORM
#   score_quality = 1 - raw_score / SCORE_WORST_CASE  in [0, 1]
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
# competitive pruning. Deliberately generous (~5x a good config) so slow-but-accurate
# trials are never pruned on time.
TIME_ABS_CAP     = 20.0
TIME_INITIAL_CAP = 5.0   # sentinel time (with coverage=0) for infeasible referredStep
COV_PRUNE_FLOOR  = 0.10  # prune if running coverage is below this after 3+ scenes

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

REFSTEP_BOUNDS     = (1, 20)        # MechMind hard limit for refStep & referredStep
DISTQ_BOUNDS       = (0.5, 3.0)     # distQuantification factor (1/2x-3x around 1.0)
VOTERATIO_BOUNDS   = (0.5, 0.9)     # maxVoteRatio (Hough threshold)
OUTPUTNUM_BOUNDS   = (1, 3)         # coarse outputNum
ANGLETHRESH_BOUNDS = (45, 135)      # edge-only axis angleThreshold
OPAPP_BOUNDS       = (0, 3)         # operationApproach
DEVCAP_BOUNDS      = (0, 2)         # deviationCorrectionCapacity

ANGLQ_CHOICES = [180, 120, 90, 60]  # angleQuantification (Hough angle bins)
PAIRS_SCALES  = [0.25, 0.5, 1.0, 2.0, 4.0]  # xwarm-start maxNumOfPointPairsPerFeature
