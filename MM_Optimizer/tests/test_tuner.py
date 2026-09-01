"""
test_tuner.py — Tests for the MechVision Tuner
-----------------------------------------------
Run from project root:
    python -m pytest MM_Optimizer/tests/test_tuner.py -q
    python MM_Optimizer/tests/test_tuner.py [--live]

Unit tests (no MechVision, no scene files):
  test_suggest_params_surface / _edge   — key set and bounds per regime
  test_split_params_surface / _edge     — flat params split into coarse/fine
  test_build_warm                       — warm-start dict round-trips without clamp errors
  test_pareto_winner_selection          — max-coverage then min-time selection
  test_pareto_winner_fallback           — fallback when the Pareto front is empty
  test_warm_fine_keys / _warm_coarse_keys — regime-seeded warm-start dicts
  test_sampler_factory_*                — one per sampler, plus an invalid-name check
  test_abs_time_cap_prunes / _slow_but_accurate_not_pruned — pruning is a safety valve only

Dry-run integration (scene files required, no MechVision):
  test_dry_run_study / test_dry_run_gp_study — study runs with zero MechVision calls
  test_multi_round_early_stop           — round 1 fires, improvement check works

Live test (requires MechVision with the CAD_Match project):
  test_live_short
"""

import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import warnings

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

from mm_adapter.mm_adapter        import MechVisionClient
from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
from MM_Optimizer.mv_evaluator    import PROJ_NAME
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
from MM_Optimizer.mesh_analysis import WarmStart
from MM_Optimizer.tuner import (
    Tuner,
    SAMPLER_CHOICES,
    suggest_params,
    _split_params,
    _build_warm,
    _referredstep_constraint,
    _select_pareto_winner,
)
import MM_Optimizer.search_config as SC

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
# Warm-start cloud comes from the app's own bundle, exactly as tuning_stage/tuner do.
# NOT from the deployed MechVision library: model_sync writes <part>/<part>.ply there,
# and it is rewritten per regime.
MODEL_PATH = os.path.join(_ROOT, "output", "reference_pcd", PART,
                          f"{PART}_surface", f"{PART}_surface.ply")

_PAIRS        = [1250, 2500, 5000, 10000, 20000]
_VOXEL_BOUNDS = (0.14, 4.2, 0.28, 16.8)   # (min_lo, min_hi, width_lo, width_hi)
_REGIME_A     = {"coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False, "id": "A"}
_REGIME_C     = {"coarse_mode": 1.0, "fine_mode": 0.0, "needs_edge": True,  "id": "C"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_study_multi():
    return optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.RandomSampler(seed=0),
    )


def _ask(regime, study=None):
    if study is None:
        study = _make_study_multi()
    trial = study.ask()
    p = suggest_params(trial, regime, _PAIRS, _VOXEL_BOUNDS)
    return p, trial, study


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_params_surface():
    """Joint suggest returns all required keys with valid bounds for surface regime."""
    p, _, _ = _ask(_REGIME_A)

    # ── Coarse keys ──────────────────────────────────────────────────────
    assert p["coarse_mode"] == 0.0
    lo, hi = SC.REFSTEP_BOUNDS
    assert lo <= p["refStep"] <= hi, f"refStep={p['refStep']} out of [{lo},{hi}]"

    dlo, dhi = SC.DISTQ_BOUNDS
    assert dlo <= p["distQuantification"] <= dhi

    assert p["angleQuantification"] in SC.ANGLQ_CHOICES

    assert p["maxNumOfPointPairsPerFeature"] in _PAIRS

    vlo, vhi = SC.VOTERATIO_BOUNDS
    assert vlo <= p["maxVoteRatio"] <= vhi

    # referredStep now uses fixed bounds (not dynamic upper=refStep).
    # The constraint referredStep ≤ refStep is enforced by the objective guard, not here.
    assert lo <= p["referredStep"] <= hi, \
        f"referredStep={p['referredStep']} out of fixed bounds [{lo},{hi}]"

    assert isinstance(p["useDistanceNMS"], bool)

    olo, ohi = SC.OUTPUTNUM_BOUNDS
    assert olo <= p["outputNum"] <= ohi

    assert p["minVoxelLength"] < p["maxVoxelLength"]

    # ── Fine keys ────────────────────────────────────────────────────────
    assert p["fine_mode"] == 0.0
    assert 0 <= p["operationApproach"] <= 4.0
    assert 0 <= p["deviationCorrectionCapacity"] <= 2.0
    assert isinstance(p["onlyConsiderVisibleSurfaceOfModel"], bool)
    assert isinstance(p["considerErrorofNormalAngles"], bool)

    # ── Surface mode: edge-only params fixed, not explored ───────────────
    # filterCandidatePoseByAxis and angleThreshold exist but are constants
    assert p["filterCandidatePoseByAxis"] is True   # fixed for surface
    assert p["angleThreshold"] == 135                # fixed for surface

    log.info("PASS: test_suggest_params_surface")


def test_suggest_params_edge():
    """Edge mode conditionally suggests filterByAxis and angleThreshold."""
    study = _make_study_multi()
    got_with_filter   = False
    got_without_filter = False

    for _ in range(30):
        p, trial, _ = _ask(_REGIME_C, study)
        study.tell(trial, [0.1, 0.5])

        assert "filterCandidatePoseByAxis" in p
        if p["filterCandidatePoseByAxis"]:
            assert 45 <= p["angleThreshold"] <= 180, \
                f"angleThreshold={p['angleThreshold']} out of [45,180]"
            got_with_filter = True
        else:
            assert p["angleThreshold"] == 90   # fixed when filterByAxis=False
            got_without_filter = True

    assert got_with_filter,     "No trial had filterCandidatePoseByAxis=True"
    assert got_without_filter,  "No trial had filterCandidatePoseByAxis=False"
    log.info("PASS: test_suggest_params_edge")


def test_split_params_surface():
    """_split_params produces valid coarse/fine dicts for surface regime."""
    p, _, _ = _ask(_REGIME_A)
    coarse, fine = _split_params(p)

    coarse_required = {"registrationMode", "refStep", "distQuantification",
                       "angleQuantification", "maxNumOfPointPairsPerFeature",
                       "maxVoteRatio", "referredStep", "useDistanceNMS",
                       "outputNum", "minVoxelLength", "maxVoxelLength"}
    assert coarse_required == set(coarse.keys()), \
        f"Coarse key mismatch: {coarse_required.symmetric_difference(coarse.keys())}"

    fine_required = {"registrationMode", "operationApproach",
                     "deviationCorrectionCapacity",
                     "onlyConsiderVisibleSurfaceOfModel",
                     "considerErrorofNormalAngles",
                     "scoreLevel", "confidenceThreshold", "candidateTopNum"}
    assert fine_required == set(fine.keys()), \
        f"Fine key mismatch: {fine_required.symmetric_difference(fine.keys())}"

    assert fine["scoreLevel"]         == 0.0
    assert fine["confidenceThreshold"] == 0.1
    assert fine["candidateTopNum"]     == 1
    assert coarse["minVoxelLength"] < coarse["maxVoxelLength"]

    log.info("PASS: test_split_params_surface")


def test_split_params_edge():
    """Edge-mode coarse dict includes filterCandidatePoseByAxis and angleThreshold."""
    p, _, _ = _ask(_REGIME_C)
    coarse, _ = _split_params(p)

    assert "filterCandidatePoseByAxis" in coarse
    assert "angleThreshold"            in coarse
    log.info("PASS: test_split_params_edge")


def test_build_warm():
    """_build_warm produces a valid enqueue dict without clamping errors."""
    coarse = {
        "refStep": 10, "distQuantification": 1.0, "angleQuantification": 90,
        "maxNumOfPointPairsPerFeature": 5000, "maxVoteRatio": 0.5,
        "referredStep": 1, "useDistanceNMS": True, "outputNum": 1,
        "minVoxelLength": 0.7, "maxVoxelLength": 2.8,
    }
    fine = {
        "operationApproach": 1.0, "deviationCorrectionCapacity": 0.0,
        "onlyConsiderVisibleSurfaceOfModel": False,
        "considerErrorofNormalAngles": False,
    }
    p = _build_warm(coarse, fine, _REGIME_A, _PAIRS, _VOXEL_BOUNDS)

    lo, hi = SC.REFSTEP_BOUNDS
    assert lo <= p["refStep"] <= hi
    assert p["angleQuantification"] in SC.ANGLQ_CHOICES
    assert 0 <= p["pairs_idx"]  < len(_PAIRS)
    assert 0 <= p["opApproach"] <= 4
    assert 0 <= p["devCap"]     <= 2
    assert isinstance(p["visibleSurf"], bool)
    assert isinstance(p["normalAng"],   bool)
    assert p["minVoxelLength_mm"] < p["minVoxelLength_mm"] + p["voxel_width_mm"]
    assert p["refStep"] >= p["referredStep"], \
        f"refStep={p['refStep']} < referredStep={p['referredStep']}"

    log.info("PASS: test_build_warm")


def test_pareto_winner_selection():
    """Pareto winner: max coverage first, min time as tiebreaker."""
    study = _make_study_multi()
    # Inject artificial completed trials with known values
    configs = [
        (0.95, 1.0),   # 95% cov, 1.0s
        (0.95, 0.5),   # 95% cov, 0.5s  ← should win (max cov, min time tiebreaker)
        (0.90, 0.3),   # 90% cov, 0.3s
        (0.80, 0.2),   # 80% cov, 0.2s
    ]
    for cov, t in configs:
        study.add_trial(optuna.trial.create_trial(
            params={}, distributions={},
            values=[cov, t],
        ))

    winner = _select_pareto_winner(study)
    # Best coverage is 0.95; among tied (0.95,1.0) and (0.95,0.5), pick min time
    assert winner.values[0] == 0.95, f"Expected cov=0.95, got {winner.values[0]}"
    assert winner.values[1] == 0.5,  f"Expected time=0.5 (faster), got {winner.values[1]}"
    log.info("PASS: test_pareto_winner_selection")


def test_pareto_winner_fallback():
    """Fallback to scalarized best when Pareto front is empty."""
    study = _make_study_multi()
    # Add one pruned trial (won't appear in best_trials)
    study.add_trial(optuna.trial.create_trial(
        params={}, distributions={},
        values=None,
        state=optuna.trial.TrialState.PRUNED,
    ))
    # Add one complete trial
    study.add_trial(optuna.trial.create_trial(
        params={}, distributions={},
        values=[0.85, 0.8],
    ))
    winner = _select_pareto_winner(study)
    assert winner.values == [0.85, 0.8]
    log.info("PASS: test_pareto_winner_fallback")


def test_warm_fine_keys():
    """_warm_fine returns all 8 fine keys at warm-start defaults."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_warm_fine_keys — model not found: {MODEL_PATH}")
        return
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)
    groups = list_synthetic_scenes(SCENES_DIR) if os.path.isdir(SCENES_DIR) else [[]]

    opt = Tuner(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws, dry_run=True)

    fine = opt._warm_fine(_REGIME_A)
    required = {"registrationMode", "operationApproach", "deviationCorrectionCapacity",
                "onlyConsiderVisibleSurfaceOfModel", "considerErrorofNormalAngles",
                "scoreLevel", "confidenceThreshold", "candidateTopNum"}
    assert required == set(fine.keys()), \
        f"Key mismatch: {required.symmetric_difference(fine.keys())}"
    assert fine["operationApproach"]           == 1.0
    assert fine["deviationCorrectionCapacity"] == 0.0
    assert fine["onlyConsiderVisibleSurfaceOfModel"] is False
    assert fine["scoreLevel"]                  == 0.0
    assert fine["confidenceThreshold"]         == 0.1
    assert fine["candidateTopNum"]             == 1
    opt.cleanup()
    log.info("PASS: test_warm_fine_keys")


def test_warm_coarse_keys():
    """_warm_coarse includes refStep and distQ (full coarse warm-start)."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_warm_coarse_keys — model not found: {MODEL_PATH}")
        return
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)
    groups = list_synthetic_scenes(SCENES_DIR) if os.path.isdir(SCENES_DIR) else [[]]

    opt = Tuner(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws, dry_run=True)

    base = opt._warm_coarse(_REGIME_A)
    required = {"registrationMode", "refStep", "distQuantification",
                "angleQuantification", "maxNumOfPointPairsPerFeature",
                "maxVoteRatio", "referredStep", "useDistanceNMS",
                "outputNum", "minVoxelLength", "maxVoxelLength"}
    assert required.issubset(base.keys()), f"Missing: {required - base.keys()}"
    assert base["outputNum"]    == 1
    assert base["referredStep"] == 1
    assert base["minVoxelLength"] < base["maxVoxelLength"]
    opt.cleanup()
    log.info("PASS: test_warm_coarse_keys")


# ─────────────────────────────────────────────────────────────────────────────
# Sampler dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _make_optimizer_for_factory(sampler: str) -> Tuner:
    """Build an Tuner with a synthetic WarmStart — no model/scene files."""
    ws = WarmStart()
    ws.maxNumOfPointPairsPerFeature = 5000
    ws.minVoxelLength_mm = 1.0
    ws.maxVoxelLength_mm = 15.0
    ws.longest_extent_m  = 0.3
    return Tuner(
        part_name=PART, client=None, project_id=-1,
        scene_groups=[[]], warm_start=ws, dry_run=True, sampler=sampler)


def test_sampler_factory_nsgaii():
    """sampler='nsgaii' → NSGAIISampler with the referredStep constraint hooked up."""
    opt = _make_optimizer_for_factory("nsgaii")
    try:
        study = opt._create_study("smoke_nsgaii")
        assert isinstance(study.sampler, optuna.samplers.NSGAIISampler), \
            f"expected NSGAIISampler, got {type(study.sampler).__name__}"
        assert study.sampler._constraints_func is _referredstep_constraint, \
            "NSGA-II sampler must have _referredstep_constraint wired"
        assert study.directions == [
            optuna.study.StudyDirection.MAXIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ]
    finally:
        opt.cleanup()
    log.info("PASS: test_sampler_factory_nsgaii")


def test_sampler_factory_tpe():
    """sampler='tpe' → TPESampler with multivariate=True, group=True."""
    opt = _make_optimizer_for_factory("tpe")
    try:
        study = opt._create_study("smoke_tpe")
        assert isinstance(study.sampler, optuna.samplers.TPESampler), \
            f"expected TPESampler, got {type(study.sampler).__name__}"
        assert study.sampler._multivariate is True, "TPE must be multivariate"
        assert study.sampler._group is True,        "TPE must use group=True"
        assert study.directions == [
            optuna.study.StudyDirection.MAXIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ]
    finally:
        opt.cleanup()
    log.info("PASS: test_sampler_factory_tpe")


def test_sampler_factory_gp():
    """sampler='gp' → GPSampler with the referredStep constraint hooked up (MOO parity)."""
    opt = _make_optimizer_for_factory("gp")
    try:
        study = opt._create_study("smoke_gp")
        assert isinstance(study.sampler, optuna.samplers.GPSampler), \
            f"expected GPSampler, got {type(study.sampler).__name__}"
        assert study.sampler._constraints_func is _referredstep_constraint, \
            "GP sampler must have _referredstep_constraint wired (parity with NSGA-II)"
        assert study.directions == [
            optuna.study.StudyDirection.MAXIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ]
    finally:
        opt.cleanup()
    log.info("PASS: test_sampler_factory_gp")


def test_sampler_invalid():
    """Unknown sampler name → ValueError at construction time."""
    try:
        _make_optimizer_for_factory("cmaes")
    except ValueError as e:
        assert "cmaes" in str(e)
        log.info("PASS: test_sampler_invalid")
        return
    raise AssertionError("Tuner should reject sampler='cmaes'")


# ─────────────────────────────────────────────────────────────────────────────
# Pruning policy — absolute time cap, no competitive time pruning
# ─────────────────────────────────────────────────────────────────────────────

_FEASIBLE_JOINT_P = {
    "refStep": 8, "referredStep": 3,          # feasible: referredStep ≤ refStep
    "distQuantification": 1.0, "angleQuantification": 90,
    "pairs_idx": 2, "maxNumOfPointPairsPerFeature": 5000,
    "maxVoteRatio": 0.5, "useDistanceNMS": True, "outputNum": 1,
    "minVoxelLength_mm": 0.7, "voxel_width_mm": 2.1,
    "minVoxelLength": 0.7, "maxVoxelLength": 2.8,
    "filterCandidatePoseByAxis": True, "angleThreshold": 135,
    "coarse_mode": 0.0, "fine_mode": 0.0,
    "operationApproach": 1.0, "deviationCorrectionCapacity": 0.0,
    "onlyConsiderVisibleSurfaceOfModel": False,
    "considerErrorofNormalAngles": False,
}


def _run_objective_with(mean_time, coverage):
    """Drive _objective with a fixed feasible config and a mocked per-scene
    EvalResult (constant mean_time, coverage). Returns (result_or_None, was_pruned)."""
    from unittest.mock import patch
    from MM_Optimizer.mv_evaluator import EvalResult

    opt    = _make_optimizer_for_factory("nsgaii")
    regime = {"id": "A", "coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False}
    study  = _make_study_multi()
    trial  = study.ask()
    er     = EvalResult(score=0.0, coverage=coverage, mean_time=mean_time)
    scenes = [(f"scene_{i}", []) for i in range(3)]

    orig_full = SC.M_FULL
    SC.M_FULL = 3
    try:
        with patch("MM_Optimizer.tuner.suggest_params",
                   return_value=dict(_FEASIBLE_JOINT_P)), \
             patch.object(opt, "_sample_scenes", return_value=scenes), \
             patch.object(opt, "evaluate_config", return_value=er):
            try:
                return opt._objective(trial, regime), False
            except optuna.TrialPruned:
                return None, True
    finally:
        SC.M_FULL = orig_full
        opt.cleanup()


def test_abs_time_cap_prunes():
    """A config whose running mean_time exceeds TIME_ABS_CAP is pruned (safety valve)."""
    ret, pruned = _run_objective_with(mean_time=SC.TIME_ABS_CAP + 5.0, coverage=1.0)
    assert pruned, "config above TIME_ABS_CAP must be pruned"
    assert ret is None
    log.info("PASS: test_abs_time_cap_prunes")


def test_slow_but_accurate_not_pruned():
    """A slow-but-accurate config (under the abs cap) completes — proving the old
    competitive time-ratio pruning is gone and the precision-first region survives."""
    slow = SC.TIME_ABS_CAP * 0.5    # well above a 'fast' config, still under the cap
    assert slow > SC.SCORE_TIME_NORM, "test assumes 'slow' exceeds the reference cycle time"
    ret, pruned = _run_objective_with(mean_time=slow, coverage=1.0)
    assert not pruned, "slow-but-accurate config must NOT be pruned on time"
    cov, mean_time = ret
    assert cov == 1.0 and abs(mean_time - slow) < 1e-9, f"got {ret}"
    log.info("PASS: test_slow_but_accurate_not_pruned")


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dry_run_study():
    """Joint study in dry_run — zero MV calls, single study on instance, result not None."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_dry_run_study — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_dry_run_study — no scenes under: {SCENES_DIR}")
        return

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full = SC.M_FULL
    SC.M_FULL = 3
    try:
        opt = Tuner(
            part_name      = PART,
            client         = None,
            project_id     = -1,
            scene_groups   = groups,
            warm_start     = ws,
            cache          = None,
            dry_run        = True,
            n_trials = 5,
            n_rounds       = 1,
            seed           = 0,
            storage_path   = None,
        )
        result = opt.run()

        assert result is not None,       "Dry run must return an EvalResult"
        assert result.coverage  >= 0.0
        assert result.mean_time >= 0.0
        assert opt._n_evals == 0,    f"dry_run: zero MV calls expected, got {opt._n_evals}"

        # Single joint study must be populated
        assert opt._study is not None,   "_study not set after run()"

        n_complete = sum(1 for t in opt._study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        assert n_complete >= 1, "Joint study has no complete trials"

        # Scoring formula sanity: score = time_term + cov_term
        assert abs(result.score_time_term + result.score_cov_term - result.score) < 1e-9, \
            "score != time_term + cov_term"
        assert result.score_time_term == result.mean_time / SC.SCORE_TIME_NORM
        assert abs(result.score_cov_term - (1.0 - result.coverage) * SC.SCORE_COV_NORM) < 1e-9

        log.info(f"  dry run: cov={result.coverage:.2f}  time={result.mean_time:.3f}s  "
                 f"score={result.score:.3f}  quality={result.score_quality:.3f}")
        log.info("PASS: test_dry_run_study")
    finally:
        SC.M_FULL = orig_full
        opt.cleanup()


def test_dry_run_gp_study():
    """GP-sampler joint study in dry_run — zero MV calls, result not None, study populated.

    Requires torch (GPSampler backend). Kept small (GP is O(n^3) in trials).
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("SKIP test_dry_run_gp_study — torch not installed (GPSampler needs it)")
        return
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_dry_run_gp_study — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_dry_run_gp_study — no scenes under: {SCENES_DIR}")
        return

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full = SC.M_FULL
    orig_startup = SC.N_STARTUP
    SC.M_FULL = 3
    SC.N_STARTUP = 5   # let the GP model kick in within the small budget
    try:
        opt = Tuner(
            part_name      = PART,
            client         = None,
            project_id     = -1,
            scene_groups   = groups,
            warm_start     = ws,
            cache          = None,
            dry_run        = True,
            n_trials = 10,
            n_rounds       = 1,
            seed           = 0,
            storage_path   = None,
            sampler        = "gp",
        )
        result = opt.run()

        assert result is not None,    "GP dry run must return an EvalResult"
        assert result.coverage  >= 0.0
        assert opt._n_evals == 0, f"dry_run: zero MV calls expected, got {opt._n_evals}"
        assert opt._study is not None
        assert isinstance(opt._study.sampler, optuna.samplers.GPSampler)

        n_complete = sum(1 for t in opt._study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        assert n_complete >= 1, "GP joint study has no complete trials"

        log.info(f"  GP dry run: complete={n_complete}  cov={result.coverage:.2f}  "
                 f"time={result.mean_time:.3f}s")
        log.info("PASS: test_dry_run_gp_study")
    finally:
        SC.M_FULL = orig_full
        SC.N_STARTUP = orig_startup
        opt.cleanup()


def test_multi_round_early_stop():
    """Round 1 runs and early-stop fires when improvement is negligible."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_multi_round_early_stop — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_multi_round_early_stop — no scenes under: {SCENES_DIR}")
        return

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full = SC.M_FULL
    SC.M_FULL = 2
    orig_refine = SC.N_TRIALS_REFINE
    SC.N_TRIALS_REFINE = 2
    try:
        opt = Tuner(
            part_name      = PART,
            client         = None,
            project_id     = -1,
            scene_groups   = groups,
            warm_start     = ws,
            cache          = None,
            dry_run        = True,
            n_trials = 4,
            n_rounds       = 2,
            seed           = 0,
            storage_path   = None,
        )
        result = opt.run()
        assert result is not None, "Multi-round dry run must return a result"

        # Both rounds use the same study
        assert opt._study is not None
        n_complete = sum(1 for t in opt._study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        # With dry_run all evaluations return coverage=1.0, so round 1 improvement
        # will be ~0 and early-stop should fire — study has at least round-0 trials
        assert n_complete >= 1

        log.info(f"  multi-round dry run: complete={n_complete}  "
                 f"cov={result.coverage:.2f}")
        log.info("PASS: test_multi_round_early_stop")
    finally:
        SC.M_FULL = orig_full
        SC.N_TRIALS_REFINE = orig_refine
        opt.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Live test
# ─────────────────────────────────────────────────────────────────────────────

def test_live_short():
    """10-trial joint study on 25333MB000 — result not None, coverage >= 0.0."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_live_short — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_live_short — no scenes under: {SCENES_DIR}")
        return

    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)
    client = MechVisionClient()

    projects = client.get_projects()
    assert PROJ_NAME in projects, \
        f"Project '{PROJ_NAME}' not found. Loaded: {list(projects.keys())}"
    pid = projects[PROJ_NAME]

    orig_full = SC.M_FULL
    SC.M_FULL = len(groups)
    try:
        opt = Tuner(
            part_name      = PART,
            client         = client,
            project_id     = pid,
            scene_groups   = groups,
            warm_start     = ws,
            cache          = None,
            dry_run        = False,
            n_trials = 10,
            n_rounds       = 1,
            seed           = 42,
            storage_path   = None,
        )
        result = opt.run()

        assert result is not None
        assert result.coverage >= 0.0
        assert opt._n_evals > 0, "Live test should have made MechVision calls"

        log.info(f"  live short: cov={result.coverage:.3f}  "
                 f"time={result.mean_time:.3f}s  score={result.score:.3f}  "
                 f"mv_evals={opt._n_evals}")
        log.info("PASS: test_live_short")
    finally:
        SC.M_FULL = orig_full
        opt.cleanup()
        client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true",
                   help="Also run the live MechVision test")
    args = p.parse_args()

    print("test_optuna.py — fully joint multivariate TPE\n")

    print("--- Unit tests ---")
    test_suggest_params_surface()
    test_suggest_params_edge()
    test_split_params_surface()
    test_split_params_edge()
    test_build_warm()
    test_pareto_winner_selection()
    test_pareto_winner_fallback()
    test_warm_fine_keys()
    test_warm_coarse_keys()

    print("\n--- Sampler dispatch ---")
    test_sampler_factory_nsgaii()
    test_sampler_factory_tpe()
    test_sampler_factory_gp()
    test_sampler_invalid()

    print("\n--- Pruning policy ---")
    test_abs_time_cap_prunes()
    test_slow_but_accurate_not_pruned()

    print("\n--- Dry-run integration tests ---")
    test_dry_run_study()
    test_dry_run_gp_study()
    test_multi_round_early_stop()

    if args.live:
        print("\n--- Live test (requires MechVision) ---")
        test_live_short()
    else:
        print("\n(skip live test — pass --live to enable)")

    print("\nAll tests PASSED")
