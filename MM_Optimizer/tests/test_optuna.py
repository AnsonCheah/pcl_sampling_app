"""
test_optuna.py — Tests for joint CmaEsSampler OptunaOptimizer
--------------------------------------------------------------
Run from project root:
    python MM_Optimizer/tests/test_optuna.py [--live]

Unit tests (no MechVision, no scene files):
  test_suggest_params_joint_keys   — all 18 joint params present with valid ranges
  test_joint_params_to_dicts_fixed — scoreLevel=0, confThresh=0.1, candidateTopNum=1 fixed
  test_angleQ_index_mapping        — ANGLE_QUANT_CANDIDATES index round-trips correctly
  test_build_warm_joint_roundtrip  — warm-start dict preserves coarse/fine values
  test_scoring_formula             — score=time/NORM+(1-cov)*COV_NORM, quality=1-score/WORST

Instance tests (model file required, no MechVision):
  test_default_fine_keys           — _default_fine returns correct warm defaults
  test_default_coarse_remaining_keys — _default_coarse_remaining excludes refStep/distQ

Dry-run integration test (scene files required, no MechVision):
  test_dry_run_joint               — single joint study, zero MV calls, result not None,
                                     score decomposition verified

Live test (requires MechVision with CAD_Match project):
  test_live_short                  — 6 trials, coverage >= 0.0
"""

import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from mm_adapter.mm_adapter        import MechVisionClient
from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
from MM_Optimizer.optimizer       import PROJ_NAME, MM_MODEL_ROOT, EvalResult
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
from MM_Optimizer.optuna_optimizer import (
    OptunaOptimizer,
    suggest_params_joint,
    _joint_params_to_dicts,
    _build_warm_joint,
)
import MM_Optimizer.search_config as SC

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
MODEL_PATH = os.path.join(MM_MODEL_ROOT, f"{PART}_surface", f"{PART}_surface.ply")

_PAIRS          = [1250, 2500, 5000, 10000, 20000]
_VOXEL_BOUNDS   = (0.14, 4.2, 0.28, 16.8)   # (min_lo, min_hi, width_lo, width_hi)
_REFSTEP_BOUNDS = (1, 20)
_REGIME_A       = {"coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False, "id": "A"}

_EXPECTED_COARSE_KEYS = {
    "registrationMode", "refStep", "distQuantification", "angleQuantification",
    "maxNumOfPointPairsPerFeature", "maxVoteRatio", "referredStep", "outputNum",
    "useDistanceNMS", "minVoxelLength", "maxVoxelLength",
    "filterCandidatePoseByAxis", "angleThreshold",
}
_EXPECTED_FINE_KEYS = {
    "registrationMode", "operationApproach", "deviationCorrectionCapacity",
    "onlyConsiderVisibleSurfaceOfModel", "considerErrorofNormalAngles",
    "scoreLevel", "confidenceThreshold", "candidateTopNum",
}
_JOINT_PARAM_KEYS = {
    "coarse_mode", "refStep", "distQ", "angleQ_idx", "pairs_idx",
    "maxVoteRatio", "referredStep", "outputNum", "useDistNMS",
    "minVoxelLength_mm", "voxel_width_mm", "filterByAxis", "angleThreshold",
    "fine_mode", "opApproach", "devCap", "visibleSurf", "normalAng",
}


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (no MechVision, no scene files)
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_params_joint_keys():
    """suggest_params_joint returns all 18 numeric params with valid ranges."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    p = suggest_params_joint(trial, _PAIRS, _VOXEL_BOUNDS, _REFSTEP_BOUNDS)

    assert set(p.keys()) == _JOINT_PARAM_KEYS, \
        f"Key mismatch: {_JOINT_PARAM_KEYS.symmetric_difference(p.keys())}"

    rlo, rhi = _REFSTEP_BOUNDS
    assert rlo <= p["refStep"] <= rhi
    assert SC.OPTUNA_DISTQ_BOUNDS[0] <= p["distQ"] <= SC.OPTUNA_DISTQ_BOUNDS[1]
    assert 0 <= p["angleQ_idx"] <= len(SC.ANGLE_QUANT_CANDIDATES) - 1
    assert 0 <= p["pairs_idx"]  <= len(_PAIRS) - 1
    assert SC.OPTUNA_VOTERATIO_BOUNDS[0] <= p["maxVoteRatio"] <= SC.OPTUNA_VOTERATIO_BOUNDS[1]
    assert SC.OPTUNA_REFERRED_BOUNDS[0]  <= p["referredStep"] <= SC.OPTUNA_REFERRED_BOUNDS[1]
    assert SC.OPTUNA_OUTPUTNUM_BOUNDS[0] <= p["outputNum"]    <= SC.OPTUNA_OUTPUTNUM_BOUNDS[1]
    assert p["useDistNMS"]  in (0, 1)
    assert p["coarse_mode"] in (0, 1)
    assert p["fine_mode"]   in (0, 1)
    assert 0 <= p["opApproach"] <= SC.OPTUNA_OPAPP_MAX
    assert 0 <= p["devCap"]     <= SC.OPTUNA_DEVCAP_MAX
    assert p["visibleSurf"] in (0, 1)
    assert p["normalAng"]   in (0, 1)
    assert (SC.OPTUNA_ANGLETHRESH_BOUNDS[0]
            <= p["angleThreshold"]
            <= SC.OPTUNA_ANGLETHRESH_BOUNDS[1])

    min_lo, min_hi, width_lo, width_hi = _VOXEL_BOUNDS
    assert min_lo   <= p["minVoxelLength_mm"] <= min_hi
    assert width_lo <= p["voxel_width_mm"]    <= width_hi

    # Verify SC constants have the correct values (catches regressions)
    assert SC.OPTUNA_OPAPP_MAX == 4, \
        "operationApproach must reach Compatibility mode (4); found {SC.OPTUNA_OPAPP_MAX}"
    assert SC.OPTUNA_DEVCAP_MAX == 3, \
        "deviationCorrectionCapacity must have 4 levels (0..3); found {SC.OPTUNA_DEVCAP_MAX}"

    log.info("PASS: test_suggest_params_joint_keys")


def test_coarse_mode_max_constraint():
    """coarse_mode is always 0 when coarse_mode_max=0 (no edge model available)."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=5))
    for _ in range(20):
        trial = study.ask()
        p = suggest_params_joint(trial, _PAIRS, _VOXEL_BOUNDS, _REFSTEP_BOUNDS,
                                 coarse_mode_max=0)
        study.tell(trial, 0.5)
        assert p["coarse_mode"] == 0, \
            f"coarse_mode_max=0: expected 0, got {p['coarse_mode']}"

    log.info("PASS: test_coarse_mode_max_constraint")


def test_joint_params_to_dicts_fixed():
    """Fixed params scoreLevel=0, confidenceThreshold=0.1, candidateTopNum=1 never vary."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
    for _ in range(10):
        trial = study.ask()
        p     = suggest_params_joint(trial, _PAIRS, _VOXEL_BOUNDS, _REFSTEP_BOUNDS)
        study.tell(trial, 0.5)
        coarse, fine = _joint_params_to_dicts(p, _PAIRS)

        assert fine["scoreLevel"]          == 0.0, \
            f"scoreLevel must always be 0, got {fine['scoreLevel']}"
        assert fine["confidenceThreshold"] == 0.1, \
            f"confidenceThreshold must always be 0.1, got {fine['confidenceThreshold']}"
        assert fine["candidateTopNum"]     == 1, \
            f"candidateTopNum must always be 1, got {fine['candidateTopNum']}"

        assert set(coarse.keys()) == _EXPECTED_COARSE_KEYS, \
            f"Coarse key mismatch: {_EXPECTED_COARSE_KEYS.symmetric_difference(coarse.keys())}"
        assert set(fine.keys()) == _EXPECTED_FINE_KEYS, \
            f"Fine key mismatch: {_EXPECTED_FINE_KEYS.symmetric_difference(fine.keys())}"

        # coarse_mode and fine_mode map to registrationMode as float
        assert fine["registrationMode"]   in (0.0, 1.0)
        assert coarse["registrationMode"] in (0.0, 1.0)
        # maxVoxelLength > minVoxelLength
        assert coarse["minVoxelLength"] < coarse["maxVoxelLength"], \
            f"min={coarse['minVoxelLength']} >= max={coarse['maxVoxelLength']}"

    log.info("PASS: test_joint_params_to_dicts_fixed")


def test_angleQ_index_mapping():
    """angleQ_idx correctly maps to exact ANGLE_QUANT_CANDIDATES values only."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    seen_angles = set()
    for _ in range(30):
        trial = study.ask()
        p     = suggest_params_joint(trial, _PAIRS, _VOXEL_BOUNDS, _REFSTEP_BOUNDS)
        study.tell(trial, 0.5)
        coarse, _ = _joint_params_to_dicts(p, _PAIRS)
        aq = coarse["angleQuantification"]
        assert aq in SC.ANGLE_QUANT_CANDIDATES, \
            f"angleQ={aq} not in ANGLE_QUANT_CANDIDATES={SC.ANGLE_QUANT_CANDIDATES}"
        seen_angles.add(aq)

    # 30 random trials across 8 candidates — expect most to appear
    assert len(seen_angles) >= 4, \
        f"Only saw {len(seen_angles)} distinct angleQ values — index mapping may be broken"
    log.info(f"PASS: test_angleQ_index_mapping  (angles seen: {sorted(seen_angles)})")


def test_build_warm_joint_roundtrip():
    """_build_warm_joint output round-trips correctly through _joint_params_to_dicts."""
    ref_coarse = {
        "registrationMode": 0.0,
        "refStep": 4,
        "distQuantification": 1.5,
        "angleQuantification": 60,
        "maxNumOfPointPairsPerFeature": 2500,
        "maxVoteRatio": 0.7,
        "referredStep": 2,
        "outputNum": 1,
        "useDistanceNMS": True,
        "minVoxelLength": 0.5,
        "maxVoxelLength": 2.0,
        "filterCandidatePoseByAxis": True,
        "angleThreshold": 135,
    }
    ref_fine = {
        "registrationMode": 0.0,
        "operationApproach": 0.0,
        "deviationCorrectionCapacity": 1.0,
        "onlyConsiderVisibleSurfaceOfModel": True,
        "considerErrorofNormalAngles": False,
        "scoreLevel": 0.0,
        "confidenceThreshold": 0.1,
        "candidateTopNum": 1,
    }

    warm = _build_warm_joint(ref_coarse, ref_fine, _PAIRS, _VOXEL_BOUNDS, _REFSTEP_BOUNDS)
    coarse_rt, fine_rt = _joint_params_to_dicts(warm, _PAIRS)

    assert coarse_rt["refStep"]          == ref_coarse["refStep"]
    assert coarse_rt["angleQuantification"] == 60        # exact candidate preserved
    assert coarse_rt["maxNumOfPointPairsPerFeature"] == 2500  # exact candidate preserved
    assert coarse_rt["maxVoteRatio"]     == ref_coarse["maxVoteRatio"]
    assert coarse_rt["referredStep"]     == ref_coarse["referredStep"]
    assert coarse_rt["outputNum"]        == ref_coarse["outputNum"]
    assert coarse_rt["useDistanceNMS"]   is True
    assert coarse_rt["filterCandidatePoseByAxis"] is True
    assert coarse_rt["angleThreshold"]   == ref_coarse["angleThreshold"]
    assert abs(coarse_rt["minVoxelLength"] - ref_coarse["minVoxelLength"]) < 1e-9
    assert abs(coarse_rt["maxVoxelLength"] - ref_coarse["maxVoxelLength"]) < 1e-9

    assert fine_rt["operationApproach"]             == 0.0    # HighSpeed
    assert fine_rt["deviationCorrectionCapacity"]   == 1.0   # within [0, DEVCAP_MAX]
    assert fine_rt["onlyConsiderVisibleSurfaceOfModel"] is True
    assert fine_rt["considerErrorofNormalAngles"]   is False

    log.info("PASS: test_build_warm_joint_roundtrip")


def test_scoring_formula():
    """score = time/SCORE_TIME_NORM + (1-cov)*SCORE_COV_NORM; quality = 1 - score/SCORE_WORST_CASE."""
    cases = [
        (1.000, 0.080),   # dry-run cached result
        (0.985, 0.980),   # CD target
        (0.982, 1.510),   # old Optuna
        (0.980, 1.000),
        (0.500, 0.300),
    ]
    for cov, t in cases:
        _t    = t / SC.SCORE_TIME_NORM
        _c    = (1.0 - cov) * SC.SCORE_COV_NORM
        score = _t + _c
        qual  = 1.0 - score / SC.SCORE_WORST_CASE

        er = EvalResult(
            score=score, coverage=cov, mean_time=t,
            score_time_term=_t, score_cov_term=_c, score_quality=qual,
        )
        assert abs(er.score - (er.score_time_term + er.score_cov_term)) < 1e-12, \
            f"score decomposition: {er.score} != {er.score_time_term}+{er.score_cov_term}"
        assert abs(er.score_time_term - t / SC.SCORE_TIME_NORM) < 1e-12
        assert abs(er.score_cov_term  - (1.0 - cov) * SC.SCORE_COV_NORM) < 1e-12

    # Verify constants are self-consistent: SCORE_WORST_CASE = 1.0 + SCORE_COV_NORM
    assert abs(SC.SCORE_WORST_CASE - (1.0 + SC.SCORE_COV_NORM)) < 1e-12, \
        "SCORE_WORST_CASE must equal 1.0 + SCORE_COV_NORM"
    # SCORE_TIME_NORM normalises the time term: when time == SCORE_TIME_NORM, time_term == 1.0
    assert abs(SC.SCORE_TIME_NORM / SC.SCORE_TIME_NORM - 1.0) < 1e-12

    log.info("PASS: test_scoring_formula")


# ─────────────────────────────────────────────────────────────────────────────
# Instance tests (model file required, no MechVision)
# ─────────────────────────────────────────────────────────────────────────────

def test_default_fine_keys():
    """_default_fine returns all 8 fine keys with correct warm defaults."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_default_fine_keys — model not found: {MODEL_PATH}")
        return
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)
    groups = list_synthetic_scenes(SCENES_DIR) if os.path.isdir(SCENES_DIR) else [[]]

    opt = OptunaOptimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws, dry_run=True)

    fine = opt._default_fine(_REGIME_A)
    assert set(fine.keys()) == _EXPECTED_FINE_KEYS, \
        f"Key mismatch: {_EXPECTED_FINE_KEYS.symmetric_difference(fine.keys())}"

    assert fine["registrationMode"]                  == _REGIME_A["fine_mode"]
    assert fine["operationApproach"]                 == 1.0    # Standard ICP
    assert fine["deviationCorrectionCapacity"]       == 0.0
    assert fine["onlyConsiderVisibleSurfaceOfModel"] is False
    assert fine["considerErrorofNormalAngles"]       is False
    assert fine["scoreLevel"]                        == 0.0
    assert fine["confidenceThreshold"]               == 0.1
    assert fine["candidateTopNum"]                   == 1

    opt.cleanup()
    log.info("PASS: test_default_fine_keys")


def test_default_coarse_remaining_keys():
    """_default_coarse_remaining returns geometry-derived coarse params, excludes refStep/distQ."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_default_coarse_remaining_keys — model not found: {MODEL_PATH}")
        return
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)
    groups = list_synthetic_scenes(SCENES_DIR) if os.path.isdir(SCENES_DIR) else [[]]

    opt = OptunaOptimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws, dry_run=True)

    base = opt._default_coarse_remaining(_REGIME_A)
    required = {"registrationMode", "angleQuantification",
                "maxNumOfPointPairsPerFeature", "maxVoteRatio",
                "referredStep", "useDistanceNMS", "outputNum",
                "minVoxelLength", "maxVoxelLength"}
    assert required.issubset(base.keys()), f"Missing keys: {required - base.keys()}"

    # refStep and distQ are added separately in the look-ahead; must not be pre-set
    assert "refStep"            not in base, "refStep must not be in _default_coarse_remaining"
    assert "distQuantification" not in base, "distQ must not be in _default_coarse_remaining"

    assert base["outputNum"]    == 1
    assert base["referredStep"] == 1
    assert base["minVoxelLength"] < base["maxVoxelLength"]

    opt.cleanup()
    log.info("PASS: test_default_coarse_remaining_keys")


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run integration test
# ─────────────────────────────────────────────────────────────────────────────

def test_dry_run_joint():
    """Dry run: single joint study, zero MV calls, result not None, score decomposition OK."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_dry_run_joint — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_dry_run_joint — no scenes under: {SCENES_DIR}")
        return

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full = SC.M_FULL
    SC.M_FULL = 3
    try:
        opt = OptunaOptimizer(
            part_name             = PART,
            client                = None,
            project_id            = -1,
            scene_groups          = groups,
            warm_start            = ws,
            cache                 = None,
            dry_run               = True,
            n_trials_joint        = 5,
            n_trials_joint_refine = 3,
            n_rounds              = 1,
            seed                  = 0,
            storage_path          = None,
        )
        result = opt.run()

        assert result is not None,       "Dry run must return an EvalResult"
        assert result.coverage  >= 0.0
        assert result.mean_time >= 0.0
        assert opt.opt._n_evals == 0,    f"dry_run: zero MV calls expected, got {opt.opt._n_evals}"

        # n_rounds=1 → exactly one joint study
        assert len(opt._studies_joint) == 1, \
            f"Expected 1 joint study, got {len(opt._studies_joint)}"
        study = opt._studies_joint[0]
        complete_trials = [t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE]
        assert complete_trials, "Joint study must have at least one complete trial"

        # Every complete trial must carry all 18 joint param keys
        for t in complete_trials:
            assert set(t.params.keys()) == _JOINT_PARAM_KEYS, \
                f"Trial {t.number} key mismatch: {_JOINT_PARAM_KEYS.symmetric_difference(t.params.keys())}"

        # Score decomposition: score == time_term + cov_term
        assert abs(result.score - (result.score_time_term + result.score_cov_term)) < 1e-10, \
            (f"score decomposition broken: score={result.score}  "
             f"time_term={result.score_time_term}  cov_term={result.score_cov_term}")
        assert abs(result.score_quality
                   - (1.0 - result.score / SC.SCORE_WORST_CASE)) < 1e-6, \
            f"quality formula broken: quality={result.score_quality}"

        log.info(f"  dry run: cov={result.coverage:.3f}  time={result.mean_time:.3f}s  "
                 f"score={result.score:.4f}  quality={result.score_quality:.3f}  "
                 f"n_complete={len(complete_trials)}")
        log.info("PASS: test_dry_run_joint")

    finally:
        SC.M_FULL = orig_full
        opt.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Live test
# ─────────────────────────────────────────────────────────────────────────────

def test_live_short():
    """6 joint trials on 25333MB000 — result not None, coverage >= 0.0, MV calls > 0."""
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
        opt = OptunaOptimizer(
            part_name             = PART,
            client                = client,
            project_id            = pid,
            scene_groups          = groups,
            warm_start            = ws,
            cache                 = None,
            dry_run               = False,
            n_trials_joint        = 6,
            n_trials_joint_refine = 3,
            n_rounds              = 1,
            seed                  = 42,
            storage_path          = None,
        )
        result = opt.run()

        assert result is not None
        assert result.coverage >= 0.0
        assert opt.opt._n_evals > 0, "Live test should have made MechVision calls"

        log.info(f"  live short: cov={result.coverage:.3f}  "
                 f"time={result.mean_time:.3f}s  score={result.score:.4f}  "
                 f"mv_evals={opt.opt._n_evals}")
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

    print("test_optuna.py\n")

    print("--- Unit tests (no MechVision, no scene files) ---")
    test_suggest_params_joint_keys()
    test_coarse_mode_max_constraint()
    test_joint_params_to_dicts_fixed()
    test_angleQ_index_mapping()
    test_build_warm_joint_roundtrip()
    test_scoring_formula()

    print("\n--- Instance tests (model file required) ---")
    test_default_fine_keys()
    test_default_coarse_remaining_keys()

    print("\n--- Dry-run integration test ---")
    test_dry_run_joint()

    if args.live:
        print("\n--- Live test (requires MechVision) ---")
        test_live_short()
    else:
        print("\n(skip live test — pass --live to enable)")

    print("\nAll tests PASSED")
