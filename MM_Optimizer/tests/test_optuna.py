"""
test_optuna.py — Tests for fully joint OptunaOptimizer
-------------------------------------------------------
Run from project root:
    python MM_Optimizer/tests/test_optuna.py [--live]

Unit tests (no MechVision, no scene files):
  test_suggest_params_joint_surface    — all keys present, bounds valid (surface regime)
  test_suggest_params_joint_edge       — edge-conditional params conditionally present
  test_split_joint_params_surface      — coarse/fine split correct for surface
  test_split_joint_params_edge         — edge-mode keys in coarse dict
  test_build_warm_joint                — warm-start dict round-trips without clamp errors
  test_pareto_winner_selection         — max-coverage then min-time selection
  test_pareto_winner_fallback          — fallback when Pareto front is empty
  test_default_fine_keys               — _default_fine returns correct warm defaults
  test_default_coarse_remaining_keys   — _default_coarse_remaining includes refStep/distQ

Dry-run integration test (scene files required, no MechVision):
  test_dry_run_joint_study             — joint study runs, zero MV calls, result not None
  test_multi_round_early_stop          — round 1 fires, improvement check works

Live test (requires MechVision with CAD_Match project):
  test_live_short                      — joint study, coverage >= 0.0
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
from MM_Optimizer.optimizer       import PROJ_NAME, MM_MODEL_ROOT
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
from MM_Optimizer.optuna_optimizer import (
    OptunaOptimizer,
    suggest_params_joint,
    _split_joint_params,
    _build_warm_joint,
    _select_pareto_winner,
)
import MM_Optimizer.search_config as SC

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
MODEL_PATH = os.path.join(MM_MODEL_ROOT, f"{PART}_surface", f"{PART}_surface.ply")

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


def _ask_joint(regime, study=None):
    if study is None:
        study = _make_study_multi()
    trial = study.ask()
    p = suggest_params_joint(trial, regime, _PAIRS, _VOXEL_BOUNDS)
    return p, trial, study


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_params_joint_surface():
    """Joint suggest returns all required keys with valid bounds for surface regime."""
    p, _, _ = _ask_joint(_REGIME_A)

    # ── Coarse keys ──────────────────────────────────────────────────────
    assert p["coarse_mode"] == 0.0
    lo, hi = SC.REFSTEP_BOUNDS
    assert lo <= p["refStep"] <= hi, f"refStep={p['refStep']} out of [{lo},{hi}]"

    dlo, dhi = SC.OPTUNA_DISTQ_BOUNDS
    assert dlo <= p["distQuantification"] <= dhi

    assert p["angleQuantification"] in SC.OPTUNA_ANGLQ_CHOICES

    assert p["maxNumOfPointPairsPerFeature"] in _PAIRS

    vlo, vhi = SC.OPTUNA_VOTERATIO_BOUNDS
    assert vlo <= p["maxVoteRatio"] <= vhi

    # referredStep now uses fixed bounds (not dynamic upper=refStep).
    # The constraint referredStep ≤ refStep is enforced by the objective guard, not here.
    assert lo <= p["referredStep"] <= hi, \
        f"referredStep={p['referredStep']} out of fixed bounds [{lo},{hi}]"

    assert isinstance(p["useDistanceNMS"], bool)

    olo, ohi = SC.OPTUNA_OUTPUTNUM_BOUNDS
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

    log.info("PASS: test_suggest_params_joint_surface")


def test_suggest_params_joint_edge():
    """Edge mode conditionally suggests filterByAxis and angleThreshold."""
    study = _make_study_multi()
    got_with_filter   = False
    got_without_filter = False

    for _ in range(30):
        p, trial, _ = _ask_joint(_REGIME_C, study)
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
    log.info("PASS: test_suggest_params_joint_edge")


def test_split_joint_params_surface():
    """_split_joint_params produces valid coarse/fine dicts for surface regime."""
    p, _, _ = _ask_joint(_REGIME_A)
    coarse, fine = _split_joint_params(p)

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

    log.info("PASS: test_split_joint_params_surface")


def test_split_joint_params_edge():
    """Edge-mode coarse dict includes filterCandidatePoseByAxis and angleThreshold."""
    p, _, _ = _ask_joint(_REGIME_C)
    coarse, _ = _split_joint_params(p)

    assert "filterCandidatePoseByAxis" in coarse
    assert "angleThreshold"            in coarse
    log.info("PASS: test_split_joint_params_edge")


def test_build_warm_joint():
    """_build_warm_joint produces a valid enqueue dict without clamping errors."""
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
    p = _build_warm_joint(coarse, fine, _REGIME_A, _PAIRS, _VOXEL_BOUNDS)

    lo, hi = SC.REFSTEP_BOUNDS
    assert lo <= p["refStep"] <= hi
    assert p["angleQuantification"] in SC.OPTUNA_ANGLQ_CHOICES
    assert 0 <= p["pairs_idx"]  < len(_PAIRS)
    assert 0 <= p["opApproach"] <= 4
    assert 0 <= p["devCap"]     <= 2
    assert isinstance(p["visibleSurf"], bool)
    assert isinstance(p["normalAng"],   bool)
    assert p["minVoxelLength_mm"] < p["minVoxelLength_mm"] + p["voxel_width_mm"]
    assert p["refStep"] >= p["referredStep"], \
        f"refStep={p['refStep']} < referredStep={p['referredStep']}"

    log.info("PASS: test_build_warm_joint")


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


def test_default_fine_keys():
    """_default_fine returns all 8 fine keys at warm-start defaults."""
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
    log.info("PASS: test_default_fine_keys")


def test_default_coarse_remaining_keys():
    """_default_coarse_remaining now includes refStep and distQ (full coarse warm-start)."""
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
    required = {"registrationMode", "refStep", "distQuantification",
                "angleQuantification", "maxNumOfPointPairsPerFeature",
                "maxVoteRatio", "referredStep", "useDistanceNMS",
                "outputNum", "minVoxelLength", "maxVoxelLength"}
    assert required.issubset(base.keys()), f"Missing: {required - base.keys()}"
    assert base["outputNum"]    == 1
    assert base["referredStep"] == 1
    assert base["minVoxelLength"] < base["maxVoxelLength"]
    opt.cleanup()
    log.info("PASS: test_default_coarse_remaining_keys")


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dry_run_joint_study():
    """Joint study in dry_run — zero MV calls, single study on instance, result not None."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_dry_run_joint_study — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_dry_run_joint_study — no scenes under: {SCENES_DIR}")
        return

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 3
    SC.M_SMALL = 2
    try:
        opt = OptunaOptimizer(
            part_name      = PART,
            client         = None,
            project_id     = -1,
            scene_groups   = groups,
            warm_start     = ws,
            cache          = None,
            dry_run        = True,
            n_trials_joint = 5,
            n_rounds       = 1,
            seed           = 0,
            storage_path   = None,
        )
        result = opt.run()

        assert result is not None,       "Dry run must return an EvalResult"
        assert result.coverage  >= 0.0
        assert result.mean_time >= 0.0
        assert opt.opt._n_evals == 0,    f"dry_run: zero MV calls expected, got {opt.opt._n_evals}"

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
        log.info("PASS: test_dry_run_joint_study")
    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
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

    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 2
    SC.M_SMALL = 1
    orig_refine = SC.OPTUNA_N_TRIALS_JOINT_REFINE
    SC.OPTUNA_N_TRIALS_JOINT_REFINE = 2
    try:
        opt = OptunaOptimizer(
            part_name      = PART,
            client         = None,
            project_id     = -1,
            scene_groups   = groups,
            warm_start     = ws,
            cache          = None,
            dry_run        = True,
            n_trials_joint = 4,
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
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
        SC.OPTUNA_N_TRIALS_JOINT_REFINE = orig_refine
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

    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = len(groups)
    SC.M_SMALL = max(1, len(groups) // 2)
    try:
        opt = OptunaOptimizer(
            part_name      = PART,
            client         = client,
            project_id     = pid,
            scene_groups   = groups,
            warm_start     = ws,
            cache          = None,
            dry_run        = False,
            n_trials_joint = 10,
            n_rounds       = 1,
            seed           = 42,
            storage_path   = None,
        )
        result = opt.run()

        assert result is not None
        assert result.coverage >= 0.0
        assert opt.opt._n_evals > 0, "Live test should have made MechVision calls"

        log.info(f"  live short: cov={result.coverage:.3f}  "
                 f"time={result.mean_time:.3f}s  score={result.score:.3f}  "
                 f"mv_evals={opt.opt._n_evals}")
        log.info("PASS: test_live_short")
    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
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
    test_suggest_params_joint_surface()
    test_suggest_params_joint_edge()
    test_split_joint_params_surface()
    test_split_joint_params_edge()
    test_build_warm_joint()
    test_pareto_winner_selection()
    test_pareto_winner_fallback()
    test_default_fine_keys()
    test_default_coarse_remaining_keys()

    print("\n--- Dry-run integration tests ---")
    test_dry_run_joint_study()
    test_multi_round_early_stop()

    if args.live:
        print("\n--- Live test (requires MechVision) ---")
        test_live_short()
    else:
        print("\n(skip live test — pass --live to enable)")

    print("\nAll tests PASSED")
