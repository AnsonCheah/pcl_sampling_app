"""
test_optuna.py — Tests for staged OptunaOptimizer
--------------------------------------------------
Run from project root:
    python MM_Optimizer/tests/test_optuna.py [--live]

Unit tests (no MechVision, no scene files):
  test_suggest_params_1a               — refStep+distQ only, bounds from SC
  test_suggest_params_1b_surface       — remaining coarse, surface regime
  test_suggest_params_1b_edge          — edge-conditional params present
  test_suggest_fine_params             — fine param dict keys and ranges
  test_default_fine_keys               — _default_fine returns correct warm defaults
  test_default_coarse_remaining_keys   — _default_coarse_remaining returns correct keys
  test_enqueue_1a_grid                 — 30 trials enqueued fast→slow scale order

Dry-run integration test (scene files required, no MechVision):
  test_dry_run_three_stages            — three studies run, zero MV calls, result not None

Live test (requires MechVision with CAD_Match project):
  test_live_short                      — 10 trials, coverage >= 0.0
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
    suggest_params_1a,
    suggest_params_1b,
    suggest_fine_params,
    _build_warm_1b,
    _build_warm_fine,
    _trial_to_coarse,
    _trial_to_fine,
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
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_params_1a():
    """Stage 1a returns only refStep and distQ with bounds from SC constants."""
    study = optuna.create_study()
    trial = study.ask()
    p = suggest_params_1a(trial)

    lo, hi   = SC.OPTUNA_REFSTEP_BOUNDS
    dlo, dhi = SC.OPTUNA_DISTQ_BOUNDS

    assert set(p.keys()) == {"refStep", "distQuantification"}, \
        f"Stage 1a should only have refStep+distQ, got {set(p.keys())}"
    assert lo <= p["refStep"] <= hi,              f"refStep={p['refStep']} out of [{lo},{hi}]"
    assert dlo <= p["distQuantification"] <= dhi, f"distQ={p['distQuantification']} out of [{dlo},{dhi}]"

    log.info("PASS: test_suggest_params_1a")


def test_suggest_params_1b_surface():
    """Stage 1b returns all remaining coarse params for surface regime."""
    study = optuna.create_study()
    trial = study.ask()
    p = suggest_params_1b(trial, _REGIME_A, _PAIRS, _VOXEL_BOUNDS)

    required = {"angleQuantification", "maxNumOfPointPairsPerFeature",
                "maxVoteRatio", "referredStep", "useDistanceNMS",
                "outputNum", "minVoxelLength", "maxVoxelLength"}
    assert required.issubset(p.keys()), f"Missing keys: {required - p.keys()}"

    vlo, vhi = SC.OPTUNA_VOTERATIO_BOUNDS
    rlo, rhi = SC.OPTUNA_REFERRED_BOUNDS
    olo, ohi = SC.OPTUNA_OUTPUTNUM_BOUNDS

    assert p["angleQuantification"]          in SC.OPTUNA_ANGLQ_CHOICES
    assert p["maxNumOfPointPairsPerFeature"] in _PAIRS
    assert vlo <= p["maxVoteRatio"] <= vhi
    assert rlo <= p["referredStep"] <= rhi
    assert isinstance(p["useDistanceNMS"], bool)
    assert olo <= p["outputNum"] <= ohi
    assert p["minVoxelLength"] < p["maxVoxelLength"]

    # Edge params absent for surface regime
    assert "filterCandidatePoseByAxis" not in p

    log.info("PASS: test_suggest_params_1b_surface")


def test_suggest_params_1b_edge():
    """Stage 1b includes edge-conditional params for edge regime."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    got_with_filter = False
    got_without_filter = False

    for _ in range(20):
        trial = study.ask()
        p = suggest_params_1b(trial, _REGIME_C, _PAIRS, _VOXEL_BOUNDS)
        study.tell(trial, 0.5)

        assert "filterCandidatePoseByAxis" in p
        assert isinstance(p["filterCandidatePoseByAxis"], bool)
        if p["filterCandidatePoseByAxis"]:
            at_choices = next(c for n, c, _ in SC.PHASE2B_PARAMS if n == "angleThreshold")
            assert p["angleThreshold"] in at_choices
            got_with_filter = True
        else:
            assert p["angleThreshold"] == 90
            got_without_filter = True

    assert got_with_filter,    "No trial had filterCandidatePoseByAxis=True"
    assert got_without_filter, "No trial had filterCandidatePoseByAxis=False"
    log.info("PASS: test_suggest_params_1b_edge")


def test_suggest_fine_params():
    """suggest_fine_params returns all required fine keys with SC-derived ranges."""
    study = optuna.create_study()
    trial = study.ask()
    p = suggest_fine_params(trial, _REGIME_A)

    assert p["registrationMode"]            == _REGIME_A["fine_mode"]
    assert p["operationApproach"]           in SC.OPTUNA_OPAPP_CHOICES
    assert p["deviationCorrectionCapacity"] in SC.OPTUNA_DEVCAP_CHOICES
    assert isinstance(p["onlyConsiderVisibleSurfaceOfModel"], bool)
    assert isinstance(p["considerErrorofNormalAngles"], bool)
    assert p["scoreLevel"]                  in SC.OPTUNA_SCORELV_CHOICES
    clo, chi = SC.OPTUNA_CONFTHRESH_BOUNDS
    assert clo <= p["confidenceThreshold"] <= chi
    assert p["candidateTopNum"]             == 1

    log.info("PASS: test_suggest_fine_params")


def test_default_fine_keys():
    """_default_fine returns all 8 fine keys at warm-start defaults."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_default_fine_keys — model not found: {MODEL_PATH}")
        return
    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)
    groups = list_synthetic_scenes(SCENES_DIR) if os.path.isdir(SCENES_DIR) else [[]]

    opt = OptunaOptimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws, dry_run=True)

    fine = opt._default_fine(_REGIME_A)
    required = {"registrationMode", "operationApproach", "deviationCorrectionCapacity",
                "onlyConsiderVisibleSurfaceOfModel", "considerErrorofNormalAngles",
                "scoreLevel", "confidenceThreshold", "candidateTopNum"}
    assert required == set(fine.keys()), f"Key mismatch: {required.symmetric_difference(fine.keys())}"

    # Verify warm defaults
    assert fine["operationApproach"]           == 1.0
    assert fine["deviationCorrectionCapacity"] == 0.0
    assert fine["onlyConsiderVisibleSurfaceOfModel"] is False
    assert fine["considerErrorofNormalAngles"] is False
    assert fine["scoreLevel"]                  == 0.0
    assert fine["confidenceThreshold"]         == 0.0
    assert fine["candidateTopNum"]             == 1

    opt.cleanup()
    log.info("PASS: test_default_fine_keys")


def test_default_coarse_remaining_keys():
    """_default_coarse_remaining returns all required coarse keys (no refStep/distQ)."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_default_coarse_remaining_keys — model not found: {MODEL_PATH}")
        return
    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)
    groups = list_synthetic_scenes(SCENES_DIR) if os.path.isdir(SCENES_DIR) else [[]]

    opt = OptunaOptimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws, dry_run=True)

    base = opt._default_coarse_remaining(_REGIME_A)
    required = {"registrationMode", "angleQuantification",
                "maxNumOfPointPairsPerFeature", "maxVoteRatio",
                "referredStep", "useDistanceNMS", "outputNum",
                "minVoxelLength", "maxVoxelLength"}
    assert required.issubset(base.keys()), f"Missing: {required - base.keys()}"

    # refStep and distQ must NOT be present (those come from Stage 1a)
    assert "refStep" not in base
    assert "distQuantification" not in base

    assert base["outputNum"]    == 1
    assert base["referredStep"] == 1
    assert base["minVoxelLength"] < base["maxVoxelLength"]

    opt.cleanup()
    log.info("PASS: test_default_coarse_remaining_keys")


def test_enqueue_1a_grid():
    """_enqueue_1a_grid creates 5×6=30 trials in fast→slow scale order."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_enqueue_1a_grid — model not found: {MODEL_PATH}")
        return
    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)
    groups = list_synthetic_scenes(SCENES_DIR) if os.path.isdir(SCENES_DIR) else [[]]

    opt = OptunaOptimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws, dry_run=True, seed=0)

    study = optuna.create_study()
    opt._enqueue_1a_grid(study)

    n_expected = len(SC.PHASE2A_REFSTEP_SCALES) * len(SC.PHASE2A_DISTQ_VALUES)
    assert len(study.trials) == n_expected, \
        f"Expected {n_expected} enqueued trials, got {len(study.trials)}"

    # Enqueued trials are WAITING — params live in system_attrs["fixed_params"]
    lo, hi = opt._refstep_bounds   # dynamic bounds, not static SC constant
    first_fixed = study.trials[0].system_attrs["fixed_params"]
    first_ref   = first_fixed["refStep"]
    largest_scale = SC.PHASE2A_REFSTEP_SCALES[0]   # 2.0 after reorder
    expected_first_ref = max(lo, min(hi, int(ws.refStep * largest_scale)))
    assert first_ref == expected_first_ref, \
        f"First grid trial refStep={first_ref}, expected {expected_first_ref} (scale×{largest_scale})"

    opt.cleanup()
    log.info(f"PASS: test_enqueue_1a_grid ({n_expected} trials, first refStep={first_ref})")


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run integration test
# ─────────────────────────────────────────────────────────────────────────────

def test_dry_run_three_stages():
    """Three Optuna studies in dry_run — zero MV calls, all three studies on instance."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_dry_run_three_stages — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_dry_run_three_stages — no scenes under: {SCENES_DIR}")
        return

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 3
    SC.M_SMALL = 2
    try:
        opt = OptunaOptimizer(
            part_name    = PART,
            client       = None,
            project_id   = -1,
            scene_groups = groups,
            warm_start   = ws,
            cache        = None,
            dry_run      = True,
            n_trials_1a  = 4,   # grid (30) capped at 4 for speed; remaining skipped
            n_trials_1b  = 3,
            n_trials_2   = 3,
            seed         = 0,
            storage_path = None,
        )
        result = opt.run()

        assert result is not None,        "Dry run must return an EvalResult"
        assert result.coverage  >= 0.0
        assert result.mean_time >= 0.0
        assert opt.opt._n_evals == 0,     f"dry_run: zero MV calls expected, got {opt.opt._n_evals}"

        # All three studies must be populated
        assert opt._study_1a is not None, "_study_1a not set after run()"
        assert opt._study_1b is not None, "_study_1b not set after run()"
        assert opt._study_2  is not None, "_study_2  not set after run()"

        # Each study must have at least one complete trial
        def n_complete(s): return sum(
            1 for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE)
        assert n_complete(opt._study_1a) >= 1, "Stage 1a has no complete trials"
        assert n_complete(opt._study_1b) >= 1, "Stage 1b has no complete trials"
        assert n_complete(opt._study_2)  >= 1, "Stage 2  has no complete trials"

        # Stage 1a complete trials must only have refStep and distQ params
        for t in opt._study_1a.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                assert set(t.params.keys()) == {"refStep", "distQ"}, \
                    f"Stage 1a trial has unexpected params: {set(t.params.keys())}"

        log.info(f"  dry run: cov={result.coverage:.2f}  time={result.mean_time:.3f}s")
        log.info("PASS: test_dry_run_three_stages")

    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
        opt.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Live test
# ─────────────────────────────────────────────────────────────────────────────

def test_live_short():
    """10 trials on 25333MB000 — result not None, coverage >= 0.0."""
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
            part_name    = PART,
            client       = client,
            project_id   = pid,
            scene_groups = groups,
            warm_start   = ws,
            cache        = None,
            dry_run      = False,
            n_trials_1a  = 5,
            n_trials_1b  = 3,
            n_trials_2   = 2,
            seed         = 42,
            storage_path = None,
        )
        result = opt.run()

        assert result is not None
        assert result.coverage >= 0.0
        assert opt.opt._n_evals > 0, "Live test should have made MechVision calls"

        log.info(f"  live short: cov={result.coverage:.3f}  "
                 f"time={result.mean_time:.3f}s  mv_evals={opt.opt._n_evals}")
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

    print("test_optuna.py\n")

    print("--- Unit tests ---")
    test_suggest_params_1a()
    test_suggest_params_1b_surface()
    test_suggest_params_1b_edge()
    test_suggest_fine_params()
    test_default_fine_keys()
    test_default_coarse_remaining_keys()
    test_enqueue_1a_grid()

    print("\n--- Dry-run integration test ---")
    test_dry_run_three_stages()

    if args.live:
        print("\n--- Live test (requires MechVision) ---")
        test_live_short()
    else:
        print("\n(skip live test — pass --live to enable)")

    print("\nAll tests PASSED")
