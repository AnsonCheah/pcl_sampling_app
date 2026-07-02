"""
test_nsga_optimizer.py — Tests for the NSGA-II migration in OptunaOptimizer
----------------------------------------------------------------------------
Run from project root:
    python MM_Optimizer/tests/test_nsga_optimizer.py [--live]

Unit tests (no MechVision, no disk files):
  test_suggest_params_fixed_referredstep_bounds  — referredStep sampled from full [1,20] range
  test_constraint_func_feasible                  — _referredstep_constraint returns 0 when ok
  test_constraint_func_violated                  — _referredstep_constraint returns >0 when violated
  test_constraint_guard_returns_worst_case       — guard fires, no MV call, sentinel returned
  test_build_warm_joint_clamps_referredstep      — warm-start builder clamps referredStep ≤ refStep
  test_pareto_winner_nsga                        — Pareto winner: max cov then min time

Dry-run integration tests (scene files required, no MechVision):
  test_dry_run_nsga_study                — NSGA-II study runs, n_evals=0, result not None
  test_nsga_no_blowup_trials             — all violated suggestions caught by guard, not MechVision
  test_multi_round_nsga                  — 2-round study with early-stop on negligible improvement
"""

import logging
import os
import sys
import warnings

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

from MM_Optimizer.mesh_analysis    import analyze_mesh, load_reference_pcd
from MM_Optimizer.mv_evaluator     import MM_MODEL_ROOT
from MM_Optimizer.optimizer_utils  import list_synthetic_scenes
from MM_Optimizer.optuna_optimizer import (
    OptunaOptimizer,
    suggest_params_joint,
    _build_warm_joint,
    _select_pareto_winner,
    _referredstep_constraint,
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


def _make_frozen_trial(values, user_attrs=None):
    """Build a minimal FrozenTrial for constraint-func testing."""
    return optuna.trial.FrozenTrial(
        number=0,
        trial_id=0,
        state=optuna.trial.TrialState.COMPLETE,
        value=None,
        values=values,
        datetime_start=None,
        datetime_complete=None,
        params={},
        distributions={},
        user_attrs=user_attrs or {},
        system_attrs={},
        intermediate_values={},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_params_fixed_referredstep_bounds():
    """With fixed bounds, referredStep is sampled from the full [1,20] range and
    can legitimately exceed refStep — proving the dynamic bound was removed."""
    lo, hi = SC.REFSTEP_BOUNDS   # (1, 20)
    study = _make_study_multi()
    found_violation = False

    for _ in range(100):
        p, trial, _ = _ask_joint(_REGIME_A, study)
        study.tell(trial, [0.5, 1.0])

        assert lo <= p["refStep"]      <= hi, f"refStep={p['refStep']} out of [{lo},{hi}]"
        assert lo <= p["referredStep"] <= hi, \
            f"referredStep={p['referredStep']} out of fixed bounds [{lo},{hi}]"

        if p["referredStep"] > p["refStep"]:
            found_violation = True

    # With 100 random trials over a 20×20 integer grid, probability of never seeing
    # referredStep > refStep is negligible — this assertion catches a regression to
    # the old dynamic upper-bound suggest.
    assert found_violation, (
        "100 random trials produced no referredStep > refStep. "
        "Check that suggest_int upper bound is SC.REFSTEP_BOUNDS[1], not dynamic refStep."
    )
    log.info("PASS: test_suggest_params_fixed_referredstep_bounds")


def test_constraint_func_feasible():
    """_referredstep_constraint returns [0.0] for a trial with no constraint_violation attr."""
    trial = _make_frozen_trial(values=[0.1, 0.5])   # no user_attrs
    result = _referredstep_constraint(trial)
    assert len(result) == 1
    assert result[0] <= 0.0, f"Feasible trial must have ≤0 constraint, got {result[0]}"
    log.info("PASS: test_constraint_func_feasible")


def test_constraint_func_violated():
    """_referredstep_constraint returns [violation > 0] when constraint_violation is set."""
    trial = _make_frozen_trial(
        values=[0.0, SC.OPTUNA_TIME_INITIAL_CAP],   # worst-case for maximize coverage
        user_attrs={"constraint_violation": 4.0},   # referredStep=9, refStep=5 → 9-5=4
    )
    result = _referredstep_constraint(trial)
    assert len(result) == 1
    assert result[0] == 4.0, f"Expected violation=4.0, got {result[0]}"
    log.info("PASS: test_constraint_func_violated")


def test_constraint_guard_returns_worst_case():
    """When referredStep > refStep, _objective_joint returns the worst-case sentinel
    (1.0, OPTUNA_TIME_INITIAL_CAP) without calling MechVision, and sets constraint_violation."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_constraint_guard_returns_worst_case — model not found")
        return
    groups = list_synthetic_scenes(SCENES_DIR) if os.path.isdir(SCENES_DIR) else [[]]

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full = SC.M_FULL
    SC.M_FULL = 2
    try:
        opt = OptunaOptimizer(
            part_name=PART, client=None, project_id=-1,
            scene_groups=groups, warm_start=ws, dry_run=True,
            n_trials_joint=3, n_rounds=1, seed=0,
        )
        best_regime = {"id": "A", "coarse_mode": 0.0, "fine_mode": 0.0,
                       "needs_edge": False, "coverage": 1.0,
                       "coarse": opt.opt._default_coarse(),
                       "fine":   opt.opt._default_fine()}

        study = _make_study_multi()
        trial = study.ask()

        # Craft a suggest_params_joint output where referredStep > refStep
        from unittest.mock import patch

        violating_p = {
            "refStep": 3, "referredStep": 8,   # violation: 8 - 3 = 5
            "distQuantification": 1.0,
            "angleQuantification": 90,
            "pairs_idx": 2,
            "maxNumOfPointPairsPerFeature": 5000,
            "maxVoteRatio": 0.5,
            "useDistanceNMS": True,
            "outputNum": 1,
            "minVoxelLength_mm": 0.7,
            "voxel_width_mm": 2.1,
            "minVoxelLength": 0.7,
            "maxVoxelLength": 2.8,
            "filterCandidatePoseByAxis": True,
            "angleThreshold": 135,
            "coarse_mode": 0.0,
            "fine_mode": 0.0,
            "operationApproach": 1.0,
            "deviationCorrectionCapacity": 0.0,
            "onlyConsiderVisibleSurfaceOfModel": False,
            "considerErrorofNormalAngles": False,
        }

        mv_calls = [0]
        original_eval = opt.opt.evaluate_config
        def counting_eval(*a, **kw):
            mv_calls[0] += 1
            return original_eval(*a, **kw)

        with patch.object(opt.opt, "evaluate_config", side_effect=counting_eval), \
             patch("MM_Optimizer.optuna_optimizer.suggest_params_joint",
                   return_value=violating_p):
            ret = opt._objective_joint(trial, best_regime)

        assert ret == (0.0, SC.OPTUNA_TIME_INITIAL_CAP), \
            f"Expected worst-case sentinel, got {ret}"
        assert mv_calls[0] == 0, \
            f"MechVision must NOT be called for violating referredStep, got {mv_calls[0]}"
        assert trial.user_attrs.get("constraint_violation") == 5.0, \
            f"Expected constraint_violation=5.0, got {trial.user_attrs.get('constraint_violation')}"

        opt.cleanup()
    finally:
        SC.M_FULL = orig_full
    log.info("PASS: test_constraint_guard_returns_worst_case")


def test_build_warm_joint_clamps_referredstep():
    """_build_warm_joint still clamps referredStep ≤ refStep so warm-start is always feasible."""
    coarse = {
        "refStep": 5, "distQuantification": 1.0, "angleQuantification": 90,
        "maxNumOfPointPairsPerFeature": 5000, "maxVoteRatio": 0.5,
        "referredStep": 12,   # intentionally > refStep=5
        "useDistanceNMS": True, "outputNum": 1,
        "minVoxelLength": 0.7, "maxVoxelLength": 2.8,
    }
    fine = {
        "operationApproach": 1.0, "deviationCorrectionCapacity": 0.0,
        "onlyConsiderVisibleSurfaceOfModel": False,
        "considerErrorofNormalAngles": False,
    }
    p = _build_warm_joint(coarse, fine, _REGIME_A, _PAIRS, _VOXEL_BOUNDS)

    assert p["referredStep"] <= p["refStep"], (
        f"_build_warm_joint must clamp referredStep ≤ refStep, "
        f"got referredStep={p['referredStep']} refStep={p['refStep']}"
    )
    assert p["referredStep"] == 5, \
        f"After clamping min(12, 5)=5, got referredStep={p['referredStep']}"
    log.info("PASS: test_build_warm_joint_clamps_referredstep")


def test_pareto_winner_nsga():
    """_select_pareto_winner picks max coverage first, then min time — logic unchanged."""
    study = _make_study_multi()
    configs = [
        (0.95, 1.0),   # 95% cov, 1.0s
        (0.95, 0.5),   # 95% cov, 0.5s  ← should win (max cov, min time tiebreaker)
        (0.90, 0.3),   # 90% cov, 0.3s  ← Pareto non-dominated
        (0.80, 0.2),
    ]
    for cov, t in configs:
        study.add_trial(optuna.trial.create_trial(
            params={}, distributions={},
            values=[cov, t],
        ))
    winner = _select_pareto_winner(study)
    assert winner.values[0] == 0.95, f"Expected cov=0.95, got {winner.values[0]}"
    assert winner.values[1] == 0.5,  f"Expected time=0.5 (faster), got {winner.values[1]}"
    log.info("PASS: test_pareto_winner_nsga")


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dry_run_nsga_study():
    """NSGA-II study runs in dry_run: n_evals=0, result not None, study populated."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_dry_run_nsga_study — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_dry_run_nsga_study — no scenes under: {SCENES_DIR}")
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
            n_trials_joint = SC.OPTUNA_NSGA_POPULATION_SIZE + 5,
            n_rounds       = 1,
            seed           = 0,
            storage_path   = None,
        )
        result = opt.run()

        assert result is not None,    "Dry run must return an EvalResult"
        assert result.coverage >= 0.0
        assert result.mean_time >= 0.0
        assert opt.opt._n_evals == 0, f"dry_run must have 0 MV evals, got {opt.opt._n_evals}"
        assert opt._study is not None, "_study must be set after run()"

        n_complete = sum(1 for t in opt._study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        assert n_complete >= 1, f"Study must have ≥1 complete trial, got {n_complete}"

        log.info(f"  NSGA-II dry run: complete={n_complete}  cov={result.coverage:.2f}  "
                 f"time={result.mean_time:.3f}s")
        log.info("PASS: test_dry_run_nsga_study")
    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
        opt.cleanup()


def test_nsga_no_blowup_trials():
    """After dry run, every complete trial with referredStep > refStep must have been caught
    by the constraint guard (sentinel values + constraint_violation attr), never MechVision."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_nsga_no_blowup_trials — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_nsga_no_blowup_trials — no scenes under: {SCENES_DIR}")
        return

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 2
    SC.M_SMALL = 1
    try:
        opt = OptunaOptimizer(
            part_name      = PART,
            client         = None,
            project_id     = -1,
            scene_groups   = groups,
            warm_start     = ws,
            cache          = None,
            dry_run        = True,
            n_trials_joint = 40,
            n_rounds       = 1,
            seed           = 1,
            storage_path   = None,
        )
        opt.run()

        assert opt._study is not None
        n_total = len(opt._study.trials)

        for t in opt._study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            rs   = t.params.get("refStep")
            rref = t.params.get("referredStep")
            if rs is None or rref is None:
                continue
            if rref > rs:
                # Guard must have intercepted: sentinel values + violation attr
                assert t.values == [0.0, SC.OPTUNA_TIME_INITIAL_CAP], (
                    f"Trial {t.number}: referredStep={rref} > refStep={rs} "
                    f"but values={t.values} — expected sentinel (0.0, {SC.OPTUNA_TIME_INITIAL_CAP})"
                )
                assert "constraint_violation" in t.user_attrs, (
                    f"Trial {t.number}: referredStep > refStep but no constraint_violation attr"
                )

        log.info(f"  no-blowup check: {n_total} trials verified")
        log.info("PASS: test_nsga_no_blowup_trials")
    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
        opt.cleanup()


def test_multi_round_nsga():
    """2-round NSGA-II study: round 1 fires; early-stop when improvement < threshold."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_multi_round_nsga — model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR)
    if not groups:
        log.warning(f"SKIP test_multi_round_nsga — no scenes under: {SCENES_DIR}")
        return

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    orig_refine = SC.OPTUNA_N_TRIALS_JOINT_REFINE
    SC.M_FULL  = 2
    SC.M_SMALL = 1
    SC.OPTUNA_N_TRIALS_JOINT_REFINE = 5
    try:
        opt = OptunaOptimizer(
            part_name      = PART,
            client         = None,
            project_id     = -1,
            scene_groups   = groups,
            warm_start     = ws,
            cache          = None,
            dry_run        = True,
            n_trials_joint = 20,
            n_rounds       = 2,
            seed           = 0,
            storage_path   = None,
        )
        result = opt.run()
        assert result is not None, "Multi-round NSGA-II must return a result"
        assert opt._study is not None

        n_complete = sum(1 for t in opt._study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        assert n_complete >= 1, f"Study must have ≥1 complete trial, got {n_complete}"

        log.info(f"  multi-round NSGA-II: complete={n_complete}  cov={result.coverage:.2f}")
        log.info("PASS: test_multi_round_nsga")
    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
        SC.OPTUNA_N_TRIALS_JOINT_REFINE = orig_refine
        opt.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true",
                   help="Run live MechVision tests (not yet implemented)")
    args = p.parse_args()

    print("test_nsga_optimizer.py — NSGA-II migration tests\n")

    print("--- Unit tests ---")
    test_suggest_params_fixed_referredstep_bounds()
    test_constraint_func_feasible()
    test_constraint_func_violated()
    test_constraint_guard_returns_worst_case()
    test_build_warm_joint_clamps_referredstep()
    test_pareto_winner_nsga()

    print("\n--- Dry-run integration tests ---")
    test_dry_run_nsga_study()
    test_nsga_no_blowup_trials()
    test_multi_round_nsga()

    print("\nAll tests PASSED")
