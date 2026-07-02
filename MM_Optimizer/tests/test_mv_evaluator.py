"""
test_mv_evaluator.py — Guards the optimizer.py → mv_evaluator.py extraction boundary
-------------------------------------------------------------------------------------
Run from project root:
    python MM_Optimizer/tests/test_mv_evaluator.py

Pure structural checks (no MechVision, no scene files):
  test_harness_methods_on_base      — MVEvaluator exposes the full eval harness
  test_cd_methods_not_on_base       — coordinate-descent phases did NOT leak into the base
  test_optimizer_is_subclass        — optimizer.Optimizer(MVEvaluator) keeps the CD phases
  test_reexports_are_same_objects   — from optimizer import X resolves to the mv_evaluator X
  test_optuna_does_not_import_optimizer — the Optuna path is detached from optimizer.py
"""

import importlib
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import MM_Optimizer.mv_evaluator as MV

HARNESS_METHODS = [
    "_prepare_scene", "_sample_scenes", "_make_params_dict", "_run_one_scene",
    "evaluate_config", "evaluate_phase_sweep", "_gate", "_default_coarse",
    "_default_fine", "phase1_regime_gate", "phase4_symmetry", "export_best",
    "_log_result_json", "cleanup",
]
CD_METHODS = [
    "phase2_coarse_cd", "_phase2a_joint_grid", "_sweep_voxel_range", "_sweep_param",
    "_sweep_param_direct", "phase3_fine_cd", "phase5_joint_refinement",
    "phase6_interval_narrowing", "run",
]


def test_harness_methods_on_base():
    """MVEvaluator exposes every shared harness method (incl. phase4/evaluate_phase_sweep)."""
    missing = [m for m in HARNESS_METHODS if not hasattr(MV.MVEvaluator, m)]
    assert not missing, f"MVEvaluator missing harness methods: {missing}"
    # Module-level harness surface used by callers/tests.
    for name in ("EvalResult", "match_poses_to_gt", "_confirm_symmetry",
                 "PROJ_NAME", "MM_MODEL_ROOT", "RESULTS_DIR", "ENABLE_CACHE"):
        assert hasattr(MV, name), f"mv_evaluator missing {name}"
    log.info("PASS: test_harness_methods_on_base")


def test_cd_methods_not_on_base():
    """The coordinate-descent phases must NOT live on the shared base."""
    leaked = [m for m in CD_METHODS if hasattr(MV.MVEvaluator, m)]
    assert not leaked, f"CD methods leaked into MVEvaluator: {leaked}"
    log.info("PASS: test_cd_methods_not_on_base")


def test_optimizer_is_subclass():
    """optimizer.Optimizer subclasses MVEvaluator and keeps the CD phases + ENABLE_TWO_PASS."""
    import MM_Optimizer.optimizer as OPT
    assert issubclass(OPT.Optimizer, MV.MVEvaluator)
    missing_cd = [m for m in CD_METHODS if not hasattr(OPT.Optimizer, m)]
    assert not missing_cd, f"Optimizer lost CD methods: {missing_cd}"
    assert hasattr(OPT, "ENABLE_TWO_PASS"), "ENABLE_TWO_PASS must remain in optimizer.py"
    log.info("PASS: test_optimizer_is_subclass")


def test_reexports_are_same_objects():
    """Back-compat: names imported from optimizer are the very objects from mv_evaluator."""
    import MM_Optimizer.optimizer as OPT
    for name in ("EvalResult", "match_poses_to_gt", "_confirm_symmetry",
                 "PROJ_NAME", "MM_MODEL_ROOT", "RESULTS_DIR", "ENABLE_CACHE",
                 "OPTIMIZER_UTILS_PATH"):
        assert getattr(OPT, name) is getattr(MV, name), \
            f"optimizer.{name} is not the same object as mv_evaluator.{name}"
    log.info("PASS: test_reexports_are_same_objects")


def test_optuna_does_not_import_optimizer():
    """Importing optuna_optimizer must NOT pull in the deprecated optimizer.py."""
    # Drop any prior import so this is a faithful cold-import check.
    for mod in ("MM_Optimizer.optimizer", "MM_Optimizer.optuna_optimizer"):
        sys.modules.pop(mod, None)
    importlib.import_module("MM_Optimizer.optuna_optimizer")
    assert "MM_Optimizer.optimizer" not in sys.modules, \
        "optuna_optimizer must not import the deprecated optimizer.py"
    import MM_Optimizer.optuna_optimizer as OO
    assert OO.SAMPLER_CHOICES == ("nsgaii", "tpe", "gp"), OO.SAMPLER_CHOICES
    log.info("PASS: test_optuna_does_not_import_optimizer")


if __name__ == "__main__":
    print("test_mv_evaluator.py — extraction boundary guards\n")
    test_harness_methods_on_base()
    test_cd_methods_not_on_base()
    test_optimizer_is_subclass()
    test_reexports_are_same_objects()
    test_optuna_does_not_import_optimizer()
    print("\nAll tests PASSED")
