"""
test_mv_evaluator.py — structural guards on the evaluation harness
-------------------------------------------------------------------
Run from project root:
    python -m pytest MM_Optimizer/tests/test_mv_evaluator.py -q

Pure structural checks (no MechVision, no scene files).
"""

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
    "evaluate_config", "evaluate_phase_sweep", "_default_coarse", "_default_fine",
    "phase1_regime_gate", "phase4_symmetry", "export_best", "_log_result_json",
    "cleanup",
]


def test_harness_methods_on_base():
    """MVEvaluator exposes every shared harness method and module-level name."""
    missing = [m for m in HARNESS_METHODS if not hasattr(MV.MVEvaluator, m)]
    assert not missing, f"MVEvaluator missing harness methods: {missing}"
    for name in ("EvalResult", "match_poses_to_gt", "_confirm_symmetry",
                 "PROJ_NAME", "MM_MODEL_ROOT", "RESULTS_DIR", "ENABLE_CACHE"):
        assert hasattr(MV, name), f"mv_evaluator missing {name}"
    log.info("PASS: test_harness_methods_on_base")


def test_tuner_extends_the_harness():
    """Tuner IS an MVEvaluator, so the harness is reached by inheritance, not a proxy."""
    import MM_Optimizer.tuner as T
    assert issubclass(T.Tuner, MV.MVEvaluator)
    assert not hasattr(T.Tuner, "opt"), "the wrapped-evaluator proxy should be gone"
    log.info("PASS: test_tuner_extends_the_harness")


def test_sampler_choices_have_one_definition():
    """The GUI reads the tuner's list rather than keeping its own copy."""
    import MM_Optimizer.tuner as T
    from stages.tuning_stage import SAMPLER_CHOICES, SAMPLER_DEFAULT
    assert T.SAMPLER_CHOICES == ("nsgaii", "tpe", "gp"), T.SAMPLER_CHOICES
    assert SAMPLER_CHOICES is T.SAMPLER_CHOICES
    assert SAMPLER_DEFAULT is T.SAMPLER_DEFAULT
    assert T.SAMPLER_DEFAULT in T.SAMPLER_CHOICES
    log.info("PASS: test_sampler_choices_have_one_definition")


def test_only_mechvision_params_are_sent():
    """The type tables gate what reaches MechVision; search-space bookkeeping is skipped."""
    assert "candidateTopNum" in MV._FINE_TYPES, "fine matching must set candidateTopNum"
    assert "minVoxelLength_mm" in MV._NON_MV_KEYS
    assert not (set(MV._COARSE_TYPES) | set(MV._FINE_TYPES)) & MV._NON_MV_KEYS
    log.info("PASS: test_only_mechvision_params_are_sent")


if __name__ == "__main__":
    test_harness_methods_on_base()
    test_tuner_extends_the_harness()
    test_sampler_choices_have_one_definition()
    test_only_mechvision_params_are_sent()
    print("\nAll tests PASSED")
