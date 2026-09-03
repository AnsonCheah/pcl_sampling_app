"""
test_mv_evaluator.py -- structural guards on the evaluation harness
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

import pytest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import MM_Optimizer.mv_evaluator as MV

HARNESS_METHODS = [
    "_prepare_scene", "_sample_scenes", "_make_params_dict", "_run_one_scene",
    "evaluate_config", "evaluate_phase_sweep", "_default_coarse", "_default_fine",
    "phase1_regime_gate", "export_best", "_log_result_json",
    "cleanup",
]


def test_harness_methods_on_base():
    """MVEvaluator exposes every shared harness method and module-level name."""
    missing = [m for m in HARNESS_METHODS if not hasattr(MV.MVEvaluator, m)]
    assert not missing, f"MVEvaluator missing harness methods: {missing}"
    for name in ("EvalResult", "match_poses_to_gt",
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


# -----------------------------------------------------------------------------
# Coverage counting -- an instance passes only on position AND orientation
# -----------------------------------------------------------------------------

_IDENTITY_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]   # [x,y,z, qw,qx,qy,qz]
_ROT_90_Z      = [0.0, 0.0, 0.0, 0.70710678, 0.0, 0.0, 0.70710678]


def _score_one(returned, ang_thresh):
    """Drive _run_one_scene against a mocked MechVision, returning instance_coverage."""
    from unittest.mock import MagicMock, patch
    from MM_Optimizer.mesh_analysis import WarmStart

    ws = WarmStart()
    ws.maxNumOfPointPairsPerFeature = 5000
    ws.minVoxelLength_mm, ws.maxVoxelLength_mm, ws.longest_extent_m = 1.0, 15.0, 0.3

    ev = MV.MVEvaluator(part_name="p", client=MagicMock(), project_id=1,
                        scene_groups=[[]], warm_start=ws, cache=None, dry_run=False)
    ev.client.run_vision = MagicMock(return_value={
        "fine_poses": returned, "coarse_time_s": 0.1, "fine_time_s": 0.1})

    with patch.object(ev, "_make_params_dict", return_value={}):
        return ev._run_one_scene({}, {}, "scene", [_IDENTITY_POSE],
                                 pos_thresh=0.005, ang_thresh=ang_thresh)


def test_coverage_rejects_orientation_error():
    """A pose at the right place but 90 degrees off must NOT count.

    The regression guard for the orientation-blind objective: scored position-only, this
    returned full coverage, which is why angleStep had no signal to optimise against.
    """
    r = _score_one([_ROT_90_Z], ang_thresh=5.0)
    assert r["instance_coverage"] == 0.0, \
        f"a 90 deg error passed a 5 deg threshold: {r['ang_errors']}"
    assert r["ang_errors"][0] == pytest.approx(90.0, abs=1e-3)
    log.info("PASS: test_coverage_rejects_orientation_error")


def test_coverage_accepts_within_both_tolerances():
    r = _score_one([_IDENTITY_POSE], ang_thresh=5.0)
    assert r["instance_coverage"] == 1.0
    log.info("PASS: test_coverage_accepts_within_both_tolerances")


def test_position_only_threshold_still_accepts_a_flip():
    """The old behaviour, kept for continuous parts whose roll is unrecoverable."""
    r = _score_one([_ROT_90_Z], ang_thresh=360.0)
    assert r["instance_coverage"] == 1.0
    log.info("PASS: test_position_only_threshold_still_accepts_a_flip")


if __name__ == "__main__":
    test_harness_methods_on_base()
    test_tuner_extends_the_harness()
    test_sampler_choices_have_one_definition()
    test_only_mechvision_params_are_sent()
    test_coverage_rejects_orientation_error()
    test_coverage_accepts_within_both_tolerances()
    test_position_only_threshold_still_accepts_a_flip()
    print("\nAll tests PASSED")
