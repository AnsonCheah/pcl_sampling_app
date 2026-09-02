"""
test_eval_callback.py -- guards the GUI-tuning hooks added to the optimizer
--------------------------------------------------------------------------
Run from project root:
    python MM_Optimizer/tests/test_eval_callback.py
    python -m pytest MM_Optimizer/tests/test_eval_callback.py -q

Covers the small, additive edits the TUNING GUI stage relies on:
  test_on_scene_eval_default_none   -- MVEvaluator.on_scene_eval defaults to None
  test_on_scene_eval_fires          -- the hook fires from _run_one_scene's live branch
                                      with the raw fine_poses + gt_poses
  test_dry_run_skips_hook           -- dry_run early-returns before the hook (documented)
  test_iter_pareto_and_trial_hook   -- (dry-run integration) iter_pareto_configs returns
                                      expandable (coarse, fine); on_trial_complete fires
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

import MM_Optimizer.search_config as SC
from MM_Optimizer.mv_evaluator     import MVEvaluator
from MM_Optimizer.mesh_analysis    import analyze_mesh, load_reference_pcd, WarmStart
from MM_Optimizer.optimizer_utils  import list_synthetic_scenes
from MM_Optimizer.tuner import Tuner

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
MODEL_PATH = os.path.join(_ROOT, "output", "reference_pcd", PART,
                          f"{PART}_surface", f"{PART}_surface.ply")

_ONE_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]   # [x,y,z,qw,qx,qy,qz]


class _StubClient:
    """Minimal MechVisionClient stand-in: canned fine_poses, no gRPC."""
    def __init__(self, poses):
        self.poses = poses

    def set_params(self, project_id, params_dict):
        pass

    def run_vision(self, project_id, timeout=None):
        return {"fine_poses": self.poses, "coarse_poses": [],
                "coarse_time_s": 0.1, "fine_time_s": 0.2}


def _make_evaluator(poses, dry_run=False):
    ev = MVEvaluator(PART, _StubClient(poses), project_id=1,
                     scene_groups=[], warm_start=None, cache=None, dry_run=dry_run)
    # Bypass the model-file param assembly -- the hook is upstream of it.
    ev._make_params_dict = lambda c, f, s: {}
    return ev


def test_on_scene_eval_default_none():
    ev = MVEvaluator(PART, None, 1, scene_groups=[], warm_start=None,
                     cache=None, dry_run=True)
    assert ev.on_scene_eval is None
    log.info("PASS: test_on_scene_eval_default_none")


def test_on_scene_eval_fires():
    ev = _make_evaluator([_ONE_POSE])
    captured = {}
    ev.on_scene_eval = lambda sd, c, f, fp, gt: captured.update(
        scene_dir=sd, coarse=c, fine=f, fine_poses=fp, gt_poses=gt)

    res = ev._run_one_scene({"a": 1}, {"b": 2}, "scene_00000",
                            gt_poses=[_ONE_POSE], pos_thresh=0.01, ang_thresh=10.0)

    assert captured, "on_scene_eval was not called"
    assert captured["scene_dir"] == "scene_00000"
    assert captured["fine_poses"] == [_ONE_POSE]
    assert captured["gt_poses"]   == [_ONE_POSE]
    assert captured["coarse"] == {"a": 1} and captured["fine"] == {"b": 2}
    # The scene still scores normally (perfect match here).
    assert res["instance_coverage"] == 1.0
    log.info("PASS: test_on_scene_eval_fires")


def test_dry_run_skips_hook():
    ev = _make_evaluator([_ONE_POSE], dry_run=True)
    fired = []
    ev.on_scene_eval = lambda *a: fired.append(a)
    ev._run_one_scene({}, {}, "scene_00000", [_ONE_POSE], 0.01, 10.0)
    assert fired == [], "dry_run must early-return before the hook"
    log.info("PASS: test_dry_run_skips_hook")


def test_iter_pareto_and_trial_hook():
    """Dry-run integration: iter_pareto_configs expands (coarse, fine) and the
    on_trial_complete callback fires during run(). Skips if scene/model missing."""
    if not os.path.exists(MODEL_PATH):
        log.warning(f"SKIP test_iter_pareto_and_trial_hook -- model not found: {MODEL_PATH}")
        return
    groups = list_synthetic_scenes(SCENES_DIR) if os.path.isdir(SCENES_DIR) else []
    if not groups:
        log.warning(f"SKIP test_iter_pareto_and_trial_hook -- no scenes: {SCENES_DIR}")
        return

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    orig_full = SC.M_FULL
    SC.M_FULL = 3
    try:
        opt = Tuner(
            part_name=PART, client=None, project_id=-1,
            scene_groups=groups, warm_start=ws, cache=None, dry_run=True,
            n_trials=5, n_rounds=1, seed=0, storage_path=None)

        trial_calls = []
        opt.on_trial_complete = lambda study, trial: trial_calls.append(trial.number)

        result = opt.run()
        assert result is not None

        # on_trial_complete passthrough fired for each completed trial.
        assert len(trial_calls) >= 1, "on_trial_complete never fired"

        # _best_regime stored so the front can be expanded.
        assert opt._best_regime, "_best_regime not stored after run()"

        pareto = opt.iter_pareto_configs()
        assert len(pareto) >= 1, "empty Pareto front"
        trial, coarse, fine = pareto[0]
        assert {"registrationMode", "refStep", "minVoxelLength"}.issubset(coarse)
        assert {"registrationMode", "operationApproach"}.issubset(fine)
        assert len(trial.values) == 2   # (coverage, mean_time)
        log.info(f"  pareto={len(pareto)}  trial_hook_calls={len(trial_calls)}")
        log.info("PASS: test_iter_pareto_and_trial_hook")
    finally:
        SC.M_FULL = orig_full
        opt.cleanup()


if __name__ == "__main__":
    print("test_eval_callback.py -- GUI-tuning hooks\n")
    test_on_scene_eval_default_none()
    test_on_scene_eval_fires()
    test_dry_run_skips_hook()
    test_iter_pareto_and_trial_hook()
    print("\nAll tests PASSED")
