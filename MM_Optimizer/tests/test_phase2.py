"""
test_phase2.py  —  Live smoke test: Phase 2 coarse coordinate descent
Requires live MechVision (CAD_Match project loaded).
Run from project root:
    python MM_Optimizer/tests/test_phase2.py
"""

import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mm_adapter.mm_adapter import MechVisionClient
from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
from MM_Optimizer.optimizer       import Optimizer, PROJ_NAME, MM_MODEL_ROOT
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
import MM_Optimizer.search_config  as SC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
MODEL_PATH = os.path.join(MM_MODEL_ROOT, f"{PART}_surface", f"{PART}_surface.ply")


def test_dry_run_phase2():
    """Dry run: Phase 2 should build all candidate grids and return without crashing."""
    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)

    orig_m_full, orig_m_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 3
    SC.M_SMALL = 2
    try:
        opt = Optimizer(
            part_name    = PART,
            client       = None,
            project_id   = -1,
            scene_groups = groups,
            warm_start   = ws,
            cache        = None,
            use_two_pass = False,
            dry_run      = True,
        )

        regime = {
            "id": "A", "coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False,
            "coarse": opt._default_coarse(),
            "fine":   opt._default_fine(),
        }

        result = opt.phase2_coarse_cd(regime)
        assert result is not None, "Phase 2 returned None"
        assert result.coverage == 1.0, \
            f"Dry run should give 1.0 coverage (got {result.coverage})"

        n_grid = len(SC.PHASE2A_REFSTEP_SCALES) * len(SC.PHASE2A_DISTQ_VALUES)
        log.info(f"  2a grid size = {n_grid}  evals={opt._n_evals}")
        log.info("PASS: dry_run_phase2")
    finally:
        SC.M_FULL  = orig_m_full
        SC.M_SMALL = orig_m_small
        opt.cleanup()


def test_phase2a_joint_grid():
    """Live: Phase 2a joint grid should find a config with better or equal coverage."""
    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)

    orig_m_full, orig_m_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = len(groups)
    SC.M_SMALL = max(1, len(groups) // 2)
    try:
        client     = MechVisionClient()
        projects   = client.get_projects()
        assert PROJ_NAME in projects
        project_id = projects[PROJ_NAME]

        opt = Optimizer(
            part_name    = PART,
            client       = client,
            project_id   = project_id,
            scene_groups = groups,
            warm_start   = ws,
            cache        = None,
            use_two_pass = False,
        )

        coarse = opt._default_coarse()
        fine   = opt._default_fine()

        log.info("Running Phase 2a joint grid ...")
        best = opt._phase2a_joint_grid(coarse, fine, ang_thresh=SC.ANG_THRESH_REGIME_GATE)

        log.info(f"  Best refStep={best.config['coarse']['refStep']}  "
                 f"distQ={best.config['coarse']['distQuantification']:.2f}  "
                 f"cov={best.coverage:.3f}  evals={opt._n_evals}")

        assert best is not None
        assert 0.0 <= best.coverage <= 1.0
        assert best.config["coarse"]["refStep"] >= 1

        log.info("PASS: phase2a_joint_grid")

    finally:
        SC.M_FULL  = orig_m_full
        SC.M_SMALL = orig_m_small
        opt.cleanup()
        client.close()


def test_phase2_full():
    """Live: Full Phase 2 coarse CD should complete without crashing."""
    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)

    orig_m_full, orig_m_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = len(groups)
    SC.M_SMALL = max(1, len(groups) // 2)
    try:
        client     = MechVisionClient()
        projects   = client.get_projects()
        assert PROJ_NAME in projects
        project_id = projects[PROJ_NAME]

        opt = Optimizer(
            part_name    = PART,
            client       = client,
            project_id   = project_id,
            scene_groups = groups,
            warm_start   = ws,
            cache        = None,
            use_two_pass = False,
        )

        regime = {
            "id": "A", "coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False,
            "coarse": opt._default_coarse(),
            "fine":   opt._default_fine(),
        }

        log.info("Running full Phase 2 coarse CD ...")
        result = opt.phase2_coarse_cd(regime)

        log.info(f"Phase 2 result: cov={result.coverage:.3f}  "
                 f"time={result.mean_time:.3f}s  evals={opt._n_evals}")
        for k, v in result.config["coarse"].items():
            log.info(f"    {k} = {v}")

        assert result is not None
        assert result.coverage >= 0.0
        log.info("PASS: phase2_full")

    finally:
        SC.M_FULL  = orig_m_full
        SC.M_SMALL = orig_m_small
        opt.cleanup()
        client.close()


def test_early_exit_fires():
    """evaluate_phase_sweep skips Pass 2 when top Pass-1 coverage >= TARGET_COVERAGE.

    dry_run=True returns coverage=1.0 for every call.  We monkey-patch
    evaluate_config to count actual calls and verify Pass 2 runs on 1 survivor
    instead of K_SURVIVORS.
    """
    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)

    orig_m_full, orig_m_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 3
    SC.M_SMALL = 2
    try:
        opt = Optimizer(
            part_name    = PART,
            client       = None,
            project_id   = -1,
            scene_groups = groups,
            warm_start   = ws,
            cache        = None,
            use_two_pass = True,
            dry_run      = True,
        )

        # Count evaluate_config calls via monkey-patch
        call_count = [0]
        _orig = opt.evaluate_config
        def _counting(*args, **kwargs):
            call_count[0] += 1
            return _orig(*args, **kwargs)
        opt.evaluate_config = _counting

        n_candidates = SC.TWO_PASS_MIN_CANDIDATES + 2   # ensure use_tp fires
        coarse_v = [opt._default_coarse() for _ in range(n_candidates)]
        fine_v   = [opt._default_fine()   for _ in range(n_candidates)]

        result = opt.evaluate_phase_sweep(coarse_v, fine_v, label="test-early-exit")

        # evaluate_config is called once per candidate (scenes are batched inside it).
        # Pass 1: n_candidates calls; Pass 2: 1 call (early-exit → 1 survivor)
        expected = n_candidates + 1
        assert call_count[0] == expected, \
            (f"Early-exit should give {expected} evaluate_config calls "
             f"({n_candidates} Pass1 + 1 Pass2 survivor), got {call_count[0]}")
        assert result.coverage == 1.0

        log.info(f"  calls={call_count[0]}  expected={expected}")
        log.info("PASS: test_early_exit_fires")

    finally:
        SC.M_FULL  = orig_m_full
        SC.M_SMALL = orig_m_small
        opt.cleanup()


if __name__ == "__main__":
    print("test_phase2.py  (requires live MechVision for live tests)\n")
    test_dry_run_phase2()
    test_early_exit_fires()
    test_phase2a_joint_grid()
    test_phase2_full()
    print("\nAll Phase 2 tests PASSED")
