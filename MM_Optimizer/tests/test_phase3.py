"""
test_phase3.py  —  Live smoke test: Phase 3 fine coordinate descent
Requires live MechVision (CAD_Match project loaded).
Run from project root:
    python MM_Optimizer/tests/test_phase3.py
"""

import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mm_adapter.mm_adapter import MechVisionClient
from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
from MM_Optimizer.optimizer       import Optimizer, PROJ_NAME, MM_MODEL_ROOT, EvalResult
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
import MM_Optimizer.search_config  as SC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
MODEL_PATH = os.path.join(MM_MODEL_ROOT, f"{PART}_surface", f"{PART}_surface.ply")


def test_dry_run_phase3():
    """Dry run: Phase 3 should build all fine param sweeps and return without crashing."""
    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)

    orig_m = SC.M_FULL
    SC.M_FULL = 3
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
        coarse = opt._default_coarse()
        fine   = opt._default_fine()

        _t = 0.5 / SC.SCORE_TIME_NORM
        _c = (1.0 - 1.0) * SC.SCORE_COV_NORM
        ph2_result = EvalResult(
            score=_t + _c, coverage=1.0, mean_time=0.5,
            per_scene=[{"pos_errors": [0.001]} for _ in range(3)],
            n_scenes=3, config={"coarse": coarse, "fine": fine},
            score_time_term=_t, score_cov_term=_c,
            score_quality=1.0 - (_t + _c) / SC.SCORE_WORST_CASE,
        )

        result = opt.phase3_fine_cd(coarse, fine, ph2_result)
        assert result is not None
        assert result.coverage == 1.0
        log.info(f"  dry run: evals={opt._n_evals}")
        log.info("PASS: dry_run_phase3")
    finally:
        SC.M_FULL = orig_m
        opt.cleanup()


def test_phase3_live():
    """Live: Phase 3 fine CD — verify it runs and produces valid result."""
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

        regime = {
            "id": "A", "coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False,
            "coarse": coarse, "fine": fine,
        }
        log.info("Running Phase 2 for baseline ...")
        ph2_result = opt.phase2_coarse_cd(regime)
        coarse = dict(ph2_result.config["coarse"])
        fine   = dict(ph2_result.config["fine"])
        log.info(f"  Phase 2 baseline cov={ph2_result.coverage:.3f}")

        log.info("Running Phase 3 fine CD ...")
        ph3_result = opt.phase3_fine_cd(coarse, fine, ph2_result)

        log.info(f"Phase 3 result: cov={ph3_result.coverage:.3f}  "
                 f"time={ph3_result.mean_time:.3f}s  evals={opt._n_evals}")
        for k, v in ph3_result.config["fine"].items():
            log.info(f"    {k} = {v}")

        assert ph3_result is not None
        assert ph3_result.coverage >= 0.0
        assert ph3_result.coverage >= ph2_result.coverage or \
               ph3_result.score <= ph2_result.score, \
               f"Phase 3 regressed: ph2={ph2_result.coverage:.3f} ph3={ph3_result.coverage:.3f}"

        log.info("PASS: phase3_live")
    finally:
        SC.M_FULL  = orig_m_full
        SC.M_SMALL = orig_m_small
        opt.cleanup()
        client.close()


if __name__ == "__main__":
    print("test_phase3.py  (requires live MechVision for live tests)\n")
    test_dry_run_phase3()
    test_phase3_live()
    print("\nAll Phase 3 tests PASSED")
