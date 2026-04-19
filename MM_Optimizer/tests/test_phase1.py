"""
test_phase1.py  —  Live smoke test: Phase 1 regime gate
Requires live MechVision (CAD_Match project loaded).
Run from project root:
    python MM_Optimizer/tests/test_phase1.py
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


def test_phase1_runs():
    """Phase 1 should complete and return at least one passing regime."""
    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)

    client     = MechVisionClient()
    projects   = client.get_projects()
    assert PROJ_NAME in projects, f"Project '{PROJ_NAME}' not found: {projects}"
    project_id = projects[PROJ_NAME]

    orig_m = SC.M_FULL
    SC.M_FULL = len(groups)   # use all available M-scenes
    try:
        opt = Optimizer(
            part_name    = PART,
            client       = client,
            project_id   = project_id,
            scene_groups = groups,
            warm_start   = ws,
            cache        = None,
            use_two_pass = False,
        )

        log.info("Running Phase 1 regime gate ...")
        passing = opt.phase1_regime_gate()

        log.info(f"Passing regimes: {[r['id'] for r in passing]}")
        for r in passing:
            log.info(f"  Regime {r['id']}: coverage={r['coverage']:.3f}  "
                     f"coarse_mode={r['coarse_mode']}  fine_mode={r['fine_mode']}")

        log.info(f"  Total MechVision calls: {opt._n_evals}")

        for r in passing:
            assert r["coverage"] >= SC.PHASE1_COVERAGE_GATE, \
                f"Regime {r['id']} coverage {r['coverage']:.2f} < gate"
            assert "coarse" in r and "fine" in r, "Missing coarse/fine in regime dict"

        if not passing:
            log.warning("No regime passed Phase 1 — warm start may be sub-optimal")
        else:
            log.info(f"Phase 1 passed with {len(passing)} regime(s)")

        log.info("PASS: phase1_runs")

    finally:
        SC.M_FULL = orig_m
        opt.cleanup()
        client.close()


def test_dry_run_phase1():
    """Dry run: Phase 1 should build param dicts and return without crashing."""
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
        passing = opt.phase1_regime_gate()
        assert len(passing) >= 1, "Dry run should produce at least one 'passing' regime"
        log.info(f"  dry run: {len(passing)} regimes  evals={opt._n_evals}")
        log.info("PASS: dry_run_phase1")

    finally:
        SC.M_FULL = orig_m
        opt.cleanup()


if __name__ == "__main__":
    print("test_phase1.py  (requires live MechVision for test_phase1_runs)\n")
    test_dry_run_phase1()
    test_phase1_runs()
    print("\nAll Phase 1 tests PASSED")
