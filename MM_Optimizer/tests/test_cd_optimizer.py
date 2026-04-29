"""
test_cd_optimizer.py  —  Coordinate Descent Optimizer test suite
-----------------------------------------------------------------
Dry-run per-phase tests (no MechVision required):
  test_dry_phase1       — Phase 1 regime gate builds param dicts, returns >= 1 regime
  test_dry_phase2       — Phase 2 coarse CD builds all candidate grids, cov=1.0
  test_dry_phase3       — Phase 3 fine CD builds all fine param sweeps, cov=1.0
  test_dry_full_run     — Full 6-phase pipeline completes without error, cov=1.0

Unit test (no MechVision required):
  test_early_exit_fires — Two-pass Pass-2 runs on 1 survivor when top-1 cov >= TARGET

Minimal live test (requires MechVision with CAD_Match project loaded):
  test_live_minimal     — Phase 1 with M_FULL=2 proves end-to-end connectivity

Run from project root:
    python MM_Optimizer/tests/test_cd_optimizer.py [--live]
"""

import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mm_adapter.mm_adapter        import MechVisionClient
from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
from MM_Optimizer.optimizer       import Optimizer, PROJ_NAME, MM_MODEL_ROOT, EvalResult
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
import MM_Optimizer.search_config  as SC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
MODEL_PATH = os.path.join(MM_MODEL_ROOT, f"{PART}_surface", f"{PART}_surface.ply")


def _load_fixtures():
    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)
    return groups, ws


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run per-phase tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dry_phase1():
    """Phase 1 regime gate: builds param dicts, returns >= 1 passing regime."""
    groups, ws = _load_fixtures()
    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 3
    SC.M_SMALL = 2
    opt = Optimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws,
        cache=None, use_two_pass=False, dry_run=True,
    )
    try:
        passing = opt.phase1_regime_gate()
        assert len(passing) >= 1, "Dry run should produce at least one passing regime"
        log.info(f"  dry phase1: {len(passing)} regimes  evals={opt._n_evals}")
        log.info("PASS: test_dry_phase1")
    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
        opt.cleanup()


def test_dry_phase2():
    """Phase 2 coarse CD: builds all candidate grids, returns cov=1.0."""
    groups, ws = _load_fixtures()
    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 3
    SC.M_SMALL = 2
    opt = Optimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws,
        cache=None, use_two_pass=False, dry_run=True,
    )
    try:
        regime = {
            "id": "A", "coarse_mode": 0.0, "fine_mode": 0.0, "needs_edge": False,
            "coarse": opt._default_coarse(),
            "fine":   opt._default_fine(),
        }
        result = opt.phase2_coarse_cd(regime)
        assert result is not None
        assert result.coverage == 1.0, f"Expected cov=1.0, got {result.coverage}"
        n_grid = len(SC.PHASE2A_REFSTEP_SCALES) * len(SC.PHASE2A_DISTQ_VALUES)
        log.info(f"  dry phase2: 2a grid size={n_grid}  evals={opt._n_evals}")
        log.info("PASS: test_dry_phase2")
    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
        opt.cleanup()


def test_dry_phase3():
    """Phase 3 fine CD: builds all fine param sweeps, returns cov=1.0."""
    groups, ws = _load_fixtures()
    orig_full = SC.M_FULL
    SC.M_FULL = 3
    opt = Optimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws,
        cache=None, use_two_pass=False, dry_run=True,
    )
    try:
        coarse = opt._default_coarse()
        fine   = opt._default_fine()
        ph2_stub = EvalResult(
            score=0.1, coverage=1.0, mean_time=0.5,
            per_scene=[{"pos_errors": [0.001]} for _ in range(3)],
            n_scenes=3, config={"coarse": coarse, "fine": fine},
        )
        result = opt.phase3_fine_cd(coarse, fine, ph2_stub)
        assert result is not None
        assert result.coverage == 1.0, f"Expected cov=1.0, got {result.coverage}"
        log.info(f"  dry phase3: evals={opt._n_evals}")
        log.info("PASS: test_dry_phase3")
    finally:
        SC.M_FULL = orig_full
        opt.cleanup()


def test_dry_full_run():
    """Full 6-phase pipeline dry run: completes without error, cov=1.0."""
    groups, ws = _load_fixtures()
    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 3
    SC.M_SMALL = 2
    opt = Optimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws,
        cache=None, use_two_pass=False, dry_run=True,
    )
    try:
        result = opt.run()
        assert result is not None, "run() returned None in dry mode"
        assert result.coverage == 1.0, f"Expected cov=1.0, got {result.coverage}"
        log.info(f"  dry full run: cov={result.coverage:.2f}  evals={opt._n_evals}")
        log.info("PASS: test_dry_full_run")
    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
        opt.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Unit test — two-pass early exit
# ─────────────────────────────────────────────────────────────────────────────

def test_early_exit_fires():
    """Two-pass Pass-2 runs on 1 survivor when top Pass-1 cov >= TARGET_COVERAGE."""
    groups, ws = _load_fixtures()
    orig_full, orig_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = 3
    SC.M_SMALL = 2
    opt = Optimizer(
        part_name=PART, client=None, project_id=-1,
        scene_groups=groups, warm_start=ws,
        cache=None, use_two_pass=True, dry_run=True,
    )
    try:
        call_count = [0]
        _orig = opt.evaluate_config
        def _counting(*args, **kwargs):
            call_count[0] += 1
            return _orig(*args, **kwargs)
        opt.evaluate_config = _counting

        n_candidates = SC.TWO_PASS_MIN_CANDIDATES + 2
        coarse_v = [opt._default_coarse() for _ in range(n_candidates)]
        fine_v   = [opt._default_fine()   for _ in range(n_candidates)]

        result = opt.evaluate_phase_sweep(coarse_v, fine_v, label="test-early-exit")

        # Pass1: n_candidates calls; Pass2: 1 call (early-exit → 1 survivor)
        expected = n_candidates + 1
        assert call_count[0] == expected, \
            (f"Early-exit: expected {expected} calls "
             f"({n_candidates} P1 + 1 P2), got {call_count[0]}")
        assert result.coverage == 1.0
        log.info(f"  early exit: calls={call_count[0]}  expected={expected}")
        log.info("PASS: test_early_exit_fires")
    finally:
        SC.M_FULL  = orig_full
        SC.M_SMALL = orig_small
        opt.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Minimal live test
# ─────────────────────────────────────────────────────────────────────────────

def test_live_minimal():
    """Phase 1 live with M_FULL=2: proves MechVision connectivity and end-to-end stack."""
    groups, ws = _load_fixtures()
    orig_full = SC.M_FULL
    SC.M_FULL = min(2, len(groups))
    client    = MechVisionClient()
    opt       = None
    try:
        projects = client.get_projects()
        assert PROJ_NAME in projects, f"Project '{PROJ_NAME}' not found: {projects}"

        opt = Optimizer(
            part_name=PART, client=client,
            project_id=projects[PROJ_NAME],
            scene_groups=groups, warm_start=ws,
            cache=None, use_two_pass=False,
        )
        passing = opt.phase1_regime_gate()

        assert len(passing) >= 1, "No regime passed Phase 1"
        assert all(r["coverage"] >= SC.PHASE1_COVERAGE_GATE for r in passing), \
            f"A regime slipped below gate: {[r['coverage'] for r in passing]}"
        log.info(f"  live minimal: {len(passing)} regime(s)  "
                 f"best cov={passing[0]['coverage']:.3f}  evals={opt._n_evals}")
        log.info("PASS: test_live_minimal")
    finally:
        SC.M_FULL = orig_full
        if opt:
            opt.cleanup()
        client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CD Optimizer test suite")
    parser.add_argument("--live", action="store_true",
                        help="Also run the live MechVision test")
    args = parser.parse_args()

    print("test_cd_optimizer.py\n")

    print("--- Dry-run per-phase tests ---")
    test_dry_phase1()
    test_dry_phase2()
    test_dry_phase3()

    print("\n--- Full pipeline dry run ---")
    test_dry_full_run()

    print("\n--- Unit test: two-pass early exit ---")
    test_early_exit_fires()

    if args.live:
        print("\n--- Live test (requires MechVision) ---")
        test_live_minimal()
    else:
        print("\n(skip live test — pass --live to enable)")

    print("\nAll tests PASSED")
