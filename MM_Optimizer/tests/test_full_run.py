"""
test_full_run.py  —  Live end-to-end optimizer run (Phases 1–6)
Requires live MechVision (CAD_Match project loaded).
Run from project root:
    python MM_Optimizer/tests/test_full_run.py [--dry_run] [--no_cache] [--two_pass]
"""

import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mm_adapter.mm_adapter import MechVisionClient
from MM_Optimizer.eval_cache      import EvalCache
from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
from MM_Optimizer.optimizer       import Optimizer, PROJ_NAME, MM_MODEL_ROOT
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
import MM_Optimizer.search_config  as SC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
MODEL_PATH = os.path.join(MM_MODEL_ROOT, f"{PART}_surface", f"{PART}_surface.ply")


def test_full_dry_run():
    """Dry run: all 6 phases should complete without error."""
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
        result = opt.run()
        assert result is not None, "run() returned None in dry mode"
        assert result.coverage == 1.0
        log.info(f"  dry run: cov={result.coverage:.2f}  evals={opt._n_evals}")
        log.info("PASS: test_full_dry_run")
    finally:
        SC.M_FULL  = orig_m_full
        SC.M_SMALL = orig_m_small
        opt.cleanup()


def test_full_live(use_cache=True, use_two_pass=False):
    """Live full run: Phases 1–6 across all available M-scenes."""
    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)

    n_per = [len(g) for g in groups]
    log.info(f"M-scenes: {len(groups)}, instances/scene: {n_per}")

    orig_m_full, orig_m_small = SC.M_FULL, SC.M_SMALL
    SC.M_FULL  = len(groups)
    SC.M_SMALL = max(1, len(groups) // 2)
    try:
        client     = MechVisionClient()
        projects   = client.get_projects()
        assert PROJ_NAME in projects
        project_id = projects[PROJ_NAME]

        cache = EvalCache(
            os.path.join(_ROOT, "MM_Optimizer", "results", f"cache_test_{PART}.json"),
            enabled=use_cache,
        ) if use_cache else None

        opt = Optimizer(
            part_name    = PART,
            client       = client,
            project_id   = project_id,
            scene_groups = groups,
            warm_start   = ws,
            cache        = cache,
            use_two_pass = use_two_pass,
        )

        log.info(f"Starting full run (cache={'ON' if use_cache else 'OFF'}, "
                 f"two_pass={'ON' if use_two_pass else 'OFF'}) ...")
        result = opt.run()

        assert result is not None, "run() returned None"
        log.info(f"\n--- Full run summary ---")
        log.info(f"  coverage  = {result.coverage:.3f}  (mean instance coverage)")
        log.info(f"  mean_time = {result.mean_time:.3f}s  (per M-scene)")
        log.info(f"  score     = {result.score:.4f}  quality={result.score_quality:.3f}")
        log.info(f"  evals     = {opt._n_evals}")
        if cache:
            log.info(f"  cache     = {cache.stats()}")

        log.info(f"\n  Final coarse params:")
        for k, v in result.config.get("coarse", {}).items():
            log.info(f"    {k} = {v}")
        log.info(f"\n  Final fine params:")
        for k, v in result.config.get("fine", {}).items():
            log.info(f"    {k} = {v}")

        out_path = opt.export_best(result)
        log.info(f"  Exported → {out_path}")
        log.info("PASS: test_full_live")
        return result

    finally:
        SC.M_FULL  = orig_m_full
        SC.M_SMALL = orig_m_small
        opt.cleanup()
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run",  action="store_true", help="Skip MechVision calls")
    parser.add_argument("--no_cache", action="store_true", help="Disable EvalCache")
    parser.add_argument("--two_pass", action="store_true", help="Enable two-pass multi-fidelity")
    args = parser.parse_args()

    print("test_full_run.py\n")

    if args.dry_run:
        test_full_dry_run()
    else:
        test_full_dry_run()
        test_full_live(use_cache=not args.no_cache, use_two_pass=args.two_pass)

    print("\nAll full-run tests PASSED")
