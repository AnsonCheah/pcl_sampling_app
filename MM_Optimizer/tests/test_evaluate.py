"""
test_evaluate.py  —  Live smoke test: single evaluate_config() call
Requires live MechVision (CAD_Match project loaded).
Run from project root:
    python MM_Optimizer/tests/test_evaluate.py
"""

import logging
import os
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mm_adapter.mm_adapter import MechVisionClient
from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
from MM_Optimizer.optimizer       import Optimizer, PROJ_NAME, MM_MODEL_ROOT
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
from MM_Optimizer.eval_cache      import EvalCache
import MM_Optimizer.search_config  as SC

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
MODEL_PATH = os.path.join(MM_MODEL_ROOT, f"{PART}_surface", f"{PART}_surface.ply")


def test_single_evaluate():
    groups = list_synthetic_scenes(SCENES_DIR)
    assert groups, f"No scene groups found in {SCENES_DIR}"
    n_per = [len(g) for g in groups]
    log.info(f"Found {len(groups)} M-scenes, {n_per} instances/scene")

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)
    log.info(f"Warm start: refStep={ws.refStep}  distQ={ws.distQuantification:.1f}")

    client     = MechVisionClient()
    projects   = client.get_projects()
    assert PROJ_NAME in projects, f"Project '{PROJ_NAME}' not loaded: {projects}"
    project_id = projects[PROJ_NAME]

    orig_m = SC.M_FULL
    SC.M_FULL = 3
    try:
        opt = Optimizer(
            part_name    = PART,
            client       = client,
            project_id   = project_id,
            scene_groups = groups,
            warm_start   = ws,
            cache        = None,
            use_two_pass = False,
            dry_run      = False,
        )

        coarse = opt._default_coarse()
        fine   = opt._default_fine()
        scenes = opt._sample_scenes(3)

        log.info("Running evaluate_config() with default warm-start params ...")
        result = opt.evaluate_config(coarse, fine, scenes,
                                     SC.POS_THRESH_TIGHT, SC.ANG_THRESH_TIGHT)
        log.info(f"  coverage  = {result.coverage:.3f}  (mean instance coverage across scenes)")
        log.info(f"  mean_time = {result.mean_time:.3f} s")
        log.info(f"  score     = {result.score:.1f}")
        log.info(f"  n_scenes  = {result.n_scenes}")
        for i, s in enumerate(result.per_scene):
            n_gt   = len(s.get("pos_errors", []))
            n_pass = sum(1 for e in s.get("pos_errors", []) if e is not None)
            log.info(f"    scene {i}: instance_coverage={s.get('instance_coverage', '?'):.2f}"
                     f"  ({n_pass}/{n_gt} instances pass)")

        assert 0.0 <= result.coverage <= 1.0, "Coverage out of range"
        assert result.mean_time > 0,          "mean_time must be positive"
        assert result.n_scenes == 3,          "Expected 3 scenes evaluated"
        assert len(result.per_scene) == 3,    "Expected 3 per_scene entries"
        for s in result.per_scene:
            assert "instance_coverage" in s
            assert "pos_errors"        in s
            assert "ang_errors"        in s
            assert "coarse_time_s"     in s
            assert "fine_time_s"       in s

        log.info("PASS: single_evaluate  (structure OK)")

    finally:
        SC.M_FULL = orig_m
        opt.cleanup()
        client.close()


def test_cache_integration():
    """Verify that a second evaluate_config() call hits the cache."""
    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)

    client     = MechVisionClient()
    projects   = client.get_projects()
    project_id = projects[PROJ_NAME]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = f.name

    orig_m = SC.M_FULL
    SC.M_FULL = 2
    try:
        cache = EvalCache(cache_path, enabled=True)
        opt   = Optimizer(PART, client, project_id, groups, ws,
                          cache=cache, use_two_pass=False)

        coarse = opt._default_coarse()
        fine   = opt._default_fine()
        scenes = opt._sample_scenes(2)

        # First call — cache miss
        r1 = opt.evaluate_config(coarse, fine, scenes)
        stats1 = cache.stats()
        assert stats1["misses"] == 1
        assert stats1["hits"]   == 0

        # Second call with same config + same scenes — cache HIT
        r2 = opt.evaluate_config(coarse, fine, scenes)
        stats2 = cache.stats()
        assert stats2["hits"] == 1, \
            f"Expected 1 hit after second call, got {stats2}"
        assert r1.score == r2.score, "Cached result score must match original"
        assert opt._n_evals == 2, "Only 2 MechVision calls (first eval, 2 scenes)"

        log.info(f"  cache stats: {cache.stats()}")
        log.info("PASS: cache_integration")

    finally:
        SC.M_FULL = orig_m
        opt.cleanup()
        client.close()
        os.unlink(cache_path)


if __name__ == "__main__":
    print("test_evaluate.py  (requires live MechVision)\n")
    test_single_evaluate()
    test_cache_integration()
    print("\nAll evaluate tests PASSED")
