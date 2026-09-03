"""
test_optimizer_utils.py  --  Unit and integration tests for optimizer utilities
-------------------------------------------------------------------------------
EvalCache tests (no MechVision):
  test_cache_basic_hit_miss          -- get after put returns correct entry
  test_cache_key_order_independent   -- scene list order doesn't affect key
  test_cache_different_configs       -- different configs produce different keys
  test_cache_persistence             -- save/reload round-trip
  test_cache_disabled                -- disabled cache always returns None

Mesh analysis tests (no MechVision):
  test_warm_start_values             -- analyze_mesh returns sane warm-start values
  test_n_instances_sets_output_num   -- outputNum tracks n_instances argument
  test_pcd_without_normals           -- analyze_mesh doesn't crash on normal-free pcd

evaluate_config integration tests (requires MechVision with CAD_Match project):
  test_evaluate_structure            -- result has correct fields and value ranges
  test_cache_integration             -- second call hits cache, evals count doesn't grow

Run from project root:
    python MM_Optimizer/tests/test_optimizer_utils.py [--live]
"""

import argparse
import logging
import os
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from MM_Optimizer.eval_cache      import EvalCache
from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
from MM_Optimizer.mv_evaluator    import MVEvaluator, PROJ_NAME
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
import MM_Optimizer.search_config  as SC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PART       = "25333MB000"
SCENES_DIR = os.path.join(_ROOT, "output", "synthetic_target", PART)
# Warm-start cloud comes from the app's own bundle, exactly as tuning_stage/tuner do.
# NOT from the deployed MechVision library: model_sync writes <part>/<part>.ply there,
# and it is rewritten per regime.
MODEL_PATH = os.path.join(_ROOT, "output", "reference_pcd", PART,
                          f"{PART}_surface", f"{PART}_surface.ply")


# -----------------------------------------------------------------------------
# EvalCache unit tests
# -----------------------------------------------------------------------------

def test_cache_basic_hit_miss():
    """get after put returns the stored entry; stats track hits and misses."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = f.name
    try:
        cache    = EvalCache(cache_path, enabled=True)
        config   = {"coarse": {"refStep": 5, "distQ": 5.0}, "fine": {"op": 1.0}}
        scenes   = ["/tmp/scene_0", "/tmp/scene_1"]
        key      = cache.make_key(config, scenes)

        assert cache.get(key) is None, "Expected miss on empty cache"

        payload = {"score": 1000.0, "coverage": 0.8, "mean_time": 0.05,
                   "per_scene": [], "n_scenes": 2, "config": config}
        cache.put(key, payload)
        hit = cache.get(key)
        assert hit is not None,           "Expected hit after put"
        assert hit["coverage"] == 0.8
        assert hit["score"]    == 1000.0

        stats = cache.stats()
        assert stats["hits"]   == 1
        assert stats["misses"] == 1
        log.info(f"  cache stats: {stats}")
        log.info("PASS: test_cache_basic_hit_miss")
    finally:
        os.unlink(cache_path)


def test_cache_key_order_independent():
    """Scene list order must not affect the cache key."""
    cache = EvalCache("unused.json", enabled=True)
    cache._store = {}
    config = {"coarse": {"a": 1, "b": 2}, "fine": {"c": 3}}
    scenes = ["/s/2", "/s/1", "/s/3"]
    key1   = cache.make_key(config, scenes)
    key2   = cache.make_key(config, list(reversed(scenes)))
    assert key1 == key2, "Key must be scene-order-independent"
    log.info("PASS: test_cache_key_order_independent")


def test_cache_different_configs():
    """Different configs must produce different cache keys."""
    cache    = EvalCache("unused.json", enabled=True)
    config_a = {"coarse": {"refStep": 5},  "fine": {}}
    config_b = {"coarse": {"refStep": 10}, "fine": {}}
    scenes   = ["/s/0", "/s/1"]
    assert cache.make_key(config_a, scenes) != cache.make_key(config_b, scenes), \
        "Different configs must give different keys"
    log.info("PASS: test_cache_different_configs")


def test_cache_key_content_sensitive():
    """Regenerating a scene dir's sample_*.ply must change the key (guards against the stale
    path-keyed cache that masked the world-Z matching offset behind a high cached coverage)."""
    cache  = EvalCache("unused.json", enabled=True)
    config = {"coarse": {"refStep": 5}, "fine": {}}
    with tempfile.TemporaryDirectory() as d:
        scene = os.path.join(d, "scene_00000")
        os.makedirs(scene)
        ply = os.path.join(scene, "sample_0.ply")
        with open(ply, "w") as f:
            f.write("ply-v1")
        key1 = cache.make_key(config, [scene])
        # Same content, recomputed -> same key.
        assert cache.make_key(config, [scene]) == key1, "Key must be stable for unchanged scenes"
        # Regenerate the scene with different content -> different key.
        import time as _t
        _t.sleep(0.01)
        with open(ply, "w") as f:
            f.write("ply-v2-regenerated-longer")
        os.utime(ply, None)
        key2 = cache.make_key(config, [scene])
        assert key2 != key1, "Regenerated scene content must change the cache key"
    log.info("PASS: test_cache_key_content_sensitive")


def test_cache_persistence():
    """Entries written and saved are readable after a fresh EvalCache load."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = f.name
    try:
        c1  = EvalCache(cache_path, enabled=True)
        key = c1.make_key({"coarse": {"x": 1}, "fine": {}}, ["/a"])
        c1.put(key, {"score": 99.0, "coverage": 0.5, "mean_time": 1.0,
                     "per_scene": [], "n_scenes": 1, "config": {}})
        c1.save()

        c2  = EvalCache(cache_path, enabled=True)
        hit = c2.get(key)
        assert hit is not None,      "Persisted entry not found after reload"
        assert hit["score"] == 99.0, "Persisted score mismatch"
        log.info(f"  loaded {len(c2)} entries from disk")
        log.info("PASS: test_cache_persistence")
    finally:
        os.unlink(cache_path)


def test_cache_disabled():
    """Disabled cache always returns None regardless of puts."""
    cache = EvalCache("unused.json", enabled=False)
    key   = cache.make_key({"coarse": {}, "fine": {}}, [])
    cache.put(key, {"score": 1.0, "coverage": 1.0, "mean_time": 0.0,
                    "per_scene": [], "n_scenes": 0, "config": {}})
    assert cache.get(key) is None, "Disabled cache must always return None"
    log.info("PASS: test_cache_disabled")


# -----------------------------------------------------------------------------
# Mesh analysis unit tests
# -----------------------------------------------------------------------------

def test_warm_start_values():
    """analyze_mesh returns geometrically sane warm-start values."""
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd, n_instances=1)

    log.info(f"  diameter        = {ws.diameter_m*1e3:.1f} mm")
    log.info(f"  flatness_ratio  = {ws.flatness_ratio:.2f}")
    log.info(f"  prefer_edge     = {ws.prefer_edge}")
    log.info(f"  distQ           = {ws.distQuantification:.2f}")
    log.info(f"  angleQ          = {ws.angleQuantification}")
    log.info(f"  maxPairs        = {ws.maxNumOfPointPairsPerFeature}")
    log.info(f"  minVoxel        = {ws.minVoxelLength_mm:.2f} mm")
    log.info(f"  maxVoxel        = {ws.maxVoxelLength_mm:.2f} mm")

    assert ws.diameter_m > 0.01,          "diameter should be > 10mm"
    assert ws.diameter_m < 1.0,           "diameter should be < 1m"
    assert ws.distQuantification > 0,     "distQuantification must be > 0"
    assert ws.angleQuantification in [30, 45, 60, 90]
    assert ws.maxNumOfPointPairsPerFeature >= 100
    assert ws.outputNum == 1
    assert isinstance(ws.prefer_edge, bool)
    assert ws.minVoxelLength_mm >= 0.5
    assert ws.maxVoxelLength_mm > ws.minVoxelLength_mm
    assert ws.maxVoxelLength_mm >= 1.0
    log.info("PASS: test_warm_start_values")


def test_n_instances_sets_output_num():
    """outputNum in WarmStart equals the n_instances argument."""
    pcd = load_reference_pcd(MODEL_PATH)
    for n in [1, 2, 3]:
        ws = analyze_mesh(pcd, n_instances=n)
        assert ws.outputNum == n, f"outputNum={ws.outputNum} != n_instances={n}"
    log.info("PASS: test_n_instances_sets_output_num")


def test_pcd_without_normals():
    """analyze_mesh must not crash on a point cloud with no normals."""
    import open3d as o3d
    import numpy as np
    pts = np.random.randn(500, 3) * 0.05
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    ws = analyze_mesh(pcd, n_instances=1)
    assert ws.diameter_m > 0
    log.info("PASS: test_pcd_without_normals")


# -----------------------------------------------------------------------------
# evaluate_config integration tests (live)
# -----------------------------------------------------------------------------

def test_evaluate_structure():
    """evaluate_config returns result with correct fields and value ranges."""
    from mm_adapter.mm_adapter import MechVisionClient

    groups = list_synthetic_scenes(SCENES_DIR)
    assert groups, f"No scene groups found in {SCENES_DIR}"

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd)

    client = MechVisionClient()
    orig_m = SC.M_FULL
    SC.M_FULL = 3
    opt = None
    try:
        projects = client.get_projects()
        assert PROJ_NAME in projects, f"Project '{PROJ_NAME}' not loaded: {projects}"

        opt    = MVEvaluator(PART, client, projects[PROJ_NAME], groups, ws,
                           cache=None, dry_run=False)
        coarse = opt._default_coarse()
        fine   = opt._default_fine()
        scenes = opt._sample_scenes(3)

        result = opt.evaluate_config(coarse, fine, scenes,
                                     SC.POS_THRESH_TIGHT, SC.ANG_THRESH_TIGHT)
        log.info(f"  coverage  = {result.coverage:.3f}")
        log.info(f"  mean_time = {result.mean_time:.3f} s")
        log.info(f"  n_scenes  = {result.n_scenes}")

        assert 0.0 <= result.coverage <= 1.0
        assert result.mean_time > 0
        assert result.n_scenes == 3
        assert len(result.per_scene) == 3
        for s in result.per_scene:
            assert "instance_coverage" in s
            assert "pos_errors"        in s
            assert "ang_errors"        in s
            assert "coarse_time_s"     in s
            assert "fine_time_s"       in s
        log.info("PASS: test_evaluate_structure")
    finally:
        SC.M_FULL = orig_m
        if opt:
            opt.cleanup()
        client.close()


def test_cache_integration():
    """Second evaluate_config call with same args hits the cache; evals don't grow."""
    from mm_adapter.mm_adapter import MechVisionClient

    groups = list_synthetic_scenes(SCENES_DIR)
    pcd    = load_reference_pcd(MODEL_PATH)
    ws     = analyze_mesh(pcd)

    client = MechVisionClient()
    orig_m = SC.M_FULL
    SC.M_FULL = 2
    opt = None
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = f.name
    try:
        projects = client.get_projects()
        cache    = EvalCache(cache_path, enabled=True)
        opt      = MVEvaluator(PART, client, projects[PROJ_NAME], groups, ws,
                             cache=cache)

        coarse = opt._default_coarse()
        fine   = opt._default_fine()
        scenes = opt._sample_scenes(2)

        r1 = opt.evaluate_config(coarse, fine, scenes)
        assert cache.stats()["misses"] == 1
        assert cache.stats()["hits"]   == 0

        r2 = opt.evaluate_config(coarse, fine, scenes)
        assert cache.stats()["hits"] == 1, \
            f"Expected 1 hit after second call, got {cache.stats()}"
        assert r1.score == r2.score, "Cached result score must match original"
        assert opt._n_evals == 2, f"Expected 2 MV calls (1 eval x 2 scenes), got {opt._n_evals}"
        log.info(f"  cache stats: {cache.stats()}")
        log.info("PASS: test_cache_integration")
    finally:
        SC.M_FULL = orig_m
        if opt:
            opt.cleanup()
        client.close()
        os.unlink(cache_path)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimizer utilities test suite")
    parser.add_argument("--live", action="store_true",
                        help="Also run the live MechVision tests")
    args = parser.parse_args()

    print("test_optimizer_utils.py\n")

    print("--- EvalCache unit tests ---")
    test_cache_basic_hit_miss()
    test_cache_key_order_independent()
    test_cache_different_configs()
    test_cache_persistence()
    test_cache_disabled()

    print("\n--- Mesh analysis unit tests ---")
    test_warm_start_values()
    test_n_instances_sets_output_num()
    test_pcd_without_normals()

    if args.live:
        print("\n--- evaluate_config live tests (requires MechVision) ---")
        test_evaluate_structure()
        test_cache_integration()
    else:
        print("\n(skip live tests -- pass --live to enable)")

    print("\nAll tests PASSED")
