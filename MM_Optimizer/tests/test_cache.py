"""
test_cache.py  —  Unit smoke tests for EvalCache (Strategy 1)
No MechVision required. Run from project root:
    python MM_Optimizer/tests/test_cache.py
"""

import os
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from MM_Optimizer.eval_cache import EvalCache


def test_basic_hit_miss():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = f.name

    try:
        cache = EvalCache(cache_path, enabled=True)
        config_a = {"coarse": {"refStep": 5, "distQ": 5.0}, "fine": {"op": 1.0}}
        scenes_a = ["/tmp/scene_0", "/tmp/scene_1"]

        key_a = cache.make_key(config_a, scenes_a)

        # Miss
        assert cache.get(key_a) is None, "Expected cache miss on empty cache"

        # Put + hit
        fake_result = {"score": 1000.0, "coverage": 0.8, "mean_time": 0.05,
                       "per_scene": [], "n_scenes": 2, "config": config_a}
        cache.put(key_a, fake_result)
        hit = cache.get(key_a)
        assert hit is not None,                    "Expected cache hit after put"
        assert hit["coverage"] == 0.8,             "Coverage mismatch"
        assert hit["score"]    == 1000.0,          "Score mismatch"

        stats = cache.stats()
        assert stats["hits"]   == 1, f"Expected 1 hit, got {stats['hits']}"
        assert stats["misses"] == 1, f"Expected 1 miss, got {stats['misses']}"
        print(f"  stats: {stats}")
    finally:
        os.unlink(cache_path)

    print("  PASS: basic_hit_miss")


def test_key_is_order_independent():
    cache = EvalCache("unused.json", enabled=True)
    cache._store = {}  # don't persist

    config = {"coarse": {"a": 1, "b": 2}, "fine": {"c": 3}}
    scenes = ["/s/2", "/s/1", "/s/3"]

    key1 = cache.make_key(config, scenes)
    key2 = cache.make_key(config, list(reversed(scenes)))

    assert key1 == key2, "Key must be scene-order-independent"
    print("  PASS: key_is_order_independent")


def test_different_configs_different_keys():
    cache = EvalCache("unused.json", enabled=True)
    config_a = {"coarse": {"refStep": 5},  "fine": {}}
    config_b = {"coarse": {"refStep": 10}, "fine": {}}
    scenes   = ["/s/0", "/s/1"]

    key_a = cache.make_key(config_a, scenes)
    key_b = cache.make_key(config_b, scenes)
    assert key_a != key_b, "Different configs must produce different keys"
    print("  PASS: different_configs_different_keys")


def test_persistence():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = f.name

    try:
        # Write
        c1 = EvalCache(cache_path, enabled=True)
        key = c1.make_key({"coarse": {"x": 1}, "fine": {}}, ["/a"])
        c1.put(key, {"score": 99.0, "coverage": 0.5, "mean_time": 1.0,
                     "per_scene": [], "n_scenes": 1, "config": {}})
        c1.save()

        # Read back
        c2 = EvalCache(cache_path, enabled=True)
        hit = c2.get(key)
        assert hit is not None,        "Persisted entry not found after reload"
        assert hit["score"] == 99.0,   "Persisted score mismatch"
        print(f"  Loaded {len(c2)} entries from disk")
    finally:
        os.unlink(cache_path)

    print("  PASS: persistence")


def test_disabled_cache():
    cache = EvalCache("unused.json", enabled=False)
    key = cache.make_key({"coarse": {}, "fine": {}}, [])
    cache.put(key, {"score": 1.0, "coverage": 1.0, "mean_time": 0.0,
                    "per_scene": [], "n_scenes": 0, "config": {}})
    assert cache.get(key) is None, "Disabled cache must always return None"
    print("  PASS: disabled_cache")


if __name__ == "__main__":
    print("test_cache.py")
    test_basic_hit_miss()
    test_key_is_order_independent()
    test_different_configs_different_keys()
    test_persistence()
    test_disabled_cache()
    print("\nAll cache tests PASSED")
