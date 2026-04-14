"""
eval_cache.py  —  Strategy 1: Transposition Table
--------------------------------------------------
Caches evaluate_config() results keyed by (config_hash, scene_set_hash).
Eliminates re-evaluation of the same config across phases (e.g., Phase 5
re-sweeps params already evaluated in Phase 2b).

Design:
- Key: first 16 chars of SHA-256 over sorted config JSON + sorted scene paths
- Value: full result dict {score, coverage, mean_time, per_run, per_instance}
- Backed by a JSON file for crash-resume and cross-run reuse
- Thread-safe for single-process use (no locking needed)

Detach: set ENABLE_CACHE=False in optimizer.py, or pass cache=None
        to evaluate_config().  EvalCache itself is never called in that path.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class EvalCache:
    """Persistent result cache for optimizer evaluations.

    Usage::

        cache = EvalCache("results/eval_cache.json")
        key   = cache.make_key(config, scene_paths)
        hit   = cache.get(key)
        if hit is None:
            result = evaluate(...)
            cache.put(key, result)
        cache.save()
    """

    def __init__(self, cache_path: str, enabled: bool = True):
        self.cache_path = cache_path
        self.enabled    = enabled
        self._store: Dict[str, dict] = {}
        self._hits   = 0
        self._misses = 0
        if enabled and os.path.exists(cache_path):
            self._load()

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(config: dict, scene_paths: List[str]) -> str:
        """Stable 16-char hex key from config + scene paths."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        scenes_str = ",".join(sorted(scene_paths))
        raw = config_str + "::" + scenes_str
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
            log.debug(f"cache HIT  {key}")
        else:
            self._misses += 1
            log.debug(f"cache MISS {key}")
        return result

    def put(self, key: str, result: dict) -> None:
        if not self.enabled:
            return
        self._store[key] = result

    def save(self) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._store, f, indent=2, default=str)
        log.info(f"Cache saved: {len(self._store)} entries → {self.cache_path}")

    def _load(self) -> None:
        try:
            with open(self.cache_path, "r") as f:
                self._store = json.load(f)
            log.info(f"Cache loaded: {len(self._store)} entries from {self.cache_path}")
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Cache load failed ({e}), starting fresh")
            self._store = {}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "entries":  len(self._store),
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        s = self.stats()
        return (f"EvalCache(entries={s['entries']}, hits={s['hits']}, "
                f"misses={s['misses']}, hit_rate={s['hit_rate']:.1%})")
