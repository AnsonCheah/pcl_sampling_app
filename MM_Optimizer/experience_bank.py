"""
experience_bank.py  —  Layer 2: Experience Bank
-------------------------------------------------
LanceDB embedded store. No server required.
Auto-written after every validated part run.
Used for ICL retrieval: top-K=5-8 similar past parts injected as demonstrations.

Schema version: 1
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running from MM_Optimizer/ or project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Schema version — bump when adding/removing columns
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1

# Default store location (relative to project root)
_DEFAULT_DB_PATH = os.path.join(_ROOT, "MM_Optimizer", "data", "experience_bank.lance")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ExperienceBank:
    """Thin wrapper around a LanceDB table with lazy import of lancedb."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db = None
        self._table = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def open(self) -> "ExperienceBank":
        """Open (or create) the LanceDB store. Returns self for chaining."""
        import lancedb  # deferred — not always available
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = lancedb.connect(self._db_path)
        if "experience" in self._db.list_tables():
            self._table = self._db.open_table("experience")
        else:
            self._table = None   # created lazily on first insert
        return self

    def close(self) -> None:
        self._db = None
        self._table = None

    def __enter__(self) -> "ExperienceBank":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def insert(
        self,
        *,
        part_name: str,
        model_id: str,
        mesh_path: str,
        mesh_features: dict,
        symmetry_class: str,
        arrangement_context: str,
        best_coverage: float,
        best_mean_time: float,
        best_params: dict,
        vsd_threshold_mm: float = 5.0,
        phase_reached: int = 0,
        n_rounds_to_converge: int = 0,
        reasoning_trace: str = "",
        trajectory: list[dict] | None = None,
        rag_version: str = "unknown",
        warmstart_params: dict | None = None,
        warmstart_coverage: float = 0.0,
        warmstart_mean_time: float = 0.0,
        fell_back_to_warmstart: bool = False,
        retrieval_similarity: float = 0.0,
        constraint_retries: int = 0,
        feature_vector: list[float] | None = None,
    ) -> None:
        """Insert one experience record. Feature vector is auto-computed if not provided."""
        if feature_vector is None:
            feature_vector = _features_from_dict(mesh_features)

        record = {
            # Identity
            "schema_version":      SCHEMA_VERSION,
            "part_name":           part_name,
            "model_id":            model_id,
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "mesh_hash":           _hash_file(mesh_path),
            "feature_vector":      feature_vector,

            # Raw geometry
            "mesh_features":       json.dumps(mesh_features),
            "symmetry_class":      symmetry_class,
            "arrangement_context": arrangement_context,

            # Result
            "best_coverage":       float(best_coverage),
            "best_mean_time":      float(best_mean_time),
            "best_params":         json.dumps(best_params),
            "vsd_threshold_mm":    float(vsd_threshold_mm),
            "phase_reached":       int(phase_reached),
            "n_rounds_to_converge": int(n_rounds_to_converge),
            "reasoning_trace":     reasoning_trace,
            "trajectory":          json.dumps(trajectory or []),

            # Baseline
            "rag_version":              rag_version,
            "warmstart_params":         json.dumps(warmstart_params or {}),
            "warmstart_coverage":       float(warmstart_coverage),
            "warmstart_mean_time":      float(warmstart_mean_time),
            "fell_back_to_warmstart":   bool(fell_back_to_warmstart),
            "retrieval_similarity":     float(retrieval_similarity),
            "constraint_retries":       int(constraint_retries),
        }

        if self._table is None:
            self._table = self._db.create_table("experience", data=[record])
        else:
            self._table.add([record])

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def query_similar(
        self,
        feature_vector: list[float],
        k: int = 8,
        symmetry_class: Optional[str] = None,
    ) -> list[dict]:
        """Return up to k records nearest to feature_vector.

        Results are sorted ascending by distance (least similar first) so the
        caller can place the MOST similar example LAST in the LLM prompt
        (Lost-in-the-Middle pattern).
        """
        if self._table is None:
            return []

        try:
            q = (
                self._table
                .search(feature_vector, vector_column_name="feature_vector")
                .limit(k * 3)   # over-fetch then filter
            )
            if symmetry_class is not None:
                q = q.where(f"symmetry_class = '{symmetry_class}'")
            rows = q.to_list()
        except Exception:
            rows = []

        # Decode JSON columns and trim to k
        results = []
        for row in rows[:k]:
            results.append(_decode_row(row))

        # Sort ascending distance → most similar LAST (caller slices [-1])
        results.sort(key=lambda r: r.get("_distance", 0.0))
        return results

    def query_by_part(self, part_name: str) -> list[dict]:
        """Return all records for a given part_name, newest first."""
        if self._table is None:
            return []
        try:
            rows = (
                self._table
                .search()
                .where(f"part_name = '{part_name}'")
                .to_list()
            )
            rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
            return [_decode_row(r) for r in rows]
        except Exception:
            return []

    def count(self) -> int:
        if self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

def _features_from_dict(mesh_features: dict) -> list[float]:
    """Build normalised ~19-d feature vector from mesh_features dict.

    Mirrors get_feature_vector() in mesh_analysis.py but works from the
    serialised dict so experience_bank has no hard dependency on mesh_analysis.
    """
    sym_map = {
        "ASYMMETRIC": [1,0,0,0,0,0,0],
        "C2":         [0,1,0,0,0,0,0],
        "C3":         [0,0,1,0,0,0,0],
        "C4":         [0,0,0,1,0,0,0],
        "C6":         [0,0,0,0,1,0,0],
        "SO2":        [0,0,0,0,0,1,0],
        "SO3":        [0,0,0,0,0,0,1],
    }
    sym_class = mesh_features.get("symmetry_class", "ASYMMETRIC")
    onehot = sym_map.get(sym_class, [0,0,0,0,0,0,0])

    return [
        float(mesh_features.get("diameter_mm", 0.0)  / 1000.0),
        float(mesh_features.get("aspect_ratio", 1.0) / 10.0),
        float(mesh_features.get("flatness_ratio", 1.0) / 10.0),
        float(mesh_features.get("normal_concentration", 0.0)),
        float(1.0 if mesh_features.get("prefer_edge", False) else 0.0),
        float(mesh_features.get("convexity", 1.0)),
        float(mesh_features.get("curvature_mean", 0.0)),
        float(mesh_features.get("curvature_std", 0.0)),
        float(mesh_features.get("n_flat_clusters", 0) / 10.0),
        float(1.0 if mesh_features.get("has_holes", False) else 0.0),
        float(mesh_features.get("surface_area_mm2", 0.0) / 1e6),
        float(mesh_features.get("volume_mm3", 0.0) / 1e6),
    ] + [float(x) for x in onehot]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_file(path: str) -> str:
    """SHA-256 of file content; returns empty string if file not found."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _decode_row(row: dict) -> dict:
    """Decode JSON-serialised columns back to Python objects."""
    r = dict(row)
    for col in ("mesh_features", "best_params", "trajectory", "warmstart_params"):
        if col in r and isinstance(r[col], str):
            try:
                r[col] = json.loads(r[col])
            except Exception:
                pass
    return r


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bank = ExperienceBank(db_path=os.path.join(tmp, "test.lance")).open()

        feats = {
            "diameter_mm": 120.0, "aspect_ratio": 2.5, "flatness_ratio": 3.0,
            "normal_concentration": 0.6, "prefer_edge": False,
            "convexity": 0.9, "curvature_mean": 0.05, "curvature_std": 0.02,
            "n_flat_clusters": 4, "has_holes": False,
            "surface_area_mm2": 45000.0, "volume_mm3": 120000.0,
            "symmetry_class": "C2",
        }
        fvec = _features_from_dict(feats)

        bank.insert(
            part_name="part_A", model_id="m001", mesh_path="/tmp/nonexistent.ply",
            mesh_features=feats, symmetry_class="C2",
            arrangement_context="random pile, 30 parts",
            best_coverage=0.88, best_mean_time=0.45,
            best_params={"refStep": 8, "distQuantification": 1.0},
            feature_vector=fvec,
        )
        bank.insert(
            part_name="part_B", model_id="m002", mesh_path="/tmp/nonexistent2.ply",
            mesh_features=feats, symmetry_class="C2",
            arrangement_context="tray, 10 parts",
            best_coverage=0.91, best_mean_time=0.62,
            best_params={"refStep": 10, "distQuantification": 0.75},
            feature_vector=[x * 1.05 for x in fvec],  # slightly different
        )

        print(f"Records inserted: {bank.count()}")
        results = bank.query_similar(fvec, k=5)
        print(f"Query returned {len(results)} result(s)")
        for r in results:
            print(f"  part={r['part_name']}  cov={r['best_coverage']:.2f}  sym={r['symmetry_class']}")

        by_part = bank.query_by_part("part_A")
        print(f"query_by_part('part_A'): {len(by_part)} record(s)")
        print("Smoke test PASSED")
