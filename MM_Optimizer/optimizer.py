"""
optimizer.py  —  Heuristic-Seeded Hierarchical Coordinate Descent
-----------------------------------------------------------------
Implements plan_1_hierarchical_decent.md with two optional strategies
from the transposition+look-ahead plan:

  Strategy 1 (Transposition Table) : EvalCache deduplicates evaluations
  Strategy 2 (Two-Pass Multi-Fidelity): cheap M_small pre-screen, full M_full
                                        for top-K survivors only
  Strategy 3 (Phase-Level Gates)    : abort / skip phases when coverage too low

Both strategies are detachable:
  ENABLE_CACHE    = False  →  bypasses EvalCache entirely
  ENABLE_TWO_PASS = False  →  all sweeps run at M_full directly

CLI:
  python optimizer.py --part 25333MB000 [--dry_run] [--scenes_dir PATH]
                       [--m_full N] [--no_cache] [--no_two_pass]
                       [--export_best]
"""

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, ".."))
for _p in [_ROOT, _DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mm_adapter.mm_adapter    import MechVisionClient, VisionRunError
from mm_adapter.mm_dataclasses import (CoarseMatchingV2, FineMatchingLite,
                                       EasyCreateStringList)
from MM_Optimizer.eval_cache        import EvalCache
from MM_Optimizer.mesh_analysis     import analyze_mesh, load_reference_pcd, WarmStart
from MM_Optimizer.optimizer_utils   import (list_synthetic_scenes,
                                  read_gt_pose_from_ply)
import MM_Optimizer.search_config as SC

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Top-level feature flags  (Strategy detach points)
# ─────────────────────────────────────────────────────────────────────────────
ENABLE_CACHE     = True   # Strategy 1 — set False to disable transposition table
ENABLE_TWO_PASS  = True   # Strategy 2 — set False to evaluate all candidates at M_FULL

# ─────────────────────────────────────────────────────────────────────────────
# MechVision project / path constants
# ─────────────────────────────────────────────────────────────────────────────
PROJ_NAME            = "CAD_Match"
MM_MODEL_ROOT        = os.path.join(_DIR, "CAD_Match", "resource", "3d_matching")
RESULTS_DIR          = os.path.join(_DIR, "results")
OPTIMIZER_UTILS_PATH = os.path.join(_DIR, "optimizer_utils.py")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    score:           float           # lower=better  mean_time/SCORE_TIME_NORM + (1-cov)*SCORE_COV_NORM
    coverage:        float           # mean fraction of instances detected across M scenes
    mean_time:       float           # mean coarse+fine cycle time (s)
    per_scene:       List[dict] = field(default_factory=list)  # per-scene diagnostics
    n_scenes:        int   = 0
    config:          dict  = field(default_factory=dict)
    score_quality:   float = 0.0    # 1 - score/SCORE_WORST_CASE ∈ [0,1]; higher=better
    score_time_term: float = 0.0    # mean_time / SCORE_TIME_NORM
    score_cov_term:  float = 0.0    # (1 - coverage) * SCORE_COV_NORM

    def soft_score(self) -> float:
        """Mean position error of passing instances — tiebreaker in Phase 5."""
        passing = [
            e for s in self.per_scene
            for e in s.get("pos_errors", [])
            if e is not None
        ]
        return float(np.mean(passing)) if passing else float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# Pose matching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_error_deg(q_pred, q_gt):
    """Rotation error in degrees between two quaternions [w, x, y, z]."""
    # Ensure unit quaternions
    q1 = np.array(q_pred, dtype=float)
    q2 = np.array(q_gt,   dtype=float)
    q1 /= (np.linalg.norm(q1) + 1e-12)
    q2 /= (np.linalg.norm(q2) + 1e-12)
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, dot)
    return math.degrees(2 * math.acos(dot))


def match_poses_to_gt(returned_poses, gt_poses):
    """Threshold-gated nearest-neighbour pose matching.

    Parameters
    ----------
    returned_poses : list of [x, y, z, qw, qx, qy, qz]
    gt_poses       : list of [x, y, z, qw, qx, qy, qz]

    Returns
    -------
    results : list of (pos_err_m, ang_err_deg) or (None, None) if unmatched
    """
    results = [(None, None)] * len(gt_poses)
    used    = set()
    for gt_idx, gt in enumerate(gt_poses):
        gt_pos = np.array(gt[:3], dtype=float)
        candidates = []
        for i, p in enumerate(returned_poses):
            if i in used:
                continue
            dist = float(np.linalg.norm(np.array(p[:3], dtype=float) - gt_pos))
            candidates.append((dist, i, p))
        if candidates:
            dist, idx, pred = min(candidates, key=lambda x: x[0])
            if dist < SC.POS_THRESH_MATCH:
                used.add(idx)
                ang_err = _rotation_error_deg(pred[3:7], gt[3:7])
                results[gt_idx] = (dist, ang_err)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer class
# ─────────────────────────────────────────────────────────────────────────────

class Optimizer:
    """Hierarchical coordinate descent optimizer for MechVision parameters.

    Parameters
    ----------
    part_name    : Part identifier (matches model dir and scene dir names).
    client       : Connected MechVisionClient.
    project_id   : Integer project ID for PROJ_NAME.
    scene_groups : List[List[str]] — one inner list per scene_MMMMM directory.
                   Each inner list contains all sample_*.ply paths for that
                   bin capture (N instances per scene).
    warm_start   : WarmStart from mesh_analysis.
    cache        : EvalCache instance (or None to disable).
    use_two_pass : Enable two-pass multi-fidelity (Strategy 2).
    dry_run      : Build param dicts but do not call MechVision.
    """

    def __init__(self,
                 part_name:    str,
                 client:       MechVisionClient,
                 project_id:   int,
                 scene_groups: List[List[str]],
                 warm_start:   WarmStart,
                 cache:        Optional[EvalCache] = None,
                 use_two_pass: bool = True,
                 dry_run:      bool = False):

        self.part_name    = part_name
        self.client       = client
        self.project_id   = project_id
        self.scene_groups = scene_groups   # List[List[str]] — one group per M-scene
        self.ws           = warm_start
        self.cache        = cache
        self.use_two_pass = use_two_pass
        self.dry_run      = dry_run

        self._model_root   = MM_MODEL_ROOT
        self._n_evals      = 0   # total MechVision calls
        self._n_gate_stops = 0   # phase gate activations

        # Best state (updated throughout)
        self._best_coarse: dict = {}
        self._best_fine:   dict = {}
        self._best_result: Optional[EvalResult] = None
        self._best_regime: dict = {}

        # GT cache — scene_dir → gt_poses (avoids re-reading PLY headers)
        self._scene_gt_cache: Dict[str, list] = {}

        os.makedirs(RESULTS_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Scene management
    # ─────────────────────────────────────────────────────────────────────

    def _prepare_scene(self, scene_plys: List[str]) -> Tuple[str, List]:
        """Return (scene_dir, gt_poses) for an M-scene's PLY list.

        scene_dir is the directory containing all PLYs — passed directly to
        MechVision's read_synthetic, which loads N individual clouds from it.
        gt_poses is a list of [x, y, z, qw, qx, qy, qz] for each PLY.
        Results are cached so PLY headers are read only once per directory.
        """
        scene_dir = os.path.dirname(scene_plys[0])
        if scene_dir not in self._scene_gt_cache:
            gt_poses = [read_gt_pose_from_ply(p, scalar_first=True)
                        for p in scene_plys]
            self._scene_gt_cache[scene_dir] = gt_poses
        return scene_dir, self._scene_gt_cache[scene_dir]

    def _sample_scenes(self, n: int) -> List[Tuple[str, List]]:
        """Sample n M-scenes from the available scene groups.

        Each M-scene is a full bin capture (all N instances from one
        scene_MMMMM directory).  Groups are drawn from a randomly shuffled
        order; wraps around if n > number of available scenes.
        """
        groups  = self.scene_groups
        indices = list(range(len(groups)))
        random.shuffle(indices)
        scenes = []
        for g in range(n):
            group = groups[indices[g % len(indices)]]
            scenes.append(self._prepare_scene(group))
        return scenes

    # ─────────────────────────────────────────────────────────────────────
    # Parameter dict construction
    # ─────────────────────────────────────────────────────────────────────

    def _make_params_dict(self, coarse_params: dict, fine_params: dict,
                          scene_dir: str) -> dict:
        """Build the params_dict for client.set_params().

        Parameters
        ----------
        scene_dir : path to the scene_MMMMM directory containing all sample_*.ply
                    files for this M-scene.  Passed to Scene_Path.strings;
                    read_synthetic loads all N PLYs from it and returns N
                    individual point clouds — one per instance in the bin.
        """
        coarse_mode = coarse_params.get("registrationMode", 0.0)
        fine_mode   = fine_params.get("registrationMode",   0.0)
        coarse_type = "edge" if coarse_mode == 1.0 else "surface"
        fine_type   = "edge" if fine_mode   == 1.0 else "surface"

        def model_path(mtype):
            return os.path.join(self._model_root,
                                f"{self.part_name}_{mtype}")

        def geo_path(mtype):
            return os.path.join(model_path(mtype), "geo_center.json")

        def ply_path(mtype):
            return os.path.join(model_path(mtype),
                                f"{self.part_name}_{mtype}.ply")

        scene = EasyCreateStringList(
            strings=(scene_dir, "string", "")
        )

        # Pre_Segmentation: read_synthetic loads all sample_*.ply from scene_dir
        # and returns a list of N individual [x,y,z,nx,ny,nz,0] arrays.
        # MechVision coarse matching processes each cloud independently, yielding
        # N × outputNum hypotheses.  Fine matching (candidateTopNum=1) reduces
        # this back to N final poses — one per instance in the bin.
        pre_seg_params = {
            "scriptFilePath": (OPTIMIZER_UTILS_PATH, "string", ""),
            "funcName":       ("read_synthetic",     "string", ""),
        }

        coarse = CoarseMatchingV2(
            name="Coarse_Match_Synthetics",
            modelSelection=(f"{self.part_name}_{coarse_type}", "string", ""),
            modelFileName=(ply_path(coarse_type), "string", ""),
            geoCenterFileName=(geo_path(coarse_type), "string", ""),
        )
        # Apply all coarse_params overrides
        for k, v in coarse_params.items():
            if k == "registrationMode":
                coarse.registrationMode = (str(v), "double", "")
            elif k == "refStep":
                coarse.refStep = (str(v), "double", "")
            elif k == "distQuantification":
                coarse.distQuantification = (str(v), "double", "")
            elif k == "angleQuantification":
                coarse.angleQuantification = (str(v), "double", "")
            elif k == "maxNumOfPointPairsPerFeature":
                coarse.maxNumOfPointPairsPerFeature = (str(v), "double", "")
            elif k == "maxVoteRatio":
                coarse.maxVoteRatio = (str(v), "double", "")
            elif k == "referredStep":
                coarse.referredStep = (str(v), "double", "")
            elif k == "useDistanceNMS":
                coarse.useDistanceNMS = (str(v), "bool", "")
            elif k == "filterCandidatePoseByAxis":
                coarse.filterCandidatePoseByAxis = (str(v), "bool", "")
            elif k == "angleThreshold":
                coarse.angleThreshold = (str(v), "double", "")
            elif k == "outputNum":
                coarse.outputNum = (str(v), "double", "")
            elif k == "minVoxelLength":
                coarse.minVoxelLength = (str(v), "double", "m")
            elif k == "maxVoxelLength":
                coarse.maxVoxelLength = (str(v), "double", "m")

        fine = FineMatchingLite(
            name="Fine_Match_Synthetics",
            modelSelection=(f"{self.part_name}_{fine_type}", "string", ""),
            modelFileName=(ply_path(fine_type), "string", ""),
            geoCenterFileName=(geo_path(fine_type), "string", ""),
        )
        for k, v in fine_params.items():
            if k == "registrationMode":
                fine.registrationMode = (str(v), "double", "")
            elif k == "operationApproach":
                fine.operationApproach = (str(v), "double", "")
            elif k == "deviationCorrectionCapacity":
                fine.deviationCorrectionCapacity = (str(v), "double", "")
            elif k == "onlyConsiderVisibleSurfaceOfModel":
                fine.onlyConsiderVisibleSurfaceOfModel = (str(v), "bool", "")
            elif k == "considerErrorofNormalAngles":
                fine.considerErrorofNormalAngles = (str(v), "bool", "")
            elif k == "scoreLevel":
                fine.scoreLevel = (str(v), "double", "")
            elif k == "confidenceThreshold":
                fine.confidenceThreshold = (str(v), "double", "")
            elif k == "rotationStrategy":
                fine.rotationStrategy = (str(v), "double", "")
            elif k == "angleStep":
                fine.angleStep = (str(v), "double", "")
            elif k == "minAngle":
                fine.minAngle = (str(v), "double", "")
            elif k == "maxAngle":
                fine.maxAngle = (str(v), "double", "")

        return {
            scene.name:      scene.to_step_params(),
            "Pre_Segmentation": pre_seg_params,
            coarse.name:     coarse.to_step_params(),
            fine.name:       fine.to_step_params(),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Single-scene evaluation
    # ─────────────────────────────────────────────────────────────────────

    def _run_one_scene(self, coarse_params, fine_params,
                       scene_dir, gt_poses,
                       pos_thresh, ang_thresh):
        """Run MechVision on one M-scene (N instances). Returns per-scene dict.

        MechVision receives N individual clouds via read_synthetic and returns
        up to N poses (candidateTopNum=1 per cloud).  Coverage is the fraction
        of the N instances that pass both pos_thresh and ang_thresh.

        Empty fine_poses = no detection = instance_coverage 0.0 (not an error).
        VisionRunError (noCloudInRoi) = treated as no-detection.
        """
        n_gt = len(gt_poses)
        if self.dry_run:
            return {"ok": True, "instance_coverage": 1.0,
                    "pos_errors": [0.001] * n_gt,
                    "ang_errors": [1.0]   * n_gt,
                    "coarse_time_s": 0.05, "fine_time_s": 0.03}

        params_dict = self._make_params_dict(coarse_params, fine_params, scene_dir)
        self.client.set_params(self.project_id, params_dict)
        self._n_evals += 1

        try:
            result = self.client.run_vision(self.project_id)
        except VisionRunError as e:
            log.debug(f"run_vision returned error (scoring as no-detection): {e}")
            return {"ok": False, "instance_coverage": 0.0,
                    "pos_errors": [None] * n_gt,
                    "ang_errors": [None] * n_gt,
                    "coarse_time_s": 0.0, "fine_time_s": 0.0}

        fine_poses = result.get("fine_poses", [])
        matched    = match_poses_to_gt(fine_poses, gt_poses)
        pos_errors = [m[0] for m in matched]
        ang_errors = [m[1] for m in matched]

        n_pass = sum(
            1 for p, a in matched
            if p is not None and p < pos_thresh and a < ang_thresh
        )
        instance_coverage = n_pass / max(1, n_gt)

        return {
            "ok":               instance_coverage == 1.0,
            "instance_coverage": instance_coverage,
            "pos_errors":        pos_errors,
            "ang_errors":        ang_errors,
            "coarse_time_s":     result.get("coarse_time_s", 0.0),
            "fine_time_s":       result.get("fine_time_s",   0.0),
        }

    # ─────────────────────────────────────────────────────────────────────
    # evaluate_config  (Strategy 1: cache wrapper)
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_config(self,
                        coarse_params: dict,
                        fine_params:   dict,
                        scenes:        List[Tuple[str, List]],
                        pos_thresh:    float = SC.POS_THRESH_TIGHT,
                        ang_thresh:    float = SC.ANG_THRESH_TIGHT
                        ) -> EvalResult:
        """Evaluate (coarse_params, fine_params) on the given scenes.

        Checks EvalCache first (Strategy 1). On miss, runs MechVision and stores.
        """
        combined    = {"coarse": coarse_params, "fine": fine_params}
        scene_paths = [ply_path for ply_path, _ in scenes]

        # Cache is only valid for live runs — dry_run mock values must never
        # be stored, as they would contaminate future live evaluations.
        use_cache = (self.cache is not None) and (not self.dry_run)

        if use_cache:
            # Include thresholds in key: Phase 2/3 (loose ang) vs Phase 5/6 (tight)
            # produce different coverage values for same config — must not share.
            cache_key_data = {**combined,
                              "_pos": pos_thresh, "_ang": ang_thresh}
            key = self.cache.make_key(cache_key_data, scene_paths)
            hit = self.cache.get(key)
            if hit is not None:
                return EvalResult(**hit)

        per_scene = []
        for scene_dir, gt_poses in scenes:
            s = self._run_one_scene(coarse_params, fine_params, scene_dir,
                                    gt_poses, pos_thresh, ang_thresh)
            per_scene.append(s)

        coverage  = float(np.mean([s["instance_coverage"] for s in per_scene]))
        mean_time = float(np.mean([s["coarse_time_s"] + s["fine_time_s"]
                                   for s in per_scene]))
        _t    = mean_time / SC.SCORE_TIME_NORM
        _c    = (1.0 - coverage) * SC.SCORE_COV_NORM
        score = _t + _c

        er = EvalResult(
            score           = score,
            coverage        = coverage,
            mean_time       = mean_time,
            per_scene       = per_scene,
            n_scenes        = len(per_scene),
            config          = combined,
            score_quality   = 1.0 - score / SC.SCORE_WORST_CASE,
            score_time_term = _t,
            score_cov_term  = _c,
        )

        if use_cache:
            self.cache.put(key, asdict(er))

        return er

    # ─────────────────────────────────────────────────────────────────────
    # evaluate_phase_sweep  (Strategy 2: two-pass multi-fidelity)
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_phase_sweep(self,
                             coarse_variants: List[dict],
                             fine_variants:   List[dict],
                             pos_thresh: float = SC.POS_THRESH_TIGHT,
                             ang_thresh: float = SC.ANG_THRESH_TIGHT,
                             label: str = "") -> EvalResult:
        """Evaluate all (coarse, fine) variant pairs with optional two-pass.

        coarse_variants and fine_variants are parallel lists of equal length —
        each index represents one candidate config to evaluate.

        Strategy 2: if ENABLE_TWO_PASS and len(candidates) > TWO_PASS_MIN_CANDIDATES:
          Pass 1: evaluate all candidates on M_SMALL scenes
          Pass 2: full M_FULL evaluation on top-K survivors from Pass 1
        Otherwise: evaluate all candidates at M_FULL directly.

        Returns the best EvalResult from Pass 2 (or direct eval).
        """
        n_candidates = len(coarse_variants)
        assert len(fine_variants) == n_candidates

        use_tp = (self.use_two_pass and
                  n_candidates > SC.TWO_PASS_MIN_CANDIDATES)

        # ---- Pass 1 (cheap screening) ----
        if use_tp:
            scenes_small = self._sample_scenes(SC.M_SMALL)
            pass1_results = []
            for i, (cp, fp) in enumerate(zip(coarse_variants, fine_variants)):
                r = self.evaluate_config(cp, fp, scenes_small,
                                         pos_thresh, ang_thresh)
                log.debug(f"  [{label}] P1 cand {i:2d}: "
                          f"cov={r.coverage:.2f} score={r.score:.1f}")
                pass1_results.append((r, i))

            # Top-K survivors by score; early-exit if top-1 already meets target
            pass1_results.sort(key=lambda x: x[0].score)
            if pass1_results[0][0].coverage >= SC.TARGET_COVERAGE:
                survivors = pass1_results[:1]
                log.info(f"[{label}] early-exit: top-1 cov="
                         f"{pass1_results[0][0].coverage:.2f} >= TARGET"
                         f" — skipping costlier candidates")
            else:
                survivors = pass1_results[:SC.K_SURVIVORS]
            log.info(f"[{label}] two-pass: {n_candidates} → {len(survivors)} survivors")
            coarse_variants_p2 = [coarse_variants[i] for _, i in survivors]
            fine_variants_p2   = [fine_variants[i]   for _, i in survivors]
        else:
            coarse_variants_p2 = coarse_variants
            fine_variants_p2   = fine_variants

        # ---- Pass 2 (full evaluation) ----
        scenes_full = self._sample_scenes(SC.M_FULL)
        pass2_results = []
        for i, (cp, fp) in enumerate(zip(coarse_variants_p2, fine_variants_p2)):
            r = self.evaluate_config(cp, fp, scenes_full,
                                     pos_thresh, ang_thresh)
            log.info(f"  [{label}] P2 cand {i:2d}: "
                     f"cov={r.coverage:.2f}  time={r.mean_time:.3f}s  "
                     f"score={r.score:.1f}")
            pass2_results.append(r)

        pass2_results.sort(key=lambda r: r.score)
        return pass2_results[0]

    # ─────────────────────────────────────────────────────────────────────
    # Phase gate  (Strategy 3)
    # ─────────────────────────────────────────────────────────────────────

    def _gate(self, gate_key: str, coverage: float) -> bool:
        """Return True if optimization should continue past this gate.

        Logs warnings and increments stop counter if gate fires.
        """
        cfg = SC.PHASE_GATES.get(gate_key)
        if cfg is None:
            return True
        threshold, action = cfg
        if coverage < threshold:
            self._n_gate_stops += 1
            log.warning(f"GATE [{gate_key}]: coverage={coverage:.2f} < {threshold} "
                        f"→ {action}")
            return False
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Phase 0 — Already done: warm start is passed in via __init__
    # ─────────────────────────────────────────────────────────────────────

    def _default_coarse(self) -> dict:
        return {
            "registrationMode":              0.0,
            "refStep":                       SC.REFSTEP_BOUNDS[1] // 2,
            "distQuantification":            self.ws.distQuantification,
            "angleQuantification":           self.ws.angleQuantification,
            "maxNumOfPointPairsPerFeature":  self.ws.maxNumOfPointPairsPerFeature,
            "maxVoteRatio":                  0.5,
            "referredStep":                  1,
            "useDistanceNMS":                True,
            "filterCandidatePoseByAxis":     True,
            "angleThreshold":                135,
            "outputNum":                     1,
        }

    def _default_fine(self) -> dict:
        return {
            "registrationMode":                   0.0,
            "operationApproach":                  1.0,
            "deviationCorrectionCapacity":         0.0,
            "onlyConsiderVisibleSurfaceOfModel":   False,
            "considerErrorofNormalAngles":         False,
            "scoreLevel":                         0.0,
            "confidenceThreshold":                0.0,
            "candidateTopNum":                    1,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Phase 1 — Regime gate
    # ─────────────────────────────────────────────────────────────────────

    def phase1_regime_gate(self) -> List[dict]:
        """Test A/B/C/D regime combos. Return passing regimes sorted by coverage."""
        log.info("=" * 60)
        log.info("PHASE 1 — Regime Gate")

        has_edge = os.path.isdir(os.path.join(self._model_root,
                                              f"{self.part_name}_edge"))

        # Order based on geometry hint
        regimes = sorted(SC.PHASE1_REGIMES,
                         key=lambda r: (r["needs_edge"] and not self.ws.prefer_edge,
                                        r["id"]))

        passing = []
        for regime in regimes:
            if regime["needs_edge"] and not has_edge:
                log.info(f"  Regime {regime['id']}: skip (no edge model)")
                continue

            cp = self._default_coarse()
            fp = self._default_fine()
            cp["registrationMode"] = regime["coarse_mode"]
            fp["registrationMode"] = regime["fine_mode"]

            scenes = self._sample_scenes(SC.M_FULL)
            # Phase 1 regime gate: position-only scoring.  Parts with rotational
            # symmetry may return orientation-flipped but geometrically valid poses.
            # Orientation accuracy is optimised in Phase 3 — here we only check
            # whether coarse+fine can localise the part within POS_THRESH_LOOSE.
            r = self.evaluate_config(cp, fp, scenes,
                                     SC.POS_THRESH_LOOSE, SC.ANG_THRESH_REGIME_GATE)
            log.info(f"  Regime {regime['id']}: cov={r.coverage:.2f}  "
                     f"time={r.mean_time:.3f}s")

            if r.coverage >= SC.PHASE1_COVERAGE_GATE:
                passing.append({**regime, "coverage": r.coverage,
                                 "coarse": cp, "fine": fp})

        if not passing:
            log.error("PHASE 1: No regime passes. Part may be un-tunable.")
            return []

        passing.sort(key=lambda x: -x["coverage"])
        log.info(f"PHASE 1 done: {len(passing)} passing regimes — "
                 f"best = {passing[0]['id']} (cov={passing[0]['coverage']:.2f})")

        if not self._gate("after_phase1", passing[0]["coverage"]):
            return []
        return passing

    # ─────────────────────────────────────────────────────────────────────
    # Phase 2 — Coarse coordinate descent
    # ─────────────────────────────────────────────────────────────────────

    def phase2_coarse_cd(self, regime: dict) -> EvalResult:
        """Full coarse coordinate descent for one regime.

        Uses position-only scoring (ANG_THRESH_REGIME_GATE) throughout.
        Phase 2 optimises coarse localisation quality — orientation accuracy
        is addressed in Phase 4 via symmetry parameters.
        """
        log.info("=" * 60)
        log.info(f"PHASE 2 — Coarse CD  (regime {regime['id']})")

        coarse = copy.deepcopy(regime["coarse"])
        fine   = copy.deepcopy(regime["fine"])
        ang    = SC.ANG_THRESH_REGIME_GATE   # position-only for Phases 1–3

        # Baseline
        scenes  = self._sample_scenes(SC.M_FULL)
        baseline = self.evaluate_config(coarse, fine, scenes,
                                        SC.POS_THRESH_TIGHT, ang)
        log.info(f"  Phase 2 baseline: cov={baseline.coverage:.2f}")
        best_result = baseline

        for round_idx in range(SC.PHASE2_MAX_ROUNDS):
            log.info(f"  -- Round {round_idx + 1} --")
            improved = False

            # ---- Phase 2a: Joint quantization grid ----
            best_2a = self._phase2a_joint_grid(coarse, fine, ang)
            if best_2a.score < best_result.score:
                best_result = best_2a
                coarse = copy.deepcopy(best_2a.config["coarse"])
                improved = True
            log.info(f"  2a result: cov={best_result.coverage:.2f}")

            if not self._gate("after_phase2a", best_result.coverage):
                log.warning("  Skipping Phase 2b — PPF quantization too poor.")
                return best_result

            # ---- Phase 2b: Remaining params ----
            for param_name, param in SC.PHASE2B_PARAMS.items():
                candidates   = param["candidates"]
                is_edge_only = param["edge_only"]
                if is_edge_only and regime["coarse_mode"] != 1.0:
                    continue

                # refStep >= referredStep: filter candidates to those <= locked refStep
                if param_name == "referredStep":
                    locked_refstep = coarse.get("refStep", float("inf"))
                    candidates = [c for c in candidates if c <= locked_refstep]
                    if not candidates:
                        continue

                # Expand runtime-computed candidates
                if candidates is None and param_name == "maxNumOfPointPairsPerFeature":
                    warm_pairs = self.ws.maxNumOfPointPairsPerFeature
                    candidates = [max(1, int(warm_pairs * s))
                                  for s in SC.PHASE2B_PAIRS_SCALES]

                # Joint pair sweep — min/max must be set together to keep min < max
                if param_name == "voxelLengthRange":
                    best_param = self._sweep_voxel_range(coarse, fine, ang_thresh=ang)
                    if best_param.score < best_result.score:
                        best_result = best_param
                        coarse = copy.deepcopy(best_param.config["coarse"])
                        improved = True
                    continue

                # Boolean sweep: skip two-pass (only 2 options)
                if len(candidates) <= 2:
                    best_param = self._sweep_param_direct(
                        param_name, candidates, coarse, fine,
                        is_coarse=True, ang_thresh=ang)
                else:
                    best_param = self._sweep_param(
                        param_name, candidates, coarse, fine,
                        is_coarse=True, label=f"2b-{param_name}",
                        ang_thresh=ang)
                if best_param.score < best_result.score:
                    best_result = best_param
                    coarse = copy.deepcopy(best_param.config["coarse"])
                    improved = True

            if not improved:
                log.info(f"  Round {round_idx + 1}: converged (no improvement)")
                break

        log.info(f"PHASE 2 done: cov={best_result.coverage:.2f}  "
                 f"time={best_result.mean_time:.3f}s")
        return best_result

    def _phase2a_joint_grid(self, coarse, fine,
                            ang_thresh=SC.ANG_THRESH_TIGHT) -> EvalResult:
        """Joint grid over (refStep × distQuantification).

        distQ is MechVision's unitless factor — swept independently of refStep.
        Grid size = len(REFSTEP_SCALES) × len(DISTQ_VALUES).
        """
        coarse_variants, fine_variants = [], []
        for ref in SC.PHASE2A_REFSTEP_VALUES:
            for distq in SC.PHASE2A_DISTQ_VALUES:
                cp   = copy.deepcopy(coarse)
                cp["refStep"]            = ref
                cp["distQuantification"] = distq
                coarse_variants.append(cp)
                fine_variants.append(copy.deepcopy(fine))

        best = self.evaluate_phase_sweep(coarse_variants, fine_variants,
                                         ang_thresh=ang_thresh,
                                         label="2a-grid")
        log.info(f"  2a best: refStep={best.config['coarse']['refStep']}  "
                 f"distQ={best.config['coarse']['distQuantification']:.2f}  "
                 f"cov={best.coverage:.2f}")
        return best

    def _sweep_voxel_range(self, coarse, fine,
                           ang_thresh=SC.ANG_THRESH_TIGHT) -> EvalResult:
        """Sweep voxelLengthRange as geometry-derived (min, max) pairs.

        Uses PHASE2B_VOXEL_SCALES × warm voxel lengths, preserving the
        1:4 min:max ratio at every scale so min < max is always guaranteed.
        """
        coarse_variants, fine_variants = [], []
        for scale in SC.PHASE2B_VOXEL_SCALES:
            cp = copy.deepcopy(coarse)
            cp["minVoxelLength"] = max(0.1, round(self.ws.minVoxelLength_mm * scale, 2))
            cp["maxVoxelLength"] = max(0.2, round(self.ws.maxVoxelLength_mm * scale, 2))
            coarse_variants.append(cp)
            fine_variants.append(copy.deepcopy(fine))
        return self.evaluate_phase_sweep(coarse_variants, fine_variants,
                                         ang_thresh=ang_thresh,
                                         label="2b-voxelLengthRange")

    def _sweep_param(self, param_name, candidates, coarse, fine,
                     is_coarse=True, label="",
                     ang_thresh=SC.ANG_THRESH_TIGHT) -> EvalResult:
        """Sweep one parameter with two-pass strategy."""
        coarse_variants, fine_variants = [], []
        for val in candidates:
            cp = copy.deepcopy(coarse)
            fp = copy.deepcopy(fine)
            if is_coarse:
                cp[param_name] = val
            else:
                fp[param_name] = val
            coarse_variants.append(cp)
            fine_variants.append(fp)
        return self.evaluate_phase_sweep(coarse_variants, fine_variants,
                                         ang_thresh=ang_thresh,
                                         label=label or param_name)

    def _sweep_param_direct(self, param_name, candidates, coarse, fine,
                            is_coarse=True,
                            ang_thresh=SC.ANG_THRESH_TIGHT) -> EvalResult:
        """Sweep all candidates at M_FULL (no two-pass — for booleans / tiny sets)."""
        scenes = self._sample_scenes(SC.M_FULL)
        best   = None
        for val in candidates:
            cp = copy.deepcopy(coarse)
            fp = copy.deepcopy(fine)
            if is_coarse:
                cp[param_name] = val
            else:
                fp[param_name] = val
            r = self.evaluate_config(cp, fp, scenes,
                                     SC.POS_THRESH_TIGHT, ang_thresh)
            log.debug(f"  {param_name}={val}: cov={r.coverage:.2f}")
            if best is None or r.score < best.score:
                best = r
        return best

    # ─────────────────────────────────────────────────────────────────────
    # Phase 3 — Fine coordinate descent
    # ─────────────────────────────────────────────────────────────────────

    def phase3_fine_cd(self, coarse: dict, fine: dict,
                       phase2_result: EvalResult) -> EvalResult:
        """Fine-matching parameter coordinate descent.

        Uses position-only scoring through Phase 3 — symmetry params are not
        yet set, so orientation may be flipped. Phase 4 handles symmetry.
        """
        log.info("=" * 60)
        log.info("PHASE 3 — Fine CD")
        ang = SC.ANG_THRESH_REGIME_GATE   # position-only, same as Phases 1–2

        # Strategy 3 look-ahead: narrow operationApproach candidates
        pos_errors = [e for s in phase2_result.per_scene
                      for e in s.get("pos_errors", []) if e is not None]
        median_err = float(np.median(pos_errors)) if pos_errors else 0.01
        approach_candidates = SC.phase3_approach_candidates(median_err)
        log.info(f"  median coarse pos err = {median_err*1e3:.2f} mm → "
                 f"operationApproach candidates = {approach_candidates}")

        scenes   = self._sample_scenes(SC.M_FULL)
        best_res = self.evaluate_config(coarse, fine, scenes,
                                        SC.POS_THRESH_TIGHT, ang)
        log.info(f"  Phase 3 baseline: cov={best_res.coverage:.2f}")

        for param_name, candidates in SC.PHASE3_PARAMS.items():
            if param_name == "operationApproach":
                candidates = approach_candidates

            # scoreLevel: sweep with confidenceThreshold=0
            if param_name == "scoreLevel":
                fp_temp = copy.deepcopy(fine)
                fp_temp["confidenceThreshold"] = 0.0
                best_sl = self._sweep_param("scoreLevel", candidates,
                                            coarse, fp_temp, is_coarse=False,
                                            label="3-scoreLevel",
                                            ang_thresh=ang)
                if best_sl.score < best_res.score:
                    best_res = best_sl
                    fine     = copy.deepcopy(best_sl.config["fine"])
                continue

            if len(candidates) <= 2:
                r = self._sweep_param_direct(param_name, candidates,
                                             coarse, fine, is_coarse=False,
                                             ang_thresh=ang)
            else:
                r = self._sweep_param(param_name, candidates, coarse, fine,
                                      is_coarse=False, label=f"3-{param_name}",
                                      ang_thresh=ang)
            if r.score < best_res.score:
                best_res = r
                fine     = copy.deepcopy(r.config["fine"])

        log.info(f"PHASE 3 done: cov={best_res.coverage:.2f}  "
                 f"time={best_res.mean_time:.3f}s")
        return best_res

    # ─────────────────────────────────────────────────────────────────────
    # Phase 4 — Symmetry confirmation (conditional)
    # ─────────────────────────────────────────────────────────────────────

    def phase4_symmetry(self, coarse: dict, fine: dict,
                        phase3_result: EvalResult) -> EvalResult:
        """Conditional symmetry parameter sweep."""
        log.info("=" * 60)
        log.info("PHASE 4 — Symmetry (conditional)")

        # Collect angular errors from Phase 3 result
        ang_errors = [e for s in phase3_result.per_scene
                      for e in s.get("ang_errors", []) if e is not None]

        sym_order = _confirm_symmetry(ang_errors, self.ws.sym_order)
        if sym_order is None:
            log.info("  No symmetry confirmed — skipping Phase 4")
            return phase3_result

        log.info(f"  Confirmed {sym_order}-fold symmetry")
        angle_steps = SC.phase4_angle_steps(sym_order)

        coarse_v, fine_v = [], []
        for axis in SC.PHASE4_ROTATION_STRATEGIES:
            for step in angle_steps:
                fp = copy.deepcopy(fine)
                # Symmetry search requires at least Standard accuracy (1.0).
                # HighSpeed (0) does not apply the rotation correctly and
                # yields cov=0 even when angleStep is set.
                fp["operationApproach"] = max(1.0, float(fine.get("operationApproach", 1.0)))
                fp["rotationStrategy"] = axis
                fp["angleStep"]        = step
                fp["minAngle"]         = -180.0
                fp["maxAngle"]         = 180.0
                coarse_v.append(copy.deepcopy(coarse))
                fine_v.append(fp)

        best = self.evaluate_phase_sweep(coarse_v, fine_v, label="4-sym")

        # When symmetry is confirmed we MUST adopt the symmetry-enabled config.
        # Phase 3 used ANG_THRESH_REGIME_GATE=360° (position-only), so its score
        # cannot be compared directly to Phase 4's tight-threshold score.
        # If Phase 4 achieves any coverage, adopt it to prevent Phase 5+ from
        # running with angleStep=0 (which yields cov=0 on symmetric parts).
        if best.coverage > 0.0:
            log.info(f"  Symmetry enabled: cov={best.coverage:.2f}  "
                     f"axis={best.config['fine'].get('rotationStrategy')}  "
                     f"step={best.config['fine'].get('angleStep')}")
            return best
        log.info(f"  Symmetry candidates all failed (cov=0) — kept Phase 3 result")
        return phase3_result

    # ─────────────────────────────────────────────────────────────────────
    # Phase 5 — Joint refinement
    # ─────────────────────────────────────────────────────────────────────

    def phase5_joint_refinement(self, coarse: dict, fine: dict) -> List[EvalResult]:
        """Re-sweep coarse params sensitive to fine quality. Return top-K results."""
        log.info("=" * 60)
        log.info("PHASE 5 — Joint Refinement")

        # Build candidate grid over Phase 5 params
        vote_ratios   = SC.PHASE5_PARAMS["maxVoteRatio"]
        output_nums   = SC.PHASE5_PARAMS["outputNum"]
        referred_stps = SC.PHASE5_PARAMS["referredStep"]

        coarse_v, fine_v = [], []
        for vr in vote_ratios:
            for on in output_nums:
                for rs in referred_stps:
                    cp = copy.deepcopy(coarse)
                    cp["maxVoteRatio"] = vr
                    cp["outputNum"]    = on
                    cp["referredStep"] = rs
                    coarse_v.append(cp)
                    fine_v.append(copy.deepcopy(fine))

        # Single evaluation pass on one shared scene set — all candidates
        # compared fairly; results collected for Phase 6 top-K selection.
        scenes = self._sample_scenes(SC.M_FULL)
        all_results = []
        for i, (cp, fp) in enumerate(zip(coarse_v, fine_v)):
            r = self.evaluate_config(cp, fp, scenes)
            log.info(f"  [5-joint] cand {i:2d}: cov={r.coverage:.2f}  "
                     f"time={r.mean_time:.3f}s  score={r.score:.1f}")
            all_results.append(r)
        all_results.sort(key=lambda r: r.score)

        noise_margin = SC.PHASE5_NOISE_MARGIN * all_results[0].mean_time
        top_k = [all_results[0]]
        for r in all_results[1:]:
            if abs(r.score - all_results[0].score) <= noise_margin:
                top_k.append(r)
            if len(top_k) >= SC.PHASE5_TOP_K:
                break

        # Secondary tiebreaker
        top_k.sort(key=lambda r: r.soft_score())
        log.info(f"PHASE 5 done: top-{len(top_k)} configs, "
                 f"best cov={top_k[0].coverage:.2f}")
        return top_k

    # ─────────────────────────────────────────────────────────────────────
    # Phase 6 — Continuous interval narrowing
    # ─────────────────────────────────────────────────────────────────────

    def phase6_interval_narrowing(self, candidates: List[EvalResult]) -> EvalResult:
        """Dense search around the best continuous-param values."""
        log.info("=" * 60)
        log.info("PHASE 6 — Interval Narrowing")

        best_overall = candidates[0]

        for seed_result in candidates:
            cp = copy.deepcopy(seed_result.config["coarse"])
            fp = copy.deepcopy(seed_result.config["fine"])

            # -- refStep  (distQuantification follows via locked ratio) --
            best_ref  = cp.get("refStep", SC.REFSTEP_BOUNDS[1] // 2)
            best_dist_q = cp.get("distQuantification", self.ws.distQuantification)
            ratio = best_dist_q / best_ref if best_ref > 0 else 1.0
            ref_candidates = list(range(max(1, best_ref - SC.PHASE6_REFSTEP_DELTA),
                                        best_ref + SC.PHASE6_REFSTEP_DELTA + 1))
            cp_v, fp_v = [], []
            for ref in ref_candidates:
                c = copy.deepcopy(cp)
                c["refStep"]           = ref
                c["distQuantification"] = max(0.5, ref * ratio)
                cp_v.append(c)
                fp_v.append(copy.deepcopy(fp))
            r_ref = self.evaluate_phase_sweep(cp_v, fp_v, label="6-refStep")
            if r_ref.score < best_overall.score:
                best_overall = r_ref
                cp = copy.deepcopy(r_ref.config["coarse"])

            # -- maxVoteRatio --
            best_vr = cp.get("maxVoteRatio", 0.5)
            vr_vals = np.linspace(
                max(0.0, best_vr - SC.PHASE6_VOTE_WINDOW),
                min(1.0, best_vr + SC.PHASE6_VOTE_WINDOW),
                SC.PHASE6_INTERVAL_STEPS).tolist()
            r_vr = self._sweep_param("maxVoteRatio", vr_vals, cp, fp,
                                     is_coarse=True, label="6-voteRatio")
            if r_vr.score < best_overall.score:
                best_overall = r_vr
                cp = copy.deepcopy(r_vr.config["coarse"])

            # -- confidenceThreshold --
            best_ct = fp.get("confidenceThreshold", 0.0)
            ct_vals = np.linspace(
                max(0.0, best_ct - SC.PHASE6_CONF_WINDOW),
                min(1.0, best_ct + SC.PHASE6_CONF_WINDOW),
                SC.PHASE6_INTERVAL_STEPS).tolist()
            r_ct = self._sweep_param("confidenceThreshold", ct_vals, cp, fp,
                                     is_coarse=False, label="6-confThresh")
            if r_ct.score < best_overall.score:
                best_overall = r_ct
                fp = copy.deepcopy(r_ct.config["fine"])

            # -- maxNumOfPointPairsPerFeature --
            best_pairs = cp.get("maxNumOfPointPairsPerFeature",
                                self.ws.maxNumOfPointPairsPerFeature)
            pairs_vals = [max(1, int(best_pairs * s))
                          for s in SC.PHASE6_PAIRS_SCALES]
            r_pairs = self._sweep_param_direct(
                "maxNumOfPointPairsPerFeature", pairs_vals, cp, fp, is_coarse=True)
            if r_pairs.score < best_overall.score:
                best_overall = r_pairs
                cp = copy.deepcopy(r_pairs.config["coarse"])

        log.info(f"PHASE 6 done: cov={best_overall.coverage:.2f}  "
                 f"time={best_overall.mean_time:.3f}s  "
                 f"score={best_overall.score:.1f}")
        return best_overall

    # ─────────────────────────────────────────────────────────────────────
    # Main run() loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> Optional[EvalResult]:
        """Execute the full optimization pipeline."""
        t0 = time.time()
        log.info(f"\n{'='*60}")
        n_per = [len(g) for g in self.scene_groups]
        log.info(f"Optimizer starting: part={self.part_name}  "
                 f"M={len(self.scene_groups)} scenes  N={n_per} inst/scene  "
                 f"cache={'ON' if self.cache else 'OFF'}  "
                 f"two_pass={'ON' if self.use_two_pass else 'OFF'}")
        log.info(f"Warm start: distQ={self.ws.distQuantification:.1f}  "
                 f"prefer_edge={self.ws.prefer_edge}")

        # ── Phase 1 ──────────────────────────────────────────────────────
        passing_regimes = self.phase1_regime_gate()
        if not passing_regimes:
            log.error("Optimization failed at Phase 1.")
            return None

        best_regime = passing_regimes[0]
        best_coarse = copy.deepcopy(best_regime["coarse"])
        best_fine   = copy.deepcopy(best_regime["fine"])

        # ── Phase 2 ──────────────────────────────────────────────────────
        ph2_result = self.phase2_coarse_cd(best_regime)
        best_coarse = copy.deepcopy(ph2_result.config["coarse"])
        best_fine   = copy.deepcopy(ph2_result.config["fine"])

        gate2 = self._gate("after_phase2", ph2_result.coverage)

        # ── Phase 3 ──────────────────────────────────────────────────────
        ph3_result = self.phase3_fine_cd(best_coarse, best_fine, ph2_result)
        best_coarse = copy.deepcopy(ph3_result.config["coarse"])
        best_fine   = copy.deepcopy(ph3_result.config["fine"])

        # ── Phase 4 (conditional) ─────────────────────────────────────────
        if gate2:
            ph4_result = self.phase4_symmetry(best_coarse, best_fine, ph3_result)
        else:
            ph4_result = ph3_result
            log.info("PHASE 4 skipped (Phase 2 gate)")

        best_coarse = copy.deepcopy(ph4_result.config["coarse"])
        best_fine   = copy.deepcopy(ph4_result.config["fine"])

        # ── Phase 5 ──────────────────────────────────────────────────────
        top_k_results = self.phase5_joint_refinement(best_coarse, best_fine)
        best_coarse = copy.deepcopy(top_k_results[0].config["coarse"])
        best_fine   = copy.deepcopy(top_k_results[0].config["fine"])

        # ── Phase 6 (conditional on Phase 3 gate) ────────────────────────
        gate3 = self._gate("after_phase3", top_k_results[0].coverage)
        if gate3:
            final_result = self.phase6_interval_narrowing(top_k_results)
        else:
            final_result = top_k_results[0]
            log.info("PHASE 6 skipped (Phase 3 gate)")

        # ── Summary ──────────────────────────────────────────────────────
        elapsed = time.time() - t0
        cache_stats = self.cache.stats() if self.cache else {}
        log.info(f"\n{'='*60}")
        log.info(f"OPTIMIZATION COMPLETE: {self.part_name}")
        log.info(f"  coverage   = {final_result.coverage:.3f}")
        log.info(f"  mean_time  = {final_result.mean_time:.3f} s")
        log.info(f"  score      = {final_result.score:.1f}")
        log.info(f"  evals      = {self._n_evals}")
        log.info(f"  wall_time  = {elapsed:.0f} s")
        if cache_stats:
            log.info(f"  cache      = {cache_stats}")

        self._log_result_json(final_result)
        return final_result

    # ─────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────

    def export_best(self, result: EvalResult, prefix:str="", suffix:str="") -> str:
        """Write best config to YAML-style JSON for production use."""
        out_path = os.path.join(RESULTS_DIR, f"{prefix}best_config_{self.part_name}{suffix}.json")
        payload  = {
            "part_name":  self.part_name,
            "coverage":   result.coverage,
            "mean_time":  result.mean_time,
            "score":      result.score,
            "coarse":     result.config.get("coarse", {}),
            "fine":       result.config.get("fine",   {}),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        log.info(f"Best config exported → {out_path}")
        return out_path

    def _log_result_json(self, result: EvalResult) -> None:
        ts  = int(time.time())
        out = os.path.join(RESULTS_DIR, f"result_{self.part_name}_{ts}.json")
        with open(out, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

    def cleanup(self):
        """Save cache and release resources."""
        if self.cache:
            self.cache.save()


# ─────────────────────────────────────────────────────────────────────────────
# Symmetry confirmation helper
# ─────────────────────────────────────────────────────────────────────────────

def _confirm_symmetry(ang_errors, geo_hint_order):
    """Return symmetry order if confirmed, else None."""
    if not ang_errors:
        return None
    for n in [2, 3, 4]:
        target = 360.0 / n
        frac   = sum(abs(e - target) < SC.SYM_ANGLE_TOL_DEG
                     for e in ang_errors) / len(ang_errors)
        if frac > SC.SYM_AMBIGUOUS_FRAC_THRESHOLD:
            if geo_hint_order is None or geo_hint_order == n:
                return n
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser():
    p = argparse.ArgumentParser(description="MM_Optimizer — auto-tune MechVision params")
    p.add_argument("--part",        required=True,  help="Part name, e.g. 25333MB000")
    p.add_argument("--scenes_dir",  default=None,   help="Override synthetic scenes dir")
    p.add_argument("--m_full",      type=int, default=None,
                   help="Override M_FULL evaluation budget (default: all available scenes)")
    p.add_argument("--dry_run",     action="store_true", help="No MechVision calls")
    p.add_argument("--no_cache",    action="store_true", help="Disable EvalCache")
    p.add_argument("--no_two_pass", action="store_true", help="Disable two-pass")
    p.add_argument("--export_best", default=True, action="store_true", help="Write best config JSON")
    p.add_argument("--seed",        type=int, default=42, help="Random seed")
    return p


def main():
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Scenes directory
    if args.scenes_dir:
        scenes_root = args.scenes_dir
    else:
        scenes_root = os.path.join(_ROOT, "output", "synthetic_target",
                                   args.part)

    # Build scene groups — one group per scene_MMMMM directory.
    # Each group is the full list of sample_*.ply for that bin capture.
    scene_groups = list_synthetic_scenes(scenes_root) if os.path.isdir(scenes_root) else []
    if not scene_groups:
        log.error(f"No scene directories found under {scenes_root}")
        sys.exit(1)
    n_per_scene = [len(g) for g in scene_groups]
    log.info(f"Found {len(scene_groups)} M-scenes, "
             f"{n_per_scene} instances/scene")

    # Override M_FULL — default to number of available scenes
    if args.m_full is not None:
        SC.M_FULL = args.m_full
    else:
        SC.M_FULL = len(scene_groups)
        SC.M_SMALL = max(1, len(scene_groups) // 2)
        log.info(f"M_FULL auto-set to {SC.M_FULL}, M_SMALL to {SC.M_SMALL}")

    # Phase 0 — mesh analysis
    model_path = os.path.join(MM_MODEL_ROOT, f"{args.part}_surface",
                              f"{args.part}_surface.ply")
    if not os.path.exists(model_path):
        log.error(f"Reference model not found: {model_path}")
        sys.exit(1)
    pcd = load_reference_pcd(model_path)
    ws  = analyze_mesh(pcd)
    log.info(f"Warm start: D={ws.diameter_m*1e3:.1f}mm  prefer_edge={ws.prefer_edge}")

    if args.dry_run:
        log.info("DRY RUN — no MechVision calls")
        client     = None
        project_id = -1
    else:
        client = MechVisionClient()
        projects   = client.get_projects()
        if PROJ_NAME not in projects:
            log.error(f"Project '{PROJ_NAME}' not found. Loaded: {projects}")
            sys.exit(1)
        project_id = projects[PROJ_NAME]

    # Cache
    cache_path = os.path.join(RESULTS_DIR, f"eval_cache_{args.part}.json")
    cache = (None if args.no_cache
             else EvalCache(cache_path, enabled=ENABLE_CACHE))

    opt = Optimizer(
        part_name    = args.part,
        client       = client,
        project_id   = project_id,
        scene_groups = scene_groups,
        warm_start   = ws,
        cache        = cache,
        use_two_pass = (not args.no_two_pass) and ENABLE_TWO_PASS,
        dry_run      = args.dry_run,
    )

    try:
        result = opt.run()
        if result and args.export_best:
            opt.export_best(result, prefix="CD_")
    finally:
        opt.cleanup()
        if client:
            client.close()


if __name__ == "__main__":
    main()
