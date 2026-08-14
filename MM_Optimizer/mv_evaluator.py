"""
mv_evaluator.py — MechVision evaluation harness
-----------------------------------------------
Sampler-agnostic base of `tuner.Tuner`. Owns the evaluation contract:

  - scene sampling (`_sample_scenes`, `_prepare_scene`)
  - MechVision param-dict construction (`_make_params_dict`)
  - single-scene execution + GT matching (`_run_one_scene`)
  - cached config evaluation (`evaluate_config`, `evaluate_phase_sweep`)
  - the regime gate (`phase1_regime_gate`) and symmetry sweep (`phase4_symmetry`)
  - result export / logging / cleanup

No sampler logic and no CLI: those live in `tuner.py`.
"""

import copy
import json
import logging
import math
import os
import random
import sys
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

from mm_adapter.mm_adapter     import MechVisionClient, VisionRunError
from mm_adapter.mm_dataclasses import (CoarseMatchingV2, FineMatchingLite,
                                       EasyCreateStringList)
from MM_Optimizer.eval_cache      import EvalCache
from MM_Optimizer.mesh_analysis   import WarmStart
from MM_Optimizer.optimizer_utils import read_gt_pose_from_ply
import MM_Optimizer.model_sync as model_sync
import MM_Optimizer.search_config as SC

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature flags / MechVision project + path constants
# ─────────────────────────────────────────────────────────────────────────────
ENABLE_CACHE = True   # Strategy 1 — set False to disable the transposition table

PROJ_NAME            = "CAD_Match"
MM_MODEL_ROOT        = os.path.join(_DIR, "CAD_Match", "resource", "3d_matching")
RESULTS_DIR          = os.path.join(_DIR, "results")
OPTIMIZER_UTILS_PATH = os.path.join(_DIR, "optimizer_utils.py")


# ─────────────────────────────────────────────────────────────────────────────
# MechVision parameter typing
# ─────────────────────────────────────────────────────────────────────────────
# name -> (mm_adapter type, unit). Only these reach MechVision.
_COARSE_TYPES = {
    "registrationMode":             ("double", ""),
    "refStep":                      ("double", ""),
    "distQuantification":           ("double", ""),
    "angleQuantification":          ("double", ""),
    "maxNumOfPointPairsPerFeature": ("double", ""),
    "maxVoteRatio":                 ("double", ""),
    "referredStep":                 ("double", ""),
    "useDistanceNMS":               ("bool",   ""),
    "filterCandidatePoseByAxis":    ("bool",   ""),
    "angleThreshold":               ("double", ""),
    "outputNum":                    ("double", ""),
    "minVoxelLength":               ("double", "m"),
    "maxVoxelLength":               ("double", "m"),
}

_FINE_TYPES = {
    "registrationMode":                  ("double", ""),
    "operationApproach":                 ("double", ""),
    "deviationCorrectionCapacity":       ("double", ""),
    "onlyConsiderVisibleSurfaceOfModel": ("bool",   ""),
    "considerErrorofNormalAngles":       ("bool",   ""),
    "scoreLevel":                        ("double", ""),
    "confidenceThreshold":               ("double", ""),
    "rotationStrategy":                  ("double", ""),
    "angleStep":                         ("double", ""),
    "minAngle":                          ("double", ""),
    "maxAngle":                          ("double", ""),
    "candidateTopNum":                   ("double", ""),
}

# Search-space bookkeeping carried alongside the MechVision keys (the Optuna trial
# parameter names). Listed so an unrecognised key is a typo, not a silent no-op.
_NON_MV_KEYS = frozenset({"minVoxelLength_mm", "maxVoxelLength_mm", "voxel_width_mm"})


def _apply_params(step, params: dict, types: dict) -> None:
    """Set `params` onto an mm_adapter step object as (value, type, unit) triples."""
    for k, v in params.items():
        spec = types.get(k)
        if spec is not None:
            setattr(step, k, (str(v), *spec))
        elif k not in _NON_MV_KEYS:
            log.warning(f"unknown MechVision parameter {k!r} ignored")


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


class MVEvaluator:
    """Shared MechVision evaluation harness (sampler-agnostic).

    Parameters
    ----------
    part_name    : Part identifier (matches model dir and scene dir names).
    client       : Connected MechVisionClient (or None for dry_run).
    project_id   : Integer project ID for PROJ_NAME.
    scene_groups : List[List[str]] — one inner list per scene_MMMMM directory.
                   Each inner list contains all sample_*.ply paths for that
                   bin capture (N instances per scene).
    warm_start   : WarmStart from mesh_analysis.
    cache        : EvalCache instance (or None to disable).
    dry_run      : Build param dicts but do not call MechVision.
    """

    def __init__(self,
                 part_name:    str,
                 client:       MechVisionClient,
                 project_id:   int,
                 scene_groups: List[List[str]],
                 warm_start:   WarmStart,
                 cache:        Optional[EvalCache] = None,
                 dry_run:      bool = False):

        self.part_name    = part_name
        self.client       = client
        self.project_id   = project_id
        self.scene_groups = scene_groups   # List[List[str]] — one group per M-scene
        self.ws           = warm_start
        self.cache        = cache
        self.dry_run      = dry_run

        # Optional live-eval hook. When set, called after every real MechVision
        # run in _run_one_scene with the raw matched poses, so a GUI can render a
        # per-instance overlay as the search proceeds. Default None → CLI unchanged.
        # Signature: (scene_dir, coarse_params, fine_params, fine_poses, gt_poses).
        self.on_scene_eval = None

        self._model_root   = MM_MODEL_ROOT
        self._n_evals = 0   # total MechVision calls

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
        # One library entry per part holding one cloud, so coarse and fine necessarily
        # share it — which is why the mixed regimes were dropped (see SC.REGIMES).
        # `registrationMode` still tells MechVision how to interpret that cloud; which
        # cloud it is was decided by `model_sync.sync_regime_model`.
        model_root_dir = model_sync.model_dir(self.part_name)
        model_name     = self.part_name
        ply_file       = model_sync.model_ply(self.part_name)
        geo_file       = model_sync.geo_center(self.part_name)

        scene = EasyCreateStringList(
            name="Scene_Path",
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
            modelSelection=(model_name, "string", ""),
            modelFileName=(ply_file, "string", ""),
            geoCenterFileName=(geo_file, "string", ""),
        )
        _apply_params(coarse, coarse_params, _COARSE_TYPES)

        fine = FineMatchingLite(
            name="Fine_Match_Synthetics",
            modelSelection=(model_name, "string", ""),
            modelFileName=(ply_file, "string", ""),
            geoCenterFileName=(geo_file, "string", ""),
        )
        _apply_params(fine, fine_params, _FINE_TYPES)

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

        # Live-eval hook (GUI overlay): fire with the raw matched poses before they
        # are reduced to error scalars. Guarded so a preview error never breaks tuning.
        if self.on_scene_eval is not None:
            try:
                self.on_scene_eval(scene_dir, coarse_params, fine_params,
                                   fine_poses, gt_poses)
            except Exception as e:
                log.debug(f"on_scene_eval hook failed (ignored): {e}")

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
        """Evaluate parallel (coarse, fine) variant lists on M_FULL scenes.

        coarse_variants and fine_variants are equal-length; index i is one candidate config.
        Returns the best-scoring EvalResult.
        """
        assert len(fine_variants) == len(coarse_variants)
        scenes = self._sample_scenes(SC.M_FULL)
        results = []
        for i, (cp, fp) in enumerate(zip(coarse_variants, fine_variants)):
            r = self.evaluate_config(cp, fp, scenes, pos_thresh, ang_thresh)
            log.info(f"  [{label}] cand {i:2d}: cov={r.coverage:.2f}  "
                     f"time={r.mean_time:.3f}s  score={r.score:.1f}")
            results.append(r)
        return min(results, key=lambda r: r.score)

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
        """Test the surface/edge regimes. Return passing regimes sorted by coverage.

        Each regime installs its cloud as the part's MechVision model before evaluating,
        which is what removes the manual copy into the model library. The install replaces
        the folder, so this loop must stay **sequential** — two regimes syncing at once
        would leave the library holding one regime's cloud while the other is scored
        against it.
        """
        log.info("=" * 60)
        log.info("PHASE 1 — Regime Gate")

        # Asked of the exported bundle, not the model library: the library holds only
        # whichever regime was synced last, so it reports history rather than options.
        available = model_sync.available_types(self.part_name)
        log.info(f"  cloud types exported for {self.part_name}: {available or 'none'}")

        # Order based on geometry hint
        regimes = sorted(SC.REGIMES,
                         key=lambda r: (r["needs_edge"] and not self.ws.prefer_edge,
                                        r["id"]))

        passing = []
        for regime in regimes:
            cloud_type = "edge" if regime["coarse_mode"] == 1.0 else "surface"
            if cloud_type not in available:
                log.info(f"  Regime {regime['id']}: skip (no {cloud_type} cloud exported)")
                continue

            if not self.dry_run:
                model_sync.sync_regime_model(self.part_name, cloud_type)
                log.info(f"  Regime {regime['id']}: synced {cloud_type} cloud into the "
                         f"model library as {self.part_name}.ply")

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

            if r.coverage >= SC.REGIME_COVERAGE_GATE:
                passing.append({**regime, "cloud_type": cloud_type,
                                "coverage": r.coverage, "coarse": cp, "fine": fp})

        if not passing:
            log.error("PHASE 1: No regime passes. Part may be un-tunable.")
            return []

        passing.sort(key=lambda x: -x["coverage"])
        log.info(f"PHASE 1 done: {len(passing)} passing regimes — "
                 f"best = {passing[0]['id']} (cov={passing[0]['coverage']:.2f})")

        # The loop leaves whichever regime ran last installed, which is not necessarily the
        # winner. Re-install the best one so the library is consistent with what the caller
        # is about to tune; a caller choosing a different regime must call lock_regime.
        self.lock_regime(passing[0])

        return passing

    def lock_regime(self, regime: dict) -> None:
        """Install ``regime``'s cloud as the part's active MechVision model.

        Call before tuning with a regime other than the gate's winner. Every subsequent
        MechVision run matches against whatever this last installed — the regime is carried
        by the model library, not by the parameter dict.
        """
        cloud_type = regime.get("cloud_type") or (
            "edge" if regime.get("coarse_mode") == 1.0 else "surface")
        if self.dry_run:
            return
        model_sync.sync_regime_model(self.part_name, cloud_type)
        log.info(f"  locked regime {regime.get('id', '?')} ({cloud_type}) into the model library")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4 — Symmetry confirmation (conditional)
    # ─────────────────────────────────────────────────────────────────────────

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
        angle_steps = SC.angle_steps(sym_order)

        coarse_v, fine_v = [], []
        for axis in SC.ROTATION_STRATEGIES:
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

    # ─────────────────────────────────────────────────────────────────────────
    # Export / logging / cleanup
    # ─────────────────────────────────────────────────────────────────────────

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
# Symmetry confirmation helper (module-level)
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

