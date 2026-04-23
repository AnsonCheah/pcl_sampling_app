"""
optuna_optimizer.py — Joint CmaEsSampler optimizer for MechVision pose estimation
----------------------------------------------------------------------------------
All coarse and fine parameters are co-optimised in a single CmaEsSampler study
(~18D), eliminating the stage isolation of the previous staged TPE architecture.

Architecture
------------
  Phase 1  : regime gate (geometry bounds, pairs_candidates, voxel_bounds)
  Pre-study: one-shot look-ahead — geometry warm-start + default fine
             → median pos error → operationApproach hint for trial 0
  Round 0  : joint CmaEsSampler study (OPTUNA_N_TRIALS_JOINT trials)
  Round 1+ : refinement pass (OPTUNA_N_TRIALS_JOINT_REFINE trials each),
             warm-started from previous round's best config
  Phase 4  : symmetry confirmation (unchanged from CD optimizer)

Parameter space (~18D — all numeric for CmaEsSampler compatibility)
--------------------------------------------------------------------
  Coarse: coarse_mode, refStep, distQ, angleQ_idx→ANGLE_QUANT_CANDIDATES,
          pairs_idx→pairs_candidates, maxVoteRatio, referredStep, outputNum,
          useDistNMS, minVoxelLength_mm, voxel_width_mm,
          filterByAxis, angleThreshold
  Fine:   fine_mode, opApproach, devCap, visibleSurf, normalAng
  Fixed:  scoreLevel=0, confidenceThreshold=0.1, candidateTopNum=1

CLI
---
  python MM_Optimizer/optuna_optimizer.py --part 25333MB000 [--dry_run]
      [--n_trials_joint N] [--n_trials_refine N] [--n_rounds N]
      [--scenes_dir PATH] [--m_full N] [--no_cache] [--seed N]
      [--export_best] [--storage PATH]
"""

import argparse
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── project root on path ──────────────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, ".."))
for _p in [_ROOT, _DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mm_adapter.mm_adapter import MechVisionClient
from MM_Optimizer.eval_cache    import EvalCache
from MM_Optimizer.mesh_analysis import analyze_mesh, load_reference_pcd
from MM_Optimizer.optimizer     import (Optimizer, EvalResult,
                                        PROJ_NAME, MM_MODEL_ROOT,
                                        RESULTS_DIR, ENABLE_CACHE)
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
import MM_Optimizer.search_config as SC

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Joint suggest / conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def suggest_params_joint(
    trial: optuna.Trial,
    pairs_candidates: List[int],
    voxel_bounds: Tuple[float, float, float, float],
    refstep_bounds: Tuple[int, int],
    coarse_mode_max: int = 1,
) -> dict:
    """Suggest all 18 joint params; returns flat trial-key → value dict.

    All params are numeric (int or float) so CmaEsSampler can handle them.
    Discrete/categorical values use index mapping; _joint_params_to_dicts
    converts to MechVision API values.

    coarse_mode_max: 0 if no edge model exists (surface-only), 1 otherwise.
    """
    min_lo, min_hi, width_lo, width_hi = voxel_bounds
    rlo, rhi = refstep_bounds
    vlo, vhi = SC.OPTUNA_VOTERATIO_BOUNDS
    flo, fhi = SC.OPTUNA_REFERRED_BOUNDS
    olo, ohi = SC.OPTUNA_OUTPUTNUM_BOUNDS

    # print(coarse_mode_max)
    return {
        # ── Coarse ──────────────────────────────────────────────────────
        "coarse_mode":       trial.suggest_int(  "coarse_mode",     0, coarse_mode_max),
        "refStep":           trial.suggest_int(  "refStep",         rlo, rhi),
        "distQ":             trial.suggest_float("distQ",           *SC.OPTUNA_DISTQ_BOUNDS),
        "angleQ_idx":        trial.suggest_int(  "angleQ_idx",      0, len(SC.ANGLE_QUANT_CANDIDATES) - 1),
        "pairs_idx":         trial.suggest_int(  "pairs_idx",       0, len(pairs_candidates) - 1),
        "maxVoteRatio":      trial.suggest_float("maxVoteRatio",    vlo, vhi),
        "referredStep":      trial.suggest_int(  "referredStep",    flo, fhi),
        "outputNum":         trial.suggest_int(  "outputNum",       olo, ohi),
        "useDistNMS":        trial.suggest_int(  "useDistNMS",      0, 1),
        "minVoxelLength_mm": trial.suggest_float("minVoxelLength_mm", min_lo, min_hi),
        "voxel_width_mm":    trial.suggest_float("voxel_width_mm",  width_lo, width_hi),
        "filterByAxis":      trial.suggest_int(  "filterByAxis",    0, 1),
        "angleThreshold":    trial.suggest_int(  "angleThreshold",
                                                  SC.OPTUNA_ANGLETHRESH_BOUNDS[0],
                                                  SC.OPTUNA_ANGLETHRESH_BOUNDS[1]),
        # ── Fine ────────────────────────────────────────────────────────
        "fine_mode":         trial.suggest_int(  "fine_mode",       0, 1),
        "opApproach":        trial.suggest_int(  "opApproach",      0, SC.OPTUNA_OPAPP_MAX),
        "devCap":            trial.suggest_int(  "devCap",          0, SC.OPTUNA_DEVCAP_MAX),
        "visibleSurf":       trial.suggest_int(  "visibleSurf",     0, 1),
        "normalAng":         trial.suggest_int(  "normalAng",       0, 1),
    }


def _joint_params_to_dicts(
    p: dict,
    pairs_candidates: List[int],
) -> Tuple[dict, dict]:
    """Convert flat trial param dict to (coarse_dict, fine_dict) for evaluate_config."""
    min_vox = float(p["minVoxelLength_mm"])
    coarse = {
        "registrationMode":             float(p["coarse_mode"]),
        "refStep":                      int(p["refStep"]),
        "distQuantification":           float(p["distQ"]),
        "angleQuantification":          SC.ANGLE_QUANT_CANDIDATES[int(p["angleQ_idx"])],
        "maxNumOfPointPairsPerFeature": pairs_candidates[int(p["pairs_idx"])],
        "maxVoteRatio":                 float(p["maxVoteRatio"]),
        "referredStep":                 int(p["referredStep"]),
        "outputNum":                    int(p["outputNum"]),
        "useDistanceNMS":               bool(p["useDistNMS"]),
        "minVoxelLength":               min_vox,
        "maxVoxelLength":               min_vox + float(p["voxel_width_mm"]),
        "filterCandidatePoseByAxis":    bool(p["filterByAxis"]),
        "angleThreshold":               int(p["angleThreshold"]),
    }
    fine = {
        "registrationMode":                  float(p["fine_mode"]),
        "operationApproach":                 float(p["opApproach"]),
        "deviationCorrectionCapacity":       float(p["devCap"]),
        "onlyConsiderVisibleSurfaceOfModel": bool(p["visibleSurf"]),
        "considerErrorofNormalAngles":       bool(p["normalAng"]),
        "scoreLevel":                        0.0,
        "confidenceThreshold":               0.1,
        "candidateTopNum":                   1,
    }
    return coarse, fine


def _build_warm_joint(
    coarse_dict: dict,
    fine_dict: dict,
    pairs_candidates: List[int],
    voxel_bounds: Tuple[float, float, float, float],
    refstep_bounds: Tuple[int, int],
) -> dict:
    """Convert (coarse, fine) config dicts to flat trial param dict for enqueue_trial."""
    min_lo, min_hi, width_lo, width_hi = voxel_bounds
    rlo, rhi = refstep_bounds
    vlo, vhi = SC.OPTUNA_VOTERATIO_BOUNDS
    flo, fhi = SC.OPTUNA_REFERRED_BOUNDS
    olo, ohi = SC.OPTUNA_OUTPUTNUM_BOUNDS

    # angleQ index — snap to nearest candidate
    aq = coarse_dict.get("angleQuantification", SC.ANGLE_QUANT_CANDIDATES[4])  # default 60
    aq_idx = min(range(len(SC.ANGLE_QUANT_CANDIDATES)),
                 key=lambda i: abs(SC.ANGLE_QUANT_CANDIDATES[i] - aq))

    # pairs index — snap to nearest candidate
    raw_pairs = coarse_dict.get("maxNumOfPointPairsPerFeature",
                                pairs_candidates[len(pairs_candidates) // 2])
    pairs_idx = min(range(len(pairs_candidates)),
                    key=lambda i: abs(pairs_candidates[i] - raw_pairs))

    min_vox = float(coarse_dict.get("minVoxelLength", (min_lo + min_hi) / 2))
    max_vox = float(coarse_dict.get("maxVoxelLength", min_vox + (width_lo + width_hi) / 2))
    vox_w   = max_vox - min_vox

    # Clamp to valid ranges
    min_vox = max(min_lo,   min(min_hi,   min_vox))
    vox_w   = max(width_lo, min(width_hi, vox_w))

    return {
        "coarse_mode":       int(round(float(coarse_dict.get("registrationMode", 0.0)))),
        "refStep":           max(rlo, min(rhi,  int(coarse_dict.get("refStep", 5)))),
        "distQ":             max(SC.OPTUNA_DISTQ_BOUNDS[0],
                                 min(SC.OPTUNA_DISTQ_BOUNDS[1],
                                     float(coarse_dict.get("distQuantification", 1.0)))),
        "angleQ_idx":        aq_idx,
        "pairs_idx":         pairs_idx,
        "maxVoteRatio":      max(vlo, min(vhi, float(coarse_dict.get("maxVoteRatio", 0.5)))),
        "referredStep":      max(flo, min(fhi, int(coarse_dict.get("referredStep", 1)))),
        "outputNum":         max(olo, min(ohi, int(coarse_dict.get("outputNum", 1)))),
        "useDistNMS":        int(bool(coarse_dict.get("useDistanceNMS", True))),
        "minVoxelLength_mm": min_vox,
        "voxel_width_mm":    vox_w,
        "filterByAxis":      int(bool(coarse_dict.get("filterCandidatePoseByAxis", True))),
        "angleThreshold":    max(SC.OPTUNA_ANGLETHRESH_BOUNDS[0],
                                 min(SC.OPTUNA_ANGLETHRESH_BOUNDS[1],
                                     int(coarse_dict.get("angleThreshold", 135)))),
        "fine_mode":         int(round(float(fine_dict.get("registrationMode", 0.0)))),
        "opApproach":        max(0, min(SC.OPTUNA_OPAPP_MAX,
                                        int(round(float(fine_dict.get("operationApproach", 1.0)))))),
        "devCap":            max(0, min(SC.OPTUNA_DEVCAP_MAX,
                                        int(round(float(fine_dict.get("deviationCorrectionCapacity", 0.0)))))),
        "visibleSurf":       int(bool(fine_dict.get("onlyConsiderVisibleSurfaceOfModel", False))),
        "normalAng":         int(bool(fine_dict.get("considerErrorofNormalAngles", False))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Budget audit callback
# ─────────────────────────────────────────────────────────────────────────────

def _budget_audit_callback(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    if (trial.state == optuna.trial.TrialState.COMPLETE
            and trial.number % 10 == 0):
        pruned = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.PRUNED]
        if pruned:
            # Bucket by first token of prune_reason (e.g. "median", "time", "cov")
            buckets: dict = {}
            for pt in pruned:
                key = pt.user_attrs.get("prune_reason", "unknown").split("@")[0]
                buckets[key] = buckets.get(key, 0) + 1
            reason_str = "  ".join(f"{k}:{v}" for k, v in sorted(buckets.items()))
            log.info(f"  [budget] trial={trial.number:3d}  best={study.best_value:.4f}  "
                     f"pruned={len(pruned)} ({reason_str})")
        else:
            log.info(f"  [budget] trial={trial.number:3d}  best={study.best_value:.4f}  "
                     f"pruned=0")


# ─────────────────────────────────────────────────────────────────────────────
# OptunaOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class OptunaOptimizer:
    """Joint CmaEsSampler optimizer for MechVision pose estimation parameters.

    Replaces Phases 2–3–5–6 of the hierarchical coordinate descent with a single
    18D CmaEsSampler study that co-optimises all coarse and fine parameters.
    Phases 0, 1, and 4 are unchanged.

    Parameters
    ----------
    part_name            : Part identifier (matches model dir and scene dir names).
    client               : Connected MechVisionClient, or None for dry_run.
    project_id           : Integer project ID for PROJ_NAME.
    scene_groups         : List[List[str]] — one inner list per scene_MMMMM directory.
    warm_start           : WarmStart from mesh_analysis.analyze_mesh().
    cache                : EvalCache instance, or None to disable.
    dry_run              : Build param dicts but do not call MechVision.
    n_trials_joint       : Round 0 joint study budget (default: SC.OPTUNA_N_TRIALS_JOINT).
    n_trials_joint_refine: Round 1+ refinement budget (default: SC.OPTUNA_N_TRIALS_JOINT_REFINE).
    n_rounds             : Number of optimisation rounds (default: SC.OPTUNA_N_ROUNDS).
    seed                 : Random seed for reproducibility.
    storage_path         : Path prefix for SQLite crash-resume DBs, e.g. "results/optuna".
                           Creates {prefix}_{part}_joint_r{k}.db per round.
                           None = in-memory studies (no persistence).
    """

    def __init__(
        self,
        part_name:             str,
        client,
        project_id:            int,
        scene_groups:          List[List[str]],
        warm_start,
        cache:                 Optional[EvalCache] = None,
        dry_run:               bool = False,
        n_trials_joint:        Optional[int] = None,
        n_trials_joint_refine: Optional[int] = None,
        n_rounds:              Optional[int] = None,
        seed:                  int = 42,
        storage_path:          Optional[str] = None,
    ):
        self.opt = Optimizer(
            part_name    = part_name,
            client       = client,
            project_id   = project_id,
            scene_groups = scene_groups,
            warm_start   = warm_start,
            cache        = cache,
            use_two_pass = False,
            dry_run      = dry_run,
        )
        self.n_trials_joint        = (n_trials_joint        if n_trials_joint        is not None
                                      else SC.OPTUNA_N_TRIALS_JOINT)
        self.n_trials_joint_refine = (n_trials_joint_refine if n_trials_joint_refine is not None
                                      else SC.OPTUNA_N_TRIALS_JOINT_REFINE)
        self.n_rounds              = (n_rounds              if n_rounds              is not None
                                      else SC.OPTUNA_N_ROUNDS)
        self._seed         = seed
        self._storage_path = storage_path

        warm_pairs = warm_start.maxNumOfPointPairsPerFeature
        self._pairs_candidates: List[int] = sorted(set(
            max(1, min(20000, int(warm_pairs * s)))
            for s in SC.PHASE2B_PAIRS_SCALES
        ))
        self._voxel_bounds: Tuple[float, float, float, float] = (
            max(0.1, warm_start.minVoxelLength_mm * 0.2),
            warm_start.minVoxelLength_mm * 6.0,
            max(0.5, warm_start.maxVoxelLength_mm * 0.1),
            warm_start.maxVoxelLength_mm * 6.0,
        )
        refstep_hi = max(SC.OPTUNA_REFSTEP_BOUNDS[1],
                         int(warm_start.refStep * max(SC.PHASE2A_REFSTEP_SCALES)))
        self._refstep_bounds: Tuple[int, int] = (SC.OPTUNA_REFSTEP_BOUNDS[0], refstep_hi)

        self._studies_joint: List[optuna.Study] = []
        self._best_mean_time: float = float("inf")

    # ─────────────────────────────────────────────────────────────────────
    # Default param helpers
    # ─────────────────────────────────────────────────────────────────────

    def _default_fine(self, regime: dict) -> dict:
        """Default fine params for look-ahead evaluation."""
        return {
            "registrationMode":                  regime["fine_mode"],
            "operationApproach":                 1.0,
            "deviationCorrectionCapacity":       0.0,
            "onlyConsiderVisibleSurfaceOfModel": False,
            "considerErrorofNormalAngles":       False,
            "scoreLevel":                        0.0,
            "confidenceThreshold":               0.1,
            "candidateTopNum":                   1,
        }

    def _default_coarse_remaining(self, regime: dict) -> dict:
        """Geometry-derived coarse params (excludes refStep/distQ) for look-ahead."""
        ws = self.opt.ws
        snapped_pairs = min(self._pairs_candidates,
                            key=lambda x: abs(x - ws.maxNumOfPointPairsPerFeature))
        base = {
            "registrationMode":             regime["coarse_mode"],
            "angleQuantification":          ws.angleQuantification,
            "maxNumOfPointPairsPerFeature": snapped_pairs,
            "maxVoteRatio":                 0.5,
            "referredStep":                 1,
            "useDistanceNMS":               True,
            "outputNum":                    1,
            "minVoxelLength":               ws.minVoxelLength_mm,
            "maxVoxelLength":               ws.maxVoxelLength_mm,
        }
        if regime["coarse_mode"] == 1.0:
            base["filterCandidatePoseByAxis"] = True
            base["angleThreshold"]            = 135
        return base

    # ─────────────────────────────────────────────────────────────────────
    # Study factories
    # ─────────────────────────────────────────────────────────────────────

    def _create_study(self, name: str, storage_suffix: str = "") -> optuna.Study:
        """TPE study — used by Phase 1/4 helpers delegated to Optimizer."""
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=SC.OPTUNA_N_STARTUP,
            seed=self._seed,
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=2, n_warmup_steps=1, interval_steps=1)
        storage = None
        if self._storage_path and storage_suffix:
            storage = f"sqlite:///{self._storage_path}{storage_suffix}.db"
        return optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner,
            storage=storage, study_name=name, load_if_exists=True)

    def _create_study_joint(self, name: str, storage_suffix: str = "") -> optuna.Study:
        """CmaEsSampler study for the joint 18D coarse+fine optimisation."""
        sampler = optuna.samplers.CmaEsSampler(
            with_margin=True,
            n_startup_trials=SC.OPTUNA_N_STARTUP,
            seed=self._seed,
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=2, n_warmup_steps=1, interval_steps=1)
        storage = None
        if self._storage_path and storage_suffix:
            storage = f"sqlite:///{self._storage_path}{storage_suffix}.db"
        return optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner,
            storage=storage, study_name=name, load_if_exists=True)

    # ─────────────────────────────────────────────────────────────────────
    # Joint objective
    # ─────────────────────────────────────────────────────────────────────

    def _objective_joint(
        self,
        trial: optuna.Trial,
        pairs_candidates: List[int],
        voxel_bounds: Tuple[float, float, float, float],
        coarse_mode_max: int = 1,
    ) -> float:
        """Joint CmaEsSampler objective — all coarse+fine params in one study."""
        flat     = suggest_params_joint(trial, pairs_candidates, voxel_bounds,
                                        self._refstep_bounds, coarse_mode_max)
        coarse_p, fine_p = _joint_params_to_dicts(flat, pairs_candidates)

        all_scenes = self.opt._sample_scenes(SC.M_FULL)
        total_cov  = 0.0
        total_time = 0.0

        for step, scene in enumerate(all_scenes):
            res = self.opt.evaluate_config(
                coarse_p, fine_p, [scene],
                SC.POS_THRESH_TIGHT, SC.ANG_THRESH_REGIME_GATE)

            total_cov  += res.coverage
            total_time += res.mean_time
            running_cov   = total_cov  / (step + 1)
            running_time  = total_time / (step + 1)
            running_score = (running_time / SC.SCORE_TIME_NORM
                             + (1.0 - running_cov) * SC.SCORE_COV_NORM)

            trial.report(running_score, step=step)
            if trial.should_prune():
                trial.set_user_attr("prune_reason", f"median@{step}")
                raise optuna.TrialPruned()

            # Time guard — prune blowup configs early
            if (step >= 1
                    and self._best_mean_time < float("inf")
                    and running_time > self._best_mean_time * SC.OPTUNA_TIME_RATIO):
                trial.set_user_attr("prune_reason",
                                    f"time@{step} {running_time:.2f}s>{self._best_mean_time*SC.OPTUNA_TIME_RATIO:.2f}s")
                raise optuna.TrialPruned()

            # Coverage floor — step is 0-indexed; fires after 3rd scene (step=2)
            if step >= 2 and running_cov < SC.OPTUNA_COV_PRUNE_FLOOR:
                trial.set_user_attr("prune_reason",
                                    f"cov@{step} {running_cov:.3f}<{SC.OPTUNA_COV_PRUNE_FLOOR}")
                raise optuna.TrialPruned()

        final_cov   = total_cov  / SC.M_FULL
        final_time  = total_time / SC.M_FULL
        final_score = (final_time / SC.SCORE_TIME_NORM
                       + (1.0 - final_cov) * SC.SCORE_COV_NORM)

        if final_cov > 0.0:
            self._best_mean_time = min(self._best_mean_time, final_time)

        return final_score

    # ─────────────────────────────────────────────────────────────────────
    # Main run loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> Optional[EvalResult]:
        """Execute full optimisation: Phase 1 → joint rounds → Phase 4."""
        t0 = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"OptunaOptimizer (joint CmaEsSampler): part={self.opt.part_name}  "
                 f"joint={self.n_trials_joint}  refine={self.n_trials_joint_refine}  "
                 f"n_rounds={self.n_rounds}  seed={self._seed}  M_FULL={SC.M_FULL}  "
                 f"cache={'ON' if self.opt.cache else 'OFF'}")

        # ── Phase 1: regime gate ──────────────────────────────────────────
        passing = self.opt.phase1_regime_gate()
        if not passing:
            log.error("Optimisation failed at Phase 1 — no regime passes.")
            return None
        best_regime     = passing[0]
        has_edge        = any(r["needs_edge"] for r in passing)
        coarse_mode_max = 1 if has_edge else 0
        log.info(f"Phase 1 done: best regime = {best_regime['id']}  "
                 f"(cov={best_regime['coverage']:.2f}  edge_available={has_edge}  "
                 f"coarse_mode_max={coarse_mode_max})")

        part = self.opt.part_name
        ws   = self.opt.ws

        # ── Pre-study look-ahead ─────────────────────────────────────────
        # Always warm-start from surface/surface (regime A); joint study explores
        # both modes freely within [0, coarse_mode_max].
        log.info("=" * 60)
        log.info("PRE-STUDY LOOK-AHEAD — geometry warm-start + default fine")
        _surface_regime = {"coarse_mode": 0.0, "fine_mode": 0.0}
        _geom_coarse = self._default_coarse_remaining(_surface_regime)
        _geom_coarse.update({"refStep": ws.refStep, "distQuantification": 1.0})
        _geom_fine   = self._default_fine(_surface_regime)
        _la_scenes   = self.opt._sample_scenes(SC.M_FULL)
        _la_result   = self.opt.evaluate_config(
            _geom_coarse, _geom_fine, _la_scenes,
            SC.POS_THRESH_TIGHT, SC.ANG_THRESH_REGIME_GATE)
        _pos_errs    = [e for s in _la_result.per_scene
                        for e in s.get("pos_errors", []) if e is not None]
        _median_err  = float(np.median(_pos_errs)) if _pos_errs else 0.01
        _op_hint     = int(SC.phase3_approach_candidates(_median_err)[0])
        log.info(f"Pre-study look-ahead: median_pos_err={_median_err*1e3:.2f}mm → "
                 f"opApproach hint={_op_hint}")

        # ── Multi-round joint optimisation ───────────────────────────────
        self._best_mean_time = SC.OPTUNA_TIME_INITIAL_CAP
        prev_best_score  = float("inf")
        prev_best_coarse: Optional[dict] = None
        prev_best_fine:   Optional[dict] = None
        best_coarse:      Optional[dict] = None
        best_fine:        Optional[dict] = None

        for round_idx in range(self.n_rounds):
            sfx      = f"_r{round_idx}"
            n_trials = (self.n_trials_joint if round_idx == 0
                        else self.n_trials_joint_refine)

            log.info("=" * 60)
            log.info(f"ROUND {round_idx} — joint CmaEsSampler ({n_trials} trials)")

            study = self._create_study_joint(
                f"{part}_joint{sfx}", f"_{part}_joint{sfx}")
            self._studies_joint.append(study)

            if not study.trials:
                if round_idx == 0:
                    _warm_fine_for_hint = dict(_geom_fine,
                                               operationApproach=float(_op_hint))
                    _warm = _build_warm_joint(
                        _geom_coarse, _warm_fine_for_hint,
                        self._pairs_candidates, self._voxel_bounds,
                        self._refstep_bounds)
                else:
                    _warm = _build_warm_joint(
                        prev_best_coarse, prev_best_fine,
                        self._pairs_candidates, self._voxel_bounds,
                        self._refstep_bounds)
                study.enqueue_trial(_warm)
                log.info(f"Enqueued warm-start trial for round {round_idx}")
            else:
                log.info(f"Resumed round {round_idx}: {len(study.trials)} prior trials")

            n_done    = sum(1 for t in study.trials
                            if t.state != optuna.trial.TrialState.WAITING)
            remaining = max(0, n_trials - n_done)
            if remaining > 0:
                study.optimize(
                    lambda t, _cmmax=coarse_mode_max: self._objective_joint(
                        t, self._pairs_candidates, self._voxel_bounds, _cmmax),
                    n_trials=remaining,
                    callbacks=[_budget_audit_callback],
                )

            complete = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
            if not complete:
                log.error(f"Round {round_idx} produced no complete trials.")
                if best_coarse is None:
                    return None
                log.warning("Using best config from previous round.")
                break

            round_score          = study.best_value
            best_coarse, best_fine = _joint_params_to_dicts(
                study.best_trial.params, self._pairs_candidates)
            log.info(f"Round {round_idx} best: score={round_score:.4f}  "
                     f"refStep={best_coarse.get('refStep')}  "
                     f"opApproach={best_fine.get('operationApproach')}")

            if round_idx > 0:
                improvement = prev_best_score - round_score
                if improvement < SC.OPTUNA_SCORE_IMPROVE_MIN:
                    log.info(f"Round {round_idx}: improvement {improvement:.4f} "
                             f"< {SC.OPTUNA_SCORE_IMPROVE_MIN}, stopping.")
                    break

            prev_best_score  = round_score
            prev_best_coarse = dict(best_coarse)
            prev_best_fine   = dict(best_fine)

        # ── Post-study: re-evaluate best with tight angular threshold ─────
        scenes_final = self.opt._sample_scenes(SC.M_FULL)
        best_result  = self.opt.evaluate_config(
            best_coarse, best_fine, scenes_final,
            SC.POS_THRESH_TIGHT, SC.ANG_THRESH_TIGHT)
        log.info(f"Re-eval (±{SC.POS_THRESH_TIGHT*1e3:.0f}mm ±{SC.ANG_THRESH_TIGHT:.0f}°): "
                 f"cov={best_result.coverage:.3f}  time={best_result.mean_time:.3f}s  "
                 f"score={best_result.score:.4f}  quality={best_result.score_quality:.3f}")

        # ── Phase 4: symmetry (conditional) ──────────────────────────────
        if best_result.coverage >= SC.PHASE_GATES["after_phase2"][0]:
            final_result = self.opt.phase4_symmetry(best_coarse, best_fine, best_result)
        else:
            log.info(f"PHASE 4 skipped: coverage {best_result.coverage:.2f} "
                     f"< gate {SC.PHASE_GATES['after_phase2'][0]}")
            final_result = best_result

        # ── Summary ──────────────────────────────────────────────────────
        elapsed    = time.time() - t0
        n_complete = sum(1 for s in self._studies_joint
                         for t in s.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned   = sum(1 for s in self._studies_joint
                         for t in s.trials
                         if t.state == optuna.trial.TrialState.PRUNED)
        cache_stats = self.opt.cache.stats() if self.opt.cache else {}

        log.info(f"\n{'='*60}")
        log.info(f"OPTUNA COMPLETE: {self.opt.part_name}")
        log.info(f"  coverage   = {final_result.coverage:.3f}")
        log.info(f"  mean_time  = {final_result.mean_time:.3f} s")
        log.info(f"  score      = {final_result.score:.4f}")
        log.info(f"  quality    = {final_result.score_quality:.3f}")
        log.info(f"  mv_evals   = {self.opt._n_evals}")
        log.info(f"  trials     = {n_complete} complete, {n_pruned} pruned")
        log.info(f"  wall_time  = {elapsed:.0f} s")
        if cache_stats:
            log.info(f"  cache      = {cache_stats}")

        self._log_result_json(final_result)
        return final_result

    # ─────────────────────────────────────────────────────────────────────
    # Export / logging
    # ─────────────────────────────────────────────────────────────────────

    def export_best(self, result: EvalResult, prefix: str = "", suffix: str = "") -> str:
        return self.opt.export_best(result, prefix=prefix, suffix=suffix)

    def _log_result_json(self, result: EvalResult) -> None:
        self.opt._log_result_json(result)

    def cleanup(self) -> None:
        self.opt.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OptunaOptimizer — joint CmaEsSampler auto-tuner for MechVision")
    p.add_argument("--part",            required=True)
    p.add_argument("--scenes_dir",      default=None)
    p.add_argument("--n_trials_joint",  type=int, default=None,
                   help=f"Round 0 joint study budget (default: {SC.OPTUNA_N_TRIALS_JOINT})")
    p.add_argument("--n_trials_refine", type=int, default=None,
                   help=f"Round 1+ refinement budget (default: {SC.OPTUNA_N_TRIALS_JOINT_REFINE})")
    p.add_argument("--n_rounds",        type=int, default=None,
                   help=f"Number of optimisation rounds (default: {SC.OPTUNA_N_ROUNDS})")
    p.add_argument("--m_full",          type=int, default=None)
    p.add_argument("--dry_run",         action="store_true")
    p.add_argument("--no_cache",        action="store_true")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--export_best",     default=True, action="store_true")
    p.add_argument("--storage",         default=None,
                   help="SQLite path prefix (e.g. MM_Optimizer/results/optuna); "
                        "creates {prefix}_{part}_joint_r{k}.db per round.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)

    scenes_root = (args.scenes_dir or
                   os.path.join(_ROOT, "output", "synthetic_target", args.part))
    scene_groups = (list_synthetic_scenes(scenes_root)
                    if os.path.isdir(scenes_root) else [])
    if not scene_groups:
        log.error(f"No scene directories found under: {scenes_root}")
        sys.exit(1)
    log.info(f"Found {len(scene_groups)} M-scenes")

    if args.m_full is not None:
        SC.M_FULL = args.m_full
    else:
        SC.M_FULL  = len(scene_groups)
        SC.M_SMALL = max(1, len(scene_groups) // 2)

    model_path = os.path.join(MM_MODEL_ROOT, f"{args.part}_surface",
                              f"{args.part}_surface.ply")
    if not os.path.exists(model_path):
        log.error(f"Reference model not found: {model_path}")
        sys.exit(1)
    pcd = load_reference_pcd(model_path)
    ws  = analyze_mesh(pcd)

    if args.dry_run:
        client     = None
        project_id = -1
    else:
        client   = MechVisionClient()
        projects = client.get_projects()
        if PROJ_NAME not in projects:
            log.error(f"Project '{PROJ_NAME}' not found.")
            sys.exit(1)
        project_id = projects[PROJ_NAME]

    cache_path = os.path.join(RESULTS_DIR, f"eval_cache_{args.part}.json")
    cache = (None if args.no_cache
             else EvalCache(cache_path, enabled=ENABLE_CACHE))

    storage_path = args.storage
    if storage_path is None and not args.dry_run:
        storage_path = os.path.join(RESULTS_DIR, "optuna")

    opt = OptunaOptimizer(
        part_name             = args.part,
        client                = client,
        project_id            = project_id,
        scene_groups          = scene_groups,
        warm_start            = ws,
        cache                 = cache,
        dry_run               = args.dry_run,
        n_trials_joint        = args.n_trials_joint,
        n_trials_joint_refine = args.n_trials_refine,
        n_rounds              = args.n_rounds,
        seed                  = args.seed,
        storage_path          = storage_path,
    )

    try:
        result = opt.run()
        if result is None:
            log.error("Optimisation did not converge.")
            sys.exit(1)
        if args.export_best:
            opt.export_best(result, prefix="optuna_")
        log.info("Done.")
    finally:
        opt.cleanup()
        if not args.dry_run and client is not None:
            client.close()


if __name__ == "__main__":
    main()
