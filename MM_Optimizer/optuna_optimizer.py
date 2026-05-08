"""
optuna_optimizer.py — Fully Joint Multivariate TPE optimizer for MechVision pose estimation
---------------------------------------------------------------------------------------------
Replaces all three staged studies (Stage 1a/1b/2) with a single joint study covering
all 16–18 coarse+fine parameters at once using multivariate TPE.

Architecture
------------
- All params (refStep, distQ, coarse remaining, fine) suggested in one `suggest_params_joint`
  call per trial. No stage isolation between coarse and fine.
- The 30-trial refStep×distQ grid is preserved as enqueued warm-start trials (with default
  fine params) so `_best_mean_time` is seeded early and initial coverage diversity is ensured.
- Edge-only params (filterCandidatePoseByAxis, angleThreshold) are conditionally suggested
  only when coarse_mode=1 (edge). group=True ensures separate joint models for surface vs edge
  trials, preventing missing-value noise from contaminating either model.
- Multi-round: same study extended by calling study.optimize() again in round 1+. TPE's
  accumulated density model is preserved across rounds — no new study creation needed.

Key design decisions
--------------------
- TPESampler(multivariate=True, group=True, n_startup_trials=20): joint kernel with automatic
  parameter grouping for conditional suggests.
- create_study(directions=["minimize","minimize"]): multi-objective (coverage_loss, mean_time).
  Winner selected from Pareto front: max coverage first, min time as tiebreaker.
- Two-level pruning: explicit time guard + coverage floor (trial.report not supported in MOO).
- Scoring (pruning signal only): raw_score = mean_time/SCORE_TIME_NORM + (1-cov)*SCORE_COV_NORM.
  Multi-objective objective returns (cov_loss, mean_time) tuple — not the scalarized score.
- Phases 0 (mesh analysis warm-start), 1 (regime gate), and 4 (symmetry) are unchanged and
  delegate to the wrapped Optimizer instance.
- Visualization: after run(), exports all Optuna plots to a timestamped folder.
- SQLite storage enables crash-resume (skipped in dry_run).

CLI
---
  python MM_Optimizer/optuna_optimizer.py --part 25333MB000 [--dry_run]
      [--n_trials_joint N] [--n_rounds N]
      [--scenes_dir PATH] [--m_full N] [--no_cache] [--seed N]
      [--export_best] [--storage PATH]
"""

import argparse
import logging
import os
import random
import sys
import time
from typing import List, Optional, Tuple

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
# suggest_params_joint — all 16/18D params in one function
# ─────────────────────────────────────────────────────────────────────────────

def suggest_params_joint(
    trial: optuna.Trial,
    regime: dict,
    pairs_candidates: List[int],
    voxel_bounds: Tuple[float, float, float, float],
) -> dict:
    """Suggest all coarse + fine params jointly.

    Conditional edge-only params (filterCandidatePoseByAxis, angleThreshold) are only
    suggested when coarse_mode=1. group=True in the sampler ensures surface and edge
    trials form separate joint groups, preventing missing-value noise.

    Returns flat dict suitable for splitting into coarse/fine via _split_joint_params().
    """
    min_lo, min_hi, width_lo, width_hi = voxel_bounds
    lo_ref, hi_ref = SC.REFSTEP_BOUNDS
    dlo, dhi       = SC.OPTUNA_DISTQ_BOUNDS
    vlo, vhi       = SC.OPTUNA_VOTERATIO_BOUNDS
    rlo            = SC.REFSTEP_BOUNDS[0]   # lower = 1
    olo, ohi       = SC.OPTUNA_OUTPUTNUM_BOUNDS

    # ── Coarse params ─────────────────────────────────────────────────────────
    coarse_mode         = regime["coarse_mode"]   # fixed by Phase 1; not a free param

    refStep             = trial.suggest_int(  "refStep",  lo_ref, hi_ref)
    distQ               = trial.suggest_float("distQ",    dlo,    dhi)
    angleQuantification = trial.suggest_categorical("angleQuantification", SC.OPTUNA_ANGLQ_CHOICES)
    pairs_i             = trial.suggest_int(  "pairs_idx",  0, len(pairs_candidates) - 1)
    voteRatio           = trial.suggest_float("maxVoteRatio", vlo, vhi)
    referredStep        = trial.suggest_int("referredStep", rlo, refStep)
    useDistNMS          = trial.suggest_categorical("useDistNMS", [True, False])
    outputNum           = trial.suggest_int("outputNum", olo, ohi)
    min_vox             = trial.suggest_float("minVoxelLength_mm", min_lo, min_hi)
    vox_w               = trial.suggest_float("voxel_width_mm",    width_lo, width_hi)

    # Edge-only params — conditionally suggested so TPE models them in a separate group
    atlo, athi = SC.OPTUNA_ANGLETHRESH_BOUNDS
    if coarse_mode == 1.0:
        filter_by_axis = trial.suggest_categorical("filterByAxis", [True, False])
        if filter_by_axis:
            angle_thresh = trial.suggest_int("angleThreshold", atlo, athi)
        else:
            angle_thresh = 90   # MechVision ignores when filterByAxis=False
    else:
        filter_by_axis = True   # surface mode: MechVision ignores these
        angle_thresh   = 135

    # ── Fine params ───────────────────────────────────────────────────────────
    fine_mode  = regime["fine_mode"]   # fixed by Phase 1
    oplo, ophi = SC.OPTUNA_OPAPP_BOUNDS
    dvlo, dvhi = SC.OPTUNA_DEVCAP_BOUNDS
    opApproach = trial.suggest_int("opApproach", oplo, ophi)
    devCap     = trial.suggest_int("devCap",     dvlo, dvhi)
    visibleSurf = trial.suggest_categorical("visibleSurf", [True, False])
    normalAng   = trial.suggest_categorical("normalAng",   [True, False])

    return {
        # ── coarse ─────────────────────────────────────────────────────────
        "coarse_mode":         coarse_mode,
        "refStep":             refStep,
        "distQuantification":  distQ,
        "angleQuantification": angleQuantification,
        "pairs_idx":           pairs_i,
        "maxNumOfPointPairsPerFeature": pairs_candidates[pairs_i],
        "maxVoteRatio":        voteRatio,
        "referredStep":        referredStep,
        "useDistanceNMS":      useDistNMS,
        "outputNum":           outputNum,
        "minVoxelLength_mm":   min_vox,
        "voxel_width_mm":      vox_w,
        "minVoxelLength":      min_vox,
        "maxVoxelLength":      min_vox + vox_w,
        "filterCandidatePoseByAxis": filter_by_axis,
        "angleThreshold":      angle_thresh,
        # ── fine ───────────────────────────────────────────────────────────
        "fine_mode":           fine_mode,
        "operationApproach":   float(opApproach),
        "deviationCorrectionCapacity": float(devCap),
        "onlyConsiderVisibleSurfaceOfModel": visibleSurf,
        "considerErrorofNormalAngles":       normalAng,
    }


def _split_joint_params(p: dict) -> Tuple[dict, dict]:
    """Convert flat joint param dict → (coarse_dict, fine_dict)."""
    coarse_mode = p["coarse_mode"]
    fine_mode   = p["fine_mode"]
    coarse = {
        "registrationMode":             coarse_mode,
        "refStep":                      p["refStep"],
        "distQuantification":           p["distQuantification"],
        "angleQuantification":          p["angleQuantification"],
        "maxNumOfPointPairsPerFeature": p["maxNumOfPointPairsPerFeature"],
        "maxVoteRatio":                 p["maxVoteRatio"],
        "referredStep":                 p["referredStep"],
        "useDistanceNMS":               p["useDistanceNMS"],
        "outputNum":                    p["outputNum"],
        "minVoxelLength":               p["minVoxelLength"],
        "maxVoxelLength":               p["maxVoxelLength"],
    }
    if coarse_mode == 1.0:
        coarse["filterCandidatePoseByAxis"] = p["filterCandidatePoseByAxis"]
        coarse["angleThreshold"]            = p["angleThreshold"]

    fine = {
        "registrationMode":                  fine_mode,
        "operationApproach":                 p["operationApproach"],
        "deviationCorrectionCapacity":       p["deviationCorrectionCapacity"],
        "onlyConsiderVisibleSurfaceOfModel": p["onlyConsiderVisibleSurfaceOfModel"],
        "considerErrorofNormalAngles":       p["considerErrorofNormalAngles"],
        "scoreLevel":                        0.0,
        "confidenceThreshold":               0.1,
        "candidateTopNum":                   1,
    }
    return coarse, fine


def _expand_winner_params(
    trial_params: dict,
    regime: dict,
    pairs_candidates: List[int],
) -> dict:
    """Reconstruct the full expanded param dict from winner.params (short trial-level names).

    winner.params only contains names as used in trial.suggest_*; this mirrors the
    transformation in suggest_params_joint to produce the same expanded dict.
    """
    p = trial_params
    pairs_i  = p["pairs_idx"]
    min_vox  = p["minVoxelLength_mm"]
    vox_w    = p["voxel_width_mm"]
    coarse_mode = regime["coarse_mode"]
    fine_mode   = regime["fine_mode"]

    if coarse_mode == 1.0:
        filter_by_axis = p.get("filterByAxis", True)
        angle_thresh   = p.get("angleThreshold", 90) if filter_by_axis else 90
    else:
        filter_by_axis = True
        angle_thresh   = 135

    return {
        "coarse_mode":         coarse_mode,
        "refStep":             p["refStep"],
        "distQuantification":  p["distQ"],
        "angleQuantification": p["angleQuantification"],
        "pairs_idx":           pairs_i,
        "maxNumOfPointPairsPerFeature": pairs_candidates[pairs_i],
        "maxVoteRatio":        p["maxVoteRatio"],
        "referredStep":        p["referredStep"],
        "useDistanceNMS":      p["useDistNMS"],
        "outputNum":           p["outputNum"],
        "minVoxelLength_mm":   min_vox,
        "voxel_width_mm":      vox_w,
        "minVoxelLength":      min_vox,
        "maxVoxelLength":      min_vox + vox_w,
        "filterCandidatePoseByAxis": filter_by_axis,
        "angleThreshold":      angle_thresh,
        "fine_mode":           fine_mode,
        "operationApproach":   float(p["opApproach"]),
        "deviationCorrectionCapacity": float(p["devCap"]),
        "onlyConsiderVisibleSurfaceOfModel": p["visibleSurf"],
        "considerErrorofNormalAngles":       p["normalAng"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_warm_joint(
    coarse_dict: dict,
    fine_dict: dict,
    regime: dict,
    pairs_candidates: List[int],
    voxel_bounds: Tuple[float, float, float, float],
) -> dict:
    """Convert a full (coarse, fine) config dict → flat joint trial params for enqueue.

    Snaps pairs and angleQ to nearest valid categorical value. Clamps continuous params to bounds.
    Clamps referredStep to refStep so the enqueued trial is valid (external dict may violate this).
    """
    min_lo, min_hi, width_lo, width_hi = voxel_bounds
    vlo, vhi       = SC.OPTUNA_VOTERATIO_BOUNDS
    olo, ohi       = SC.OPTUNA_OUTPUTNUM_BOUNDS
    lo_ref, hi_ref = SC.REFSTEP_BOUNDS
    rlo, rhi       = SC.REFSTEP_BOUNDS   # same as REFSTEP_BOUNDS (1–20)
    dlo, dhi       = SC.OPTUNA_DISTQ_BOUNDS

    # Snap angleQ to nearest valid categorical value
    aq_choices = SC.OPTUNA_ANGLQ_CHOICES
    aq_val = coarse_dict.get("angleQuantification", aq_choices[0])
    aq_snap = min(aq_choices, key=lambda c: abs(c - aq_val))

    # Snap pairs to nearest candidate index
    pairs_val = coarse_dict.get("maxNumOfPointPairsPerFeature", pairs_candidates[0])
    pairs_idx = min(range(len(pairs_candidates)),
                    key=lambda i: abs(pairs_candidates[i] - pairs_val))

    min_vox = float(np.clip(coarse_dict.get("minVoxelLength",
                                            coarse_dict.get("minVoxelLength_mm", 1.0)),
                            min_lo, min_hi))
    max_vox = coarse_dict.get("maxVoxelLength",
                               coarse_dict.get("maxVoxelLength_mm", min_vox + 14.0))
    vox_w   = float(np.clip(max_vox - min_vox, width_lo, width_hi))

    # Fine integer indices
    oplo, ophi = SC.OPTUNA_OPAPP_BOUNDS
    dvlo, dvhi = SC.OPTUNA_DEVCAP_BOUNDS
    op  = int(np.clip(round(fine_dict.get("operationApproach", 1.0)),           oplo, ophi))
    dev = int(np.clip(round(fine_dict.get("deviationCorrectionCapacity", 0.0)), dvlo, dvhi))

    ref_val      = int(np.clip(coarse_dict.get("refStep", 10), lo_ref, hi_ref))
    referred_val = int(np.clip(coarse_dict.get("referredStep", 1), rlo, rhi))
    referred_val = min(referred_val, ref_val)   # enforce referredStep ≤ refStep

    p: dict = {
        "refStep":              ref_val,
        "distQ":                float(np.clip(coarse_dict.get("distQuantification", 1.0), dlo, dhi)),
        "angleQuantification":  aq_snap,
        "pairs_idx":            pairs_idx,
        "maxVoteRatio":         float(np.clip(coarse_dict.get("maxVoteRatio", 0.5), vlo, vhi)),
        "referredStep":         referred_val,
        "useDistNMS":           bool(coarse_dict.get("useDistanceNMS", True)),
        "outputNum":            int(np.clip(coarse_dict.get("outputNum", 1),       olo, ohi)),
        "minVoxelLength_mm":    min_vox,
        "voxel_width_mm":       vox_w,
        "opApproach":           op,
        "devCap":               dev,
        "visibleSurf":          bool(fine_dict.get("onlyConsiderVisibleSurfaceOfModel", False)),
        "normalAng":            bool(fine_dict.get("considerErrorofNormalAngles", False)),
    }
    if regime["coarse_mode"] == 1.0:
        atlo, athi = SC.OPTUNA_ANGLETHRESH_BOUNDS
        p["filterByAxis"]   = bool(coarse_dict.get("filterCandidatePoseByAxis", True))
        p["angleThreshold"] = int(np.clip(coarse_dict.get("angleThreshold", 135), atlo, athi))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Budget audit callback
# ─────────────────────────────────────────────────────────────────────────────

def _budget_audit_callback(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    if (trial.state in (optuna.trial.TrialState.COMPLETE,
                        optuna.trial.TrialState.PRUNED)
            and trial.number % 10 == 0):
        n_complete = sum(1 for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned   = sum(1 for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED)
        reasons    = [t.user_attrs.get("prune_reason", "?")
                      for t in study.trials
                      if t.state == optuna.trial.TrialState.PRUNED]
        cov_p  = sum(1 for r in reasons if r.startswith("cov"))
        med_p  = sum(1 for r in reasons if r.startswith("median"))
        time_p = sum(1 for r in reasons if r.startswith("time"))
        best_str = ""
        if study.best_trials:
            best = min(study.best_trials, key=lambda t: (t.values[0], t.values[1]))
            best_str = (f"  best=({1-best.values[0]:.3f}cov, "
                        f"{best.values[1]:.2f}s)")
        log.info(f"  [budget] trial={trial.number:3d}  complete={n_complete}  "
                 f"pruned={n_pruned} (cov:{cov_p} median:{med_p} time:{time_p})"
                 f"{best_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Pareto front winner selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_pareto_winner(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Select winner from Pareto front: max coverage first, then min time."""
    pareto = study.best_trials
    if not pareto:
        # Fallback: best complete trial by scalarized score
        complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise RuntimeError("No complete trials in study — all were pruned.")
        log.warning("Pareto front is empty — falling back to best scalarized score.")
        return min(complete,
                   key=lambda t: (t.values[0] * SC.SCORE_COV_NORM
                                  + t.values[1] / SC.SCORE_TIME_NORM))
    # values = (cov_loss, mean_time); sort by cov_loss first (lower=higher cov), then time
    return min(pareto, key=lambda t: (t.values[0], t.values[1]))


# ─────────────────────────────────────────────────────────────────────────────
# OptunaOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class OptunaOptimizer:
    """Fully joint multivariate TPE optimizer for MechVision pose estimation.

    Replaces Phases 2–3–5–6 of the hierarchical coordinate descent with a
    single joint multi-objective TPE study. Phases 0, 1, and 4 are unchanged.

    Parameters
    ----------
    part_name      : Part identifier (matches model dir and scene dir names).
    client         : Connected MechVisionClient, or None for dry_run.
    project_id     : Integer project ID for PROJ_NAME.
    scene_groups   : List[List[str]] — one inner list per scene_MMMMM directory.
    warm_start     : WarmStart from mesh_analysis.analyze_mesh().
    cache          : EvalCache instance, or None to disable.
    dry_run        : Build param dicts but do not call MechVision.
    n_trials_joint : Round 0 trial budget (default: SC.OPTUNA_N_TRIALS_JOINT).
    n_rounds       : Number of optimization rounds (default: SC.OPTUNA_N_ROUNDS).
    seed           : Random seed for reproducibility.
    storage_path   : SQLite DB prefix for crash-resume, e.g. "results/optuna".
                     DB created as {prefix}_{part}_joint.db.
                     None = in-memory study (no persistence).
    """

    def __init__(
        self,
        part_name:      str,
        client,
        project_id:     int,
        scene_groups:   List[List[str]],
        warm_start,
        cache:          Optional[EvalCache] = None,
        dry_run:        bool = False,
        n_trials_joint: Optional[int] = None,
        n_rounds:       Optional[int] = None,
        seed:           int = 42,
        storage_path:   Optional[str] = None,
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
        self.n_trials_joint  = (n_trials_joint if n_trials_joint is not None
                                else SC.OPTUNA_N_TRIALS_JOINT)
        self.n_rounds        = n_rounds if n_rounds is not None else SC.OPTUNA_N_ROUNDS
        self._seed           = seed
        self._storage_path   = storage_path
        self._best_mean_time: float = SC.OPTUNA_TIME_INITIAL_CAP

        ws = warm_start
        warm_pairs = ws.maxNumOfPointPairsPerFeature
        self._pairs_candidates: List[int] = sorted(set(
            max(1, min(10000, int(warm_pairs * s)))
            for s in SC.PHASE2B_PAIRS_SCALES
        ))
        self._voxel_bounds: Tuple[float, float, float, float] = (
            max(0.1, ws.minVoxelLength_mm * 0.2),
            ws.minVoxelLength_mm * 6.0,
            max(0.5, ws.maxVoxelLength_mm * 0.1),
            ws.maxVoxelLength_mm * 6.0,
        )
        self._study: Optional[optuna.Study] = None

    # ─────────────────────────────────────────────────────────────────────
    # Default param helpers
    # ─────────────────────────────────────────────────────────────────────

    def _default_fine(self, regime: dict) -> dict:
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
        ws = self.opt.ws
        snapped_pairs = min(self._pairs_candidates,
                            key=lambda x: abs(x - ws.maxNumOfPointPairsPerFeature))
        base = {
            "registrationMode":             regime["coarse_mode"],
            "refStep":                      SC.REFSTEP_BOUNDS[1] // 2,
            "distQuantification":           ws.distQuantification,
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
    # Study factory
    # ─────────────────────────────────────────────────────────────────────

    def _create_study_joint(self, name: str, storage_suffix: str = "") -> optuna.Study:
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            n_startup_trials=SC.OPTUNA_N_STARTUP_JOINT,
            seed=self._seed,
        )
        # trial.report/should_prune are not supported in multi-objective mode;
        # pruning is handled explicitly via time guard + coverage floor in _objective_joint.
        storage = None
        if self._storage_path and storage_suffix:
            storage = f"sqlite:///{self._storage_path}{storage_suffix}.db"
        return optuna.create_study(
            directions     = ["minimize", "minimize"],
            sampler        = sampler,
            storage        = storage,
            study_name     = name,
            load_if_exists = True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────
    # Objective
    # ─────────────────────────────────────────────────────────────────────

    def _objective_joint(
        self,
        trial: optuna.Trial,
        regime: dict,
    ) -> Tuple[float, float]:
        """Scene-by-scene evaluation with three-level pruning.

        Returns (coverage_loss, mean_time) for multi-objective optimization.
        Pruning uses a scalarized running_score so trial.report() receives a scalar.
        """
        p        = suggest_params_joint(trial, regime, self._pairs_candidates,
                                        self._voxel_bounds)
        coarse_p, fine_p = _split_joint_params(p)

        ang        = SC.ANG_THRESH_REGIME_GATE
        all_scenes = self.opt._sample_scenes(SC.M_FULL)
        total_cov  = 0.0
        total_time = 0.0

        for step, scene in enumerate(all_scenes):
            res = self.opt.evaluate_config(
                coarse_p, fine_p, [scene], SC.POS_THRESH_TIGHT, ang)

            total_cov  += res.coverage
            total_time += res.mean_time
            running_cov   = total_cov  / (step + 1)
            running_time  = total_time / (step + 1)
            # logging.info(f"  Trial {trial.number:3d}  Scene {step+1}/{SC.M_FULL}  "
            #              f"cov={res.coverage:.3f}  mean_time={res.mean_time:.2f}s  "
            #              f"running_cov={running_cov:.3f}  running_time={running_time:.2f}s")
            # Level 2: Time guard  (trial.report/should_prune not available in multi-objective)
            if running_time > self._best_mean_time * SC.OPTUNA_TIME_RATIO:
                trial.set_user_attr("prune_reason", f"time@{step}")
                raise optuna.TrialPruned()

            # Level 3: Coverage floor (after 3rd scene)
            if step >= 2 and running_cov < SC.OPTUNA_COV_PRUNE_FLOOR:
                trial.set_user_attr("prune_reason", f"cov@{step}, running_cov={running_cov:.2f}")
                raise optuna.TrialPruned()

        final_cov  = total_cov  / SC.M_FULL
        final_time = total_time / SC.M_FULL

        if final_cov > 0.0:
            self._best_mean_time = min(self._best_mean_time, final_time)

        return (1.0 - final_cov, final_time)

    # ─────────────────────────────────────────────────────────────────────
    # Main run loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> Optional[EvalResult]:
        """Execute full optimization: Phase 1 → joint TPE study (multi-round) → Phase 4."""
        t0 = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"OptunaOptimizer: part={self.opt.part_name}  "
                 f"n_trials_joint={self.n_trials_joint}  n_rounds={self.n_rounds}  "
                 f"seed={self._seed}  M_FULL={SC.M_FULL}  "
                 f"cache={'ON' if self.opt.cache else 'OFF'}")

        # ── Phase 1: regime gate ──────────────────────────────────────────
        passing = self.opt.phase1_regime_gate()
        if not passing:
            log.error("Optimization failed at Phase 1 — no regime passes.")
            return None
        best_regime = passing[0]
        log.info(f"Phase 1 done: best regime = {best_regime['id']}  "
                 f"(cov={best_regime['coverage']:.2f})")

        part = self.opt.part_name

        # ── Pre-study look-ahead: geometry coarse → opApproach hint ──────
        log.info("Pre-study look-ahead: evaluating geometry warm-start config...")
        _geom_coarse = self._default_coarse_remaining(best_regime)
        _geom_fine   = self._default_fine(best_regime)
        _la_result   = self.opt.evaluate_config(
            _geom_coarse, _geom_fine,
            self.opt._sample_scenes(SC.M_FULL),
            SC.POS_THRESH_TIGHT, SC.ANG_THRESH_REGIME_GATE,
        )
        _pos_errs   = [e for s in _la_result.per_scene
                       for e in s.get("pos_errors", []) if e is not None]
        _median_err = float(np.median(_pos_errs)) if _pos_errs else 0.01
        _op_hint    = SC.phase3_approach_candidates(_median_err)[0]
        log.info(f"  median coarse pos err = {_median_err*1e3:.2f}mm → "
                 f"opApproach hint = {_op_hint}")

        # ── Joint study ───────────────────────────────────────────────────
        log.info("=" * 60)
        log.info("JOINT STUDY — fully joint coarse+fine multivariate TPE (multi-objective)")
        self._best_mean_time = SC.OPTUNA_TIME_INITIAL_CAP

        self._study = self._create_study_joint(
            f"{part}_joint", f"_{part}_joint")

        if not self._study.trials:
            warm_p = _build_warm_joint(
                _geom_coarse, _geom_fine, best_regime,
                self._pairs_candidates, self._voxel_bounds,
            )
            self._study.enqueue_trial(warm_p)
            log.info("Enqueued geometry warm-start trial (TPE explores all 16 params jointly)")
        else:
            log.info(f"Resumed study: {len(self._study.trials)} prior trials")

        # ── Multi-round loop (same study, extended) ───────────────────────
        prev_best_score = float("inf")
        for round_idx in range(self.n_rounds):
            n_done = sum(1 for t in self._study.trials
                         if t.state != optuna.trial.TrialState.WAITING)
            if round_idx == 0:
                n_target = self.n_trials_joint
            else:
                n_target = n_done + SC.OPTUNA_N_TRIALS_JOINT_REFINE

            remaining = max(0, n_target - n_done)
            log.info(f"  Round {round_idx}: running {remaining} trials "
                     f"(total target {n_target})")

            if remaining > 0:
                self._study.optimize(
                    lambda t: self._objective_joint(t, best_regime),
                    n_trials=remaining,
                    callbacks=[_budget_audit_callback],
                )

            if not self._study.best_trials:
                log.error(f"Round {round_idx}: no complete trials.")
                if round_idx == 0:
                    return None
                break

            winner      = _select_pareto_winner(self._study)
            round_score = winner.values[0] + winner.values[1]
            log.info(f"  Round {round_idx} best: "
                     f"cov={1-winner.values[0]:.3f}  "
                     f"time={winner.values[1]:.3f}s  "
                     f"score={round_score:.3f}  "
                     f"Pareto front size={len(self._study.best_trials)}")

            if (round_idx > 0
                    and (prev_best_score - round_score) < SC.OPTUNA_SCORE_IMPROVE_MIN):
                log.info(f"  Round {round_idx}: improvement "
                         f"{prev_best_score - round_score:.4f} < threshold — stopping.")
                break
            prev_best_score = round_score

        # ── Extract best config ───────────────────────────────────────────
        winner      = _select_pareto_winner(self._study)
        expanded    = _expand_winner_params(winner.params, best_regime, self._pairs_candidates)
        best_coarse, best_fine = _split_joint_params(expanded)

        log.info(f"Best config: refStep={best_coarse.get('refStep')}  "
                 f"distQ={best_coarse.get('distQuantification', 0):.2f}  "
                 f"opApproach={best_fine.get('operationApproach')}  "
                 f"outputNum={best_coarse.get('outputNum')}")

        # ── Post-study re-eval with tight angular threshold ───────────────
        scenes_final = self.opt._sample_scenes(SC.M_FULL)
        best_result  = self.opt.evaluate_config(
            best_coarse, best_fine, scenes_final,
            SC.POS_THRESH_TIGHT, SC.ANG_THRESH_TIGHT,
        )
        log.info(f"Re-eval (tight ±{SC.POS_THRESH_TIGHT*1e3:.0f}mm "
                 f"±{SC.ANG_THRESH_TIGHT:.0f}°): "
                 f"cov={best_result.coverage:.3f}  "
                 f"time={best_result.mean_time:.3f}s")

        # ── Phase 4: symmetry (conditional) ──────────────────────────────
        if best_result.coverage >= SC.PHASE_GATES["after_phase2"][0]:
            final_result = self.opt.phase4_symmetry(best_coarse, best_fine, best_result)
        else:
            log.info(f"PHASE 4 skipped: coverage {best_result.coverage:.2f} "
                     f"< gate {SC.PHASE_GATES['after_phase2'][0]}")
            final_result = best_result

        # ── Summary ──────────────────────────────────────────────────────
        elapsed    = time.time() - t0
        n_complete = sum(1 for t in self._study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned   = sum(1 for t in self._study.trials
                         if t.state == optuna.trial.TrialState.PRUNED)
        cache_stats = self.opt.cache.stats() if self.opt.cache else {}

        log.info(f"\n{'='*60}")
        log.info(f"OPTUNA COMPLETE: {self.opt.part_name}")
        log.info(f"  coverage      = {final_result.coverage:.3f}")
        log.info(f"  mean_time     = {final_result.mean_time:.3f} s")
        log.info(f"  score         = {final_result.score:.3f}")
        log.info(f"  score_quality = {final_result.score_quality:.3f}")
        log.info(f"  mv_evals      = {self.opt._n_evals}")
        log.info(f"  trials        = {n_complete} complete, {n_pruned} pruned")
        log.info(f"  Pareto size   = {len(self._study.best_trials)}")
        log.info(f"  wall_time     = {elapsed:.0f} s")
        if cache_stats:
            log.info(f"  cache         = {cache_stats}")

        ts = int(time.time())
        self._log_result_json(final_result, ts=ts)
        return final_result

    # ─────────────────────────────────────────────────────────────────────
    # Export / logging
    # ─────────────────────────────────────────────────────────────────────

    def export_best(self, result: EvalResult, prefix: str = "", suffix: str = "") -> str:
        return self.opt.export_best(result, prefix=prefix, suffix=suffix)

    def _log_result_json(self, result: EvalResult, ts: Optional[int] = None) -> None:
        if ts is None:
            ts = int(time.time())
        self.opt._log_result_json(result)

    def cleanup(self) -> None:
        self.opt.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OptunaOptimizer — fully joint multivariate TPE for MechVision")
    p.add_argument("--part",           required=True)
    p.add_argument("--scenes_dir",     default=None)
    p.add_argument("--n_trials_joint", type=int, default=None,
                   help=f"Joint study trial budget (default: {SC.OPTUNA_N_TRIALS_JOINT})")
    p.add_argument("--n_rounds",       type=int, default=None,
                   help=f"Number of optimization rounds (default: {SC.OPTUNA_N_ROUNDS})")
    p.add_argument("--m_full",         type=int, default=None)
    p.add_argument("--dry_run",        action="store_true")
    p.add_argument("--no_cache",       action="store_true")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--export_best",    default=True, action="store_true")
    p.add_argument("--storage",        default=None,
                   help="SQLite path prefix (e.g. MM_Optimizer/results/optuna); "
                        "creates {prefix}_{part}_joint.db")
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
        part_name      = args.part,
        client         = client,
        project_id     = project_id,
        scene_groups   = scene_groups,
        warm_start     = ws,
        cache          = cache,
        dry_run        = args.dry_run,
        n_trials_joint = args.n_trials_joint,
        n_rounds       = args.n_rounds,
        seed           = args.seed,
        storage_path   = storage_path,
    )

    try:
        result = opt.run()
        if result is None:
            log.error("Optimization did not converge.")
            sys.exit(1)
        if args.export_best:
            out = opt.export_best(result, prefix="optuna_")
            log.info(f"Best config exported → {out}")
        log.info("Done.")
    finally:
        opt.cleanup()
        if not args.dry_run and client is not None:
            client.close()


if __name__ == "__main__":
    main()
