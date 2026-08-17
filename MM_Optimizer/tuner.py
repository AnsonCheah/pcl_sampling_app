"""
tuner.py — MechVision pose-estimation parameter tuner
-----------------------------------------------------
`Tuner` extends `MVEvaluator` with a multi-objective Optuna study over all 16-18 coarse
and fine parameters at once. Three samplers share one search space and warm-start seed:
NSGA-II, TPE, and GP; select with --sampler or the `sampler` argument.

Non-obvious behaviour
---------------------
- The objective returns (coverage, mean_time) and the winner comes off the Pareto front:
  max coverage first, min time as tiebreaker. The trade-off is resolved there, not by
  pruning.
- referredStep <= refStep is enforced twice: `constraints_func` on the sampler, plus an
  early-return guard in `_objective`. The guard is what prevents a MechVision blowup;
  the sampler constraint only steers away from that region.
- Pruning is deliberately minimal: a coverage floor and a FIXED absolute time cap. There
  is no competitive time pruning — a `best_mean_time × ratio` guard ratchets down after
  one fast low-quality trial and then prunes most of the search, biasing away from the
  slow-but-accurate region that matters here.
- Edge-only params are suggested only when coarse_mode=1, and the samplers use
  group=True so surface and edge trials form separate joint groups.
- SQLite storage gives crash-resume; rounds extend the same study, so the sampler keeps
  its model across them.

CLI
---
  python MM_Optimizer/tuner.py --part 25333MB000 [--dry_run]
      [--n_trials N] [--n_rounds N] [--sampler nsgaii|tpe|gp]
      [--scenes_dir PATH] [--m_full N] [--no_cache] [--seed N] [--storage PATH]
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
from MM_Optimizer.mv_evaluator  import (MVEvaluator, EvalResult,
                                        PROJ_NAME, MM_MODEL_ROOT,
                                        RESULTS_DIR, ENABLE_CACHE)
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
import MM_Optimizer.search_config as SC

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# suggest_params — all 16/18D params in one function
# ─────────────────────────────────────────────────────────────────────────────

def suggest_params(
    trial: optuna.Trial,
    regime: dict,
    pairs_candidates: List[int],
    voxel_bounds: Tuple[float, float, float, float],
) -> dict:
    """Suggest all coarse + fine params jointly.

    Conditional edge-only params (filterCandidatePoseByAxis, angleThreshold) are only
    suggested when coarse_mode=1. group=True in the sampler ensures surface and edge
    trials form separate joint groups, preventing missing-value noise.

    Returns flat dict suitable for splitting into coarse/fine via _split_params().
    """
    min_lo, min_hi, width_lo, width_hi = voxel_bounds
    lo_ref, hi_ref = SC.REFSTEP_BOUNDS
    dlo, dhi       = SC.DISTQ_BOUNDS
    vlo, vhi       = SC.VOTERATIO_BOUNDS
    rlo            = SC.REFSTEP_BOUNDS[0]   # lower = 1
    olo, ohi       = SC.OUTPUTNUM_BOUNDS

    # ── Coarse params ─────────────────────────────────────────────────────────
    coarse_mode         = regime["coarse_mode"]   # fixed by Phase 1; not a free param

    refStep             = trial.suggest_int(  "refStep",  lo_ref, hi_ref)
    distQ               = trial.suggest_float("distQ",    dlo,    dhi)
    angleQuantification = trial.suggest_categorical("angleQuantification", SC.ANGLQ_CHOICES)
    pairs_i             = trial.suggest_int(  "pairs_idx",  0, len(pairs_candidates) - 1)
    voteRatio           = trial.suggest_float("maxVoteRatio", vlo, vhi)
    referredStep        = trial.suggest_int("referredStep", rlo, hi_ref)
    useDistNMS          = trial.suggest_categorical("useDistNMS", [True, False])
    outputNum           = trial.suggest_int("outputNum", olo, ohi)
    min_vox             = trial.suggest_float("minVoxelLength_mm", min_lo, min_hi)
    vox_w               = trial.suggest_float("voxel_width_mm",    width_lo, width_hi)

    # Edge-only params — conditionally suggested so TPE models them in a separate group
    atlo, athi = SC.ANGLETHRESH_BOUNDS
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
    oplo, ophi = SC.OPAPP_BOUNDS
    dvlo, dvhi = SC.DEVCAP_BOUNDS
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


def _split_params(p: dict) -> Tuple[dict, dict]:
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
    transformation in suggest_params to produce the same expanded dict.
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

def _build_warm(
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
    vlo, vhi       = SC.VOTERATIO_BOUNDS
    olo, ohi       = SC.OUTPUTNUM_BOUNDS
    lo_ref, hi_ref = SC.REFSTEP_BOUNDS
    rlo, rhi       = SC.REFSTEP_BOUNDS   # same as REFSTEP_BOUNDS (1–20)
    dlo, dhi       = SC.DISTQ_BOUNDS

    # Snap angleQ to nearest valid categorical value
    aq_choices = SC.ANGLQ_CHOICES
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
    oplo, ophi = SC.OPAPP_BOUNDS
    dvlo, dvhi = SC.DEVCAP_BOUNDS
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
        atlo, athi = SC.ANGLETHRESH_BOUNDS
        p["filterByAxis"]   = bool(coarse_dict.get("filterCandidatePoseByAxis", True))
        p["angleThreshold"] = int(np.clip(coarse_dict.get("angleThreshold", 135), atlo, athi))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# NSGA-II constraint function
# ─────────────────────────────────────────────────────────────────────────────

def _referredstep_constraint(trial: optuna.trial.FrozenTrial) -> List[float]:
    """NSGA-II constraints_func: returns [violation] where >0 means infeasible.

    Violation = referredStep - refStep when referredStep > refStep, else 0.
    The value is written by _objective's early-return guard before the
    trial completes, so it is available on the FrozenTrial passed here.
    """
    return [float(trial.user_attrs.get("constraint_violation", 0.0))]


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
        time_p = sum(1 for r in reasons if r.startswith("time"))   # absolute cap only
        best_str = ""
        if study.best_trials:
            best = max(study.best_trials, key=lambda t: (t.values[0], -t.values[1]))
            best_str = (f"  best=({best.values[0]:.3f}cov, "
                        f"{best.values[1]:.2f}s)")
        log.info(f"  [budget] trial={trial.number:3d}  complete={n_complete}  "
                 f"pruned={n_pruned} (cov:{cov_p} time:{time_p})"
                 f"{best_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Pareto front winner selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_pareto_winner(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Select winner from Pareto front: max coverage first, then min time."""
    pareto = study.best_trials
    if not pareto:
        # Fallback: best complete trial by scalarized score.
        # Prefer feasible trials (no constraint violation) over sentinel infeasible ones.
        complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise RuntimeError("No complete trials in study — all were pruned.")
        feasible = [t for t in complete
                    if t.user_attrs.get("constraint_violation", 0.0) <= 0.0]
        pool = feasible if feasible else complete
        log.warning(f"Pareto front empty — fallback (feasible={len(feasible)}/{len(complete)})")
        return max(pool,
                   key=lambda t: t.values[0] - t.values[1] / SC.SCORE_TIME_NORM)
    # values = (coverage, mean_time); sort by coverage first (higher=better), then time
    return max(pareto, key=lambda t: (t.values[0], -t.values[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Tuner
# ─────────────────────────────────────────────────────────────────────────────

SAMPLER_CHOICES = SC.SAMPLER_CHOICES
SAMPLER_DEFAULT = SC.SAMPLER_DEFAULT


class Tuner(MVEvaluator):
    """Fully joint Optuna optimizer for MechVision pose estimation.

    Replaces Phases 2–3–5–6 of the hierarchical coordinate descent with a
    single joint multi-objective study. Phases 0, 1, and 4 are unchanged.

    Parameters
    ----------
    part_name      : Part identifier (matches model dir and scene dir names).
    client         : Connected MechVisionClient, or None for dry_run.
    project_id     : Integer project ID for PROJ_NAME.
    scene_groups   : List[List[str]] — one inner list per scene_MMMMM directory.
    warm_start     : WarmStart from mesh_analysis.analyze_mesh().
    cache          : EvalCache instance, or None to disable.
    dry_run        : Build param dicts but do not call MechVision.
    n_trials : Round 0 trial budget (default: SC.N_TRIALS).
    n_rounds       : Number of optimization rounds (default: SC.N_ROUNDS).
    seed           : Random seed for reproducibility.
    storage_path   : SQLite DB path prefix for crash-resume, e.g. "results/"
                     (include a trailing separator to keep DBs in a directory).
                     DB created as {prefix}{part}_{sampler}.db.
                     None = in-memory study (no persistence).
    sampler        : Joint-study sampler — "nsgaii" (default), "tpe", or "gp".
                     All three run multi-objective (coverage, mean_time) over the
                     same suggest_params search space and warm-start seed.
                     "nsgaii" enforces referredStep ≤ refStep via constraints_func;
                     "tpe" uses multivariate=True, group=True to handle the
                     conditional edge-only params;
                     "gp" (Gaussian process) also enforces the referredStep
                     constraint via constraints_func — strong in the low-trial
                     regime but requires torch (CPU build is sufficient).
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
        n_trials: Optional[int] = None,
        n_rounds:       Optional[int] = None,
        seed:           int = 42,
        storage_path:   Optional[str] = None,
        pos_thresh_k:     float = 0.01,
        adaptive_thresh:  bool  = True,
        sampler:          str   = SAMPLER_DEFAULT,
    ):
        if sampler not in SAMPLER_CHOICES:
            raise ValueError(
                f"sampler={sampler!r} not in {SAMPLER_CHOICES}")
        super().__init__(
            part_name    = part_name,
            client       = client,
            project_id   = project_id,
            scene_groups = scene_groups,
            warm_start   = warm_start,
            cache        = cache,
            dry_run      = dry_run,
        )
        self._sampler = sampler
        self.n_trials  = (n_trials if n_trials is not None
                                else SC.N_TRIALS)
        self.n_rounds        = n_rounds if n_rounds is not None else SC.N_ROUNDS
        self._seed           = seed
        self._storage_path   = storage_path

        ws = warm_start
        warm_pairs = ws.maxNumOfPointPairsPerFeature
        self._pairs_candidates: List[int] = sorted(set(
            max(1, min(10000, int(warm_pairs * s)))
            for s in SC.PAIRS_SCALES
        ))
        self._voxel_bounds: Tuple[float, float, float, float] = (
            max(0.1, ws.minVoxelLength_mm * 0.2),
            ws.minVoxelLength_mm * 6.0,
            max(0.5, ws.maxVoxelLength_mm * 0.1),
            ws.maxVoxelLength_mm * 6.0,
        )
        # Adaptive study threshold: clip(0.01 × longest_extent, 2mm, 5mm).
        # Looser than POS_THRESH_TIGHT during the study so NSGA-II gets a
        # gradient signal on flat/long parts where coarse matching lands
        # 2–5mm from GT but fine can still converge. Final re-eval always
        # uses POS_THRESH_TIGHT (2mm).
        self._pos_thresh_study: float = (
            float(np.clip(pos_thresh_k * ws.longest_extent_m,
                          SC.POS_THRESH_TIGHT, SC.POS_THRESH_LOOSE))
            if adaptive_thresh else SC.POS_THRESH_TIGHT
        )
        self._study: Optional[optuna.Study] = None
        # Regime chosen by Phase 1, stored in run() so iter_pareto_configs() can
        # expand Pareto trial params after the study finishes.
        self._best_regime: dict = {}
        # Optional per-trial hook (GUI progress). When set, appended to the study
        # callbacks; Optuna calls it (study, trial) after each trial. Default None
        # → CLI unchanged.
        self.on_trial_complete = None

    # ─────────────────────────────────────────────────────────────────────
    # Default param helpers
    # ─────────────────────────────────────────────────────────────────────

    def _warm_fine(self, regime: dict) -> dict:
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

    def _warm_coarse(self, regime: dict) -> dict:
        ws = self.ws
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

    def _create_study(self, name: str, storage_suffix: str = "") -> optuna.Study:
        """Build the multi-objective study for `self._sampler`.

        All three samplers are multi-objective and accept `constraints_func`, so they are
        interchangeable here; they differ in how many trials they need to pay off. Optuna's
        own recommended budgets, against this project's default of ~150-250 trials:

        nsgaii : NSGA-II genetic algorithm. Recommended 100-10,000 trials — this budget is
                 at the very bottom of its range, and Optuna notes it handles float and
                 integer parameters inefficiently, which is most of this search space.
                 Optuna also documents it as NOT supporting dynamic search spaces.
        tpe    : Tree-structured Parzen Estimator, O(d·n·log n) per trial. Recommended
                 100-1,000 trials. Plain TPE does support dynamic search spaces, but this
                 study needs `multivariate=True`, and multivariate TPE does not: it cannot
                 reuse trials recorded before a range changed, and without `group=True`
                 conditional params degrade to independent random sampling.
        gp     : Gaussian-process Bayesian optimization, O(n³) per trial — the most expensive
                 per trial but the most sample-efficient. Recommended up to 500 trials, which
                 fits this budget; the default. It infers its relative search space via
                 `intersection_search_space()` and delegates anything outside it to the
                 independent sampler, so a parameter whose range moved, or one suggested
                 conditionally, loses GP guidance individually while the rest stay modelled.
                 Needs scipy and torch (a CPU build is enough).

        Why that matters here, in two places:

        1. `referredStep <= refStep`. The natural encoding is a dynamic upper bound,
           `suggest_int("referredStep", 1, refStep)`, which makes the space dynamic by
           construction on every trial. That is why it is NOT written that way: instead
           referredStep takes fixed bounds and the coupling is enforced by
           `_referredstep_constraint` plus the early-return guard in `_objective`. Note the
           guard is the part that actually prevents the MechVision blowup — `constraints_func`
           only steers — and it is wired for nsgaii and gp but not tpe, so under tpe the
           coupling is enforced solely by the guard.
        2. `_voxel_bounds` and `_pairs_candidates` are derived from the part's warm start,
           and studies resume with `load_if_exists=True` in "extend" mode. Regenerate a
           reference cloud and the bounds move while the DB still holds trials sampled under
           the old ones — a dynamic value range within one study. GP degrades per-parameter
           there; NSGA-II has no support for it at all.

        `deterministic_objective=False` for GP because the objective genuinely is noisy:
        scenes are sampled at random and MechVision timings jitter.
        """
        if self._sampler == "nsgaii":
            sampler = optuna.samplers.NSGAIISampler(
                population_size=SC.NSGA_POPULATION_SIZE,
                seed=self._seed,
                constraints_func=_referredstep_constraint,
            )
        elif self._sampler == "tpe":
            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                group=True,
                n_startup_trials=SC.N_STARTUP,
                seed=self._seed,
            )
        elif self._sampler == "gp":
            sampler = optuna.samplers.GPSampler(
                seed=self._seed,
                n_startup_trials=SC.N_STARTUP,
                deterministic_objective=False,
                constraints_func=_referredstep_constraint,
            )
        else:
            raise ValueError(f"Unknown sampler: {self._sampler!r}")
        # trial.report/should_prune are unsupported in multi-objective mode, so pruning is
        # explicit in _objective. TPE ignores the constraint_violation user-attr, which is
        # harmless metadata there.
        storage = None
        if self._storage_path and storage_suffix:
            storage = f"sqlite:///{self._storage_path}{storage_suffix}.db"
        study = optuna.create_study(
            directions     = ["maximize", "minimize"],
            sampler        = sampler,
            storage        = storage,
            study_name     = name,
            load_if_exists = True,
        )
        study.set_metric_names(["coverage", "mean_time"])
        return study

    # ─────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────
    # Objective
    # ─────────────────────────────────────────────────────────────────────

    def _objective(
        self,
        trial: optuna.Trial,
        regime: dict,
    ) -> Tuple[float, float]:
        """Scene-by-scene evaluation returning (coverage, mean_time) for MOO.

        Pruning is minimal and non-competitive: (0) referredStep feasibility
        short-circuit, (1) coverage floor after ≥3 scenes, (2) a fixed absolute
        time safety cap after ≥3 scenes. No time-ratio pruning — the sampler owns
        the time/accuracy trade-off via the Pareto front.
        """
        p        = suggest_params(trial, regime, self._pairs_candidates,
                                        self._voxel_bounds)
        coarse_p, fine_p = _split_params(p)

        # Level 0: referredStep feasibility guard.
        # NSGA-II uses fixed bounds (1–20) for referredStep, so it can suggest
        # referredStep > refStep. Guard here prevents MechVision blowup (5–10 min/scene).
        if p["referredStep"] > p["refStep"]:
            trial.set_user_attr("constraint_violation",
                                float(p["referredStep"] - p["refStep"]))
            return (0.0, SC.TIME_INITIAL_CAP)

        ang        = SC.ANG_THRESH_REGIME_GATE
        all_scenes = self._sample_scenes(SC.M_FULL)
        total_cov  = 0.0
        total_time = 0.0

        for step, scene in enumerate(all_scenes):
            res = self.evaluate_config(
                coarse_p, fine_p, [scene], self._pos_thresh_study, ang)

            total_cov  += res.coverage
            total_time += res.mean_time
            running_cov   = total_cov  / (step + 1)
            running_time  = total_time / (step + 1)
            # Absolute time cap, after 3+ scenes so first-scene noise cannot trip it. Fixed,
            # never competitive -- see the pruning note in this module's docstring.
            if step >= 2 and running_time > SC.TIME_ABS_CAP:
                trial.set_user_attr("prune_reason", f"time@{step}")
                raise optuna.TrialPruned()

            # Level 2: Coverage floor (after 3rd scene)
            if step >= 2 and running_cov < SC.COV_PRUNE_FLOOR:
                trial.set_user_attr("prune_reason", f"cov@{step}, running_cov={running_cov:.2f}")
                raise optuna.TrialPruned()

        final_cov  = total_cov  / SC.M_FULL
        final_time = total_time / SC.M_FULL

        return (final_cov, final_time)

    # ─────────────────────────────────────────────────────────────────────
    # Main run loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> Optional[EvalResult]:
        """Execute full optimization: Phase 1 → joint NSGA-II study (multi-round) → Phase 4."""
        t0 = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"Tuner: part={self.part_name}  "
                 f"n_trials={self.n_trials}  n_rounds={self.n_rounds}  "
                 f"seed={self._seed}  M_FULL={SC.M_FULL}  "
                 f"cache={'ON' if self.cache else 'OFF'}  "
                 f"pos_thresh_study={self._pos_thresh_study*1e3:.1f}mm  "
                 f"pos_thresh_final={SC.POS_THRESH_TIGHT*1e3:.1f}mm")

        # ── Phase 1: regime gate ──────────────────────────────────────────
        passing = self.phase1_regime_gate()
        if not passing:
            log.error("Optimization failed at Phase 1 — no regime passes.")
            return None
        best_regime = passing[0]
        self._best_regime = best_regime   # kept for iter_pareto_configs() after run()
        log.info(f"Phase 1 done: best regime = {best_regime['id']}  "
                 f"(cov={best_regime['coverage']:.2f})")

        part = self.part_name

        # ── Pre-study look-ahead: geometry coarse → opApproach hint ──────
        log.info("Pre-study look-ahead: evaluating geometry warm-start config...")
        _geom_coarse = self._warm_coarse(best_regime)
        _geom_fine   = self._warm_fine(best_regime)
        _la_result   = self.evaluate_config(
            _geom_coarse, _geom_fine,
            self._sample_scenes(SC.M_FULL),
            SC.POS_THRESH_TIGHT, SC.ANG_THRESH_REGIME_GATE,
        )
        _pos_errs   = [e for s in _la_result.per_scene
                       for e in s.get("pos_errors", []) if e is not None]
        _median_err = float(np.median(_pos_errs)) if _pos_errs else 0.01
        _op_hint    = SC.approach_candidates(_median_err)[0]
        log.info(f"  median coarse pos err = {_median_err*1e3:.2f}mm → "
                 f"opApproach hint = {_op_hint}")

        # ── Joint study ───────────────────────────────────────────────────
        log.info("=" * 60)
        log.info(f"JOINT STUDY — fully joint coarse+fine {self._sampler.upper()} "
                 f"(multi-objective)")

        self._study = self._create_study(
            f"{part}_{self._sampler}", f"{part}_{self._sampler}")

        if not self._study.trials:
            warm_p = _build_warm(
                _geom_coarse, _geom_fine, best_regime,
                self._pairs_candidates, self._voxel_bounds,
            )
            self._study.enqueue_trial(warm_p)
            log.info("Enqueued geometry warm-start trial (feasible seed for NSGA-II population)")
        else:
            log.info(f"Resumed study: {len(self._study.trials)} prior trials")

        # ── Multi-round loop (same study, extended) ───────────────────────
        prev_best_score = float("inf")
        for round_idx in range(self.n_rounds):
            n_done = sum(1 for t in self._study.trials
                         if t.state != optuna.trial.TrialState.WAITING)
            if round_idx == 0:
                n_target = self.n_trials
            else:
                n_target = n_done + SC.N_TRIALS_REFINE

            remaining = max(0, n_target - n_done)
            log.info(f"  Round {round_idx}: running {remaining} trials "
                     f"(total target {n_target})")

            if remaining > 0:
                callbacks = [_budget_audit_callback]
                if self.on_trial_complete is not None:
                    callbacks.append(self.on_trial_complete)
                self._study.optimize(
                    lambda t: self._objective(t, best_regime),
                    n_trials=remaining,
                    callbacks=callbacks,
                )

            if not self._study.best_trials:
                log.error(f"Round {round_idx}: no complete trials.")
                if round_idx == 0:
                    return None
                break

            winner      = _select_pareto_winner(self._study)
            round_score = ((1.0 - winner.values[0]) * SC.SCORE_COV_NORM
                           + winner.values[1] / SC.SCORE_TIME_NORM)
            log.info(f"  Round {round_idx} best: "
                     f"cov={winner.values[0]:.3f}  "
                     f"time={winner.values[1]:.3f}s  "
                     f"score={round_score:.3f}  "
                     f"Pareto front size={len(self._study.best_trials)}")

            if (round_idx > 0
                    and (prev_best_score - round_score) < SC.SCORE_IMPROVE_MIN):
                log.info(f"  Round {round_idx}: improvement "
                         f"{prev_best_score - round_score:.4f} < threshold — stopping.")
                break
            prev_best_score = round_score

        # ── Extract best config ───────────────────────────────────────────
        winner      = _select_pareto_winner(self._study)
        expanded    = _expand_winner_params(winner.params, best_regime, self._pairs_candidates)
        best_coarse, best_fine = _split_params(expanded)

        log.info(f"Best config: refStep={best_coarse.get('refStep')}  "
                 f"distQ={best_coarse.get('distQuantification', 0):.2f}  "
                 f"opApproach={best_fine.get('operationApproach')}  "
                 f"outputNum={best_coarse.get('outputNum')}")

        # ── Post-study re-eval with tight angular threshold ───────────────
        scenes_final = self._sample_scenes(SC.M_FULL)
        best_result  = self.evaluate_config(
            best_coarse, best_fine, scenes_final,
            SC.POS_THRESH_TIGHT, SC.ANG_THRESH_TIGHT,
        )
        log.info(f"Re-eval (tight ±{SC.POS_THRESH_TIGHT*1e3:.0f}mm "
                 f"±{SC.ANG_THRESH_TIGHT:.0f}°): "
                 f"cov={best_result.coverage:.3f}  "
                 f"time={best_result.mean_time:.3f}s")

        # ── Phase 4: symmetry (conditional) ──────────────────────────────
        if best_result.coverage >= SC.SYMMETRY_COVERAGE_GATE:
            final_result = self.phase4_symmetry(best_coarse, best_fine, best_result)
        else:
            log.info(f"PHASE 4 skipped: coverage {best_result.coverage:.2f} "
                     f"< gate {SC.SYMMETRY_COVERAGE_GATE}")
            final_result = best_result

        # ── Summary ──────────────────────────────────────────────────────
        elapsed    = time.time() - t0
        n_complete = sum(1 for t in self._study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned   = sum(1 for t in self._study.trials
                         if t.state == optuna.trial.TrialState.PRUNED)
        cache_stats = self.cache.stats() if self.cache else {}

        log.info(f"\n{'='*60}")
        log.info(f"OPTUNA COMPLETE: {self.part_name}")
        log.info(f"  coverage      = {final_result.coverage:.3f}")
        log.info(f"  mean_time     = {final_result.mean_time:.3f} s")
        log.info(f"  score         = {final_result.score:.3f}")
        log.info(f"  score_quality = {final_result.score_quality:.3f}")
        log.info(f"  mv_evals      = {self._n_evals}")
        log.info(f"  trials        = {n_complete} complete, {n_pruned} pruned")
        log.info(f"  Pareto size   = {len(self._study.best_trials)}")
        log.info(f"  wall_time     = {elapsed:.0f} s")
        if cache_stats:
            log.info(f"  cache         = {cache_stats}")

        self._log_result_json(final_result)
        return final_result

    # ─────────────────────────────────────────────────────────────────────
    # Pareto front access (for the GUI TUNING stage)
    # ─────────────────────────────────────────────────────────────────────

    def iter_pareto_configs(self) -> List[Tuple[optuna.trial.FrozenTrial, dict, dict]]:
        """Return the Pareto front as [(trial, coarse, fine), …].

        Each trial's short params are expanded into full coarse/fine config dicts
        (ready for evaluate_config / _run_one_scene) using the Phase-1 regime and
        pairs candidates. Sorted by coverage desc, then time asc — the same order
        _select_pareto_winner prefers. Empty if no study has run yet.
        """
        if self._study is None or not self._best_regime:
            return []
        trials = sorted(self._study.best_trials,
                        key=lambda t: (-t.values[0], t.values[1]))
        out: List[Tuple[optuna.trial.FrozenTrial, dict, dict]] = []
        for t in trials:
            expanded = _expand_winner_params(t.params, self._best_regime,
                                             self._pairs_candidates)
            coarse, fine = _split_params(expanded)
            out.append((t, coarse, fine))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tuner — fully joint MechVision tuning "
                    "(NSGA-II / TPE / GP samplers)")
    p.add_argument("--part",           required=True)
    p.add_argument("--scenes_dir",     default=None)
    p.add_argument("--n_trials", type=int, default=None,
                   help=f"Joint study trial budget (default: {SC.N_TRIALS})")
    p.add_argument("--n_rounds",       type=int, default=None,
                   help=f"Number of optimization rounds (default: {SC.N_ROUNDS})")
    p.add_argument("--m_full",         type=int, default=None)
    p.add_argument("--dry_run",        action="store_true")
    p.add_argument("--no_cache",       action="store_true")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--storage",        default=None,
                   help="SQLite path prefix (e.g. MM_Optimizer/results/); "
                        "creates {prefix}{part}_{sampler}.db")
    p.add_argument("--no_adaptive_thresh", dest="adaptive_thresh", action="store_false",
                   help="Disable adaptive study threshold; use fixed POS_THRESH_TIGHT=2mm.")
    p.set_defaults(adaptive_thresh=True)
    p.add_argument("--pos_thresh_k",   type=float, default=0.01,
                   help="k for adaptive study threshold: clip(k × longest_OBB_m, 2mm, 5mm). "
                        "Only used when adaptive_thresh is enabled (default: 0.01).")
    p.add_argument("--sampler",        choices=list(SAMPLER_CHOICES), default=SAMPLER_DEFAULT,
                   help="Optuna sampler for the joint study (default: gp)")
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
        SC.M_FULL = len(scene_groups)

    model_path = os.path.join(_ROOT, "output", "reference_pcd", args.part,
                              f"{args.part}_surface", f"{args.part}_surface.ply")
    if not os.path.exists(model_path):
        log.error(f"Reference model not found: {model_path} — re-run the sampling pipeline to generate it.")
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
        storage_path = os.path.join(RESULTS_DIR, "")

    opt = Tuner(
        part_name      = args.part,
        client         = client,
        project_id     = project_id,
        scene_groups   = scene_groups,
        warm_start     = ws,
        cache          = cache,
        dry_run        = args.dry_run,
        n_trials = args.n_trials,
        n_rounds       = args.n_rounds,
        seed           = args.seed,
        storage_path   = storage_path,
        pos_thresh_k    = args.pos_thresh_k,
        adaptive_thresh = args.adaptive_thresh,
        sampler         = args.sampler,
    )

    try:
        result = opt.run()
        if result is None:
            log.error("Optimization did not converge.")
            sys.exit(1)
        out = opt.export_best(result, prefix=f"{args.sampler.upper()}_")
        log.info(f"Best config exported -> {out}")
        log.info("Done.")
    finally:
        opt.cleanup()
        if not args.dry_run and client is not None:
            client.close()


if __name__ == "__main__":
    main()
