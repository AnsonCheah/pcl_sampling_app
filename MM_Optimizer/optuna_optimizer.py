"""
optuna_optimizer.py — Staged Optuna TPE optimizer for MechVision pose estimation
----------------------------------------------------------------------------------
Replaces Phases 2–3–5–6 of the hierarchical coordinate descent with three
sequential Optuna studies that mirror the CD structure:

  Stage 1a : refStep × distQ          (2D, mirrors Phase 2a joint grid)
  Stage 1b : remaining coarse params  (8–10D, mirrors Phase 2b CD)
  Stage 2  : fine params              (6D, mirrors Phase 3 CD)

Phases 0 (mesh analysis warm-start), 1 (regime gate), and 4 (symmetry) are
unchanged and delegate to the wrapped Optimizer instance.

Key design decisions
--------------------
- Stage 1a enqueues the full 5×6=30 refStep×distQ grid (fast→slow scale order)
  before TPE, so the time guard is seeded early by cheap configs.
- Time guard (Stage 1a only): prune if running mean_time > best×OPTUNA_TIME_RATIO.
  refStep is locked after Stage 1a, so Stage 1b and 2 have no time guard —
  this lets outputNum and fine params be explored freely.
- Coarse stages (1a, 1b) use fixed default fine params (_default_fine) so fine
  param variation does not contaminate coarse optimisation.
- MedianPruner with n_startup_trials=2, n_warmup_steps=1, interval_steps=1
  checks after every scene from the 2nd trial onward.
- EvalCache from the existing Optimizer is preserved.
- SQLite storage enables crash-resume (skipped in dry_run).

CLI
---
  python MM_Optimizer/optuna_optimizer.py --part 25333MB000 [--dry_run]
      [--n_trials_1a N] [--n_trials_1b N] [--n_trials_2 N]
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
# Stage-specific suggest_params functions
# All bounds/choices reference SC.* constants — no literals.
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_voxel_params(
    trial: optuna.Trial,
    voxel_bounds: Tuple[float, float, float, float],
) -> dict:
    """Suggest minVoxelLength and maxVoxelLength as min + width (guarantees min < max)."""
    min_lo, min_hi, width_lo, width_hi = voxel_bounds
    min_vox = trial.suggest_float("minVoxelLength_mm", min_lo, min_hi)
    vox_w   = trial.suggest_float("voxel_width_mm",   width_lo, width_hi)
    return {
        "minVoxelLength": min_vox,
        "maxVoxelLength": min_vox + vox_w,
    }


def suggest_params_1a(
    trial: optuna.Trial,
    refstep_bounds: Tuple[int, int] = SC.OPTUNA_REFSTEP_BOUNDS,
) -> dict:
    """Stage 1a: suggest refStep (int) and distQ (float) only.

    refstep_bounds is computed per-part in OptunaOptimizer.__init__ so the
    upper bound scales with warm_start.refStep for large parts.
    """
    lo, hi   = refstep_bounds
    dlo, dhi = SC.OPTUNA_DISTQ_BOUNDS
    return {
        "refStep":            trial.suggest_int(  "refStep", lo, hi),
        "distQuantification": trial.suggest_float("distQ",   dlo, dhi),
    }


def suggest_params_1b(
    trial: optuna.Trial,
    regime: dict,
    pairs_candidates: List[int],
    voxel_bounds: Tuple[float, float, float, float],
) -> dict:
    """Stage 1b: suggest remaining coarse params (refStep/distQ are fixed externally)."""
    rlo, rhi = SC.OPTUNA_REFERRED_BOUNDS
    vlo, vhi = SC.OPTUNA_VOTERATIO_BOUNDS
    olo, ohi = SC.OPTUNA_OUTPUTNUM_BOUNDS

    coarse = {
        "angleQuantification":          trial.suggest_categorical(
                                            "angleQ", SC.OPTUNA_ANGLQ_CHOICES),
        "maxNumOfPointPairsPerFeature": trial.suggest_categorical(
                                            "pairs", pairs_candidates),
        "maxVoteRatio":                 trial.suggest_float(
                                            "maxVoteRatio", vlo, vhi),
        "referredStep":                 trial.suggest_int(
                                            "referredStep", rlo, rhi),
        "useDistanceNMS":               trial.suggest_categorical(
                                            "useDistNMS", [True, False]),
        "outputNum":                    trial.suggest_int(
                                            "outputNum", olo, ohi),
        **_suggest_voxel_params(trial, voxel_bounds),
    }
    if regime["coarse_mode"] == 1.0:
        at_choices = next(c for n, c, _ in SC.PHASE2B_PARAMS if n == "angleThreshold")
        fa = trial.suggest_categorical("filterCandidatePoseByAxis", [True, False])
        coarse["filterCandidatePoseByAxis"] = fa
        coarse["angleThreshold"] = (
            trial.suggest_categorical("angleThreshold", at_choices) if fa else 90
        )
    return coarse


def suggest_fine_params(trial: optuna.Trial, regime: dict) -> dict:
    """Stage 2: suggest fine registration params only."""
    clo, chi = SC.OPTUNA_CONFTHRESH_BOUNDS
    return {
        "registrationMode":                  regime["fine_mode"],
        "operationApproach":                 trial.suggest_categorical(
                                                 "opApproach", SC.OPTUNA_OPAPP_CHOICES),
        "deviationCorrectionCapacity":       trial.suggest_categorical(
                                                 "devCap", SC.OPTUNA_DEVCAP_CHOICES),
        "onlyConsiderVisibleSurfaceOfModel": trial.suggest_categorical(
                                                 "visibleSurf", [True, False]),
        "considerErrorofNormalAngles":       trial.suggest_categorical(
                                                 "normalAng", [True, False]),
        "scoreLevel":                        trial.suggest_categorical(
                                                 "scoreLevel", SC.OPTUNA_SCORELV_CHOICES),
        "confidenceThreshold":               trial.suggest_float("confThresh", clo, chi),
        "candidateTopNum":                   1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_warm_1b(
    warm,
    regime: dict,
    pairs_candidates: List[int],
    voxel_bounds: Tuple[float, float, float, float],
) -> dict:
    """Geometry-derived warm-start for Stage 1b (remaining coarse params)."""
    snapped_pairs = min(pairs_candidates,
                        key=lambda x: abs(x - warm.maxNumOfPointPairsPerFeature))
    min_lo, min_hi, width_lo, width_hi = voxel_bounds
    params = {
        "angleQ":              warm.angleQuantification,
        "pairs":               snapped_pairs,
        "maxVoteRatio":        0.5,
        "referredStep":        1,
        "useDistNMS":          True,
        "outputNum":           1,
        "minVoxelLength_mm":   warm.minVoxelLength_mm,
        "voxel_width_mm":      warm.maxVoxelLength_mm - warm.minVoxelLength_mm,
    }
    if regime["coarse_mode"] == 1.0:
        params["filterCandidatePoseByAxis"] = True
        params["angleThreshold"]            = 135

    # Guard bounds
    vlo, vhi = SC.OPTUNA_VOTERATIO_BOUNDS
    rlo, rhi = SC.OPTUNA_REFERRED_BOUNDS
    olo, ohi = SC.OPTUNA_OUTPUTNUM_BOUNDS
    assert vlo <= params["maxVoteRatio"] <= vhi
    assert rlo <= params["referredStep"] <= rhi
    assert olo <= params["outputNum"]    <= ohi
    assert params["angleQ"] in SC.OPTUNA_ANGLQ_CHOICES, \
        f"angleQ={params['angleQ']} not in {SC.OPTUNA_ANGLQ_CHOICES}"
    assert params["pairs"] in pairs_candidates
    assert min_lo <= params["minVoxelLength_mm"] <= min_hi
    assert width_lo <= params["voxel_width_mm"] <= width_hi
    return params


def _build_warm_fine() -> dict:
    """Default warm-start for Stage 2 (fine params)."""
    return {
        "opApproach":  1.0,   # Standard ICP
        "devCap":      0.0,
        "visibleSurf": False,
        "normalAng":   False,
        "scoreLevel":  0.0,
        "confThresh":  0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruct param dicts from completed trial params
# ─────────────────────────────────────────────────────────────────────────────

def _trial_to_coarse(
    best_1a: dict,
    best_1b_params: dict,
    regime: dict,
) -> dict:
    """Merge Stage 1a best and Stage 1b best_trial.params into a full coarse dict."""
    min_vox = best_1b_params.get("minVoxelLength_mm", 1.0)
    vox_w   = best_1b_params.get("voxel_width_mm",   14.0)
    coarse = {
        "registrationMode":             regime["coarse_mode"],
        "refStep":                      best_1a["refStep"],
        "distQuantification":           best_1a["distQuantification"],
        "angleQuantification":          best_1b_params["angleQ"],
        "maxNumOfPointPairsPerFeature": best_1b_params["pairs"],
        "maxVoteRatio":                 best_1b_params["maxVoteRatio"],
        "referredStep":                 best_1b_params["referredStep"],
        "useDistanceNMS":               best_1b_params["useDistNMS"],
        "outputNum":                    best_1b_params["outputNum"],
        "minVoxelLength":               min_vox,
        "maxVoxelLength":               min_vox + vox_w,
    }
    if regime["coarse_mode"] == 1.0:
        coarse["filterCandidatePoseByAxis"] = best_1b_params.get(
            "filterCandidatePoseByAxis", True)
        coarse["angleThreshold"] = best_1b_params.get("angleThreshold", 90)
    return coarse


def _trial_to_fine(fine_params: dict, regime: dict) -> dict:
    """Convert Stage 2 best_trial.params to a fine registration dict."""
    return {
        "registrationMode":                  regime["fine_mode"],
        "operationApproach":                 fine_params["opApproach"],
        "deviationCorrectionCapacity":       fine_params["devCap"],
        "onlyConsiderVisibleSurfaceOfModel": fine_params["visibleSurf"],
        "considerErrorofNormalAngles":       fine_params["normalAng"],
        "scoreLevel":                        fine_params["scoreLevel"],
        "confidenceThreshold":               fine_params["confThresh"],
        "candidateTopNum":                   1,
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
        n_pruned = sum(1 for t in study.trials
                       if t.state == optuna.trial.TrialState.PRUNED)
        log.info(f"  [budget] trial={trial.number:3d}  "
                 f"best_score={study.best_value:.3f}  pruned={n_pruned}")


# ─────────────────────────────────────────────────────────────────────────────
# OptunaOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class OptunaOptimizer:
    """Staged Optuna TPE optimizer for MechVision pose estimation parameters.

    Replaces Phases 2–3–5–6 of the hierarchical coordinate descent with three
    sequential studies. Phases 0, 1, and 4 are unchanged.

    Parameters
    ----------
    part_name    : Part identifier (matches model dir and scene dir names).
    client       : Connected MechVisionClient, or None for dry_run.
    project_id   : Integer project ID for PROJ_NAME.
    scene_groups : List[List[str]] — one inner list per scene_MMMMM directory.
    warm_start   : WarmStart from mesh_analysis.analyze_mesh().
    cache        : EvalCache instance, or None to disable.
    dry_run      : Build param dicts but do not call MechVision.
    n_trials_1a  : Stage 1a trial budget (default: SC.OPTUNA_N_TRIALS_1A).
    n_trials_1b  : Stage 1b trial budget (default: SC.OPTUNA_N_TRIALS_1B).
    n_trials_2   : Stage 2  trial budget (default: SC.OPTUNA_N_TRIALS_2).
    seed         : Random seed for reproducibility.
    storage_path : Path to SQLite DB prefix for crash-resume, e.g. "results/optuna".
                   Three DBs are created: {prefix}_1a.db, {prefix}_1b.db, {prefix}_2.db.
                   None = in-memory studies (no persistence).
    """

    def __init__(
        self,
        part_name:    str,
        client,
        project_id:   int,
        scene_groups: List[List[str]],
        warm_start,
        cache:        Optional[EvalCache] = None,
        dry_run:      bool = False,
        n_trials_1a:  Optional[int] = None,
        n_trials_1b:  Optional[int] = None,
        n_trials_2:   Optional[int] = None,
        seed:         int = 42,
        storage_path: Optional[str] = None,
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
        self.n_trials_1a   = n_trials_1a if n_trials_1a is not None else SC.OPTUNA_N_TRIALS_1A
        self.n_trials_1b   = n_trials_1b if n_trials_1b is not None else SC.OPTUNA_N_TRIALS_1B
        self.n_trials_2    = n_trials_2  if n_trials_2  is not None else SC.OPTUNA_N_TRIALS_2
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
        # Upper bound scales with warm_start.refStep so large parts aren't
        # capped at 20 — e.g. D=0.5m part has ws.refStep=25, needs hi >= 50.
        refstep_hi = max(SC.OPTUNA_REFSTEP_BOUNDS[1],
                         int(warm_start.refStep * max(SC.PHASE2A_REFSTEP_SCALES)))
        self._refstep_bounds: Tuple[int, int] = (SC.OPTUNA_REFSTEP_BOUNDS[0], refstep_hi)

        # Populated by run() for post-run inspection / testing
        self._study_1a: Optional[optuna.Study] = None
        self._study_1b: Optional[optuna.Study] = None
        self._study_2:  Optional[optuna.Study] = None
        self._best_mean_time: float = float("inf")

    # ─────────────────────────────────────────────────────────────────────
    # Default param helpers
    # ─────────────────────────────────────────────────────────────────────

    def _default_fine(self, regime: dict) -> dict:
        """Fixed fine params for Stage 1a/1b — warm-start defaults."""
        return {
            "registrationMode":                  regime["fine_mode"],
            "operationApproach":                 1.0,
            "deviationCorrectionCapacity":       0.0,
            "onlyConsiderVisibleSurfaceOfModel": False,
            "considerErrorofNormalAngles":       False,
            "scoreLevel":                        0.0,
            "confidenceThreshold":               0.0,
            "candidateTopNum":                   1,
        }

    def _default_coarse_remaining(self, regime: dict) -> dict:
        """Default coarse params (excluding refStep/distQ) for Stage 1a evaluation."""
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
    # Study factory
    # ─────────────────────────────────────────────────────────────────────

    def _create_study(self, name: str, storage_suffix: str = "") -> optuna.Study:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=SC.OPTUNA_N_STARTUP,
            seed=self._seed,
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=1,
            interval_steps=1,
        )
        storage = None
        if self._storage_path and storage_suffix:
            storage = f"sqlite:///{self._storage_path}{storage_suffix}.db"
        return optuna.create_study(
            direction      = "minimize",
            sampler        = sampler,
            pruner         = pruner,
            storage        = storage,
            study_name     = name,
            load_if_exists = True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1a grid enqueue
    # ─────────────────────────────────────────────────────────────────────

    def _enqueue_1a_grid(self, study: optuna.Study) -> None:
        """Enqueue full refStep×distQ grid (fast→slow scale order)."""
        ws = self.opt.ws
        lo, hi = self._refstep_bounds
        for ref_scale in SC.PHASE2A_REFSTEP_SCALES:   # [2.0, 1.5, 1.0, 0.75, 0.5]
            ref = max(lo, min(hi, int(ws.refStep * ref_scale)))
            for dq in SC.PHASE2A_DISTQ_VALUES:
                study.enqueue_trial({"refStep": ref, "distQ": dq})

    # ─────────────────────────────────────────────────────────────────────
    # Streaming objective
    # ─────────────────────────────────────────────────────────────────────

    def _objective_streaming(
        self,
        trial: optuna.Trial,
        regime: dict,
        voxel_bounds: Optional[Tuple[float, float, float, float]],
        fixed_1a: Optional[Dict]  = None,
        fixed_coarse: Optional[Dict] = None,
    ) -> float:
        """Scene-by-scene evaluation with per-scene pruning + stage-conditional time guard.

        Stage 1a (fixed_1a=None,  fixed_coarse=None): suggest {refStep, distQ}.
        Stage 1b (fixed_1a=dict,  fixed_coarse=None): suggest remaining coarse.
        Stage 2  (fixed_coarse=dict):                 suggest fine params only.

        Time guard is Stage 1a only — refStep varies there and can reach 97s/scene.
        Stage 1b has no time guard so outputNum is explored freely.
        """
        is_stage_1a = (fixed_1a is None and fixed_coarse is None)
        is_stage_2  = (fixed_coarse is not None)

        if is_stage_2:
            coarse_p = fixed_coarse
            fine_p   = suggest_fine_params(trial, regime)
        elif is_stage_1a:
            _1a      = suggest_params_1a(trial, self._refstep_bounds)
            coarse_p = {**self._default_coarse_remaining(regime),
                        "refStep":            _1a["refStep"],
                        "distQuantification": _1a["distQuantification"]}
            fine_p   = self._default_fine(regime)
        else:
            # Stage 1b
            _1b      = suggest_params_1b(trial, regime,
                                         self._pairs_candidates, voxel_bounds)
            coarse_p = {**fixed_1a,
                        "registrationMode": regime["coarse_mode"],
                        **_1b}
            fine_p   = self._default_fine(regime)

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
            running_score = (1.0 - running_cov) * 1e6 + running_time

            trial.report(running_score, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Time guard — Stage 1a only (refStep can cause 97s/scene blowup)
            if (is_stage_1a
                    and step >= 1
                    and self._best_mean_time < float("inf")
                    and running_time > self._best_mean_time * SC.OPTUNA_TIME_RATIO):
                raise optuna.TrialPruned()

            # Hard abort: zero coverage after 3+ scenes (all stages)
            if step >= 2 and running_cov == 0.0:
                raise optuna.TrialPruned()

        final_cov   = total_cov  / SC.M_FULL
        final_time  = total_time / SC.M_FULL
        final_score = (1.0 - final_cov) * 1e6 + final_time

        if is_stage_1a and final_cov > 0.0:
            self._best_mean_time = min(self._best_mean_time, final_time)

        return final_score

    # ─────────────────────────────────────────────────────────────────────
    # Main run loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> Optional[EvalResult]:
        """Execute full optimization: Phase 1 → three Optuna stages → Phase 4."""
        t0 = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"OptunaOptimizer: part={self.opt.part_name}  "
                 f"1a={self.n_trials_1a}  1b={self.n_trials_1b}  2={self.n_trials_2}  "
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

        # ── Stage 1a: refStep × distQ (time guard active) ────────────────
        log.info("=" * 60)
        log.info("STAGE 1a — refStep × distQ grid + TPE")
        self._best_mean_time = float("inf")
        self._study_1a = self._create_study(f"{part}_1a", f"_{part}_1a")
        if not self._study_1a.trials:
            self._enqueue_1a_grid(self._study_1a)
            log.info(f"Enqueued {len(SC.PHASE2A_REFSTEP_SCALES) * len(SC.PHASE2A_DISTQ_VALUES)}"
                     f" grid trials (fast→slow scale order)")
        else:
            log.info(f"Resumed Stage 1a: {len(self._study_1a.trials)} prior trials")

        n_done_1a = sum(1 for t in self._study_1a.trials
                        if t.state != optuna.trial.TrialState.WAITING)
        remaining_1a = max(0, self.n_trials_1a - n_done_1a)
        if remaining_1a > 0:
            self._study_1a.optimize(
                lambda t: self._objective_streaming(t, best_regime, None),
                n_trials=remaining_1a,
                callbacks=[_budget_audit_callback],
            )

        complete_1a = [t for t in self._study_1a.trials
                       if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete_1a:
            log.error("Stage 1a produced no complete trials.")
            return None
        best_1a = {
            "refStep":            self._study_1a.best_trial.params["refStep"],
            "distQuantification": self._study_1a.best_trial.params["distQ"],
        }
        log.info(f"Stage 1a best: refStep={best_1a['refStep']}  "
                 f"distQ={best_1a['distQuantification']:.2f}  "
                 f"score={self._study_1a.best_value:.3f}")

        # ── Stage 1b: remaining coarse (no time guard) ───────────────────
        log.info("=" * 60)
        log.info("STAGE 1b — remaining coarse params (outputNum freely explored)")
        self._best_mean_time = float("inf")
        self._study_1b = self._create_study(f"{part}_1b", f"_{part}_1b")
        if not self._study_1b.trials:
            warm_1b = _build_warm_1b(self.opt.ws, best_regime,
                                     self._pairs_candidates, self._voxel_bounds)
            self._study_1b.enqueue_trial(warm_1b)
            log.info("Enqueued warm-start for Stage 1b")
        else:
            log.info(f"Resumed Stage 1b: {len(self._study_1b.trials)} prior trials")

        n_done_1b = sum(1 for t in self._study_1b.trials
                        if t.state != optuna.trial.TrialState.WAITING)
        remaining_1b = max(0, self.n_trials_1b - n_done_1b)
        if remaining_1b > 0:
            self._study_1b.optimize(
                lambda t: self._objective_streaming(
                    t, best_regime, self._voxel_bounds, fixed_1a=best_1a),
                n_trials=remaining_1b,
                callbacks=[_budget_audit_callback],
            )

        complete_1b = [t for t in self._study_1b.trials
                       if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete_1b:
            log.error("Stage 1b produced no complete trials.")
            return None
        best_coarse = _trial_to_coarse(best_1a, self._study_1b.best_trial.params,
                                       best_regime)
        log.info(f"Stage 1b best score={self._study_1b.best_value:.3f}  "
                 f"outputNum={best_coarse.get('outputNum', '?')}  "
                 f"angleQ={best_coarse.get('angleQuantification', '?')}")

        # ── Stage 2: fine params (no time guard) ─────────────────────────
        log.info("=" * 60)
        log.info("STAGE 2 — fine registration params")
        self._best_mean_time = float("inf")
        self._study_2 = self._create_study(f"{part}_2", f"_{part}_2")
        if not self._study_2.trials:
            warm_2 = _build_warm_fine()
            self._study_2.enqueue_trial(warm_2)
            log.info("Enqueued warm-start for Stage 2")
        else:
            log.info(f"Resumed Stage 2: {len(self._study_2.trials)} prior trials")

        n_done_2 = sum(1 for t in self._study_2.trials
                       if t.state != optuna.trial.TrialState.WAITING)
        remaining_2 = max(0, self.n_trials_2 - n_done_2)
        if remaining_2 > 0:
            self._study_2.optimize(
                lambda t: self._objective_streaming(
                    t, best_regime, None, fixed_coarse=best_coarse),
                n_trials=remaining_2,
                callbacks=[_budget_audit_callback],
            )

        complete_2 = [t for t in self._study_2.trials
                      if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete_2:
            log.error("Stage 2 produced no complete trials.")
            return None
        best_fine = _trial_to_fine(self._study_2.best_trial.params, best_regime)
        log.info(f"Stage 2 best score={self._study_2.best_value:.3f}  "
                 f"opApproach={best_fine.get('operationApproach', '?')}")

        # ── Post-study: re-evaluate best with tight angular threshold ─────
        scenes_final = self.opt._sample_scenes(SC.M_FULL)
        best_result  = self.opt.evaluate_config(
            best_coarse, best_fine, scenes_final,
            SC.POS_THRESH_TIGHT, SC.ANG_THRESH_TIGHT,
        )
        log.info(f"Re-eval (tight ±{SC.POS_THRESH_TIGHT*1e3:.0f}mm ±{SC.ANG_THRESH_TIGHT:.0f}°): "
                 f"cov={best_result.coverage:.3f}  time={best_result.mean_time:.3f}s")

        # ── Phase 4: symmetry (conditional) ──────────────────────────────
        if best_result.coverage >= SC.PHASE_GATES["after_phase2"][0]:
            final_result = self.opt.phase4_symmetry(best_coarse, best_fine, best_result)
        else:
            log.info(f"PHASE 4 skipped: coverage {best_result.coverage:.2f} "
                     f"< gate {SC.PHASE_GATES['after_phase2'][0]}")
            final_result = best_result

        # ── Summary ──────────────────────────────────────────────────────
        elapsed = time.time() - t0
        n_complete = (len(complete_1a) + len(complete_1b) + len(complete_2))
        n_pruned   = sum(
            1 for study in (self._study_1a, self._study_1b, self._study_2)
            for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        )
        cache_stats = self.opt.cache.stats() if self.opt.cache else {}

        log.info(f"\n{'='*60}")
        log.info(f"OPTUNA COMPLETE: {self.opt.part_name}")
        log.info(f"  coverage   = {final_result.coverage:.3f}")
        log.info(f"  mean_time  = {final_result.mean_time:.3f} s")
        log.info(f"  score      = {final_result.score:.3f}")
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

    def export_best(self, result: EvalResult, prefix:str="", suffix:str="") -> str:
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
        description="OptunaOptimizer — staged TPE auto-tuner for MechVision")
    p.add_argument("--part",          required=True)
    p.add_argument("--scenes_dir",    default=None)
    p.add_argument("--n_trials_1a",   type=int, default=None,
                   help=f"Stage 1a budget (default: {SC.OPTUNA_N_TRIALS_1A})")
    p.add_argument("--n_trials_1b",   type=int, default=None,
                   help=f"Stage 1b budget (default: {SC.OPTUNA_N_TRIALS_1B})")
    p.add_argument("--n_trials_2",    type=int, default=None,
                   help=f"Stage 2  budget (default: {SC.OPTUNA_N_TRIALS_2})")
    p.add_argument("--m_full",        type=int, default=None)
    p.add_argument("--dry_run",       action="store_true")
    p.add_argument("--no_cache",      action="store_true")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--export_best",   default=True, action="store_true")
    p.add_argument("--storage",       default=None,
                   help="SQLite path prefix (e.g. MM_Optimizer/results/optuna); "
                        "creates {prefix}_{part}_1a.db etc.")
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
        part_name    = args.part,
        client       = client,
        project_id   = project_id,
        scene_groups = scene_groups,
        warm_start   = ws,
        cache        = cache,
        dry_run      = args.dry_run,
        n_trials_1a  = args.n_trials_1a,
        n_trials_1b  = args.n_trials_1b,
        n_trials_2   = args.n_trials_2,
        seed         = args.seed,
        storage_path = storage_path,
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
