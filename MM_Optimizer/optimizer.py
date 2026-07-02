"""
optimizer.py — [DEPRECATED] Heuristic-Seeded Hierarchical Coordinate Descent
----------------------------------------------------------------------------
DEPRECATED: superseded by optuna_optimizer.py (fully-joint NSGA-II / TPE / GP
study). Kept runnable for reference and as a fallback.

The sampler-agnostic MechVision evaluation harness now lives in
mv_evaluator.py. This module only adds the legacy coordinate-descent phases
(2 / 3 / 5 / 6) and the `run()` orchestration on top of `MVEvaluator`, and
re-exports the harness symbols so existing
`from MM_Optimizer.optimizer import ...` imports keep working.

  Strategy 1 (Transposition Table)   : EvalCache deduplicates evaluations
  Strategy 2 (Two-Pass Multi-Fidelity): cheap M_small pre-screen, full M_full
                                        for top-K survivors only
  Strategy 3 (Phase-Level Gates)      : abort / skip phases when coverage too low

CLI:
  python optimizer.py --part 25333MB000 [--dry_run] [--scenes_dir PATH]
                       [--m_full N] [--no_cache] [--no_two_pass]
                       [--export_best]
"""

import argparse
import copy
import logging
import os
import random
import sys
import time
import warnings
from typing import List, Optional

import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, ".."))
for _p in [_ROOT, _DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mm_adapter.mm_adapter    import MechVisionClient
from MM_Optimizer.eval_cache      import EvalCache
from MM_Optimizer.mesh_analysis   import analyze_mesh, load_reference_pcd
from MM_Optimizer.optimizer_utils import list_synthetic_scenes
import MM_Optimizer.search_config as SC

# Re-export the shared harness so existing imports of these names from
# MM_Optimizer.optimizer keep resolving after the extraction to mv_evaluator.
from MM_Optimizer.mv_evaluator import (   # noqa: F401  (re-exported for back-compat)
    MVEvaluator, EvalResult, match_poses_to_gt, _rotation_error_deg,
    _confirm_symmetry, ENABLE_CACHE, PROJ_NAME, MM_MODEL_ROOT, RESULTS_DIR,
    OPTIMIZER_UTILS_PATH,
)

log = logging.getLogger(__name__)

ENABLE_TWO_PASS = True   # Strategy 2 — set False to evaluate all candidates at M_FULL


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer (legacy coordinate descent)
# ─────────────────────────────────────────────────────────────────────────────

class Optimizer(MVEvaluator):
    """[DEPRECATED] Hierarchical coordinate-descent optimizer.

    Inherits the shared evaluation harness from MVEvaluator and adds the legacy
    Phase 2/3/5/6 coordinate-descent sweeps plus run() orchestration. New work
    should use optuna_optimizer.OptunaOptimizer (NSGA-II / TPE / GP).
    """

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

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 5 — Joint refinement  /  Phase 6 — Interval narrowing  /  run()
    # ─────────────────────────────────────────────────────────────────────────

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
    warnings.warn(
        "optimizer.py (coordinate descent) is deprecated; use "
        "optuna_optimizer.py (NSGA-II / TPE / GP joint study).",
        DeprecationWarning, stacklevel=2)
    log.warning("optimizer.py is DEPRECATED — prefer optuna_optimizer.py")
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
    model_path = os.path.join(_ROOT, "output", "reference_pcd", args.part,
                              f"{args.part}_surface", f"{args.part}_surface.ply")
    if not os.path.exists(model_path):
        log.error(f"Reference model not found: {model_path} — re-run the sampling pipeline to generate it.")
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
