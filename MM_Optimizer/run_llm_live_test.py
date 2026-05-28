"""
run_llm_live_test.py
--------------------
Runs one live test round of LLMOptimizer against a real MechVision instance.

Usage:
    python MM_Optimizer/run_llm_live_test.py [--n_rounds N] [--llm_model MODEL]
                                              [--arrangement TEXT] [--dry_run]

Defaults:
    part         : 25333MB000
    n_rounds     : 1
    llm_model    : phi4-reasoning:latest
    arrangement  : random pile
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import MM_Optimizer.search_config as SC
from MM_Optimizer.llm_optimizer import LLMOptimizer, llm_params_to_coarse_fine
from MM_Optimizer.mesh_analysis import analyze_mesh, load_reference_pcd
from MM_Optimizer.optimizer import Optimizer, PROJ_NAME
from MM_Optimizer.optimizer_utils import list_synthetic_scenes

log = logging.getLogger(__name__)

PART = "25333MB000"


def _build_validate_fn(opt: Optimizer, ws, n_scenes: int = 1):
    """Return a validate_fn(params) -> (coverage, mean_time) for LLMOptimizer.

    The scene set is sampled once at construction so all rounds compare on the
    same scenes (removes inter-round noise from random scene selection).
    """
    fixed_scenes = opt._sample_scenes(n_scenes)
    log.info(f"Evaluation scene(s) fixed: {[s[0] for s in fixed_scenes]}")

    def validate_fn(params: dict) -> tuple[float, float]:
        coarse, fine = llm_params_to_coarse_fine(params, ws)
        result = opt.evaluate_config(
            coarse, fine, fixed_scenes,
            SC.POS_THRESH_TIGHT,
            SC.ANG_THRESH_TIGHT,
        )
        log.info(
            f"  [eval] cov={result.coverage:.3f}  time={result.mean_time:.3f}s"
            f"  refStep={params['refStep']}  distQ={params['distQuantification']:.2f}"
            f"  referred={params['referredStep']}"
        )
        return result.coverage, result.mean_time

    return validate_fn


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="LLM live test against MechVision")
    parser.add_argument("--part",        default=PART)
    parser.add_argument("--n_rounds",    type=int,   default=1)
    parser.add_argument("--llm_model",   default="phi4-reasoning:latest")
    parser.add_argument("--arrangement", default="random pile, single part instance")
    parser.add_argument("--dry_run",     action="store_true")
    args = parser.parse_args()

    # ── Mesh path ──────────────────────────────────────────────────────────────
    mesh_path = os.path.join(
        _ROOT, "output", "reference_pcd",
        args.part, f"{args.part}_surface", f"{args.part}_surface.ply",
    )
    if not os.path.exists(mesh_path):
        log.error(f"Reference model not found: {mesh_path}")
        sys.exit(1)
    log.info(f"Mesh path : {mesh_path}")

    # ── Scene groups ───────────────────────────────────────────────────────────
    scenes_root = os.path.join(_ROOT, "output", "synthetic_target", args.part)
    if not os.path.isdir(scenes_root):
        log.error(f"Scenes directory not found: {scenes_root}")
        sys.exit(1)
    scene_groups = list_synthetic_scenes(scenes_root)
    if not scene_groups:
        log.error(f"No scene_MMMMM directories found under: {scenes_root}")
        sys.exit(1)
    log.info(f"Scenes found: {len(scene_groups)} M-scene(s)")

    # ── Mesh analysis ──────────────────────────────────────────────────────────
    log.info("Running mesh analysis …")
    pcd = load_reference_pcd(mesh_path)
    ws  = analyze_mesh(pcd)
    log.info(
        f"  sym={ws.symmetry_class}  diam={ws.diameter_mm:.1f} mm"
        f"  prefer_edge={ws.prefer_edge}"
    )

    # ── MechVision connection ──────────────────────────────────────────────────
    if args.dry_run:
        log.info("[dry_run] Skipping MechVision connection")
        client = None
        project_id = -1
    else:
        log.info("Connecting to MechVision …")
        from mm_adapter.mm_adapter import MechVisionClient
        client = MechVisionClient()
        projects = client.get_projects()
        if PROJ_NAME not in projects:
            log.error(f"Project '{PROJ_NAME}' not found. Available: {list(projects.keys())}")
            sys.exit(1)
        project_id = projects[PROJ_NAME]
        log.info(f"Connected — project_id={project_id}")

    # ── Build Optimizer (wraps MechVision calls) ───────────────────────────────
    opt = Optimizer(
        part_name    = args.part,
        client       = client,
        project_id   = project_id,
        scene_groups = scene_groups,
        warm_start   = ws,
        use_two_pass = False,
        dry_run      = args.dry_run,
    )

    # validate_fn: 1 scene per eval (fast live test)
    validate_fn = _build_validate_fn(opt, ws, n_scenes=1)

    # ── LLM optimizer ──────────────────────────────────────────────────────────
    log.info(f"Starting LLMOptimizer — model={args.llm_model}  n_rounds={args.n_rounds}")
    llm_opt = LLMOptimizer(
        experience_bank = None,
        n_rounds        = args.n_rounds,
        llm_model       = args.llm_model,
        dry_run         = args.dry_run,
    )

    result = llm_opt.run(
        mesh_path           = mesh_path,
        arrangement_context = args.arrangement,
        validate_fn         = validate_fn,
        store_result        = False,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LLM Live Test — Result")
    print("=" * 60)
    print(f"symmetry_class : {result['symmetry_class']}")
    print(f"best_coverage  : {result['best_coverage']:.3f}")
    print(f"best_mean_time : {result['best_mean_time']:.3f} s")
    print(f"\nTrajectory ({len(result['trajectory'])} rounds):")
    for r in result["trajectory"]:
        tag = " [violated]" if r["violated"] else ""
        print(
            f"  round {r['round']}: cov={r['coverage']:.3f}"
            f"  time={r['mean_time']:.3f}s{tag}"
        )
    import json
    print(f"\nbest_params:\n{json.dumps(result['best_params'], indent=2)}")


if __name__ == "__main__":
    main()
