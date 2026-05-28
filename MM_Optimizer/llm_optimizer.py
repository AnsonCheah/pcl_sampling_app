"""
llm_optimizer.py  —  LLM-guided parameter optimizer (Layer 1+2+3 pipeline)
---------------------------------------------------------------------------
Closed-loop agent: suggest → validate → refine, N rounds per part.

Integration with existing NSGA-II:
  LLM proposes params → they are enqueued into Optuna study via
  study.enqueue_trial(build_warm_joint(...)) before NSGA-II runs.
  This seeds the initial population with informed guesses instead of random.

Local LLM: phi4-reasoning:latest 14B via Ollama (fits in 12 GB VRAM at Q4_K_M).
Fallback:  qwen3:8b if 14B is too slow.

Usage:
  # Dry-run (no Ollama required, no MechVision required):
  python MM_Optimizer/llm_optimizer.py --dry_run --model_path path/to/model.ply

  # Integrated:
  from llm_optimizer import LLMOptimizer
  opt = LLMOptimizer(experience_bank=bank, n_rounds=5)
  best = opt.run(mesh_path, arrangement_context="random pile, 20 parts",
                 validate_fn=my_evaluate_fn)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

_log = logging.getLogger(__name__)

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from MM_Optimizer.mesh_analysis import (
    analyze_mesh, load_reference_pcd,
    generate_geometry_summary, get_feature_vector,
    WarmStart,
)
from MM_Optimizer.tcs import RoundResult, summarize as tcs_summarize
from MM_Optimizer.experience_bank import ExperienceBank, _features_from_dict
import MM_Optimizer.search_config as SC


# ---------------------------------------------------------------------------
# Pydantic schema for LLM structured output
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field, model_validator
except ImportError:
    raise ImportError("pip install 'pydantic>=2.0'")


class ParamProposal(BaseModel):
    refStep:                    int   = Field(ge=SC.REFSTEP_BOUNDS[0],
                                             le=SC.REFSTEP_BOUNDS[1])
    distQuantification:         float = Field(ge=SC.OPTUNA_DISTQ_BOUNDS[0],
                                             le=SC.OPTUNA_DISTQ_BOUNDS[1])
    angleQuantification:        int   = Field(ge=min(SC.OPTUNA_ANGLQ_CHOICES),
                                             le=max(SC.OPTUNA_ANGLQ_CHOICES),
                                             default=SC.OPTUNA_ANGLQ_CHOICES[-1])
    referredStep:               int   = Field(ge=SC.REFSTEP_BOUNDS[0],
                                             le=SC.REFSTEP_BOUNDS[1])
    maxVoteRatio:               float = Field(ge=SC.OPTUNA_VOTERATIO_BOUNDS[0],
                                             le=SC.OPTUNA_VOTERATIO_BOUNDS[1])
    outputNum:                  int   = Field(ge=SC.OPTUNA_OUTPUTNUM_BOUNDS[0],
                                             le=SC.OPTUNA_OUTPUTNUM_BOUNDS[1])
    operationApproach:          int   = Field(ge=SC.OPTUNA_OPAPP_BOUNDS[0],
                                             le=SC.OPTUNA_OPAPP_BOUNDS[1])
    deviationCorrectionCapacity: int  = Field(ge=SC.OPTUNA_DEVCAP_BOUNDS[0],
                                             le=SC.OPTUNA_DEVCAP_BOUNDS[1])
    scoreLevel:                 float = Field(ge=min(SC.OPTUNA_SCORELV_CHOICES),
                                             le=max(SC.OPTUNA_SCORELV_CHOICES),
                                             default=0.0)
    confidenceThreshold:        float = Field(ge=SC.OPTUNA_CONFTHRESH_BOUNDS[0],
                                             le=SC.OPTUNA_CONFTHRESH_BOUNDS[1],
                                             default=0.0)
    reasoning: str = Field(description="One sentence explaining the rationale.")

    @model_validator(mode="after")
    def referred_le_ref(self) -> "ParamProposal":
        if self.referredStep > self.refStep:
            self.referredStep = self.refStep
        return self

    def to_flat_dict(self) -> dict:
        """Return params without the reasoning field."""
        d = self.model_dump()
        d.pop("reasoning", None)
        return d


# ---------------------------------------------------------------------------
# LLM client wrapper
# ---------------------------------------------------------------------------

_RAG_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "parameter_reference.md")
try:
    with open(_RAG_PATH, encoding="utf-8") as _f:
        _RAG_CONTENT = _f.read()
except FileNotFoundError:
    _RAG_CONTENT = ""

_SYSTEM_PROMPT = (
    "You are an expert in MechMind CAD matching pose estimation parameter tuning.\n"
    "Your role: propose parameter configurations that maximise detection coverage "
    "while minimising cycle time.  These are INDEPENDENT objectives — do NOT trade "
    "them off yourself; propose a config and let the optimizer place it on the "
    "Pareto front.\n\n"
    "Hard constraints you must never violate:\n"
    "  - referredStep <= refStep  (otherwise the run is rejected and wasted)\n"
    "  - All values within the stated bounds\n\n"
    "Units: all distances in mm.  Times in seconds.\n\n"
    "Output: JSON matching the ParamProposal schema.  One sentence of reasoning only."
    + (("\n\n---\n\n" + _RAG_CONTENT) if _RAG_CONTENT else "")
)

# Similarity score below which we fall back to geometry warm-start only
_RETRIEVAL_FALLBACK_THRESHOLD = SC.LLM_RETRIEVAL_FALLBACK_THRESHOLD


class LLMClient:
    """Thin wrapper around Ollama + Instructor for structured output."""

    def __init__(self, model: str = SC.LLM_DEFAULT_MODEL, max_retries: int = SC.LLM_MAX_RETRIES) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import instructor
            from openai import OpenAI
            raw = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self._client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise Ollama+Instructor: {exc}\n"
                "Ensure Ollama is running and 'pip install instructor openai' is done."
            ) from exc
        return self._client

    def propose(self, user_prompt: str) -> ParamProposal:
        client = self._get_client()
        proposal, _ = client.chat.completions.create_with_completion(
            model=self.model,
            response_model=ParamProposal,
            max_retries=self.max_retries,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        _log.info(
            "[LLM proposal] reasoning: %s | params: %s",
            proposal.reasoning,
            json.dumps(proposal.to_flat_dict()),
        )
        return proposal


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    ws: WarmStart,
    arrangement_context: str,
    similar_parts: list[dict],
    tcs_block: Optional[str],
    scale_anchors: str,
) -> str:
    sections = []

    # 1. Scale anchors
    sections.append(f"## Scale anchors\n{scale_anchors}")

    # 2. Geometry summary
    sections.append(f"## Part geometry\n{generate_geometry_summary(ws)}")

    # 3. Numeric features
    fvec_dict = {
        "diameter_mm": ws.diameter_mm,
        "aspect_ratio": ws.aspect_ratio,
        "flatness_ratio": ws.flatness_ratio,
        "normal_concentration": ws.normal_concentration,
        "prefer_edge": ws.prefer_edge,
        "convexity": ws.convexity,
        "curvature_mean": ws.curvature_mean,
        "curvature_std": ws.curvature_std,
        "n_flat_clusters": ws.n_flat_clusters,
        "has_holes": ws.has_holes,
        "bbox_mm": f"{ws.bbox_x_mm:.0f} x {ws.bbox_y_mm:.0f} x {ws.bbox_z_mm:.0f}",
        "symmetry_class": ws.symmetry_class,
    }
    sections.append("## Numeric features\n" + "\n".join(f"  {k}: {v}" for k, v in fvec_dict.items()))

    # 4. Task context
    sections.append(f"## Task context\n{arrangement_context}")

    # 5. Similar past parts (ascending similarity — most similar LAST)
    if similar_parts:
        ex_lines = ["## Similar past parts (most similar last — pay most attention to it)"]
        for i, part in enumerate(similar_parts):
            sim = part.get("_distance", "?")
            trace = part.get("reasoning_trace", "(no trace)")
            bp = part.get("best_params", {})
            ex_lines.append(
                f"\n### Example {i+1}  (similarity score: {sim})\n"
                f"Part: {part.get('part_name', '?')}  sym={part.get('symmetry_class', '?')}\n"
                f"Best params: {json.dumps(bp, indent=2)}\n"
                f"Result: coverage={part.get('best_coverage', 0):.2f}  "
                f"time={part.get('best_mean_time', 0):.2f}s\n"
                f"Why: {trace}"
            )
        sections.append("\n".join(ex_lines))

    # 6. TCS block (refinement rounds only)
    if tcs_block:
        sections.append(f"## Trajectory so far\n{tcs_block}")

    # 7. The ask
    ask = (
        "## Your task\n"
        "Propose the next parameter configuration as JSON matching ParamProposal.\n"
        "Include one sentence explaining your reasoning.\n"
        "Remember: referredStep MUST be <= refStep."
    )
    sections.append(ask)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

@dataclass
class LLMOptimizer:
    experience_bank: Optional[ExperienceBank] = None
    n_rounds: int = SC.LLM_N_ROUNDS
    k_similar: int = SC.LLM_K_SIMILAR
    llm_model: str = SC.LLM_DEFAULT_MODEL
    dry_run: bool = False
    scale_anchors: str = "Tray: 760x590 mm. Smallest expected part: 30x30x10 mm."
    _client: Optional[LLMClient] = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if not self.dry_run:
            self._client = LLMClient(model=self.llm_model)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(
        self,
        mesh_path: str,
        arrangement_context: str = "random pile",
        validate_fn: Optional[Callable[[dict], tuple[float, float]]] = None,
        store_result: bool = True,
    ) -> dict:
        """Run the closed-loop LLM optimisation for one part.

        Parameters
        ----------
        mesh_path         : Path to the reference PLY model.
        arrangement_context : Free-text description of the bin/tray scene.
        validate_fn       : Callable(params_dict) -> (coverage, mean_time).
                            If None (dry_run mode), returns random mock values.
        store_result      : If True, saves the best result to the experience bank.

        Returns
        -------
        dict with keys: best_params, best_coverage, best_mean_time, trajectory
        """
        # ── Geometry analysis ──────────────────────────────────────────
        pcd = load_reference_pcd(mesh_path)
        ws  = analyze_mesh(pcd)
        fvec = get_feature_vector(ws)
        mesh_features = _ws_to_dict(ws)

        # ── ICL retrieval from experience bank ────────────────────────
        similar_parts: list[dict] = []
        retrieval_sim = 0.0
        if self.experience_bank is not None:
            similar_parts = self.experience_bank.query_similar(
                fvec, k=self.k_similar,
                symmetry_class=ws.symmetry_class,
            )
            if similar_parts:
                retrieval_sim = float(similar_parts[-1].get("_distance", 0.0))

        # Fallback: if no similar parts or too different, rely on warm-start only
        use_llm = (not self.dry_run) and (retrieval_sim > _RETRIEVAL_FALLBACK_THRESHOLD or len(similar_parts) == 0)
        fell_back = not use_llm

        # ── Geometry warm-start params (always available) ──────────────
        warmstart_params = _ws_to_params(ws)

        # ── Validate warm-start baseline ──────────────────────────────
        ws_cov, ws_time = self._validate(warmstart_params, validate_fn)

        # ── Refinement rounds ─────────────────────────────────────────
        history: list[RoundResult] = []
        history.append(RoundResult(0, warmstart_params, ws_cov, ws_time))

        best_params   = warmstart_params
        best_coverage = ws_cov
        best_time     = ws_time
        constraint_retries = 0

        for round_idx in range(1, self.n_rounds + 1):
            tcs_block = tcs_summarize(history) if len(history) >= 1 else None

            if use_llm or self.dry_run:
                prompt = _build_prompt(
                    ws, arrangement_context, similar_parts, tcs_block, self.scale_anchors
                )
                proposal = self._propose(prompt, ws)
                params = proposal.to_flat_dict()
                reasoning = proposal.reasoning
            else:
                params = warmstart_params
                reasoning = "Fell back to geometry warm-start (low retrieval similarity)"
                fell_back = True

            # Validate constraint before evaluation
            if params.get("referredStep", 1) > params.get("refStep", 20):
                constraint_retries += 1
                history.append(RoundResult(round_idx, params, 0.0, 0.0, violated=True))
                continue

            cov, t = self._validate(params, validate_fn)
            history.append(RoundResult(round_idx, params, cov, t))

            if cov > best_coverage or (cov >= best_coverage - 0.01 and t < best_time):
                best_params   = params
                best_coverage = cov
                best_time     = t

            # Early stop if target coverage reached
            if best_coverage >= SC.TARGET_COVERAGE:
                break

        # ── Store result ───────────────────────────────────────────────
        trajectory = [
            {"round": r.round_idx, "params": r.params,
             "coverage": r.coverage, "mean_time": r.mean_time,
             "violated": r.violated}
            for r in history
        ]

        if store_result and self.experience_bank is not None:
            part_name = os.path.splitext(os.path.basename(mesh_path))[0]
            self.experience_bank.insert(
                part_name=part_name,
                model_id=part_name,
                mesh_path=mesh_path,
                mesh_features=mesh_features,
                symmetry_class=ws.symmetry_class,
                arrangement_context=arrangement_context,
                best_coverage=best_coverage,
                best_mean_time=best_time,
                best_params=best_params,
                n_rounds_to_converge=len([r for r in history if not r.violated]),
                reasoning_trace=reasoning if use_llm else "warm-start fallback",
                trajectory=trajectory,
                warmstart_params=warmstart_params,
                warmstart_coverage=ws_cov,
                warmstart_mean_time=ws_time,
                fell_back_to_warmstart=fell_back,
                retrieval_similarity=retrieval_sim,
                constraint_retries=constraint_retries,
                feature_vector=fvec,
            )

        return {
            "best_params":   best_params,
            "best_coverage": best_coverage,
            "best_mean_time": best_time,
            "trajectory":    trajectory,
            "symmetry_class": ws.symmetry_class,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _validate(
        self,
        params: dict,
        validate_fn: Optional[Callable],
    ) -> tuple[float, float]:
        if self.dry_run or validate_fn is None:
            # Mock: return plausible random values for dry-run
            rng = np.random.default_rng(sum(hash(str(v)) for v in params.values()) % (2**31))
            cov  = float(np.clip(rng.normal(0.75, 0.15), 0.0, 1.0))
            time = float(np.clip(rng.normal(0.6, 0.2), 0.1, 3.0))
            return cov, time
        return validate_fn(params)

    def _propose(self, prompt: str, ws: WarmStart) -> ParamProposal:
        if self.dry_run:
            # Return geometry warm-start as a mock proposal
            p = _ws_to_params(ws)
            return ParamProposal(
                refStep=p.get("refStep", 10),
                distQuantification=p.get("distQuantification", 1.0),
                angleQuantification=p.get("angleQuantification", 60),
                referredStep=p.get("referredStep", 5),
                maxVoteRatio=p.get("maxVoteRatio", 0.7),
                outputNum=p.get("outputNum", 1),
                operationApproach=p.get("operationApproach", 1),
                deviationCorrectionCapacity=p.get("deviationCorrectionCapacity", 1),
                reasoning="[dry_run] geometry warm-start proposal",
            )
        return self._client.propose(prompt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def llm_params_to_coarse_fine(params: dict, ws: WarmStart) -> tuple[dict, dict]:
    """Convert a flat LLM ParamProposal dict → (coarse_dict, fine_dict).

    The coarse dict uses surface mode (registrationMode=0.0) as the default;
    OptunaOptimizer.run() overrides this with the best regime found in Phase 1
    when injecting via extra_warm_coarse_fine.
    Missing params are filled from the WarmStart geometry defaults.
    """
    coarse = {
        "registrationMode":              0.0,
        "refStep":                       params["refStep"],
        "distQuantification":            params["distQuantification"],
        "angleQuantification":           params["angleQuantification"],
        "maxNumOfPointPairsPerFeature":  ws.maxNumOfPointPairsPerFeature,
        "maxVoteRatio":                  params["maxVoteRatio"],
        "referredStep":                  params["referredStep"],
        "useDistanceNMS":                True,
        "outputNum":                     params["outputNum"],
        "minVoxelLength":                ws.minVoxelLength_mm,
        "maxVoxelLength":                ws.maxVoxelLength_mm,
    }
    fine = {
        "registrationMode":                   0.0,
        "operationApproach":                  float(params["operationApproach"]),
        "deviationCorrectionCapacity":        float(params["deviationCorrectionCapacity"]),
        "onlyConsiderVisibleSurfaceOfModel":  False,
        "considerErrorofNormalAngles":        False,
        "scoreLevel":                         float(params.get("scoreLevel", 0.0)),
        "confidenceThreshold":                float(params.get("confidenceThreshold", 0.0)),
        "candidateTopNum":                    1,
    }
    return coarse, fine


def _ws_to_params(ws: WarmStart) -> dict:
    """Convert WarmStart geometry hints to a flat param dict."""
    return {
        "refStep":                    10,
        "distQuantification":         float(ws.distQuantification),
        "angleQuantification":        int(ws.angleQuantification),
        "referredStep":               5,
        "maxVoteRatio":               0.7,
        "outputNum":                  int(ws.outputNum),
        "operationApproach":          1,
        "deviationCorrectionCapacity": 1,
        "scoreLevel":                 0.0,
        "confidenceThreshold":        0.0,
    }


def _ws_to_dict(ws: WarmStart) -> dict:
    """Serialise WarmStart to a plain dict (for experience bank storage)."""
    return {
        "diameter_mm":          ws.diameter_mm,
        "aspect_ratio":         ws.aspect_ratio,
        "flatness_ratio":       ws.flatness_ratio,
        "normal_concentration": ws.normal_concentration,
        "prefer_edge":          ws.prefer_edge,
        "convexity":            ws.convexity,
        "curvature_mean":       ws.curvature_mean,
        "curvature_std":        ws.curvature_std,
        "n_flat_clusters":      ws.n_flat_clusters,
        "has_holes":            ws.has_holes,
        "bbox_x_mm":            ws.bbox_x_mm,
        "bbox_y_mm":            ws.bbox_y_mm,
        "bbox_z_mm":            ws.bbox_z_mm,
        "surface_area_mm2":     ws.surface_area_mm2,
        "volume_mm3":           ws.volume_mm3,
        "symmetry_class":       ws.symmetry_class,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import logging
    import random

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description=(
            "LLM-guided parameter optimizer.\n"
            "  python MM_Optimizer/llm_optimizer.py --part 25333MB000\n"
            "  python MM_Optimizer/llm_optimizer.py --part 25333MB000 --run_nsga\n"
            "  python MM_Optimizer/llm_optimizer.py --part 25333MB000 --dry_run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--part",        required=True,
                        help="Part name, e.g. 25333MB000")
    parser.add_argument("--scenes_dir",  default=None,
                        help="Override scene directory (default: output/synthetic_target/<part>)")
    parser.add_argument("--n_rounds",    type=int, default=SC.LLM_N_ROUNDS,
                        help=f"LLM refinement rounds (default: {SC.LLM_N_ROUNDS})")
    parser.add_argument("--llm_model",   default=SC.LLM_DEFAULT_MODEL)
    parser.add_argument("--arrangement", default="random pile, single part instance")
    parser.add_argument("--run_nsga",    action="store_true",
                        help="After LLM rounds, continue with NSGA-II seeded by LLM best params")
    parser.add_argument("--nsga_trials", type=int, default=None,
                        help="NSGA-II trial budget (default: SC.OPTUNA_N_TRIALS_JOINT)")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Skip Ollama and MechVision; use mock validate_fn")
    parser.add_argument("--store",       action="store_true",
                        help="Write result to experience bank after LLM rounds")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Paths (same conventions as optimizer.py / optuna_optimizer.py)
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mesh_path = os.path.join(
        _root, "output", "reference_pcd",
        args.part, f"{args.part}_surface", f"{args.part}_surface.ply",
    )
    scenes_root = args.scenes_dir or os.path.join(
        _root, "output", "synthetic_target", args.part
    )

    if not os.path.exists(mesh_path):
        _log.error(f"Reference model not found: {mesh_path}")
        raise SystemExit(1)

    from MM_Optimizer.optimizer_utils import list_synthetic_scenes
    scene_groups = list_synthetic_scenes(scenes_root) if os.path.isdir(scenes_root) else []
    if not scene_groups:
        _log.error(f"No scenes found under: {scenes_root}")
        raise SystemExit(1)
    _log.info(f"Part={args.part}  scenes={len(scene_groups)}  dry_run={args.dry_run}")

    # MechVision connection
    if args.dry_run:
        _client, _project_id = None, -1
    else:
        from mm_adapter.mm_adapter import MechVisionClient
        from MM_Optimizer.optimizer import PROJ_NAME
        _client = MechVisionClient()
        _projects = _client.get_projects()
        if PROJ_NAME not in _projects:
            _log.error(f"Project '{PROJ_NAME}' not found. Loaded: {list(_projects.keys())}")
            raise SystemExit(1)
        _project_id = _projects[PROJ_NAME]
        _log.info(f"Connected — project_id={_project_id}")

    # Mesh analysis (needed for validate_fn defaults and NSGA-II warm-start)
    _pcd = load_reference_pcd(mesh_path)
    _ws  = analyze_mesh(_pcd)
    _log.info(f"  sym={_ws.symmetry_class}  diam={_ws.diameter_mm:.1f} mm  "
              f"prefer_edge={_ws.prefer_edge}")

    # Build Optimizer (wraps MechVision)
    from MM_Optimizer.optimizer import Optimizer
    _opt = Optimizer(
        part_name    = args.part,
        client       = _client,
        project_id   = _project_id,
        scene_groups = scene_groups,
        warm_start   = _ws,
        use_two_pass = False,
        dry_run      = args.dry_run,
    )

    # Fixed 1-scene validate_fn for LLM rounds (fast per-round feedback)
    _fixed_scenes = _opt._sample_scenes(1)
    _log.info(f"LLM eval scene: {_fixed_scenes[0][0]}")

    def _validate_fn(params: dict) -> tuple[float, float]:
        coarse, fine = llm_params_to_coarse_fine(params, _ws)
        result = _opt.evaluate_config(
            coarse, fine, _fixed_scenes,
            SC.POS_THRESH_TIGHT, SC.ANG_THRESH_TIGHT,
        )
        _log.info(f"  [eval] cov={result.coverage:.3f}  time={result.mean_time:.3f}s"
                  f"  refStep={params['refStep']}  distQ={params['distQuantification']:.2f}"
                  f"  referred={params['referredStep']}")
        return result.coverage, result.mean_time

    # Run LLM optimizer
    _llm_opt = LLMOptimizer(
        experience_bank = None,
        n_rounds        = args.n_rounds,
        llm_model       = args.llm_model,
        dry_run         = args.dry_run,
    )
    _result = _llm_opt.run(
        mesh_path           = mesh_path,
        arrangement_context = args.arrangement,
        validate_fn         = _validate_fn if not args.dry_run else None,
        store_result        = args.store,
    )

    print(f"\n{'='*60}")
    print(f"LLM optimizer — {args.part}")
    print(f"{'='*60}")
    print(f"symmetry_class : {_result['symmetry_class']}")
    print(f"best_coverage  : {_result['best_coverage']:.3f}")
    print(f"best_mean_time : {_result['best_mean_time']:.3f} s")
    print(f"\nTrajectory ({len(_result['trajectory'])} rounds):")
    for _r in _result["trajectory"]:
        _tag = " [violated]" if _r["violated"] else ""
        print(f"  round {_r['round']}: cov={_r['coverage']:.3f}  time={_r['mean_time']:.3f}s{_tag}")
    print(f"\nbest_params:\n{json.dumps(_result['best_params'], indent=2)}")

    # Optionally continue with NSGA-II seeded by LLM best params
    if args.run_nsga:
        from MM_Optimizer.optuna_optimizer import OptunaOptimizer
        import MM_Optimizer.search_config as _SC

        _SC.M_FULL  = len(scene_groups)
        _SC.M_SMALL = max(1, len(scene_groups) // 2)

        _llm_coarse, _llm_fine = llm_params_to_coarse_fine(_result["best_params"], _ws)

        _log.info("\nStarting NSGA-II with LLM warm-start …")
        _nsga = OptunaOptimizer(
            part_name            = args.part,
            client               = _client,
            project_id           = _project_id,
            scene_groups         = scene_groups,
            warm_start           = _ws,
            dry_run              = args.dry_run,
            n_trials_joint       = args.nsga_trials,
            seed                 = args.seed,
            extra_warm_coarse_fine = [(_llm_coarse, _llm_fine)],
        )
        _nsga_result = _nsga.run()
        if _nsga_result:
            print(f"\n{'='*60}")
            print("NSGA-II result")
            print(f"{'='*60}")
            print(f"coverage  : {_nsga_result.coverage:.3f}")
            print(f"mean_time : {_nsga_result.mean_time:.3f} s")
