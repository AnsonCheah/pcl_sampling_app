"""Run the reference-cloud / vote-weight ablation across parts and scenes.

Run::

    python -m registration.ppf_saliency.bench.ablation --parts obj_000018 --arms A_uniform,F_ppf_weight
    python -m registration.ppf_saliency.bench.ablation --all --out bench_ablation.json
    python -m registration.ppf_saliency.bench.ablation --all --arms A_uniform,E_heat_weight,G_combined

Results are stratified by **symmetry class** (continuous / discrete / asymmetric, taken from
BOP's published ``models_info.json``) rather than pooled. A mean over mixed geometry would
hide the thing the sweep exists to find: the mission is that a method works for part #5000
as well as for part #1, so an arm that wins on cylinders and loses on brackets is a failure
of exactly that claim, and pooling would report it as a modest win.

Every recall figure is printed with a 95% confidence interval. That is not decoration -- at
26 instances the half-width is +/- 19 points, wider than any arm difference measured so far.
Reading a 4-point gap as a result at that sample size is the single easiest mistake to make
here, so the interval is carried all the way to the summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

# The repo root, four levels up, so `geometry` and friends resolve when run with -m
# from anywhere. registration/ppf_saliency/bench/ablation.py -> ../../..
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))

from registration.ppf_saliency.bench import SYNTH_ROOT, MESH_ROOT, REPO_ROOT
from registration.ppf_saliency.bench.arms import (ARMS, ArmContext, build_arm,
                                                  resolve_weights)
from registration.ppf.bench.dataset import list_scenes, load_reference, load_scene
from geometry.geom_utils import reference_frames_agree
from registration.ppf_saliency.bench.metrics import (LOOSE, TIGHT, evaluate_pose, summarise,
                           symmetry_transforms_from_bop,
                           symmetry_transforms_from_profile)
from registration.ppf_saliency import PPFConfig, PPFModel, downsample, match

_ROOT = REPO_ROOT


def _bop_models_info() -> Dict[str, dict]:
    """Map ``obj_000005`` -> its BOP ``models_info`` entry, if the dataset is present."""
    out: Dict[str, dict] = {}
    for root, _, files in os.walk(MESH_ROOT):
        if "models_info.json" not in files:
            continue
        with open(os.path.join(root, "models_info.json")) as f:
            for key, rec in json.load(f).items():
                out[f"obj_{int(key):06d}"] = rec
    return out


def symmetry_class(rec: Optional[dict]) -> str:
    if rec is None:
        return "unknown"
    if rec.get("symmetries_continuous"):
        return "continuous"
    if rec.get("symmetries_discrete"):
        return "discrete"
    return "asymmetric"


def _bundle_mesh(part: str) -> str:
    """The part's own exported STL -- the only mesh guaranteed to share the cloud's frame."""
    return os.path.join(REPO_ROOT, "output", "reference_pcd", part, f"{part}.stl")


def _needs_profile(arm_names: List[str], rec: Optional[dict], mode: str) -> bool:
    """Does anything in this run actually consume the ambiguity profile?

    The analysis costs 1-3 min per part, so it is not paid unless something reads it: a
    heat-weighted arm, or a part with no BOP annotation, where the profile's global axes are
    the only symmetry group available to score against. ``always`` forces it so the
    ``ambiguity_tagged`` and axis/phase diagnostics are populated on uniform-only runs too.
    """
    if mode == "never":
        return False
    if mode == "always":
        return True
    return any(ARMS[a].needs_heat for a in arm_names) or rec is None


def _compute_profile(part: str, ref_pts: np.ndarray, ref_nrm: np.ndarray):
    """Analyse this part's ambiguity, in the frame its scenes are actually in.

    Returns ``(profile, None)`` or ``(None, reason)``.

    Two things make this correct, and both are easy to get wrong:

    * **The mesh must be the bundle's own STL.** ``SaveStage`` writes
      ``output/reference_pcd/<part>/<part>.stl`` in the same run and the same model frame as
      ``<part>_surface.ply``, which is the cloud each scene's ``reference_cloud.ply`` is
      written from. A BOP mesh under ``mesh_raw/`` is in the pre-recentre frame *and* in
      millimetres; pairing one with a scene cloud does not raise, it silently produces a
      visibility sweep over a sliver (measured: 13% of points ever visible against 99%, and
      12 reported axes against 3). ``pairing_error`` is what rejects that, so it runs before
      the analysis rather than after something looks wrong.
    * **Analysing on the scene's own reference cloud** makes ``per_point_discriminative``
      index-aligned with ``ref_pts`` by construction. The sidecar this replaced could only
      hope for that alignment and check it after the fact.
    """
    from geometry.ambiguity import AmbiguityConfig, analyse_ambiguity
    from geometry.geom_utils import pairing_error
    import open3d as o3d

    path = _bundle_mesh(part)
    if not os.path.exists(path):
        return None, f"no bundle STL at {os.path.relpath(path, REPO_ROOT)}"

    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        return None, f"bundle STL has no triangles: {os.path.relpath(path, REPO_ROOT)}"
    mesh.compute_vertex_normals()

    why = pairing_error(mesh, ref_pts)
    if why is not None:
        return None, f"bundle STL does not match the scene cloud -- {why}"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ref_pts)
    pcd.normals = o3d.utility.Vector3dVector(ref_nrm)
    print("    analysing pose ambiguity (1-3 min)...", flush=True)
    return analyse_ambiguity(mesh, pcd, AmbiguityConfig()), None


def run_part(part: str, arm_names: List[str], models_info: Dict[str, dict],
             max_scenes: Optional[int] = None,
             model_points: int = 500,
             max_instances: Optional[int] = None,
             ambiguity: str = "auto") -> Dict[str, list]:
    """Every arm against the instances of one part.

    ``max_instances`` caps the per-part budget, and matters more than it looks. Instance
    count per scene scales with how many copies fit in the bin, so a 63 mm part yields ~160
    per scene while a 190 mm part yields ~20 -- an 8x imbalance. Left uncapped, the pooled
    table becomes a weighted average dominated by whichever parts happen to be small, which
    is precisely the "works on part #1 but not part #5000" failure the sweep is meant to
    detect. Capping equalises the contribution per part.
    """
    scenes = list_scenes(part, SYNTH_ROOT)
    if max_scenes:
        scenes = scenes[:max_scenes]
    if not scenes:
        return {}
    used = 0

    rec = models_info.get(part)
    results: Dict[str, list] = defaultdict(list)

    # ONCE per part, not per scene. Every scene of a part shares one reference cloud --
    # `generate_scenes.py` refuses to mix model frames within a part -- so the analysis is
    # valid for all of them, and paying for it per scene would multiply a 1-3 min cost by the
    # scene count for an identical answer.
    profile = None
    analysed_pts = None
    if _needs_profile(arm_names, rec, ambiguity):
        base_pts, base_nrm = load_reference(scenes[0])
        if len(base_pts) >= 100:
            profile, why = _compute_profile(part, base_pts, base_nrm)
            if why is not None:
                print(f"    [no ambiguity] {why}")
            else:
                analysed_pts = base_pts

    for scene_dir in scenes:
        if max_instances is not None and used >= max_instances:
            break
        ref_pts, ref_nrm = load_reference(scene_dir)
        if len(ref_pts) < 100:
            continue
        cfg = PPFConfig.derive(ref_pts, model_target_points=model_points)
        heat = None
        if profile is not None and profile.per_point_discriminative.size == len(ref_pts):
            # The heat map is index-aligned with the cloud it was analysed on. Scenes of one
            # part are supposed to share that cloud, but `generate_scenes.py` is resumable and
            # a part can accumulate scenes across sessions, so agreement is checked rather
            # than assumed -- scoring the right values against the wrong points is silent.
            why = reference_frames_agree(ref_pts, analysed_pts)
            if why is None:
                heat = profile.per_point_discriminative
            else:
                print(f"    [no heat] {os.path.basename(scene_dir)}: {why}")

        # Normals estimated over 2*tau: the same neighbourhood PPFConfig assumes when it
        # turns depth noise into an angular bin. It moves recall by ~20 points, so it is
        # derived from tau rather than defaulted.
        remaining = None if max_instances is None else max_instances - used
        scene = load_scene(scene_dir, normal_radius=2.0 * cfg.tau,
                           load_scene_cloud=False, max_instances=remaining)
        if not scene.instances:
            continue
        used += len(scene.instances)

        # Symmetry group for scoring: BOP's published annotation when we have it, else the
        # global axes the ambiguity analysis found. Never the view-dependent axes -- those
        # describe genuine failures, not equivalent poses.
        if rec is not None:
            syms = symmetry_transforms_from_bop(rec, rec.get("diameter", 0.0) * 1e-3)
        else:
            syms = symmetry_transforms_from_profile(profile)

        ctx = ArmContext(points=ref_pts, normals=ref_nrm, tau=cfg.tau, heat=heat, part=part)
        base_model = None
        for arm in arm_names:
            spec = ARMS[arm]
            if spec.needs_heat and heat is None:
                continue                            # nothing to weight by on this part
            try:
                a_pts, a_nrm, w = build_arm(arm, ctx)
            except ValueError:
                continue

            if spec.retrains:
                m_pts, m_nrm = downsample(a_pts, a_nrm, cfg.tau)
                if len(m_pts) < 20:
                    continue
                model = PPFModel.train(m_pts, m_nrm, cfg)
                if arm == "A_uniform":
                    base_model = model
            else:
                if base_model is None:              # weight arms reuse arm A's exact table
                    m_pts, m_nrm = downsample(ref_pts, ref_nrm, cfg.tau)
                    base_model = PPFModel.train(m_pts, m_nrm, cfg)
                model = base_model

            weights = resolve_weights(arm, model, w)
            m = model.with_weights(weights) if weights is not None else model

            for inst in scene.instances:
                t0 = time.perf_counter()
                best = match(m, inst.points, inst.normals).best
                dt = time.perf_counter() - t0
                if best is None:
                    results[arm].append(None)
                    continue
                e = evaluate_pose(best.T, inst.T_gt, model.points, syms, profile,
                                  diameter=cfg.diameter)
                results[arm].append({
                    "mssd": e.mssd, "te": e.te, "re_sym": e.re_sym_deg, "re": e.re_deg,
                    "bop": e.passes_bop(), "tight": e.passes(TIGHT), "loose": e.passes(LOOSE),
                    "tagged": e.ambiguity_tagged, "margin": best.peak_margin,
                    "score": best.score, "sec": dt, "overlap": inst.overlap,
                })
    return results


def _agg(rows: list) -> dict:
    found = [r for r in rows if r]
    n = len(rows)
    if not found:
        return {"n": n, "found": 0.0}
    bop = float(np.mean([r["bop"] for r in found]))
    return {
        "n": n,
        "found": len(found) / n,
        "bop": bop,
        "tight": float(np.mean([r["tight"] for r in found])),
        "loose": float(np.mean([r["loose"] for r in found])),
        # CI on the BOP gate, which is the headline number the arms are compared on.
        "ci": 1.96 * float(np.sqrt(max(bop * (1 - bop), 1e-9) / n)),
        "mssd_p50": float(np.median([r["mssd"] for r in found])),
        "te_p50": float(np.median([r["te"] for r in found])),
        "re_p50": float(np.median([r["re_sym"] for r in found])),
        "margin": float(np.mean([r["margin"] for r in found])),
        "tagged": float(np.mean([r["tagged"] for r in found])),
        "sec": float(np.mean([r["sec"] for r in found])),
    }


def _print_table(title: str, per_arm: Dict[str, list]) -> None:
    print(f"\n  {title}")
    print(f"    {'arm':<15} {'n':>5} {'BOP mssd<0.2D':>15} {'@5mm/10d':>9} {'@2mm/5d':>9} "
          f"{'mssd_mm':>8} {'re_deg':>7} {'margin':>7} {'tag':>5} {'s/inst':>7}")
    for arm, rows in per_arm.items():
        a = _agg(rows)
        if not a.get("found"):
            print(f"    {arm:<15} {a['n']:>5}   no detections")
            continue
        gate = f"{a['bop']:.2f} +/-{a['ci']:.2f}"
        print(f"    {arm:<15} {a['n']:>5} {gate:>15} {a['loose']:>9.2f} {a['tight']:>9.2f} "
              f"{a['mssd_p50'] * 1e3:>8.2f} {a['re_p50']:>7.2f} {a['margin']:>7.2f} "
              f"{a['tagged']:>5.2f} {a['sec']:>7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default=None, help="comma-separated part names")
    ap.add_argument("--all", action="store_true", help="every part with scenes on disk")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated arm names")
    ap.add_argument("--scenes", type=int, default=None, help="cap scenes per part")
    ap.add_argument("--max-instances", type=int, default=100,
                    help="instances per part. Equalises each part's weight in the pooled "
                         "table: a 63mm part packs ~160 instances per scene against ~20 for "
                         "a 190mm one, so uncapped pooling is dominated by the small parts")
    ap.add_argument("--model-points", type=int, default=500)
    ap.add_argument("--ambiguity", choices=("auto", "always", "never"), default="auto",
                    help="when to run the 1-3 min per-part ambiguity analysis. auto = only "
                         "when a heat arm needs it or the part has no BOP annotation; always "
                         "= also populate the 'tag' column on uniform-only runs; never = skip")
    ap.add_argument("--out", default=None,
                    help="write raw per-instance results as JSON. A bare filename lands in "
                         "output/bench/, which is already gitignored -- these run to several "
                         "MB and are regenerable, so they do not belong in the tree")
    args = ap.parse_args()

    if args.all:
        parts = sorted(d for d in os.listdir(SYNTH_ROOT)
                       if os.path.isdir(os.path.join(SYNTH_ROOT, d)))
    elif args.parts:
        parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    else:
        sys.exit("give --parts or --all")

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_names:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; choose from {', '.join(ARMS)}")

    models_info = _bop_models_info()
    print(f"{len(parts)} parts x {len(arm_names)} arms: {', '.join(arm_names)}")

    raw: Dict[str, Dict[str, list]] = {}
    by_class: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pooled: Dict[str, list] = defaultdict(list)

    t0 = time.time()
    for i, part in enumerate(parts, 1):
        cls = symmetry_class(models_info.get(part))
        print(f"[{i}/{len(parts)}] {part} ({cls}) ...", flush=True)
        res = run_part(part, arm_names, models_info, args.scenes, args.model_points,
                       args.max_instances, args.ambiguity)
        if not res:
            print("    no scenes -- skipped")
            continue
        raw[part] = res
        for arm, rows in res.items():
            pooled[arm].extend(rows)
            by_class[cls][arm].extend(rows)
        _print_table(f"{part} ({cls})", res)

    print(f"\n{'=' * 100}")
    for cls in sorted(by_class):
        _print_table(f"SYMMETRY CLASS: {cls}", by_class[cls])
    _print_table("POOLED (all parts -- read the per-class tables first)", pooled)

    n_tot = max((len(v) for v in pooled.values()), default=0)
    half = 1.96 * float(np.sqrt(0.25 / n_tot)) if n_tot else float("nan")
    print(f"\n  {n_tot} instances/arm, {(time.time() - t0) / 60:.1f} min")
    print(f"  95% CI half-width at ~0.5 recall: +/- {half * 100:.1f} points -> "
          f"{'a 5-point arm difference IS resolvable' if half < 0.025 else 'a 5-point arm difference is NOT resolvable; generate more scenes'}")
    print("  'tag' = share of ALL detections that are failures a view-dependent ambiguity")
    print("          axis predicts (it is set only on instances that miss the LOOSE gate),")
    print("          so it is bounded above by the miss rate, not by 1.")

    if args.out:
        out = args.out
        if not os.path.isabs(out) and os.path.dirname(out) == "":
            out = os.path.join(_ROOT, "output", "bench", out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"parts": raw,
                       "symmetry_class": {p: symmetry_class(models_info.get(p))
                                          for p in raw}}, f)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
