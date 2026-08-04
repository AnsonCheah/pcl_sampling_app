"""Benchmark the vanilla PPF matcher against synthetic scenes with ground-truth poses.

Run::

    python -m registration.ppf.bench.run --parts obj_000018 --max-instances 50
    python -m registration.ppf.bench.run --all --scenes 2 --out ppf_bench.json
    python -m registration.ppf.bench.run --all --models-info mesh_raw/tless/models_info.json

Scenes come from ``--scenes-root`` (or ``PPF_SCENES_ROOT``), in the directory format
``dataset.load_scene`` reads.  Symmetry annotations come from a BOP ``models_info.json`` when
one is supplied; without it every part is scored as asymmetric, which **understates** recall
on symmetric parts rather than flattering it -- a correct pose in the wrong symmetry orbit is
counted wrong.  Supply the file whenever you have it.

Two measurement choices worth stating, because both move the numbers:

* **Normals are re-estimated from the points**, not read from the stored PLY.  The stored
  normals came from the mesh via raycast and are exact; a real sensor delivers depth and
  normals get estimated from noisy points, so benchmarking against the stored ones flatters
  the matcher.  ``--normals stored`` isolates normal-estimation error from matcher error.
* **The normal-estimation radius is ``2 * tau``**, tied to the matcher's own quantisation.
  It is not cosmetic: moving it from ``2*tau`` to 4x the reference spacing moved recall
  @5mm/10deg from 0.77 to 0.62 on one scene and reordered the arms against each other.

The headline is **BOP recall (MSSD < 0.2 D)**, not the 2mm/5deg gate.  PPF is a coarse stage
with no refinement, so the tight gate judges it for missing a step it does not contain; it is
reported alongside, and the gap between them is roughly what a fine ICP stage would have to
close.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

# Relative throughout: the package must not know what it is installed as. Absolute
# ``registration.ppf...`` imports would work in this repository and nowhere else, which is
# exactly the coupling the extraction was meant to remove. Run with ``-m``, not by path.
from .. import PPFConfig, PPFModel, cupy_available, downsample, match, match_many
from .dataset import default_scenes_root, list_parts, list_scenes, load_reference, load_scene
from .metrics import PoseError, evaluate_pose, summarise, symmetry_transforms_from_bop


def load_models_info(path: Optional[str]) -> Dict[str, dict]:
    """Map ``obj_000005`` -> its BOP ``models_info`` entry. Empty dict when unavailable."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return {f"obj_{int(k):06d}": v for k, v in json.load(f).items()}


def symmetry_class(rec: Optional[dict]) -> str:
    """BOP's own three-way split. Parts differ enormously in difficulty by this axis, so a
    pooled mean over mixed geometry hides more than it shows."""
    # `is None` on purpose: a BOP entry for an asymmetric part is a real record that simply
    # carries no symmetry keys, and `not rec` would file it as "unknown" -- losing the
    # distinction between "annotated asymmetric" and "never annotated".
    if rec is None:
        return "unknown"
    if rec.get("symmetries_continuous"):
        return "continuous"
    if rec.get("symmetries_discrete"):
        return "discrete"
    return "asymmetric"


def benchmark_part(part: str,
                   scenes_root: str,
                   n_scenes: Optional[int] = None,
                   max_instances: Optional[int] = None,
                   model_target_points: int = 500,
                   normals: str = "estimated",
                   models_info: Optional[dict] = None,
                   top_k: int = 1,
                   workers: Optional[int] = None,
                   backend: str = "numpy",
                   verbose: bool = True) -> dict:
    """Train once per part, then match every instance of every scene."""
    scene_dirs = list_scenes(part, scenes_root)
    if n_scenes:
        scene_dirs = scene_dirs[:n_scenes]
    if not scene_dirs:
        return {"part": part, "error": "no scenes"}

    # The reference cloud is identical across a part's scenes, so parameters and the trained
    # table are derived once. Re-deriving per scene would also let tau drift between scenes
    # and make their numbers incomparable.
    ref_pts, ref_nrm = load_reference(scene_dirs[0])
    if len(ref_pts) < 16:
        return {"part": part, "error": "reference cloud too small"}

    t0 = time.perf_counter()
    cfg = PPFConfig.derive(ref_pts, model_target_points=model_target_points)
    m_pts, m_nrm = downsample(ref_pts, ref_nrm, cfg.tau)
    model = PPFModel.train(m_pts, m_nrm, cfg)
    t_train = time.perf_counter() - t0

    rec = (models_info or {}).get(part)
    syms = symmetry_transforms_from_bop(rec, cfg.diameter) if rec else []
    sym_class = symmetry_class(rec)

    if verbose:
        print(f"\n=== {part}  [{sym_class}] ===")
        print(cfg.describe())
        print(f"  model         {model.n_points:8d} points, {len(model.keys)} entries, "
              f"trained in {t_train:.2f}s")
        print(f"  symmetry      {len(syms) if syms else 1} transform(s) from "
              f"{'models_info' if rec else 'no annotation (scored as asymmetric)'}")

    errors: List[PoseError] = []
    n_seen = n_failed = 0
    t_match = 0.0
    backend_used = backend
    for sd in scene_dirs:
        scene = load_scene(sd, normal_radius=2.0 * cfg.tau, normals=normals,
                           load_scene_cloud=False, max_instances=max_instances)
        if not scene.instances:
            continue
        # Whole scene at once so the thread pool has something to fill: matching one instance
        # per pool costs more in setup than it saves. Order is preserved, which matters --
        # results are joined to ground truth by position.
        t1 = time.perf_counter()
        results = match_many(model, [(i.points, i.normals) for i in scene.instances],
                             workers=workers, top_k=top_k, backend=backend)
        t_match += time.perf_counter() - t1
        if results:
            backend_used = results[0].backend
        for inst, res in zip(scene.instances, results):
            n_seen += 1
            if res.best is None:
                n_failed += 1
                continue
            errors.append(evaluate_pose(res.best.T, inst.T_gt, model.points,
                                        syms, diameter=cfg.diameter))

    out = {"part": part, "symmetry_class": sym_class, "n_scenes": len(scene_dirs),
           "n_instances": n_seen, "n_no_pose": n_failed,
           "diameter_mm": cfg.diameter * 1e3, "tau_mm": cfg.tau * 1e3,
           "model_points": int(model.n_points),
           "train_s": t_train,
           # Wall-clock per instance. With `workers > 1` this is throughput, not latency:
           # it is the elapsed time for the whole scene divided by its instances.
           "s_per_instance": t_match / max(n_seen, 1),
           "backend": backend_used,
           "workers": workers,
           **summarise(errors, n_expected=n_seen)}

    if verbose:
        print(f"  instances     {n_seen:8d}  ({n_failed} returned no pose)")
        if errors:
            print(f"  BOP recall    {out['recall_bop']:8.3f}  +/- {out['ci95_half']:.3f}"
                  f"   (MSSD < 0.2 x D)")
            print(f"  @5mm/10deg    {out['recall_loose']:8.3f}")
            print(f"  @2mm/5deg     {out['recall_tight']:8.3f}")
            print(f"  MSSD p50      {out['mssd_p50'] * 1e3:8.2f} mm")
            print(f"  TE p50        {out['te_p50'] * 1e3:8.2f} mm")
            print(f"  RE_sym p50    {out['re_sym_p50']:8.2f} deg")
        print(f"  s/instance    {out['s_per_instance']:8.3f}  "
              f"[{out['backend']}, workers={out['workers'] or 'auto'}]")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", help="comma-separated part names")
    ap.add_argument("--all", action="store_true", help="every part under the scenes root")
    ap.add_argument("--scenes-root", default=None)
    ap.add_argument("--scenes", type=int, default=None, help="max scenes per part")
    ap.add_argument("--max-instances", type=int, default=None, help="max instances per scene")
    ap.add_argument("--model-points", type=int, default=500)
    ap.add_argument("--normals", default="estimated", choices=["estimated", "stored"])
    ap.add_argument("--models-info", default=None, help="BOP models_info.json for symmetry")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None,
                    help="threads for instance-level parallelism (default: min(8, cores); "
                         "1 on the cupy backend, where the device is already saturated)")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "cupy", "auto"],
                    help="array module for the vote stage; cupy falls back to numpy when no "
                         "usable GPU is present, and the table reports what actually ran")
    ap.add_argument("--out", default=None, help="write per-part JSON here")
    args = ap.parse_args()

    root = args.scenes_root or default_scenes_root()
    if args.parts:
        parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    elif args.all:
        parts = list_parts(root)
    else:
        ap.error("pass --parts or --all")

    if not parts:
        print(f"no parts found under {root}")
        return

    info = load_models_info(args.models_info)
    print(f"scenes root: {root}")
    print(f"parts:       {len(parts)}")
    print(f"symmetry:    {'models_info loaded' if info else 'none - all scored asymmetric'}")
    print(f"backend:     {args.backend}"
          f"{'' if args.backend == 'numpy' else ' (gpu available: %s)' % cupy_available()}")
    print(f"workers:     {args.workers if args.workers else 'auto'}")

    rows = []
    for p in parts:
        try:
            rows.append(benchmark_part(
                p, root, n_scenes=args.scenes, max_instances=args.max_instances,
                model_target_points=args.model_points, normals=args.normals,
                models_info=info, top_k=args.top_k, workers=args.workers,
                backend=args.backend))
        except Exception as exc:                       # one bad part must not end the sweep
            print(f"  [FAIL] {p}: {type(exc).__name__}: {exc}")
            rows.append({"part": p, "error": f"{type(exc).__name__}: {exc}"})

    _print_table(rows)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


def _print_table(rows: List[dict]) -> None:
    good = [r for r in rows if r.get("n", 0)]
    if not good:
        print("\nno scored instances")
        return

    print(f"\n{'part':<16}{'class':<12}{'n':>6}{'BOP':>8}{'+/-':>7}"
          f"{'5/10':>8}{'2/5':>8}{'MSSDp50':>10}{'s/inst':>9}")
    print("-" * 84)
    for r in sorted(good, key=lambda r: -r["recall_bop"]):
        print(f"{r['part']:<16}{r['symmetry_class']:<12}{r['n_instances']:>6}"
              f"{r['recall_bop']:>8.3f}{r['ci95_half']:>7.3f}"
              f"{r['recall_loose']:>8.3f}{r['recall_tight']:>8.3f}"
              f"{r['mssd_p50'] * 1e3:>9.2f}m{r['s_per_instance']:>9.3f}")

    # Pooled overall, then per symmetry class. The split is the informative one: a method
    # that wins on cylinders and loses on brackets is a failure of the "works for part #5000
    # as well as part #1" claim, and pooling would report it as a modest win.
    n_all = sum(r["n_instances"] for r in good)
    w = lambda key: sum(r[key] * r["n_instances"] for r in good) / max(n_all, 1)
    print("-" * 84)
    print(f"{'ALL':<16}{'':<12}{n_all:>6}{w('recall_bop'):>8.3f}{'':>7}"
          f"{w('recall_loose'):>8.3f}{w('recall_tight'):>8.3f}"
          f"{w('mssd_p50') * 1e3:>9.2f}m{w('s_per_instance'):>9.3f}")

    for cls in ("asymmetric", "discrete", "continuous", "unknown"):
        sub = [r for r in good if r["symmetry_class"] == cls]
        if not sub:
            continue
        n = sum(r["n_instances"] for r in sub)
        ws = lambda key: sum(r[key] * r["n_instances"] for r in sub) / max(n, 1)
        print(f"{'  ' + cls:<16}{'':<12}{n:>6}{ws('recall_bop'):>8.3f}{'':>7}"
              f"{ws('recall_loose'):>8.3f}{ws('recall_tight'):>8.3f}"
              f"{ws('mssd_p50') * 1e3:>9.2f}m{ws('s_per_instance'):>9.3f}")

    failed = [r for r in rows if r.get("error")]
    if failed:
        print(f"\n{len(failed)} part(s) failed:")
        for r in failed:
            print(f"  {r['part']}: {r['error']}")


if __name__ == "__main__":
    main()
