"""Check ``geometry/ambiguity.py`` against BOP's published symmetry annotations.

Run::

    python bench/validate_ambiguity.py
    python bench/validate_ambiguity.py --verbose

Until now the module has only been tested against synthetic primitives whose answers we
wrote ourselves, which cannot catch a shared misconception between the test and the code.
BOP ships per-object ``symmetries_discrete`` and ``symmetries_continuous`` in
``models_info.json`` -- independently authored ground truth for 30 industrial parts -- so this
is the first external check it has had.

Profiles are computed here, straight from the BOP meshes in ``mesh_raw/``.  This script needs
no scene generation and no exported bundle, so it runs immediately after
``bench/fetch_dataset.py``.  That is sound because the comparison reads only ``is_global`` and
``fold``, both of which are frame-independent, and ``analyse_ambiguity`` centres its input
itself -- so a mesh in BOP's own coordinates, in millimetres, is fine once unit-scaled.

The verdict moves with sampling density
    ``analyse_ambiguity`` is deterministic for a fixed (mesh, cloud, cfg, seed) but **not**
    stable against a changed input: measured on 25333MB000, three sample densities gave
    dominant folds C4 / C1 / C2, one of them on an axis 45 degrees from the others.  The cloud
    is therefore sampled at a pinned count and a pinned seed, and any agreement figure below is
    only meaningful together with that density.  Re-running at a different ``--n-points`` is a
    legitimate robustness probe; comparing two runs taken at different densities is not.

Reading the result
    The comparison is only meaningful on **global** symmetry. BOP annotates symmetries of
    the whole object; ``AmbiguityProfile`` additionally reports *view-dependent* axes, which
    map a visible patch elsewhere without mapping the model onto itself. A part BOP calls
    asymmetric can legitimately carry view-dependent axes -- that is the capability the module
    exists for and precisely what the BOP annotation cannot express. So a disagreement in the
    ``is_global`` column is a bug; a view-dependent axis on an "asymmetric" part is not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MESH_ROOT = os.path.join(_ROOT, "mesh_raw")

MESH_EXT = (".ply", ".stl", ".obj")


def bop_expectation(rec: dict) -> tuple:
    """``(class, fold)`` implied by a BOP entry.

    ``symmetries_discrete`` lists the non-identity members of the group, so N entries means
    a C(N+1) rotation -- a single 180-degree transform is C2, not C1.
    """
    if rec.get("symmetries_continuous"):
        return "continuous", 0
    disc = rec.get("symmetries_discrete", [])
    if disc:
        return "discrete", len(disc) + 1
    return "asymmetric", 1


def mesh_paths() -> Dict[str, str]:
    """Map ``obj_000005`` -> its mesh file under ``mesh_raw/``.

    Keyed on the bare filename stem, which is exactly how ``models_info.json`` keys are
    rendered (``obj_%06d``), so the two dictionaries intersect directly.
    """
    out: Dict[str, str] = {}
    for root, _, files in os.walk(MESH_ROOT):
        for f in sorted(files):
            stem, ext = os.path.splitext(f)
            if ext.lower() in MESH_EXT and stem not in out:
                out[stem] = os.path.join(root, f)
    return out


def compute_profiles(parts: List[str], n_points: int, cfg) -> Dict[str, object]:
    """Analyse each part straight from its mesh. Minutes per part -- see the module docstring."""
    import open3d as o3d

    from geometry.ambiguity import analyse_ambiguity
    from geometry.mesh_repair import analyze_mesh

    paths = mesh_paths()
    out: Dict[str, object] = {}
    for i, part in enumerate(parts, 1):
        path = paths.get(part)
        if path is None:
            print(f"  [{i}/{len(parts)}] {part:<14} no mesh under mesh_raw/ -- skipped")
            continue
        print(f"  [{i}/{len(parts)}] {part:<14} ", end="", flush=True)
        t0 = time.time()

        mesh = o3d.io.read_triangle_mesh(path)
        cleaned, report = analyze_mesh(mesh)
        if report.errors:
            print(f"refused: {'; '.join(report.errors)}")
            continue
        # BOP ships millimetres; every tolerance in AmbiguityConfig that is not part-relative
        # is an absolute sensor-physics floor in metres, so this conversion is load-bearing
        # rather than cosmetic. Same call and same center as ImportMeshStage.
        if report.unit_scale != 1.0:
            cleaned.scale(report.unit_scale, center=(0, 0, 0))
        cleaned.compute_vertex_normals()

        # Pinned seed + pinned count: the answer moves with density, so the sampling must not
        # be free to drift between runs. Matches visualize_ambiguity._prepare.
        o3d.utility.random.seed(0)
        pcd = cleaned.sample_points_uniformly(n_points, use_triangle_normal=False)

        out[part] = analyse_ambiguity(cleaned, pcd, cfg)
        print(f"{time.time() - t0:6.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="one line per part")
    ap.add_argument("--parts", nargs="+", default=None, metavar="PART",
                    help="only these parts (e.g. obj_000005 obj_000013). The full sweep is "
                         "30 parts at minutes each, so this is the way to spot-check.")
    ap.add_argument("--n-points", type=int, default=6000,
                    help="points sampled from each mesh. The answer is density-sensitive "
                         "(see the module docstring) -- changing this changes the verdict, so "
                         "report it alongside any figure taken from this script.")
    ap.add_argument("--n-views", type=int, default=None,
                    help="viewpoints in the visibility sweep (default: AmbiguityConfig's)")
    args = ap.parse_args()

    from geometry.ambiguity import AmbiguityConfig
    cfg = AmbiguityConfig() if args.n_views is None else AmbiguityConfig(n_views=args.n_views)

    info: Dict[str, dict] = {}
    for root, _, files in os.walk(MESH_ROOT):
        if "models_info.json" in files:
            with open(os.path.join(root, "models_info.json")) as f:
                for k, rec in json.load(f).items():
                    info[f"obj_{int(k):06d}"] = rec
    if not info:
        sys.exit("no models_info.json under mesh_raw/ -- run bench/fetch_dataset.py first")

    wanted = sorted(info) if args.parts is None else list(args.parts)
    unknown = [p for p in wanted if p not in info]
    if unknown:
        sys.exit(f"no BOP annotation for: {', '.join(unknown)}\n"
                 f"known parts: {', '.join(sorted(info))}")

    print(f"Analysing {len(wanted)} part(s) at {args.n_points} points, "
          f"n_views={cfg.n_views}. This takes minutes per part.\n")
    profiles = compute_profiles(wanted, args.n_points, cfg)

    shared = sorted(set(info) & set(profiles))
    if not shared:
        sys.exit("no parts could be analysed -- check that mesh_raw/ holds the meshes named "
                 "in models_info.json (run bench/fetch_dataset.py)")

    print(f"\n{len(shared)} parts with both a BOP annotation and a computed profile\n")
    if args.verbose:
        print(f"  {'part':<14} {'BOP':<12} {'fold':>5} | {'global axes':>11} "
              f"{'fold':>5} {'view-dep':>9}  {'agree':>6}")

    confusion: Dict[tuple, int] = Counter()
    fold_ok = fold_bad = 0
    for part in shared:
        exp_cls, exp_fold = bop_expectation(info[part])
        prof = profiles[part]
        g = [a for a in prof.axes if a.is_global]
        v = [a for a in prof.axes if not a.is_global]

        if not g:
            got_cls, got_fold = "asymmetric", 1
        elif any(a.fold == 0 for a in g):
            got_cls, got_fold = "continuous", 0
        else:
            got_cls = "discrete"
            got_fold = max(a.fold for a in g)

        confusion[(exp_cls, got_cls)] += 1
        agree = exp_cls == got_cls
        if agree and exp_cls == "discrete":
            fold_ok += int(got_fold == exp_fold)
            fold_bad += int(got_fold != exp_fold)
        if args.verbose:
            print(f"  {part:<14} {exp_cls:<12} {exp_fold:>5} | {len(g):>11} "
                  f"{got_fold:>5} {len(v):>9}  {'ok' if agree else 'MISMATCH':>6}")

    classes = ["asymmetric", "discrete", "continuous"]
    print(f"\n  confusion (rows = BOP, cols = ambiguity module):")
    print(f"    {'':<12}" + "".join(f"{c:>12}" for c in classes))
    for e in classes:
        row = "".join(f"{confusion.get((e, g), 0):>12}" for g in classes)
        print(f"    {e:<12}{row}")

    agree = sum(confusion[(c, c)] for c in classes)
    print(f"\n  class agreement: {agree}/{len(shared)} ({agree / len(shared):.0%})")
    if fold_ok + fold_bad:
        print(f"  fold order on agreed discrete parts: {fold_ok}/{fold_ok + fold_bad} exact")
    print("\n  A view-dependent axis on a BOP-'asymmetric' part is expected, not an error:")
    print("  BOP annotates whole-object symmetry and has no way to express one.")


if __name__ == "__main__":
    main()
