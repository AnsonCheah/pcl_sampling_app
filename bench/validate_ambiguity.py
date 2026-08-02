"""Check ``geometry/ambiguity.py`` against BOP's published symmetry annotations.

Run::

    python bench/validate_ambiguity.py
    python bench/validate_ambiguity.py --verbose

Until now the module has only been tested against synthetic primitives whose answers we
wrote ourselves, which cannot catch a shared misconception between the test and the code.
BOP ships per-object ``symmetries_discrete`` and ``symmetries_continuous`` in
``models_info.json`` — independently authored ground truth for 30 industrial parts — so this
is the first external check it has had.

Profiles are read from the sidecars the scene sweep already wrote, not recomputed; the
analysis costs minutes per part and the sweep has already paid for it.

Reading the result
    The comparison is only meaningful on **global** symmetry. BOP annotates symmetries of
    the whole object; ``AmbiguityProfile`` additionally reports *view-dependent* axes, which
    map a visible patch elsewhere without mapping the model onto itself. A part BOP calls
    asymmetric can legitimately carry view-dependent axes — that is the capability the module
    exists for and precisely what the BOP annotation cannot express. So a disagreement in the
    ``is_global`` column is a bug; a view-dependent axis on an "asymmetric" part is not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SYNTH_ROOT = os.path.join(_ROOT, "output", "synthetic_target")


def bop_expectation(rec: dict) -> tuple:
    """``(class, fold)`` implied by a BOP entry.

    ``symmetries_discrete`` lists the non-identity members of the group, so N entries means
    a C(N+1) rotation — a single 180-degree transform is C2, not C1.
    """
    if rec.get("symmetries_continuous"):
        return "continuous", 0
    disc = rec.get("symmetries_discrete", [])
    if disc:
        return "discrete", len(disc) + 1
    return "asymmetric", 1


def load_profiles() -> Dict[str, object]:
    from geometry.ambiguity import load_ambiguity_profile

    out = {}
    if not os.path.isdir(SYNTH_ROOT):
        return out
    for part in sorted(os.listdir(SYNTH_ROOT)):
        pdir = os.path.join(SYNTH_ROOT, part)
        if not os.path.isdir(pdir):
            continue
        for scene in sorted(os.listdir(pdir)):
            f = os.path.join(pdir, scene, "ambiguity_profile.json")
            if os.path.exists(f):
                try:
                    out[part] = load_ambiguity_profile(f)
                except Exception:
                    pass
                break                      # one profile per part is enough; they agree
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="one line per part")
    args = ap.parse_args()

    info: Dict[str, dict] = {}
    for root, _, files in os.walk(os.path.join(_ROOT, "mesh_raw")):
        if "models_info.json" in files:
            with open(os.path.join(root, "models_info.json")) as f:
                for k, rec in json.load(f).items():
                    info[f"obj_{int(k):06d}"] = rec
    if not info:
        sys.exit("no models_info.json under mesh_raw/ — run bench/fetch_dataset.py first")

    profiles = load_profiles()
    shared = sorted(set(info) & set(profiles))
    if not shared:
        sys.exit("no parts have both a BOP annotation and a saved ambiguity profile "
                 "— run bench/generate_scenes.py first")

    print(f"{len(shared)} parts with both a BOP annotation and a computed profile\n")
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
