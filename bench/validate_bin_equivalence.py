"""Does a dynamically-sized bin model the same stacking as the full-size bin?

This is the empirical check behind dynamic bin sizing. The claim being tested is that stacking
depth is set by fill rate and part geometry, NOT by bin size -- so a smaller bin at the same fill
rate produces the same *kind* of pile with far fewer parts, and the parts removed were only
widening the pile rather than deepening it.

What it compares, for one part at one fill rate, between the max bin and the solved bin:

  floor-contact fraction  parts resting on the bin floor vs on OTHER parts. THE headline metric:
                          floor-resting parts land in flat stable poses, and only part-on-part
                          support produces the tilted poses a real bin-picking scene contains.
  tilt distribution       angle of each part's local +Z from world +Z (mean/median/p90, plus a
                          two-sample KS statistic between the bins).
  neighbour count         mean number of distinct other parts each part touches.
  pile depth              realised pile height in effective-layer (h_eff) units.

A matching floor-contact fraction and tilt distribution justify substituting the small bin.
An EXCESS of floor contacts in the small bin means stacking depth was lost; an excess of
wall contacts means MIN_PARTS_PER_LAYER is too low and the footprint is too cramped.

Usage:
  python bench/validate_bin_equivalence.py --shape cube --fill 0.2
  python bench/validate_bin_equivalence.py --mesh part.stl --fill 0.5 --save equiv.json

NOTE: the max-bin arm uses the legacy auto-count, which for a small part is hundreds of parts
and can take many minutes. That asymmetry is the point of the exercise.
"""

import sys
import os
import argparse
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
from rich import print as rp
from scipy.spatial.transform import Rotation as R

from physics.mujoco_bin_scene import (
    MAX_BIN_DIM, BIN_TOP_MARGIN_FRAC, MIN_AUTO_PARTS, MAX_AUTO_PARTS,
    MujocoBinScene, load_part, obb_packing_factor, solve_bin_dim, stable_layer_heights,
)


def legacy_part_count(part_mesh, fill_rate):
    """The count SceneStage._auto_part_count derives for the fixed max bin."""
    bw, bl, bh, _ = MAX_BIN_DIM
    bin_vol = bw * bl * bh * (1.0 - BIN_TOP_MARGIN_FRAC)
    obb_vol = max(float(part_mesh.bounding_box_oriented.volume), 1e-9)
    n = round(fill_rate * obb_packing_factor(part_mesh) * bin_vol / obb_vol)
    return int(np.clip(n, MIN_AUTO_PARTS, MAX_AUTO_PARTS))


def analyse(scene, h_eff):
    """Pose/contact statistics for a settled scene."""
    model, data = scene.model, scene.data
    body_ids = list(scene._body_ids)
    part_bodies = set(body_ids)

    # Tilt: polar angle of each part's local +Z away from world +Z.
    rot = R.from_quat(data.xquat[body_ids], scalar_first=True).as_matrix()
    tilt = np.degrees(np.arccos(np.clip(rot[:, 2, 2], -1.0, 1.0)))

    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_floor")
    geom_body = np.asarray(model.geom_bodyid)

    on_floor, on_wall = set(), set()
    neighbours = defaultdict(set)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(geom_body[g1]), int(geom_body[g2])
        p1, p2 = b1 in part_bodies, b2 in part_bodies
        if p1 and p2 and b1 != b2:
            neighbours[b1].add(b2)
            neighbours[b2].add(b1)
        elif p1 or p2:                       # part vs bin fixture
            part_b = b1 if p1 else b2
            fixture_g = g2 if p1 else g1
            (on_floor if fixture_g == floor_gid else on_wall).add(part_b)

    z = data.xpos[body_ids, 2]
    n = max(len(body_ids), 1)
    return {
        "n_parts": len(body_ids),
        "floor_frac": len(on_floor) / n,
        "wall_frac": len(on_wall) / n,
        "mean_neighbours": float(np.mean([len(neighbours[b]) for b in body_ids])),
        "tilt_mean": float(tilt.mean()),
        "tilt_median": float(np.median(tilt)),
        "tilt_p90": float(np.percentile(tilt, 90)),
        "pile_layers": float((z.max() - z.min()) / h_eff) if h_eff > 0 else 0.0,
        "tilt": tilt.tolist(),
    }


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap); 0 = identical distributions."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a)
                               - np.searchsorted(b, grid, "right") / len(b))))


def run_arm(part_mesh, convex, bin_dim, n_parts, settle_time, h_eff, seed=17, batch_cap=None):
    np.random.seed(seed)
    scene = MujocoBinScene(part_mesh, convex, n_parts=n_parts, bin_dim=bin_dim,
                           settle_time=settle_time, render=False,
                           max_release_batches=batch_cap)
    scene.simulate()
    stats = analyse(scene, h_eff)
    stats["escaped"] = scene.verify_parts_in_bin()["n_out"]
    stats["bin_dim"] = [float(v) for v in bin_dim]
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mesh", type=str, default=None)
    src.add_argument("--shape", type=str, default="cube")
    ap.add_argument("--fill", type=float, default=0.2)
    ap.add_argument("--settle-time", type=float, default=5.0)
    ap.add_argument("--batch-cap", type=int, default=None,
                    help="override MAX_RELEASE_BATCHES for BOTH arms. The batch-wave floor "
                         "fires on the max bin too (500 parts: 50 waves of 10 -> 8 of 63), so "
                         "the default max-bin arm is NOT the historical baseline. Pass a large "
                         "value (e.g. 999) to restore the old 10-per-wave release and A/B the "
                         "settling change on its own.")
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    if args.mesh:
        part_mesh, convex = load_part(args.mesh)
    else:
        from physics.tests.test_bin_scene import generate_test_mesh
        part_mesh, convex = generate_test_mesh(args.shape)

    layer_h, _ = stable_layer_heights(part_mesh)
    h_eff = layer_h / obb_packing_factor(part_mesh)

    dyn_bin, dyn_n, dyn_layers = solve_bin_dim(part_mesh, args.fill)
    leg_n = legacy_part_count(part_mesh, args.fill)

    rp(f"\n[bold]Bin equivalence[/bold]  fill={args.fill:.0%}  h_eff={h_eff * 1000:.1f} mm")
    rp(f"  max bin     {MAX_BIN_DIM[0]:.3f} x {MAX_BIN_DIM[1]:.3f} x {MAX_BIN_DIM[2]:.3f} m"
       f"  n={leg_n}")
    rp(f"  solved bin  {dyn_bin[0]:.3f} x {dyn_bin[1]:.3f} x {dyn_bin[2]:.3f} m"
       f"  n={dyn_n}   (~{dyn_layers:.2f} layers)\n")

    rows = {
        "max_bin": run_arm(part_mesh, convex, MAX_BIN_DIM, leg_n, args.settle_time, h_eff,
                           batch_cap=args.batch_cap),
        "solved_bin": run_arm(part_mesh, convex, dyn_bin, dyn_n, args.settle_time, h_eff,
                              batch_cap=args.batch_cap),
    }

    fields = [("n_parts", "{:.0f}"), ("floor_frac", "{:.3f}"), ("wall_frac", "{:.3f}"),
              ("mean_neighbours", "{:.2f}"), ("tilt_mean", "{:.1f}"), ("tilt_median", "{:.1f}"),
              ("tilt_p90", "{:.1f}"), ("pile_layers", "{:.2f}"), ("escaped", "{:.0f}")]
    rp(f"{'metric':<18}{'max bin':>12}{'solved bin':>14}{'delta':>12}")
    rp("-" * 56)
    for key, fmt in fields:
        a, b = rows["max_bin"][key], rows["solved_bin"][key]
        rp(f"{key:<18}{fmt.format(a):>12}{fmt.format(b):>14}{fmt.format(b - a):>12}")

    ks = ks_statistic(rows["max_bin"]["tilt"], rows["solved_bin"]["tilt"])
    rp(f"\ntilt distribution KS statistic: {ks:.3f}  (0 = identical)")

    d_floor = rows["solved_bin"]["floor_frac"] - rows["max_bin"]["floor_frac"]
    if d_floor > 0.15:
        rp("[yellow]Solved bin has materially MORE floor-resting parts - stacking depth was "
           "lost. Raise the fill rate, or LAYERS_AT_FULL_FILL.[/yellow]")
    elif rows["solved_bin"]["wall_frac"] - rows["max_bin"]["wall_frac"] > 0.20:
        rp("[yellow]Solved bin has materially MORE wall-contacting parts - the footprint is "
           "cramped. Raise MIN_PARTS_PER_LAYER.[/yellow]")
    else:
        rp("[green]Distributions comparable - the smaller bin is a valid substitute.[/green]")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(rows, f, indent=2, default=float)
        rp(f"\nSaved {args.save}")


if __name__ == "__main__":
    main()
