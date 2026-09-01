"""Profile scene generation + settling across arrangements, to find where the time actually goes.

Tray scenes are reputed to be slow, but the cause has never been measured. This harness runs the
SAME part through random / structured-partition / structured-tray and prints a comparison table,
so the optimisation lever is chosen from data rather than from reading code.

The table discriminates the three standing hypotheses for tray cost:
  (a) _instance_tray_hfield concatenating a ~48k-face mesh per slot -> `hfield_inst` dominates
  (b) MuJoCo hfield narrow-phase per step                          -> `col_narrow` dominates
  (c) settling never converging, so the full settle_time is spent  -> `steps` hits the cap

Always headless (render=False): the passive viewer is blocking and must never run in batch.

Usage:
  python physics/profile_structured_scene.py --shape cube
  python physics/profile_structured_scene.py --shape cuboid_flat --modes tray partition
  python physics/profile_structured_scene.py --mesh path/to/part.stl --repeat 3 --save prof.json
"""

import sys
import os
import argparse
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rich import print as rp

from physics.mujoco_bin_scene import MujocoBinScene, load_part

MODES = {                       # label -> (arrangement, structure_type)
    "random":    ("random", "none"),
    "partition": ("structured", "partition"),
    "tray":      ("structured", "tray"),
}


def _mesh_from_shape(shape):
    """Reuse the synthetic primitives from the bin-scene tests — no STL required."""
    from physics.tests.test_bin_scene import generate_test_mesh
    return generate_test_mesh(shape)


def run_one(part_mesh, convex_meshes, mode, n_parts, settle_time):
    arrangement, structure_type = MODES[mode]
    t0 = time.perf_counter()
    scene = MujocoBinScene(part_mesh, convex_meshes, n_parts=n_parts,
                           render=False, arrangement=arrangement,
                           structure_type=structure_type, settle_time=settle_time,
                           profile=True)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    scene.simulate()
    t_settle = time.perf_counter() - t0

    rep = scene.profile_report()
    timers = rep.get("timers_ms", {})
    return {
        "mode": mode,
        "n_parts": int(scene.n_parts),
        "build_s": t_build,
        "settle_s": t_settle,
        "steps": rep["steps"],
        "sim_time_s": float(scene.data.time),
        "ms_per_step": rep["ms_per_step"],
        "mean_ncon": rep["mean_ncon"],
        "max_ncon": rep["max_ncon"],
        "mean_nefc": rep["mean_nefc"],
        "step_ms": timers.get("mjTIMER_STEP", 0.0),
        "col_ms": timers.get("mjTIMER_POS_COLLISION", 0.0),
        "broad_ms": timers.get("mjTIMER_COL_BROAD", 0.0),
        "narrow_ms": timers.get("mjTIMER_COL_NARROW", 0.0),
        "solve_ms": timers.get("mjTIMER_CONSTRAINT", 0.0),
        "phases_s": rep["phases_s"],
        "census": rep["census"],
    }


def print_report(rows, settle_time):
    rp("\n[bold]Scene profile[/bold] "
       f"(settle_time={settle_time}s, headless, MuJoCo timers via mjcb_time)\n")
    hdr = (f"{'mode':<10}{'parts':>6}{'build':>8}{'settle':>9}{'steps':>8}{'ms/step':>9}"
           f"{'ncon':>7}{'narrow':>9}{'solve':>8}{'geoms':>7}{'hf':>4}{'bin_tris':>10}")
    rp(hdr)
    rp("-" * len(hdr))
    for r in rows:
        c = r["census"]
        rp(f"{r['mode']:<10}{r['n_parts']:>6}{r['build_s']:>7.2f}s{r['settle_s']:>8.2f}s"
           f"{r['steps']:>8}{r['ms_per_step']:>9.3f}{r['mean_ncon']:>7.0f}"
           f"{r['narrow_ms']:>8.3f}m{r['solve_ms']:>7.3f}m"
           f"{c.get('ngeom', 0):>7}{c.get('hfield_geoms', 0):>4}{c.get('bin_tris', 0):>10,}")

    rp("\n[bold]Setup phases (s)[/bold]")
    keys = sorted({k for r in rows for k in r["phases_s"]})
    rp(f"{'mode':<10}" + "".join(f"{k:>18}" for k in keys))
    for r in rows:
        rp(f"{r['mode']:<10}" + "".join(f"{r['phases_s'].get(k, 0.0):>18.3f}" for k in keys))

    rp("\n[dim]narrow/solve are mean ms per mj_step call (mjTIMER_COL_NARROW / "
       "mjTIMER_CONSTRAINT). hf = hfield geom count.[/dim]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mesh", type=str, default=None, help="path to an STL part")
    src.add_argument("--shape", type=str, default="cube",
                     help="synthetic primitive (cube, cube_big, cuboid_long, rod_long, "
                          "cuboid_flat, plate_big)")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--n_parts", type=int, default=12,
                    help="requested count (structured modes override this with grid capacity)")
    ap.add_argument("--settle-time", type=float, default=5.0)
    ap.add_argument("--repeat", type=int, default=1, help="runs per mode; the median is kept")
    ap.add_argument("--save", type=str, default=None, help="write raw results to this JSON file")
    args = ap.parse_args()

    if args.mesh:
        part_mesh, convex_meshes = load_part(args.mesh)
    else:
        part_mesh, convex_meshes = _mesh_from_shape(args.shape)

    rows = []
    for mode in args.modes:
        runs = [run_one(part_mesh, convex_meshes, mode, args.n_parts, args.settle_time)
                for _ in range(args.repeat)]
        # Median by wall-clock settle time — robust to a single noisy run.
        runs.sort(key=lambda r: r["settle_s"])
        rows.append(runs[len(runs) // 2])

    print_report(rows, args.settle_time)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(rows, f, indent=2, default=float)
        rp(f"\n[green]Saved[/green] {args.save}")


if __name__ == "__main__":
    main()
