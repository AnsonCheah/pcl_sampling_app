"""Batch scene generation: meshes in ``mesh_raw/`` -> scenes in ``output/synthetic_target/``.

Run::

    python bench/generate_scenes.py --meshes mesh_raw/tless --scenes 4
    python bench/generate_scenes.py --meshes mesh_raw/tless --limit 3 --scenes 1   # smoke test
    python bench/generate_scenes.py --meshes mesh_raw/tless --scenes 20 --only obj_000018

Expect hours, not minutes: roughly 1-3 min per part for raycast + downsample + ambiguity,
plus 1-2 min per scene for the MuJoCo settle and the sensor simulation.  The run is
therefore **resumable** — a part already holding the requested number of scenes is skipped,
and a part that fails is logged and stepped over rather than taking the sweep down with it.

Two headless-only hazards this works around, both of which fail *silently*:

* ``app.main_thread(fn)`` drops ``fn`` entirely when headless (``app.py``), and
  ``_express_sampling_worker`` sets the stage through it.  So ``app.stage`` never becomes
  ``SAVE``, ``SaveStage.worker()`` matches neither of its two branches, and the reference
  bundle is simply not written — no error, no output.  Every stage transition here is
  therefore made with a direct ``app.set_stage(...)``.
* ``SceneStage.rendering_flag`` opens MuJoCo's *blocking* passive viewer.  Left on, a batch
  run stops at the first scene and waits for a window nobody is watching.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SYNTH_ROOT = os.path.join(_ROOT, "output", "synthetic_target")

MESH_EXT = (".stl", ".ply", ".obj")


def existing_scene_count(part: str) -> int:
    d = os.path.join(SYNTH_ROOT, part)
    if not os.path.isdir(d):
        return 0
    return sum(1 for n in os.listdir(d)
               if n.startswith("scene_") and os.path.isdir(os.path.join(d, n)))


def generate_for_mesh(mesh_path: str, n_scenes: int, fill_rate: float,
                      arrangement: str, run_ambiguity: bool,
                      adaptive: bool) -> dict:
    """Full pipeline for one mesh. Returns a summary dict."""
    from app import MeshSamplingApp
    from enums import Stage

    part = os.path.splitext(os.path.basename(mesh_path))[0]
    t_start = time.time()

    app = MeshSamplingApp(headless=True, mesh_path=mesh_path)
    app.stages[Stage.IMPORT_MESH]._run_worker()
    if app.target_mesh is None:
        raise RuntimeError("mesh failed to import")

    # Every path through the app centres the mesh before sampling
    # (`start_express_sampling`, `run_headless`, `_batch_sampling_worker`); this harness
    # was the only one that did not, so it sampled parts wherever their STL happened to
    # sit -- 0.85 m out for 25333MB000, which is in assembly coordinates. That put the
    # scenes generated here on a different footing from anything produced through the app.
    app.stages[Stage.IMPORT_MESH].center_mesh()

    app.stages[Stage.DOWNSAMPLE].use_adaptive = adaptive
    app.stages[Stage.DOWNSAMPLE].run_ambiguity = run_ambiguity

    t0 = time.time()
    app._express_sampling_worker()          # raycast -> downsample -> recenter
    t_sample = time.time() - t0

    # Direct, not via main_thread: see the module docstring. Without this the reference
    # bundle is silently skipped.
    app.set_stage(Stage.SAVE)
    app.stages[Stage.SAVE].worker()

    t0 = time.time()
    app.stages[Stage.DECOMPOSE]._run_worker()
    t_decomp = time.time() - t0

    scene = app.stages[Stage.SCENE]
    render = app.stages[Stage.RENDER]
    scene.rendering_flag = False            # the passive viewer blocks a batch run
    scene.arrangement = arrangement
    scene.generate_mode = "fill_rate"
    scene.fill_rate = fill_rate
    scene.structure_type = "none"

    # Required so save_synthetic_targets() -> SaveStage.worker takes the RENDER branch and
    # writes reference_cloud.ply (and the ambiguity sidecar) into each scene directory.
    app.set_stage(Stage.RENDER)

    made, instances = 0, 0
    for k in range(n_scenes):
        t0 = time.time()
        scene._run_worker()
        render._run_worker()
        n = len(app.synthetic_targets)
        render.save_synthetic_targets()
        instances += n
        made += 1
        print(f"    scene {k + 1}/{n_scenes}: {n} instances in {time.time() - t0:.0f}s",
              flush=True)

    return {"part": part, "scenes": made, "instances": instances,
            "n_hulls": len(app.convex_meshes),
            "t_sample": t_sample, "t_decomp": t_decomp,
            "t_total": time.time() - t_start}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", default=os.path.join(_ROOT, "mesh_raw", "tless"),
                    help="directory of meshes, or a single mesh file")
    ap.add_argument("--scenes", type=int, default=4, help="scenes per part")
    ap.add_argument("--fill", type=float, default=0.3,
                    help="bin fill rate; drives how many instances land per scene")
    ap.add_argument("--arrangement", choices=("random", "structured"), default="random")
    ap.add_argument("--adaptive", action="store_true",
                    help="curvature-adaptive downsampling instead of uniform")
    ap.add_argument("--no-ambiguity", action="store_true",
                    help="skip the ambiguity analysis (saves ~1-3 min/part, but the "
                         "heat-map weighting arm then has nothing to read)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N meshes")
    ap.add_argument("--only", default=None, help="substring filter on the mesh name")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even for parts that already have enough scenes")
    args = ap.parse_args()

    if os.path.isfile(args.meshes):
        meshes = [args.meshes]
    else:
        meshes = [os.path.join(args.meshes, f) for f in sorted(os.listdir(args.meshes))
                  if f.lower().endswith(MESH_EXT)]
    if args.only:
        meshes = [m for m in meshes if args.only in os.path.basename(m)]
    if args.limit:
        meshes = meshes[: args.limit]
    if not meshes:
        sys.exit(f"no meshes found under {args.meshes}")

    print(f"{len(meshes)} meshes, {args.scenes} scene(s) each, fill={args.fill:.0%}, "
          f"arrangement={args.arrangement}, ambiguity={not args.no_ambiguity}")

    done, skipped, failed = [], [], []
    t_all = time.time()
    for i, mesh in enumerate(meshes, 1):
        part = os.path.splitext(os.path.basename(mesh))[0]
        have = existing_scene_count(part)
        if have >= args.scenes and not args.force:
            print(f"[{i}/{len(meshes)}] {part}: already has {have} scenes - skipping")
            skipped.append(part)
            continue
        print(f"[{i}/{len(meshes)}] {part} ...", flush=True)
        try:
            r = generate_for_mesh(mesh, args.scenes - (0 if args.force else have),
                                  args.fill, args.arrangement,
                                  not args.no_ambiguity, args.adaptive)
            done.append(r)
            print(f"    OK {r['instances']} instances over {r['scenes']} scenes, "
                  f"{r['n_hulls']} hulls, {r['t_total']:.0f}s "
                  f"(sample {r['t_sample']:.0f}s, decomp {r['t_decomp']:.0f}s)", flush=True)
        except Exception as exc:                       # one bad mesh must not end the sweep
            failed.append((part, repr(exc)))
            print(f"    FAILED: {exc}", flush=True)
            traceback.print_exc()

    total_inst = sum(r["instances"] for r in done)
    print(f"\n{'=' * 70}")
    print(f"generated {total_inst} instances across {len(done)} parts "
          f"in {(time.time() - t_all) / 60:.1f} min "
          f"({len(skipped)} skipped, {len(failed)} failed)")
    if total_inst:
        # The number that actually decides whether the ablation can conclude anything:
        # at ~0.5 recall the 95% CI half-width is 1.96*sqrt(0.25/n).
        half = 1.96 * (0.25 / total_inst) ** 0.5
        print(f"95% CI half-width on a ~0.5 recall: +/- {half * 100:.1f} points "
              f"({'enough to resolve a 5-point arm difference' if half < 0.025 else 'NOT yet enough - a 5-point arm difference stays inside the noise'})")
    for part, err in failed:
        print(f"  FAILED {part}: {err}")


if __name__ == "__main__":
    main()
