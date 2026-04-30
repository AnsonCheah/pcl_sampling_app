"""
Test random and structured bin scene generation.

Stops at mesh scene construction (no raycasting/rendering).
Run with --display to open the Open3D viewer after each test.

Usage:
  python physics/tests/test_bin_scene.py --arrangement both --display
  python physics/tests/test_bin_scene.py --arrangement structured
  python physics/tests/test_bin_scene.py --arrangement random --n_parts 15 --display
"""

import sys
import os
import argparse
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from rich import print as rp

from geometry.geom_utils import o3d_to_trimesh, trimesh_to_o3d, o3d_display, init_open3d
from physics.mujoco_bin_scene import MujocoBinScene
from app_v2 import MeshSamplingApp
from enums import Stage


# -- Mesh loading --------------------------------------------------------------

def load_part(verbose: bool = True):
    if verbose:
        rp("[bold]Loading part mesh via MeshSamplingApp...[/bold]")
    app = MeshSamplingApp(headless=True)
    app.stages[Stage.IMPORT_MESH]._run_worker()
    app._express_sampling_worker()
    part_mesh = o3d_to_trimesh(app.target_mesh)
    if verbose:
        rp(f"  vertices: {len(part_mesh.vertices):,}  faces: {len(part_mesh.faces):,}")
        rp(f"  bounding sphere r = {part_mesh.bounding_sphere.primitive.radius:.4f} m")
    return part_mesh, app.convex_meshes


# -- Test helpers --------------------------------------------------------------

def _display_scene(scene, scene_state):
    import open3d as o3d
    o3d_scene = scene.mujoco_scene_to_o3d(scene_state)
    geom_list = []
    for obj in o3d_scene.values():
        mesh = copy.deepcopy(obj.geom)
        geom_list.append(mesh.transform(obj.T_gt))
    vis = o3d_display(geom_list, dynamic_color=True)
    vis.run()
    vis.destroy_window()


def _check_scene(scene, scene_state, arrangement):
    n_parts = len(scene_state)
    bin_result = scene.verify_parts_in_bin()
    rp(f"  Parts in scene_state : {n_parts}")
    rp(f"  scene.n_parts        : {scene.n_parts}")
    rp(f"  In-bin / escaped     : {bin_result['n_in']} / {bin_result['n_out']}")

    assert n_parts > 0, "scene_state is empty"
    assert n_parts == scene.n_parts, (
        f"n_parts mismatch: scene.n_parts={scene.n_parts} but scene_state has {n_parts} entries"
    )

    if arrangement == "structured":
        assert bin_result["n_out"] == 0, (
            f"Structured mode: {bin_result['n_out']} parts escaped the bin"
        )
        # import mujoco
        assert scene.data.time == 0.0, (
            "Structured mode: data.time > 0 implies simulate() ran physics steps"
        )


# -- Individual test functions ------------------------------------------------─

def test_random(part_mesh, convex_meshes, n_parts: int, display: bool):
    rp("\n[bold cyan]=== RANDOM ARRANGEMENT ===[/bold cyan]")
    scene = MujocoBinScene(
        part_mesh, convex_meshes,
        n_parts=n_parts,
        render=False,
        arrangement="random",
    )
    scene.simulate()
    scene_state = scene.extract_scene_state()
    _check_scene(scene, scene_state, "random")
    rp(f"[green]  PASS -random ({len(scene_state)} parts)[/green]")

    if display:
        rp("  Opening viewer...")
        _display_scene(scene, scene_state)


def test_structured(part_mesh, convex_meshes, display: bool):
    rp("\n[bold cyan]=== STRUCTURED ARRANGEMENT ===[/bold cyan]")
    scene = MujocoBinScene(
        part_mesh, convex_meshes,
        n_parts=1,          # ignored -grid capacity overrides
        render=False,
        arrangement="structured",
    )
    scene.simulate()        # should be a no-op
    scene_state = scene.extract_scene_state()
    _check_scene(scene, scene_state, "structured")
    rp(f"[green]  PASS -structured ({len(scene_state)} parts, physics skipped)[/green]")

    if display:
        rp("  Opening viewer...")
        _display_scene(scene, scene_state)


# -- Entry point --------------------------------------------------------------─

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bin scene generation tests")
    parser.add_argument(
        "--arrangement",
        choices=["random", "structured", "both"],
        default="both",
        help="Which arrangement to test (default: both)",
    )
    parser.add_argument(
        "--n_parts",
        type=int,
        default=10,
        help="Number of parts for random mode (structured ignores this)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open Open3D viewer after each test",
    )
    args = parser.parse_args()

    init_open3d()
    part_mesh, convex_meshes = load_part()

    passed = []
    failed = []

    def run(name, fn, *a, **kw):
        try:
            fn(*a, **kw)
            passed.append(name)
        except Exception as e:
            print(f"  FAIL - {name}: {e}")
            failed.append(name)

    if args.arrangement in ("random", "both"):
        run("random", test_random, part_mesh, convex_meshes, args.n_parts, args.display)

    if args.arrangement in ("structured", "both"):
        run("structured", test_structured, part_mesh, convex_meshes, args.display)

    rp(f"\n{'-'*40}")
    rp(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        rp(f"[red]Failed: {failed}[/red]")
        sys.exit(1)
    else:
        rp("[green bold]All tests passed.[/green bold]")
