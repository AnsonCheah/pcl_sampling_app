"""
Test random and structured bin scene generation.

Stops at mesh scene construction (no raycasting/rendering).
Run with --display to open the Open3D viewer after each test.

Usage:
  python physics/tests/test_bin_scene.py --mesh path/to/part.stl --arrangement both --display
  python physics/tests/test_bin_scene.py --arrangement structured
  python physics/tests/test_bin_scene.py --arrangement random --n_parts 15 --display
  python physics/tests/test_bin_scene.py --shape cuboid_long --arrangement random --n_parts 10
  python physics/tests/test_bin_scene.py --shape cuboid_flat --arrangement both --n_parts 8
  python physics/tests/test_bin_scene.py --shape cube --arrangement random --n_parts 15
"""

import sys
import os
import argparse
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import open3d as o3d
import trimesh
import mujoco
from trimesh.collision import CollisionManager
from rich import print as rp
from scipy.spatial.transform import Rotation as R

from geometry.geom_utils import o3d_to_trimesh, init_open3d
from physics.mujoco_bin_scene import MujocoBinScene, load_part, _display_scene


# -- Synthetic mesh generator --------------------------------------------------

def generate_test_mesh(shape: str, bin_dim=(0.76, 0.585, 0.25, 0.005)):
    """
    Return (part_mesh: trimesh.Trimesh, convex_meshes: list[o3d.TriangleMesh])
    for a synthetic box shape — no STL file required.

    Dimensions are expressed as multiples of bin_dim so shapes stay proportionate
    to whatever bin is configured.  Default bin_dim matches MujocoBinScene default.

    Shapes:
      cube        — small cube; max extent < bin_height → exercises spherical code path
      cuboid_long — longest dim > bin_height → exercises constrained tilt
      cuboid_flat — wide and flat; stable pose is almost always floor-facing
    """
    bw, bl, bh = bin_dim[0], bin_dim[1], bin_dim[2]
    shapes = {
        "cube":        [0.30 * bh,  0.30 * bh,  0.30 * bh],
        "cuboid_long": [0.50 * bw,  0.10 * bh,  0.10 * bh],
        "cuboid_flat": [0.35 * bw,  0.25 * bl,  0.08 * bh],
    }
    if shape not in shapes:
        raise ValueError(f"Unknown shape {shape!r}. Choose from: {list(shapes)}")

    tri_mesh = trimesh.creation.box(extents=shapes[shape])   # centred at origin
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tri_mesh.vertices),
        triangles=o3d.utility.Vector3iVector(tri_mesh.faces),
    )
    o3d_mesh.compute_vertex_normals()
    return tri_mesh, [o3d_mesh]



# -- Test helpers --------------------------------------------------------------

def _check_no_bin_intersection(scene, scene_state):
    """Assert that no settled part mesh intersects the bin geometry."""
    bin_trimesh = o3d_to_trimesh(scene.bin_mesh)
    cm = CollisionManager()
    cm.add_object("bin", bin_trimesh)
    colliding = []
    for body_name, data in scene_state.items():
        pos  = data["position"]
        quat = data["quaternion"]          # w x y z (MuJoCo convention)
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
        T[:3, 3]  = pos
        hit = cm.in_collision_single(scene.part_mesh, transform=T)
        if hit:
            colliding.append(body_name)
    assert not colliding, (
        f"{len(colliding)} part(s) intersect the bin: {colliding[:5]}"
        f"{'  ...' if len(colliding) > 5 else ''}"
    )
    rp(f"  No part-bin intersections (checked {len(scene_state)} parts)")


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
        assert scene.data.time == 0.0, (
            "Structured mode: data.time > 0 implies simulate() ran physics steps"
        )

        _check_no_bin_intersection(scene, scene_state)


# -- Individual test functions -------------------------------------------------

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
    try:
        _check_scene(scene, scene_state, "random")
        rp(f"[green]  PASS - random ({len(scene_state)} parts)[/green]")
    finally:
        if display:
            rp("  Opening viewer...")
            _display_scene(scene, scene_state)


def test_structured(part_mesh, convex_meshes, stable_pose_R: np.ndarray, pose_idx: int, prob: float, display: bool):
    rp(f"\n[bold cyan]=== STRUCTURED ARRANGEMENT (pose {pose_idx}, prob={prob:.3f}) ===[/bold cyan]")
    scene = MujocoBinScene(
        part_mesh, convex_meshes,
        n_parts=1,           # ignored - grid capacity overrides
        render=False,
        arrangement="structured",
        stable_pose_R=stable_pose_R,
    )
    scene.simulate()         # should be a no-op
    scene_state = scene.extract_scene_state()
    try:
        _check_scene(scene, scene_state, "structured")
        rp(f"[green]  PASS - structured pose {pose_idx} ({len(scene_state)} parts, physics skipped)[/green]")
    finally:
        if display:
            rp("  Opening viewer...")
            _display_scene(scene, scene_state)


def test_constrained_rotation(part_mesh, convex_meshes, n_parts: int, display: bool):
    rp("\n[bold cyan]=== CONSTRAINED ROTATION TEST ===[/bold cyan]")
    scene = MujocoBinScene(
        part_mesh, convex_meshes,
        n_parts=n_parts,
        render=False,
        arrangement="random",
    )
    # Random mode does not call mj_forward in generate_scene; populate xquat now
    # so we can check spawn orientations before physics runs.
    mujoco.mj_forward(scene.model, scene.data)

    z_limit = 0.9 * 2 * scene.hh * 1.05   # 5 % float tolerance
    verts   = np.asarray(part_mesh.vertices)
    violations = []
    for obj in scene.scene_objects:
        bid   = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
        R_mat = R.from_quat(scene.data.xquat[bid], scalar_first=True).as_matrix()
        sv    = (R_mat @ verts.T).T
        lz    = float(sv[:, 2].max() - sv[:, 2].min())
        if lz > z_limit:
            violations.append((obj.body_name, lz))

    assert not violations, (
        f"{len(violations)} part(s) exceed z-extent limit {z_limit:.4f} m: "
        + ", ".join(f"{name}={lz:.4f}" for name, lz in violations[:5])
    )
    rp(f"  All {len(scene.scene_objects)} spawned parts satisfy z-extent <= {z_limit:.4f} m")

    scene.simulate()
    result = scene.verify_parts_in_bin()
    rp(f"[green]  PASS - constrained rotation ({len(scene.scene_objects)} parts, "
       f"{result['n_in']} in bin)[/green]")

    if display:
        rp("  Opening viewer...")
        _display_scene(scene, scene.extract_scene_state())


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bin scene generation tests")
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Path to STL mesh file (skips file dialog)",
    )
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
    parser.add_argument(
        "--shape",
        choices=["stl", "cube", "cuboid_long", "cuboid_flat"],
        default="stl",
        help="Mesh source: 'stl' requires --mesh; others generate synthetic box shapes",
    )
    args = parser.parse_args()

    init_open3d()
    if args.shape == "stl":
        part_mesh, convex_meshes = load_part(mesh_path=args.mesh)
    else:
        rp(f"[bold]Generating synthetic mesh: {args.shape}[/bold]")
        part_mesh, convex_meshes = generate_test_mesh(args.shape)

    # Enumerate all stable poses once; structured tests iterate over them.
    stable_poses = MujocoBinScene.get_stable_poses(part_mesh)
    rp(f"\nFound {len(stable_poses)} stable pose(s) for this part:")
    for i, (_, p) in enumerate(stable_poses):
        rp(f"  pose {i}: probability = {p:.4f}")

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
        run("constrained_rotation", test_constrained_rotation,
            part_mesh, convex_meshes, args.n_parts, args.display)

    if args.arrangement in ("structured", "both"):
        for i, (R_stable, prob) in enumerate(stable_poses):
            run(f"structured_pose_{i}", test_structured,
                part_mesh, convex_meshes, R_stable, i, prob, args.display)

    rp(f"\n{'-'*40}")
    rp(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        rp(f"[red]Failed: {failed}[/red]")
        sys.exit(1)
    else:
        rp("[green bold]All tests passed.[/green bold]")
