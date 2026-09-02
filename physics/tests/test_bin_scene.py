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

# Primitive shapes spanning aspect ratio (AR = longest/shortest extent) and absolute size.
# Used by the big-part drop regression to reproduce the high-velocity penetration that big /
# elongated parts caused with the old layer-stacking spawn.
PRIMITIVE_SHAPES = ["cube", "cube_big", "cuboid_long", "rod_long", "cuboid_flat", "plate_big"]


def generate_test_mesh(shape: str, bin_dim=(0.76, 0.585, 0.25, 0.005)):
    """
    Return (part_mesh: trimesh.Trimesh, convex_meshes: list[o3d.TriangleMesh])
    for a synthetic box shape -- no STL file required.

    Dimensions are expressed as multiples of bin_dim so shapes stay proportionate
    to whatever bin is configured.  Default bin_dim matches MujocoBinScene default.

    Shapes (see PRIMITIVE_SHAPES):
      cube        -- small cube; max extent < bin_height -> exercises spherical code path
      cube_big    -- large cube near bin height; blocky (AR ~1) but big
      cuboid_long -- longest dim > bin_height -> exercises constrained tilt
      rod_long    -- long thin rod, high AR (worst case for packing/placement)
      cuboid_flat -- wide and flat; stable pose is almost always floor-facing
      plate_big   -- large flat plate, high AR
    """
    bw, bl, bh = bin_dim[0], bin_dim[1], bin_dim[2]
    shapes = {
        "cube":        [0.30 * bh,  0.30 * bh,  0.30 * bh],
        "cube_big":    [0.80 * bh,  0.80 * bh,  0.80 * bh],
        "cuboid_long": [0.50 * bw,  0.10 * bh,  0.10 * bh],
        "rod_long":    [0.50 * bw,  0.06 * bh,  0.06 * bh],
        "cuboid_flat": [0.35 * bw,  0.25 * bl,  0.08 * bh],
        "plate_big":   [0.55 * bw,  0.45 * bl,  0.04 * bh],
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


def realistic_part_count(part_mesh, requested: int, bin_dim=(0.76, 0.585, 0.25, 0.005),
                         fill: float = 0.6) -> int:
    """
    Cap a requested part count to a physically realistic fill for the part/bin, mirroring the
    shape-aware packing heuristic in SceneStage._auto_part_count. This keeps the big-part
    drop test from pathologically over-filling the bin (e.g. 10 large cubes -> unstable tower
    that spills), while leaving room for many thin parts. Returns min(requested, auto_count).
    """
    bw, bl, bh, _ = bin_dim
    bin_vol = bw * bl * bh * 0.8                                   # 20% top headroom
    obb_vol = max(float(part_mesh.bounding_box_oriented.volume), 1e-9)
    ext     = np.sort(part_mesh.bounding_box_oriented.extents)[::-1]
    ar      = float(ext[0] / max(ext[2], 1e-9))
    packing = max(0.18, 0.62 / np.sqrt(ar))
    auto    = max(2, int(round(fill * packing * bin_vol / obb_vol)))
    return int(min(requested, auto))



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


def _pose_to_T(data):
    """MuJoCo scene_state entry (position + wxyz quaternion) -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(data["quaternion"], scalar_first=True).as_matrix()
    T[:3, 3]  = data["position"]
    return T


def _check_no_penetration(scene, scene_state, max_depth: float = 0.003):
    """
    Assert no settled part interpenetrates another part OR the bin beyond max_depth.

    Uses fcl contact *depth* (via trimesh CollisionManager), not a boolean collision test:
    a part merely resting on the floor / on another part reports ~0 mm penetration, whereas a
    tunneled part reports a large depth. The bin is added as a collision object so deep
    floor/wall sinking is caught by the same metric. The 3 mm default tolerates the sub-mm
    resting overlap the constraint solver allows while still catching gross interpenetration --
    the core regression for high-velocity drops of big parts.
    """
    cm = CollisionManager()
    cm.add_object("bin", o3d_to_trimesh(scene.bin_mesh))
    for body_name, data in scene_state.items():
        cm.add_object(body_name, scene.part_mesh, transform=_pose_to_T(data))

    hit, _names, data = cm.in_collision_internal(return_names=True, return_data=True)
    if not hit:
        rp("  No part-part / part-bin contacts at all")
        return

    worst = max(data, key=lambda cd: cd.depth)
    deep  = [cd for cd in data if cd.depth > max_depth]
    assert not deep, (
        f"{len(deep)} penetration(s) > {max_depth * 1000:.1f} mm "
        f"(worst {worst.depth * 1000:.2f} mm between {tuple(worst.names)}): "
        + ", ".join(f"{tuple(cd.names)}={cd.depth * 1000:.2f}mm" for cd in deep[:5])
    )
    rp(f"  Penetration OK (worst {worst.depth * 1000:.2f} mm <= {max_depth * 1000:.1f} mm, "
       f"{len(data)} contacts)")


def _check_bounded_height(scene, margin_layers: int = 3):
    """
    Assert no settled part centre rose far above the bin -- guards the old 'spawned meters
    high' creep regression. Allowance = bin height + margin_layers x worst-case tilted part
    height (so a legitimately tall pile still passes, but a runaway does not).
    """
    body_ids = scene._body_ids if scene._body_ids else list(range(1, scene.model.nbody))
    z_max    = float(scene.data.xpos[body_ids, 2].max())
    h_layer  = scene._h_layer if scene._h_layer > 0 else 0.05
    z_allow  = 2 * scene.hh + margin_layers * h_layer
    assert z_max <= z_allow, (
        f"settled part too high: max centre z={z_max:.3f} m > allowed {z_allow:.3f} m "
        f"(bin height {2 * scene.hh:.3f} + {margin_layers}xh_layer {h_layer:.3f})"
    )
    rp(f"  Bounded height OK (max part z={z_max:.3f} m <= {z_allow:.3f} m)")


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


def test_big_part_drop(part_mesh, convex_meshes, n_parts: int, display: bool):
    """
    Batched random drop, then assert the settled pile is physically valid: nothing escaped or
    tunneled out, no deep part-part / part-bin penetration, and no runaway spawn height. This
    is the regression for big / elongated parts under the high-velocity penetration fix.
    """
    rp("\n[bold cyan]=== BIG-PART DROP (random, batched) ===[/bold cyan]")
    n = realistic_part_count(part_mesh, n_parts)
    if n < n_parts:
        rp(f"  capping requested n_parts {n_parts} -> {n} (realistic fill for this shape)")
    scene = MujocoBinScene(
        part_mesh, convex_meshes,
        n_parts=n,
        render=False,
        arrangement="random",
    )
    scene.simulate()
    scene_state = scene.extract_scene_state()
    try:
        bin_result = scene.verify_parts_in_bin()
        assert len(scene_state) == scene.n_parts, (
            f"n_parts mismatch: scene.n_parts={scene.n_parts}, scene_state has {len(scene_state)}"
        )

        # Floor tunneling (a part dropped through the bin floor) is the real penetration bug and
        # must never happen; spilling over a wall from a full pile is benign overfill, tolerated.
        z_floor_escape = scene.bin_transform[2, 3] - 2 * scene.hh
        tunneled = [nm for nm, d in scene_state.items() if d["position"][2] < z_floor_escape]
        assert not tunneled, f"{len(tunneled)} part(s) tunneled through the floor: {tunneled[:5]}"
        spill_tol = max(1, int(0.15 * scene.n_parts))
        assert bin_result["n_out"] <= spill_tol, (
            f"{bin_result['n_out']}/{scene.n_parts} parts spilled out (tol {spill_tol}): "
            f"{bin_result['out_of_bin'][:5]}"
        )

        _check_no_penetration(scene, scene_state)
        _check_bounded_height(scene)
        rp(f"[green]  PASS - big_part_drop ({len(scene_state)} parts)[/green]")
    finally:
        if display:
            rp("  Opening viewer...")
            _display_scene(scene, scene_state)


# -- Per-step vectorization tests ----------------------------------------------

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden")


def test_index_arrays_match_adr(part_mesh, convex_meshes):
    """The precomputed qpos/qvel index matrices must address the same DOFs as the
    per-body (qpos_adr, qvel_adr) pairs they replace."""
    rp("\n[bold cyan]=== INDEX ARRAYS MATCH _adr_of_body ===[/bold cyan]")
    np.random.seed(0)
    sc = MujocoBinScene(part_mesh, convex_meshes, n_parts=6, render=False, arrangement="random")
    for i, (q, v) in enumerate(sc._adr_of_body):
        assert np.array_equal(sc._qpos_idx7[i],  q + np.arange(7)), f"qpos idx row {i}"
        assert np.array_equal(sc._qvel_idx6[i],  v + np.arange(6)), f"qvel idx row {i}"
        assert np.array_equal(sc._lin_vel_idx[i], v + np.arange(3)), f"lin idx row {i}"
    rp(f"  index matrices match _adr_of_body for all {len(sc._adr_of_body)} parts")


def test_clamp_velocities_spec(part_mesh, convex_meshes):
    """Vectorized clamp: linear speed capped at vel_cap, sub-cap linear velocities
    unchanged, angular velocity untouched."""
    rp("\n[bold cyan]=== CLAMP VELOCITIES SPEC ===[/bold cyan]")
    np.random.seed(0)
    sc = MujocoBinScene(part_mesh, convex_meshes, n_parts=5, render=False, arrangement="random")
    cap = sc.vel_cap
    qvel = sc.data.qvel
    qvel[:] = 0.0
    li = sc._lin_vel_idx
    ai = sc._qvel_idx6[:, 3:]                       # angular DOFs per part
    qvel[li[0]] = np.array([3.0 * cap, 0.0, 0.0])   # over cap
    qvel[li[1]] = np.array([0.0, 0.5 * cap, 0.0])   # under cap
    qvel[ai[0]] = np.array([7.0, -2.0, 1.0])        # nonzero angular
    under_before = qvel[li[1]].copy()
    ang_before   = qvel[ai[0]].copy()

    sc._clamp_velocities()

    sp0 = float(np.linalg.norm(qvel[li[0]]))
    assert abs(sp0 - cap) < 1e-9, f"over-cap not clamped to {cap}: got {sp0}"
    assert np.allclose(qvel[li[1]], under_before), "sub-cap linear velocity changed"
    assert np.allclose(qvel[ai[0]], ang_before),   "angular velocity was modified"
    rp(f"  over-cap clamped to {cap:.3f}, sub-cap and angular untouched")


def test_freeze_parked_machinery(part_mesh, convex_meshes):
    """Vectorized parked-body freeze restores the snapshot qpos and zeros qvel for
    parked parts only, leaving active parts untouched."""
    rp("\n[bold cyan]=== FREEZE PARKED MACHINERY ===[/bold cyan]")
    np.random.seed(0)
    sc = MujocoBinScene(part_mesh, convex_meshes, n_parts=15, render=False, arrangement="random")
    parked = [i for i, b in enumerate(sc._batch_of_body) if b > 0]
    assert len(parked) > 0, "fixture must produce >=2 batches so some bodies are parked"
    active = np.array([i for i, b in enumerate(sc._batch_of_body) if b == 0], dtype=np.intp)
    pid = np.array(parked, dtype=np.intp)

    snapshot = sc.data.qpos[sc._qpos_idx7].copy()
    active_qpos_before = sc.data.qpos[sc._qpos_idx7[active]].copy()
    # Perturb parked bodies (simulate gravity pulling them while contype=0).
    sc.data.qpos[sc._qpos_idx7[pid].ravel()] += 1.234
    sc.data.qvel[sc._qvel_idx6[pid].ravel()] = 9.9

    # Vectorized freeze (mirrors simulate()'s freeze_parked closure).
    sc.data.qpos[sc._qpos_idx7[pid].ravel()] = snapshot[pid].ravel()
    sc.data.qvel[sc._qvel_idx6[pid].ravel()] = 0.0

    assert np.allclose(sc.data.qpos[sc._qpos_idx7[pid]], snapshot[pid]), "parked qpos not restored"
    assert np.all(sc.data.qvel[sc._qvel_idx6[pid]] == 0.0), "parked qvel not zeroed"
    assert np.allclose(sc.data.qpos[sc._qpos_idx7[active]], active_qpos_before), "active qpos changed"
    rp(f"  {len(parked)} parked restored & zeroed, {len(active)} active untouched")


def test_vectorization_parity():
    """End-to-end: the vectorized per-step loops must reproduce the pre-refactor
    settled poses bitwise (same np seed, baseline solver settings). Guards against any
    behavioural drift from replacing the scalar clamp/freeze loops."""
    rp("\n[bold cyan]=== VECTORIZATION PARITY (vs golden) ===[/bold cyan]")
    gold_path = os.path.join(_GOLDEN, "settle_cube.npz")
    np.random.seed(0)
    pm, cv = generate_test_mesh("cube")
    sc = MujocoBinScene(pm, cv, n_parts=8, render=False, arrangement="random")
    sc.simulate()
    st = sc.extract_scene_state()
    names = list(st.keys())
    xpos = np.array([st[n]["position"] for n in names])
    xquat = np.array([st[n]["quaternion"] for n in names])

    # Golden .npz is gitignored; on a fresh checkout regenerate from the current run so
    # the parity check becomes a forward regression lock on the current settled poses.
    if not os.path.exists(gold_path):
        os.makedirs(_GOLDEN, exist_ok=True)
        np.savez_compressed(gold_path, names=np.array(names), xpos=xpos, xquat=xquat,
                            n_parts=np.array([sc.n_parts]))
        rp(f"  [yellow]golden missing; baselined current poses to {gold_path}[/yellow]")
    g = np.load(gold_path)
    assert list(g["names"]) == names, "body order changed vs golden"
    dpos = float(np.abs(xpos - g["xpos"]).max())
    dquat = float(np.abs(xquat - g["xquat"]).max())
    assert dpos < 1e-9,  f"settled xpos drifted from golden: max abs diff={dpos:.2e}"
    assert dquat < 1e-9, f"settled xquat drifted from golden: max abs diff={dquat:.2e}"
    rp(f"  bitwise match vs golden (max|dpos|={dpos:.1e}, max|dquat|={dquat:.1e})")


def test_simulate_speedup():
    """Informational: report simulate() wall-clock at a higher part count. Never fails."""
    rp("\n[bold cyan]=== SIMULATE TIMING (informational) ===[/bold cyan]")
    np.random.seed(0)
    pm, cv = generate_test_mesh("cube")
    n = realistic_part_count(pm, 60)
    sc = MujocoBinScene(pm, cv, n_parts=n, render=False, arrangement="random")
    import time as _t
    t0 = _t.time(); sc.simulate(); dt = _t.time() - t0
    rp(f"  simulate() n_parts={sc.n_parts}: {dt:.2f}s")


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
        choices=["stl", "all"] + PRIMITIVE_SHAPES,
        default="stl",
        help="Mesh source: 'stl' requires --mesh; 'all' loops every primitive; "
             "or pick one primitive box shape",
    )
    args = parser.parse_args()

    init_open3d()

    # Resolve the list of (name, part_mesh, convex_meshes) parts to exercise.
    if args.shape == "stl":
        parts = [("stl", *load_part(mesh_path=args.mesh))]
    elif args.shape == "all":
        parts = [(s, *generate_test_mesh(s)) for s in PRIMITIVE_SHAPES]
        rp(f"[bold]Looping every primitive shape: {PRIMITIVE_SHAPES}[/bold]")
    else:
        rp(f"[bold]Generating synthetic mesh: {args.shape}[/bold]")
        parts = [(args.shape, *generate_test_mesh(args.shape))]

    passed = []
    failed = []

    def run(name, fn, *a, **kw):
        try:
            fn(*a, **kw)
            passed.append(name)
        except Exception as e:
            print(f"  FAIL - {name}: {e}")
            failed.append(name)

    # Per-step vectorization unit tests (shape-independent; use a cube primitive).
    _u_mesh, _u_convex = generate_test_mesh("cube")
    run("index_arrays_match_adr",  test_index_arrays_match_adr,  _u_mesh, _u_convex)
    run("clamp_velocities_spec",   test_clamp_velocities_spec,   _u_mesh, _u_convex)
    run("freeze_parked_machinery", test_freeze_parked_machinery, _u_mesh, _u_convex)
    run("vectorization_parity",    test_vectorization_parity)
    run("simulate_speedup",        test_simulate_speedup)

    for name, part_mesh, convex_meshes in parts:
        rp(f"\n[bold magenta]######## PART: {name} ########[/bold magenta]")
        if args.arrangement in ("random", "both"):
            # Synthetic primitives use the big-part-drop regression (penetration + bounded
            # height); an STL part keeps the original random + constrained-rotation checks.
            if name == "stl":
                run(f"random[{name}]", test_random,
                    part_mesh, convex_meshes, args.n_parts, args.display)
                run(f"constrained_rotation[{name}]", test_constrained_rotation,
                    part_mesh, convex_meshes, args.n_parts, args.display)
            else:
                run(f"big_part_drop[{name}]", test_big_part_drop,
                    part_mesh, convex_meshes, args.n_parts, args.display)

        if args.arrangement in ("structured", "both"):
            stable_poses = MujocoBinScene.get_stable_poses(part_mesh)
            rp(f"Found {len(stable_poses)} stable pose(s) for {name}")
            for i, (R_stable, prob) in enumerate(stable_poses):
                run(f"structured_pose_{i}[{name}]", test_structured,
                    part_mesh, convex_meshes, R_stable, i, prob, args.display)

    rp(f"\n{'-'*40}")
    rp(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        rp(f"[red]Failed: {failed}[/red]")
        sys.exit(1)
    else:
        rp("[green bold]All tests passed.[/green bold]")
