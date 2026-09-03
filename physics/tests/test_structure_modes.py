"""Structured-scene structure modes: bare grid (none), partition dividers, and molded tray.

Verifies that partition/tray add fixture geometry to the bin body, widen the grid pitch, settle the
parts under gravity without any escaping, and seat tray parts in their pockets.

Run:  python -m pytest physics/tests/test_structure_modes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mujoco
import numpy as np
import open3d as o3d
import pytest
import trimesh
from scipy.spatial.transform import Rotation as R

from geometry.geom_utils import rotation_aligning_vector_to_axis
from physics.mujoco_bin_scene import MujocoBinScene, TRAY_BASE, PARTITION_THICKNESS
from physics.tests.test_bin_scene import generate_test_mesh


def _bin_geom_count(scene) -> int:
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    return int(scene.model.body_geomnum[bid])


def _make(structure, shape="cube_big"):
    part, convex = generate_test_mesh(shape)
    scene = MujocoBinScene(part, convex, n_parts=1, render=False, arrangement="structured",
                           structure_type=structure, structure_height_frac=0.7,
                           clearance_mode="medium", settle_time=2.0)
    scene.simulate()
    return scene


@pytest.fixture(scope="module")
def scenes():
    return {s: _make(s) for s in ("none", "partition", "tray")}


def test_grid_populated_all_modes(scenes):
    for s, sc in scenes.items():
        assert sc.n_parts > 0, f"{s}: empty grid"


def test_partition_and_tray_add_bin_geoms(scenes):
    base = _bin_geom_count(scenes["none"])
    assert _bin_geom_count(scenes["partition"]) > base, "partition added no divider geoms"
    assert _bin_geom_count(scenes["tray"]) > base, "tray added no pocket geoms"


def test_fixtures_merged_into_bin_mesh(scenes):
    base = len(scenes["none"].bin_mesh.vertices)
    assert len(scenes["partition"].bin_mesh.vertices) > base
    assert len(scenes["tray"].bin_mesh.vertices) > base


def test_structure_widens_pitch(scenes):
    # Wider pitch (fixture clearance) => no more slots than the bare grid.
    assert scenes["partition"].n_parts <= scenes["none"].n_parts
    assert scenes["tray"].n_parts <= scenes["none"].n_parts


def test_no_parts_escape_after_settle(scenes):
    for s in ("partition", "tray"):
        chk = scenes[s].verify_parts_in_bin()
        assert chk["n_out"] == 0, f"{s}: {chk['n_out']} parts escaped the bin"


def test_tray_parts_rest_above_pocket_floor(scenes):
    # Every part centre must sit at/above the pocket floor (z = TRAY_BASE) -- i.e. inside the pocket,
    # not sunk through the tray base.
    state = scenes["tray"].extract_scene_state()
    zs = np.array([bd["position"][2] for bd in state.values()])
    assert (zs >= TRAY_BASE - 1e-3).all(), f"tray parts below pocket floor: min z={zs.min():.4f}"


def test_export_records_structure_metadata(scenes):
    st = scenes["tray"].export_scene_state()
    assert str(st["structure_type"]) == "tray"
    assert 0.0 < float(st["structure_height_frac"]) <= 1.0
    assert float(st["clearance_m"]) > 0.0


def test_none_is_static_no_freejoint(scenes):
    # Structured/none stays static (no freejoint) -- qpos has only the bin's 0 DOFs + no part joints.
    assert scenes["none"].model.njnt == 0


def test_custom_face_up_pose_drives_structured_grid():
    """The GUI face-picker turns a clicked face normal into scene.stable_pose_R. Feeding a
    forced pose (built exactly the same way) must override the auto stable pose: the aligned
    grid orientation keeps the chosen normal pointing along world +Z (packing may add a yaw
    about Z, which preserves the up axis)."""
    part, convex = generate_test_mesh("cube_big")
    n_up = np.array([0.0, 1.0, 0.0])                       # a non-stable "this face up" pick
    R_user = rotation_aligning_vector_to_axis(n_up, (0.0, 0.0, 1.0))

    scene = MujocoBinScene(part, convex, n_parts=1, render=False, arrangement="structured",
                           structure_type="tray", structure_height_frac=0.7,
                           clearance_mode="medium", stable_pose_R=R_user, settle_time=1.0)

    # The forced pose is honored (not replaced by the auto most-probable stable pose).
    assert scene.stable_pose_R is R_user
    assert scene.n_parts > 0

    xs, ys, R_aligned, raw_pos_z, pitch_x, pitch_y, part_h = scene._compute_structured_grid(R_user)
    assert np.isfinite(xs).all() and np.isfinite(ys).all()
    assert np.isfinite(R_aligned).all()
    assert part_h > 0 and pitch_x > 0 and pitch_y > 0
    up = R_aligned @ n_up
    assert abs(up[2]) > 0.99, f"chosen up axis not preserved: R_aligned @ n_up = {up}"


def _lshape_with_hull_collision():
    """Concave L part whose collision is its single convex hull -- the worst-case bulge. The hull fills
    the L's notch, so a pocket traced from the MESH silhouette would overlap it and eject the part; a
    pocket traced from the convex-collision footprint must not."""
    base = trimesh.creation.box(extents=[0.14, 0.14, 0.03])
    notch = trimesh.creation.box(extents=[0.07, 0.07, 0.06])
    notch.apply_translation([0.035, 0.035, 0.0])
    part = trimesh.boolean.difference([base, notch], engine="manifold")
    hull = part.convex_hull
    hull_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hull.vertices),
                                         o3d.utility.Vector3iVector(hull.faces))
    hull_o3d.compute_vertex_normals()
    return part, [hull_o3d]


def _asymmetric_wedge():
    """Right-triangular prism centred at its centroid, so its footprint AABB centre does NOT
    coincide with the body origin (nonzero cx/cy) -- the case the divider-centering fix targets.
    A prism is convex, so its collision hull equals the mesh."""
    from shapely.geometry import Polygon
    wedge = trimesh.creation.extrude_polygon(Polygon([(0, 0), (0.10, 0), (0, 0.05)]), height=0.05)
    wedge.apply_translation(-wedge.centroid)
    hull = wedge.convex_hull
    hull_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hull.vertices),
                                         o3d.utility.Vector3iVector(hull.faces))
    hull_o3d.compute_vertex_normals()
    return wedge, [hull_o3d]


def _divider_positions(scene):
    """Read x/y divider plane positions from the compiled model (thin boxes on the bin body)."""
    half = PARTITION_THICKNESS / 2.0
    bin_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    act_x, act_y = [], []
    for gi in range(scene.model.ngeom):
        if scene.model.geom_bodyid[gi] != bin_bid or scene.model.geom_type[gi] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        gs, gp = scene.model.geom_size[gi], scene.model.geom_pos[gi]
        if abs(gs[0] - half) < 1e-6:      # thin in x -> an x-divider
            act_x.append(float(gp[0]))
        elif abs(gs[1] - half) < 1e-6:    # thin in y -> a y-divider
            act_y.append(float(gp[1]))
    return sorted(act_x), sorted(act_y)


def _expected_boundaries(centres, pitch, inner_h, half):
    lines = [centres[0] - pitch / 2.0,
             *[(centres[i] + centres[i + 1]) / 2.0 for i in range(len(centres) - 1)],
             centres[-1] + pitch / 2.0]
    return sorted(l for l in lines if abs(l) + half <= inner_h + 1e-9)


def test_partition_dividers_centered_on_asymmetric_footprint():
    """Regression: partition dividers must sit at every cell boundary midway between adjacent part
    FOOTPRINTS (incl. the outer perimeter ring), not between body origins. For an asymmetric part
    the footprint centre is offset from the body origin (cx/cy != 0); placing dividers at
    body-origin midpoints put them off-centre and clipped the part, ejecting non-stable parts."""
    part, convex = _asymmetric_wedge()
    bin_dim = (0.40, 0.40, 0.20, 0.005)
    scene = MujocoBinScene(part, convex, n_parts=1, render=False, arrangement="structured",
                           structure_type="partition", structure_height_frac=0.7,
                           clearance_mode="snug", bin_dim=bin_dim, settle_time=0.1)
    xs, ys, R_aligned, _rpz, pitch_x, pitch_y, _ph = scene._compute_structured_grid(scene._find_stable_pose())
    cx, cy = scene._grid_fp_offset
    assert abs(cx) > 1e-3 or abs(cy) > 1e-3, "wedge not asymmetric enough to exercise the offset"

    half = PARTITION_THICKNESS / 2.0
    inner_hx, inner_hy = scene.hx - bin_dim[3], scene.hy - bin_dim[3]
    exp_x = _expected_boundaries(np.asarray(xs) + cx, pitch_x, inner_hx, half)
    exp_y = _expected_boundaries(np.asarray(ys) + cy, pitch_y, inner_hy, half)

    act_x, act_y = _divider_positions(scene)
    # Full egg-crate incl. perimeter => nx+1 / ny+1 planes (minus any clamped past the wall).
    assert np.allclose(act_x, exp_x, atol=1e-6), f"x-dividers: {act_x} vs {exp_x}"
    assert np.allclose(act_y, exp_y, atol=1e-6), f"y-dividers: {act_y} vs {exp_y}"


@pytest.mark.parametrize("mode,expected_mm", [("snug", 1.0), ("medium", 2.5)])
def test_partition_clearance_matches_setting(mode, expected_mm):
    """The running part-to-divider gap must equal the clearance setting (not the old ~5%-of-footprint
    looseness). Pitch is convex-footprint + PARTITION_THICKNESS + 2*clearance, so
    (pitch - convex_footprint - PARTITION_THICKNESS)/2 == clearance on each axis."""
    part, convex = generate_test_mesh("cube_big")
    scene = MujocoBinScene(part, convex, n_parts=1, render=False, arrangement="structured",
                           structure_type="partition", structure_height_frac=0.7,
                           clearance_mode=mode, settle_time=0.1)
    _xs, _ys, R_aligned, _rpz, pitch_x, pitch_y, _ph = scene._compute_structured_grid(scene._find_stable_pose())
    cvx = np.vstack([np.asarray(m.vertices) for m in convex])
    sv = (R_aligned @ cvx.T).T
    fp_x, fp_y = float(np.ptp(sv[:, 0])), float(np.ptp(sv[:, 1]))
    gap_x = (pitch_x - fp_x - PARTITION_THICKNESS) / 2.0
    gap_y = (pitch_y - fp_y - PARTITION_THICKNESS) / 2.0
    assert gap_x == pytest.approx(expected_mm / 1000.0, abs=1e-5)
    assert gap_y == pytest.approx(expected_mm / 1000.0, abs=1e-5)


def test_partition_no_spawn_penetration_true_snug():
    """True-snug cells sized to the convex footprint + a bounded lean must not penetrate the dividers
    or floor at spawn (asymmetric part, full-height dividers, snug)."""
    part, convex = _asymmetric_wedge()
    scene = MujocoBinScene(part, convex, n_parts=1, render=False, arrangement="structured",
                           structure_type="partition", structure_height_frac=1.0,
                           clearance_mode="snug", bin_dim=(0.40, 0.40, 0.20, 0.005), settle_time=0.1)
    mujoco.mj_forward(scene.model, scene.data)
    bb = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    worst = 0.0
    for i in range(scene.data.ncon):
        c = scene.data.contact[i]
        if (scene.model.geom_bodyid[c.geom1] == bb) ^ (scene.model.geom_bodyid[c.geom2] == bb):
            worst = min(worst, float(c.dist))
    assert worst > -1e-3, f"spawn penetration {worst * 1000:.2f} mm"


def test_partition_tilt_bounded_and_scales_with_clearance():
    """The spawn lean must never exceed theta_max(clearance), and loose must allow more lean than snug."""
    part, convex = generate_test_mesh("cube_big")
    tmax = {}
    for mode in ("snug", "loose"):
        scene = MujocoBinScene(part, convex, n_parts=1, render=False, arrangement="structured",
                               structure_type="partition", structure_height_frac=0.7,
                               clearance_mode=mode, settle_time=0.1)
        _xs, _ys, R_aligned, *_ = scene._compute_structured_grid(scene._find_stable_pose())
        theta_max = scene._structured_theta_max
        mujoco.mj_forward(scene.model, scene.data)
        for bd in scene.extract_scene_state().values():
            R_local = R.from_quat(bd["quaternion"], scalar_first=True).as_matrix()
            ang = (R.from_matrix(R_aligned).inv() * R.from_matrix(R_local)).magnitude()
            assert ang <= theta_max + 1e-3, f"{mode}: lean {np.degrees(ang):.2f} > theta_max {np.degrees(theta_max):.2f}"
        tmax[mode] = theta_max
    assert tmax["loose"] > tmax["snug"]


def test_concave_part_stays_seated_in_tray():
    # Regression for the pocket-ejection bug: the pocket collision must follow the part's convex
    # collision, not its concave mesh, or the bulging hull overlaps the walls and gets pushed out.
    part, convex = _lshape_with_hull_collision()
    sc = MujocoBinScene(part, convex, n_parts=1, render=False, arrangement="structured",
                        structure_type="tray", structure_height_frac=0.7, clearance_mode="medium",
                        settle_time=3.0)
    mujoco.mj_forward(sc.model, sc.data)
    pre = {k: v["position"].copy() for k, v in sc.extract_scene_state().items()}
    sc.simulate()
    post = {k: v["position"].copy() for k, v in sc.extract_scene_state().items()}
    drift = max(np.linalg.norm(post[k][:2] - pre[k][:2]) for k in pre)
    assert sc.n_parts > 0
    assert drift < 0.003, f"concave part drifted {1000 * drift:.1f} mm -- pocket is ejecting the part"
