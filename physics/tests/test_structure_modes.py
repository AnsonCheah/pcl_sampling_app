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

from physics.mujoco_bin_scene import MujocoBinScene, TRAY_BASE
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
    # Every part centre must sit at/above the pocket floor (z = TRAY_BASE) — i.e. inside the pocket,
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
    # Structured/none stays static (no freejoint) — qpos has only the bin's 0 DOFs + no part joints.
    assert scenes["none"].model.njnt == 0


def _lshape_with_hull_collision():
    """Concave L part whose collision is its single convex hull — the worst-case bulge. The hull fills
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
    assert drift < 0.003, f"concave part drifted {1000 * drift:.1f} mm — pocket is ejecting the part"
