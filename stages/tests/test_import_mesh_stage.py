"""Tests for ImportMeshStage: unit conversion, basename, empty-mesh handling, centering,
and the stale-state wipe on loading a new mesh."""

from pathlib import Path

import numpy as np
import open3d as o3d

from enums import Stage


def test_worker_converts_mm_to_m(headless_app, make_box_mesh):
    stage = headless_app.stages[Stage.IMPORT_MESH]
    stage.file_path = Path("part_in_mm.stl")
    # extent 50 is in [5, 5000] -> treated as mm, scaled x0.001.
    stage.worker(mesh=make_box_mesh((50.0, 50.0, 50.0)))

    extent = headless_app.target_mesh.get_axis_aligned_bounding_box().get_extent().max()
    assert extent < 5.0
    assert np.isclose(extent, 0.05, atol=1e-6)
    assert headless_app.mesh_basename == "part_in_mm"


def test_worker_keeps_metre_scale(headless_app, make_box_mesh):
    stage = headless_app.stages[Stage.IMPORT_MESH]
    stage.file_path = Path("part_in_m.stl")
    stage.worker(mesh=make_box_mesh((0.05, 0.05, 0.05)))

    extent = headless_app.target_mesh.get_axis_aligned_bounding_box().get_extent().max()
    assert np.isclose(extent, 0.05, atol=1e-6)  # unchanged


def test_worker_empty_mesh_no_crash(headless_app):
    stage = headless_app.stages[Stage.IMPORT_MESH]
    stage.file_path = Path("empty.stl")
    stage.worker(mesh=o3d.geometry.TriangleMesh())  # empty
    assert headless_app.target_mesh is None


def test_center_mesh_moves_centroid_to_origin(headless_app, box_mesh):
    app = headless_app
    box_mesh.translate((1.0, 2.0, 3.0))
    app.target_mesh = box_mesh
    # a downstream cloud should ride along with the same offset
    app.raw_pcd = box_mesh.sample_points_uniformly(200)
    before = np.asarray(app.raw_pcd.get_center())

    app.stages[Stage.IMPORT_MESH].center_mesh()

    assert np.linalg.norm(app.target_mesh.get_center()) < 1e-6
    after = np.asarray(app.raw_pcd.get_center())
    assert np.allclose(after - before, [-1.0, -2.0, -3.0], atol=1e-6)


def test_loading_new_mesh_wipes_downstream(headless_app, make_box_mesh):
    app = headless_app
    app.down_pcd = "stale"
    app.convex_meshes = [1, 2]
    app.synthetic_targets = {"k": 1}

    stage = app.stages[Stage.IMPORT_MESH]
    stage.file_path = Path("fresh.stl")
    stage.worker(mesh=make_box_mesh())

    assert app.target_mesh is not None
    assert app.down_pcd is None
    assert app.convex_meshes == []
    assert app.synthetic_targets == {}
