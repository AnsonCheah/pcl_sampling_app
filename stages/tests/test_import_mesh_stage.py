"""Tests for ImportMeshStage: unit conversion, basename, empty-mesh handling, centering,
debris validation, and the stale-state wipe on loading a new mesh."""

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from enums import Stage


def _mesh_with_debris():
    """A dense part plus a 12-triangle speck 1 m away -- the 96330MB000.STL failure shape."""
    part = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    part.apply_scale([0.20, 0.07, 0.19])
    speck = trimesh.creation.box(extents=[1e-5, 1e-5, 1e-5])
    speck.apply_translation([1.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([part, speck])
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(combined.vertices),
        triangles=o3d.utility.Vector3iVector(combined.faces))
    mesh.compute_vertex_normals()
    return mesh, len(part.faces)


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


def test_worker_decimates_after_unit_conversion(headless_app):
    """Decimation must run AFTER the mm->m conversion. Its absolute voxel clamps are in metres,
    so on a still-in-mm mesh the target would clamp to the 2 mm ceiling and destroy the part
    (a 150-unit diagonal x 0.4% = 0.6 "mm-units", i.e. 0.6 m once converted).

    Feeds a dense 50 mm sphere: the final mesh must be BOTH converted to metres and decimated.
    """
    import trimesh
    tri = trimesh.creation.icosphere(subdivisions=7, radius=25.0)   # 50 mm diameter, in mm
    dense = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tri.vertices),
        triangles=o3d.utility.Vector3iVector(tri.faces))
    dense.compute_vertex_normals()
    n_before = len(dense.triangles)

    stage = headless_app.stages[Stage.IMPORT_MESH]
    stage.file_path = Path("dense_part_mm.stl")
    stage.worker(mesh=dense)

    extent = headless_app.target_mesh.get_axis_aligned_bounding_box().get_extent().max()
    # 0.05 m, not 50: the conversion ran. Tolerance is one voxel, since vertex clustering pulls
    # extreme vertices inward by up to ~voxel/2 (bounded tightly by test_preserves_extents).
    assert np.isclose(extent, 0.05, atol=1e-3), "unit conversion lost"
    assert len(headless_app.target_mesh.triangles) < n_before, "mesh was not decimated"


def test_worker_empty_mesh_no_crash(headless_app):
    stage = headless_app.stages[Stage.IMPORT_MESH]
    stage.file_path = Path("empty.stl")
    stage.worker(mesh=o3d.geometry.TriangleMesh())  # empty
    assert headless_app.target_mesh is None


def test_center_mesh_moves_bbox_centre_to_origin(headless_app, box_mesh):
    app = headless_app
    box_mesh.translate((1.0, 2.0, 3.0))
    app.target_mesh = box_mesh
    # a downstream cloud should ride along with the same offset
    app.raw_pcd = box_mesh.sample_points_uniformly(200)
    before = np.asarray(app.raw_pcd.get_center())

    app.stages[Stage.IMPORT_MESH].center_mesh()

    bbox_centre = np.asarray(app.target_mesh.get_axis_aligned_bounding_box().get_center())
    assert np.linalg.norm(bbox_centre) < 1e-6
    after = np.asarray(app.raw_pcd.get_center())
    assert np.allclose(after - before, [-1.0, -2.0, -3.0], atol=1e-6)


def test_center_mesh_anchors_on_bbox_not_vertex_mean(headless_app):
    """A tessellation-biased part: one face densely subdivided drags the vertex mean off the
    bounding-box centre. Centering must land the bbox centre on the origin regardless."""
    tri = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    dense = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    for _ in range(4):
        dense = dense.subdivide()
    dense.apply_translation([0.09, 0.0, 0.0])          # cluster of vertices at one end
    combined = trimesh.util.concatenate([tri, dense])
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(combined.vertices),
        triangles=o3d.utility.Vector3iVector(combined.faces))
    headless_app.target_mesh = mesh

    vertex_mean = np.asarray(mesh.get_center())
    bbox_centre = np.asarray(mesh.get_axis_aligned_bounding_box().get_center())
    assert np.linalg.norm(vertex_mean - bbox_centre) > 1e-3   # the fixture really is biased

    headless_app.stages[Stage.IMPORT_MESH].center_mesh()

    after = np.asarray(headless_app.target_mesh.get_axis_aligned_bounding_box().get_center())
    assert np.linalg.norm(after) < 1e-6


def test_headless_import_auto_removes_debris(headless_app):
    """Headless has no dialog: choice_dialog resolves to its first option, which removes."""
    mesh, n_part_faces = _mesh_with_debris()
    stage = headless_app.stages[Stage.IMPORT_MESH]
    stage.file_path = Path("debris_part.stl")
    stage.worker(mesh=mesh)

    assert stage._pending is None                     # decision was resolved
    assert len(np.asarray(headless_app.target_mesh.triangles)) == n_part_faces
    extent = np.asarray(
        headless_app.target_mesh.get_axis_aligned_bounding_box().get_extent())
    assert np.allclose(extent, [0.20, 0.07, 0.19], atol=1e-6)


def test_clean_mesh_leaves_no_pending_decision(headless_app, make_box_mesh):
    stage = headless_app.stages[Stage.IMPORT_MESH]
    stage.file_path = Path("clean.stl")
    stage.worker(mesh=make_box_mesh((0.05, 0.05, 0.05)))
    assert stage._pending is None


def test_debris_does_not_leak_into_the_unit_heuristic(headless_app):
    """A mm part with a speck 6000 mm out: the raw extent exceeds the mm window, so without
    debris-first analysis the part would silently stay in millimetres."""
    part = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    part.apply_scale([200.0, 70.0, 190.0])
    speck = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    speck.apply_translation([6000.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([part, speck])
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(combined.vertices),
        triangles=o3d.utility.Vector3iVector(combined.faces))

    stage = headless_app.stages[Stage.IMPORT_MESH]
    stage.file_path = Path("mm_with_debris.stl")
    stage.worker(mesh=mesh)

    extent = np.asarray(
        headless_app.target_mesh.get_axis_aligned_bounding_box().get_extent())
    assert np.allclose(extent, [0.200, 0.070, 0.190], atol=1e-6)


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
