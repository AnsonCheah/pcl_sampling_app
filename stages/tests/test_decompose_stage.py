"""Tests for DecomposeStage and headless robustness of the stage pipeline.

Covers:
  * DecomposeStage.worker() populates app.convex_meshes synchronously.
  * SceneStage/RenderStage.reset() do not crash in headless mode (regression: the
    combobox attributes do not exist when build_panel() returns early).
  * The Stage enum / app stages dict include DECOMPOSE ordered between
    SAVE and SCENE/RENDER.

Run:  python -m pytest stages/tests/test_decompose_stage.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import open3d as o3d
import trimesh

from app import MeshSamplingApp
from enums import Stage


def _box_mesh():
    tri = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tri.vertices),
        triangles=o3d.utility.Vector3iVector(tri.faces),
    )
    mesh.compute_vertex_normals()
    return mesh


def test_decompose_populates_convex_meshes():
    app = MeshSamplingApp(headless=True)
    app.target_mesh = _box_mesh()
    app.convex_meshes = []

    app.stages[Stage.DECOMPOSE].worker()

    assert len(app.convex_meshes) > 0
    assert all(isinstance(m, o3d.geometry.TriangleMesh) for m in app.convex_meshes)


def test_decompose_reset_clears_meshes():
    app = MeshSamplingApp(headless=True)
    app.target_mesh = _box_mesh()
    app.stages[Stage.DECOMPOSE].worker()
    assert len(app.convex_meshes) > 0

    app.stages[Stage.DECOMPOSE].reset()
    assert app.convex_meshes == []


def test_scene_render_reset_headless_no_crash():
    # Regression: reset() touched combobox widgets that never exist in headless.
    app = MeshSamplingApp(headless=True)
    app.stages[Stage.SCENE].reset()   # must not raise
    app.stages[Stage.RENDER].reset()  # must not raise
    assert app.synthetic_targets == {}
    assert app.synthetic_scenes == {}
    assert app.o3d_scene == {}
    assert app.mj_scene is None


def test_decompose_stage_registered_and_ordered():
    app = MeshSamplingApp(headless=True)
    assert Stage.DECOMPOSE in app.stages
    assert Stage.SCENE in app.stages and Stage.RENDER in app.stages
    assert Stage.SAVE.value < Stage.DECOMPOSE.value < Stage.SCENE.value < Stage.RENDER.value
