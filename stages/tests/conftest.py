"""Shared fixtures for the stage test suite.

All tests run headless (`MeshSamplingApp(headless=True)`): every stage worker branches on
`self.app.headless` and skips the Open3D GUI, so no window/renderer is created.

Run the fast subset:   python -m pytest stages/tests -q -m "not slow"
Run everything:        python -m pytest stages/tests -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import open3d as o3d
import trimesh
import pytest

from app import MeshSamplingApp
from enums import Stage


@pytest.fixture(scope="session")
def make_box_mesh():
    """Factory: a watertight box o3d mesh with vertex normals. `extents` are in the mesh's
    own units — pass values in [5, 5000] to exercise the import mm->m auto-conversion."""
    def _make(extents=(0.05, 0.05, 0.05)):
        tri = trimesh.creation.box(extents=list(extents))
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(tri.vertices),
            triangles=o3d.utility.Vector3iVector(tri.faces),
        )
        mesh.compute_vertex_normals()
        return mesh
    return _make


@pytest.fixture
def box_mesh(make_box_mesh):
    """A fresh 5 cm box mesh (function-scoped so per-test in-place transforms don't leak)."""
    return make_box_mesh()


@pytest.fixture
def headless_app():
    return MeshSamplingApp(headless=True)


@pytest.fixture(scope="module")
def sampled_app(make_box_mesh):
    """An app with IMPORT->RAYCAST->DOWNSAMPLE already run on a box (uniform downsample).

    Module-scoped so the slow raycast runs once per heavy test module. Used by the
    SAVE/SCENE/RENDER tests, which need a real reference cloud + point-count range.
    """
    app = MeshSamplingApp(headless=True)
    app.target_mesh = make_box_mesh()
    app.mesh_basename = "test_box"
    app.stages[Stage.RAYCAST].worker()
    app.stages[Stage.DOWNSAMPLE].use_adaptive = False
    app.stages[Stage.DOWNSAMPLE].worker()
    return app
