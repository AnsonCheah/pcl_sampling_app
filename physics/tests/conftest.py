"""Shared fixtures for the physics test suite.

Mirrors `stages/tests/conftest.py`: registers the `slow` marker (previously only registered
there, so `pytestmark = pytest.mark.slow` under `physics/tests/` emitted PytestUnknownMarkWarning)
and performs the repo-root `sys.path` insert every physics test module repeats.

Run the fast subset:   python -m pytest physics/tests -q -m "not slow"
Run everything:        python -m pytest physics/tests -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import open3d as o3d
import trimesh
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: slow tests (raycast / MuJoCo physics / sensor sim)")


@pytest.fixture(scope="session")
def make_box_part():
    """Factory: (trimesh box, [o3d convex mesh]) for a part of the given extents in metres.

    Boxes are the right primitive for bin-sizing tests: OBB volume, footprint and stable-pose
    height are all exact, so expected bin dimensions can be computed in closed form.
    """
    def _make(extents=(0.02, 0.02, 0.02)):
        tri = trimesh.creation.box(extents=list(extents))       # centred at origin
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(tri.vertices),
            triangles=o3d.utility.Vector3iVector(tri.faces),
        )
        o3d_mesh.compute_vertex_normals()
        return tri, [o3d_mesh]
    return _make


@pytest.fixture(scope="session")
def cube_part(make_box_part):
    """The 20 mm cube used as the worked example throughout the dynamic-bin design."""
    return make_box_part((0.02, 0.02, 0.02))
