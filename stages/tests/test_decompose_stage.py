"""Tests for DecomposeStage and headless robustness of the stage pipeline.

Covers:
  * DecomposeStage.worker() populates app.convex_meshes synchronously.
  * A concave part is split into multiple genuinely-convex hulls (exercises the
    CoACD backend, which the box fixture does not since a box is already convex).
  * The Stage enum / app stages dict include DECOMPOSE ordered between
    SAVE and SCENE/RENDER.

(reset()/headless-clear robustness now lives in test_clear_state.py.)

Run:  python -m pytest stages/tests/test_decompose_stage.py
"""

import numpy as np
import open3d as o3d
import trimesh

from enums import Stage


def _concave_l_bracket_o3d():
    """L-shaped bracket (box with a corner notch removed) as an o3d mesh: the
    simplest part that cannot be one convex hull, so CoACD must return >1 piece."""
    big = trimesh.creation.box(extents=[0.10, 0.10, 0.02])
    notch = trimesh.creation.box(extents=[0.06, 0.06, 0.04])
    notch.apply_translation([0.02, 0.02, 0])
    tri = big.difference(notch)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(tri.vertices)),
        triangles=o3d.utility.Vector3iVector(np.asarray(tri.faces)),
    )
    mesh.compute_vertex_normals()
    return mesh


def test_decompose_populates_convex_meshes(headless_app, box_mesh):
    app = headless_app
    app.target_mesh = box_mesh
    app.convex_meshes = []

    app.stages[Stage.DECOMPOSE].worker()

    assert len(app.convex_meshes) > 0
    assert all(isinstance(m, o3d.geometry.TriangleMesh) for m in app.convex_meshes)


def test_decompose_concave_part_into_convex_hulls(headless_app):
    app = headless_app
    app.target_mesh = _concave_l_bracket_o3d()
    app.convex_meshes = []

    app.stages[Stage.DECOMPOSE].worker()

    # CoACD must split the concavity into more than one piece...
    assert len(app.convex_meshes) > 1
    # ...and every piece it returns must actually be convex.
    for m in app.convex_meshes:
        assert trimesh.Trimesh(
            vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles), process=False
        ).is_convex


def test_decompose_reset_clears_meshes(headless_app, box_mesh):
    app = headless_app
    app.target_mesh = box_mesh
    app.stages[Stage.DECOMPOSE].worker()
    assert len(app.convex_meshes) > 0

    app.stages[Stage.DECOMPOSE].reset()
    assert app.convex_meshes == []


def test_decompose_stage_registered_and_ordered(headless_app):
    app = headless_app
    assert Stage.DECOMPOSE in app.stages
    assert Stage.SCENE in app.stages and Stage.RENDER in app.stages
    assert Stage.SAVE.value < Stage.DECOMPOSE.value < Stage.SCENE.value < Stage.RENDER.value
