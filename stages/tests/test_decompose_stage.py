"""Tests for DecomposeStage and headless robustness of the stage pipeline.

Covers:
  * DecomposeStage.worker() populates app.convex_meshes synchronously.
  * The Stage enum / app stages dict include DECOMPOSE ordered between
    SAVE and SCENE/RENDER.

(reset()/headless-clear robustness now lives in test_clear_state.py.)

Run:  python -m pytest stages/tests/test_decompose_stage.py
"""

import open3d as o3d

from enums import Stage


def test_decompose_populates_convex_meshes(headless_app, box_mesh):
    app = headless_app
    app.target_mesh = box_mesh
    app.convex_meshes = []

    app.stages[Stage.DECOMPOSE].worker()

    assert len(app.convex_meshes) > 0
    assert all(isinstance(m, o3d.geometry.TriangleMesh) for m in app.convex_meshes)


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
