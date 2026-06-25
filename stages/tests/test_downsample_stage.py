"""Tests for DownsampleStage: uniform vs adaptive output shape and the early return when
there is no cropped cloud. Uses a directly-sampled box cloud so no slow raycast is needed."""

import numpy as np

from enums import Stage


def _seed_cropped(app, box_mesh, n=5000):
    app.target_mesh = box_mesh
    app.cropped_pcd = box_mesh.sample_points_uniformly(n)


def test_uniform_downsample(headless_app, box_mesh):
    app = headless_app
    _seed_cropped(app, box_mesh)
    stage = app.stages[Stage.DOWNSAMPLE]
    stage.use_adaptive = False

    stage.worker()

    assert app.down_pcd is not None
    assert len(app.down_pcd.points) <= len(app.cropped_pcd.points)
    # uniform mode aliases the surface cloud to down_pcd; the recenter logic relies on this.
    assert app.down_pcd_surface is app.down_pcd
    assert app.geocenter.shape == (4, 4)
    assert app.down_pcd_edge is not None


def test_adaptive_downsample(headless_app, box_mesh):
    app = headless_app
    _seed_cropped(app, box_mesh)
    stage = app.stages[Stage.DOWNSAMPLE]
    stage.use_adaptive = True

    stage.worker()

    assert app.feature_pcd is not None
    assert app.pcd_flat is not None
    # adaptive mode keeps surface as an independent deepcopy of down_pcd.
    assert app.down_pcd_surface is not app.down_pcd
    assert app.down_pcd_edge is not None


def test_no_cropped_cloud_early_return(headless_app):
    app = headless_app
    app.cropped_pcd = None
    app.stages[Stage.DOWNSAMPLE].worker()   # must not raise
    assert app.down_pcd is None
