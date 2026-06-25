"""Tests for RaycastStage: reference-cloud production and point-count stats.
Marked slow (renders over the full view sphere)."""

import numpy as np
import pytest

from enums import Stage

pytestmark = pytest.mark.slow


def test_worker_produces_reference_cloud(headless_app, box_mesh):
    app = headless_app
    app.target_mesh = box_mesh

    app.stages[Stage.RAYCAST].worker()

    assert app.raw_pcd is not None and len(app.raw_pcd.points) > 0
    assert app.raw_pcd.has_normals()
    # cropped_pcd is an independent working copy of raw_pcd.
    assert app.cropped_pcd is not app.raw_pcd
    assert len(app.cropped_pcd.points) == len(app.raw_pcd.points)


def test_point_count_stats(headless_app, box_mesh):
    app = headless_app
    app.target_mesh = box_mesh

    app.stages[Stage.RAYCAST].worker()

    assert isinstance(app.point_count_mean, int) and app.point_count_mean > 0
    lo, hi = app.point_count_range
    assert lo <= hi


def test_no_mesh_early_return(headless_app):
    app = headless_app
    app.target_mesh = None
    app.stages[Stage.RAYCAST].worker()   # must not raise
    assert app.raw_pcd is None
