"""Tests for CropStage: the point-deletion mask logic, downstream clearing, and the
special reset() that restores cropped_pcd from raw_pcd. All headless / no GUI."""

import copy

import numpy as np
import open3d as o3d

from enums import Stage


def _cloud(n=100):
    rng = np.random.default_rng(0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rng.random((n, 3)))
    pcd.normals = o3d.utility.Vector3dVector(rng.random((n, 3)))  # mask_point_cloud carries normals
    return pcd


def test_worker_deletes_selected_points(headless_app):
    app = headless_app
    app.raw_pcd = _cloud(100)
    app.cropped_pcd = copy.deepcopy(app.raw_pcd)
    app.down_pcd = "stale"            # downstream marker

    stage = app.stages[Stage.CROP]
    stage.selected_indices = [3, 7, 11]
    stage.worker()

    assert len(app.cropped_pcd.points) == 100 - 3
    assert stage.selected_indices == []      # scratch reset
    assert app.down_pcd is None              # downstream cleared


def test_reset_restores_cropped_from_raw(headless_app):
    app = headless_app
    app.raw_pcd = _cloud(100)
    app.cropped_pcd = _cloud(40)             # pretend a prior crop shrank it
    app.down_pcd = "stale"

    app.stages[Stage.CROP].reset()

    assert len(app.cropped_pcd.points) == 100          # restored to raw
    assert app.cropped_pcd is not app.raw_pcd          # independent deepcopy
    assert app.down_pcd is None                        # downstream cleared
