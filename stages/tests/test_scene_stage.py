"""Smoke test for SceneStage: build + settle a tiny random bin scene under MuJoCo and hand
off o3d_scene / mj_scene. Slow (physics)."""

import numpy as np
import pytest

from enums import Stage
from geometry.geom_utils import O3DSceneObject

pytestmark = pytest.mark.slow


def test_scene_worker_builds_settled_scene(sampled_app):
    app = sampled_app
    app.stages[Stage.DECOMPOSE].worker()
    assert len(app.convex_meshes) > 0

    scene = app.stages[Stage.SCENE]
    scene.arrangement = "random"
    scene.generate_mode = "count"
    scene.num_targets = 2
    scene.structure_type = "none"
    scene.stable_pose_R = None

    scene.worker()

    assert app.mj_scene is not None
    assert len(app.o3d_scene) > 0
    obj = next(iter(app.o3d_scene.values()))
    assert isinstance(obj, O3DSceneObject)
    assert np.asarray(obj.T_gt).shape == (4, 4)
