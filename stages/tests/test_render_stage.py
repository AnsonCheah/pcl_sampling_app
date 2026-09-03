"""Smoke test for RenderStage: run the sensor sim + segmentation over a settled scene.
Heaviest test in the suite -- primarily a no-crash / output-shape guard. Slow."""

import numpy as np
import pytest

from enums import Stage

pytestmark = pytest.mark.slow


def test_render_worker_produces_synthetic_data(sampled_app):
    app = sampled_app
    app.stages[Stage.DECOMPOSE].worker()

    scene = app.stages[Stage.SCENE]
    scene.arrangement = "random"
    scene.generate_mode = "count"
    scene.num_targets = 2
    scene.structure_type = "none"
    scene.stable_pose_R = None
    scene.worker()
    assert app.mj_scene is not None

    app.set_stage(Stage.RENDER)
    app.stages[Stage.RENDER].worker()

    # The noise-stage scenes are always produced; targets only if instances pass filtering.
    assert len(app.synthetic_scenes) > 0
    for obj in app.synthetic_targets.values():
        assert np.asarray(obj.T_gt).shape == (4, 4)
        assert hasattr(obj, "overlap")
