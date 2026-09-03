"""SceneStage: the bin/count sizing policy (fast) and a build + settle smoke test (slow).

The policy tests exercise `_resolve_bin_and_count()` directly rather than `worker()`, so they
cost no physics: the decision under test is which bin to hand MujocoBinScene, not how the
resulting pile settles (that is covered in physics/tests/test_dynamic_bin_settle.py).
"""

import numpy as np
import pytest
import trimesh

from enums import Stage
from geometry.geom_utils import O3DSceneObject
from physics.mujoco_bin_scene import MAX_BIN_DIM


def _scene_stage(headless_app):
    stage = headless_app.stages[Stage.SCENE]
    stage.arrangement = "random"
    stage.generate_mode = "fill_rate"
    stage.fill_rate = 0.2
    stage.structure_type = "none"
    stage.dynamic_bin = True
    return stage


@pytest.fixture
def small_part():
    """20 mm cube -- small enough that dynamic sizing shrinks the bin on every axis it can."""
    return trimesh.creation.box(extents=[0.02, 0.02, 0.02])


def test_dynamic_bin_shrinks_for_small_part(headless_app, small_part):
    stage = _scene_stage(headless_app)
    bin_dim, n = stage._resolve_bin_and_count(small_part)

    assert bin_dim[0] < MAX_BIN_DIM[0] and bin_dim[1] < MAX_BIN_DIM[1]
    assert bin_dim[2] < MAX_BIN_DIM[2]
    assert n == 48                      # closed-form worked example, see physics/tests
    assert stage.num_targets == n       # worker() hands this to MujocoBinScene


def test_dynamic_bin_checkbox_off_uses_max_bin(headless_app, small_part):
    """The escape hatch must reproduce today's behaviour exactly: max bin, legacy count."""
    stage = _scene_stage(headless_app)
    stage.dynamic_bin = False
    bin_dim, n = stage._resolve_bin_and_count(small_part)

    assert bin_dim == pytest.approx(MAX_BIN_DIM, abs=1e-12)
    assert n == stage._auto_part_count(small_part)


def test_override_count_uses_max_bin(headless_app, small_part):
    """Override Count is a manual escape hatch -- exact count, full bin, no dynamic sizing."""
    stage = _scene_stage(headless_app)
    stage.generate_mode = "count"
    stage.num_targets = 7
    bin_dim, n = stage._resolve_bin_and_count(small_part)

    assert bin_dim == pytest.approx(MAX_BIN_DIM, abs=1e-12)
    assert n == 7


def test_structured_arrangement_uses_max_bin(headless_app, small_part):
    """Structured grids set their own count from grid capacity, so the bin must not shrink."""
    stage = _scene_stage(headless_app)
    stage.arrangement = "structured"
    stage.structure_type = "partition"
    bin_dim, _ = stage._resolve_bin_and_count(small_part)

    assert bin_dim == pytest.approx(MAX_BIN_DIM, abs=1e-12)


@pytest.mark.slow
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
