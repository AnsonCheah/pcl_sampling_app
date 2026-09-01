"""`bin_transform` places the bin, its fixtures and the spawned parts in world.

It is identity for every current caller, which is exactly why it needs a test: an unused
parameter that is never exercised silently stops working. These pin the contract a
non-identity world frame (e.g. camera_frame == world_frame) would rely on.

Run:  python -m pytest physics/tests/test_bin_transform.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from physics.mujoco_bin_scene import MujocoBinScene
from physics.tests.test_bin_scene import generate_test_mesh

_SHIFT = np.array([0.30, -0.20, 0.10])


def _scene(bin_transform=None):
    part, convex = generate_test_mesh("cube_big")
    return MujocoBinScene(part, convex, n_parts=2, render=False, arrangement="structured",
                          structure_type="none", settle_time=0.5,
                          bin_transform=bin_transform)


def _translation(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T


@pytest.fixture(scope="module")
def identity_scene():
    return _scene()


@pytest.fixture(scope="module")
def shifted_scene():
    return _scene(_translation(_SHIFT))


def test_defaults_to_identity(identity_scene):
    assert np.allclose(identity_scene.bin_transform, np.eye(4))


def test_accepts_a_list_not_just_an_array():
    """Callers should not have to pre-convert; np.asarray handles the rest."""
    sc = _scene(_translation(_SHIFT).tolist())
    assert isinstance(sc.bin_transform, np.ndarray)
    assert np.allclose(sc.bin_transform, _translation(_SHIFT))


def test_bin_mesh_moves_with_the_transform(identity_scene, shifted_scene):
    base = np.asarray(identity_scene.bin_mesh.vertices).mean(axis=0)
    moved = np.asarray(shifted_scene.bin_mesh.vertices).mean(axis=0)
    assert np.allclose(moved - base, _SHIFT, atol=1e-9), (
        f"bin mesh centroid moved by {moved - base}, expected {_SHIFT}")


def test_parts_spawn_inside_the_moved_bin(shifted_scene):
    """Spawned poses are placed through bin_transform, so parts follow the bin."""
    state = shifted_scene.extract_scene_state()
    assert state, "no bodies spawned"
    xy = np.array([state[n]["position"][:2] for n in state])
    lo = _SHIFT[:2] - [shifted_scene.hx, shifted_scene.hy]
    hi = _SHIFT[:2] + [shifted_scene.hx, shifted_scene.hy]
    assert np.all((xy >= lo) & (xy <= hi)), (
        f"parts at {xy} fall outside the shifted bin footprint [{lo}, {hi}]")


def test_exported_state_records_the_transform(shifted_scene):
    """scene_state.npz carries it, so a reader can recover the world placement."""
    exported = shifted_scene.export_scene_state()
    assert np.allclose(exported["bin_transform"], _translation(_SHIFT))
