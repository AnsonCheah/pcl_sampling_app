"""Unit tests for small pure-math helpers in geometry.geom_utils.

Run:  python -m pytest geometry/tests/test_geom_utils.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from geometry.geom_utils import rotation_aligning_vector_to_axis


def _is_rotation(R):
    return (np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(R), 1.0, atol=1e-6))


@pytest.mark.parametrize("src", [
    (0.0, 0.0, 1.0),     # already aligned with default +Z
    (0.0, 0.0, -1.0),    # antiparallel to +Z (degenerate)
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 2.0, 3.0),
    (-2.0, 0.5, -1.0),
    (0.05, 0.0, 1.0),    # small angle, above the identity-snap threshold
])
def test_aligns_src_to_plus_z(src):
    R = rotation_aligning_vector_to_axis(src, (0.0, 0.0, 1.0))
    assert _is_rotation(R), "result is not a proper rotation matrix"
    src_u = np.asarray(src, float)
    src_u /= np.linalg.norm(src_u)
    assert np.allclose(R @ src_u, [0.0, 0.0, 1.0], atol=1e-6)


def test_aligns_to_arbitrary_axis():
    src = np.array([0.3, -0.7, 0.2])
    dst = np.array([1.0, 1.0, 0.0])
    R = rotation_aligning_vector_to_axis(src, dst)
    assert _is_rotation(R)
    assert np.allclose(R @ (src / np.linalg.norm(src)),
                       dst / np.linalg.norm(dst), atol=1e-6)


def test_input_not_mutated():
    src = np.array([1.0, 2.0, 3.0])
    before = src.copy()
    rotation_aligning_vector_to_axis(src)
    assert np.array_equal(src, before)


def test_parallel_returns_identity():
    R = rotation_aligning_vector_to_axis((0.0, 0.0, 5.0), (0.0, 0.0, 1.0))
    assert np.allclose(R, np.eye(3), atol=1e-6)
