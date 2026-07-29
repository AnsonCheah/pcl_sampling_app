"""Unit tests for small pure-math helpers in geometry.geom_utils.

Run:  python -m pytest geometry/tests/test_geom_utils.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest
import trimesh

from geometry.geom_utils import rotation_aligning_vector_to_axis, face_facet_map


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


def test_face_facet_map_box_groups_coplanar_faces():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.merge_vertices()
    face_to_facet, facets, facets_normal = face_facet_map(box)

    assert len(facets) == 6                                   # a box has 6 flat sides
    assert all(len(grp) == 2 for grp in facets)               # each side = 2 coplanar triangles
    assert (face_to_facet >= 0).all()                         # every triangle is grouped
    assert facets_normal.shape == (6, 3)
    # facet normals are axis-aligned, and match each member triangle's own normal direction
    axis_sorted = np.sort(np.abs(facets_normal), axis=1)
    assert np.allclose(axis_sorted[:, :2], 0.0, atol=1e-6)
    for f, fi in enumerate(face_to_facet):
        assert abs(abs(np.dot(facets_normal[fi], box.face_normals[f])) - 1.0) < 1e-6


def test_face_facet_map_curved_is_all_singletons():
    sphere = trimesh.creation.icosphere(subdivisions=1)       # no coplanar-adjacent faces
    sphere.merge_vertices()
    face_to_facet, facets, _ = face_facet_map(sphere)
    assert len(facets) == 0
    assert (face_to_facet == -1).all()
