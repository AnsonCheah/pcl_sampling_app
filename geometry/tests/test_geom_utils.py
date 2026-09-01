"""Unit tests for small pure-math helpers in geometry.geom_utils.

Run:  python -m pytest geometry/tests/test_geom_utils.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy

import numpy as np
import open3d as o3d
import pytest
import trimesh

from geometry.ambiguity import AmbiguityAxis
from geometry.geom_utils import (
    face_facet_map,
    median_spacing,
    pcd_geocenter,
    reference_frames_agree,
    rotation_aligning_vector_to_axis,
)


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


# ─────────────────────────────────────────────────────────────────────────────
# pcd_geocenter — the model frame MechVision's rotationStrategy is aimed at
# ─────────────────────────────────────────────────────────────────────────────

def _box_cloud(dims=(0.100, 0.030, 0.020), n=20000, seed=0):
    """Long box surface, AABB-centred. PCA-1 is X, PCA-2 Y, PCA-3 Z.

    Sampled here rather than via ``sample_points_uniformly``, which takes no seed in this
    Open3D build — and these tests assert on a frame that must be reproducible. Deliberately
    not a regular lattice either: a lattice is self-similar under a lattice-vector shift, so
    it would understate what ``reference_frames_agree`` sees on a real cloud.
    """
    rng = np.random.default_rng(seed)
    half = np.asarray(dims, dtype=float) / 2.0
    # One face pair per axis; sampled in proportion to area so the surface density is uniform.
    areas = np.array([dims[1] * dims[2], dims[0] * dims[2], dims[0] * dims[1]], dtype=float)
    counts = np.maximum(2, np.rint(n * areas / areas.sum()).astype(int))

    pts, nrm = [], []
    for axis, count in enumerate(counts):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        block = np.zeros((count, 3))
        sign = rng.choice([-1.0, 1.0], size=count)
        block[:, axis] = sign * half[axis]
        block[:, u] = rng.uniform(-half[u], half[u], count)
        block[:, v] = rng.uniform(-half[v], half[v], count)
        normals = np.zeros((count, 3))
        normals[:, axis] = sign
        pts.append(block)
        nrm.append(normals)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pts))
    pcd.normals = o3d.utility.Vector3dVector(np.vstack(nrm))
    return pcd


def _apply(pcd, T):
    moved = copy.deepcopy(pcd)
    moved.transform(T)
    return moved


def _axis_distance(query, direction, point):
    """Perpendicular distance from ``query`` to the line (point, direction)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    v = np.asarray(query, float) - np.asarray(point, float)
    return float(np.linalg.norm(v - (v @ d) * d))


def test_geocenter_translation_is_in_column_three():
    """The convention `SaveStage` reads. It used to read row 3, which is always [0,0,0,1] —
    so `geocenter_x/y/z` was 0.0 in every PLY ever exported, whatever the frame."""
    T = pcd_geocenter(_box_cloud())
    assert T.shape == (4, 4)
    assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])
    # A box centred on the origin has a near-zero frame origin, so offset the cloud to make
    # the translation unambiguous.
    shifted = _apply(_box_cloud(), np.array([[1.0, 0, 0, 0.011],
                                             [0, 1.0, 0, -0.023],
                                             [0, 0, 1.0, 0.007],
                                             [0, 0, 0, 1.0]]))
    T2 = pcd_geocenter(shifted)
    assert np.linalg.norm(T2[:3, 3]) > 1e-3, "translation must live in column 3"
    assert np.allclose(T2[3, :], [0.0, 0.0, 0.0, 1.0])


def test_geocenter_is_deterministic():
    """Pure function: the frame is baked into every exported cloud, so it must not drift
    run to run. `_fix_axis_signs` exists to kill the eigen-decomposition's sign freedom."""
    pcd = _box_cloud()
    assert np.array_equal(pcd_geocenter(pcd), pcd_geocenter(pcd))
    ax = AmbiguityAxis(direction=np.array([0.0, 0.0, 1.0]), point=np.array([0.004, -0.002, 0.0]),
                       fold=2, angles_deg=[180.0], is_global=True,
                       view_fraction=1.0, area_fraction=0.99)
    assert np.array_equal(pcd_geocenter(pcd, axis=ax), pcd_geocenter(pcd, axis=ax))


def test_pca_frame_centres_the_cloud():
    pcd = _box_cloud()
    moved = _apply(pcd, pcd_geocenter(pcd))
    assert np.allclose(np.asarray(moved.points).mean(axis=0), 0.0, atol=1e-9)


def test_ambiguity_frame_puts_the_axis_on_the_origin_as_z():
    """The whole contract: `rotationStrategy` can only rotate about a frame axis through the
    frame origin, so the ambiguity axis must be exactly that."""
    pcd = _box_cloud()
    offset = 0.006
    ax = AmbiguityAxis(direction=np.array([0.0, 0.0, 1.0]),
                       point=np.array([offset, 0.0, 0.0]),
                       fold=2, angles_deg=[180.0], is_global=True,
                       view_fraction=1.0, area_fraction=0.99)
    T = pcd_geocenter(pcd, axis=ax)
    moved_axis = ax.transformed(T)
    d = moved_axis.direction / np.linalg.norm(moved_axis.direction)

    assert abs(abs(float(d @ [0.0, 0.0, 1.0])) - 1.0) < 1e-9, "axis must become frame Z"
    assert _axis_distance([0.0, 0.0, 0.0], d, moved_axis.point) < 1e-9, \
        "axis must pass through the frame origin"
    # ...and the part is therefore NOT centred: it ends up displaced by exactly the axis's
    # off-centroid distance, which a rigid transform preserves. That is the documented trade.
    before = np.asarray(pcd.points).mean(axis=0)
    expected = _axis_distance(before, ax.direction, ax.point)
    centroid = np.asarray(_apply(pcd, T).points).mean(axis=0)
    assert expected > 0.005, "fixture must actually put the axis off the centroid"
    assert np.isclose(np.linalg.norm(centroid[:2]), expected, atol=1e-9)


def test_ambiguity_frame_origin_is_the_projected_cloud_mean():
    """The remaining along-axis freedom is pinned to the projection of the cloud mean, so the
    part sits as close to the origin as the perpendicular constraint allows."""
    pcd = _box_cloud()
    ax = AmbiguityAxis(direction=np.array([0.0, 0.0, 1.0]), point=np.array([0.006, 0.0, 0.004]),
                       fold=2, angles_deg=[180.0], is_global=True,
                       view_fraction=1.0, area_fraction=0.99)
    centroid = np.asarray(_apply(pcd, pcd_geocenter(pcd, axis=ax)).points).mean(axis=0)
    assert abs(centroid[2]) < 1e-9, "along-axis offset of the mean must be zero"


def test_no_shortcut_when_the_pca_frame_nearly_agrees():
    """Regression for the deleted "PCA already agrees" shortcut.

    It fired whenever a PCA axis sat within 5 deg of the ambiguity axis and the axis passed
    within 1% of the extent of the PCA origin, and returned the PCA frame instead. Measured on
    exactly this input, that put the axis in frame **X** sitting **0.83 mm off the origin** —
    so a rotation search about frame Z swept a completely different line.
    """
    pcd = _box_cloud()                                   # extent diagonal ~106 mm -> 1% ~1.06 mm
    ax = AmbiguityAxis(direction=np.array([1.0, 0.0, 0.0]),   # parallel to PCA-1
                       point=np.array([0.0, 0.0, 0.0008]),    # 0.8 mm off -> inside the old tol
                       fold=2, angles_deg=[180.0], is_global=True,
                       view_fraction=1.0, area_fraction=0.99)
    moved_axis = ax.transformed(pcd_geocenter(pcd, axis=ax))
    d = moved_axis.direction / np.linalg.norm(moved_axis.direction)

    assert abs(abs(float(d @ [0.0, 0.0, 1.0])) - 1.0) < 1e-9, "axis must be frame Z, not frame X"
    assert _axis_distance([0.0, 0.0, 0.0], d, moved_axis.point) < 1e-9


def test_ambiguity_frame_is_idempotent():
    """A second recentre must be a no-op, which is why `recenter_mesh_pcd` needs no
    "already applied" guard. The floor is `np.round(rot, decimals=6)` inside `_pack`."""
    pcd = _box_cloud()
    ax = AmbiguityAxis(direction=np.array([0.02, 0.0, 0.9998]), point=np.array([0.006, 0.002, 0.0]),
                       fold=2, angles_deg=[180.0], is_global=True,
                       view_fraction=1.0, area_fraction=0.99)
    T1 = pcd_geocenter(pcd, axis=ax)
    once = _apply(pcd, T1)

    T2 = pcd_geocenter(once, axis=ax.transformed(T1))
    twice = _apply(once, T2)

    moved = np.linalg.norm(np.asarray(twice.points) - np.asarray(once.points), axis=1)
    assert moved.max() < 1e-9, f"second recentre moved points by {moved.max() * 1000:.6f} mm"


# ─────────────────────────────────────────────────────────────────────────────
# reference_frames_agree — the stale-scene guard
# ─────────────────────────────────────────────────────────────────────────────

def test_identical_clouds_agree():
    pcd = _box_cloud(n=4000)
    assert reference_frames_agree(pcd, copy.deepcopy(pcd)) is None


def test_a_displaced_cloud_is_rejected():
    """One median spacing is the smallest displacement worth calling a different frame; a real
    frame change moves everything by millimetres."""
    pcd = _box_cloud(n=4000)
    shift = np.eye(4)
    shift[:3, 3] = [2.0 * median_spacing(pcd), 0.0, 0.0]
    why = reference_frames_agree(_apply(pcd, shift), pcd)
    assert why is not None and "different model frames" in why


def test_a_rotated_cloud_is_rejected():
    pcd = _box_cloud(n=4000)
    rot = np.eye(4)
    rot[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, np.deg2rad(3.0)))
    assert reference_frames_agree(_apply(pcd, rot), pcd) is not None


def test_a_different_point_count_is_rejected_outright():
    """Not the same export at all — no geometric comparison is meaningful."""
    pcd = _box_cloud(n=4000)
    why = reference_frames_agree(_box_cloud(n=3960), pcd)
    assert why is not None and "point counts differ" in why


def test_float32_ply_round_trip_still_agrees(tmp_path):
    """The tolerance has to absorb the precision the bundle is actually stored at."""
    from geometry.file_utils import pointcloud_to_ply

    pcd = _box_cloud(n=4000)
    path = tmp_path / "ref.ply"
    pointcloud_to_ply(pcd, str(path))
    assert reference_frames_agree(o3d.io.read_point_cloud(str(path)), pcd) is None
