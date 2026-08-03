"""Mesh/cloud pairing guard for the ambiguity viewer.

The viewer resolves a mesh and a cloud from two different places on disk. If they are not
the same part in the same model frame, nothing errors — ``_visibility_masks`` raycasts the
mesh, fails to snap the hits onto the cloud, and the analysis proceeds against a few
hundred accidental points, reporting axes that have nothing to do with the part.

That is exactly what a stale MechVision resource copy paired with a re-exported STL did:
13% of points ever visible instead of 99%, and 12 reported axes instead of 3. These tests
pin the guard that now rejects it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation as Rot

from geometry.visualize_ambiguity import _mesh_candidates, pairing_error


def _part():
    """A part with no rotational symmetry, so a rotated cloud really does leave the mesh."""
    mesh = o3d.geometry.TriangleMesh.create_box(0.100, 0.030, 0.020)
    mesh.translate(-mesh.get_axis_aligned_bounding_box().get_center())
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(0)
    return mesh, mesh.sample_points_uniformly(6000, use_triangle_normal=False)


def test_matching_pair_is_accepted():
    mesh, pcd = _part()
    assert pairing_error(mesh, pcd) is None


def test_translation_of_the_whole_bundle_is_still_a_match():
    """_prepare re-centres both together; a shared rigid move must not trip the guard."""
    mesh, pcd = _part()
    shift = np.array([0.031, -0.017, 0.229])
    mesh.translate(shift)
    pcd.translate(shift)
    assert pairing_error(mesh, pcd) is None


def test_cloud_in_a_different_model_frame_is_rejected():
    """The real failure: the export was re-run into an ambiguity-aligned frame and only one
    of the two files was refreshed."""
    mesh, pcd = _part()
    pcd.rotate(Rot.from_euler("xyz", [0.0, 90.0, 180.0], degrees=True).as_matrix(),
               center=(0, 0, 0))
    why = pairing_error(mesh, pcd)
    assert why is not None and "mm of the mesh surface" in why


@pytest.mark.parametrize("angle_deg", [2.0, 10.0, 45.0])
def test_rejection_does_not_need_a_large_misalignment(angle_deg):
    mesh, pcd = _part()
    pcd.rotate(Rot.from_rotvec(np.deg2rad(angle_deg) * np.array([0.0, 1.0, 0.0])).as_matrix(),
               center=(0, 0, 0))
    assert pairing_error(mesh, pcd) is not None


def test_tolerance_scales_with_the_cloud_not_the_part():
    """A sparse cloud of a big part and a dense cloud of a small one must be judged the
    same way — the tolerance is a multiple of the cloud's own spacing."""
    for scale, n in ((0.02, 800), (0.5, 40000)):
        mesh = o3d.geometry.TriangleMesh.create_box(scale, scale * 0.3, scale * 0.2)
        mesh.compute_vertex_normals()
        o3d.utility.random.seed(0)
        pcd = mesh.sample_points_uniformly(n, use_triangle_normal=False)
        assert pairing_error(mesh, pcd) is None
        moved = o3d.geometry.PointCloud(pcd)
        moved.translate([scale * 0.1, 0.0, 0.0])   # 10% of the part, in both cases
        assert pairing_error(mesh, moved) is not None


def test_bundle_stl_is_preferred_over_any_other_copy():
    """The mesh exported beside a cloud is the only one guaranteed to share its frame, so
    it must be tried before REF_ROOT or the raw CAD at the repo root."""
    ply = os.path.join("output", "reference_pcd", "P", "P_surface", "P_surface.ply")
    cands = _mesh_candidates(ply, "P")
    assert os.path.basename(os.path.dirname(cands[0])) == "P"
    assert cands[0].endswith(os.path.join("P", "P.stl"))
