"""Tests for resolution-based mesh decimation in geometry/geom_utils.py.

The target resolution is a fraction of the part's OBB diagonal, clamped by absolute metric
bounds — so it is scale-invariant in the middle of its range (part #1 and part #5000 get
comparable triangle density) and deliberately NOT scale-invariant at the clamps.

Run:  python -m pytest geometry/tests/test_decimate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import open3d as o3d
import pytest
import trimesh

from geometry.geom_utils import (
    DECIMATE_DIAG_FRAC,
    DECIMATE_MIN_VOXEL_M,
    DECIMATE_MAX_VOXEL_M,
    DECIMATE_MAX_VOLUME_ERR,
    decimate_mesh_to_resolution,
    o3d_to_trimesh,
)


def _sphere(radius, subdivisions=5):
    """A dense watertight icosphere — stands in for a high-resolution scanned/CAD part."""
    tri = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tri.vertices),
        triangles=o3d.utility.Vector3iVector(tri.faces),
    )
    mesh.compute_vertex_normals()
    return mesh


def _obb_diagonal(mesh):
    # `.primitive.extents` = the OBB's own side lengths (see trimesh #871/#1865); the inherited
    # `.extents` is the axis-aligned bounds of the rotated box and inflates toward its diagonal.
    return float(np.linalg.norm(
        o3d_to_trimesh(mesh).bounding_box_oriented.primitive.extents))


# -- the core requirement ------------------------------------------------------


def test_scale_invariance():
    """The same shape at two scales must decimate to the same triangle count, with the voxel
    scaling by exactly the size ratio.

    Explicit (wide) clamps and a coarser fraction are passed so both meshes are genuinely
    over-resolved relative to the target; with the shipped defaults an icosphere is already at
    its target resolution and is skipped (see test_skips_mesh_already_at_target_resolution).
    """
    kw = dict(diag_frac=0.02, min_voxel_m=1e-6, max_voxel_m=1.0)
    small = _sphere(0.05, subdivisions=6)
    large = _sphere(0.10, subdivisions=6)

    out_s, st_s = decimate_mesh_to_resolution(small, **kw)
    out_l, st_l = decimate_mesh_to_resolution(large, **kw)

    assert not st_s["skipped"] and not st_l["skipped"]
    assert st_l["voxel_size"] == pytest.approx(2.0 * st_s["voxel_size"], rel=1e-9)
    # Same shape, proportional voxel -> same tessellation density.
    assert st_l["tri_after"] == pytest.approx(st_s["tri_after"], rel=0.05)


def test_skips_mesh_already_at_target_resolution():
    """A mesh with plenty of triangles but edges already coarser than the target is returned
    untouched — the second skip test, independent of the DECIMATE_MIN_TRIANGLES guard."""
    mesh = _sphere(0.05, subdivisions=4)     # 5120 faces (> min_triangles) at ~3.8 mm edges
    out, stats = decimate_mesh_to_resolution(mesh)
    assert out is mesh
    assert stats["skipped"] is True
    assert stats["reason"] == "median edge >= voxel"


def test_clamps_break_scale_invariance_by_design():
    """Below/above the clamp range the target voxel stops tracking part size. This is
    intentional — it stops a tiny part being decimated into a tetrahedron and a huge part
    being left coarser than the render voxel. Documented so nobody "fixes" it."""
    tiny = _sphere(0.0005)     # diag ~1.7 mm -> 0.4% would be ~7 um, below the floor
    huge = _sphere(2.0)        # diag ~6.9 m  -> 0.4% would be ~28 mm, above the ceiling

    assert decimate_mesh_to_resolution(tiny)[1]["voxel_size"] == pytest.approx(
        DECIMATE_MIN_VOXEL_M, rel=1e-9)
    assert decimate_mesh_to_resolution(huge)[1]["voxel_size"] == pytest.approx(
        DECIMATE_MAX_VOXEL_M, rel=1e-9)


def test_voxel_follows_obb_diagonal():
    mesh = _sphere(0.05)
    _, stats = decimate_mesh_to_resolution(mesh)
    assert stats["voxel_size"] == pytest.approx(
        DECIMATE_DIAG_FRAC * _obb_diagonal(mesh), rel=1e-6)


# -- quality guarantees --------------------------------------------------------


def test_reduces_triangle_count():
    """A genuinely over-resolved mesh must be cut down by the SHIPPED defaults.

    subdiv 7 is ~328k faces at ~0.47 mm edges against a ~0.69 mm target, so the expected
    reduction is ~(0.69/0.47)^2 ~ 2x — clustering merges vertices per voxel, so the count falls
    with the square of the edge/voxel ratio, not arbitrarily far.
    """
    mesh = _sphere(0.05, subdivisions=7)
    out, stats = decimate_mesh_to_resolution(mesh)
    assert not stats["skipped"]
    assert stats["tri_before"] > 300000
    assert stats["tri_after"] < 0.6 * stats["tri_before"]
    assert len(out.triangles) == stats["tri_after"]


def test_preserves_volume_within_tolerance():
    out, stats = decimate_mesh_to_resolution(_sphere(0.05, subdivisions=7))
    assert stats["volume_err"] <= DECIMATE_MAX_VOLUME_ERR
    assert not out.is_empty()


def test_preserves_extents():
    """Bin sizing reads the part's extents, so decimation must not shrink the part materially.
    Asserted on the AABB, which is exact — trimesh's OBB is unreliable for rounded shapes
    (it reports a box LARGER than the AABB for a sphere)."""
    mesh = _sphere(0.05, subdivisions=7)
    before = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    out, stats = decimate_mesh_to_resolution(mesh)
    after = np.asarray(out.get_axis_aligned_bounding_box().get_extent())
    # Vertex clustering pulls extreme vertices inward by at most ~voxel/2 per side.
    assert np.all(np.abs(after - before) <= stats["voxel_size"])
    assert np.all(np.abs(after - before) / before < 0.02)


def test_output_is_vhacd_ready():
    """The result feeds VHACD, whose fillMode="flood" wants a clean watertight mesh."""
    out, _ = decimate_mesh_to_resolution(_sphere(0.05, subdivisions=7))
    assert out.has_vertex_normals()
    assert len(out.triangles) > 4
    assert not out.is_empty()
    tri = o3d_to_trimesh(out)
    assert tri.is_watertight
    assert tri.volume > 0


# -- skip / degenerate paths ---------------------------------------------------


def test_skips_already_coarse_mesh():
    """A 12-triangle box is already far coarser than any target voxel — return it untouched,
    by identity, so callers and tests can tell nothing happened."""
    tri = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tri.vertices),
        triangles=o3d.utility.Vector3iVector(tri.faces),
    )
    mesh.compute_vertex_normals()
    out, stats = decimate_mesh_to_resolution(mesh)
    assert out is mesh
    assert stats["skipped"] is True
    assert stats["tri_after"] == stats["tri_before"] == 12


def test_empty_and_tiny_mesh_no_crash():
    empty = o3d.geometry.TriangleMesh()
    out, stats = decimate_mesh_to_resolution(empty)
    assert out is empty and stats["skipped"] is True

    one = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)),
        triangles=o3d.utility.Vector3iVector(np.array([[0, 1, 2]])),
    )
    out, stats = decimate_mesh_to_resolution(one)
    assert out is one and stats["skipped"] is True
