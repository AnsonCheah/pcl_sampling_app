"""Tests for geometry.mesh_repair.

The debris rule has three parts (negligible area, spatially detached, materially inflates the
bounding box) plus a triangle-count cap. Each part is load-bearing against a real file in the
repo, so each gets a test built from the same shape:

  speck            -> 96330MB000.STL: a 2-triangle sliver 1 m out, 76% bbox inflation
  nested shell     -> 96330MB100.STL: 11 components, all inside the main AABB
  scattered bodies -> 40mm.STL:       208 components, 42% of triangles, 0% bbox inflation
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import open3d as o3d
import trimesh
import pytest

from geometry.mesh_repair import (
    analyze_mesh, DEBRIS_MAX_TRI_FRAC, MM_TO_M,
)


def _o3d_box(extents, translate=(0.0, 0.0, 0.0)):
    tri = trimesh.creation.box(extents=list(extents))
    tri.apply_translation(translate)
    return tri


def _part(extents, translate=(0.0, 0.0, 0.0)):
    """A densely tessellated part (~1280 triangles), so a 12-triangle speck is a realistic
    fraction of the mesh. A bare box has only 12 triangles, which would trip the assembly
    cap against a single speck."""
    tri = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    tri.apply_scale(list(extents))
    tri.apply_translation(translate)
    return tri


def _to_o3d(tri):
    return o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tri.vertices),
        triangles=o3d.utility.Vector3iVector(tri.faces))


def _concat(*tris):
    return _to_o3d(trimesh.util.concatenate(list(tris)))


# --------------------------------------------------------------------------------------
# Debris: the 96330MB000.STL shape
# --------------------------------------------------------------------------------------

def test_detached_speck_is_removed_and_extent_restored():
    part = _part((0.20, 0.07, 0.19))
    speck = _o3d_box((1e-5, 5e-5, 1e-5), translate=(1.0, 0.0, 0.0))
    mesh = _concat(part, speck)

    cleaned, report = analyze_mesh(mesh)

    assert report.has_debris()
    assert not report.refused_debris_removal
    assert len(report.debris) == 1
    assert report.debris[0].gap > 0.5
    # bbox inflated ~1.10 m -> back to the part's 0.20 m
    assert report.diag_shrink_frac > 0.5
    ext = np.asarray(cleaned.get_axis_aligned_bounding_box().get_extent())
    assert np.allclose(ext, [0.20, 0.07, 0.19], atol=1e-6)


def test_debris_removal_preserves_the_part_triangles():
    part = _part((0.20, 0.07, 0.19))
    n_part = len(part.faces)
    speck = _o3d_box((1e-5, 1e-5, 1e-5), translate=(1.0, 0.0, 0.0))
    cleaned, report = analyze_mesh(_concat(part, speck))

    assert len(np.asarray(cleaned.triangles)) == n_part   # only the speck's 12 are gone
    assert report.debris_triangles == 12


# --------------------------------------------------------------------------------------
# Nested shells: the 96330MB100.STL shape — detached but inside the main AABB
# --------------------------------------------------------------------------------------

def test_nested_inner_shell_is_kept():
    outer = _o3d_box((0.20, 0.20, 0.20))
    inner = _o3d_box((0.002, 0.002, 0.002))   # tiny, but concentric => gap == 0
    mesh = _concat(outer, inner)
    n_before = len(np.asarray(mesh.triangles))

    cleaned, report = analyze_mesh(mesh)

    assert report.n_clusters == 2
    assert not report.has_debris()
    assert not report.refused_debris_removal
    assert len(np.asarray(cleaned.triangles)) == n_before


def test_small_component_that_does_not_inflate_bbox_is_kept():
    """Detached and negligible, but tucked inside the main body's AABB footprint."""
    outer = _o3d_box((0.20, 0.20, 0.20))
    inside = _o3d_box((0.001, 0.001, 0.001), translate=(0.05, 0.05, 0.05))
    cleaned, report = analyze_mesh(_concat(outer, inside))

    assert not report.has_debris()
    assert len(np.asarray(cleaned.triangles)) == 24


# --------------------------------------------------------------------------------------
# Assembly guard: the 40mm.STL shape
# --------------------------------------------------------------------------------------

def test_scattered_assembly_is_refused_not_cut():
    """Many comparable bodies spread over a metre: each is individually negligible and outside
    the largest body's AABB, so the area+gap rule alone would delete most of the file."""
    rng = np.random.default_rng(0)
    parts = [_o3d_box((0.02, 0.02, 0.02))]
    for _ in range(60):
        t = rng.uniform(-0.5, 0.5, size=3)
        parts.append(_o3d_box((0.02, 0.02, 0.02), translate=tuple(t)))
    mesh = _concat(*parts)
    n_before = len(np.asarray(mesh.triangles))

    cleaned, report = analyze_mesh(mesh)

    assert report.n_clusters > 10
    assert not report.has_debris()
    assert len(np.asarray(cleaned.triangles)) == n_before   # nothing cut


def test_triangle_cap_blocks_wholesale_deletion():
    """Even when detached bodies DO inflate the bbox, cutting more than the cap is refused.
    Here 200 specks are each individually negligible in area (so they pass the area+gap test)
    but together make up 62% of the triangles — deleting them would gut the file."""
    main = _part((0.20, 0.20, 0.20))
    far = [_o3d_box((1e-4, 1e-4, 1e-4), translate=(1.0 + 0.01 * i, 0.0, 0.0))
           for i in range(200)]
    mesh = _concat(main, *far)
    n_before = len(np.asarray(mesh.triangles))

    cleaned, report = analyze_mesh(mesh)

    assert report.debris                                   # candidates were identified
    assert report.debris_triangles / n_before >= DEBRIS_MAX_TRI_FRAC
    assert report.refused_debris_removal
    assert len(np.asarray(cleaned.triangles)) == n_before   # nothing cut


# --------------------------------------------------------------------------------------
# Topology hygiene, units, bin fit
# --------------------------------------------------------------------------------------

def test_duplicate_vertices_are_merged_without_changing_geometry():
    tri = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    # explode into a vertex soup, the way an STL stores it
    soup = trimesh.Trimesh(vertices=tri.vertices[tri.faces].reshape(-1, 3),
                           faces=np.arange(len(tri.faces) * 3).reshape(-1, 3),
                           process=False)
    mesh = _to_o3d(soup)
    ext_before = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())

    cleaned, report = analyze_mesh(mesh)

    assert report.dup_vertices_removed > 0
    assert len(np.asarray(cleaned.vertices)) == 8
    assert np.allclose(np.asarray(cleaned.get_axis_aligned_bounding_box().get_extent()),
                       ext_before, atol=1e-9)


def test_unit_heuristic_uses_the_debris_free_extent():
    """A mm-authored part with a speck 6000 mm out: the raw extent (>5000) would skip the
    conversion and silently leave the part in millimetres."""
    part = _part((200.0, 70.0, 190.0))
    speck = _o3d_box((0.01, 0.01, 0.01), translate=(6000.0, 0.0, 0.0))
    _cleaned, report = analyze_mesh(_concat(part, speck))

    assert report.has_debris()
    assert report.unit_scale == MM_TO_M


def test_metre_scale_part_is_not_rescaled():
    _cleaned, report = analyze_mesh(_to_o3d(_o3d_box((0.2, 0.07, 0.19))))
    assert report.unit_scale == 1.0


def test_bin_fit_flags_oversize_part():
    _cleaned, report = analyze_mesh(_to_o3d(_o3d_box((1.2, 0.05, 0.05))),
                                    bin_limit=(0.76, 0.585, 0.25))
    assert not report.fits_in_bin
    assert report.has_findings()


def test_bin_fit_allows_part_needing_reorientation():
    """Sorted-extent comparison: a part taller than the bin height still fits if it can lie down."""
    _cleaned, report = analyze_mesh(_to_o3d(_o3d_box((0.05, 0.05, 0.5))),
                                    bin_limit=(0.76, 0.585, 0.25))
    assert report.fits_in_bin


def test_watertight_box_is_reported_watertight():
    _cleaned, report = analyze_mesh(_to_o3d(_o3d_box((0.1, 0.1, 0.1))),
                                    bin_limit=(0.76, 0.585, 0.25))
    assert report.is_watertight
    assert report.is_edge_manifold
    assert not report.has_findings()      # a clean part must not interrupt the operator


# --------------------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------------------

def test_empty_mesh_reports_error():
    _cleaned, report = analyze_mesh(o3d.geometry.TriangleMesh())
    assert report.errors
    assert report.has_findings()


def test_non_finite_vertices_report_error():
    mesh = _to_o3d(_o3d_box((0.1, 0.1, 0.1)))
    v = np.asarray(mesh.vertices).copy()
    v[0] = [np.nan, 0.0, 0.0]
    mesh.vertices = o3d.utility.Vector3dVector(v)

    _cleaned, report = analyze_mesh(mesh)
    assert report.errors
    assert "NaN" in report.summary()


def test_analyze_does_not_mutate_input():
    part = _part((0.20, 0.07, 0.19))
    speck = _o3d_box((1e-5, 1e-5, 1e-5), translate=(1.0, 0.0, 0.0))
    mesh = _concat(part, speck)
    n_before = len(np.asarray(mesh.triangles))

    cleaned, _report = analyze_mesh(mesh)

    assert len(np.asarray(mesh.triangles)) == n_before        # original untouched
    assert len(np.asarray(cleaned.triangles)) < n_before


def test_summary_is_printable():
    part = _part((0.20, 0.07, 0.19))
    speck = _o3d_box((1e-5, 1e-5, 1e-5), translate=(1.0, 0.0, 0.0))
    _cleaned, report = analyze_mesh(_concat(part, speck), bin_limit=(0.76, 0.585, 0.25))
    text = report.summary()
    assert "DEBRIS DETECTED" in text
    text.encode("ascii")     # must survive a cp1252 Windows console
