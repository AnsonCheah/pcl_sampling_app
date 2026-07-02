"""Tests for geometry/tray_utils.py — footprint silhouette + per-pocket tray tile.

Run:  python -m pytest geometry/tests/test_tray_utils.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import trimesh
from shapely.geometry import Polygon

from geometry.tray_utils import (footprint_polygon, convex_footprint_polygon, _coalesce,
                                  build_tray_collision_frame, tray_visual_tile,
                                  build_tray_conforming_hfield)


def _lshape():
    """Concave L-shaped prism (a box with a corner notch removed)."""
    base = trimesh.creation.box(extents=[0.06, 0.06, 0.02])
    notch = trimesh.creation.box(extents=[0.03, 0.03, 0.05])
    notch.apply_translation([0.015, 0.015, 0.0])
    return trimesh.boolean.difference([base, notch], engine="manifold")


def test_footprint_polygon_box_is_rectangle():
    box = trimesh.creation.box(extents=[0.08, 0.04, 0.02])
    poly = footprint_polygon(box, np.eye(3), clearance=0.0)
    assert isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty
    # ~ the 8x4 cm rectangle (no clearance); allow slack for the buffer/rounding path.
    assert 0.0030 < poly.area < 0.0036


def test_footprint_clearance_grows_outline():
    box = trimesh.creation.box(extents=[0.08, 0.04, 0.02])
    a0 = footprint_polygon(box, np.eye(3), clearance=0.0).area
    a1 = footprint_polygon(box, np.eye(3), clearance=0.003).area
    assert a1 > a0   # outward offset enlarges the footprint


def test_footprint_concave_is_smaller_than_bbox():
    poly = footprint_polygon(_lshape(), np.eye(3), clearance=0.0)
    # The L footprint must be clearly less than its 6x6 cm bounding box (i.e. concavity kept).
    assert poly.is_valid and poly.area < 0.06 * 0.06 * 0.95


def test_footprint_fill_holes_makes_solid():
    # A washer (ring) projects to an annulus; fill_holes must yield a hole-free pocket so the part
    # always seats (no interfering central post).
    outer = trimesh.creation.cylinder(radius=0.03, height=0.02, sections=48)
    inner = trimesh.creation.cylinder(radius=0.015, height=0.05, sections=48)
    ring = trimesh.boolean.difference([outer, inner], engine="manifold")
    filled = footprint_polygon(ring, np.eye(3), clearance=0.0, fill_holes=True)
    assert len(filled.interiors) == 0
    assert filled.area > np.pi * 0.03 ** 2 * 0.9   # ~ full disk, not an annulus


def test_convex_footprint_unions_all_pieces():
    # A connected 2-piece part: the collision footprint must cover BOTH hulls. Regression for the bug
    # where the mesh-projection silhouette fractured and only the largest lobe was kept, shrinking the
    # pocket below the part (poses that overlapped the tray walls and ejected the part).
    a = trimesh.creation.box(extents=[0.04, 0.04, 0.02]); a.apply_translation([-0.015, 0, 0])
    b = trimesh.creation.box(extents=[0.04, 0.04, 0.02]); b.apply_translation([0.015, 0, 0])
    poly = convex_footprint_polygon([a, b], np.eye(3), clearance=0.0)
    assert poly.geom_type == "Polygon"
    assert poly.bounds[0] < -0.034 and poly.bounds[2] > 0.034   # spans both pieces, none dropped
    assert poly.area > 0.04 * 0.07 * 0.95                       # ~ the full 7x4 cm combined footprint


def test_coalesce_merges_touching_lobes_without_dropping_area():
    from shapely.geometry import box, MultiPolygon
    p = MultiPolygon([box(0, 0, 1, 1), box(1.0, 0, 2, 1)])     # two edge-adjacent unit squares
    m = _coalesce(p)
    assert m.geom_type == "Polygon"
    assert abs(m.area - 2.0) < 0.05                            # both lobes kept (old max() kept one → 1.0)


def test_build_tray_collision_frame_decomposed():
    poly = footprint_polygon(_lshape(), np.eye(3), clearance=0.0025)
    frame, pieces = build_tray_collision_frame(poly, pitch_xy=(0.075, 0.075), pocket_depth=0.014,
                                               base_z=0.004, max_convex_hulls=12, vhacd_resolution=200000)
    # Frame is the wall ring at z in [base_z, base_z + pocket_depth] — no base under the pocket.
    assert frame.is_watertight
    assert abs(frame.bounds[0][2] - 0.004) < 1e-3
    assert abs(frame.bounds[1][2] - 0.018) < 1e-3
    assert len(pieces) >= 1
    for v, f in pieces:
        assert len(v) >= 4 and len(f) >= 4    # each convex piece is a real closed mesh


def test_conforming_hfield_cradles_a_tilted_bottom():
    # A box tilted about Y has a ramped bottom; the conforming pocket floor must ramp to match it
    # (and provide a uniform clearance gap), with walls at the cell top outside the footprint.
    box = trimesh.creation.box(extents=[0.08, 0.05, 0.02])
    box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(15), [0, 1, 0]))
    hf = build_tray_conforming_hfield([box.convex_hull], np.eye(3), (0.10, 0.07),
                                      pocket_depth=0.02, base_thickness=0.004, clearance=0.0025)
    el = hf["elevation"]
    assert el.ndim == 2 and el.min() >= 0.0 and el.max() <= 1.0
    assert len(hf["size"]) == 4 and hf["seat_dz"] > 0
    assert len(hf["visual"].vertices) > 0
    assert hf["z_offset"] >= 0.0 and hf["size"][2] > 0.0
    mujoco_z = hf["z_offset"] + el.astype(float) * hf["size"][2]
    assert np.allclose(hf["collision_surface"], mujoco_z, atol=1e-7)
    vz = np.asarray(hf["visual"].vertices)[:, 2].reshape(hf["visual_shape"])
    contact = hf["visual_contact_mask"]
    top_z = hf["z_offset"] + hf["size"][2]
    assert np.allclose(vz[contact], hf["visual_floor"][contact], atol=1e-7)
    rim = (~contact) & (vz > hf["visual_floor"] + 1e-7) & (vz < top_z - 1e-7)
    assert rim.any(), "visual mesh has no smoothed wall/rim band"
    # walls present (elevation hits the cell top = 1) AND a lower conforming floor
    assert el.max() > 0.99 and el.min() < el.max() - 0.05
    # the floor ramps along the tilt axis (x = columns): mean elevation should trend across columns
    floor_rows = el[el.min(axis=1) < 0.99]          # rows that contain floor (not all-wall)
    col_means = el.mean(axis=0)
    assert np.ptp(col_means) > 0.02, "conforming floor shows no ramp for a tilted bottom"


def test_tray_visual_tile_watertight():
    poly = footprint_polygon(_lshape(), np.eye(3), clearance=0.0025)
    tile = tray_visual_tile(poly, pitch_xy=(0.075, 0.075), pocket_depth=0.014, base_thickness=0.004)
    assert tile.is_watertight
    # Tile spans z in [0, base+depth]; the pocket is cut from the top.
    assert abs(tile.bounds[0][2] - 0.0) < 1e-6
    assert abs(tile.bounds[1][2] - 0.018) < 1e-3
